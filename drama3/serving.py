"""Reference streaming service with request-owned model state and bounded queues.

The default uses interleaved eager execution; --batched enables continuous
batched decode. Performance/quality qualification remains separate.
"""
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import uuid

from aiohttp import web


@dataclass
class Session:
    request_id: str
    payload: dict
    output: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=8))
    stream: object = None
    cancelled: bool = False
    generated_frames: int = 0
    emitted_frames: int = 0
    codes: list = field(default_factory=list)


class ReferenceBackend:
    def __init__(self, args):
        import torch
        import torch_npu  # noqa: F401
        from transformers import AutoTokenizer
        from drama3.generation import ReferenceGenerator
        from drama3.model.backbone import Drama3Model
        from drama3.model.codec import AudioCodec

        torch.set_num_threads(args.cpu_threads)
        torch.npu.set_device(args.devices[0])
        torch.manual_seed(args.seed)
        torch.npu.manual_seed_all(args.seed)
        self.model = Drama3Model.load(args.model, args.devices, cache_slots=args.max_active,
                                     max_length=args.context_length, use_npu_moe=args.npu_moe,
                                     use_npu_attention=args.npu_attention,
                                     use_npu_gdn_prefill=args.npu_gdn_prefill,
                                     use_npu_gdn_decode=args.npu_gdn_decode,
                                     use_npu_norm=args.npu_norm,
                                     use_npu_gdn_graph=args.npu_gdn_graph,
                                     use_npu_layer_graph=args.npu_layer_graph)
        self.codec = AudioCodec(args.model / 'codec.pth', args.devices[-1], torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
        self.generator = ReferenceGenerator(self.model)
        self.first_frames = args.first_chunk_frames
        self.chunk_frames = args.chunk_frames
        self.left_frames = args.left_context_frames
        self.context_length = args.context_length

    def shutdown(self):
        """Release NPU tensors on the same worker thread that owns execution."""
        import gc
        import torch

        devices = self.model.devices
        for device in devices:
            torch.npu.synchronize(device)
        for module in self.model.modules():
            clear = getattr(module, "clear_graphs", None)
            if clear is not None:
                clear()
        self.generator = None
        self.codec = None
        self.model = None
        self.tokenizer = None
        gc.collect()
        for device in devices:
            with torch.npu.device(device):
                torch.npu.empty_cache()
                torch.npu.synchronize(device)
        print('STANDALONE_BACKEND_SHUTDOWN_COMPLETE', flush=True)

    def prepare(self, session):
        import torch
        from drama3.generation import Sampling
        from drama3.model.prompt import Reference, build_prompt_frames

        references = []
        if 'ref_audio_codes' in session.payload:
            references.append(Reference(
                session.payload['ref_text'],
                torch.tensor(session.payload['ref_audio_codes'], dtype=torch.long).T,
            ))
        prompt = build_prompt_frames(
            self.tokenizer, session.payload['input'], references,
            semantic_begin_id=self.model.config['semantic_start_token_id'],
            num_codebooks=self.model.tower.num_codebooks,
        )
        sampling = Sampling(**session.payload['sampling'])
        if len(prompt) + sampling.max_frames > self.context_length:
            raise ValueError('Prompt and requested output exceed context length')
        return prompt, sampling

    def step(self, session):
        import torch

        torch.npu.set_device(self.model.devices[0])
        if session.stream is None:
            session.stream = self.generator.frames(*self.prepare(session))
        return self.consume(session, next(session.stream, None))

    def consume(self, session, code):
        import torch

        finished = code is None
        if code is not None:
            if not (0 <= code[0] < 4096 and ((code[1:] >= 0) & (code[1:] < 1024)).all()):
                raise RuntimeError('Invalid generated acoustic frame')
            session.codes.append(code)
            session.generated_frames += 1
        pending = session.generated_frames - session.emitted_frames
        target = self.first_frames if session.emitted_frames == 0 else self.chunk_frames
        pcm = None
        if pending and (finished or pending >= target):
            context = len(session.codes) - pending
            indices = torch.stack(session.codes).T.unsqueeze(0).to(self.model.devices[-1])
            audio = self.codec.decode(indices)[0, context * self.codec.hop_length:]
            if not torch.isfinite(audio).all():
                raise RuntimeError('Nonfinite decoded audio')
            pcm = (audio.clamp(-1, 1) * 32767).short().cpu().numpy().astype('<i2').tobytes()
            session.emitted_frames = session.generated_frames
            session.codes = session.codes[-self.left_frames:] if self.left_frames else []
        if finished and session.generated_frames == 0:
            raise RuntimeError('EOS without acoustic frames')
        return pcm, finished

    def close(self, session):
        if session.stream is not None:
            session.stream.close()
        session.codes.clear()


class BatchedBackend(ReferenceBackend):
    def __init__(self, args):
        super().__init__(args)
        from drama3.batching import BatchedGenerator

        self.generator = BatchedGenerator(self.model)

    def step_batch(self, sessions):
        import torch

        torch.npu.set_device(self.model.devices[0])
        results = [None] * len(sessions)
        prepared = []
        for index, session in enumerate(sessions):
            try:
                if session.stream is None:
                    session.stream = self.generator.prefill(*self.prepare(session))
                prepared.append((index, session))
            except Exception as error:
                results[index] = error
        if prepared:
            try:
                frames = self.generator.step([session.stream for _, session in prepared])
            except Exception as error:
                for index, _ in prepared:
                    results[index] = error
            else:
                for (index, session), frame in zip(prepared, frames):
                    try:
                        results[index] = self.consume(session, frame)
                    except Exception as error:
                        results[index] = error
        return results

    def close(self, session):
        if session.stream is not None:
            self.generator.close(session.stream)
        session.codes.clear()


class ReferenceService:
    def __init__(self, backend, executor, max_active, max_queued, telemetry):
        self.backend = backend
        self.executor = executor
        self.max_active = max_active
        self.pending = asyncio.Queue(maxsize=max_queued)
        self.sessions = {}
        self.telemetry = telemetry
        self.stopping = False

    async def advance(self, sessions):
        loop = asyncio.get_running_loop()
        batch_step = getattr(self.backend, 'step_batch', None)
        if batch_step is not None:
            try:
                results = await loop.run_in_executor(self.executor, batch_step, sessions)
                if len(results) != len(sessions):
                    raise RuntimeError('Backend returned an incomplete batch')
            except Exception as error:
                results = [error] * len(sessions)
            for session, result in zip(sessions, results):
                yield session, result
        else:
            for session in sessions:
                try:
                    result = await loop.run_in_executor(self.executor, self.backend.step, session)
                except Exception as error:
                    result = error
                yield session, result

    async def worker(self):
        active = []
        while not self.stopping:
            while len(active) < self.max_active and not self.pending.empty():
                active.append(self.pending.get_nowait())
            ready = []
            for session in list(active):
                if session.cancelled:
                    await self.finish(session, active, 'cancelled')
                elif session.output.qsize() <= session.output.maxsize - 2:
                    ready.append(session)
            if ready:
                async for session, result in self.advance(ready):
                    if isinstance(result, Exception):
                        await self.finish(session, active, f'{type(result).__name__}: {result}')
                        session.output.put_nowait(result)
                        continue
                    pcm, finished = result
                    if pcm:
                        session.output.put_nowait(pcm)
                    if finished:
                        await self.finish(session, active, 'complete')
                        session.output.put_nowait(None)
            else:
                await asyncio.sleep(.001)
        while not self.pending.empty():
            active.append(self.pending.get_nowait())
        for session in list(active):
            await self.finish(session, active, 'shutdown')
            if not session.output.full():
                session.output.put_nowait(RuntimeError('Service shutting down'))

    async def finish(self, session, active, status):
        await asyncio.get_running_loop().run_in_executor(self.executor, self.backend.close, session)
        active.remove(session)
        self.sessions.pop(session.request_id, None)
        self.telemetry.write(json.dumps({'request_id': session.request_id,
                                        'generated_frames': session.generated_frames,
                                        'emitted_frames': session.emitted_frames,
                                        'status': status}) + '\n')
        self.telemetry.flush()

    async def speech(self, request):
        payload = await request.json()
        allowed = {'model', 'input', 'stream', 'stream_format', 'response_format',
                   'temperature', 'top_k', 'top_p', 'max_new_tokens',
                   'ref_audio_codes', 'ref_text'}
        if not isinstance(payload, dict) or set(payload) - allowed:
            raise web.HTTPBadRequest(text='Unsupported request fields')
        if not isinstance(payload.get('input'), str) or not payload['input'].strip():
            raise web.HTTPBadRequest(text='Nonempty input text required')
        if payload.get('stream') is not True or payload.get('response_format') != 'pcm':
            raise web.HTTPBadRequest(text='Reference endpoint requires stream=true and PCM output')
        if 'ref_audio_codes' in payload or 'ref_text' in payload:
            codes = payload.get('ref_audio_codes')
            transcript = payload.get('ref_text')
            valid = (isinstance(transcript, str) and bool(transcript.strip())
                     and isinstance(codes, list) and len(codes) == 10
                     and all(isinstance(row, list) and bool(row) for row in codes))
            if valid:
                valid = (len({len(row) for row in codes}) == 1
                         and all(type(code) is int and 0 <= code < (4096 if i == 0 else 1024)
                                 for i, row in enumerate(codes) for code in row))
            if not valid:
                raise web.HTTPBadRequest(text='Reference requires transcript and integer [10, frames] codec indices')
        sampling = {key: payload[key] for key in ('temperature', 'top_k', 'top_p') if key in payload}
        if 'max_new_tokens' in payload:
            sampling['max_frames'] = payload['max_new_tokens']
        for key in ('top_k', 'max_frames'):
            if key in sampling and (type(sampling[key]) is not int or sampling[key] < 1):
                raise web.HTTPBadRequest(text=f'{key} must be a positive integer')
        for key in ('temperature', 'top_p'):
            if key in sampling and (type(sampling[key]) not in (int, float) or not math.isfinite(sampling[key])):
                raise web.HTTPBadRequest(text=f'{key} must be finite')
        from drama3.generation import Sampling
        try:
            sampling = Sampling(**sampling)
        except (ValueError, TypeError) as error:
            raise web.HTTPBadRequest(text=str(error)) from error
        payload['sampling'] = vars(sampling)
        request_id = request.headers.get('X-Request-Id') or uuid.uuid4().hex
        if request_id in self.sessions:
            raise web.HTTPConflict(text='Duplicate active request ID')
        if self.pending.full() or self.stopping:
            raise web.HTTPServiceUnavailable(text='Admission queue full or shutting down')
        session = Session(request_id, payload)
        self.sessions[request_id] = session
        self.pending.put_nowait(session)
        response = web.StreamResponse(headers={'Content-Type': 'audio/pcm', 'X-Request-Id': request_id})
        try:
            first = await session.output.get()
            if isinstance(first, Exception):
                raise web.HTTPInternalServerError(text=str(first))
            if first is None:
                raise web.HTTPInternalServerError(text='No audio')
            await response.prepare(request)
            await response.write(first)
            while True:
                chunk = await session.output.get()
                if isinstance(chunk, Exception):
                    # A truncated stream must fail at the client, not become HTTP 200 success.
                    request.transport.close()
                    return response
                if chunk is None:
                    break
                await response.write(chunk)
            await response.write_eof()
            return response
        finally:
            session.cancelled = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--devices', nargs='+', default=['npu:0', 'npu:1'])
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8093)
    parser.add_argument('--max-active', type=int, default=8)
    parser.add_argument('--max-queued', type=int, default=128)
    parser.add_argument('--context-length', type=int, default=2048)
    parser.add_argument('--first-chunk-frames', type=int, default=2)
    parser.add_argument('--chunk-frames', type=int, default=25)
    parser.add_argument('--left-context-frames', type=int, default=25)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--npu-moe', action='store_true')
    parser.add_argument('--npu-layer-graph', action='store_true', help='Replay GDN decoder blocks including grouped MoE')
    parser.add_argument('--npu-gdn-graph', action='store_true', help='Replay cached GDN decode in NPU graphs')
    parser.add_argument('--npu-norm', action='store_true', help='Use fused zero-centered RMSNorm')
    parser.add_argument('--npu-gdn-decode', action='store_true', help='Use standalone FP32-state fused GDN decode')
    parser.add_argument('--npu-gdn-prefill', action='store_true', help='Use optional FP32-state Ascend GDN prefill kernels')
    parser.add_argument('--npu-attention', action='store_true', help='Use native paged attention for batched tower decode')
    parser.add_argument('--batched', action='store_true', help='Batch active backbone/tower decode requests')
    parser.add_argument('--telemetry', type=Path, required=True)
    args = parser.parse_args()
    if min(args.max_active, args.max_queued, args.context_length, args.first_chunk_frames,
           args.chunk_frames, args.cpu_threads) < 1 or args.left_context_frames < 0:
        parser.error('Invalid cache, queue, thread or streaming dimensions')
    if args.npu_layer_graph and (args.npu_gdn_graph or not args.npu_moe):
        parser.error('--npu-layer-graph requires --npu-moe and excludes --npu-gdn-graph')
    if args.npu_attention and not args.batched:
        parser.error('--npu-attention requires --batched')
    args.telemetry.parent.mkdir(parents=True, exist_ok=True)
    with args.telemetry.open('x') as telemetry, ThreadPoolExecutor(max_workers=1) as executor:
        backend_class = BatchedBackend if args.batched else ReferenceBackend
        backend = executor.submit(backend_class, args).result()
        service = ReferenceService(backend, executor, args.max_active, args.max_queued, telemetry)
        app = web.Application()
        app.router.add_post('/v1/audio/speech', service.speech)
        async def lifecycle(app):
            worker = asyncio.create_task(service.worker())
            try:
                yield
            finally:
                service.stopping = True
                await worker
        app.cleanup_ctx.append(lifecycle)
        try:
            web.run_app(app, host=args.host, port=args.port, handler_cancellation=True)
        finally:
            executor.submit(backend.shutdown).result()


if __name__ == '__main__':
    main()
