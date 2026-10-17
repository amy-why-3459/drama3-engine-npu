import asyncio
from pathlib import Path
import tempfile
import unittest

import aiohttp
from aiohttp import web

from drama3.benchmark import adapt_drama3_request, attach_frames, request
from drama3.metrics import RequestTrace, summarize


class FrameJoinTests(unittest.TestCase):
    def test_server_assigned_id_joins_without_changing_client_artifact_id(self):
        trace = RequestTrace('client', 0, .1, 1, 2, 2048, 44100, None, True,
                             server_request_id='speech-server')
        joined = attach_frames([trace], [{'request_id': 'speech-server', 'generated_frames': 1}])
        self.assertEqual(joined[0].request_id, 'client')
        self.assertEqual(joined[0].generated_frames, 1)
        with self.assertRaises(ValueError):
            attach_frames([trace, trace], [])

    def test_reference_adapter_preserves_codes_and_rejects_ignored_sampling(self):
        codes = [[1, 2]] * 10
        payload = {'input': 'hello', 'ref_text': 'reference', 'ref_audio_codes': codes,
                   'stream': True, 'response_format': 'pcm'}
        sglang = adapt_drama3_request(payload, 'sglang-omni')
        self.assertEqual(sglang['references'], [{'vq_codes': codes, 'text': 'reference'}])
        self.assertNotIn('ref_audio_codes', sglang)
        self.assertEqual(payload['ref_audio_codes'], codes)
        vllm = adapt_drama3_request(payload, 'vllm-omni')
        self.assertEqual(vllm['stream_format'], 'audio')
        self.assertEqual(vllm['ref_audio_codes'], codes)
        with self.assertRaises(ValueError):
            adapt_drama3_request({**payload, 'top_k': 30}, 'vllm-omni')

    def test_join_uses_ids_and_rejects_ambiguous_counts(self):
        traces = [RequestTrace("a", 0, .1, 1, 2, 44100, 44100, None, True),
                  RequestTrace("b", 0, .1, 1, 2, 44100, 44100, None, True)]
        records = [{"request_id": "b", "generated_frames": 30},
                   {"request_id": "a", "generated_frames": 20}]
        joined = attach_frames(traces, records)
        self.assertEqual([t.generated_frames for t in joined], [20, 30])
        self.assertEqual(summarize(joined, 2)["throughput_frames_s"], 25)
        with self.assertRaises(ValueError):
            attach_frames(traces, records * 2)
        with self.assertRaises(ValueError):
            attach_frames(traces, [{"request_id": "a", "generated_frames": True}])
        self.assertIsNone(attach_frames(traces, records[:1])[0].generated_frames)


class StreamingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        app = web.Application()

        async def speech(req):
            mode = (await req.json())["mode"]
            if mode == "error":
                return web.json_response({"error": "failed"}, status=503)
            if mode == "json":
                return web.json_response({"error": "not audio"})
            response = web.StreamResponse(headers={"Content-Type": "audio/pcm",
                                                   "X-Request-Id": "server-" + mode})
            await response.prepare(req)
            await response.write(b"\x01")
            await asyncio.sleep(.01)
            await response.write(b"\x00\x02\x00")
            if mode == "odd":
                await response.write(b"\x03")
            await response.write_eof()
            return response

        app.router.add_post("/v1/audio/speech", speech)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{self.runner.addresses[0][1]}/v1/audio/speech"
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=2))

    async def asyncTearDown(self):
        await self.session.close()
        await self.runner.cleanup()
        self.directory.cleanup()

    async def test_pcm_split_sample_is_retained_without_inventing_frames(self):
        path = Path(self.directory.name) / "audio.pcm"
        trace = await request(self.session, self.url, {"mode": "pcm"}, "a", path, 44100)
        trace.validate()
        self.assertTrue(trace.success)
        self.assertEqual(trace.server_request_id, 'server-pcm')
        self.assertEqual(trace.audio_samples, 2)
        self.assertEqual(path.read_bytes(), b"\x01\x00\x02\x00")
        self.assertIsNone(trace.generated_frames)
        self.assertFalse(summarize([trace], 1)["eligible_for_comparison"])

    async def test_http_error_json_and_partial_sample_remain_failed_requests(self):
        for mode in ("error", "json", "odd"):
            trace = await request(self.session, self.url, {"mode": mode}, mode,
                                  Path(self.directory.name) / f"{mode}.pcm", 44100)
            self.assertFalse(trace.success)
            self.assertIsNotNone(trace.error)
            self.assertEqual(summarize([trace], 1)["failed_requests"], 1)
