import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
import unittest

import aiohttp
from aiohttp import web

from drama3.serving import ReferenceService, Session


class Backend:
    def __init__(self):
        self.events = []
        self.closed = []
        self.fail_after = None

    def step(self, session):
        self.events.append(session.request_id)
        if session.payload.get('fail') or session.generated_frames == self.fail_after:
            raise RuntimeError('backend failure')
        session.generated_frames += 1
        session.emitted_frames += 1
        return b'\x01\x00', session.generated_frames == 3

    def close(self, session):
        self.closed.append(session.request_id)


class ServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = Backend()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.telemetry = io.StringIO()
        self.service = ReferenceService(self.backend, self.executor, 2, 4, self.telemetry)
        self.worker = asyncio.create_task(self.service.worker())

    async def asyncTearDown(self):
        self.service.stopping = True
        await asyncio.wait_for(self.worker, 2)
        self.executor.shutdown()

    def submit(self, name, **payload):
        session = Session(name, payload)
        self.service.sessions[name] = session
        self.service.pending.put_nowait(session)
        return session

    async def complete(self, session):
        chunks = []
        while True:
            chunk = await asyncio.wait_for(session.output.get(), 2)
            if chunk is None or isinstance(chunk, Exception):
                return chunks, chunk
            chunks.append(chunk)

    async def test_interleaving_frame_telemetry_and_cleanup(self):
        a, b = self.submit('a'), self.submit('b')
        results = await asyncio.gather(self.complete(a), self.complete(b))
        self.assertEqual([len(chunks) for chunks, _ in results], [3, 3])
        self.assertEqual(self.backend.events, ['a', 'b', 'a', 'b', 'a', 'b'])
        self.service.stopping = True
        await asyncio.wait_for(self.worker, 2)
        records = [json.loads(line) for line in self.telemetry.getvalue().splitlines()]
        self.assertEqual({r['request_id']: r['generated_frames'] for r in records}, {'a': 3, 'b': 3})
        self.assertEqual(self.service.sessions, {})
        self.assertEqual(sorted(self.backend.closed), ['a', 'b'])

    async def test_backpressure_does_not_block_other_requests_or_cancellation(self):
        slow = self.submit('slow')
        for _ in range(slow.output.maxsize):
            slow.output.put_nowait(b'queued')
        fast = self.submit('fast')
        chunks, error = await self.complete(fast)
        self.assertEqual(len(chunks), 3)
        self.assertIsNone(error)
        self.assertNotIn('slow', self.backend.events)
        slow.cancelled = True
        for _ in range(100):
            if 'slow' in self.backend.closed:
                break
            await asyncio.sleep(.001)
        self.assertIn('slow', self.backend.closed)

    async def test_backend_failure_is_not_successful_completion(self):
        session = self.submit('failure', fail=True)
        chunks, error = await self.complete(session)
        self.assertEqual(chunks, [])
        self.assertIsInstance(error, RuntimeError)
        self.service.stopping = True
        await asyncio.wait_for(self.worker, 2)
        record = json.loads(self.telemetry.getvalue())
        self.assertIn('backend failure', record['status'])
        self.assertEqual(record['generated_frames'], 0)

    async def test_http_pcm_and_request_id_telemetry_contract(self):
        app = web.Application()
        app.router.add_post('/v1/audio/speech', self.service.speech)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        url = f'http://127.0.0.1:{runner.addresses[0][1]}/v1/audio/speech'
        try:
            async with aiohttp.ClientSession() as client:
                async with client.post(url, json={'input': 'hello', 'stream': True,
                                                  'response_format': 'pcm'},
                                       headers={'X-Request-Id': 'http-request'}) as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(await response.read(), b'\x01\x00' * 3)
                record = json.loads(self.telemetry.getvalue())
                self.assertEqual(record['request_id'], 'http-request')
                self.assertEqual(record['generated_frames'], 3)
                self.assertEqual(record['status'], 'complete')
                async with client.post(url, json={'input': 'hello', 'stream': False,
                                                  'response_format': 'wav'}) as response:
                    self.assertEqual(response.status, 400)
                reference = {'input': 'hello', 'stream': True, 'response_format': 'pcm',
                             'ref_text': 'reference speech', 'ref_audio_codes': [[1, 2]] * 10}
                async with client.post(url, json=reference) as response:
                    self.assertEqual(response.status, 200)
                    await response.read()
                for invalid in ([[1]] * 9, [[4096]] + [[0]] * 9,
                                [[0]] + [[1024]] * 9, [[True]] * 10,
                                [[0, 1]] + [[0]] * 9):
                    async with client.post(url, json={**reference, 'ref_audio_codes': invalid}) as response:
                        self.assertEqual(response.status, 400)
                self.backend.fail_after = 1
                async with client.post(url, json={'input': 'hello', 'stream': True,
                                                  'response_format': 'pcm'},
                                       headers={'X-Request-Id': 'http-failure'}) as response:
                    self.assertEqual(response.status, 200)
                    with self.assertRaises(aiohttp.ClientPayloadError):
                        await response.read()
        finally:
            await runner.cleanup()


class BatchBackend(Backend):
    def __init__(self):
        super().__init__()
        self.batches = []

    def step_batch(self, sessions):
        self.batches.append(tuple(session.request_id for session in sessions))
        results = []
        for session in sessions:
            try:
                results.append(self.step(session))
            except Exception as error:
                results.append(error)
        return results


class BatchedServiceTests(ServiceTests):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.backend = BatchBackend()
        self.service.backend = self.backend

    async def test_actual_batch_dispatch_and_per_request_error(self):
        bad, good = self.submit('bad', fail=True), self.submit('good')
        failed, successful = await asyncio.gather(self.complete(bad), self.complete(good))
        self.assertIsInstance(failed[1], RuntimeError)
        self.assertEqual(len(successful[0]), 3)
        self.assertEqual(self.backend.batches[0], ('bad', 'good'))
        self.assertEqual(self.backend.batches[1:], [('good',), ('good',)])
        self.assertEqual(sorted(self.backend.closed), ['bad', 'good'])
