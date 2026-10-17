import os
import time
import unittest
from types import SimpleNamespace
from drama3.process_backend import ProcessBackend
from drama3.serving import Session


class FakeBackend:
    def __init__(self, args):
        self.args = args
    def step_batch(self, sessions):
        if self.args.mode == 'crash':
            os._exit(7)  # Deliberate fake-worker failure, never a production exit path.
        if self.args.mode == 'timeout':
            time.sleep(10)
        results = []
        for session in sessions:
            if session.payload.get('fail'):
                results.append(ValueError('bad request'))
                continue
            session.generated_frames += 1
            session.emitted_frames += 1
            results.append((bytes([session.payload['code'], session.generated_frames]),
                            session.generated_frames >= 2))
        return results
    def close(self, session):
        session.closed = True
    def shutdown(self):
        if self.args.mode == 'shutdown_failure':
            raise RuntimeError('injected cleanup failure')


class ProcessBackendTest(unittest.TestCase):
    def make_backend(self, mode='normal', **kwargs):
        return ProcessBackend(SimpleNamespace(mode=mode), factory=FakeBackend,
                              startup_timeout=30, shutdown_timeout=10, **kwargs)

    def test_state_counters_errors_close_and_id_reuse(self):
        backend = self.make_backend()
        try:
            first, bad = Session('a', {'code':3}), Session('b', {'fail':True})
            result = backend.step_batch([first,bad])
            self.assertEqual(result[0], (b'\x03\x01',False))
            self.assertIsInstance(result[1],RuntimeError)
            self.assertEqual((first.generated_frames,first.emitted_frames),(1,1))
            self.assertEqual(backend.step_batch([first]),[(b'\x03\x02',True)])
            backend.close(first)
            backend.close(bad)
            replacement=Session('a',{'code':9})
            self.assertEqual(backend.step_batch([replacement]),[(b'\x09\x01',False)])
            backend.close(replacement)
        finally:
            backend.shutdown()
        self.assertEqual(backend.process.exitcode,0)

    def test_worker_crash_is_reported_and_cannot_pass_shutdown(self):
        backend=self.make_backend('crash')
        with self.assertRaisesRegex(RuntimeError,'process'):
            backend.step_batch([Session('a',{'code':1})])
        self.assertEqual(backend.process.exitcode,7)
        with self.assertRaisesRegex(RuntimeError,'exit 7'):
            backend.shutdown()

    def test_cleanup_failure_is_reported(self):
        backend=self.make_backend('shutdown_failure')
        with self.assertRaisesRegex(RuntimeError,'cleanup failed'):
            backend.shutdown()
        self.assertEqual(backend.process.exitcode,1)

    def test_timeout_terminates_only_owned_worker(self):
        backend=self.make_backend('timeout',rpc_timeout=.2)
        started=time.monotonic()
        with self.assertRaises(TimeoutError):
            backend.step_batch([Session('a',{'code':1})])
        self.assertFalse(backend.process.is_alive())
        self.assertLess(time.monotonic()-started,8)
        with self.assertRaisesRegex(RuntimeError,'cleanup failed'):
            backend.shutdown()
