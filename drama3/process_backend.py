"""A spawned main-thread owner for NPU inference, with explicit exit verification.

Only plain request data, PCM bytes, counters and error strings cross the pipe.
No device tensors or model objects are transferred to the HTTP process.
"""
import multiprocessing
import time


def _serve(connection, args, factory):
    from drama3.serving import BatchedBackend, Session

    backend = None
    sessions = {}
    try:
        backend = (factory or BatchedBackend)(args)
        connection.send(('ready', None))
        while True:
            command, payload = connection.recv()
            if command == 'step':
                active = []
                for request_id, request in payload:
                    if request_id not in sessions:
                        if request is None:
                            raise RuntimeError('Missing new request payload')
                        sessions[request_id] = Session(request_id, request)
                    active.append(sessions[request_id])
                results = backend.step_batch(active)
                if len(results) != len(active):
                    raise RuntimeError('Backend returned an incomplete batch')
                rows = []
                for session, result in zip(active, results):
                    error = f'{type(result).__name__}: {result}' if isinstance(result, Exception) else None
                    rows.append((None if error else result, error,
                                 session.generated_frames, session.emitted_frames))
                connection.send(('ok', rows))
            elif command == 'close':
                session = sessions.pop(payload, None)
                if session is not None:
                    backend.close(session)
                connection.send(('ok', None))
            elif command == 'shutdown':
                break
            else:
                raise RuntimeError(f'Unknown backend command: {command}')
    finally:
        try:
            if backend is not None:
                for session in sessions.values():
                    backend.close(session)
                backend.shutdown()
        finally:
            connection.close()
    # Normal interpreter exit invokes the runtime's original atexit handlers
    # on this same main thread. The parent must join and inspect exitcode.


class ProcessBackend:
    def __init__(self, args, *, factory=None, startup_timeout=900, rpc_timeout=120,
                 shutdown_timeout=120):
        self.rpc_timeout = rpc_timeout
        self.shutdown_timeout = shutdown_timeout
        self.known = set()
        self.closed = False
        context = multiprocessing.get_context('spawn')
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(target=_serve, args=(child_connection, args, factory),
                                       name='drama3-inference')
        self.process.start()
        child_connection.close()
        try:
            kind, _ = self._receive(startup_timeout)
            if kind != 'ready':
                raise RuntimeError('Invalid inference process startup response')
        except BaseException:
            self._abort()
            raise

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.connection.poll(min(.1, max(0., deadline-time.monotonic()))):
                try:
                    return self.connection.recv()
                except EOFError as error:
                    self.process.join(1)
                    raise RuntimeError(f'Inference process disconnected (exit {self.process.exitcode})') from error
            if not self.process.is_alive():
                raise RuntimeError(f'Inference process exited with {self.process.exitcode}')
        raise TimeoutError('Inference process response timed out')

    def _abort(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(5)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(5)
        self.connection.close()
        self.closed = True

    def _rpc(self, command, payload):
        if self.closed:
            raise RuntimeError('Inference process is closed')
        try:
            self.connection.send((command, payload))
            kind, result = self._receive(self.rpc_timeout)
            if kind != 'ok':
                raise RuntimeError('Invalid inference process response')
            return result
        except BaseException:
            # A timed-out pipe is no longer synchronized. Never consume a stale
            # response as the result of a different request.
            self._abort()
            raise

    def step_batch(self, sessions):
        payload = [(s.request_id, None if s.request_id in self.known else s.payload) for s in sessions]
        rows = self._rpc('step', payload)
        if len(rows) != len(sessions):
            self._abort()
            raise RuntimeError('Incomplete inference process batch')
        results = []
        for session, (result, error, generated, emitted) in zip(sessions, rows):
            self.known.add(session.request_id)
            session.generated_frames, session.emitted_frames = generated, emitted
            results.append(RuntimeError(error) if error else result)
        return results

    def close(self, session):
        if session.request_id in self.known:
            if not self.closed and self.process.is_alive():
                self._rpc('close', session.request_id)
            self.known.discard(session.request_id)

    def shutdown(self):
        if not self.closed and self.process.is_alive():
            try:
                self.connection.send(('shutdown', None))
            except (BrokenPipeError, EOFError):
                pass
            self.process.join(self.shutdown_timeout)
        if self.process.is_alive():
            self._abort()
            raise RuntimeError('Inference process did not exit after cleanup')
        self.connection.close()
        self.closed = True
        if self.process.exitcode != 0:
            raise RuntimeError(f'Inference process cleanup failed (exit {self.process.exitcode})')
        print('INFERENCE_PROCESS_EXIT_0', flush=True)
