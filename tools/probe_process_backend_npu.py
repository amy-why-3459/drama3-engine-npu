"""Verify main-thread NPU graph ownership and normal child interpreter exit."""
import sys
import threading
from types import SimpleNamespace
from drama3.process_backend import ProcessBackend
from drama3.serving import Session


class GraphBackend:
    def __init__(self, args):
        import torch
        import torch_npu
        assert threading.current_thread() is threading.main_thread()
        torch.set_num_threads(2)
        self.torch = torch
        self.entries = {}
        with torch.inference_mode():
            for device in args.devices:
                entries = self.entries[device] = []
                with torch.npu.device(device):
                    for _ in range(40):
                        x = torch.ones(16,device=device)
                        graph = torch.npu.NPUGraph()
                        stream = torch.npu.Stream(device=device)
                        with torch.npu.graph(graph,stream=stream):
                            y = x*2+1
                        entries.append((graph,stream,x,y))
    def step_batch(self, sessions):
        assert threading.current_thread() is threading.main_thread()
        torch = self.torch
        with torch.inference_mode():
            for device, entries in self.entries.items():
                with torch.npu.device(device):
                    for graph,stream,x,y in entries:
                        x.fill_(2)
                        graph.replay()
                    value = int(y[0].item())
        results=[]
        for session in sessions:
            session.generated_frames+=1
            session.emitted_frames+=1
            results.append((bytes([value,session.generated_frames]),session.generated_frames==2))
        return results
    def close(self,session):
        pass
    def shutdown(self):
        import gc
        assert threading.current_thread() is threading.main_thread()
        torch = self.torch
        for device, entries in self.entries.items():
            with torch.npu.device(device):
                torch.npu.synchronize()
                while entries:
                    entry=entries.pop()
                    entry[0].reset()
                    del entry
                gc.collect()
                torch.npu.empty_cache()
        print('CHILD_MAIN_THREAD_GRAPH_CLEANUP_COMPLETE',flush=True)


def main():
    backend=ProcessBackend(SimpleNamespace(devices=['npu:0','npu:1']),factory=GraphBackend)
    session=Session('probe',{})
    try:
        assert backend.step_batch([session])==[(b'\x05\x01',False)]
        assert backend.step_batch([session])==[(b'\x05\x02',True)]
        backend.close(session)
    finally:
        backend.shutdown()
    assert backend.process.exitcode == 0
    assert 'torch_npu' not in sys.modules
    print('PROCESS_NPU_GRAPH_PASS_PARENT_HAS_NO_TORCH_NPU',flush=True)


if __name__=='__main__': main()
