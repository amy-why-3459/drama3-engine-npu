"""Stress graph/stream lifetime at full-model graph counts in a worker thread."""
import gc
from concurrent.futures import ThreadPoolExecutor
import torch
import torch_npu


def probe():
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    entries = []
    with torch.inference_mode(), torch.npu.device(0):
        for i in range(400):
            x = torch.ones(256, device='npu:0')
            graph = torch.npu.NPUGraph()
            stream = torch.npu.Stream(device='npu:0')
            with torch.npu.graph(graph, stream=stream):
                output = x * 2 + 1
            graph.replay()
            entries.append((graph, stream, x, output))
        torch.npu.synchronize()
        assert entries[-1][-1].cpu().tolist() == [3.] * 256
        del graph, stream, x, output
        while entries:
            entry = entries.pop()
            entry[0].reset()
            del entry
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        print('400_GRAPHS_RELEASED', flush=True)


with ThreadPoolExecutor(max_workers=1) as executor:
    executor.submit(probe).result()
print('EXECUTOR_CLOSED', flush=True)
