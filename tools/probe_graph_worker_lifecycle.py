"""Reproduce serving-style graph ownership and cleanup in an executor thread."""
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor


devices = sys.argv[1:] or ["npu:0"]


def probe():
    import gc
    import torch
    import torch_npu
    sys.argv = ['probe_batched_generation_npu.py', '--npu-layer-graph', '--npu-tower-graph', '--npu-norm', '--devices', *devices]
    runpy.run_path('tools/probe_batched_generation_npu.py', run_name='__main__')
    gc.collect()
    torch.npu.synchronize()
    torch.npu.empty_cache()
    print('WORKER_GRAPH_CLEANUP_COMPLETE', flush=True)


with ThreadPoolExecutor(max_workers=1) as executor:
    executor.submit(probe).result()
print('EXECUTOR_CLOSED', flush=True)
