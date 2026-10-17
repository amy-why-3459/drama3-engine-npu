"""Profile real streaming backend steps; timings are diagnostic, not an RTF benchmark."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch_npu
from drama3.serving import BatchedBackend, Session


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--workload',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--optimized',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    config=SimpleNamespace(model=args.model,devices=['npu:0','npu:1'],max_active=1,
        context_length=2048,cpu_threads=4,seed=42,npu_moe=True,npu_attention=True,
        npu_gdn_prefill=True,npu_gdn_decode=True,npu_norm=True,npu_gdn_graph=False,
        npu_layer_graph=False,npu_tower_graph=False,prefill_batch_size=1,
        first_chunk_frames=2,chunk_frames=25,left_context_frames=25)
    if args.optimized:
        config.npu_layer_graph = True
        config.npu_tower_graph = True
        config.npu_conv = True
        config.codec_batch_size = 8
    backend=BatchedBackend(config)
    payload=json.loads(args.workload.read_text().splitlines()[0])
    payload['sampling']={'temperature':1.,'top_k':30,'top_p':.9,'max_frames':100}
    session=Session('device-profile',payload)
    def step():
        result=backend.step_batch([session])[0]
        if isinstance(result,Exception):raise result
        if result[1]:raise RuntimeError('EOS before diagnostic window completed')
    try:
        for _ in range(5):step()
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,with_stack=False,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output)),
        ) as profiler:
            for _ in range(3):
                step()
                profiler.step()
        print('DEVICE_PROFILE_COMPLETE_NOT_BENCHMARK',flush=True)
    finally:
        backend.close(session)
        backend.shutdown()


if __name__=='__main__':main()
