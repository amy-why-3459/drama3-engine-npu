"""Full codec weights: equal-length batch parity and component timing."""
import json
import time
from pathlib import Path
import torch
import torch_npu
from drama3.model.codec import AudioCodec

torch.set_num_threads(4)
torch.npu.set_device(0)
torch.npu.conv.allow_hf32 = False
torch.npu.matmul.allow_hf32 = False
torch.manual_seed(42)
codec=AudioCodec(Path('/dataset/models/drama-3/codec.pth'),'npu:0',torch.float32)
records=[]
for frames in (2,27,50):
    codes=torch.randint(0,1024,(8,10,frames),device='npu')
    codes[:,0]=torch.randint(0,4096,(8,frames),device='npu')
    def run(batch):
        return codec.decode(codes) if batch else torch.cat([codec.decode(x[None]) for x in codes])
    expected=run(False)
    actual=run(True)
    error=(actual-expected).abs().max().item()
    torch.testing.assert_close(actual,expected,atol=3e-5,rtol=1e-4)
    timings=[]
    for batch in (False,True,True,False):
        torch.npu.synchronize()
        start=time.perf_counter()
        for _ in range(2):run(batch)
        torch.npu.synchronize()
        timings.append({'batch':batch,'ms_per_eight_chunks':(time.perf_counter()-start)*500})
    records.append({'frames':frames,'max_abs_error':error,'timings':timings})
print(json.dumps(records,indent=2),flush=True)
print('CODEC_BATCH_COMPONENT_PASS_NOT_QUALITY_OR_RTF',flush=True)
