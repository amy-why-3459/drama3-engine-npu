"""Compare output and mutable state against CPU FP32 convolution reference."""
import json
import time
import torch
import torch_npu
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import torch_causal_conv1d_update
from drama3.model.npu_conv import causal_conv_update

torch.set_num_threads(4)
torch.npu.set_device(0)
torch.manual_seed(23)
records=[]
for batch in (1,8,64):
    width=8192
    weight=torch.randn(width,4,dtype=torch.bfloat16)*.1
    state=torch.randn(batch,width,4,dtype=torch.bfloat16)*.1
    actual_state=state.npu()
    actual_weight=weight.npu()
    error=0.
    for _ in range(16):
        x=torch.randn(batch,width,1,dtype=torch.bfloat16)*.1
        expected=torch_causal_conv1d_update(x,state,weight,activation='silu')
        actual=causal_conv_update(x.npu(),actual_state,actual_weight,activation='silu').cpu()
        torch.testing.assert_close(actual,expected,atol=5e-4,rtol=.02)
        torch.testing.assert_close(actual_state.cpu(),state,atol=0,rtol=0)
        error=max(error,(actual-expected).abs().max().item())
    x=x.npu()
    row={'batch':batch,'max_abs_error':error,'dispatch':'fused' if batch <= 8 else 'native_fallback','timings':[]}
    for fused in (False,True,True,False):
        fn=causal_conv_update if fused else torch_causal_conv1d_update
        for _ in range(3):fn(x,actual_state,actual_weight,activation='silu')
        torch.npu.synchronize();start=time.perf_counter()
        for _ in range(20):fn(x,actual_state,actual_weight,activation='silu')
        torch.npu.synchronize()
        row['timings'].append({'fused':fused,'ms':(time.perf_counter()-start)*50})
    records.append(row)
print(json.dumps(records,indent=2),flush=True)
print('CONV_UPDATE_COMPONENT_PASS',flush=True)
