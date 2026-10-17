"""Experimental W8A8 grouped experts: eager/graph parity, not quality acceptance."""
import json
import torch
import torch_npu
from torch import nn
from drama3.model.npu_experts import NpuExperts
from drama3.model.quantization import quantize_expert_weight
from drama3.model.npu_tower_graph import RowGraph


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(731)
    torch.npu.set_device(0)
    reference=nn.Module()
    reference.num_experts=256
    reference.gate_up_proj=nn.Parameter(torch.randn(256,1024,2048,dtype=torch.bfloat16)*.03,requires_grad=False)
    reference.down_proj=nn.Parameter(torch.randn(256,2048,512,dtype=torch.bfloat16)*.03,requires_grad=False)
    for name in ('gate_up_proj','down_proj'):
        q,scale=quantize_expert_weight(getattr(reference,name))
        setattr(reference,name,nn.Parameter(q.npu(),requires_grad=False))
        reference.register_buffer(name+'_scale',scale.npu())
    module=NpuExperts(reference)
    del reference
    for batch in (1,3,8,64):
        hidden=torch.randn(batch,2048,device='npu:0',dtype=torch.bfloat16)
        probs,ids=torch.randn(batch,256,device='npu:0').softmax(-1).topk(8)
        probs/=probs.sum(-1,keepdim=True)
        graph=RowGraph(module,(hidden,ids,probs),batch)
        for _ in range(3):
            hidden.normal_()
            expected=module(hidden,ids,probs)
            actual=graph.replay((hidden,ids,probs))
            torch.testing.assert_close(actual.cpu(),expected.cpu(),atol=0,rtol=0)
            assert torch.isfinite(actual).all()
        graph.graph.reset()
        del graph
        print(json.dumps(dict(batch=batch,status='EXACT_INT8_EAGER_GRAPH')),flush=True)
    print('INT8_GRAPH_PARITY_ONLY_QUALITY_NOT_VALIDATED',flush=True)


if __name__=='__main__': main()
