"""Real DRAMA dimensions with synthetic weights: state parity and process exit."""
import gc
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
import torch_npu
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
from drama3.model.configuration_fish_omni import to_qwen3_5_moe_text_config
from drama3.model.npu_experts import NpuExperts
from drama3.model.npu_graph import GraphGDNDecoder, StaticGDNCache
from drama3.model.npu_recurrent import recurrent_decode
from drama3.model.npu_conv import causal_conv_update


@torch.inference_mode()
def probe():
    torch.set_num_threads(4)
    torch.manual_seed(731)
    config = to_qwen3_5_moe_text_config(json.loads(Path('/dataset/models/drama-3/config.json').read_text())['text_config'])
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        for device in ('npu:0', 'npu:1'):
            with torch.npu.device(device):
                with torch.device('meta'):
                    layer = Qwen3_5MoeDecoderLayer(config, 0)
                layer.to_empty(device=device)
                for p in layer.parameters(): p.normal_(0, .01)
                layer.mlp.experts = NpuExperts(layer.mlp.experts)
                layer.linear_attn.recurrent_gated_delta_rule = recurrent_decode
                layer.linear_attn.causal_conv1d_update = causal_conv_update
                graph = GraphGDNDecoder(layer)
                for batch in (1, 3, 8, 17, 32, 64, 2, 1):
                    conv = torch.randn(batch, 8192, 4, device=device, dtype=torch.bfloat16)*.01
                    state = torch.randn(batch, 32, 128, 128, device=device, dtype=torch.float32)*.01
                    eager = StaticGDNCache(0, conv.clone(), state.clone())
                    actual = StaticGDNCache(0, conv.clone(), state.clone())
                    actual.layers[0].number_of_states = 1
                    actual.layers[0].record_past = False
                    for step in range(2):
                        hidden = torch.randn(batch, 1, 2048, device=device)
                        expected = layer(hidden, position_embeddings=None, past_key_values=eager)
                        output = graph(hidden, position_embeddings=None, past_key_values=actual)
                        torch.testing.assert_close(output.cpu(), expected.cpu(), atol=0, rtol=0)
                        for name in ('conv_states', 'recurrent_states'):
                            torch.testing.assert_close(getattr(actual.layers[0], name)[0].cpu(),
                                getattr(eager.layers[0], name)[0].cpu(), atol=0, rtol=0)
                    print(json.dumps(dict(device=device, batch=batch, status='EXACT')), flush=True)
                graph.clear_graphs()
                del graph, layer, actual, eager, output, expected, state, conv, hidden, p
                gc.collect()
                torch.npu.empty_cache()
                torch.npu.synchronize()
    finally:
        torch.set_default_dtype(previous)
    print('REAL_SHAPE_GDN_GRAPH_CLEANUP_COMPLETE', flush=True)


with ThreadPoolExecutor(max_workers=1) as executor:
    executor.submit(probe).result()
print('EXECUTOR_CLOSED', flush=True)
