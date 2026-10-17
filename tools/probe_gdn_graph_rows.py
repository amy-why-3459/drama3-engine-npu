"""Check real graph replay state, bucket padding and both NPU contexts."""
import argparse
import copy
import json
import tempfile
from pathlib import Path
import torch
import torch_npu
from drama3.model.backbone import Drama3Model
from drama3.model.npu_graph import GraphGDN, GraphGDNDecoder
from tests.test_model import tiny_checkpoint


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--whole-layer', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tiny_checkpoint(root, overrides={'text_config': {
            'linear_key_head_dim': 128, 'linear_value_head_dim': 128,
            'num_hidden_layers': 4, 'layer_types': ['linear_attention', 'full_attention'] * 2,
            'mlp_layer_types': ['moe'] * 4}})
        model = Drama3Model.load(root, ['npu:0', 'npu:1'], max_length=32,
                                use_npu_gdn_decode=True, use_npu_gdn_prefill=True, use_npu_norm=True, use_npu_moe=args.whole_layer)
        rows = []
        for index in (0, 2):
            layer = model.backbone.model.layers[index]
            module = layer if args.whole_layer else layer.linear_attn
            wrapper = GraphGDNDecoder(module) if args.whole_layer else GraphGDN(module)
            device = next(module.parameters()).device
            for batch in (1, 3, 2, 1):
                prompt = torch.tensor([[[1, 0, 0, 0], [3, 0, 0, 0]]] * batch)
                seed = model.hidden(prompt).past_key_values
                eager, graphed = copy.deepcopy(seed), copy.deepcopy(seed)
                for step in range(4):
                    inputs = torch.randn(batch, 1, 32, dtype=torch.bfloat16, device=device)
                    if args.whole_layer:
                        expected = module(inputs, position_embeddings=None, past_key_values=eager, attention_mask=None)
                        actual = wrapper(inputs, position_embeddings=None, past_key_values=graphed, attention_mask=None, use_cache=True)
                    else:
                        expected = module(inputs, cache_params=eager, attention_mask=None)
                        actual = wrapper(inputs, cache_params=graphed, attention_mask=None, use_cache=True)
                    torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=.002, rtol=.002)
                    for name in ('conv_states', 'recurrent_states'):
                        torch.testing.assert_close(getattr(eager.layers[index], name)[0].cpu(),
                                                   getattr(graphed.layers[index], name)[0].cpu(), atol=2e-5, rtol=2e-5)
                rows.append({'device': str(device), 'batch': batch, 'buckets': sorted(wrapper.entries)})
            for entry in wrapper.entries.values():
                assert entry.capture_stream.device == device
            wrapper.clear_graphs()
        print(json.dumps({'status': 'PASS', 'cases': rows}), flush=True)


if __name__ == '__main__':
    main()
