"""Prototype graph replay for a cached standalone GDN module; no serving dispatch."""
import argparse
import copy
import json
import tempfile
from pathlib import Path

import torch
import torch_npu

from drama3.model.backbone import Drama3Model
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
        tiny_checkpoint(root, overrides={
            'text_config': {'linear_key_head_dim': 128, 'linear_value_head_dim': 128}})
        model = Drama3Model.load(root, ['npu:0'], max_length=32,
                                use_npu_gdn_decode=True, use_npu_gdn_prefill=True,
                                use_npu_norm=True, use_npu_moe=args.whole_layer)
        prompt = torch.tensor([[[1, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]]])
        seed_cache = model.hidden(prompt).past_key_values
        layer = model.backbone.model.layers[0]
        def run(inputs, cache):
            if args.whole_layer:
                return layer(inputs, position_embeddings=None, past_key_values=cache, attention_mask=None)
            return layer.linear_attn(inputs, cache_params=cache, attention_mask=None)
        static_cache = copy.deepcopy(seed_cache)
        inputs = torch.randn(1, 1, 32, dtype=torch.bfloat16, device='npu')
        for _ in range(3):
            run(inputs, static_cache)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = run(inputs, static_cache)
        results = []
        for step in range(4):
            inputs.copy_(torch.randn_like(inputs))
            reference_cache = copy.deepcopy(seed_cache)
            expected = run(inputs, reference_cache)
            for name in ('conv_states', 'recurrent_states'):
                getattr(static_cache.layers[0], name)[0].copy_(getattr(seed_cache.layers[0], name)[0])
            graph.replay()
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), expected.cpu(), atol=.02, rtol=.02)
            for name in ('conv_states', 'recurrent_states'):
                torch.testing.assert_close(getattr(static_cache.layers[0], name)[0].cpu(),
                                           getattr(reference_cache.layers[0], name)[0].cpu(), atol=.002, rtol=.002)
            results.append({'step': step, 'max_abs': float((output-expected).abs().max())})
            seed_cache = reference_cache
        graph.reset()
        print(json.dumps({'status': 'PASS', 'whole_layer': args.whole_layer, 'scope': 'GDN graph prototype only', 'steps': results}), flush=True)


if __name__ == '__main__':
    main()
