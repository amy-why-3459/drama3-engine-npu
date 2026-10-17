"""Run the mixed-arrival generation regression on a real NPU with tiny weights."""
import argparse
import unittest
from unittest.mock import patch

import torch
import torch_npu

from drama3.model.backbone import Drama3Model
from tests.test_batching import BatchedGenerationTest
from tests.test_model import tiny_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npu-int8-experts', action='store_true')
    parser.add_argument('--npu-conv', action='store_true')
    parser.add_argument('--npu-tower-graph', action='store_true')
    parser.add_argument('--npu-layer-graph', action='store_true')
    parser.add_argument('--npu-gdn-graph', action='store_true')
    parser.add_argument('--npu-norm', action='store_true')
    parser.add_argument('--devices', nargs='+', default=['npu:0'])
    args = parser.parse_args()
    torch.npu.set_device(0)
    load = Drama3Model.load
    loaded = []

    def on_npu(directory, devices, **kwargs):
        model = load(directory, args.devices, use_npu_attention=True,
                    use_npu_gdn_prefill=True, use_npu_gdn_decode=True, use_npu_norm=args.npu_norm, use_npu_gdn_graph=args.npu_gdn_graph, use_npu_layer_graph=args.npu_layer_graph, use_npu_tower_graph=args.npu_tower_graph,
                    use_npu_moe=args.npu_layer_graph or args.npu_int8_experts, use_npu_int8_experts=args.npu_int8_experts, use_npu_conv=args.npu_conv, **kwargs)
        loaded.append(model)
        return model

    def checkpoint(directory):
        return tiny_checkpoint(directory, overrides={
            'text_config': {'linear_key_head_dim': 128, 'linear_value_head_dim': 128,
                            'num_hidden_layers': 4,
                            'layer_types': ['linear_attention', 'full_attention'] * 2,
                            'mlp_layer_types': ['moe'] * 4},
            'audio_decoder_config': {'head_dim': 128},
        })

    with patch.object(Drama3Model, 'load', side_effect=on_npu), patch('tests.test_batching.tiny_checkpoint', side_effect=checkpoint):
        suite = unittest.TestSuite([BatchedGenerationTest('test_matches_reference_with_mixed_lengths_and_arrivals'),
                                    BatchedGenerationTest('test_three_request_padding_and_removal')])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.npu_gdn_graph or args.npu_layer_graph:
        from drama3.model.npu_graph import GraphGDN

        wrappers = [module for model in loaded for module in model.modules() if isinstance(module, GraphGDN)]
        buckets = [sorted(module.entries) for module in wrappers]
        print('CAPTURED_GDN_BUCKETS', buckets, flush=True)
        assert wrappers and all(entries for entries in buckets), 'Graph path was not exercised'
        for module in wrappers:
            module.clear_graphs()
    if args.npu_tower_graph:
        from drama3.model.npu_tower_graph import GraphTowerLayer
        wrappers = [m for model in loaded for m in model.modules() if isinstance(m, GraphTowerLayer)]
        assert wrappers and all(m.entries for m in wrappers)
        with torch.inference_mode():
            for m in wrappers:
                width = m.module.eh_proj.out_features
                for rows in (1, 3, 2):
                    hidden = torch.randn(rows, width, device=m.device, dtype=next(m.parameters()).dtype)
                    codes = torch.zeros(rows, device=m.device, dtype=torch.long)
                    valid = torch.ones(rows, device=m.device, dtype=torch.bool)
                    extra = torch.randn_like(hidden)
                    positions = torch.arange(rows, device=m.device)
                    attention = m.module.transformer_layer.attention
                    attended = torch.randn(rows, attention.n_head, attention.head_dim, device=m.device, dtype=hidden.dtype)
                    gate = torch.randn(rows, attention.n_head * attention.head_dim, device=m.device, dtype=hidden.dtype)
                    for stage, tensors, function in (
                        ('before', (hidden, codes, valid, extra, positions), m.before_attention),
                        ('after', (attended, gate, hidden), m.after_attention),
                    ):
                        expected = function(*tensors)
                        actual = m.run_graph(stage, tensors)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        print('CAPTURED_TOWER_BUCKETS', [sorted(m.entries) for m in wrappers], flush=True)
        for m in wrappers:
            m.clear_graphs()
    print('NPU_TINY_BATCHED_GENERATION_PASS', flush=True)


if __name__ == '__main__':
    main()
