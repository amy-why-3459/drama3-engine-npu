"""Exercise graph buckets, changing slots/positions and all cache writes."""
import json
import torch
import torch_npu
from drama3.model.configuration_fish_omni import FishOmniAudioDecoderConfig
from drama3.model.tower import TowerLayer, TowerBatch
from drama3.model.npu_tower_graph import GraphTowerLayer


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.manual_seed(731)
    torch.npu.set_device(0)
    config = FishOmniAudioDecoderConfig(dim=128, intermediate_size=256,
        n_head=4, n_local_heads=2, head_dim=128, rotary_dim=64, vocab_size=128)
    module = TowerLayer(config, 'codebook_embedding').to(device='npu:0', dtype=torch.bfloat16).eval()
    results = []
    for native in (False, True):
        module.transformer_layer.attention.use_npu_paged_attention = native
        wrapper = GraphTowerLayer(module)
        keys = torch.randn(6, 256, 2, 128, device='npu:0', dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        actual_keys, actual_values = keys.clone(), values.clone()
        for slots, positions in (([4], [1]), ([4, 1, 3], [2, 17, 65]),
                                 ([3, 4], [66, 3]), ([1], [18]),
                                 ([3, 1, 4], [67, 19, 4])):
            rows = len(slots)
            hidden = torch.randn(rows, 128, device='npu:0', dtype=torch.bfloat16)
            codes = torch.randint(128, (rows,), device='npu:0')
            valid = torch.tensor([i % 2 == 0 for i in range(rows)], device='npu:0')
            extra = torch.randn_like(hidden)
            batch = TowerBatch.decode(torch.tensor(slots, device='npu:0'),
                torch.tensor(positions, device='npu:0'), positions_cpu=positions, cache_length=256)
            expected = module(hidden, codes, valid, extra, batch, keys, values)
            actual = wrapper(hidden, codes, valid, extra, batch, actual_keys, actual_values)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
            torch.testing.assert_close(actual_keys.cpu(), keys.cpu(), atol=0, rtol=0)
            torch.testing.assert_close(actual_values.cpu(), values.cpu(), atol=0, rtol=0)
            results.append(dict(native=native, slots=slots, positions=positions, exact=True))
        wrapper.clear_graphs()
    print(json.dumps(dict(status='PASS', cases=results)), flush=True)


if __name__ == '__main__':
    main()
