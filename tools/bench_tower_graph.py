"""Component-only ABBA timing of a real-shape tower layer, not an RTF result."""
import json
import time
import torch
import torch_npu
from drama3.model.configuration_fish_omni import FishOmniAudioDecoderConfig
from drama3.model.tower import TowerLayer, TowerBatch
from drama3.model.npu_tower_graph import GraphTowerLayer


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(731)
    torch.npu.set_device(0)
    config = FishOmniAudioDecoderConfig(dim=2048, intermediate_size=6144,
        n_head=16, n_local_heads=2, head_dim=256, rotary_dim=64, vocab_size=4096)
    module = TowerLayer(config, 'codebook_embedding').to(device='npu:0', dtype=torch.bfloat16).eval()
    module.transformer_layer.attention.use_npu_paged_attention = True
    wrapper = GraphTowerLayer(module)
    results = []
    for rows in (1, 8, 64):
        keys = torch.randn(rows, 256, 2, 256, device='npu:0', dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        hidden = torch.randn(rows, 2048, device='npu:0', dtype=torch.bfloat16)
        codes = torch.randint(4096, (rows,), device='npu:0')
        valid = torch.ones(rows, device='npu:0', dtype=torch.bool)
        extra = torch.randn_like(hidden)
        batch = TowerBatch.decode(torch.arange(rows, device='npu:0'),
            torch.full((rows,), 128, device='npu:0'), positions_cpu=[128]*rows, cache_length=256)
        args = (hidden, codes, valid, extra, batch, keys, values)
        expected = module(*args).clone()
        actual = wrapper(*args)
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
        for _ in range(5): module(*args); wrapper(*args)
        for label, function in (('eager', module), ('graph', wrapper), ('graph', wrapper), ('eager', module)):
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(30): function(*args)
            torch.npu.synchronize()
            results.append(dict(batch=rows, arm=label, mean_ms=(time.perf_counter()-start)*1000/30))
    wrapper.clear_graphs()
    print(json.dumps(dict(status='PASS', correctness='exact', scope='one real-shape tower layer, shared host', rounds=results)), flush=True)


if __name__ == '__main__': main()
