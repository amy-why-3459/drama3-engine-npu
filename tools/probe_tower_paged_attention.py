"""Check standalone native tower attention against an independent CPU reference."""
import json

import torch
import torch_npu

from drama3.model.npu_attention import paged_decode


def main():
    torch.set_num_threads(2)
    torch.manual_seed(731)
    torch.npu.set_device(0)
    results = []
    for dimension in (128, 256):
        keys = torch.randn(6, 96, 2, dimension, dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        for slots, lengths in (([4, 1, 3], [1, 17, 65]), ([3, 4], [66, 2]), ([1], [96])):
            query = torch.randn(len(slots), 16, dimension, dtype=torch.bfloat16)
            blocks = torch.tensor(slots, dtype=torch.int32)[:, None] * 6 + torch.arange(6, dtype=torch.int32)[None]
            actual = paged_decode(query.npu(), keys.npu(), values.npu(), blocks.npu(), torch.tensor(lengths, dtype=torch.int32)).cpu().float()
            expected = []
            for row, (slot, length) in enumerate(zip(slots, lengths)):
                k = keys[slot, :length].float().repeat_interleave(8, dim=1)
                v = values[slot, :length].float().repeat_interleave(8, dim=1)
                scores = torch.einsum('hd,thd->ht', query[row].float(), k) * dimension ** -.5
                expected.append(torch.einsum('ht,thd->hd', scores.softmax(-1), v))
            expected = torch.stack(expected)
            torch.testing.assert_close(actual, expected, atol=.02, rtol=.02)
            results.append(dict(head_dim=dimension, slots=slots, lengths=lengths, max_abs=float((actual - expected).abs().max())))
    print(json.dumps(dict(status='PASS', cases=results), indent=2), flush=True)


if __name__ == '__main__':
    main()
