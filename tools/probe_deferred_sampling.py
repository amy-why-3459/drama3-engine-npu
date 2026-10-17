"""Matched component timing; not a serving RTF or speech quality benchmark."""
import json
import time
import torch
import torch_npu
from drama3.model.tower import sample_rows


def main():
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    rows = []
    for batch in (1, 64):
        logits = torch.randn(batch, 1024, device='npu', dtype=torch.bfloat16)
        args = (torch.ones(batch, device='npu'), torch.full((batch,), 30, device='npu'),
                torch.full((batch,), .9, device='npu'))
        def run(deferred):
            flags = [] if deferred else None
            values = [sample_rows(logits, *args, invalid_flags=flags) for _ in range(9)]
            if flags is not None and torch.stack(flags).any().item():
                raise ValueError('Invalid logits')
            return torch.stack(values)
        torch.manual_seed(42)
        before = run(False)
        torch.manual_seed(42)
        after = run(True)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        for mode in (False, True):
            for _ in range(3): run(mode)
        for mode in (False, True, True, False):
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(10): run(mode)
            torch.npu.synchronize()
            rows.append({'batch': batch, 'deferred': mode, 'ms_per_nine_samples':
                         (time.perf_counter() - start) * 100})
    print(json.dumps({'status': 'COMPONENT_ONLY', 'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
