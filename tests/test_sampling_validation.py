import unittest
import torch
from drama3.model.tower import sample_rows


class SamplingValidationTest(unittest.TestCase):
    def test_deferred_validation_preserves_rng_and_outputs(self):
        logits = torch.arange(128, dtype=torch.float32).reshape(4, 32) / 100
        args = (torch.ones(4), torch.full((4,), 30), torch.full((4,), .9))
        torch.manual_seed(42)
        expected = [sample_rows(logits, *args) for _ in range(9)]
        torch.manual_seed(42)
        flags = []
        actual = [sample_rows(logits, *args, invalid_flags=flags) for _ in range(9)]
        self.assertFalse(torch.stack(flags).any().item())
        torch.testing.assert_close(torch.stack(actual), torch.stack(expected), rtol=0, atol=0)

    def test_invalid_rows_are_retained_by_deferred_check(self):
        args = (torch.ones(1), torch.tensor([2]), torch.ones(1))
        for logits in (torch.tensor([[float('nan'), 0.]]),
                       torch.tensor([[float('inf'), 0.]]),
                       torch.full((1, 2), -float('inf'))):
            with self.assertRaises(ValueError):
                sample_rows(logits, *args)
            flags = []
            sample_rows(logits, *args, invalid_flags=flags)
            self.assertTrue(torch.stack(flags).any().item())
