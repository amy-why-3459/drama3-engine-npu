import unittest

import torch

from drama3.model.checkpoint import expand_block_fp8


class BlockFp8Tests(unittest.TestCase):
    def test_partial_blocks_and_float32_scales(self):
        weight = torch.tensor([[1.0, -2.0, 3.0], [4.0, 5.0, -6.0], [7.0, 8.0, 9.0]]).to(
            torch.float8_e4m3fn
        )
        scale = torch.tensor([[0.3333333, 2.0], [0.125, 0.75]], dtype=torch.float32)
        expected = torch.tensor(
            [
                [1 * 0.3333333, -2 * 0.3333333, 6.0],
                [4 * 0.3333333, 5 * 0.3333333, -12.0],
                [0.875, 1.0, 6.75],
            ]
        ).to(torch.bfloat16)
        self.assertTrue(torch.equal(expand_block_fp8(weight, scale, (2, 2)), expected))

    def test_rejects_incorrect_scale_shape(self):
        with self.assertRaises(ValueError):
            expand_block_fp8(torch.zeros(128, 128), torch.ones(2, 1))


if __name__ == "__main__":
    unittest.main()
