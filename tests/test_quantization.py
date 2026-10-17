import unittest
import torch
from drama3.model.quantization import quantize_expert_weight


class QuantizationTest(unittest.TestCase):
    def test_per_output_channel_range_and_zero_channels(self):
        weight=torch.tensor([[[0.,0.,0.],[-127.,0.,127.]],
                             [[-1.,.5,1.],[4.,-2.,0.]]])
        q,scale=quantize_expert_weight(weight)
        self.assertEqual(q.dtype,torch.int8)
        self.assertEqual(scale.dtype,torch.bfloat16)
        self.assertTrue(torch.isfinite(scale).all())
        self.assertTrue((scale>0).all())
        self.assertEqual(q[0,0].tolist(),[0,0,0])
        self.assertEqual(q[0,1].tolist(),[-127,0,127])
        restored=q.float()*scale.float()[...,None]
        # Include BF16 scale rounding in the error bound.
        error=(weight-restored).abs()
        bound=scale.float()[...,None]*.51 + weight.abs()*.004
        self.assertTrue((error<=bound).all())
