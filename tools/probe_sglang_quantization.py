"""Check the NPU baseline's checkpoint expansion independently of serving."""

import torch

from sglang_omni.models.fishaudio_drama3.npu_quantization import (
    AscendExpertFp8Method,
    expand_expert_blocks,
)
from sglang_omni.models.fishaudio_drama3.quantization import ExpertFp8Config

torch.manual_seed(12)
weight = torch.randn(3, 259, 130).to(torch.float8_e4m3fn)
scales = torch.rand(3, 3, 2)
expected = (
    weight.float()
    * scales.repeat_interleave(128, 1).repeat_interleave(128, 2)[:, :259, :130]
).bfloat16()
actual = expand_expert_blocks(weight, scales)
torch.testing.assert_close(actual, expected, rtol=0, atol=0)
method = AscendExpertFp8Method(ExpertFp8Config().expert_fp8_config)
print(type(method).__name__, "CPU_FP8_EXPANSION_EXACT", flush=True)
