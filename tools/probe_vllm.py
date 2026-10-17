"""Read-only quantization compatibility check against the installed Ascend runtime."""

import importlib.util
import sys
from pathlib import Path

from vllm.platforms import current_platform
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.layer import RoutedExperts
from types import SimpleNamespace

root = Path(sys.argv[1])
path = (
    root
    if root.is_file()
    else root / "vllm_omni/model_executor/models/fish_omni/quantization.py"
)
spec = importlib.util.spec_from_file_location("drama3_baseline_quant", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print("platform:", type(current_platform).__name__, flush=True)
print("supported quantization:", current_platform.supported_quantization, flush=True)
config = module.ExpertFp8Config.from_config(
    {"weight_block_size": [128, 128], "activation_scheme": "dynamic"}
)
print("quant config:", config, flush=True)
current_platform.verify_quantization(config.get_name())
print("platform verification passed", flush=True)


parallel = FusedMoEParallelConfig(
    tp_size=1,
    pcp_size=1,
    dp_size=1,
    ep_size=1,
    tp_rank=0,
    pcp_rank=0,
    dp_rank=0,
    ep_rank=0,
    sp_size=1,
    use_ep=False,
    all2all_backend="naive",
    enable_eplb=False,
)
moe = FusedMoEConfig(
    num_experts=256,
    experts_per_token=8,
    hidden_dim=2048,
    intermediate_size=512,
    num_local_experts=256,
    num_logical_experts=256,
    activation=MoEActivation.SILU,
    device="npu:0",
    routing_method=RoutingMethodType.Default,
    moe_parallel_config=parallel,
    in_dtype=torch.bfloat16,
)
layer = RoutedExperts.__new__(RoutedExperts)
torch.nn.Module.__init__(layer)
layer.moe_config = moe
fixture_config = VllmConfig()
fixture_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
with set_current_vllm_config(fixture_config):
    method = config.get_quant_method(layer, "model.layers.0.mlp.experts")
    print("selected MoE method:", type(method).__name__, flush=True)
    print(
        "backend:",
        getattr(method, "fp8_backend", None),
        getattr(method, "experts_cls", None),
        flush=True,
    )
    if hasattr(method, "experts_cls") and method.experts_cls is None:
        print(
            "BLOCKED: registered Drama3 quantization selected no executable MoE experts backend",
            flush=True,
        )
        sys.exit(2)
