import json
import torch
import torch_npu  # noqa: F401 - registers the NPU device backend

torch.npu.set_device(0)
for name, operation in [
    (
        "bf16_matmul",
        lambda: torch.ones((128, 128), device="npu", dtype=torch.bfloat16)
        @ torch.ones((128, 128), device="npu", dtype=torch.bfloat16),
    ),
    (
        "fp8_allocation",
        lambda: torch.empty((2, 1024, 2048), device="npu", dtype=torch.float8_e4m3fn),
    ),
    (
        "fp8_cpu_to_npu",
        lambda: torch.ones((128, 128)).to(torch.float8_e4m3fn).to("npu"),
    ),
]:
    try:
        result = operation()
        torch.npu.synchronize()
        print(
            json.dumps(
                {"operation": name, "status": "PASS", "dtype": str(result.dtype)}
            ),
            flush=True,
        )
    except Exception as exc:
        print(
            json.dumps({"operation": name, "status": "FAIL", "error": str(exc)}),
            flush=True,
        )
