"""Run the upstream scheduler with synthetic DRAMA-3 weights on an NPU."""

import copy
import os
from functools import partial
from pathlib import Path
import queue
import threading
import time

import torch
import torch_npu  # noqa: F401

from sglang_omni.models.fishaudio_drama3.engine_builder import Drama3EngineBuilder
from sglang_omni.models.fishaudio_drama3.request_builders import Drama3GenerationInput
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.fishaudio_drama3.mtp_fixtures import CHECKPOINT_CONFIG, write_synthetic_checkpoint

torch.set_num_threads(4)
torch.npu.set_device(0)
root = Path("/home/why/drama3-research/tiny-sglang")
config = copy.deepcopy(CHECKPOINT_CONFIG)
config["text_config"].update(
    head_dim=256,
    moe_intermediate_size=256,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
)
config["audio_decoder_config"].update(head_dim=256, rotary_percent=0.25)
write_synthetic_checkpoint(root, config)
print("WROTE_SYNTHETIC_CHECKPOINT", flush=True)
builder = Drama3EngineBuilder(
    context_length=128, max_running_requests=2, max_prompt_rows=64,
    max_new_tokens_cap=8, first_frame_cuda_graphs=False,
    triton_sampling=False, tower_fused_layers=False,
)
scheduler = builder.build(
    str(root), device="npu", gpu_id=0, dtype="bfloat16",
    server_args_overrides={"max_total_tokens": 256, "max_prefill_tokens": 128, "mem_fraction_static": 0.15},
)
print("BUILT_SYNTHETIC_ENGINE", flush=True)
def check_head(module, inputs, output):
    if not torch.isfinite(output).all():
        raise RuntimeError("Nonfinite codebook logits")


for head in builder.sglang_runner.model.audio_decoder.codebook_heads:
    head.register_forward_hook(check_head)
if os.environ.get("DRAMA3_PROBE_ALL_FINITE") == "1":
    def check_module(module, inputs, output, *, name):
        values = output if isinstance(output, (tuple, list)) else (output,)
        for value in values:
            if isinstance(value, torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                raise RuntimeError(f"Nonfinite output at {name}: {tuple(value.shape)}")

    for name, module in builder.sglang_runner.model.named_modules():
        module.register_forward_hook(partial(check_module, name=name))
failures = []


def serve():
    try:
        torch.npu.set_device(0)
        scheduler.start()
    except BaseException as error:
        failures.append(error)
        raise


thread = threading.Thread(target=serve, daemon=True)
thread.start()
try:
    prompt = torch.zeros((5, 11), dtype=torch.long)
    prompt[:, 0] = torch.tensor([1, 8, 9, 10, 11])
    payload = StagePayload(
        request_id="smoke", request=OmniRequest(inputs="synthetic"),
        data=Drama3GenerationInput(prompt_rows=prompt, max_new_tokens=4, temperature=1.0, top_k=1, top_p=1.0),
    )
    scheduler.inbox.put(IncomingMessage(request_id="smoke", type="new_request", data=payload))
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if failures:
            raise RuntimeError("Scheduler failed") from failures[0]
        try:
            message = scheduler.outbox.get(timeout=1)
        except queue.Empty:
            continue
        print(f"OUTBOX {message.type}", flush=True)
        if message.type == "error":
            raise RuntimeError(str(message))
        if message.type == "result":
            frames = message.data.data.frames
            assert frames.ndim == 2 and frames.shape[1] == 11
            assert 0 < len(frames) <= 4
            assert ((frames[:, 1:] >= 0) & (frames[:, 1:] < config["audio_decoder_config"]["vocab_size"])).all()
            assert (frames[:, 0] == config["semantic_start_token_id"] + frames[:, 1]).all()
            print(f"VALID_SYNTHETIC_FRAMES {frames.tolist()}", flush=True)
            print("SYNTHETIC_SGLANG_ENGINE_NPU_PASS", flush=True)
            break
    else:
        raise TimeoutError("Synthetic request did not finish")
finally:
    scheduler.stop()
    thread.join(timeout=10)
    if thread.is_alive():
        raise RuntimeError("Scheduler did not stop")
