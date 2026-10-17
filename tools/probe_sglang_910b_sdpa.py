"""Diagnose tower numerics using the existing Torch SDPA attention backend."""
import runpy
from pathlib import Path

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang_omni.models.fishaudio_drama3 import engine_builder, model_runner

engine_builder.ATTENTION_BACKEND = 'torch_native'
model_runner.FULL_ATTENTION_BACKEND = TorchNativeAttnBackend
runpy.run_path(str(Path(__file__).with_name('probe_sglang_910b_prefill.py')), run_name='__main__')
