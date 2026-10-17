"""Test a scoped FP32-state NPU prefill substitution in the synthetic scheduler."""
import runpy
from pathlib import Path

from sgl_kernel_npu.fla.chunk import chunk_gated_delta_rule_npu
from sglang.srt.hardware_backend.npu.attention.ascend_gdn_backend import AscendGDNAttnBackend


def chunk_gdn(self, q, k, v, g, beta, initial_state, cu_seqlens):
    if self.forward_metadata.has_mamba_track_mask:
        raise NotImplementedError('Probe does not support radix state tracking')
    if q.shape[-1] != v.shape[-1]:
        raise NotImplementedError('Probe is scoped to equal DRAMA-3 K/V head dimensions')
    output, final_state, _ = chunk_gated_delta_rule_npu(
        q=q.unsqueeze(0), k=k.unsqueeze(0), v=v.unsqueeze(0),
        g=g.unsqueeze(0), beta=beta.unsqueeze(0),
        initial_state=initial_state, cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=True, output_final_state=True,
    )
    return output.squeeze(0), final_state, None


AscendGDNAttnBackend.chunk_gdn = chunk_gdn
runpy.run_path(str(Path(__file__).with_name('probe_sglang_engine_npu.py')), run_name='__main__')
