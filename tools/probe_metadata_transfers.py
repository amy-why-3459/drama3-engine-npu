"""Compare metadata reuse with the original per-layer transfer hooks."""
import tempfile
from functools import partial
from pathlib import Path
import torch
import torch_npu
from drama3.model.backbone import Drama3Model, move_tensors
from tests.test_model import tiny_checkpoint


def legacy_hook(module, inputs, keywords, *, device):
    return move_tensors(inputs, device), move_tensors(keywords, device)


@torch.inference_mode()
def main():
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        tiny_checkpoint(root, overrides={'text_config': {
            'linear_key_head_dim':128,'linear_value_head_dim':128,
            'num_hidden_layers':4,'layer_types':['linear_attention','full_attention']*2,
            'mlp_layer_types':['moe']*4}})
        options=dict(devices=['npu:0','npu:1'],use_npu_moe=True,
                     use_npu_gdn_decode=True,use_npu_gdn_prefill=True)
        reference=Drama3Model.load(root,**options)
        candidate=Drama3Model.load(root,**options)
        for hook in reference.layer_hooks: hook.remove()
        for layer in reference.backbone.model.layers:
            layer.register_forward_pre_hook(partial(legacy_hook,device=next(layer.parameters()).device),with_kwargs=True)
        for batch in (1,3,2):
            frames=torch.tensor([[[1,0,0,0],[3,0,0,0],[2,0,0,0]]]*batch)
            expected=reference.hidden(frames)
            actual=candidate.hidden(frames)
            torch.testing.assert_close(actual.last_hidden_state.cpu(),expected.last_hidden_state.cpu(),atol=0,rtol=0)
            assert not candidate.metadata_transfers
            for step in range(4):
                frames=torch.tensor([[[3,0,0,0]]]*batch)
                expected=reference.hidden(frames,cache=expected.past_key_values)
                actual=candidate.hidden(frames,cache=actual.past_key_values)
                torch.testing.assert_close(actual.last_hidden_state.cpu(),expected.last_hidden_state.cpu(),atol=0,rtol=0)
                assert not candidate.metadata_transfers
            print(f'B{batch}_PREFILL_AND_DECODE_EXACT',flush=True)
    print('METADATA_TRANSFER_PARITY_PASS',flush=True)


if __name__=='__main__': main()
