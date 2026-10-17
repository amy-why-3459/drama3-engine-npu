"""Validate left-padded prefill and continued decode on Ascend."""
import unittest
from unittest.mock import patch
import torch
import torch_npu
from drama3.model.backbone import Drama3Model
from tests.test_model import tiny_checkpoint
from tests.test_prefill import PrefillTest

torch.npu.set_device(0)
load=Drama3Model.load

def npu_load(directory,devices,**kwargs):
    return load(directory,['npu:0'],use_npu_moe=True,use_npu_attention=True,
                use_npu_gdn_prefill=True,use_npu_gdn_decode=True,**kwargs)

def checkpoint(directory):
    return tiny_checkpoint(directory,overrides={
        'text_config':{'linear_key_head_dim':128,'linear_value_head_dim':128,
                       'num_hidden_layers':4,'layer_types':['linear_attention','full_attention']*2,
                       'mlp_layer_types':['moe']*4},
        'audio_decoder_config':{'head_dim':128}})

with patch.object(Drama3Model,'load',side_effect=npu_load),patch('tests.test_prefill.tiny_checkpoint',side_effect=checkpoint):
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrefillTest))
if not result.wasSuccessful():raise SystemExit(1)
print('BATCHED_PREFILL_NPU_PASS',flush=True)
