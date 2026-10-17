import tempfile
import unittest
from pathlib import Path

import torch

from drama3.model.backbone import Drama3Model
from drama3.model.batch_cache import DecodeCacheBatch
from tests.test_model import tiny_checkpoint


class BatchCacheTest(unittest.TestCase):
    def test_variable_lengths_reordering_and_removed_request(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_checkpoint(root)
            model = Drama3Model.load(root, ['cpu'], cache_slots=3, max_length=32)
            prompts = [torch.tensor([[1, 0, 0, 0], [3, 0, 0, 0]]),
                       torch.tensor([[1, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0], [6, 0, 0, 0]])]
            with torch.inference_mode():
                reference = [model.hidden(p[None]).past_key_values for p in prompts]
                actual = [model.hidden(p[None]).past_key_values for p in prompts]
                for order in ([0, 1], [1, 0], [0]):
                    frames = [torch.tensor([[[64 + i, 0, 1, 2]]]) for i in order]
                    expected = [model.hidden(f, cache=reference[i]).last_hidden_state
                                for i, f in zip(order, frames)]
                    batch = DecodeCacheBatch([actual[i] for i in order])
                    before = [cache.get_seq_length() for cache in actual]
                    snapshots = [[layer.keys.clone() if hasattr(layer, 'keys') else layer.recurrent_states[0].clone()
                                  for layer in actual[i].layers] for i in order]
                    result = model.hidden(torch.cat(frames), cache=batch.cache,
                                          **batch.model_inputs('cpu'))
                    self.assertEqual(before, [cache.get_seq_length() for cache in actual])
                    for i, snapshot in zip(order, snapshots):
                        for layer, previous in zip(actual[i].layers, snapshot):
                            value = layer.keys if hasattr(layer, 'keys') else layer.recurrent_states[0]
                            torch.testing.assert_close(value, previous, rtol=0, atol=0)
                    torch.testing.assert_close(result.last_hidden_state, torch.cat(expected), atol=.025, rtol=.025)
                    batch.commit()
                    with self.assertRaises(RuntimeError):
                        batch.commit()
                    for i in order:
                        self.assertEqual(actual[i].get_seq_length(), reference[i].get_seq_length())
                        for got, want in zip(actual[i].layers, reference[i].layers):
                            if hasattr(got, 'keys'):
                                torch.testing.assert_close(got.keys, want.keys, atol=.02, rtol=.02)
                                torch.testing.assert_close(got.values, want.values, atol=.02, rtol=.02)
                            else:
                                torch.testing.assert_close(got.conv_states[0], want.conv_states[0], atol=.02, rtol=.02)
                                torch.testing.assert_close(got.recurrent_states[0], want.recurrent_states[0], atol=.02, rtol=.02)
                with self.assertRaises(ValueError):
                    DecodeCacheBatch([actual[0], actual[0]])
                batch = DecodeCacheBatch([actual[0]])
                with self.assertRaises(ValueError):
                    batch.commit()
