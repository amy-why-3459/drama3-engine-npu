import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from drama3.generation import ReferenceGenerator, Sampling
from drama3.model.backbone import Drama3Model
from drama3.model.tower import sample_top_k_top_p
from tests.test_model import tiny_checkpoint


class GenerationTest(unittest.TestCase):
    def test_invalid_logits_cannot_become_legal_code_ids(self):
        for logits in ([0., float('nan')], [0., float('inf')],
                       [float('-inf'), float('-inf')]):
            with self.assertRaises(ValueError):
                sample_top_k_top_p(torch.tensor([logits]), 1., 2, .9)
        result = sample_top_k_top_p(torch.tensor([[0., float('-inf')]]), 0., 2, .9)
        self.assertEqual(result.item(), 0)

    def test_frames_and_early_close_release_state(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tiny_checkpoint(root)
            model = Drama3Model.load(root, ["cpu"], max_length=32)
            generator = ReferenceGenerator(model)
            prompt = torch.tensor([[1, 0, 0, 0], [3, 0, 0, 0]])
            # Force a semantic token to exercise every decode/cache iteration.
            logits = torch.full((1, 17), -100.0)
            logits[0, 1] = 100.0
            with patch.object(model, "semantic_logits", return_value=logits):
                stream = generator.frames(prompt, Sampling(max_frames=3))
                self.assertEqual(next(stream).shape, (3,))
                self.assertFalse(torch.is_inference_mode_enabled())
                self.assertEqual(len(generator.active_slots), 1)
                stream.close()
                self.assertEqual(generator.active_slots, set())
                result = list(generator.frames(prompt, Sampling(max_frames=3)))
                self.assertEqual(len(result), 3)
                self.assertTrue(all(frame[0].item() == 0 for frame in result))
                self.assertEqual(generator.active_slots, set())
            logits[0, 0] = 200.0
            with patch.object(model, "semantic_logits", return_value=logits):
                self.assertEqual(
                    list(generator.frames(prompt, Sampling(max_frames=3))), []
                )
                self.assertEqual(generator.active_slots, set())
