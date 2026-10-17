import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from drama3.batching import BatchedGenerator
from drama3.generation import ReferenceGenerator, Sampling
from drama3.model.backbone import Drama3Model
from tests.test_model import tiny_checkpoint


class BatchedGenerationTest(unittest.TestCase):
    def test_matches_reference_with_mixed_lengths_and_arrivals(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_checkpoint(root)
            model = Drama3Model.load(root, ['cpu'], cache_slots=3, max_length=32)
            prompts = [torch.tensor([[1, 0, 0, 0], [3, 0, 0, 0]]),
                       torch.tensor([[1, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0]])]
            sampling = Sampling(temperature=0, max_frames=4)

            def semantic(hidden):
                logits = torch.full((len(hidden), 17), -100., device=hidden.device)
                logits[:, 1] = 100.
                return logits

            with patch.object(model, 'semantic_logits', side_effect=semantic):
                ref = ReferenceGenerator(model)
                expected = [list(ref.frames(prompt, sampling)) for prompt in prompts]
                generator = BatchedGenerator(model)
                first = generator.prefill(prompts[0], sampling)
                actual = [[generator.step([first])[0]], []]
                second = generator.prefill(prompts[1], sampling)
                for _ in range(3):
                    rows = generator.step([second, first])
                    actual[0].append(rows[1])
                    actual[1].append(rows[0])
                self.assertIsNone(generator.step([first])[0])
                generator.close(first)
                actual[1].append(generator.step([second])[0])
                for got, want in zip(actual, expected):
                    torch.testing.assert_close(torch.stack(got), torch.stack(want), rtol=0, atol=0)
                generator.close(second)
                generator.close(second)
                self.assertEqual(generator.active_slots, set())
                with self.assertRaises(ValueError):
                    generator.step([second])
                replacement = generator.prefill(prompts[0], sampling)
                with self.assertRaises(ValueError):
                    generator.step([replacement, replacement])
                generator.close(replacement)

    def test_three_request_padding_and_removal(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_checkpoint(root)
            model = Drama3Model.load(root, ['cpu'], cache_slots=3, max_length=32)
            prompts = [torch.tensor([[1, 0, 0, 0]] + [[i + 3, 0, 0, 0]] * length)
                       for i, length in enumerate((1, 2, 3))]
            sampling = Sampling(temperature=0, max_frames=4)

            def semantic(hidden):
                logits = torch.full((len(hidden), 17), -100., device=hidden.device)
                logits[:, 1] = 100.
                return logits

            with patch.object(model, 'semantic_logits', side_effect=semantic):
                expected = [list(ReferenceGenerator(model).frames(prompt, sampling)) for prompt in prompts]
                generator = BatchedGenerator(model)
                states = [generator.prefill(prompt, sampling) for prompt in prompts]
                actual = [[], [], []]
                for _ in range(2):
                    for i, frame in enumerate(generator.step(states)):
                        actual[i].append(frame)
                generator.close(states[1])
                for _ in range(2):
                    frames = generator.step([states[2], states[0]])
                    actual[2].append(frames[0])
                    actual[0].append(frames[1])
                for i in range(3):
                    torch.testing.assert_close(torch.stack(actual[i]), torch.stack(expected[i][:len(actual[i])]), atol=0, rtol=0)
                generator.close(states[0])
                generator.close(states[2])
                self.assertEqual(generator.active_slots, set())

    def test_eos_and_prefill_error_release(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tiny_checkpoint(root)
            model = Drama3Model.load(root, ['cpu'], cache_slots=2, max_length=32)
            generator = BatchedGenerator(model)
            prompt = torch.tensor([[1, 0, 0, 0], [3, 0, 0, 0]])
            with patch.object(model, 'hidden', side_effect=RuntimeError('prefill failed')):
                with self.assertRaisesRegex(RuntimeError, 'prefill failed'):
                    generator.prefill(prompt, Sampling(max_frames=3))
            self.assertEqual(generator.active_slots, set())
            state = generator.prefill(prompt, Sampling(max_frames=3))
            logits = torch.full((1, 17), -100.)
            logits[:, 0] = 100.
            with patch.object(model, 'semantic_logits', return_value=logits):
                self.assertEqual(generator.step([state]), [None])
                self.assertEqual(state.generated, 0)
            generator.close(state)
            self.assertEqual(generator.active_slots, set())
