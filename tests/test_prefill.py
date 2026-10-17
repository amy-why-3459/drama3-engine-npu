import tempfile
import unittest
from pathlib import Path
import torch
from drama3.batching import BatchedGenerator
from drama3.generation import Sampling
from drama3.model.backbone import Drama3Model
from drama3.prefill import prefill_many
from tests.test_model import tiny_checkpoint


class PrefillTest(unittest.TestCase):
    lengths = (1, 3, 2)
    capacity = 32

    def test_chunk_boundaries(self):
        self.lengths = (1, 65, 129)
        self.capacity = 160
        self.test_left_padding_cache_state_and_continuation()

    def test_left_padding_cache_state_and_continuation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            tiny_checkpoint(root)
            model=Drama3Model.load(root,['cpu'],cache_slots=6,max_length=self.capacity)
            generator=BatchedGenerator(model)
            prompts=[torch.tensor([[1,0,0,0]] + [[i+3,0,0,0]]*(n-1)) for i,n in enumerate(self.lengths)]
            sampling=[Sampling(temperature=0,max_frames=4) for _ in prompts]
            expected=[generator.prefill(p,s) for p,s in zip(prompts,sampling)]
            actual=prefill_many(generator,prompts,sampling)
            for got,want in zip(actual,expected):
                self.assertEqual(got.cache.get_seq_length(),want.cache.get_seq_length())
                torch.testing.assert_close(got.hidden,want.hidden,atol=.025,rtol=.025)
                for a,b in zip(got.cache.layers,want.cache.layers):
                    if hasattr(a,'keys'):
                        torch.testing.assert_close(a.keys,b.keys,atol=.025,rtol=.025)
                        torch.testing.assert_close(a.values,b.values,atol=.025,rtol=.025)
                    else:
                        torch.testing.assert_close(a.conv_states[0],b.conv_states[0],atol=.025,rtol=.025)
                        torch.testing.assert_close(a.recurrent_states[0],b.recurrent_states[0],atol=.025,rtol=.025)
            for _ in range(3):
                torch.manual_seed(42)
                a=generator.step(actual)
                torch.manual_seed(42)
                b=generator.step(expected)
                for x,y in zip(a,b):
                    if x is None or y is None:self.assertIs(x,y)
                    else:torch.testing.assert_close(x,y,rtol=0,atol=0)
            for state in actual+expected:generator.close(state)
            self.assertFalse(generator.active_slots)
            with self.assertRaises(RuntimeError):
                prefill_many(generator, [prompts[0]]*7, [sampling[0]]*7)
            self.assertFalse(generator.active_slots)
