import unittest
from types import SimpleNamespace
import torch
from drama3.codec_batch import consume_many
from drama3.serving import ReferenceBackend, Session


class CodecBatchTest(unittest.TestCase):
    def test_stream_boundaries_pcm_and_eos_match_serial(self):
        calls = []
        class Codec:
            hop_length = 2
            def decode(self, codes):
                calls.append(tuple(codes.shape))
                return codes[:, 0].float().repeat_interleave(2, dim=-1)/100
        backend = SimpleNamespace(codec=Codec(), model=SimpleNamespace(devices=['cpu']),
                                  first_frames=2, chunk_frames=3, left_frames=2)
        serial = [Session(str(i), {}) for i in range(3)]
        batched = [Session(str(i), {}) for i in range(3)]
        for step in range(7):
            frames = [torch.tensor([step+1]*10) if step < limit else None for limit in (4, 6, 5)]
            expected = [ReferenceBackend.consume(backend, s, f) for s, f in zip(serial, frames)]
            actual = consume_many(backend, batched, frames, 2)
            self.assertEqual(actual, expected)
            for a, b in zip(batched, serial):
                self.assertEqual((a.generated_frames, a.emitted_frames), (b.generated_frames, b.emitted_frames))
                self.assertEqual(len(a.codes), len(b.codes))
        self.assertTrue(any(shape[0] == 2 for shape in calls))

    def test_nonfinite_row_is_not_emitted_and_other_row_succeeds(self):
        class Codec:
            hop_length = 2
            def decode(self, codes):
                return torch.tensor([[float('nan'), 0.], [.1, .2]])
        backend = SimpleNamespace(codec=Codec(),model=SimpleNamespace(devices=['cpu']),
                                  first_frames=1,chunk_frames=3,left_frames=2)
        sessions = [Session(str(i), {}) for i in range(2)]
        results = consume_many(backend, sessions, [torch.zeros(10,dtype=torch.long)]*2, 2)
        self.assertIsInstance(results[0], RuntimeError)
        self.assertEqual(sessions[0].emitted_frames, 0)
        self.assertEqual(sessions[1].emitted_frames, 1)
        self.assertEqual(len(results[1][0]), 4)

    def test_generated_model_indices_reach_codec_and_invalid_indices_fail(self):
        received = []
        class Codec:
            hop_length = 2
            def decode(self, codes):
                received.append(codes.clone())
                return torch.zeros(len(codes), codes.shape[-1]*2)
        backend = SimpleNamespace(codec=Codec(), model=SimpleNamespace(devices=['cpu']),
                                  first_frames=1, chunk_frames=3, left_frames=2)
        frame = torch.tensor([4095, 1024, 4095, 0, 0, 0, 0, 0, 0, 0])
        expected = ReferenceBackend.consume(backend, Session('serial', {}), frame)
        actual = consume_many(backend, [Session('batched', {})], [frame], 1)[0]
        self.assertEqual(actual, expected)
        self.assertTrue(all(torch.equal(x[0, :, 0], frame) for x in received))
        for invalid in (torch.full((10,), 4096), torch.full((10,), -1),
                        torch.zeros(9, dtype=torch.long), torch.zeros(10)):
            with self.assertRaisesRegex(RuntimeError, 'Invalid generated'):
                ReferenceBackend.consume(backend, Session('bad', {}), invalid)
            self.assertIsInstance(consume_many(backend, [Session('bad', {})], [invalid], 1)[0], RuntimeError)

    def test_upstream_dac_clamps_model_residual_vocabulary(self):
        from drama3.model.dac_modules.rvq import DownsampleResidualVectorQuantize
        seen = []
        def from_codes(codes):
            seen.append(codes.clone())
            return (torch.zeros(len(codes), 1, codes.shape[-1]),)
        codec = SimpleNamespace(
            semantic_quantizer=SimpleNamespace(codebook_size=4096, from_codes=from_codes),
            quantizer=SimpleNamespace(codebook_size=1024, from_codes=from_codes),
            post_module=lambda x: x, upsample=lambda x: x)
        codes = torch.tensor([[[4095], [1024], [4095], [1], [2], [3], [4], [5], [6], [7]]])
        DownsampleResidualVectorQuantize.decode(codec, codes)
        self.assertEqual(seen[0].flatten().tolist(), [4095])
        self.assertEqual(seen[1].flatten().tolist(), [1023, 1023, 1, 2, 3, 4, 5, 6, 7])
