import copy
import unittest

from drama3.metrics import RequestTrace, compare, summarize


class MetricsTests(unittest.TestCase):
    def traces(self):
        return [
            RequestTrace("a", 0, 0.1, 1, 1.2, 44100, 44100, 22, True),
            RequestTrace("b", 0, 0.3, 2, 2.2, 88200, 44100, 43, True),
        ]

    def test_last_audio_not_terminal_controls_rtf(self):
        result = summarize(self.traces(), 2)
        self.assertAlmostEqual(result["throughput_frames_s"], 65 / 2.2)
        self.assertEqual(result["ttfa_p50_ms"], 200)
        self.assertEqual(result["ttfa_p95_ms"], 290)
        self.assertEqual(result["rtf_mean"], 1)

    def test_missing_frame_telemetry_is_not_inferred_from_pcm(self):
        traces = self.traces()
        traces[0] = RequestTrace("a", 0, 0.1, 1, 1.2, 44100, 44100, None, True)
        result = summarize(traces, 2)
        self.assertIsNone(result["throughput_frames_s"])
        self.assertFalse(result["eligible_for_comparison"])

    def test_failure_remains_in_wall_time_and_blocks_promotion(self):
        traces = self.traces() + [
            RequestTrace("c", 0, None, None, 10, 0, 44100, None, False, "timeout")
        ]
        result = summarize(traces, 3)
        self.assertEqual(result["wall_time_s"], 10)
        self.assertFalse(result["eligible_for_comparison"])
        self.assertEqual(result["failed_requests"], 1)

    def test_missing_or_duplicate_requests_are_rejected(self):
        with self.assertRaises(ValueError):
            summarize(self.traces(), 3)
        with self.assertRaises(ValueError):
            summarize([self.traces()[0]] * 2, 2)

    def test_comparison_requires_stronger_baseline_and_quality(self):
        contract = dict.fromkeys(
            [
                "model_revision",
                "model_hashes",
                "hardware",
                "device_count",
                "workload_sha256",
                "concurrency",
                "sampling",
                "frame_count_convention",
                "quality_contract_sha256",
                "precision",
                "streaming_policy",
            ],
            "fixture",
        )
        candidate = {
            "engine": "candidate",
            "contract": contract,
            "source_revision": "fixture",
            "clean_exit": True,
            "quality": "PASS",
            "quality_evidence": "fixture.json",
            "metrics": {"eligible_for_comparison": True, "throughput_frames_s": 200},
        }
        baselines = [copy.deepcopy(candidate), copy.deepcopy(candidate)]
        baselines[0]["engine"] = "vllm-omni"
        baselines[0]["metrics"]["throughput_frames_s"] = 80
        baselines[1]["engine"] = "sglang-omni"
        baselines[1]["metrics"]["throughput_frames_s"] = 110
        self.assertEqual(compare(candidate, baselines)["verdict"], "FAIL")
        baselines[1]["metrics"]["throughput_frames_s"] = 100
        self.assertEqual(compare(candidate, baselines)["verdict"], "PASS")
        candidate["quality"] = "NOT_RUN"
        self.assertEqual(compare(candidate, baselines)["verdict"], "INCONCLUSIVE")

    def test_invalid_timing_and_nan_fail_closed(self):
        bad = RequestTrace("a", 0, 1, 0.5, 2, 44100, 44100, 22, True)
        with self.assertRaises(ValueError):
            summarize([bad], 1)
        bad = RequestTrace("a", 0, 0.1, 1, float("nan"), 44100, 44100, 22, True)
        with self.assertRaises(ValueError):
            summarize([bad], 1)


if __name__ == "__main__":
    unittest.main()
