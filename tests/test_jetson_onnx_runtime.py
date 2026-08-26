import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "uav_payload_lab"
    / "uav_payload_lab"
    / "tasks"
    / "direct"
    / "uav_payload_sim2real"
)
sys.path.insert(0, str(MODULE_DIR))

from jetson_inference_trace import InferenceTraceWriter  # noqa: E402
from jetson_onnx_runtime import FastSlowOnnxRuntime  # noqa: E402


class FakeSession:
    def __init__(self, kind):
        self.kind = kind
        self.calls = 0

    def run(self, _outputs, feed):
        self.calls += 1
        value = next(iter(feed.values()))
        if self.kind == "slow":
            return [np.repeat(value[:, -1, :1], 2, axis=1).astype(np.float32)]
        if self.kind == "fast":
            return [np.repeat(value[:, -1, 1:2], 3, axis=1).astype(np.float32)]
        return [np.array([[1.0, -3.0, 3.0, 2.0]], dtype=np.float32)]


class FastSlowOnnxRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.slow = FakeSession("slow")
        self.fast = FakeSession("fast")
        self.actor = FakeSession("actor")
        self.runtime = FastSlowOnnxRuntime(
            self.slow, self.fast, self.actor, input_names=("history", "history", "actor_input")
        )

    def test_history_fill_readiness_schedule_and_clamp(self):
        records = []
        for step in range(242):
            observation = np.zeros(21, dtype=np.float32)
            observation[17:21] = float(step)
            records.append(self.runtime.step(observation, observation_age_sec=0.01))

        self.assertEqual(records[0]["history_fill_count"], 1)
        self.assertEqual(records[49]["history_fill_count"], 50)
        self.assertFalse(records[178]["candidate_ready"])
        self.assertTrue(records[179]["candidate_ready"])
        self.assertTrue(records[180]["slow_updated"])
        self.assertFalse(records[181]["slow_updated"])
        self.assertTrue(records[240]["slow_updated"])
        self.assertEqual(self.slow.calls, 182)
        self.assertEqual(self.fast.calls, 242)
        np.testing.assert_allclose(
            records[-1]["action_clamped"], [0.0, -2.5, 2.5, 1.5]
        )
        np.testing.assert_allclose(records[-1]["previous_executed_ctbr"], [241.0] * 4)

    def test_invalid_input_revokes_ready_and_resets_history(self):
        for _ in range(180):
            ready = self.runtime.step(np.zeros(21, dtype=np.float32))
        self.assertTrue(ready["candidate_ready"])

        rejected = self.runtime.step(
            np.full(21, np.nan, dtype=np.float32), observation_age_sec=0.0
        )

        self.assertFalse(rejected["candidate_ready"])
        self.assertEqual(rejected["reject_reason"], "nonfinite_observation")
        self.assertEqual(rejected["history_fill_count"], 0)
        recovered = self.runtime.step(np.zeros(21, dtype=np.float32))
        self.assertEqual(recovered["history_fill_count"], 1)
        self.assertFalse(recovered["candidate_ready"])

    def test_stale_input_is_rejected(self):
        record = self.runtime.step(np.zeros(21), observation_age_sec=0.2)
        self.assertEqual(record["reject_reason"], "stale_observation")
        self.assertFalse(record["observation_valid"])

    def test_context_is_clamped_and_severe_overflow_blocks_candidate(self):
        for _ in range(179):
            self.runtime.step(np.zeros(21, dtype=np.float32))
        moderate = np.zeros(21, dtype=np.float32)
        moderate[0] = 1.1
        record = self.runtime.step(moderate)
        self.assertTrue(record["context_out_of_range"])
        self.assertFalse(record["context_severe_out_of_range"])
        self.assertAlmostEqual(float(record["context_raw"][0]), 1.1, places=6)
        self.assertAlmostEqual(float(record["context_clamped"][0]), 1.0, places=6)
        self.assertTrue(record["candidate_ready"])

        severe = np.zeros(21, dtype=np.float32)
        severe[0] = 2.0
        blocked = self.runtime.step(severe)
        self.assertTrue(blocked["context_severe_out_of_range"])
        self.assertFalse(blocked["candidate_ready"])
        self.assertEqual(blocked["reject_reason"], "context_severe_out_of_range")

    def test_trace_contains_reconstructable_fields(self):
        record = self.runtime.step(np.arange(21, dtype=np.float32))
        record.update(
            wall_time_ns=1,
            monotonic_time_ns=2,
            px4_timestamp_us=3,
            observation_timestamp_ns=4,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.csv"
            with InferenceTraceWriter(path, flush_every=1) as writer:
                writer.write(record)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 1)
        row = rows[0]
        for name in (
            "obs_0",
            "obs_20",
            "z0",
            "z4",
            "z_raw0",
            "z_clamped4",
            "context_out_of_range",
            "context_severe_out_of_range",
            "slow_raw_0",
            "slow_target_1",
            "slow_cache_0",
            "action_raw_thrust",
            "action_clamped_pitch_rate",
            "previous_ctbr_yaw_rate",
            "slow_latency_us",
            "end_to_end_latency_us",
            "candidate_ready",
            "reject_reason",
        ):
            self.assertIn(name, row)


if __name__ == "__main__":
    unittest.main()
