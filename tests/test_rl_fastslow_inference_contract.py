import unittest
from pathlib import Path


NODE = (
    Path(__file__).resolve().parents[1]
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real"
    / "rl_fastslow_inference_onnx_v1.py"
)


class RosInferenceContractTests(unittest.TestCase):
    def test_node_is_candidate_only_and_logs_context(self):
        source = NODE.read_text(encoding="utf-8")
        self.assertIn("/uav_payload/observation21", source)
        self.assertIn("/uav_payload/rl_ctbr_candidate", source)
        self.assertIn("/uav_payload/rl_context", source)
        self.assertIn("/uav_payload/rl_inference_status", source)
        self.assertIn("create_timer(1.0 / 60.0", source)
        self.assertIn("config/manifest.json", source)
        self.assertIn("/tmp/vio_current_run_dir", source)
        self.assertNotIn("VehicleCommand", source)
        self.assertNotIn("OffboardControlMode", source)
        self.assertNotIn("VehicleRatesSetpoint", source)


if __name__ == "__main__":
    unittest.main()
