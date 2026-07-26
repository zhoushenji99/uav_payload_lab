import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/audit_z_dataset.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_z_dataset", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ZDatasetAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_module()

    def _write_dataset(self, root: Path, slow_offset: float = 0.0):
        inputs = torch.zeros(6, 50, 21, dtype=torch.float16)
        labels_ml = torch.tensor(
            [[0.0, 0.0], [0.2, 0.8], [1.0, 1.0]] * 2,
            dtype=torch.float32,
        )
        labels = torch.cat(
            [
                labels_ml + slow_offset,
                torch.zeros(6, 3, dtype=torch.float32),
            ],
            dim=1,
        )
        torch.save(
            {"inputs": inputs, "labels": labels, "labels_ml": labels_ml},
            root / "shard_0000.pt",
        )
        torch.save(
            {
                "teacher_context_mode": "split_hard",
                "history_len": 50,
                "input_dim": 21,
                "z_dim": 5,
                "z_exp_dim": 2,
                "total_samples": 6,
            },
            root / "meta.pt",
        )

    def test_hard_explicit_dataset_passes_exact_identity_and_coverage_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_dataset(root)

            report = self.module.audit_dataset(root)

        self.assertTrue(report["passed"])
        self.assertEqual(report["slow_label_identity_max_abs"], 0.0)
        self.assertEqual(report["normalized_slow_coverage_span"], [1.0, 1.0])
        self.assertEqual(report["total_samples"], 6)

    def test_hard_explicit_dataset_rejects_any_slow_label_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_dataset(root, slow_offset=1e-4)

            report = self.module.audit_dataset(root)

        self.assertFalse(report["passed"])
        self.assertGreater(report["slow_label_identity_max_abs"], 0.0)
        self.assertFalse(report["hard_identity_ok"])

    def test_nonfinite_input_is_reported(self):
        inputs = torch.zeros(2, 50, 21)
        inputs[0, 0, 0] = float("nan")
        labels_ml = torch.zeros(2, 2)
        labels = torch.zeros(2, 5)

        report = self.module.audit_shard_tensors(inputs, labels, labels_ml)

        self.assertFalse(report["all_finite"])
        self.assertEqual(report["inputs_nonfinite"], 1)


if __name__ == "__main__":
    unittest.main()
