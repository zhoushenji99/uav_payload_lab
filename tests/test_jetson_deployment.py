import sys
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


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

from jetson_deployment import (  # noqa: E402
    CNNContextEncoder,
    NormalizedActor,
    load_deployment_models,
    validate_student_checkpoint,
)
from jetson_reference_runtime import (  # noqa: E402
    FastSlowRuntime,
    causal_ema_alpha,
    load_torchscript_runtime,
)
from export_jetson_bundle import (  # noqa: E402
    REQUIRED_ARTIFACTS,
    _copy_required,
    build_manifest,
    load_parity_history,
    sha256_file,
    validate_existing_bundle_lineage,
)
from verify_jetson_bundle import (  # noqa: E402
    verify_checksums,
    verify_manifest,
    verify_torchscript,
)


class JetsonDeploymentTests(unittest.TestCase):
    def test_normalized_actor_matches_training_normalization(self):
        actor = nn.Sequential(nn.Linear(3, 2, bias=False))
        with torch.no_grad():
            actor[0].weight.copy_(
                torch.tensor([[1.0, 2.0, -1.0], [-2.0, 0.5, 3.0]])
            )
        mean = torch.tensor([[1.0, -2.0, 0.5]])
        std = torch.tensor([[2.0, 4.0, 0.25]])
        module = NormalizedActor(actor, mean, std, eps=1.0e-2)
        x = torch.tensor([[3.0, 2.0, 1.0]])

        expected = actor((x - mean) / (std + 1.0e-2))
        torch.testing.assert_close(module(x), expected, rtol=0.0, atol=0.0)

    def test_context_encoder_uses_fixed_history_contract(self):
        encoder = CNNContextEncoder(input_dim=21, history_len=50, output_dim=2)
        output = encoder(torch.zeros(4, 50, 21))
        self.assertEqual(tuple(output.shape), (4, 2))

        with self.assertRaisesRegex(ValueError, "history shape"):
            encoder(torch.zeros(4, 49, 21))

    def test_student_checkpoint_requires_split_hard_lineage(self):
        checkpoint = {
            "student_context_mode": "split",
            "teacher_context_mode": "monolithic",
            "input_dim": 21,
            "history_len": 50,
            "z_dim": 5,
            "z_slow_dim": 2,
            "z_fast_dim": 3,
            "state_dict": {},
        }
        with self.assertRaisesRegex(ValueError, "split_hard"):
            validate_student_checkpoint(checkpoint)

    def test_actual_checkpoints_load_expected_deployment_shapes(self):
        base = Path(
            "/home/shenji/uav_payload_lab/uav_payload_lab/logs/rsl_rl/"
            "uav_payload_sim2real_hover_rl/"
            "2026-08-19_21-26-54_hardexplicit_teacher_hover_seed42"
        )
        teacher = base / "model_3000.pt"
        student = (
            base
            / "StudentFastSlow_hover_model3000_seed42_noprobe"
            / "best_fast_slow_student_encoder_z.pth"
        )
        if not teacher.is_file() or not student.is_file():
            self.skipTest("Run-specific deployment checkpoints are unavailable")

        models, metadata = load_deployment_models(teacher, student)
        history = torch.zeros(1, 50, 21)
        slow = models.slow_encoder(history)
        fast = models.fast_encoder(history)
        action = models.actor(torch.cat([history[:, -1, :], slow, fast], dim=-1))

        self.assertEqual(tuple(slow.shape), (1, 2))
        self.assertEqual(tuple(fast.shape), (1, 3))
        self.assertEqual(tuple(action.shape), (1, 4))
        self.assertEqual(metadata["student_epoch"], 473)
        self.assertEqual(metadata["teacher_iteration"], 3000)
        self.assertEqual(metadata["teacher_context_mode"], "split_hard")


class _HistoryProbe(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.last_history = None

    def forward(self, history):
        self.last_history = history.detach().clone()
        value = history[:, -1, :1]
        return value.repeat(1, self.output_dim)


class _ConstantActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("action", torch.tensor([[1.0, -3.0, 3.0, 2.0]]))

    def forward(self, actor_input):
        return self.action.to(dtype=actor_input.dtype).repeat(actor_input.shape[0], 1)


class JetsonReferenceRuntimeTests(unittest.TestCase):
    def test_runtime_zero_pads_history_and_clamps_ctbr(self):
        slow = _HistoryProbe(2)
        fast = _HistoryProbe(3)
        runtime = FastSlowRuntime(slow, fast, _ConstantActor())

        result = runtime.step(torch.ones(21))

        self.assertEqual(tuple(slow.last_history.shape), (1, 50, 21))
        torch.testing.assert_close(
            slow.last_history[:, :-1], torch.zeros(1, 49, 21)
        )
        torch.testing.assert_close(
            slow.last_history[:, -1], torch.ones(1, 21)
        )
        torch.testing.assert_close(
            result["action_raw"], torch.tensor([[1.0, -3.0, 3.0, 2.0]])
        )
        torch.testing.assert_close(
            result["action_clamped"], torch.tensor([[0.0, -2.5, 2.5, 1.5]])
        )

    def test_runtime_matches_fast_slow_update_schedule(self):
        slow = _HistoryProbe(2)
        fast = _HistoryProbe(3)
        runtime = FastSlowRuntime(slow, fast, _ConstantActor())

        slow_flags = []
        fast_flags = []
        for step in range(242):
            result = runtime.step(torch.full((21,), float(step)))
            slow_flags.append(result["slow_updated"])
            fast_flags.append(result["fast_updated"])

        self.assertTrue(all(slow_flags[:181]))
        self.assertFalse(slow_flags[181])
        self.assertFalse(slow_flags[239])
        self.assertTrue(slow_flags[240])
        self.assertFalse(slow_flags[241])
        self.assertTrue(all(fast_flags))
        self.assertEqual(runtime.slow_call_count, 182)
        self.assertEqual(runtime.fast_call_count, 242)

    def test_causal_filter_matches_training_runtime(self):
        alpha = causal_ema_alpha(policy_dt=1.0 / 60.0, tau_sec=0.25)
        self.assertAlmostEqual(alpha, 0.06449301496838222, places=14)

    def test_load_torchscript_runtime_uses_bundle_model_names(self):
        slow = torch.jit.trace(_HistoryProbe(2), torch.zeros(1, 50, 21))
        fast = torch.jit.trace(_HistoryProbe(3), torch.zeros(1, 50, 21))
        actor = torch.jit.trace(_ConstantActor(), torch.zeros(1, 26))
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            torch.jit.save(slow, bundle / "slow_encoder.ts")
            torch.jit.save(fast, bundle / "fast_encoder.ts")
            torch.jit.save(actor, bundle / "actor.ts")

            runtime = load_torchscript_runtime(bundle)
            result = runtime.step(torch.zeros(21))

        self.assertEqual(tuple(result["context"].shape), (1, 5))
        self.assertEqual(tuple(result["action_clamped"].shape), (1, 4))

    def test_load_torchscript_runtime_supports_nested_model_directory(self):
        slow = torch.jit.trace(_HistoryProbe(2), torch.zeros(1, 50, 21))
        fast = torch.jit.trace(_HistoryProbe(3), torch.zeros(1, 50, 21))
        actor = torch.jit.trace(_ConstantActor(), torch.zeros(1, 26))
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            models = bundle / "models"
            models.mkdir()
            torch.jit.save(slow, models / "slow_encoder.ts")
            torch.jit.save(fast, models / "fast_encoder.ts")
            torch.jit.save(actor, models / "actor.ts")
            runtime = load_torchscript_runtime(bundle)
            result = runtime.step(torch.zeros(21))
        self.assertEqual(tuple(result["context"].shape), (1, 5))


class JetsonExporterTests(unittest.TestCase):
    def test_manifest_records_complete_runtime_contract(self):
        metadata = {
            "teacher_iteration": 3000,
            "student_epoch": 473,
            "student_best_val": 4.246531203534687e-4,
            "seed": 42,
            "teacher_context_mode": "split_hard",
            "student_context_mode": "split",
            "input_dim": 21,
            "history_len": 50,
            "z_dim": 5,
            "z_slow_dim": 2,
            "z_fast_dim": 3,
            "normalizer_eps": 1.0e-2,
        }
        manifest = build_manifest(
            metadata,
            source_hashes={"teacher": "abc", "student": "def"},
        )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["models"]["actor"]["input_shape"], ["B", 26])
        self.assertEqual(
            manifest["models"]["slow_encoder"]["input_shape"], ["B", 50, 21]
        )
        self.assertEqual(manifest["runtime"]["slow_warmup_steps"], 180)
        self.assertEqual(manifest["runtime"]["slow_period_steps"], 60)
        self.assertEqual(manifest["runtime"]["fast_period_steps"], 1)
        self.assertEqual(manifest["action"]["low"], [-1.0, -2.5, -2.5, -1.5])
        self.assertTrue(manifest["safety"]["startup_guard_required_before_flight"])
        self.assertEqual(manifest["training_ranges"]["payload_mass_kg"], [0.31265, 0.93265])
        self.assertEqual(manifest["training_ranges"]["rope_length_m"], [0.25, 0.8])
        self.assertEqual(manifest["models"]["actor"]["onnx"], "models/actor.onnx")
        self.assertEqual(manifest["student_source"], "models/source_student_best.pth")
        self.assertNotIn("last_checkpoint.pth", "\n".join(manifest["artifacts"]))
        self.assertEqual(set(manifest["artifacts"]), set(REQUIRED_ARTIFACTS))

    def test_lineage_guard_rejects_different_source_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "manifest.json").write_text(
                json.dumps(
                    {"source_hashes": {"teacher": "old", "student": "old"}}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "different checkpoint lineage"):
                validate_existing_bundle_lineage(
                    output, {"teacher": "new", "student": "new"}
                )

    def test_sha256_file_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "payload.bin"
            path.write_bytes(b"jetson-bundle")
            self.assertEqual(
                sha256_file(path),
                "fd2e48431be6b10054d6fa6b494c029aa39d893c890b542560cdd08d935600c0",
            )

    def test_copy_required_can_replace_read_only_previous_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.py"
            destination = root / "destination.py"
            source.write_text("new", encoding="utf-8")
            destination.write_text("old", encoding="utf-8")
            destination.chmod(0o444)
            _copy_required(source, destination)
            self.assertEqual(destination.read_text(encoding="utf-8"), "new")

    def test_parity_history_uses_float32_physical_dataset_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary)
            values = torch.arange(8 * 50 * 21, dtype=torch.float16).reshape(
                8, 50, 21
            )
            torch.save({"inputs": values}, data_dir / "shard_0000.pt")

            history, audit = load_parity_history(data_dir, sample_count=4)

        self.assertEqual(tuple(history.shape), (4, 50, 21))
        self.assertEqual(history.dtype, torch.float32)
        self.assertEqual(audit["shard_name"], "shard_0000.pt")
        self.assertEqual(audit["sample_count"], 4)
        self.assertEqual(len(audit["shard_sha256"]), 64)

    def test_bundle_checksum_verifier_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            artifact = bundle / "actor.ts"
            artifact.write_bytes(b"original")
            (bundle / "sha256sums.txt").write_text(
                f"{sha256_file(artifact)}  actor.ts\n", encoding="utf-8"
            )

            clean = verify_checksums(bundle)
            artifact.write_bytes(b"tampered")
            tampered = verify_checksums(bundle)

        self.assertTrue(clean["passed"])
        self.assertFalse(tampered["passed"])
        self.assertEqual(tampered["mismatched"], ["actor.ts"])

    def test_torchscript_verifier_reads_nested_models_and_vectors(self):
        slow = torch.jit.trace(_HistoryProbe(2), torch.zeros(1, 50, 21))
        fast = torch.jit.trace(_HistoryProbe(3), torch.zeros(1, 50, 21))
        actor = torch.jit.trace(_ConstantActor(), torch.zeros(1, 26))
        history = torch.zeros(1, 50, 21)
        actor_input = torch.zeros(1, 26)
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            models = bundle / "models"
            verification = bundle / "verification"
            models.mkdir()
            verification.mkdir()
            torch.jit.save(slow, models / "slow_encoder.ts")
            torch.jit.save(fast, models / "fast_encoder.ts")
            torch.jit.save(actor, models / "actor.ts")
            import numpy as np

            np.savez(
                verification / "parity_vectors.npz",
                history=history.numpy(),
                actor_input=actor_input.numpy(),
                slow_expected=slow(history).detach().numpy(),
                fast_expected=fast(history).detach().numpy(),
                actor_expected=actor(actor_input).detach().numpy(),
            )
            report = verify_torchscript(bundle)
        self.assertTrue(report["passed"])

    def test_required_artifacts_use_complete_nested_bundle_layout(self):
        required = set(REQUIRED_ARTIFACTS)
        self.assertIn("models/actor.onnx", required)
        self.assertIn("runtime/rl_fastslow_inference_onnx_v1.py", required)
        self.assertIn("verification/parity_vectors.npz", required)
        self.assertIn("config/manifest.json", required)
        self.assertIn("models/source_student_best.pth", required)
        self.assertNotIn("models/last_checkpoint.pth", required)

    def test_manifest_verifier_rejects_extra_student_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "config").mkdir()
            (bundle / "models").mkdir()
            (bundle / "models/source_student_best.pth").write_bytes(b"best")
            (bundle / "models/last_checkpoint.pth").write_bytes(b"last")
            (bundle / "config/manifest.json").write_text(
                json.dumps(
                    {
                        "student_source": "models/source_student_best.pth",
                        "artifacts": ["models/source_student_best.pth"],
                        "observation_21": {"topic": "/uav_payload/observation21"},
                    }
                ),
                encoding="utf-8",
            )
            report = verify_manifest(bundle)
        self.assertFalse(report["passed"])
        self.assertEqual(report["unexpected_student_checkpoints"], ["models/last_checkpoint.pth"])


if __name__ == "__main__":
    unittest.main()
