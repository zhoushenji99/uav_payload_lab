import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/train_student_z.py"
)


def _load_trainer_module():
    spec = importlib.util.spec_from_file_location("sim2real_train_student_z", TRAINER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


trainer = _load_trainer_module()


def test_seed_argument_is_required(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_student_z.py", "--data_dir", "/tmp/data"])
    with pytest.raises(SystemExit):
        trainer.parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_student_z.py",
            "--data_dir",
            "/tmp/data",
            "--split_manifest",
            "/tmp/manifest.json",
            "--source_weights",
            "teacher=0.5,position=0.5",
            "--seed",
            "42",
        ],
    )
    args = trainer.parse_args()
    assert args.seed == 42


def test_configure_reproducibility_repeats_all_rngs():
    trainer.configure_reproducibility(73)
    first = (
        random.random(),
        float(np.random.random()),
        torch.rand(8),
    )

    trainer.configure_reproducibility(73)
    second = (
        random.random(),
        float(np.random.random()),
        torch.rand(8),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])
    assert torch.are_deterministic_algorithms_enabled()


def test_rng_state_round_trip_including_dataloader_generator():
    trainer.configure_reproducibility(91)
    train_generator = torch.Generator().manual_seed(91)
    checkpoint_state = {
        "seed": 91,
        "rng_state": trainer.capture_rng_state(),
        "train_generator_state": train_generator.get_state(),
    }

    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(8)
    expected_order = torch.randperm(32, generator=train_generator)

    trainer.restore_training_rng(checkpoint_state, requested_seed=91, train_generator=train_generator)

    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(8), expected_torch)
    assert torch.equal(torch.randperm(32, generator=train_generator), expected_order)


@pytest.mark.parametrize(
    "checkpoint, message",
    [
        ({}, "legacy or incomplete"),
        (
            {"seed": 7, "rng_state": {}, "train_generator_state": torch.get_rng_state()},
            "seed mismatch",
        ),
    ],
)
def test_resume_rejects_uncontrolled_or_mismatched_seed(checkpoint, message):
    generator = torch.Generator().manual_seed(42)
    with pytest.raises(RuntimeError, match=message):
        trainer.restore_training_rng(checkpoint, requested_seed=42, train_generator=generator)


def _write_tiny_dataset(data_dir: Path):
    data_dir.mkdir(parents=True)
    generator = torch.Generator().manual_seed(20260802)
    for shard_idx in range(10):
        inputs = torch.randn(2, 50, 21, generator=generator)
        labels = torch.randn(2, 5, generator=generator)
        dataset_seed = 42 if shard_idx < 8 else 101
        source = shard_idx % 2
        torch.save(
            {
                "inputs": inputs,
                "labels": labels,
                "labels_ml": labels[:, :2].clone(),
                "seed": torch.full((2,), dataset_seed, dtype=torch.int64),
                "env_id": torch.tensor([0, 0]),
                "episode_id": torch.full((2,), shard_idx, dtype=torch.int64),
                "episode_step": torch.tensor([50, 55]),
                "history_source": torch.full((2,), source, dtype=torch.uint8),
                "episode_keys": [[dataset_seed, shard_idx]],
            },
            data_dir / f"shard_{shard_idx:04d}.pt",
        )
    torch.save(
        {
            "teacher_context_mode": "split_hard",
            "history_source_encoding": {
                "0": "teacher_closed_loop",
                "1": "causal_position_closed_loop",
            },
            "z_stats": {"mean": [0.0] * 5, "std": [1.0] * 5},
        },
        data_dir / "meta.pt",
    )
    (data_dir / "dataset_audit.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8"
    )
    shards = []
    train = []
    validation = []
    for shard_idx in range(10):
        path = (data_dir / f"shard_{shard_idx:04d}.pt").resolve()
        dataset_seed = 42 if shard_idx < 8 else 101
        shard_id = f"shard_{shard_idx:06d}"
        shards.append(
            {
                "id": shard_id,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sample_count": 2,
                "seed": dataset_seed,
                "episode_ids": [shard_idx],
            }
        )
        (train if dataset_seed == 42 else validation).append(
            {"seed": dataset_seed, "episode_id": shard_idx, "shard_ids": [shard_id]}
        )
    manifest = {
        "schema_version": 1,
        "dataset_roots": [str(data_dir.resolve())],
        "teacher_context_mode": "split_hard",
        "required_sources": ["teacher_closed_loop", "causal_position_closed_loop"],
        "source_encoding": {
            "0": "teacher_closed_loop",
            "1": "causal_position_closed_loop",
        },
        "shards": shards,
        "train_seeds": [42],
        "validation_seeds": [101],
        "train": train,
        "validation": validation,
        "source_counts": {
            "all": {"teacher_closed_loop": 10, "causal_position_closed_loop": 10},
            "train": {"teacher_closed_loop": 8, "causal_position_closed_loop": 8},
            "validation": {"teacher_closed_loop": 2, "causal_position_closed_loop": 2},
        },
    }
    (data_dir / "dataset_split_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def _run_trainer(data_dir: Path, out_dir: Path, epochs: int, resume: bool = False):
    command = [
        sys.executable,
        str(TRAINER_PATH),
        "--data_dir",
        str(data_dir),
        "--out_dir",
        str(out_dir),
        "--split_manifest",
        str(data_dir / "dataset_split_manifest.json"),
        "--source_weights",
        "teacher=0.5,position=0.5",
        "--student_context_mode",
        "split",
        "--epochs",
        str(epochs),
        "--batch_size",
        "2",
        "--lr",
        "2e-4",
        "--num_workers",
        "0",
        "--aux_ml_coef",
        "0",
        "--seed",
        "123",
        "--save_name",
        "best.pth",
        "--checkpoint_interval",
        "1",
    ]
    if resume:
        command.extend(
            ["--resume", "--resume_path", str(out_dir / "last_checkpoint.pth")]
        )
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_two_stage_resume_matches_uninterrupted_training_exactly(tmp_path):
    data_dir = tmp_path / "data"
    full_dir = tmp_path / "full"
    resumed_dir = tmp_path / "resumed"
    _write_tiny_dataset(data_dir)

    _run_trainer(data_dir, full_dir, epochs=2)
    _run_trainer(data_dir, resumed_dir, epochs=1)
    _run_trainer(data_dir, resumed_dir, epochs=2, resume=True)

    full = _load_checkpoint(full_dir / "last_checkpoint.pth")
    resumed = _load_checkpoint(resumed_dir / "last_checkpoint.pth")

    assert full["seed"] == resumed["seed"] == 123
    assert full["train_hist"] == resumed["train_hist"]
    assert full["val_hist"] == resumed["val_hist"]
    assert full["epoch"] == resumed["epoch"] == 1
    assert full["state_dict"].keys() == resumed["state_dict"].keys()
    for name in full["state_dict"]:
        assert torch.equal(full["state_dict"][name], resumed["state_dict"][name]), name

    report = json.loads((full_dir / "training_report.json").read_text())
    assert report["source_weights"] == {
        "teacher_closed_loop": 0.5,
        "causal_position_closed_loop": 0.5,
    }
    assert report["source_sample_counts_drawn"]
    assert report["top5_checkpoints"]
    assert all(
        Path(item["path"]).is_absolute() for item in report["top5_checkpoints"]
    )
