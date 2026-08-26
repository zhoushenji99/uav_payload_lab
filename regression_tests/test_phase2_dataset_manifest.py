import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/phase2_dataset_manifest.py"
)
AUDIT_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/audit_z_dataset.py"
)
COLLECT_PATH = (
    REPO_ROOT
    / "source/uav_payload_lab/uav_payload_lab/tasks/direct/uav_payload_sim2real/collect_z_dataset.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("phase2_dataset_manifest", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_z_dataset_manifest", AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_manifest_never_shares_episode_or_seed():
    module = _load_module()
    records = [
        {"seed": 42, "episode_id": 1, "shard_id": "a"},
        {"seed": 43, "episode_id": 2, "shard_id": "b"},
        {"seed": 101, "episode_id": 3, "shard_id": "c"},
    ]
    manifest = module.build_split_manifest(
        records,
        train_seeds=[42, 43],
        val_seeds=[101],
    )
    train_keys = {(x["seed"], x["episode_id"]) for x in manifest["train"]}
    val_keys = {
        (x["seed"], x["episode_id"]) for x in manifest["validation"]
    }
    assert train_keys.isdisjoint(val_keys)
    assert {x["seed"] for x in manifest["train"]}.isdisjoint(
        {x["seed"] for x in manifest["validation"]}
    )


def test_collector_saves_required_per_sample_episode_metadata():
    source = COLLECT_PATH.read_text(encoding="utf-8")
    for key in (
        '"seed": sample_seed',
        '"env_id": sample_env_id',
        '"episode_id": sample_episode_id',
        '"episode_step": sample_episode_step',
        '"history_source": history_source',
        '"episode_keys": episode_keys',
    ):
        assert key in source


def test_cli_manifest_records_absolute_shards_checksums_and_sources(tmp_path):
    module = _load_module()
    roots = []
    for seed in (42, 101):
        root = tmp_path / f"seed_{seed}"
        root.mkdir()
        shard = root / "shard_0000.pt"
        torch.save(
            {
                "inputs": torch.zeros(4, 50, 21),
                "labels": torch.zeros(4, 5),
                "labels_ml": torch.zeros(4, 2),
                "seed": torch.full((4,), seed, dtype=torch.int64),
                "env_id": torch.tensor([0, 0, 1, 1]),
                "episode_id": torch.tensor([0, 0, 1, 1]),
                "episode_step": torch.tensor([50, 55, 50, 55]),
                "history_source": torch.tensor([0, 0, 1, 1], dtype=torch.uint8),
                "episode_keys": [[seed, 0], [seed, 1]],
            },
            shard,
        )
        torch.save(
            {
                "teacher_context_mode": "split_hard",
                "history_source_encoding": {
                    "0": "teacher_closed_loop",
                    "1": "causal_position_closed_loop",
                },
            },
            root / "meta.pt",
        )
        roots.append(root)

    manifest = module.create_manifest_from_roots(
        roots,
        train_seeds=[42],
        val_seeds=[101],
        required_sources=[
            "teacher_closed_loop",
            "causal_position_closed_loop",
        ],
    )

    assert Path(manifest["shards"][0]["path"]).is_absolute()
    expected = hashlib.sha256(
        Path(manifest["shards"][0]["path"]).read_bytes()
    ).hexdigest()
    assert manifest["shards"][0]["sha256"] == expected
    assert manifest["source_counts"]["all"]["teacher_closed_loop"] == 4
    assert manifest["source_counts"]["all"]["causal_position_closed_loop"] == 4
    json.dumps(manifest)

    manifest_path = tmp_path / "dataset_split_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    audit_module = _load_audit_module()
    audit = audit_module.audit_split_manifest(manifest_path)
    assert audit["passed"] is True
    dataset_audit = audit_module.audit_dataset(
        tmp_path,
        require_hard_identity=True,
        split_manifest=manifest_path,
    )
    assert dataset_audit["passed"] is True

    first_shard = Path(manifest["shards"][0]["path"])
    first_shard.write_bytes(first_shard.read_bytes() + b"tamper")
    tampered = audit_module.audit_split_manifest(manifest_path)
    assert tampered["passed"] is False
    assert tampered["checksum_ok"] is False
