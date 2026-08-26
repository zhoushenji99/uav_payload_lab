#!/usr/bin/env python3
"""Audit every Phase-II history-to-context dataset shard.

The collector performs the same checks before writing each shard.  This
standalone command rechecks an existing dataset before Student training and
writes a durable JSON report beside ``meta.pt``.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_split_manifest(path: str | os.PathLike[str]) -> dict[str, object]:
    """Verify manifest checksums and enforce disjoint seed/episode splits."""
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_seeds = {int(seed) for seed in manifest.get("train_seeds", [])}
    val_seeds = {int(seed) for seed in manifest.get("validation_seeds", [])}
    seed_overlap = sorted(train_seeds & val_seeds)
    train_keys = {
        (int(item["seed"]), int(item["episode_id"]))
        for item in manifest.get("train", [])
    }
    val_keys = {
        (int(item["seed"]), int(item["episode_id"]))
        for item in manifest.get("validation", [])
    }
    episode_overlap = sorted(train_keys & val_keys)

    shard_reports = []
    shard_paths = []
    checksum_ok = True
    for item in manifest.get("shards", []):
        shard_path = Path(item["path"]).expanduser().resolve()
        exists = shard_path.is_file()
        actual = _sha256(shard_path) if exists else None
        matches = bool(exists and actual == str(item["sha256"]))
        checksum_ok = checksum_ok and matches
        shard_reports.append(
            {
                "id": str(item["id"]),
                "path": str(shard_path),
                "exists": exists,
                "expected_sha256": str(item["sha256"]),
                "actual_sha256": actual,
                "checksum_ok": matches,
            }
        )
        if exists:
            shard_paths.append(str(shard_path))

    source_counts = manifest.get("source_counts", {})
    required_sources = [str(name) for name in manifest.get("required_sources", [])]
    missing_sources = {
        split: [
            name
            for name in required_sources
            if int(source_counts.get(split, {}).get(name, 0)) <= 0
        ]
        for split in ("train", "validation")
    }
    passed = bool(
        shard_reports
        and checksum_ok
        and not seed_overlap
        and not episode_overlap
        and not missing_sources["train"]
        and not missing_sources["validation"]
    )
    return {
        "path": str(manifest_path),
        "manifest": manifest,
        "shard_paths": shard_paths,
        "shards": shard_reports,
        "checksum_ok": checksum_ok,
        "seed_overlap": seed_overlap,
        "episode_overlap": [list(key) for key in episode_overlap],
        "source_counts": source_counts,
        "missing_required_sources": missing_sources,
        "passed": passed,
    }


def audit_shard_tensors(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    labels_ml: torch.Tensor,
    *,
    history_len: int | None = None,
    input_dim: int = 21,
    z_dim: int = 5,
    slow_dim: int = 2,
) -> dict[str, object]:
    """Validate one in-memory shard and return JSON-safe diagnostics."""

    if inputs.ndim != 3:
        raise RuntimeError(f"inputs must have shape (N,H,D), got {tuple(inputs.shape)}")
    if labels.ndim != 2 or labels_ml.ndim != 2:
        raise RuntimeError(
            f"labels/labels_ml must be matrices, got {tuple(labels.shape)} and "
            f"{tuple(labels_ml.shape)}"
        )
    sample_count = int(inputs.shape[0])
    if labels.shape[0] != sample_count or labels_ml.shape[0] != sample_count:
        raise RuntimeError(
            "Shard sample counts differ: "
            f"inputs={sample_count}, labels={labels.shape[0]}, labels_ml={labels_ml.shape[0]}"
        )
    if history_len is not None and int(inputs.shape[1]) != int(history_len):
        raise RuntimeError(
            f"Expected history_len={history_len}, got inputs.shape[1]={inputs.shape[1]}"
        )
    if int(inputs.shape[2]) != int(input_dim):
        raise RuntimeError(f"Expected input_dim={input_dim}, got {inputs.shape[2]}")
    if int(labels.shape[1]) != int(z_dim):
        raise RuntimeError(f"Expected z_dim={z_dim}, got {labels.shape[1]}")
    if int(labels_ml.shape[1]) != int(slow_dim):
        raise RuntimeError(f"Expected slow_dim={slow_dim}, got {labels_ml.shape[1]}")

    inputs_nonfinite = int(torch.count_nonzero(~torch.isfinite(inputs)).item())
    labels_nonfinite = int(torch.count_nonzero(~torch.isfinite(labels)).item())
    labels_ml_nonfinite = int(torch.count_nonzero(~torch.isfinite(labels_ml)).item())
    identity_error = torch.abs(labels[:, :slow_dim] - labels_ml)
    slow_identity_max_abs = (
        float(torch.max(identity_error).item()) if identity_error.numel() else 0.0
    )

    return {
        "sample_count": sample_count,
        "inputs_shape": [int(x) for x in inputs.shape],
        "labels_shape": [int(x) for x in labels.shape],
        "labels_ml_shape": [int(x) for x in labels_ml.shape],
        "inputs_dtype": str(inputs.dtype),
        "labels_dtype": str(labels.dtype),
        "labels_ml_dtype": str(labels_ml.dtype),
        "inputs_nonfinite": inputs_nonfinite,
        "labels_nonfinite": labels_nonfinite,
        "labels_ml_nonfinite": labels_ml_nonfinite,
        "slow_identity_max_abs": slow_identity_max_abs,
        "all_finite": (inputs_nonfinite + labels_nonfinite + labels_ml_nonfinite) == 0,
    }


def _empty_stats(dim: int) -> dict[str, torch.Tensor | int]:
    return {
        "count": 0,
        "sum": torch.zeros(dim, dtype=torch.float64),
        "sumsq": torch.zeros(dim, dtype=torch.float64),
        "min": torch.full((dim,), float("inf"), dtype=torch.float64),
        "max": torch.full((dim,), -float("inf"), dtype=torch.float64),
    }


def _update_stats(stats: dict[str, object], values: torch.Tensor) -> None:
    flat = values.detach().to(dtype=torch.float64, device="cpu").reshape(-1, values.shape[-1])
    finite_rows = torch.isfinite(flat).all(dim=1)
    flat = flat[finite_rows]
    if flat.numel() == 0:
        return
    stats["count"] = int(stats["count"]) + int(flat.shape[0])
    stats["sum"] += flat.sum(dim=0)
    stats["sumsq"] += (flat * flat).sum(dim=0)
    stats["min"] = torch.minimum(stats["min"], flat.min(dim=0).values)
    stats["max"] = torch.maximum(stats["max"], flat.max(dim=0).values)


def _finalize_stats(stats: dict[str, object]) -> dict[str, object]:
    count = int(stats["count"])
    if count == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = stats["sum"] / count
    variance = stats["sumsq"] / count - mean * mean
    std = torch.sqrt(torch.clamp(variance, min=0.0))
    return {
        "count": count,
        "mean": [float(x) for x in mean.tolist()],
        "std": [float(x) for x in std.tolist()],
        "min": [float(x) for x in stats["min"].tolist()],
        "max": [float(x) for x in stats["max"].tolist()],
    }


def audit_dataset(
    data_dir: str | os.PathLike[str],
    *,
    require_hard_identity: bool = False,
    split_manifest: str | os.PathLike[str] | None = None,
) -> dict[str, object]:
    """Load and validate all shards in ``data_dir``."""

    data_path = Path(data_dir).expanduser().resolve()
    manifest_audit = audit_split_manifest(split_manifest) if split_manifest else None
    if manifest_audit is not None:
        shard_paths = list(manifest_audit["shard_paths"])
    else:
        shard_paths = sorted(glob.glob(str(data_path / "shard_*.pt")))
    if not shard_paths:
        raise RuntimeError(f"No shard_*.pt files found in {data_path}")

    meta_path = data_path / "meta.pt"
    meta = (
        torch.load(meta_path, map_location="cpu", weights_only=False)
        if meta_path.exists()
        else {}
    )
    if manifest_audit is not None and not meta:
        first_root = Path(manifest_audit["manifest"]["dataset_roots"][0])
        root_meta = first_root / "meta.pt"
        meta = torch.load(root_meta, map_location="cpu", weights_only=False)
    history_len = int(meta.get("history_len", 50))
    input_dim = int(meta.get("input_dim", 21))
    z_dim = int(meta.get("z_dim", 5))
    slow_dim = int(meta.get("z_exp_dim", 2))
    context_mode = str(meta.get("teacher_context_mode", "unknown"))
    require_identity = bool(require_hard_identity or context_mode == "split_hard")

    total_samples = 0
    total_nonfinite = {"inputs": 0, "labels": 0, "labels_ml": 0}
    identity_max = 0.0
    label_stats = _empty_stats(z_dim)
    ml_stats = _empty_stats(slow_dim)
    shard_reports = []

    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        missing = [key for key in ("inputs", "labels", "labels_ml") if key not in shard]
        if missing:
            raise RuntimeError(f"{shard_path} is missing keys: {missing}")
        report = audit_shard_tensors(
            shard["inputs"],
            shard["labels"],
            shard["labels_ml"],
            history_len=history_len,
            input_dim=input_dim,
            z_dim=z_dim,
            slow_dim=slow_dim,
        )
        report["file"] = os.path.basename(shard_path)
        shard_reports.append(report)
        total_samples += int(report["sample_count"])
        total_nonfinite["inputs"] += int(report["inputs_nonfinite"])
        total_nonfinite["labels"] += int(report["labels_nonfinite"])
        total_nonfinite["labels_ml"] += int(report["labels_ml_nonfinite"])
        identity_max = max(identity_max, float(report["slow_identity_max_abs"]))
        _update_stats(label_stats, shard["labels"])
        _update_stats(ml_stats, shard["labels_ml"])

    labels_summary = _finalize_stats(label_stats)
    ml_summary = _finalize_stats(ml_stats)
    normalized_slow_coverage = None
    if ml_summary["min"] is not None:
        normalized_slow_coverage = [
            float(hi - lo)
            for lo, hi in zip(ml_summary["min"], ml_summary["max"])
        ]

    finite_ok = sum(total_nonfinite.values()) == 0
    identity_ok = (not require_identity) or identity_max == 0.0
    meta_samples = None if manifest_audit is not None else meta.get("total_samples")
    meta_count_ok = meta_samples is None or int(meta_samples) == total_samples
    report = {
        "data_dir": str(data_path),
        "teacher_context_mode": context_mode,
        "require_hard_identity": require_identity,
        "num_shards": len(shard_paths),
        "total_samples": total_samples,
        "meta_total_samples": int(meta_samples) if meta_samples is not None else None,
        "meta_count_ok": bool(meta_count_ok),
        "history_len": history_len,
        "input_dim": input_dim,
        "z_dim": z_dim,
        "z_exp_dim": slow_dim,
        "nonfinite_counts": total_nonfinite,
        "all_finite": finite_ok,
        "slow_label_identity_max_abs": identity_max,
        "hard_identity_ok": identity_ok,
        "labels_stats": labels_summary,
        "labels_ml_stats": ml_summary,
        "normalized_slow_coverage_span": normalized_slow_coverage,
        "shards": shard_reports,
        "split_manifest_audit": (
            {key: value for key, value in manifest_audit.items() if key != "manifest"}
            if manifest_audit is not None
            else None
        ),
    }
    report["passed"] = bool(
        finite_ok
        and identity_ok
        and meta_count_ok
        and (manifest_audit is None or manifest_audit["passed"])
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit all Phase-II z-dataset shards.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--split_manifest",
        default="",
        help="Episode/seed split manifest whose shard hashes must all verify.",
    )
    parser.add_argument(
        "--require_hard_identity",
        action="store_true",
        help="Require labels[:,:2] to equal labels_ml exactly, regardless of meta.pt.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output JSON. Default: <data_dir>/dataset_audit_recheck.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_dataset(
        args.data_dir,
        require_hard_identity=bool(args.require_hard_identity),
        split_manifest=args.split_manifest or None,
    )
    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else Path(args.data_dir).expanduser().resolve() / "dataset_audit_recheck.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
    print(f"[Audit] passed={report['passed']} -> {out_path}")
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
