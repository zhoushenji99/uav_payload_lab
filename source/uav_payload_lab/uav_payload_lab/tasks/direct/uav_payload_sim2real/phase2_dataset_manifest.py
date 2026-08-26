#!/usr/bin/env python3
"""Build a deterministic, checksum-locked Phase-II episode split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_split_manifest(
    records: Iterable[dict[str, object]],
    *,
    train_seeds: Iterable[int],
    val_seeds: Iterable[int],
) -> dict[str, object]:
    """Split complete episode records using disjoint, explicitly assigned seeds."""
    train_seed_set = {int(seed) for seed in train_seeds}
    val_seed_set = {int(seed) for seed in val_seeds}
    overlap = train_seed_set & val_seed_set
    if overlap:
        raise ValueError(f"train and validation seeds overlap: {sorted(overlap)}")
    if not train_seed_set or not val_seed_set:
        raise ValueError("train_seeds and val_seeds must both be non-empty")

    grouped: dict[tuple[int, int], dict[str, object]] = {}
    for raw in records:
        seed = int(raw["seed"])
        episode_id = int(raw["episode_id"])
        if seed not in train_seed_set | val_seed_set:
            raise ValueError(f"record seed {seed} was not assigned to a split")
        key = (seed, episode_id)
        entry = grouped.setdefault(
            key,
            {"seed": seed, "episode_id": episode_id, "shard_ids": []},
        )
        shard_ids = raw.get("shard_ids")
        if shard_ids is None and raw.get("shard_id") is not None:
            shard_ids = [raw["shard_id"]]
        for shard_id in shard_ids or []:
            shard_id = str(shard_id)
            if shard_id not in entry["shard_ids"]:
                entry["shard_ids"].append(shard_id)

    train = []
    validation = []
    for key in sorted(grouped):
        entry = grouped[key]
        entry["shard_ids"] = sorted(entry["shard_ids"])
        (train if key[0] in train_seed_set else validation).append(entry)

    train_keys = {(item["seed"], item["episode_id"]) for item in train}
    val_keys = {(item["seed"], item["episode_id"]) for item in validation}
    if train_keys & val_keys:
        raise RuntimeError("episode leakage detected while building manifest")
    return {
        "train_seeds": sorted(train_seed_set),
        "validation_seeds": sorted(val_seed_set),
        "train": train,
        "validation": validation,
    }


def _source_name_map(meta: dict[str, object]) -> dict[int, str]:
    raw = meta.get("history_source_encoding", {})
    mapping = {int(key): str(value) for key, value in raw.items()}
    if not mapping:
        raise RuntimeError("dataset meta.pt is missing history_source_encoding")
    return mapping


def create_manifest_from_roots(
    dataset_roots: Iterable[str | Path],
    *,
    train_seeds: Iterable[int],
    val_seeds: Iterable[int],
    required_sources: Iterable[str],
) -> dict[str, object]:
    """Scan shard metadata and create a complete episode-level split manifest."""
    roots = [Path(root).expanduser().resolve() for root in dataset_roots]
    if not roots:
        raise ValueError("at least one dataset root is required")
    required = [str(name) for name in required_sources]
    if len(required) != len(set(required)):
        raise ValueError("required_sources contains duplicates")

    records: list[dict[str, object]] = []
    shards: list[dict[str, object]] = []
    source_counts = {
        "all": Counter(),
        "train": Counter(),
        "validation": Counter(),
    }
    train_seed_set = {int(seed) for seed in train_seeds}
    val_seed_set = {int(seed) for seed in val_seeds}
    context_modes: set[str] = set()
    source_name_by_id: dict[int, str] = {}

    for root in roots:
        meta_path = root / "meta.pt"
        if not meta_path.is_file():
            raise RuntimeError(f"missing dataset metadata: {meta_path}")
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)
        context_modes.add(str(meta.get("teacher_context_mode", "unknown")))
        local_sources = _source_name_map(meta)
        for source_id, name in local_sources.items():
            previous = source_name_by_id.get(source_id)
            if previous is not None and previous != name:
                raise RuntimeError(
                    f"history source id {source_id} maps to both {previous!r} and {name!r}"
                )
            source_name_by_id[source_id] = name

        shard_paths = sorted(root.glob("shard_*.pt"))
        if not shard_paths:
            raise RuntimeError(f"no shard_*.pt files found in {root}")
        for shard_path in shard_paths:
            data = torch.load(shard_path, map_location="cpu", weights_only=False)
            required_keys = (
                "inputs",
                "labels",
                "labels_ml",
                "seed",
                "env_id",
                "episode_id",
                "episode_step",
                "history_source",
            )
            missing = [key for key in required_keys if key not in data]
            if missing:
                raise RuntimeError(f"{shard_path} is missing metadata keys: {missing}")
            sample_count = int(data["inputs"].shape[0])
            for key in required_keys[1:]:
                if int(data[key].shape[0]) != sample_count:
                    raise RuntimeError(
                        f"{shard_path}: {key} has {data[key].shape[0]} rows, expected {sample_count}"
                    )

            seed_values = data["seed"].to(dtype=torch.int64).reshape(-1)
            episode_values = data["episode_id"].to(dtype=torch.int64).reshape(-1)
            source_values = data["history_source"].to(dtype=torch.int64).reshape(-1)
            unique_seeds = sorted({int(value) for value in seed_values.tolist()})
            if len(unique_seeds) != 1:
                raise RuntimeError(f"{shard_path} mixes seeds: {unique_seeds}")
            seed = unique_seeds[0]
            if seed not in train_seed_set | val_seed_set:
                raise RuntimeError(f"{shard_path} uses unassigned seed {seed}")
            split = "train" if seed in train_seed_set else "validation"

            shard_id = f"shard_{len(shards):06d}"
            episode_ids = sorted({int(value) for value in episode_values.tolist()})
            for episode_id in episode_ids:
                records.append(
                    {"seed": seed, "episode_id": episode_id, "shard_id": shard_id}
                )

            local_counts = Counter()
            for source_id, count in zip(*torch.unique(source_values, return_counts=True)):
                source_id_int = int(source_id.item())
                if source_id_int not in local_sources:
                    raise RuntimeError(
                        f"{shard_path} uses unknown history source id {source_id_int}"
                    )
                local_counts[local_sources[source_id_int]] += int(count.item())
            source_counts["all"].update(local_counts)
            source_counts[split].update(local_counts)
            shards.append(
                {
                    "id": shard_id,
                    "path": str(shard_path.resolve()),
                    "sha256": _sha256(shard_path),
                    "sample_count": sample_count,
                    "seed": seed,
                    "episode_ids": episode_ids,
                    "source_counts": dict(sorted(local_counts.items())),
                }
            )

    if len(context_modes) != 1:
        raise RuntimeError(f"dataset roots mix Teacher context modes: {sorted(context_modes)}")
    split_manifest = build_split_manifest(
        records,
        train_seeds=train_seed_set,
        val_seeds=val_seed_set,
    )
    for split in ("train", "validation"):
        missing = [name for name in required if source_counts[split][name] <= 0]
        if missing:
            raise RuntimeError(f"{split} split is missing required sources: {missing}")

    return {
        "schema_version": 1,
        "dataset_roots": [str(root) for root in roots],
        "teacher_context_mode": next(iter(context_modes)),
        "required_sources": required,
        "source_encoding": {
            str(source_id): name
            for source_id, name in sorted(source_name_by_id.items())
        },
        "shards": shards,
        **split_manifest,
        "source_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in source_counts.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--dataset-roots", nargs="+", required=True)
    build.add_argument("--train-seeds", nargs="+", type=int, required=True)
    build.add_argument("--validation-seeds", nargs="+", type=int, required=True)
    build.add_argument("--required-sources", nargs="+", required=True)
    build.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command != "build":
        raise RuntimeError(f"unsupported command: {args.command}")
    manifest = create_manifest_from_roots(
        args.dataset_roots,
        train_seeds=args.train_seeds,
        val_seeds=args.validation_seeds,
        required_sources=args.required_sources,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[Manifest] train_episodes={len(manifest['train'])}")
    print(f"[Manifest] validation_episodes={len(manifest['validation'])}")
    print(f"[Manifest] saved {output}")


if __name__ == "__main__":
    main()
