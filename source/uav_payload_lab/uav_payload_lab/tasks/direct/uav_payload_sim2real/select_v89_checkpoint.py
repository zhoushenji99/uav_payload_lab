#!/usr/bin/env python3
"""Select the lowest-score checkpoint only from fully passing V8.9 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def select_passing_checkpoint(evaluation: dict[str, object]):
    passing = [item for item in evaluation.get("checkpoints", []) if item.get("passed")]
    if not passing:
        return None
    return min(passing, key=lambda item: (float(item["score"]), item["checkpoint"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-kind", choices=["teacher", "student"], required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluation_root = Path(args.evaluation_root).expanduser().resolve()
    evaluation = json.loads((evaluation_root / "summary.json").read_text())
    if evaluation.get("policy_kind") != args.policy_kind:
        raise RuntimeError("evaluation policy kind does not match selection request")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_file = output / "selected_checkpoint.txt"
    if selected_file.exists():
        selected_file.unlink()

    selected = select_passing_checkpoint(evaluation)
    report = {
        "policy_kind": args.policy_kind,
        "evaluation_root": str(evaluation_root),
        "contract": str(Path(args.contract).expanduser().resolve()),
        "selected": selected,
        "passed_candidate_count": sum(
            bool(item.get("passed")) for item in evaluation.get("checkpoints", [])
        ),
    }
    if selected is None:
        (output / "selection_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print("[V8.9 Select] no checkpoint passed every hard gate")
        raise SystemExit(2)

    checkpoint = Path(selected["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise RuntimeError(f"selected checkpoint is missing: {checkpoint}")
    actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_sha != selected["sha256"]:
        raise RuntimeError("selected checkpoint SHA256 changed after evaluation")
    selected_file.write_text(str(checkpoint) + "\n", encoding="utf-8")
    report["selected_sha256"] = actual_sha
    (output / "selection_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[V8.9 Select] selected {checkpoint}")


if __name__ == "__main__":
    main()
