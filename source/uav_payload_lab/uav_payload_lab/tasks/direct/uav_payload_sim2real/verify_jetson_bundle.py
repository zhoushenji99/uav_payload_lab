"""Target-side checksum and golden-vector verification for a Jetson bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(bundle_dir: str | Path) -> dict[str, Any]:
    root = Path(bundle_dir).resolve()
    checksum_path = root / "sha256sums.txt"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing checksum file: {checksum_path}")
    missing: list[str] = []
    mismatched: list[str] = []
    checked: list[str] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.strip()
        path = (root / name).resolve()
        if root not in path.parents:
            raise ValueError(f"Checksum entry escapes bundle directory: {name}")
        if not path.is_file():
            missing.append(name)
            continue
        checked.append(name)
        if _sha256(path) != expected:
            mismatched.append(name)
    return {
        "passed": not missing and not mismatched,
        "checked_count": len(checked),
        "missing": missing,
        "mismatched": mismatched,
    }


def _load_vectors(bundle_dir: Path) -> dict[str, np.ndarray]:
    path = bundle_dir / "parity_vectors.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing parity vectors: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def verify_torchscript(
    bundle_dir: str | Path,
    *,
    device: str = "cpu",
    tolerance: float = 1.0e-6,
) -> dict[str, Any]:
    root = Path(bundle_dir).resolve()
    vectors = _load_vectors(root)
    device_value = torch.device(device)
    cases = {
        "slow_encoder": ("history", "slow_expected"),
        "fast_encoder": ("history", "fast_expected"),
        "actor": ("actor_input", "actor_expected"),
    }
    reports: dict[str, Any] = {}
    with torch.inference_mode():
        for name, (input_key, expected_key) in cases.items():
            module = torch.jit.load(
                str(root / f"{name}.ts"), map_location=device_value
            ).eval()
            value = torch.from_numpy(vectors[input_key]).to(device_value)
            expected = torch.from_numpy(vectors[expected_key]).to(device_value)
            output = module(value)
            max_abs = float(torch.max(torch.abs(output - expected)).item())
            finite = bool(torch.isfinite(output).all().item())
            reports[name] = {
                "max_abs": max_abs,
                "tolerance": tolerance,
                "finite": finite,
                "passed": finite and max_abs <= tolerance,
            }
    return {
        "backend": "torchscript",
        "device": str(device_value),
        "models": reports,
        "passed": all(item["passed"] for item in reports.values()),
    }


def verify_onnx(
    bundle_dir: str | Path,
    *,
    tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    root = Path(bundle_dir).resolve()
    vectors = _load_vectors(root)
    cases = {
        "slow_encoder": ("history", "slow_expected"),
        "fast_encoder": ("history", "fast_expected"),
        "actor": ("actor_input", "actor_expected"),
    }
    reports: dict[str, Any] = {}
    try:
        import onnxruntime as ort

        backend = "onnxruntime"
        for name, (input_key, expected_key) in cases.items():
            session = ort.InferenceSession(
                str(root / f"{name}.onnx"), providers=["CPUExecutionProvider"]
            )
            input_name = session.get_inputs()[0].name
            output = np.asarray(
                session.run(None, {input_name: vectors[input_key]})[0]
            )
            expected = vectors[expected_key]
            max_abs = float(np.max(np.abs(output - expected)))
            finite = bool(np.isfinite(output).all())
            reports[name] = {
                "max_abs": max_abs,
                "tolerance": tolerance,
                "finite": finite,
                "passed": finite and max_abs <= tolerance,
            }
    except ModuleNotFoundError:
        from onnx.reference import ReferenceEvaluator

        backend = "onnx_reference"
        for name, (input_key, expected_key) in cases.items():
            evaluator = ReferenceEvaluator(str(root / f"{name}.onnx"))
            input_name = evaluator.input_names[0]
            output = np.asarray(
                evaluator.run(None, {input_name: vectors[input_key]})[0]
            )
            expected = vectors[expected_key]
            max_abs = float(np.max(np.abs(output - expected)))
            finite = bool(np.isfinite(output).all())
            reports[name] = {
                "max_abs": max_abs,
                "tolerance": tolerance,
                "finite": finite,
                "passed": finite and max_abs <= tolerance,
            }
    return {
        "backend": backend,
        "models": reports,
        "passed": all(item["passed"] for item in reports.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default=".")
    parser.add_argument(
        "--backend", choices=["torchscript", "onnx", "all"], default="all"
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    root = Path(args.bundle).resolve()
    report: dict[str, Any] = {"checksums": verify_checksums(root)}
    if args.backend in ("torchscript", "all"):
        report["torchscript"] = verify_torchscript(root, device=args.device)
    if args.backend in ("onnx", "all"):
        report["onnx"] = verify_onnx(root)
    report["passed"] = all(
        value.get("passed", False)
        for key, value in report.items()
        if key != "passed"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
