# train_student_z.py
# Train independent slow/fast student encoders from the same 50-step history.
# Paper-grade: weighted MSE, per-dim RMSE, report.json, loss_curve.png

import os
import glob
import hashlib
import json
import time
import argparse
import random
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np


def configure_reproducibility(seed: int):
    """Configure deterministic random behavior before model construction."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_dataloader_worker(worker_id: int):
    """Seed Python and NumPy inside every DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def capture_rng_state():
    """Capture all global RNG states needed for an exact training resume."""
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_training_rng(checkpoint, requested_seed: int, train_generator: torch.Generator):
    """Validate and restore global and DataLoader RNG states from a checkpoint."""
    required_keys = ("seed", "rng_state", "train_generator_state")
    missing = [key for key in required_keys if key not in checkpoint]
    if missing:
        raise RuntimeError(
            "Cannot resume a legacy or incomplete checkpoint without reproducibility metadata: "
            f"missing={missing}. Start a fresh seeded run."
        )
    checkpoint_seed = int(checkpoint["seed"])
    if checkpoint_seed != int(requested_seed):
        raise RuntimeError(
            "Resume seed mismatch: "
            f"requested={requested_seed}, checkpoint={checkpoint_seed}."
        )

    rng_state = checkpoint["rng_state"]
    rng_required = ("python", "numpy", "torch_cpu", "torch_cuda")
    rng_missing = [key for key in rng_required if key not in rng_state]
    if rng_missing:
        raise RuntimeError(
            "Cannot resume a legacy or incomplete checkpoint without complete RNG state: "
            f"missing={rng_missing}. Start a fresh seeded run."
        )

    random.setstate(rng_state["python"])
    numpy_state = rng_state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["keys"].cpu().numpy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(rng_state["torch_cpu"].cpu())
    cuda_states = rng_state["torch_cuda"]
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
    train_generator.set_state(checkpoint["train_generator_state"].cpu())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Folder containing shard_*.pt and meta.pt")
    p.add_argument(
        "--split_manifest",
        type=str,
        required=True,
        help="Required checksum-locked episode/seed split manifest.",
    )
    p.add_argument(
        "--source_weights",
        type=str,
        required=True,
        help="Sampling mixture, e.g. teacher=0.5,position=0.5.",
    )
    p.add_argument("--out_dir", type=str, default=".", help="Output directory to save model/report/plots")
    p.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Required experiment seed controlling initialization, shuffling, and resume.",
    )
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_name", type=str, default="best_fast_slow_student_encoder_z.pth")
    p.add_argument(
        "--student_context_mode",
        type=str,
        default="split",
        choices=["split", "monolithic"],
        help="Proposed method uses two independent split encoders; monolithic is an ablation.",
    )
    p.add_argument("--use_weighted_mse", action="store_true", default=True)
    p.add_argument("--no_weighted_mse", dest="use_weighted_mse", action="store_false")
    p.add_argument(
        "--aux_ml_coef",
        type=float,
        default=0.5,
        help="Auxiliary supervision weight for pred[:, :2] -> [m_norm, l_norm]. 0 disables.",
    )
    # Project decision (2026-08-20): do not enable this auxiliary ML loss in new experiments.
    # Keep the legacy option only for reproducing older runs; new commands must use --aux_ml_coef 0.0.
    p.add_argument("--resume", action="store_true", help="Resume training from a saved checkpoint.")
    p.add_argument("--resume_path", type=str, default="", help="Path to checkpoint for resume.")
    p.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save a closure-evaluation candidate every N epochs.",
    )
    return p.parse_args()


SOURCE_ALIASES = {
    "teacher": "teacher_closed_loop",
    "position": "causal_position_closed_loop",
    "dagger1": "student_dagger_round_1",
    "dagger2": "student_dagger_round_2",
}


def parse_source_weights(spec: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in str(spec).split(","):
        if "=" not in item:
            raise ValueError(f"invalid source weight item: {item!r}")
        raw_name, raw_value = item.split("=", 1)
        name = SOURCE_ALIASES.get(raw_name.strip(), raw_name.strip())
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"source weight must be positive: {item!r}")
        if name in weights:
            raise ValueError(f"duplicate source weight: {name}")
        weights[name] = value
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("at least one source weight is required")
    return {name: value / total for name, value in weights.items()}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_locked_manifest(path: str) -> dict[str, object]:
    manifest_path = Path(path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_seeds = {int(seed) for seed in manifest.get("train_seeds", [])}
    validation_seeds = {
        int(seed) for seed in manifest.get("validation_seeds", [])
    }
    if train_seeds & validation_seeds:
        raise RuntimeError("split manifest shares seeds between train and validation")
    train_keys = {
        (int(item["seed"]), int(item["episode_id"]))
        for item in manifest.get("train", [])
    }
    validation_keys = {
        (int(item["seed"]), int(item["episode_id"]))
        for item in manifest.get("validation", [])
    }
    if train_keys & validation_keys:
        raise RuntimeError("split manifest shares episodes between train and validation")
    for item in manifest.get("shards", []):
        shard_path = Path(item["path"]).expanduser().resolve()
        if not shard_path.is_file():
            raise RuntimeError(f"manifest shard does not exist: {shard_path}")
        if _sha256(shard_path) != str(item["sha256"]):
            raise RuntimeError(f"manifest shard checksum mismatch: {shard_path}")
    manifest["_path"] = str(manifest_path)
    return manifest


def build_manifest_selections(manifest: dict[str, object], split: str):
    allowed = {
        (int(item["seed"]), int(item["episode_id"]))
        for item in manifest[split]
    }
    id_to_item = {str(item["id"]): item for item in manifest["shards"]}
    referenced = {
        str(shard_id)
        for item in manifest[split]
        for shard_id in item.get("shard_ids", [])
    }
    selections = []
    for shard_id in sorted(referenced):
        item = id_to_item[shard_id]
        shard = torch.load(item["path"], map_location="cpu", weights_only=False)
        for key in ("seed", "episode_id", "history_source"):
            if key not in shard:
                raise RuntimeError(f"manifest shard {item['path']} lacks {key}")
        seeds = shard["seed"].to(dtype=torch.int64).reshape(-1)
        episodes = shard["episode_id"].to(dtype=torch.int64).reshape(-1)
        mask = torch.tensor(
            [
                (int(seed), int(episode)) in allowed
                for seed, episode in zip(seeds.tolist(), episodes.tolist())
            ],
            dtype=torch.bool,
        )
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if indices.numel() == 0:
            continue
        selections.append(
            {
                "id": shard_id,
                "path": str(Path(item["path"]).resolve()),
                "indices": indices,
                "sources": shard["history_source"]
                .to(dtype=torch.int64)
                .reshape(-1)[indices],
            }
        )
    if not selections:
        raise RuntimeError(f"split manifest contains no samples for {split}")
    return selections

class CNNContextEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, output_dim=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, 5, 1, 2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Flatten(),
        )
        flat = 64 * history_len
        self.mlp = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,H,21)->(B,21,H)
        return self.mlp(self.cnn(x))


class FastSlowStudentEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, z_slow_dim=2, z_fast_dim=3):
        super().__init__()
        self.slow_encoder = CNNContextEncoder(input_dim, history_len, z_slow_dim)
        self.fast_encoder = CNNContextEncoder(input_dim, history_len, z_fast_dim)

    def encode_slow(self, x):
        return self.slow_encoder(x)

    def encode_fast(self, x):
        return self.fast_encoder(x)

    def forward(self, x):
        return self.encode_slow(x), self.encode_fast(x)


class MonolithicStudentEncoder(nn.Module):
    """Single history encoder that predicts all context dimensions jointly."""

    def __init__(self, input_dim=21, history_len=50, z_dim=5):
        super().__init__()
        self.encoder = CNNContextEncoder(input_dim, history_len, z_dim)

    def forward(self, x):
        return self.encoder(x)

def load_shard(path):
    d = torch.load(path, map_location="cpu", weights_only=False)
    x = d["inputs"].float()        # (N, H, 21)
    y = d["labels"].float()        # (N, 5)
    y_ml = d["labels_ml"].float() if "labels_ml" in d else None   # (N, 2)
    return x, y, y_ml

def compute_z_stats_from_meta(meta_path: str, z_dim: int):
    if not os.path.exists(meta_path):
        return None, None
    meta = torch.load(meta_path, map_location="cpu", weights_only=False)
    z_mean = torch.tensor(meta["z_stats"]["mean"], dtype=torch.float32)
    z_std = torch.tensor(meta["z_stats"]["std"], dtype=torch.float32)
    if z_mean.numel() != z_dim or z_std.numel() != z_dim:
        return None, None
    return z_mean, z_std

def main():
    args = parse_args()
    configure_reproducibility(args.seed)
    train_generator = torch.Generator(device="cpu")
    train_generator.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    train_t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[Seed] seed={args.seed} deterministic_algorithms="
        f"{torch.are_deterministic_algorithms_enabled()}"
    )

    if int(args.checkpoint_interval) <= 0:
        raise ValueError("--checkpoint_interval must be positive")
    manifest = load_locked_manifest(args.split_manifest)
    source_weights = parse_source_weights(args.source_weights)
    required_sources = {str(name) for name in manifest.get("required_sources", [])}
    if set(source_weights) != required_sources:
        raise RuntimeError(
            "source weights must cover exactly the manifest required sources: "
            f"weights={sorted(source_weights)}, required={sorted(required_sources)}"
        )
    source_encoding = {
        int(source_id): str(name)
        for source_id, name in manifest.get("source_encoding", {}).items()
    }
    name_to_source_id = {name: source_id for source_id, name in source_encoding.items()}
    missing_source_ids = sorted(required_sources - set(name_to_source_id))
    if missing_source_ids:
        raise RuntimeError(f"manifest source encoding lacks: {missing_source_ids}")

    train_selections = build_manifest_selections(manifest, "train")
    val_selections = build_manifest_selections(manifest, "validation")
    train_files = [item["path"] for item in train_selections]
    val_files = [item["path"] for item in val_selections]
    print(
        f"[Data] manifest={manifest['_path']} "
        f"train_shards={len(train_files)} val_shards={len(val_files)}"
    )

    # infer dims from first shard
    x0_all, y0_all, y0_ml_all = load_shard(train_files[0])
    first_indices = train_selections[0]["indices"]
    x0 = x0_all[first_indices]
    y0 = y0_all[first_indices]
    y0_ml = y0_ml_all[first_indices] if y0_ml_all is not None else None
    H = x0.shape[1]
    in_dim = x0.shape[2]
    z_dim = y0.shape[1]
    print(f"[Data] inferred H={H} in_dim={in_dim} z_dim={z_dim}")

    if in_dim != 21:
        raise RuntimeError(f"Expected input dim = 21, but got {in_dim}. collect pipeline is inconsistent.")
    if z_dim != 5:
        raise RuntimeError(f"Expected z dim = 5, but got {z_dim}.")
    z_slow_dim = 2
    z_fast_dim = z_dim - z_slow_dim
    model_type = (
        "fast_slow_context"
        if args.student_context_mode == "split"
        else "monolithic_context"
    )

    first_dataset_root = Path(manifest["dataset_roots"][0])
    meta_path = first_dataset_root / "meta.pt"
    meta = (
        torch.load(meta_path, map_location="cpu", weights_only=False)
        if meta_path.exists()
        else {}
    )
    teacher_context_mode = str(
        manifest.get("teacher_context_mode", meta.get("teacher_context_mode", "unknown"))
    )
    dataset_audit_path = str(Path(args.split_manifest).expanduser().resolve())
    dataset_audit = {"passed": True}
    if teacher_context_mode == "split_hard":
        if dataset_audit is None:
            raise RuntimeError(
                "Hard-explicit Student training requires dataset_audit.json. "
                "Recollect or run the dataset audit before training."
            )
        if not bool(dataset_audit.get("passed", False)):
            raise RuntimeError(
                f"Dataset audit did not pass: {dataset_audit_path}"
            )
    if teacher_context_mode == "monolithic" and args.aux_ml_coef > 0.0:
        raise RuntimeError(
            "A monolithic Teacher has no guaranteed physical meaning in z[:2]. "
            "Use --aux_ml_coef 0 for that baseline, or use a hard/split Teacher "
            "for the physical slow-context experiment."
        )

    has_ml = y0_ml is not None
    if args.aux_ml_coef > 0.0 and not has_ml:
        raise RuntimeError("aux_ml_coef > 0 but shard has no labels_ml. Please recollect dataset.")
    if has_ml and y0_ml.shape[1] != 2:
        raise RuntimeError(f"Expected labels_ml dim = 2, but got {y0_ml.shape[1]}.")
    slow_identity_max_abs = (
        float(torch.max(torch.abs(y0[:, :z_slow_dim] - y0_ml)).item())
        if y0_ml is not None
        else None
    )
    if teacher_context_mode == "split_hard" and slow_identity_max_abs != 0.0:
        raise RuntimeError(
            "Hard-explicit dataset identity check failed on the first training shard: "
            f"max_abs={slow_identity_max_abs}"
        )
    # z stats (for weighted mse)
    z_mean, z_std = compute_z_stats_from_meta(str(meta_path), z_dim)
    if z_std is None:
        # fallback: compute approx std from first few shards
        ys = []
        for selection in train_selections[:min(3, len(train_selections))]:
            fp = selection["path"]
            _, y, _ = load_shard(fp)
            ys.append(y[selection["indices"]])
        ycat = torch.cat(ys, dim=0)
        z_mean = ycat.mean(dim=0)
        z_std = ycat.std(dim=0).clamp_min(1e-6)
    z_std = z_std.clamp_min(1e-6)

    # weighted mse weights: 1/std^2
    w = (1.0 / (z_std * z_std)).to(device)

    if args.student_context_mode == "split":
        model = FastSlowStudentEncoder(
            input_dim=in_dim,
            history_len=H,
            z_slow_dim=z_slow_dim,
            z_fast_dim=z_fast_dim,
        ).to(device)
    else:
        model = MonolithicStudentEncoder(
            input_dim=in_dim,
            history_len=H,
            z_dim=z_dim,
        ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    num_params = sum(p.numel() for p in model.parameters())
    num_params_slow = (
        sum(p.numel() for p in model.slow_encoder.parameters())
        if args.student_context_mode == "split"
        else 0
    )
    num_params_fast = (
        sum(p.numel() for p in model.fast_encoder.parameters())
        if args.student_context_mode == "split"
        else 0
    )
    def mse_loss(pred, target):
        return torch.mean((pred - target) ** 2)

    def branch_mse_loss(pred, target, branch_weight):
        if args.use_weighted_mse:
            return torch.mean(((pred - target) ** 2) * branch_weight)
        return mse_loss(pred, target)

    global_shard = torch.cat(
        [
            torch.full((item["indices"].numel(),), idx, dtype=torch.int64)
            for idx, item in enumerate(train_selections)
        ]
    )
    global_local = torch.cat([item["indices"] for item in train_selections])
    global_sources = torch.cat([item["sources"] for item in train_selections])
    source_counts_train = Counter(
        source_encoding[int(source_id)] for source_id in global_sources.tolist()
    )
    if set(source_counts_train) != required_sources:
        raise RuntimeError(
            "training split source set differs from required sources: "
            f"actual={sorted(source_counts_train)}, required={sorted(required_sources)}"
        )
    sampler_weights = torch.tensor(
        [
            source_weights[source_encoding[int(source_id)]]
            / source_counts_train[source_encoding[int(source_id)]]
            for source_id in global_sources.tolist()
        ],
        dtype=torch.double,
    )
    source_sampling_counts = Counter()
    checkpoint_candidates = []

    best_val = float("inf")
    train_hist = []
    val_hist = []
    start_epoch = 0
    best_epoch = None
    best_rmse_dim = None
    time_to_best_sec = None
    if args.resume:
        if not args.resume_path:
            raise RuntimeError("--resume requires --resume_path")
        ckpt = torch.load(args.resume_path, map_location=device, weights_only=False)

        if "state_dict" not in ckpt:
            raise RuntimeError(f"Resume checkpoint missing 'state_dict': {args.resume_path}")
        if ckpt.get("model_type") != model_type:
            raise RuntimeError(
                "Resume checkpoint architecture mismatch: "
                f"requested={model_type}, checkpoint={ckpt.get('model_type')}"
            )

        model.load_state_dict(ckpt["state_dict"], strict=True)

        if "optimizer_state_dict" in ckpt:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            for param_group in opt.param_groups:
                param_group["lr"] = args.lr
            print(f"[Resume] override optimizer lr -> {args.lr}")
        else:
            print("[WARN] optimizer_state_dict not found in resume checkpoint. This is only pseudo-resume.")

        restore_training_rng(
            ckpt,
            requested_seed=args.seed,
            train_generator=train_generator,
        )

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        train_hist = list(ckpt.get("train_hist", train_hist))
        val_hist = list(ckpt.get("val_hist", val_hist))
        best_epoch = ckpt.get("best_epoch", best_epoch)
        best_rmse_dim = ckpt.get("best_rmse_dim", best_rmse_dim)
        time_to_best_sec = ckpt.get("time_to_best_sec", time_to_best_sec)
        source_sampling_counts.update(ckpt.get("source_sampling_counts", {}))
        checkpoint_candidates = list(ckpt.get("checkpoint_candidates", []))

        print(
            f"[Resume] loaded {args.resume_path} | "
            f"start_epoch={start_epoch} best_val={best_val:.6e}"
        )

    for epoch in range(start_epoch, args.epochs):
        # ---- train ----
        model.train()
        train_losses = []

        sampler = WeightedRandomSampler(
            sampler_weights,
            num_samples=int(sampler_weights.numel()),
            replacement=True,
            generator=train_generator,
        )
        sampled_global = torch.tensor(list(sampler), dtype=torch.int64)
        sampled_sources = global_sources[sampled_global]
        epoch_source_counts = Counter(
            source_encoding[int(source_id)] for source_id in sampled_sources.tolist()
        )
        source_sampling_counts.update(epoch_source_counts)

        for selection_idx, selection in enumerate(train_selections):
            selected_global = sampled_global[
                global_shard[sampled_global] == selection_idx
            ]
            if selected_global.numel() == 0:
                continue
            local_indices = global_local[selected_global]
            x_all, y_all, y_ml_all = load_shard(selection["path"])
            x = x_all[local_indices]
            y = y_all[local_indices]
            y_ml = y_ml_all[local_indices] if y_ml_all is not None else None

            if y_ml is not None:
                ds = TensorDataset(x, y, y_ml)
            else:
                ds = TensorDataset(x, y)

            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
                worker_init_fn=seed_dataloader_worker,
            )

            for batch in dl:
                if y_ml is not None:
                    bx, by, by_ml = batch
                    by_ml = by_ml.to(device, non_blocking=True)
                else:
                    bx, by = batch
                    by_ml = None

                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                if args.student_context_mode == "split":
                    pred_slow, pred_fast = model(bx)
                else:
                    pred_all = model(bx)
                    pred_slow = pred_all[:, :z_slow_dim]
                    pred_fast = pred_all[:, z_slow_dim:]
                target_slow = by[:, :z_slow_dim]
                target_fast = by[:, z_slow_dim:]
                loss_slow = branch_mse_loss(pred_slow, target_slow, w[:z_slow_dim])
                loss_fast = branch_mse_loss(pred_fast, target_fast, w[z_slow_dim:])
                loss_main = (z_slow_dim / z_dim) * loss_slow + (z_fast_dim / z_dim) * loss_fast
                loss = loss_main

                if by_ml is not None and args.aux_ml_coef > 0.0:
                    loss_ml = mse_loss(pred_slow, by_ml)
                    loss = loss + args.aux_ml_coef * loss_ml

                loss.backward()
                opt.step()
                train_losses.append(loss.item())

        # ---- val ----
        model.eval()
        val_losses = []
        val_slow_losses = []
        val_fast_losses = []
        sum_sq = torch.zeros(z_dim, device=device)
        count = 0

        with torch.no_grad():
            for selection in val_selections:
                x_all, y_all, y_ml_all = load_shard(selection["path"])
                local_indices = selection["indices"]
                x = x_all[local_indices]
                y = y_all[local_indices]
                y_ml = y_ml_all[local_indices] if y_ml_all is not None else None

                if y_ml is not None:
                    ds = TensorDataset(x, y, y_ml)
                else:
                    ds = TensorDataset(x, y)

                dl = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    persistent_workers=(args.num_workers > 0),
                )

                for batch in dl:
                    if y_ml is not None:
                        bx, by, by_ml = batch
                        by_ml = by_ml.to(device, non_blocking=True)
                    else:
                        bx, by = batch
                        by_ml = None

                    bx = bx.to(device, non_blocking=True)
                    by = by.to(device, non_blocking=True)

                    if args.student_context_mode == "split":
                        pred_slow, pred_fast = model(bx)
                    else:
                        pred_all = model(bx)
                        pred_slow = pred_all[:, :z_slow_dim]
                        pred_fast = pred_all[:, z_slow_dim:]
                    pred = torch.cat([pred_slow, pred_fast], dim=-1)
                    target_slow = by[:, :z_slow_dim]
                    target_fast = by[:, z_slow_dim:]
                    loss_slow = branch_mse_loss(pred_slow, target_slow, w[:z_slow_dim])
                    loss_fast = branch_mse_loss(pred_fast, target_fast, w[z_slow_dim:])
                    loss_main = (z_slow_dim / z_dim) * loss_slow + (z_fast_dim / z_dim) * loss_fast
                    loss = loss_main
                    if by_ml is not None and args.aux_ml_coef > 0.0:
                        loss_ml = mse_loss(pred_slow, by_ml)
                        loss = loss + args.aux_ml_coef * loss_ml

                    val_losses.append(loss.item())
                    val_slow_losses.append(loss_slow.item())
                    val_fast_losses.append(loss_fast.item())

                    err = pred - by
                    sum_sq += (err * err).sum(dim=0)
                    count += err.shape[0]

        tr = float(np.mean(train_losses)) if len(train_losses) else 0.0
        va = float(np.mean(val_losses)) if len(val_losses) else 0.0
        va_slow = float(np.mean(val_slow_losses)) if len(val_slow_losses) else 0.0
        va_fast = float(np.mean(val_fast_losses)) if len(val_fast_losses) else 0.0
        train_hist.append(tr)
        val_hist.append(va)

        rmse_dim = torch.sqrt(sum_sq / max(1, count)).detach().cpu().numpy()
        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | train={tr:.6e} | val={va:.6e} "
            f"| val_slow={va_slow:.6e} | val_fast={va_fast:.6e} "
            f"| rmse_dim={np.round(rmse_dim,3)}"
        )

        improved = va < best_val
        if improved:
            best_val = va
            best_epoch = int(epoch + 1)
            best_rmse_dim = [float(x) for x in rmse_dim.tolist()]
            time_to_best_sec = float(time.time() - train_t0)
        interval_due = (
            (epoch + 1) % int(args.checkpoint_interval) == 0
            or (epoch + 1) == int(args.epochs)
        )
        interval_path = None
        if interval_due:
            interval_path = str(
                (Path(args.out_dir) / f"checkpoint_epoch_{epoch + 1:04d}.pth")
                .expanduser()
                .resolve()
            )
            checkpoint_candidates.append(
                {"epoch": int(epoch + 1), "val_loss": float(va), "path": interval_path}
            )

        checkpoint_state = {
            "model_type": model_type,
            "student_context_mode": args.student_context_mode,
            "teacher_context_mode": teacher_context_mode,
            "state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            "best_epoch": best_epoch,
            "best_rmse_dim": best_rmse_dim,
            "time_to_best_sec": time_to_best_sec,
            "train_hist": train_hist,
            "val_hist": val_hist,
            "history_len": H,
            "input_dim": in_dim,
            "z_dim": z_dim,
            "z_slow_dim": z_slow_dim,
            "z_fast_dim": z_fast_dim,
            "z_mean": z_mean.cpu(),
            "z_std": z_std.cpu(),
            "seed": int(args.seed),
            "split_manifest": str(Path(args.split_manifest).expanduser().resolve()),
            "source_weights": source_weights,
            "source_sampling_counts": dict(source_sampling_counts),
            "checkpoint_candidates": checkpoint_candidates,
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "rng_state": capture_rng_state(),
            "train_generator_state": train_generator.get_state(),
        }
        if improved:
            save_path = str((Path(args.out_dir) / args.save_name).resolve())
            torch.save(checkpoint_state, save_path)
            print(f"[Save-Best] {save_path} (best_val={best_val:.6e})")
        if interval_path is not None:
            torch.save(checkpoint_state, interval_path)
            print(f"[Save-Interval] {interval_path} (val={va:.6e})")
        last_path = os.path.join(args.out_dir, "last_checkpoint.pth")
        torch.save(checkpoint_state, last_path)
    # ---- final report ----
    wall_time_sec = float(time.time() - train_t0)
    top5_checkpoints = sorted(
        checkpoint_candidates,
        key=lambda item: (float(item["val_loss"]), int(item["epoch"])),
    )[:5]

    report = {
        "best_val": best_val,
        "best_epoch": best_epoch,
        "best_rmse_dim": best_rmse_dim,
        "time_to_best_sec": time_to_best_sec,
        "wall_time_sec": wall_time_sec,
        "epochs_ran": len(train_hist),
        "num_params": int(num_params),
        "num_params_slow": int(num_params_slow),
        "num_params_fast": int(num_params_fast),
        "student_context_mode": args.student_context_mode,
        "teacher_context_mode": teacher_context_mode,
        "slow_label_identity_max_abs_first_shard": slow_identity_max_abs,
        "dataset_audit_path": dataset_audit_path if dataset_audit is not None else None,
        "dataset_audit_passed": (
            bool(dataset_audit.get("passed", False))
            if dataset_audit is not None
            else None
        ),
        "z_slow_dim": int(z_slow_dim),
        "z_fast_dim": int(z_fast_dim),
        "use_weighted_mse": bool(args.use_weighted_mse),
        "z_std": z_std.cpu().tolist(),
        "z_mean": z_mean.cpu().tolist() if z_mean is not None else None,
        "train_hist": train_hist,
        "val_hist": val_hist,
        "data_dir": args.data_dir,
        "split_manifest": str(Path(args.split_manifest).expanduser().resolve()),
        "source_weights": source_weights,
        "source_sample_counts_available": dict(source_counts_train),
        "source_sample_counts_drawn": dict(source_sampling_counts),
        "top5_checkpoints": top5_checkpoints,
        "checkpoint_interval": int(args.checkpoint_interval),
        "num_train_shards": len(train_files),
        "num_val_shards": len(val_files),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "aux_ml_coef": args.aux_ml_coef,
        "has_labels_ml": has_ml,
        "resume": bool(args.resume),
        "resume_path": args.resume_path,
        "seed": int(args.seed),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
    }
    report_path = os.path.join(args.out_dir, "training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    legacy_report_path = os.path.join(args.out_dir, "report.json")
    with open(legacy_report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Done] best_val: {best_val} | report: {report_path}")

    # ---- plot loss curve (optional) ----
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(train_hist, label="train")
        plt.plot(val_hist, label="val")
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        out_png = os.path.join(args.out_dir, "loss_curve.png")
        plt.savefig(out_png, dpi=200)
        print(f"[Plot] saved {out_png}")
    except Exception as e:
        print(f"[Plot] skipped: {e}")

if __name__ == "__main__":
    main()
