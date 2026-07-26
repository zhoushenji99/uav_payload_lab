# train_student_z.py
# Train independent slow/fast student encoders from the same 50-step history.
# Paper-grade: weighted MSE, per-dim RMSE, report.json, loss_curve.png

import os
import glob
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Folder containing shard_*.pt and meta.pt")
    p.add_argument("--out_dir", type=str, default=".", help="Output directory to save model/report/plots")
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
    p.add_argument("--resume", action="store_true", help="Resume training from a saved checkpoint.")
    p.add_argument("--resume_path", type=str, default="", help="Path to checkpoint for resume.")
    return p.parse_args()

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
    d = torch.load(path, map_location="cpu")
    x = d["inputs"].float()        # (N, H, 21)
    y = d["labels"].float()        # (N, 5)
    y_ml = d["labels_ml"].float() if "labels_ml" in d else None   # (N, 2)
    return x, y, y_ml

def compute_z_stats_from_meta(meta_path: str, z_dim: int):
    if not os.path.exists(meta_path):
        return None, None
    meta = torch.load(meta_path, map_location="cpu")
    z_mean = torch.tensor(meta["z_stats"]["mean"], dtype=torch.float32)
    z_std = torch.tensor(meta["z_stats"]["std"], dtype=torch.float32)
    if z_mean.numel() != z_dim or z_std.numel() != z_dim:
        return None, None
    return z_mean, z_std

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    train_t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shard_files = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.pt")))
    assert len(shard_files) > 0, f"No shard_*.pt found in {args.data_dir}"

    # split by files (stable)
    n = len(shard_files)
    n_val = max(1, int(0.1 * n))
    val_files = shard_files[-n_val:]
    train_files = shard_files[:-n_val]
    print(f"[Data] shards={n} train={len(train_files)} val={len(val_files)}")

    # infer dims from first shard
    x0, y0, y0_ml = load_shard(train_files[0])
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

    meta_path = os.path.join(args.data_dir, "meta.pt")
    meta = torch.load(meta_path, map_location="cpu") if os.path.exists(meta_path) else {}
    teacher_context_mode = str(meta.get("teacher_context_mode", "unknown"))
    dataset_audit_path = os.path.join(args.data_dir, "dataset_audit.json")
    dataset_audit = None
    if os.path.exists(dataset_audit_path):
        with open(dataset_audit_path, "r") as audit_file:
            dataset_audit = json.load(audit_file)
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
    z_mean, z_std = compute_z_stats_from_meta(os.path.join(args.data_dir, "meta.pt"), z_dim)
    if z_std is None:
        # fallback: compute approx std from first few shards
        ys = []
        for fp in train_files[:min(3, len(train_files))]:
            _, y, _ = load_shard(fp)
            ys.append(y)
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
        ckpt = torch.load(args.resume_path, map_location=device)

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

        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("best_val", best_val))
        train_hist = list(ckpt.get("train_hist", train_hist))
        val_hist = list(ckpt.get("val_hist", val_hist))

        print(
            f"[Resume] loaded {args.resume_path} | "
            f"start_epoch={start_epoch} best_val={best_val:.6e}"
        )

    for epoch in range(start_epoch, args.epochs):
        # ---- train ----
        model.train()
        train_losses = []

        for fp in train_files:
            x, y, y_ml = load_shard(fp)

            if y_ml is not None:
                ds = TensorDataset(x, y, y_ml)
            else:
                ds = TensorDataset(x, y)

            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=True,
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
            for fp in val_files:
                x, y, y_ml = load_shard(fp)

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

        if va < best_val:
            best_val = va
            best_epoch = int(epoch + 1)
            best_rmse_dim = [float(x) for x in rmse_dim.tolist()]
            time_to_best_sec = float(time.time() - train_t0)
            save_path = os.path.join(args.out_dir, args.save_name)
            torch.save(
                {
                    "model_type": model_type,
                    "student_context_mode": args.student_context_mode,
                    "teacher_context_mode": teacher_context_mode,
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "epoch": epoch,
                    "best_val": best_val,
                    "train_hist": train_hist,
                    "val_hist": val_hist,
                    "history_len": H,
                    "input_dim": in_dim,
                    "z_dim": z_dim,
                    "z_slow_dim": z_slow_dim,
                    "z_fast_dim": z_fast_dim,
                    "z_mean": z_mean.cpu(),
                    "z_std": z_std.cpu(),
                },
                save_path,
            )
            print(f"[Save-Best] {save_path} (best_val={best_val:.6e})")
        last_path = os.path.join(args.out_dir, "last_checkpoint.pth")
        torch.save(
            {
                "model_type": model_type,
                "student_context_mode": args.student_context_mode,
                "teacher_context_mode": teacher_context_mode,
                "state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "epoch": epoch,
                "best_val": best_val,
                "train_hist": train_hist,
                "val_hist": val_hist,
                "history_len": H,
                "input_dim": in_dim,
                "z_dim": z_dim,
                "z_slow_dim": z_slow_dim,
                "z_fast_dim": z_fast_dim,
                "z_mean": z_mean.cpu(),
                "z_std": z_std.cpu(),
            },
            last_path,
        )
    # ---- final report ----
    wall_time_sec = float(time.time() - train_t0)

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
        "num_train_shards": len(train_files),
        "num_val_shards": len(val_files),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "aux_ml_coef": args.aux_ml_coef,
        "has_labels_ml": has_ml,
        "resume": bool(args.resume),
        "resume_path": args.resume_path,
    }
    report_path = os.path.join(args.out_dir, "report.json")
    with open(report_path, "w") as f:
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
