# train_student_z.py
# Train student encoder phi(history) -> z_teacher (5-dim)
# Paper-grade: weighted MSE, per-dim RMSE, report.json, loss_curve.png

import os
import glob
import json
import argparse
from dataclasses import dataclass
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
    p.add_argument("--save_name", type=str, default="best_student_encoder_z.pth")
    p.add_argument("--use_weighted_mse", action="store_true", default=True)
    p.add_argument("--no_weighted_mse", dest="use_weighted_mse", action="store_false")
    p.add_argument("--slow_dims", type=int, default=2, help="Number of slow latent dims (z0..z{slow_dims-1}).")
    p.add_argument("--slow_loss_coef", type=float, default=2.0, help="Loss multiplier for slow dims.")
    p.add_argument("--fast_loss_coef", type=float, default=1.0, help="Loss multiplier for fast dims.")
    p.add_argument("--delta_loss_coef", type=float, default=0.5, help="Auxiliary fast-delta loss multiplier.")
    p.add_argument("--fast_huber_beta", type=float, default=0.05, help="Huber beta for fast-dim loss.")
    return p.parse_args()

class CNNStudentEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, output_dim=5, slow_dims=2):
        super().__init__()
        self.slow_dims = int(slow_dims)
        self.fast_dims = max(0, int(output_dim) - self.slow_dims)
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, 5, 1, 2), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 64, 3, 1, 1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Flatten(),
        )
        flat = 64 * history_len
        self.backbone = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.z_head = nn.Linear(128, output_dim)
        self.fast_delta_head = nn.Linear(128, self.fast_dims) if self.fast_dims > 0 else None

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,H,21)->(B,21,H)
        feat = self.backbone(self.cnn(x))
        z = self.z_head(feat)
        delta_fast = self.fast_delta_head(feat) if self.fast_delta_head is not None else None
        return z, delta_fast

@dataclass
class ShardData:
    x: torch.Tensor
    y: torch.Tensor
    fast_delta: torch.Tensor
    fast_delta_valid: torch.Tensor

def load_shard(path, num_envs: int, slow_dims: int):
    d = torch.load(path, map_location="cpu")
    x = d["inputs"].float()   # (N, H, 21)
    y = d["labels"].float()   # (N, 5)
    fast_dims = max(0, y.shape[1] - slow_dims)

    if fast_dims > 0 and (num_envs > 0) and (x.shape[0] >= num_envs):
        fast = y[:, slow_dims:]
        prev_fast = torch.roll(fast, shifts=num_envs, dims=0)
        fast_delta = fast - prev_fast
        valid = torch.ones(x.shape[0], 1, dtype=torch.float32)
        valid[:num_envs] = 0.0
    else:
        fast_delta = torch.zeros((x.shape[0], fast_dims), dtype=torch.float32)
        valid = torch.zeros((x.shape[0], 1), dtype=torch.float32)

    return ShardData(x=x, y=y, fast_delta=fast_delta, fast_delta_valid=valid)

def compute_z_stats_from_meta(meta_path: str, z_dim: int):
    if not os.path.exists(meta_path):
        return None, None, None
    meta = torch.load(meta_path, map_location="cpu")
    z_mean = torch.tensor(meta["z_stats"]["mean"], dtype=torch.float32)
    z_std = torch.tensor(meta["z_stats"]["std"], dtype=torch.float32)
    num_envs = int(meta.get("num_envs", 0))
    if z_mean.numel() != z_dim or z_std.numel() != z_dim:
        return None, None, num_envs
    return z_mean, z_std, num_envs

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

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
    slow_dims = int(args.slow_dims)
    meta_path = os.path.join(args.data_dir, "meta.pt")
    meta = torch.load(meta_path, map_location="cpu") if os.path.exists(meta_path) else {}
    num_envs = int(meta.get("num_envs", 0))

    s0 = load_shard(train_files[0], num_envs=num_envs, slow_dims=slow_dims)
    x0, y0 = s0.x, s0.y
    H = x0.shape[1]
    in_dim = x0.shape[2]
    z_dim = y0.shape[1]
    fast_dims = max(0, z_dim - slow_dims)
    print(f"[Data] inferred H={H} in_dim={in_dim} z_dim={z_dim}")
    print(f"[Data] slow_dims={slow_dims} fast_dims={fast_dims} num_envs(meta)={num_envs}")

    # z stats (for weighted mse)
    z_mean, z_std, meta_num_envs = compute_z_stats_from_meta(meta_path, z_dim)
    if num_envs == 0:
        num_envs = int(meta_num_envs or 0)
    if z_std is None:
        # fallback: compute approx std from first few shards
        ys = []
        for fp in train_files[:min(3, len(train_files))]:
            ys.append(load_shard(fp, num_envs=num_envs, slow_dims=slow_dims).y)
        ycat = torch.cat(ys, dim=0)
        z_mean = ycat.mean(dim=0)
        z_std = ycat.std(dim=0).clamp_min(1e-6)
    z_std = z_std.clamp_min(1e-6)

    # weighted mse weights: 1/std^2
    w = (1.0 / (z_std * z_std)).to(device)

    model = CNNStudentEncoder(input_dim=in_dim, history_len=H, output_dim=z_dim, slow_dims=slow_dims).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    def split_loss(pred_z, target_z):
        if slow_dims > 0:
            slow_err = ((pred_z[:, :slow_dims] - target_z[:, :slow_dims]) ** 2)
            if args.use_weighted_mse:
                slow_err = slow_err * w[:slow_dims]
            slow_loss = slow_err.mean()
        else:
            slow_loss = torch.zeros((), device=device)

        if fast_dims > 0:
            fast_err = pred_z[:, slow_dims:] - target_z[:, slow_dims:]
            fast_huber = nn.functional.smooth_l1_loss(
                fast_err, torch.zeros_like(fast_err), beta=float(args.fast_huber_beta), reduction="none"
            )
            if args.use_weighted_mse:
                fast_huber = fast_huber * w[slow_dims:]
            fast_loss = fast_huber.mean()
        else:
            fast_loss = torch.zeros((), device=device)
        total = float(args.slow_loss_coef) * slow_loss + float(args.fast_loss_coef) * fast_loss
        return total, slow_loss.detach(), fast_loss.detach()

    best_val = float("inf")
    train_hist = []
    val_hist = []
    train_slow_hist = []
    train_fast_hist = []
    val_slow_hist = []
    val_fast_hist = []
    val_delta_hist = []

    for epoch in range(args.epochs):
        # ---- train ----
        model.train()
        train_losses = []
        train_slow_losses = []
        train_fast_losses = []

        for fp in train_files:
            shard = load_shard(fp, num_envs=num_envs, slow_dims=slow_dims)
            ds = TensorDataset(shard.x, shard.y, shard.fast_delta, shard.fast_delta_valid)
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
            )
            for bx, by, bdf, bvalid in dl:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                bdf = bdf.to(device, non_blocking=True)
                bvalid = bvalid.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred_z, pred_df = model(bx)
                loss_main, slow_l, fast_l = split_loss(pred_z, by)

                loss_delta = torch.zeros((), device=device)
                if fast_dims > 0 and pred_df is not None:
                    delta_err = (pred_df - bdf) ** 2
                    valid_mass = bvalid.sum().clamp_min(1.0)
                    loss_delta = (delta_err * bvalid).sum() / (valid_mass * fast_dims)

                loss = loss_main + float(args.delta_loss_coef) * loss_delta
                loss.backward()
                opt.step()
                train_losses.append(loss.item())
                train_slow_losses.append(float(slow_l.item()))
                train_fast_losses.append(float(fast_l.item()))

        # ---- val ----
        model.eval()
        val_losses = []
        val_slow_losses = []
        val_fast_losses = []
        val_delta_losses = []
        # also compute per-dim rmse on a running basis
        sum_sq = torch.zeros(z_dim, device=device)
        count = 0
        with torch.no_grad():
            for fp in val_files:
                shard = load_shard(fp, num_envs=num_envs, slow_dims=slow_dims)
                ds = TensorDataset(shard.x, shard.y, shard.fast_delta, shard.fast_delta_valid)
                dl = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    persistent_workers=(args.num_workers > 0),
                )
                for bx, by, bdf, bvalid in dl:
                    bx = bx.to(device, non_blocking=True)
                    by = by.to(device, non_blocking=True)
                    bdf = bdf.to(device, non_blocking=True)
                    bvalid = bvalid.to(device, non_blocking=True)
                    pred_z, pred_df = model(bx)
                    loss_main, slow_l, fast_l = split_loss(pred_z, by)

                    loss_delta = torch.zeros((), device=device)
                    if fast_dims > 0 and pred_df is not None:
                        delta_err = (pred_df - bdf) ** 2
                        valid_mass = bvalid.sum().clamp_min(1.0)
                        loss_delta = (delta_err * bvalid).sum() / (valid_mass * fast_dims)

                    loss = loss_main + float(args.delta_loss_coef) * loss_delta
                    val_losses.append(loss.item())
                    val_slow_losses.append(float(slow_l.item()))
                    val_fast_losses.append(float(fast_l.item()))
                    val_delta_losses.append(float(loss_delta.item()))

                    err = pred_z - by
                    sum_sq += (err * err).sum(dim=0)
                    count += err.shape[0]

        tr = float(np.mean(train_losses)) if len(train_losses) else 0.0
        va = float(np.mean(val_losses)) if len(val_losses) else 0.0
        tr_slow = float(np.mean(train_slow_losses)) if len(train_slow_losses) else 0.0
        tr_fast = float(np.mean(train_fast_losses)) if len(train_fast_losses) else 0.0
        va_slow = float(np.mean(val_slow_losses)) if len(val_slow_losses) else 0.0
        va_fast = float(np.mean(val_fast_losses)) if len(val_fast_losses) else 0.0
        va_delta = float(np.mean(val_delta_losses)) if len(val_delta_losses) else 0.0
        train_hist.append(tr)
        val_hist.append(va)
        train_slow_hist.append(tr_slow)
        train_fast_hist.append(tr_fast)
        val_slow_hist.append(va_slow)
        val_fast_hist.append(va_fast)
        val_delta_hist.append(va_delta)

        rmse_dim = torch.sqrt(sum_sq / max(1, count)).detach().cpu().numpy()

        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | train={tr:.6e} (slow={tr_slow:.3e}, fast={tr_fast:.3e}) "
            f"| val={va:.6e} (slow={va_slow:.3e}, fast={va_fast:.3e}, dfast={va_delta:.3e}) "
            f"| rmse_dim={np.round(rmse_dim,3)}"
        )

        if va < best_val:
            best_val = va
            save_path = os.path.join(args.out_dir, args.save_name)
            torch.save(model.state_dict(), save_path)
            print(f"[Save] {save_path} (best_val={best_val:.6e})")

    # ---- final report ----
    report = {
        "best_val": best_val,
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
        "slow_dims": slow_dims,
        "fast_dims": fast_dims,
        "slow_loss_coef": args.slow_loss_coef,
        "fast_loss_coef": args.fast_loss_coef,
        "delta_loss_coef": args.delta_loss_coef,
        "fast_huber_beta": args.fast_huber_beta,
        "train_slow_hist": train_slow_hist,
        "train_fast_hist": train_fast_hist,
        "val_slow_hist": val_slow_hist,
        "val_fast_hist": val_fast_hist,
        "val_delta_hist": val_delta_hist,
        "meta_num_envs": num_envs,
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
