# train_student_z.py
# Train student encoder phi(history) -> z_teacher (5-dim)
# Paper-grade: weighted MSE, per-dim RMSE, report.json, loss_curve.png

import os
import glob
import json
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
    p.add_argument("--save_name", type=str, default="best_student_encoder_z.pth")
    p.add_argument("--use_weighted_mse", action="store_true", default=True)
    p.add_argument("--no_weighted_mse", dest="use_weighted_mse", action="store_false")
    return p.parse_args()

class CNNStudentEncoder(nn.Module):
    def __init__(self, input_dim=21, history_len=50, output_dim=5):
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

def load_shard(path):
    d = torch.load(path, map_location="cpu")
    x = d["inputs"].float()   # (N, H, 21)
    y = d["labels"].float()   # (N, 5)
    return x, y

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
    x0, y0 = load_shard(train_files[0])
    H = x0.shape[1]
    in_dim = x0.shape[2]
    z_dim = y0.shape[1]
    print(f"[Data] inferred H={H} in_dim={in_dim} z_dim={z_dim}")

    # z stats (for weighted mse)
    z_mean, z_std = compute_z_stats_from_meta(os.path.join(args.data_dir, "meta.pt"), z_dim)
    if z_std is None:
        # fallback: compute approx std from first few shards
        ys = []
        for fp in train_files[:min(3, len(train_files))]:
            _, y = load_shard(fp)
            ys.append(y)
        ycat = torch.cat(ys, dim=0)
        z_mean = ycat.mean(dim=0)
        z_std = ycat.std(dim=0).clamp_min(1e-6)
    z_std = z_std.clamp_min(1e-6)

    # weighted mse weights: 1/std^2
    w = (1.0 / (z_std * z_std)).to(device)

    model = CNNStudentEncoder(input_dim=in_dim, history_len=H, output_dim=z_dim).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    def mse_loss(pred, target):
        return torch.mean((pred - target) ** 2)

    def weighted_mse_loss(pred, target):
        # pred/target: (B,z_dim)
        return torch.mean(((pred - target) ** 2) * w)

    best_val = float("inf")
    train_hist = []
    val_hist = []

    for epoch in range(args.epochs):
        # ---- train ----
        model.train()
        train_losses = []

        for fp in train_files:
            x, y = load_shard(fp)
            ds = TensorDataset(x, y)
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=(args.num_workers > 0),
            )
            for bx, by in dl:
                bx = bx.to(device, non_blocking=True)
                by = by.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                pred = model(bx)
                loss = weighted_mse_loss(pred, by) if args.use_weighted_mse else mse_loss(pred, by)
                loss.backward()
                opt.step()
                train_losses.append(loss.item())

        # ---- val ----
        model.eval()
        val_losses = []
        # also compute per-dim rmse on a running basis
        sum_sq = torch.zeros(z_dim, device=device)
        count = 0
        with torch.no_grad():
            for fp in val_files:
                x, y = load_shard(fp)
                ds = TensorDataset(x, y)
                dl = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    persistent_workers=(args.num_workers > 0),
                )
                for bx, by in dl:
                    bx = bx.to(device, non_blocking=True)
                    by = by.to(device, non_blocking=True)
                    pred = model(bx)
                    loss = weighted_mse_loss(pred, by) if args.use_weighted_mse else mse_loss(pred, by)
                    val_losses.append(loss.item())

                    err = pred - by
                    sum_sq += (err * err).sum(dim=0)
                    count += err.shape[0]

        tr = float(np.mean(train_losses)) if len(train_losses) else 0.0
        va = float(np.mean(val_losses)) if len(val_losses) else 0.0
        train_hist.append(tr)
        val_hist.append(va)

        rmse_dim = torch.sqrt(sum_sq / max(1, count)).detach().cpu().numpy()

        print(f"Epoch {epoch+1:03d}/{args.epochs} | train={tr:.6e} | val={va:.6e} | rmse_dim={np.round(rmse_dim,3)}")

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
