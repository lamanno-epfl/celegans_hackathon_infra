"""Train the domain-adapted baseline on simulated (labeled) + real (unlabeled)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baselines.common.rotation6d import matrix_to_rotation_6d, rotation_6d_to_matrix
from baselines.domain_adapted.model import DomainAdaptedRegressor


class SimDataset(Dataset):
    def __init__(self, images_dir: Path, poses: dict):
        self.images_dir = Path(images_dir)
        self.names = sorted(poses.keys())
        self.poses = poses

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = np.load(self.images_dir / name).astype(np.float32)
        p = self.poses[name]
        R = torch.tensor(p["rotation"], dtype=torch.float32)
        t = torch.tensor(p["translation"], dtype=torch.float32)
        return torch.from_numpy(img), R, t


class RealDataset(Dataset):
    def __init__(self, images_dir: Path):
        self.images_dir = Path(images_dir)
        self.names = sorted(p.name for p in self.images_dir.glob("*.npy"))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        img = np.load(self.images_dir / self.names[i]).astype(np.float32)
        return torch.from_numpy(img)


def train(sim_dir: Path, real_dir: Path, out_path: Path, epochs: int = 5, batch_size: int = 8, lr: float = 1e-3):
    poses = json.loads((sim_dir / "poses.json").read_text())
    sim_ds = SimDataset(sim_dir / "images", poses)
    real_ds = RealDataset(real_dir / "images")
    sim_dl = DataLoader(sim_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    real_dl = DataLoader(real_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DomainAdaptedRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        real_iter = iter(real_dl)
        total, count = 0.0, 0
        for img, R, t in sim_dl:
            img, R, t = img.to(device), R.to(device), t.to(device)
            try:
                real_img = next(real_iter).to(device)
            except StopIteration:
                real_iter = iter(real_dl)
                real_img = next(real_iter).to(device)

            alpha = 0.1 + 0.9 * (epoch / max(1, epochs - 1))

            emb_s, rot6_s, trans_s, dom_s = model(img, alpha=alpha)
            _, _, _, dom_r = model(real_img, alpha=alpha)

            R_pred = rotation_6d_to_matrix(rot6_s)
            rot_target = matrix_to_rotation_6d(R)
            rot_loss = nn.functional.mse_loss(rot6_s, rot_target) + ((R_pred - R) ** 2).mean()
            trans_loss = nn.functional.mse_loss(trans_s, t)

            dom_labels_s = torch.zeros(img.shape[0], dtype=torch.long, device=device)
            dom_labels_r = torch.ones(real_img.shape[0], dtype=torch.long, device=device)
            dom_loss = ce(dom_s, dom_labels_s) + ce(dom_r, dom_labels_r)

            loss = rot_loss + trans_loss + dom_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * img.shape[0]
            count += img.shape[0]
        print(f"epoch {epoch + 1}/{epochs} loss={total / max(1, count):.4f} alpha={alpha:.2f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print("saved", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim-dir", required=True)
    p.add_argument("--real-dir", required=True)
    p.add_argument("--out", default="baselines/domain_adapted/model.pt")
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()
    train(Path(args.sim_dir), Path(args.real_dir), Path(args.out), epochs=args.epochs)


if __name__ == "__main__":
    main()
