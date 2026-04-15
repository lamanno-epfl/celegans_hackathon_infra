"""Train the trivial baseline on public simulated data."""
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
from baselines.trivial.model import PoseRegressor


class SimulatedSliceDataset(Dataset):
    def __init__(self, images_dir: Path, poses: dict):
        self.images_dir = Path(images_dir)
        self.names = sorted(poses.keys())
        self.poses = poses

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = np.load(self.images_dir / name).astype(np.float32)
        p = self.poses[name]
        R = torch.tensor(p["rotation"], dtype=torch.float32)
        t = torch.tensor(p["translation"], dtype=torch.float32)
        return torch.from_numpy(img), R, t


def train(public_dir: Path, out_path: Path, epochs: int = 5, batch_size: int = 8, lr: float = 1e-3):
    poses = json.loads((public_dir / "poses.json").read_text())
    ds = SimulatedSliceDataset(public_dir / "images", poses)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseRegressor().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total = 0.0
        count = 0
        for img, R, t in dl:
            img, R, t = img.to(device), R.to(device), t.to(device)
            emb, rot6, trans = model(img)
            R_pred = rotation_6d_to_matrix(rot6)
            rot_target = matrix_to_rotation_6d(R)
            rot_loss = nn.functional.mse_loss(rot6, rot_target)
            # Additional geodesic-ish loss: ||R_pred - R||_F
            frob_loss = ((R_pred - R) ** 2).mean()
            trans_loss = nn.functional.mse_loss(trans, t)
            loss = rot_loss + frob_loss + trans_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * img.shape[0]
            count += img.shape[0]
        print(f"epoch {epoch + 1}/{epochs} loss={total / max(1, count):.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print("saved", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--public-dir", required=True)
    p.add_argument("--out", default="baselines/trivial/model.pt")
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()
    train(Path(args.public_dir), Path(args.out), epochs=args.epochs)


if __name__ == "__main__":
    main()
