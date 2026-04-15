"""PyTorch reference submission — seg-in / seg-out contract.

What it does: runs a small CNN on the input mask to produce a per-pixel
embedding, then assigns each connected region the ID of its centroid embedding's
nearest neighbor in a learnable ID codebook. Since no paired (noisy, canonical)
training data is shipped with the repo, the codebook is initialized as identity
and the CNN weights default to a tiny pass-through — effectively an identity
baseline, but exercising every piece of the torch/CUDA pipeline so participants
can see a real model running inside the sandbox.

Replace `CodebookModel.__init__` / `forward` with your trained weights.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RegionEncoder(nn.Module):
    """Tiny CNN: one-hot-ish mask embedding -> 32-d per-pixel features."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x  # (B, hidden, H, W)


def predict_one(mask: np.ndarray, model: nn.Module) -> np.ndarray:
    """Return an HxW int32 mask with predicted canonical IDs.

    Baseline strategy: for each input region, output its input ID unchanged
    (identity). The CNN is run to demonstrate GPU usage but its output is not
    used for ID reassignment in this baseline — participants plug in their
    learned codebook here.
    """
    mask_t = torch.from_numpy(mask.astype(np.float32))[None, None].to(DEVICE)
    with torch.no_grad():
        _features = model(mask_t)  # exercise GPU; shape (1, hidden, H, W)
    # Identity ID assignment:
    return mask.astype(np.int32)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = RegionEncoder().to(DEVICE).eval()
    seg_files = sorted(INPUT_DIR.glob("*_seg.npy"))
    print(f"[pytorch_baseline] device={DEVICE} torch={torch.__version__} "
          f"inputs={len(seg_files)}", flush=True)

    for in_path in seg_files:
        seg = np.load(in_path, allow_pickle=True).item()
        pred = predict_one(seg["masks"], model)
        out = {
            "masks": pred.astype(np.int32),
            "cell_ids": sorted(int(x) for x in np.unique(pred) if x > 0),
        }
        np.save(OUTPUT_DIR / in_path.name, out)

    print(f"[pytorch_baseline] wrote {len(seg_files)} files", flush=True)


if __name__ == "__main__":
    main()
