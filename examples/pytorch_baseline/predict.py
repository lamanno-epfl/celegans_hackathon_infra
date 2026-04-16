"""PyTorch baseline: seg-in / seg-out + domain-adaptation embeddings.

The server mounts TWO held-out sets into `/input/` at eval time — you don't
see either locally:

  - `/input/sample_*_seg.npy`          simulated held-out (857 samples)
  - `/input/real_manual/*_seg.npy`     REAL held-out embryos (manually
                                       annotated; ground-truth real data)

The 4D reference atlas is bind-mounted READ-ONLY at `/atlas/`:

  - `/atlas/reference.ome.zarr/`       OME-Zarr v3, (T=255, Z=214, Y=356, X=256)
                                       int16 cell labels + uint8 membrane/nucleus
  - `/atlas/name_dictionary.csv`       cell ID -> Sulston lineage name

This baseline opens the atlas (proves the mount + zarr stack work) but does
not USE it for assignment — your job is to slice this 4D volume to derive
canonical cell IDs for each input sample.

In one run, the container must:

1. For every sim sample, emit `/output/<same_name>` with predicted atlas IDs
   (scored → seg_accuracy, 70% of final).
2. For every cell across BOTH the sim and real held-out inputs, emit a
   per-cell feature vector into `/output/embeddings.npz` with a per-row
   domain label ("sim" or "real"). A 5-fold LR classifier tries to separate
   sim from real in that feature space; the less-separable, the higher
   integration_score (30% of final).

Final = 0.7 * seg_accuracy + 0.3 * integration_score.

This baseline runs a tiny CNN per mask (exercises GPU) and uses the mean of
the CNN's feature map over each cell as that cell's embedding. IDs default to
identity — replace `predict_one` and/or `embed_one` with your trained model.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")
ATLAS_DIR = Path("/atlas")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def open_atlas() -> dict:
    """Open the 4D reference atlas lazily. Returns a dict of zarr arrays so
    only the slices you actually index get loaded into RAM."""
    import zarr
    root = zarr.open_group(str(ATLAS_DIR / "reference.ome.zarr"), mode="r")
    return {
        "labels":   root["labels"],    # (255, 214, 356, 256) int16
        "membrane": root["membrane"],  # (255, 214, 356, 256) uint8
        "nucleus":  root["nucleus"],   # (255,  92, 512, 712) uint8
    }


class RegionEncoder(nn.Module):
    """Tiny CNN: HxW mask -> (hidden, H, W) per-pixel features."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


def _encode(mask: np.ndarray, model: nn.Module) -> torch.Tensor:
    """Run the CNN on a single (H,W) int mask; return (hidden, H, W) features."""
    x = torch.from_numpy(mask.astype(np.float32))[None, None].to(DEVICE)
    with torch.no_grad():
        return model(x)[0]  # (hidden, H, W)


def predict_one(mask: np.ndarray, features: torch.Tensor) -> np.ndarray:
    """Return an HxW int32 mask with predicted atlas IDs.

    Baseline: identity — pass input labels through. Replace with your
    learned assignment using `features` + the 4D reference atlas you baked in.
    """
    return mask.astype(np.int32)


def embed_one(features: torch.Tensor, mask: np.ndarray, cell_id: int) -> np.ndarray:
    """Per-cell embedding = spatial mean of CNN features within that cell.

    Replace with however your model scores cell identity (e.g. a learned
    projection head).
    """
    ys, xs = np.where(mask == cell_id)
    if ys.size == 0:
        return np.zeros(features.shape[0], dtype=np.float32)
    # features: (hidden, H, W) torch on DEVICE
    vec = features[:, ys, xs].mean(dim=1)
    return vec.detach().cpu().numpy().astype(np.float32)


def _process(seg_path: Path, model: nn.Module) -> Tuple[np.ndarray, List[np.ndarray], List[int]]:
    seg = np.load(seg_path, allow_pickle=True).item()
    mask = seg["masks"]
    features = _encode(mask, model)
    pred = predict_one(mask, features)
    ids = sorted(int(x) for x in np.unique(mask) if x > 0)
    embs = [embed_one(features, mask, cid) for cid in ids]
    return pred, embs, ids


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = RegionEncoder().to(DEVICE).eval()

    atlas = open_atlas()
    print(f"[pytorch_baseline] atlas labels={atlas['labels'].shape} "
          f"dtype={atlas['labels'].dtype}", flush=True)

    sim_files = sorted(INPUT_DIR.glob("sample_*_seg.npy"))
    print(f"[pytorch_baseline] device={DEVICE} torch={torch.__version__} "
          f"sim_inputs={len(sim_files)}", flush=True)

    sim_embs: List[np.ndarray] = []
    for in_path in sim_files:
        pred, embs, _ = _process(in_path, model)
        np.save(OUTPUT_DIR / in_path.name, {
            "masks":    pred.astype(np.int32),
            "cell_ids": sorted(int(x) for x in np.unique(pred) if x > 0),
        })
        sim_embs.extend(embs)

    real_embs: List[np.ndarray] = []
    real_dir = INPUT_DIR / "real_manual"
    if real_dir.is_dir():
        real_files = sorted(real_dir.glob("*_seg.npy"))
        print(f"[pytorch_baseline] real_inputs={len(real_files)}", flush=True)
        for real_path in real_files:
            _, embs, _ = _process(real_path, model)
            real_embs.extend(embs)

    all_embs = np.vstack(sim_embs + real_embs).astype(np.float32)
    domain = np.array(["sim"] * len(sim_embs) + ["real"] * len(real_embs))
    np.savez(OUTPUT_DIR / "embeddings.npz", embeddings=all_embs, domain=domain)

    print(f"[pytorch_baseline] wrote {len(sim_files)} seg files + embeddings.npz "
          f"(sim={len(sim_embs)} real={len(real_embs)} D={all_embs.shape[1]})",
          flush=True)


if __name__ == "__main__":
    main()
