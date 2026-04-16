"""Participant template: seg-in / seg-out + domain-adaptation embeddings.

The server mounts TWO held-out sets into `/input/` at eval time — you don't
see either locally:

  - `/input/sample_*_seg.npy`          simulated held-out (857 samples)
  - `/input/real_manual/*_seg.npy`     REAL held-out embryos (manually
                                       annotated; ground-truth real data)

Your container must, in ONE run, do both of the following:

1. For every sim sample, emit `/output/<same_name>` with predicted atlas IDs
   (scored against gold atlas masks → seg_accuracy, 70% of final).
2. For every cell across BOTH the sim and real held-out inputs, emit a
   per-cell feature vector into `/output/embeddings.npz` with a per-row
   domain label ("sim" or "real"). A 5-fold LR classifier tries to tell
   sim from real in your feature space; the better it FAILS (i.e. the more
   your sim and real embeddings mix), the higher integration_score, which
   is 30% of the final.

Final = 0.7 * seg_accuracy + 0.3 * integration_score.

This template is IDENTITY for seg (~0 seg accuracy) and uses 3-D shape
features (centroid_y, centroid_x, area) for the embedding. Replace
`predict_ids` with your classifier and `cell_embedding` with your learned
feature extractor.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")


def predict_ids(mask: np.ndarray) -> np.ndarray:
    """HxW int mask in -> HxW int mask out with predicted canonical atlas IDs.

    Baseline: pass input labels through unchanged (will score ~0).
    """
    return mask.astype(np.int32, copy=True)


def cell_embedding(mask: np.ndarray, cell_id: int) -> np.ndarray:
    """Return a D-length float32 feature vector for one cell in `mask`.

    Baseline: (centroid_y_norm, centroid_x_norm, area_norm). Replace with a
    learned embedding from your model for a higher integration score.
    """
    yx = np.argwhere(mask == cell_id)
    if yx.size == 0:
        return np.zeros(3, dtype=np.float32)
    h, w = mask.shape
    cy = yx[:, 0].mean() / h
    cx = yx[:, 1].mean() / w
    area = yx.shape[0] / (h * w)
    return np.array([cy, cx, area], dtype=np.float32)


def _iter_cells(seg_path: Path) -> Tuple[np.ndarray, List[int]]:
    seg = np.load(seg_path, allow_pickle=True).item()
    m = seg["masks"]
    ids = sorted(int(x) for x in np.unique(m) if x > 0)
    return m, ids


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Per-sample seg output + gather sim embeddings ---
    sim_seg_files = sorted(INPUT_DIR.glob("sample_*_seg.npy"))
    sim_embs: List[np.ndarray] = []
    for in_path in sim_seg_files:
        mask, ids = _iter_cells(in_path)
        pred_mask = predict_ids(mask)
        np.save(OUTPUT_DIR / in_path.name, {
            "masks":    pred_mask.astype(np.int32),
            "cell_ids": sorted(int(x) for x in np.unique(pred_mask) if x > 0),
        })
        for cid in ids:
            sim_embs.append(cell_embedding(mask, cid))

    # --- 2. Gather real embeddings (domain-adaptation reference) ---
    real_dir = INPUT_DIR / "real_manual"
    real_embs: List[np.ndarray] = []
    if real_dir.is_dir():
        for real_path in sorted(real_dir.glob("*_seg.npy")):
            mask, ids = _iter_cells(real_path)
            for cid in ids:
                real_embs.append(cell_embedding(mask, cid))

    # --- 3. Save embeddings in the required shape ---
    all_embs = np.vstack(sim_embs + real_embs).astype(np.float32)
    domain = np.array(["sim"] * len(sim_embs) + ["real"] * len(real_embs))
    np.savez(OUTPUT_DIR / "embeddings.npz", embeddings=all_embs, domain=domain)

    print(f"[predict] wrote {len(sim_seg_files)} seg files + embeddings.npz "
          f"(sim={len(sim_embs)} real={len(real_embs)} D={all_embs.shape[1]})",
          flush=True)


if __name__ == "__main__":
    main()
