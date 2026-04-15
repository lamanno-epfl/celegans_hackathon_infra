"""Participant template for the seg-in / seg-out contract (v2, 2026-04-15).

Reads every `/input/*_seg.npy` (Cellpose-style dict with key "masks" = HxW int
array of atlas cell IDs, 0 = background) and writes a corresponding
`/output/<same_name>_seg.npy` with the SAME pixel regions but with the IDs your
model thinks are the canonical atlas cell IDs for each region.

This template is an IDENTITY baseline: it copies the input IDs through unchanged.
That scores 0% if the gold IDs use a different convention than the input — which
is exactly the point of the competition.

Replace `predict_ids(mask)` with your model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")


def predict_ids(mask: np.ndarray) -> np.ndarray:
    """Take an HxW int mask, return an HxW int mask with your predicted IDs.

    Baseline: pass input IDs through unchanged.
    """
    return mask.astype(np.int32, copy=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    seg_files = sorted(INPUT_DIR.glob("*_seg.npy"))
    print(f"[predict] {len(seg_files)} input seg files", flush=True)

    for in_path in seg_files:
        seg = np.load(in_path, allow_pickle=True).item()
        pred_mask = predict_ids(seg["masks"])
        out = {
            "masks": pred_mask.astype(np.int32),
            "cell_ids": sorted(int(x) for x in np.unique(pred_mask) if x > 0),
        }
        np.save(OUTPUT_DIR / in_path.name, out)

    print(f"[predict] wrote {len(seg_files)} files to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
