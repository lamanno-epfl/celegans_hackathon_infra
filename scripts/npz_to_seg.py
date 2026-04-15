#!/usr/bin/env python3
"""Convert masks from npz to _seg.npy format.

Output _seg.npy contains:
  - masks: (554, 554) int32 — cell instance labels (atlas integer IDs)
  - cell_ids: list of int — unique non-zero cell IDs in the mask

Cell IDs are the original atlas labels (e.g. 12, 128, 500), NOT sequential.
This preserves cell identity for matching against the reference.

Usage:
    # From train.npz (bulk, e.g. first 100)
    python hackathon/npz_to_seg.py hackathon/02_slice_database/train.npz \
        -o hackathon/02_slice_database/seg_files/ --max-samples 100

    # From eval masks directory
    python hackathon/npz_to_seg.py hackathon/evaluation_annotation_SEALED/masks/ \
        -o hackathon/evaluation_annotation_SEALED/seg_files/

    # From ground truth masks (ref_mask key)
    python hackathon/npz_to_seg.py hackathon/evaluation_annotation_SEALED/ground_truth_masks/ \
        -o hackathon/evaluation_annotation_SEALED/gt_seg_files/ --key ref_mask
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def mask_to_seg(mask: np.ndarray) -> dict:
    """Convert a 2D label mask to _seg.npy dict.

    Args:
        mask: (H, W) integer label array. 0 = background, >0 = cell IDs.

    Returns:
        Dict with 'masks' and 'cell_ids'.
    """
    mask = mask.astype(np.int32)
    cell_ids = sorted(int(x) for x in np.unique(mask) if x > 0)
    return {
        "masks": mask,
        "cell_ids": cell_ids,
    }


def convert_stacked_npz(
    npz_path: Path, out_dir: Path, mask_key: str, max_samples: int | None,
):
    """Convert a stacked npz to individual _seg.npy files."""
    print(f"Loading {npz_path.name}...", flush=True)
    data = np.load(npz_path, allow_pickle=True)
    masks = data[mask_key]
    n = min(masks.shape[0], max_samples) if max_samples else masks.shape[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Converting {n} masks -> {out_dir}/", flush=True)

    for i in range(n):
        seg = mask_to_seg(masks[i])
        np.save(out_dir / f"sample_{i:04d}_seg.npy", seg)
        if (i + 1) % 500 == 0 or i == n - 1:
            print(f"  {i + 1}/{n} done", flush=True)


def convert_individual_npz(masks_dir: Path, out_dir: Path, mask_key: str):
    """Convert individual sample_XXXX.npz files to _seg.npy files."""
    npz_files = sorted(masks_dir.glob("sample_*.npz"))
    n = len(npz_files)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Converting {n} mask files -> {out_dir}/", flush=True)

    for i, f in enumerate(npz_files):
        data = np.load(f, allow_pickle=True)
        seg = mask_to_seg(data[mask_key])
        np.save(out_dir / f"{f.stem}_seg.npy", seg)
        if (i + 1) % 100 == 0 or i == n - 1:
            print(f"  {i + 1}/{n} done", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Convert masks npz -> _seg.npy")
    parser.add_argument("input", type=Path,
                        help="Stacked .npz file or directory of sample_XXXX.npz")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="Output directory for _seg.npy files")
    parser.add_argument("--key", type=str, default="masks",
                        help="Mask key in npz (default: 'masks', use 'ref_mask' for GT)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples (stacked npz only)")
    args = parser.parse_args()

    if args.input.is_file():
        convert_stacked_npz(args.input, args.output, args.key, args.max_samples)
    elif args.input.is_dir():
        convert_individual_npz(args.input, args.output, args.key)
    else:
        print(f"Error: {args.input} not found")
        return
    print(f"Done! Output: {args.output}/")


if __name__ == "__main__":
    main()
