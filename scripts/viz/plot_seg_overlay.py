#!/usr/bin/env python3
"""Visualize evaluation: input mask + GT ref_mask + predicted mask, side by side,
with regions colored by correct/wrong so mis-identified cells jump out.

Usage
-----
    # Single sample:
    python scripts/viz/plot_seg_overlay.py \
        --input  data/real/held_out/evaluation_annotation_SEALED/masks/sample_0000.npz \
        --gt     data/real/held_out/evaluation_annotation_SEALED/ground_truth_masks/sample_0000.npz \
        --pred   /path/to/pred_dir/sample_0000_seg.npy \
        -o runtime/plots/sample_0000_overlay.png

    # Batch: use a predictions dir (produced by a baseline), pick N random
    # samples, write one PNG per sample:
    python scripts/viz/plot_seg_overlay.py \
        --eval-root data/real/held_out/evaluation_annotation_SEALED \
        --pred-dir  /tmp/out \
        --n 6 \
        -o runtime/plots/overlay_grid/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load_mask_from_npz(p: Path, key_candidates=("masks", "ref_mask")) -> np.ndarray:
    d = np.load(p, allow_pickle=True)
    if hasattr(d, "keys"):
        for k in key_candidates:
            if k in d.keys():
                return np.asarray(d[k])
    obj = d.item() if hasattr(d, "item") else d
    for k in key_candidates:
        if isinstance(obj, dict) and k in obj:
            return np.asarray(obj[k])
    raise KeyError(f"{p}: none of {key_candidates} present")


def _scramble(mask: np.ndarray, seed: int = 0) -> np.ndarray:
    """Remap instance IDs to random small ints for readable coloring.
    Background (0) is pinned to 0 so it renders identically across panels."""
    nz = np.unique(mask[mask > 0])
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(nz)) + 1  # reserve 0 for background
    lut = np.zeros(max(mask.max() + 1, 1), dtype=np.int32)
    for k, u in enumerate(nz):
        lut[u] = perm[k]
    return lut[mask]


def _per_region_correctness(gt: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Produce an HxW image where each GT region is shaded by correctness:
    0 = background, 1 = wrong, 2 = correct. Majority vote per GT region.
    """
    out = np.zeros_like(gt, dtype=np.uint8)
    gt_ids = [int(x) for x in np.unique(gt) if x > 0]
    n_ok = 0
    for gid in gt_ids:
        reg = gt == gid
        pred_vals = pred[reg]
        ids, counts = np.unique(pred_vals, return_counts=True)
        nonzero = ids > 0
        if nonzero.any():
            pid = int(ids[nonzero][counts[nonzero].argmax()])
        else:
            pid = 0
        ok = (pid == gid)
        out[reg] = 2 if ok else 1
        n_ok += int(ok)
    return out, n_ok, len(gt_ids)


def make_overlay_figure(inp: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                        title: str) -> plt.Figure:
    correctness, n_ok, n_gt = _per_region_correctness(gt, pred)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    cmap_inst = matplotlib.cm.get_cmap("nipy_spectral").copy()
    cmap_inst.set_under("#111111")
    kw = dict(cmap=cmap_inst, interpolation="nearest", vmin=0.5)
    axes[0].imshow(_scramble(inp, 1), **kw)
    axes[0].set_title(f"input (noised)\n{int((inp > 0).sum() > 0) * (len(np.unique(inp)) - 1)} regions")
    axes[1].imshow(_scramble(gt, 2), **kw)
    axes[1].set_title(f"gold ref_mask\n{n_gt} regions")
    axes[2].imshow(_scramble(pred, 3), **kw)
    axes[2].set_title(f"prediction\n{len(np.unique(pred)) - 1} regions")
    cmap = matplotlib.colors.ListedColormap(["#222222", "#d94141", "#46c46b"])
    axes[3].imshow(correctness, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    axes[3].set_title(f"correctness\n{n_ok}/{n_gt} ({n_ok / max(n_gt, 1):.1%})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return fig


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, help="single-sample input mask .npz")
    ap.add_argument("--gt", type=Path, help="single-sample ground_truth_masks .npz")
    ap.add_argument("--pred", type=Path, help="single-sample prediction _seg.npy")
    ap.add_argument("--eval-root", type=Path,
                    help="evaluation_annotation_SEALED root (for batch mode)")
    ap.add_argument("--pred-dir", type=Path, help="dir of prediction _seg.npy (batch mode)")
    ap.add_argument("--n", type=int, default=6, help="batch mode: how many samples")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    if args.input and args.gt and args.pred:
        inp = _load_mask_from_npz(args.input, ("masks",))
        gt = _load_mask_from_npz(args.gt, ("ref_mask", "masks"))
        pred = np.load(args.pred, allow_pickle=True).item()["masks"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig = make_overlay_figure(inp, gt, pred, title=args.input.stem)
        fig.savefig(args.output, dpi=100)
        print("wrote", args.output)
        return

    if not (args.eval_root and args.pred_dir):
        ap.error("provide either --input/--gt/--pred or --eval-root + --pred-dir")

    masks_dir = args.eval_root / "masks"
    gt_dir = args.eval_root / "ground_truth_masks"
    all_stems = sorted({p.stem for p in masks_dir.glob("sample_*.npz")}
                       & {p.stem for p in gt_dir.glob("sample_*.npz")})
    # Also filter to samples with a prediction
    all_stems = [s for s in all_stems if (args.pred_dir / f"{s}_seg.npy").exists()]
    if not all_stems:
        raise SystemExit(f"no overlapping samples between {masks_dir}, {gt_dir}, {args.pred_dir}")

    rng = np.random.default_rng(0)
    picked = rng.choice(all_stems, size=min(args.n, len(all_stems)), replace=False)
    args.output.mkdir(parents=True, exist_ok=True)
    for stem in picked:
        inp = _load_mask_from_npz(masks_dir / f"{stem}.npz", ("masks",))
        gt = _load_mask_from_npz(gt_dir / f"{stem}.npz", ("ref_mask", "masks"))
        pred = np.load(args.pred_dir / f"{stem}_seg.npy", allow_pickle=True).item()["masks"]
        fig = make_overlay_figure(inp, gt, pred, title=stem)
        out = args.output / f"{stem}_overlay.png"
        fig.savefig(out, dpi=100)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    _main()
