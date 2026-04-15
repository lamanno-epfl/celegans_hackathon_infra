#!/usr/bin/env python3
"""Score predicted _seg.npy against ground truth _seg.npy.

The predicted and GT masks share identical segmentation geometry (same pixel
regions). The model's task is to assign the correct atlas cell ID to each
region. Scoring simply checks whether each region received the right ID.

Scoring:
  1. For each GT cell region, look up the predicted cell ID in that region
  2. A cell is "correct" if pred_cell_id == gt_cell_id
  3. Accuracy = n_correct / n_gt_cells

Usage:
    # Score a single sample
    python hackathon/score_seg.py \
        --pred pred_seg/sample_0000_seg.npy \
        --gt gt_seg/sample_0000_seg.npy

    # Score all samples in a directory
    python hackathon/score_seg.py \
        --pred-dir pred_seg/ \
        --gt-dir gt_seg/

    # Score with detailed per-cell breakdown
    python hackathon/score_seg.py \
        --pred-dir pred_seg/ --gt-dir gt_seg/ --verbose
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def score_single(gt_seg: dict, pred_seg: dict) -> dict:
    """Score a single predicted seg against ground truth.

    Masks share the same geometry — each GT region maps to exactly one pred
    region. We check whether the assigned cell ID matches.

    Args:
        gt_seg: dict with 'masks' (H,W int32) and 'cell_ids'.
        pred_seg: dict with 'masks' (H,W int32) and 'cell_ids'.

    Returns:
        Dict with scoring metrics.
    """
    gt_mask = gt_seg["masks"]
    pred_mask = pred_seg["masks"]
    gt_ids = sorted(int(x) for x in np.unique(gt_mask) if x > 0)

    n_gt = len(gt_ids)
    if n_gt == 0:
        n_pred = len([x for x in np.unique(pred_mask) if x > 0])
        return {
            "n_gt": 0, "n_pred": n_pred,
            "n_correct": 0, "accuracy": 1.0 if n_pred == 0 else 0.0,
            "matches": [],
        }

    matches = []
    n_correct = 0

    for gid in gt_ids:
        region = gt_mask == gid
        # The pred mask covers the same region — grab the dominant pred ID
        pred_in_region = pred_mask[region]
        pred_ids_in_region, counts = np.unique(pred_in_region, return_counts=True)
        # Exclude background
        nonzero = pred_ids_in_region > 0
        if nonzero.any():
            pid = int(pred_ids_in_region[nonzero][counts[nonzero].argmax()])
        else:
            pid = 0

        correct = gid == pid
        if correct:
            n_correct += 1
        matches.append({"gt_id": gid, "pred_id": pid, "correct": correct})

    accuracy = n_correct / n_gt

    return {
        "n_gt": n_gt,
        "n_pred": len([x for x in np.unique(pred_mask) if x > 0]),
        "n_correct": n_correct,
        "accuracy": accuracy,
        "matches": matches,
    }


def main():
    parser = argparse.ArgumentParser(description="Score predicted _seg.npy vs ground truth")
    parser.add_argument("--pred", type=Path, help="Single predicted _seg.npy file")
    parser.add_argument("--gt", type=Path, help="Single ground truth _seg.npy file")
    parser.add_argument("--pred-dir", type=Path, help="Directory of predicted _seg.npy files")
    parser.add_argument("--gt-dir", type=Path, help="Directory of ground truth _seg.npy files")
    parser.add_argument("--verbose", action="store_true", help="Print per-cell breakdown")
    args = parser.parse_args()

    if args.pred and args.gt:
        # Single file mode
        pred_seg = np.load(args.pred, allow_pickle=True).item()
        gt_seg = np.load(args.gt, allow_pickle=True).item()
        result = score_single(gt_seg, pred_seg)

        print(f"GT cells: {result['n_gt']}, Pred cells: {result['n_pred']}")
        print(f"Correct IDs: {result['n_correct']} / {result['n_gt']}")
        print(f"Accuracy: {result['accuracy']:.1%}")

        if args.verbose and result["matches"]:
            print(f"\nPer-cell breakdown:")
            for m in result["matches"]:
                status = "OK" if m["correct"] else "WRONG"
                print(f"  GT {m['gt_id']:>4d} <-> Pred {m['pred_id']:>4d}  {status}")

    elif args.pred_dir and args.gt_dir:
        # Directory mode
        gt_files = sorted(args.gt_dir.glob("*_seg.npy"))
        if not gt_files:
            print(f"No _seg.npy files found in {args.gt_dir}")
            return

        all_results = []
        total_correct = 0
        total_gt = 0

        for gt_path in gt_files:
            pred_path = args.pred_dir / gt_path.name
            if not pred_path.exists():
                print(f"  MISSING: {pred_path.name}")
                continue

            pred_seg = np.load(pred_path, allow_pickle=True).item()
            gt_seg = np.load(gt_path, allow_pickle=True).item()
            result = score_single(gt_seg, pred_seg)
            all_results.append(result)
            total_correct += result["n_correct"]
            total_gt += result["n_gt"]

            if args.verbose:
                print(f"  {gt_path.stem}: {result['n_correct']}/{result['n_gt']} "
                      f"({result['accuracy']:.1%})")

        n = len(all_results)
        if n == 0:
            print("No matching files found")
            return

        accuracies = [r["accuracy"] for r in all_results]

        print(f"\n{'=' * 60}")
        print(f"SCORING SUMMARY ({n} samples)")
        print(f"{'=' * 60}")
        print(f"Overall accuracy:     {total_correct}/{total_gt} "
              f"({total_correct/total_gt:.1%})")
        print(f"Mean per-sample acc:  {np.mean(accuracies):.1%} "
              f"+/- {np.std(accuracies):.1%}")
        print(f"Samples >= 90% acc:   {sum(1 for a in accuracies if a >= 0.9)}/{n}")
        print(f"Samples >= 80% acc:   {sum(1 for a in accuracies if a >= 0.8)}/{n}")
        print(f"Samples >= 50% acc:   {sum(1 for a in accuracies if a >= 0.5)}/{n}")
        print(f"{'=' * 60}")
    else:
        parser.print_help()
        print("\nProvide either --pred/--gt (single file) or --pred-dir/--gt-dir (batch)")


if __name__ == "__main__":
    main()
