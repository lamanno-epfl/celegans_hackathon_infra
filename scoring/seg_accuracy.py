"""Seg-in / seg-out scoring (Xinyi's 2026-04-15 contract clarification).

Participant container reads `/input/<sample>_seg.npy`, writes `/output/<sample>_seg.npy`
with the SAME pixel regions but (potentially) remapped cell IDs. The target is a
gold `_seg.npy` carrying the canonical atlas ID for each region. Score = fraction
of regions whose predicted ID matches the gold ID, majority-voted within each
gold region.

This module wraps `scripts/score_seg.py::score_single` as an importable API so the
worker can call it directly without shelling out.

STATUS: unit-testable, but BLOCKED on Xinyi shipping the gold `*_seg.npy` files
(or the atlas-ID → canonical-ID lookup from which we can derive them). The
shipped `evaluation_annotation_SEALED/ground_truth.npz` has pose params only and
no per-pixel target. See `docs/TODO_PENDING.md` item X1 / X6.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

GT_MISSING_REASON = (
    "seg_accuracy is a placeholder (returns 0.0). Pending Xinyi's gold "
    "`ground_truth_masks/*.npz` (referenced in her own npz_to_seg.py docstring "
    "but not shipped). See docs/TODO_PENDING.md item X1."
)


def score_single(gt_seg: dict, pred_seg: dict) -> dict:
    """Score one predicted seg dict against its gold seg dict.

    Mirrors `scripts/score_seg.py::score_single` — kept in sync intentionally.
    """
    gt_mask = gt_seg["masks"]
    pred_mask = pred_seg["masks"]
    gt_ids = sorted(int(x) for x in np.unique(gt_mask) if x > 0)

    if not gt_ids:
        n_pred = len([x for x in np.unique(pred_mask) if x > 0])
        return {"n_gt": 0, "n_pred": n_pred, "n_correct": 0,
                "accuracy": 1.0 if n_pred == 0 else 0.0}

    n_correct = 0
    for gid in gt_ids:
        region = gt_mask == gid
        pred_in_region = pred_mask[region]
        ids, counts = np.unique(pred_in_region, return_counts=True)
        nonzero = ids > 0
        if nonzero.any():
            pid = int(ids[nonzero][counts[nonzero].argmax()])
        else:
            pid = 0
        if pid == gid:
            n_correct += 1

    return {"n_gt": len(gt_ids),
            "n_pred": len([x for x in np.unique(pred_mask) if x > 0]),
            "n_correct": n_correct,
            "accuracy": n_correct / len(gt_ids)}


def score_directory(pred_dir: Path, gt_dir: Path,
                    sample_names: Iterable[str] | None = None) -> dict:
    """Score every `<name>_seg.npy` in `gt_dir` against `pred_dir/<name>_seg.npy`.

    Returns `{score, n_scored, n_missing, per_sample}`. If `gt_dir` is empty or
    absent, returns the placeholder zero with a `note`.
    """
    gt_dir = Path(gt_dir)
    if not gt_dir.is_dir():
        return {"score": 0.0, "n_scored": 0, "note": GT_MISSING_REASON}
    gt_files = sorted(gt_dir.glob("*_seg.npy"))
    if not gt_files:
        return {"score": 0.0, "n_scored": 0, "note": GT_MISSING_REASON}

    total_correct = total_gt = n_missing = 0
    per_sample = {}
    for gt_path in gt_files:
        name = gt_path.stem
        if sample_names is not None and name not in set(sample_names):
            continue
        pred_path = Path(pred_dir) / gt_path.name
        if not pred_path.exists():
            n_missing += 1
            continue
        gt_seg = np.load(gt_path, allow_pickle=True).item()
        pred_seg = np.load(pred_path, allow_pickle=True).item()
        r = score_single(gt_seg, pred_seg)
        per_sample[name] = r["accuracy"]
        total_correct += r["n_correct"]
        total_gt += r["n_gt"]

    if total_gt == 0:
        return {"score": 0.0, "n_scored": 0, "n_missing": n_missing,
                "note": "no overlap between predictions and gt"}
    return {"score": total_correct / total_gt,
            "n_scored": len(per_sample),
            "n_missing": n_missing,
            "per_sample": per_sample}
