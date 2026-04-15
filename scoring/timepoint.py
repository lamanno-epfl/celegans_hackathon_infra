"""Timepoint prediction scoring (v2 contract).

Each held-out sample has a gold integer timepoint. Participants predict an int
per sample. We report exact accuracy and an optional within-tolerance accuracy
(useful when the model is one bin off on a coarse-grained dataset).
"""
from __future__ import annotations

from typing import Dict, Iterable


def timepoint_accuracy(
    pred: Dict[str, int],
    gt: Dict[str, int],
    sample_ids: Iterable[str],
    tolerance: int = 0,
) -> Dict[str, float]:
    """Return {'exact': ..., 'within_<tol>': ...} fractions over sample_ids.

    Missing predictions count as wrong. Extra predictions are ignored.
    """
    exact = 0
    within = 0
    n = 0
    for sid in sample_ids:
        n += 1
        if sid not in gt:
            continue
        g = int(gt[sid])
        if sid not in pred:
            continue
        try:
            p = int(pred[sid])
        except (TypeError, ValueError):
            continue
        if p == g:
            exact += 1
        if abs(p - g) <= tolerance:
            within += 1
    if n == 0:
        return {"exact": 0.0, f"within_{tolerance}": 0.0}
    return {
        "exact": exact / n,
        f"within_{tolerance}": within / n,
    }
