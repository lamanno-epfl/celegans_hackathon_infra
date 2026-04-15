"""Sulston cell-name accuracy (v2 contract) — placeholder.

The participant's `cell_predictions.json` maps `{sample_id: {instance_id: name}}`.
The gold `name_lookup(timepoint, center, angles, u_value, mask) -> {instance_id: name}`
is **not yet available** — it must come from Xinyi (atlas + cutting-plane → Sulston
names per labeled blob). Until it lands, `score_naming` returns 0.0 with a clear
TODO marker so the rest of the pipeline can run.

When the snippet arrives, replace `_PLACEHOLDER_LOOKUP` with the real function and
the metric below ("fraction of cells correctly named, Hungarian-matched per sample,
averaged across samples") works as-is.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np

PLACEHOLDER_REASON = (
    "cell_naming_score is a placeholder (returns 0.0). Pending Xinyi's "
    "atlas mapping snippet — see docs/TODO_PENDING.md item X1."
)

# Type alias: gold name lookup signature.
NameLookupFn = Callable[[int, np.ndarray, np.ndarray, float, np.ndarray], Dict[int, str]]


def _placeholder_lookup(*args, **kwargs) -> Dict[int, str]:
    raise NotImplementedError(PLACEHOLDER_REASON)


_PLACEHOLDER_LOOKUP: NameLookupFn = _placeholder_lookup


def hungarian_name_accuracy(pred: Dict[int, str], gold: Dict[int, str]) -> float:
    """Best one-to-one matching of predicted instance names to gold instance names.

    With instance IDs being arbitrary labels, the cost matrix is `1 − exact_match`
    over all (pred_instance, gold_instance) pairs. We do not weight by mask size;
    every gold cell counts equally.
    """
    if not gold:
        return 0.0
    pred_ids = list(pred.keys())
    gold_ids = list(gold.keys())
    if not pred_ids:
        return 0.0
    cost = np.ones((len(pred_ids), len(gold_ids)), dtype=np.float32)
    for i, pi in enumerate(pred_ids):
        for j, gj in enumerate(gold_ids):
            if pred[pi] == gold[gj]:
                cost[i, j] = 0.0
    # SciPy is already a dependency via sklearn; use its Hungarian.
    from scipy.optimize import linear_sum_assignment

    row_idx, col_idx = linear_sum_assignment(cost)
    matched = int((cost[row_idx, col_idx] == 0).sum())
    return matched / len(gold_ids)


def score_naming(
    predictions: Dict[str, Dict[str, str]],
    gt_by_sample: Dict[str, Dict[int, str]],
    sample_ids: Iterable[str],
    name_lookup: Optional[NameLookupFn] = None,
) -> Dict[str, float]:
    """Mean Hungarian-matched cell-name accuracy across `sample_ids`.

    `gt_by_sample[sid]` is the gold `{instance_id_int: sulston_name_str}` for
    sample `sid`. If `gt_by_sample` is empty (because the lookup function isn't
    available yet) we return the placeholder zero with a `note`.

    `name_lookup` is reserved for the future when we'll derive `gt_by_sample`
    from the raw `(timepoint, center, angles, u, mask)` rows in `ground_truth.npz`.
    Pass `None` to use the placeholder behavior.
    """
    if not gt_by_sample or name_lookup is _PLACEHOLDER_LOOKUP:
        return {"score": 0.0, "n_scored": 0, "note": PLACEHOLDER_REASON}

    per_sample = []
    for sid in sample_ids:
        if sid not in gt_by_sample:
            continue
        gold = gt_by_sample[sid]
        raw_pred = predictions.get(sid, {})
        pred = {}
        for k, v in raw_pred.items():
            try:
                pred[int(k)] = str(v)
            except (TypeError, ValueError):
                continue
        per_sample.append(hungarian_name_accuracy(pred, gold))
    if not per_sample:
        return {"score": 0.0, "n_scored": 0, "note": "no overlap between predictions and gt"}
    return {"score": float(np.mean(per_sample)), "n_scored": len(per_sample)}
