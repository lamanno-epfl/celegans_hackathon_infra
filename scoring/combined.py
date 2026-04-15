"""Combined final-score computation."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from config import CONFIG
from .integration import compute_integration_score
from .registration import compute_registration_accuracy


def compute_final_score(
    poses_pred: dict,
    poses_gt: dict,
    embeddings: np.ndarray,
    domain_labels: np.ndarray,
    volume_diameter: float,
    simulated_filenames: list,
    filename_order: list,
) -> Tuple[float, dict]:
    """Main scoring entry point; see startingprompt.md §6.3."""
    sc = CONFIG.scoring
    # Registration only over simulated held-out.
    sim_preds = {f: poses_pred[f] for f in simulated_filenames if f in poses_pred}
    reg_accuracy, reg_details = compute_registration_accuracy(sim_preds, poses_gt, volume_diameter)

    # Integration on all embeddings (rows matched to filename_order).
    if embeddings.shape[0] != len(filename_order):
        raise ValueError(
            f"embedding rows {embeddings.shape[0]} != filename_order length {len(filename_order)}"
        )
    int_score, int_details = compute_integration_score(
        embeddings,
        np.asarray(domain_labels),
        n_folds=sc.integration_k_folds,
        collapse_threshold=sc.collapse_std_threshold,
    )

    if reg_accuracy < sc.registration_threshold:
        final = reg_accuracy
        formula_used = "registration_only (below threshold)"
    else:
        final = sc.registration_weight * reg_accuracy + sc.integration_weight * int_score
        formula_used = "weighted_combination"

    return float(final), {
        "final_score": float(final),
        "registration_accuracy": float(reg_accuracy),
        "integration_score": float(int_score),
        "formula_used": formula_used,
        "threshold": sc.registration_threshold,
        "weights": {
            "registration": sc.registration_weight,
            "integration": sc.integration_weight,
        },
        "registration_details": reg_details,
        "integration_details": int_details,
    }
