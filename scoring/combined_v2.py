"""v2 combined scorer (registration + timepoint + cell-naming + integration).

Not yet wired into the worker — the worker still calls v1's `combined.compute_final_score`.
This module exists so we can unit-test the math now and switch the worker over the
moment Xinyi's atlas mapping snippet (TODO X1) lands.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class V2Weights:
    registration: float = 0.30
    timepoint: float = 0.20
    cell_naming: float = 0.30
    integration: float = 0.20

    def normalize(self) -> "V2Weights":
        s = self.registration + self.timepoint + self.cell_naming + self.integration
        if s <= 0:
            raise ValueError("weights must sum to > 0")
        return V2Weights(
            self.registration / s,
            self.timepoint / s,
            self.cell_naming / s,
            self.integration / s,
        )


def compute_final_v2(
    *,
    registration_score: float,
    timepoint_score: float,
    cell_naming_score: float,
    integration_score: float,
    registration_threshold: float = 0.30,
    weights: Optional[V2Weights] = None,
    cell_naming_is_placeholder: bool = False,
) -> dict:
    """Mirror of v1's threshold-gated weighted combination, extended to four
    components.

    If `registration_score < registration_threshold`, the model is too weak to
    fairly weight cell naming or integration — final collapses to registration.

    If `cell_naming_is_placeholder` (Xinyi snippet not in yet), redistribute its
    weight onto the other three so the placeholder zero doesn't drag final down
    artificially during the bring-up phase.
    """
    w = (weights or V2Weights()).normalize()

    if registration_score < registration_threshold:
        return {
            "final_score": registration_score,
            "formula_used": "registration_only_below_threshold",
            "threshold": registration_threshold,
            "components": {
                "registration": registration_score,
                "timepoint": timepoint_score,
                "cell_naming": cell_naming_score,
                "integration": integration_score,
            },
            "weights": w.__dict__,
        }

    if cell_naming_is_placeholder:
        leftover = w.cell_naming
        denom = w.registration + w.timepoint + w.integration
        if denom <= 0:
            denom = 1.0
        eff = V2Weights(
            registration=w.registration + leftover * (w.registration / denom),
            timepoint=w.timepoint + leftover * (w.timepoint / denom),
            cell_naming=0.0,
            integration=w.integration + leftover * (w.integration / denom),
        )
    else:
        eff = w

    final = (
        eff.registration * registration_score
        + eff.timepoint * timepoint_score
        + eff.cell_naming * cell_naming_score
        + eff.integration * integration_score
    )
    return {
        "final_score": float(final),
        "formula_used": "weighted_combination_v2"
        + ("_cell_naming_placeholder" if cell_naming_is_placeholder else ""),
        "threshold": registration_threshold,
        "components": {
            "registration": registration_score,
            "timepoint": timepoint_score,
            "cell_naming": cell_naming_score,
            "integration": integration_score,
        },
        "weights": eff.__dict__,
    }
