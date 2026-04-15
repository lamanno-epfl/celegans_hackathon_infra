"""Unit tests for v2 scoring scaffolding."""
from __future__ import annotations

import pytest

from scoring.cell_naming import hungarian_name_accuracy, score_naming
from scoring.combined_v2 import V2Weights, compute_final_v2
from scoring.timepoint import timepoint_accuracy


def test_timepoint_exact_and_tolerance():
    pred = {"a": 5, "b": 6, "c": 4, "d": 10}
    gt = {"a": 5, "b": 5, "c": 5, "d": 5}  # only 'a' exact; 'b','c' off by 1
    out = timepoint_accuracy(pred, gt, sample_ids=["a", "b", "c", "d"], tolerance=1)
    assert out["exact"] == pytest.approx(0.25)
    assert out["within_1"] == pytest.approx(0.75)


def test_timepoint_missing_pred_counts_wrong():
    out = timepoint_accuracy({"a": 5}, {"a": 5, "b": 6}, sample_ids=["a", "b"])
    assert out["exact"] == pytest.approx(0.5)


def test_hungarian_perfect_match_swapped_ids():
    # Pred and gold use different instance IDs; matching should still recover.
    pred = {1: "ABala", 2: "MSpaap", 3: "Cppap"}
    gold = {7: "MSpaap", 8: "ABala", 9: "Cppap"}
    assert hungarian_name_accuracy(pred, gold) == 1.0


def test_hungarian_partial():
    pred = {1: "ABala", 2: "WRONG", 3: "Cppap"}
    gold = {1: "ABala", 2: "MSpaap", 3: "Cppap"}
    assert hungarian_name_accuracy(pred, gold) == pytest.approx(2 / 3)


def test_hungarian_empty_pred_zero():
    assert hungarian_name_accuracy({}, {1: "ABala"}) == 0.0


def test_score_naming_placeholder_when_gt_empty():
    out = score_naming({}, {}, sample_ids=["a"])
    assert out["score"] == 0.0
    assert "placeholder" in out["note"].lower()


def test_score_naming_real_values():
    preds = {"s1": {"1": "A", "2": "B"}, "s2": {"1": "X", "2": "Y"}}
    gt = {"s1": {1: "A", 2: "B"}, "s2": {1: "X", 2: "Z"}}
    out = score_naming(preds, gt, sample_ids=["s1", "s2"])
    assert out["score"] == pytest.approx((1.0 + 0.5) / 2)
    assert out["n_scored"] == 2


def test_combined_v2_below_threshold_collapses_to_registration():
    out = compute_final_v2(
        registration_score=0.20, timepoint_score=0.99,
        cell_naming_score=0.99, integration_score=0.99,
    )
    assert out["final_score"] == pytest.approx(0.20)
    assert "below_threshold" in out["formula_used"]


def test_combined_v2_weighted_combination():
    out = compute_final_v2(
        registration_score=0.50, timepoint_score=0.80,
        cell_naming_score=0.60, integration_score=0.40,
        weights=V2Weights(0.30, 0.20, 0.30, 0.20),
    )
    expected = 0.30*0.50 + 0.20*0.80 + 0.30*0.60 + 0.20*0.40
    assert out["final_score"] == pytest.approx(expected)


def test_combined_v2_placeholder_redistributes_weight():
    out = compute_final_v2(
        registration_score=0.50, timepoint_score=0.80,
        cell_naming_score=0.0, integration_score=0.40,
        weights=V2Weights(0.30, 0.20, 0.30, 0.20),
        cell_naming_is_placeholder=True,
    )
    # cell weight (0.30) redistributed proportionally over the other three (sum 0.70)
    eff_reg = 0.30 + 0.30 * (0.30/0.70)
    eff_tp  = 0.20 + 0.30 * (0.20/0.70)
    eff_int = 0.20 + 0.30 * (0.20/0.70)
    expected = eff_reg*0.50 + eff_tp*0.80 + 0.0 + eff_int*0.40
    assert out["final_score"] == pytest.approx(expected)
    assert "placeholder" in out["formula_used"]
