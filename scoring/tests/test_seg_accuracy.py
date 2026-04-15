"""Tests for seg-in / seg-out scoring."""
from __future__ import annotations

import numpy as np
import pytest

from scoring.seg_accuracy import score_single, score_directory, GT_MISSING_REASON


def _seg(mask):
    return {"masks": np.asarray(mask, dtype=np.int32),
            "cell_ids": sorted(int(x) for x in np.unique(mask) if x > 0)}


def test_score_single_perfect():
    m = np.zeros((8, 8), dtype=np.int32)
    m[1:3, 1:3] = 42
    m[5:7, 5:7] = 99
    r = score_single(_seg(m), _seg(m))
    assert r["accuracy"] == 1.0
    assert r["n_gt"] == 2 and r["n_correct"] == 2


def test_score_single_one_wrong():
    gt = np.zeros((8, 8), dtype=np.int32); gt[1:3, 1:3] = 42; gt[5:7, 5:7] = 99
    pred = gt.copy(); pred[pred == 99] = 100  # wrong id for the second region
    r = score_single(_seg(gt), _seg(pred))
    assert r["n_correct"] == 1
    assert r["accuracy"] == pytest.approx(0.5)


def test_score_single_majority_vote():
    gt = np.zeros((4, 4), dtype=np.int32); gt[:, :] = 7
    pred = np.zeros((4, 4), dtype=np.int32)
    pred[:, :3] = 7  # 12 pixels right
    pred[:, 3] = 8   # 4 pixels wrong
    r = score_single(_seg(gt), _seg(pred))
    assert r["n_correct"] == 1  # majority wins


def test_score_single_empty_gt():
    empty = np.zeros((4, 4), dtype=np.int32)
    r = score_single(_seg(empty), _seg(empty))
    assert r["accuracy"] == 1.0 and r["n_gt"] == 0


def test_score_directory_placeholder_when_missing(tmp_path):
    out = score_directory(tmp_path / "nope", tmp_path / "also_nope")
    assert out["score"] == 0.0
    assert GT_MISSING_REASON == out["note"]


def test_score_directory_end_to_end(tmp_path):
    pred_dir = tmp_path / "pred"; pred_dir.mkdir()
    gt_dir = tmp_path / "gt"; gt_dir.mkdir()
    gt = np.zeros((6, 6), dtype=np.int32); gt[1:3, 1:3] = 10; gt[4:6, 4:6] = 20
    pred = gt.copy()
    np.save(gt_dir / "sample_0000_seg.npy", _seg(gt))
    np.save(pred_dir / "sample_0000_seg.npy", _seg(pred))
    out = score_directory(pred_dir, gt_dir)
    assert out["score"] == 1.0 and out["n_scored"] == 1
