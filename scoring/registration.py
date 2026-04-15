"""Registration accuracy: pose comparison on simulated held-out slices."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def geodesic_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic distance between two rotation matrices, normalized to [0, 1].

    0 = identical, 1 = 180 degrees apart.
    """
    R_pred = np.asarray(R_pred, dtype=np.float64)
    R_gt = np.asarray(R_gt, dtype=np.float64)
    if R_pred.shape != (3, 3) or R_gt.shape != (3, 3):
        raise ValueError(f"Rotation matrices must be 3x3, got {R_pred.shape} and {R_gt.shape}")
    R_diff = R_pred @ R_gt.T
    trace_val = np.trace(R_diff)
    cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return float(angle / np.pi)


def translation_error(t_pred: np.ndarray, t_gt: np.ndarray, volume_diameter: float) -> float:
    """Euclidean translation error, normalized by volume diameter to [0, 1]."""
    t_pred = np.asarray(t_pred, dtype=np.float64).reshape(-1)
    t_gt = np.asarray(t_gt, dtype=np.float64).reshape(-1)
    if t_pred.shape != t_gt.shape:
        raise ValueError(f"Translation shape mismatch: {t_pred.shape} vs {t_gt.shape}")
    if volume_diameter <= 0:
        raise ValueError("volume_diameter must be positive")
    return float(np.clip(np.linalg.norm(t_pred - t_gt) / volume_diameter, 0.0, 1.0))


def _validate_rotation(R: np.ndarray, atol: float = 1e-3) -> bool:
    if R.shape != (3, 3):
        return False
    if not np.allclose(R @ R.T, np.eye(3), atol=atol):
        return False
    if not np.isclose(np.linalg.det(R), 1.0, atol=atol):
        return False
    return True


def compute_registration_accuracy(
    predictions: Dict[str, dict],
    ground_truth: Dict[str, dict],
    volume_diameter: float,
) -> Tuple[float, dict]:
    """Compute mean registration accuracy over ground-truth filenames.

    Returns (accuracy, per-sample detail dict).
    """
    if not ground_truth:
        return 0.0, {"per_sample": {}, "n": 0}

    per_sample: Dict[str, dict] = {}
    scores = []
    for fname, gt in ground_truth.items():
        if fname not in predictions:
            per_sample[fname] = {"score": 0.0, "missing": True}
            scores.append(0.0)
            continue
        pred = predictions[fname]
        try:
            R_pred = np.array(pred["rotation"], dtype=np.float64)
            R_gt = np.array(gt["rotation"], dtype=np.float64)
            t_pred = np.array(pred["translation"], dtype=np.float64)
            t_gt = np.array(gt["translation"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            per_sample[fname] = {"score": 0.0, "error": str(exc)}
            scores.append(0.0)
            continue

        if np.any(~np.isfinite(R_pred)) or np.any(~np.isfinite(t_pred)):
            per_sample[fname] = {"score": 0.0, "error": "non-finite prediction"}
            scores.append(0.0)
            continue

        rot_err = geodesic_rotation_error(R_pred, R_gt)
        trans_err = translation_error(t_pred, t_gt, volume_diameter)
        score = 0.5 * (1.0 - rot_err) + 0.5 * (1.0 - trans_err)
        per_sample[fname] = {
            "score": float(score),
            "rotation_error": float(rot_err),
            "translation_error": float(trans_err),
        }
        scores.append(score)

    accuracy = float(np.mean(scores))
    return accuracy, {"per_sample": per_sample, "n": len(scores)}
