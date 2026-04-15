import numpy as np
import pytest

from scoring.registration import (
    compute_registration_accuracy,
    geodesic_rotation_error,
    translation_error,
)


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def test_geodesic_identity():
    R = np.eye(3)
    assert geodesic_rotation_error(R, R) == pytest.approx(0.0, abs=1e-9)


def test_geodesic_180_degrees():
    R_gt = np.eye(3)
    R_pred = _rot_x(np.pi)
    assert geodesic_rotation_error(R_pred, R_gt) == pytest.approx(1.0, abs=1e-6)


def test_geodesic_90_degrees():
    R_pred = _rot_x(np.pi / 2)
    assert geodesic_rotation_error(R_pred, np.eye(3)) == pytest.approx(0.5, abs=1e-6)


def test_geodesic_bad_shape():
    with pytest.raises(ValueError):
        geodesic_rotation_error(np.eye(4), np.eye(3))


def test_translation_zero():
    assert translation_error(np.zeros(3), np.zeros(3), 10.0) == 0.0


def test_translation_clipped():
    assert translation_error(np.array([100.0, 0, 0]), np.zeros(3), 1.0) == 1.0


def test_translation_scale():
    assert translation_error(np.array([5.0, 0, 0]), np.zeros(3), 10.0) == pytest.approx(0.5)


def test_registration_accuracy_perfect():
    gt = {"a": {"rotation": np.eye(3).tolist(), "translation": [0, 0, 0]}}
    pred = {"a": {"rotation": np.eye(3).tolist(), "translation": [0, 0, 0]}}
    acc, details = compute_registration_accuracy(pred, gt, volume_diameter=10.0)
    assert acc == pytest.approx(1.0)
    assert details["n"] == 1


def test_registration_accuracy_missing_file():
    gt = {"a": {"rotation": np.eye(3).tolist(), "translation": [0, 0, 0]}}
    acc, details = compute_registration_accuracy({}, gt, volume_diameter=10.0)
    assert acc == 0.0
    assert details["per_sample"]["a"]["missing"] is True


def test_registration_non_finite_gives_zero():
    gt = {"a": {"rotation": np.eye(3).tolist(), "translation": [0, 0, 0]}}
    pred = {"a": {"rotation": [[np.nan] * 3] * 3, "translation": [0, 0, 0]}}
    acc, _ = compute_registration_accuracy(pred, gt, volume_diameter=10.0)
    assert acc == 0.0


def test_registration_empty_gt():
    acc, details = compute_registration_accuracy({}, {}, volume_diameter=10.0)
    assert acc == 0.0
    assert details["n"] == 0
