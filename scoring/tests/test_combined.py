import numpy as np

from scoring.combined import compute_final_score


def _perfect_pose():
    return {"rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]}


def test_below_threshold_returns_registration_only():
    rng = np.random.default_rng(0)
    # Bad predictions.
    gt = {f"{i:04d}": _perfect_pose() for i in range(5)}
    bad_rot = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]  # 90 deg
    pred = {k: {"rotation": bad_rot, "translation": [100, 100, 100]} for k in gt}
    emb = rng.normal(size=(10, 32))
    labels = np.array([0] * 5 + [1] * 5)
    order = list(gt.keys()) + [f"r{i}" for i in range(5)]
    # Add "real" predictions so poses_pred has everything
    for k in order[5:]:
        pred[k] = _perfect_pose()

    final, details = compute_final_score(
        poses_pred=pred,
        poses_gt=gt,
        embeddings=emb,
        domain_labels=labels,
        volume_diameter=10.0,
        simulated_filenames=list(gt.keys()),
        filename_order=order,
    )
    assert details["formula_used"].startswith("registration_only")
    assert final == details["registration_accuracy"]


def test_above_threshold_weighted():
    rng = np.random.default_rng(1)
    gt = {f"{i:04d}": _perfect_pose() for i in range(5)}
    pred = {k: _perfect_pose() for k in gt}
    order = list(gt.keys()) + [f"r{i}" for i in range(5)]
    for k in order[5:]:
        pred[k] = _perfect_pose()
    emb = rng.normal(size=(10, 32))
    labels = np.array([0] * 5 + [1] * 5)
    final, details = compute_final_score(
        poses_pred=pred,
        poses_gt=gt,
        embeddings=emb,
        domain_labels=labels,
        volume_diameter=10.0,
        simulated_filenames=list(gt.keys()),
        filename_order=order,
    )
    assert details["formula_used"] == "weighted_combination"
    assert details["registration_accuracy"] == 1.0
    expected = 0.8 * 1.0 + 0.2 * details["integration_score"]
    assert abs(final - expected) < 1e-9
