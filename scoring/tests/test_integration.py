import numpy as np
import pytest

from scoring.integration import (
    check_embedding_collapse,
    compute_integration_score,
    domain_classifier_accuracy,
)


def test_collapse_detection_constant():
    emb = np.ones((100, 32))
    assert check_embedding_collapse(emb) is True


def test_collapse_detection_normal():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(100, 32))
    assert check_embedding_collapse(emb) is False


def test_integration_score_collapse_is_zero():
    emb = np.zeros((50, 16))
    labels = np.array([0] * 25 + [1] * 25)
    score, details = compute_integration_score(emb, labels)
    assert score == 0.0
    assert details["reason"] == "embedding_collapse"


def test_integration_score_indistinguishable():
    rng = np.random.default_rng(1)
    # Both classes drawn from same distribution -> classifier near 0.5 -> score near 1.
    emb = rng.normal(size=(200, 16))
    labels = np.array([0] * 100 + [1] * 100)
    score, details = compute_integration_score(emb, labels)
    assert score > 0.6
    assert details["collapse_detected"] is False


def test_integration_score_separable():
    rng = np.random.default_rng(2)
    sim = rng.normal(loc=0.0, scale=0.1, size=(100, 16))
    real = rng.normal(loc=10.0, scale=0.1, size=(100, 16))
    emb = np.vstack([sim, real])
    labels = np.array([0] * 100 + [1] * 100)
    score, details = compute_integration_score(emb, labels)
    assert score < 0.1
    assert details["classifier_accuracy"] > 0.95


def test_integration_deterministic():
    rng = np.random.default_rng(3)
    emb = rng.normal(size=(100, 8))
    labels = np.array([0] * 50 + [1] * 50)
    s1, _ = compute_integration_score(emb, labels)
    s2, _ = compute_integration_score(emb, labels)
    assert s1 == s2


def test_integration_nan_gives_zero():
    emb = np.full((20, 4), np.nan)
    labels = np.array([0] * 10 + [1] * 10)
    score, details = compute_integration_score(emb, labels)
    assert score == 0.0
    assert details["reason"] == "non_finite_embeddings"


def test_classifier_single_class():
    emb = np.random.default_rng(4).normal(size=(10, 4))
    labels = np.zeros(10, dtype=int)
    acc = domain_classifier_accuracy(emb, labels)
    assert acc == 1.0
