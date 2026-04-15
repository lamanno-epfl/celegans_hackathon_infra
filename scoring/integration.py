"""Integration score via domain-classifier two-sample test."""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def check_embedding_collapse(embeddings: np.ndarray, threshold: float = 1e-6) -> bool:
    """Returns True if embeddings have collapsed to near-constant values."""
    if embeddings.size == 0:
        return True
    return bool(np.std(embeddings) < threshold)


def domain_classifier_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
) -> float:
    """Mean k-fold cross-validated accuracy of a logistic regression domain classifier."""
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    if embeddings.shape[0] != labels.shape[0]:
        raise ValueError("embeddings and labels must have same length")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2:
        # Only one class — not a meaningful test. Treat as perfect separation (worst).
        return 1.0
    n_folds = int(min(n_folds, counts.min()))
    if n_folds < 2:
        n_folds = 2

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    accuracies = []
    for train_idx, test_idx in skf.split(embeddings, labels):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(embeddings[train_idx])
        X_test = scaler.transform(embeddings[test_idx])
        y_train, y_test = labels[train_idx], labels[test_idx]
        clf = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, solver="lbfgs")
        clf.fit(X_train, y_train)
        accuracies.append(clf.score(X_test, y_test))
    return float(np.mean(accuracies))


def compute_integration_score(
    embeddings: np.ndarray,
    domain_labels: np.ndarray,
    n_folds: int = 5,
    collapse_threshold: float = 1e-6,
) -> Tuple[float, dict]:
    """Compute integration score; 1.0 = indistinguishable, 0.0 = perfectly separable."""
    embeddings = np.asarray(embeddings)
    if not np.all(np.isfinite(embeddings)):
        return 0.0, {"reason": "non_finite_embeddings", "classifier_accuracy": None}
    if check_embedding_collapse(embeddings, collapse_threshold):
        return 0.0, {"reason": "embedding_collapse", "classifier_accuracy": None}

    clf_acc = domain_classifier_accuracy(embeddings, domain_labels, n_folds=n_folds)
    score = 1.0 - 2.0 * abs(clf_acc - 0.5)
    score = float(max(0.0, min(1.0, score)))
    return score, {"classifier_accuracy": clf_acc, "collapse_detected": False}
