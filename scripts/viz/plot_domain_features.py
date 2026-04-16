#!/usr/bin/env python3
"""Eyeball the simulated-vs-real domain gap.

Extracts per-sample shape features from segmentation masks (no raw images
needed), z-scores, PCA-reduces to 20 components, then UMAPs to 2D. Colors points
by domain (simulated = noised eval masks, real = manual segs). If the two
clouds separate linearly, domain adaptation has work to do; if they already
mix, DA is a no-op for this feature set (but a *pixel-space* DA signal could
still be present — this script is an upper-bound sanity check).

Usage
-----
    python scripts/viz/plot_domain_features.py \
        --sim-masks-dir data/real/held_out/evaluation_annotation_SEALED/masks \
        --real-segs-dir data/real/held_out/05_manual_segmentation \
        --n-sim 200 \
        -o runtime/plots/domain_umap.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURE_NAMES = [
    "n_cells", "mean_area", "std_area", "median_area",
    "min_area", "max_area", "area_p25", "area_p75",
    "total_coverage", "centroid_x_mean", "centroid_y_mean",
    "centroid_x_std", "centroid_y_std", "aspect_mean",
    "compactness_mean", "nearest_neighbor_mean",
]


def _mask_features(mask: np.ndarray) -> np.ndarray:
    ids = [int(x) for x in np.unique(mask) if x > 0]
    if not ids:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float64)

    areas, centroids, aspects, compactness = [], [], [], []
    for cid in ids:
        reg = mask == cid
        ys, xs = np.where(reg)
        a = ys.size
        areas.append(a)
        cy, cx = ys.mean(), xs.mean()
        centroids.append((cy, cx))
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        aspects.append(h / max(w, 1))
        # crude compactness: area / bbox_area
        compactness.append(a / max(h * w, 1))
    areas = np.array(areas, dtype=np.float64)
    centroids = np.array(centroids, dtype=np.float64)

    if len(ids) >= 2:
        d = np.sqrt(((centroids[:, None, :] - centroids[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        nn_mean = float(d.min(axis=1).mean())
    else:
        nn_mean = 0.0

    return np.array([
        len(ids), areas.mean(), areas.std(), np.median(areas),
        areas.min(), areas.max(), np.percentile(areas, 25),
        np.percentile(areas, 75),
        float((mask > 0).sum()) / mask.size,
        centroids[:, 1].mean(), centroids[:, 0].mean(),
        centroids[:, 1].std(), centroids[:, 0].std(),
        float(np.mean(aspects)), float(np.mean(compactness)),
        nn_mean,
    ], dtype=np.float64)


def _iter_sim(masks_dir: Path, n: int) -> Iterable[np.ndarray]:
    files = sorted(masks_dir.glob("sample_*.npz"))
    if n > 0:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(files), size=min(n, len(files)), replace=False)
        files = [files[i] for i in idx]
    for f in files:
        d = np.load(f)
        yield np.asarray(d["masks"])


def _iter_real(segs_dir: Path) -> Iterable[np.ndarray]:
    for f in sorted(segs_dir.glob("*_seg.npy")):
        d = np.load(f, allow_pickle=True).item()
        if isinstance(d, dict) and "masks" in d:
            yield np.asarray(d["masks"])


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-masks-dir", type=Path, required=True)
    ap.add_argument("--real-segs-dir", type=Path, required=True)
    ap.add_argument("--n-sim", type=int, default=200,
                    help="how many sim masks to sample (real set is always used fully)")
    ap.add_argument("-o", "--output", type=Path, required=True)
    args = ap.parse_args()

    print("extracting sim features...", flush=True)
    sim = np.array([_mask_features(m) for m in _iter_sim(args.sim_masks_dir, args.n_sim)])
    print(f"  sim: {sim.shape}")
    print("extracting real features...", flush=True)
    real = np.array([_mask_features(m) for m in _iter_real(args.real_segs_dir)])
    print(f"  real: {real.shape}")
    if len(real) < 2:
        raise SystemExit(f"need >= 2 real segs; got {len(real)}")

    X = np.vstack([sim, real])
    labels = np.array([0] * len(sim) + [1] * len(real))

    # Z-score
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)

    # PCA to min(k, n_samples, n_features)
    from sklearn.decomposition import PCA
    k = min(20, X.shape[0], X.shape[1])
    pca = PCA(n_components=k, random_state=0)
    Xp = pca.fit_transform(X)
    var = pca.explained_variance_ratio_.sum()

    # UMAP if available, else plain 2-D PCA
    method = "pca(2)"
    try:
        import umap  # type: ignore
        red = umap.UMAP(n_components=2, random_state=0,
                        n_neighbors=min(15, len(X) - 1)).fit_transform(Xp)
        method = "umap"
    except Exception as exc:
        print(f"  umap unavailable ({exc}); falling back to 2D PCA")
        red = Xp[:, :2]

    # Classifier sanity check: logistic regression accuracy
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    clf = LogisticRegression(max_iter=1000)
    scores = cross_val_score(clf, Xp, labels, cv=min(5, len(real)))
    cls_acc = float(scores.mean())
    integration_score = 1.0 - 2.0 * abs(cls_acc - 0.5)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(red[labels == 0, 0], red[labels == 0, 1],
               s=12, alpha=0.6, c="#4c72b0", label=f"sim (n={len(sim)})")
    ax.scatter(red[labels == 1, 0], red[labels == 1, 1],
               s=90, alpha=0.95, c="#c44e52", marker="X",
               edgecolors="black", linewidths=0.8, label=f"real (n={len(real)})")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.legend(loc="best")
    ax.set_title(
        f"Domain gap: mask-shape features ({X.shape[1]} dims, PCA {k} comps, "
        f"{var:.1%} var)\n"
        f"Logistic-regression sim-vs-real CV accuracy = {cls_acc:.3f} -> "
        f"integration_score = {integration_score:.3f}\n"
        f"(0 = indistinguishable / perfect DA, 1 = trivial to separate)"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=100)
    print("wrote", args.output)


if __name__ == "__main__":
    _main()
