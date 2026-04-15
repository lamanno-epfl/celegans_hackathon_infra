"""Fake participant #1: random poses + Gaussian embeddings.

Expected outcome: poor registration, moderate integration (embeddings are random
so the classifier cannot learn a stable boundary across folds).
"""
import json
import os
from pathlib import Path

import numpy as np

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))

rng = np.random.default_rng(7)


def random_rotation():
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((INPUT_DIR / "manifest.json").read_text())
    poses = {
        n: {"rotation": random_rotation().tolist(), "translation": rng.normal(size=3).tolist()}
        for n in manifest
    }
    D = 128
    emb = rng.normal(size=(len(manifest), D)).astype(np.float32)
    (OUTPUT_DIR / "poses.json").write_text(json.dumps(poses))
    np.save(OUTPUT_DIR / "embeddings.npy", emb)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps({"embedding_dim": D, "model_name": "random_guesser"})
    )


if __name__ == "__main__":
    main()
