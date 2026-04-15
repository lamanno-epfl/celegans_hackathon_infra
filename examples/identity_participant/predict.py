"""Fake participant #2: identity pose + image-statistics embeddings.

Identity rotation is actually the mean rotation under geodesic metric, so this
gives non-trivial registration on a uniformly-sampled dataset. Embeddings are
intensity stats that will clearly separate simulated from real (domain gap is
large), so integration will be low.
"""
import json
import os
from pathlib import Path

import numpy as np

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
D = 128


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((INPUT_DIR / "manifest.json").read_text())
    poses = {}
    embs = np.zeros((len(manifest), D), dtype=np.float32)
    for i, name in enumerate(manifest):
        img = np.load(INPUT_DIR / "images" / name).astype(np.float32)
        poses[name] = {"rotation": np.eye(3).tolist(), "translation": [0.0, 0.0, 0.0]}
        stats = []
        for ch in range(img.shape[0]):
            a = img[ch]
            stats.extend(
                [a.mean(), a.std(), np.percentile(a, 10), np.percentile(a, 50), np.percentile(a, 90)]
            )
        s = np.array(stats, dtype=np.float32)
        embs[i, : len(s)] = s
    (OUTPUT_DIR / "poses.json").write_text(json.dumps(poses))
    np.save(OUTPUT_DIR / "embeddings.npy", embs)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps({"embedding_dim": D, "model_name": "identity_stats"})
    )


if __name__ == "__main__":
    main()
