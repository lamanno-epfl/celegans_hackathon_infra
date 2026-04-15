"""Fake participant #3: "blurred" identity baseline.

Heavy gaussian blur before computing intensity stats. The blurred embeddings
should look much more similar across simulated and real (domain gap shrinks
after low-pass filtering), so integration should improve relative to the
identity_stats submission, while registration stays at the identity baseline.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

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
            blurred = gaussian_filter(img[ch], sigma=4.0)
            # Normalize per-image so global intensity differences don't drive the embedding.
            blurred = (blurred - blurred.mean()) / (blurred.std() + 1e-6)
            stats.extend(
                [np.percentile(blurred, p) for p in [10, 25, 50, 75, 90]]
            )
        s = np.array(stats, dtype=np.float32)
        embs[i, : len(s)] = s
    (OUTPUT_DIR / "poses.json").write_text(json.dumps(poses))
    np.save(OUTPUT_DIR / "embeddings.npy", embs)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps({"embedding_dim": D, "model_name": "blur_stats_normalized"})
    )


if __name__ == "__main__":
    main()
