"""Minimal participant template.

Replace the model logic in `predict_one` with your own. Inputs are already parsed
for you.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

INPUT_DIR = Path(os.environ.get("INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/output"))
EMBEDDING_DIM = 128


def predict_one(image: np.ndarray, mask: np.ndarray, reference: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rotation 3x3, translation 3-vec, embedding D-vec) for one slice.

    Your real model goes here.
    """
    # Identity rotation, zero translation, embedding = feature vector from pooled stats.
    R = np.eye(3)
    t = np.zeros(3)
    # Toy embedding: per-channel mean/std/percentiles of the image.
    stats = []
    for ch in range(image.shape[0]):
        arr = image[ch]
        stats.extend([arr.mean(), arr.std(), np.percentile(arr, 25), np.percentile(arr, 75)])
    pad = EMBEDDING_DIM - len(stats)
    emb = np.concatenate([stats, np.zeros(pad)]) if pad > 0 else np.array(stats[:EMBEDDING_DIM])
    return R, t, emb.astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((INPUT_DIR / "manifest.json").read_text())
    reference = {
        "nuclei": np.load(INPUT_DIR / "reference_3d" / "volume_nuclei.npy"),
        "membrane": np.load(INPUT_DIR / "reference_3d" / "volume_membrane.npy"),
        "masks": np.load(INPUT_DIR / "reference_3d" / "volume_masks.npy"),
    }

    poses = {}
    embeddings = np.zeros((len(manifest), EMBEDDING_DIM), dtype=np.float32)

    for i, name in enumerate(manifest):
        img = np.load(INPUT_DIR / "images" / name)
        mask_path = INPUT_DIR / "masks" / name
        mask = np.load(mask_path) if mask_path.exists() else None
        R, t, emb = predict_one(img, mask, reference)
        poses[name] = {"rotation": R.tolist(), "translation": t.tolist()}
        embeddings[i] = emb

    (OUTPUT_DIR / "poses.json").write_text(json.dumps(poses))
    np.save(OUTPUT_DIR / "embeddings.npy", embeddings)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps({"embedding_dim": EMBEDDING_DIM, "model_name": "template", "notes": "starter"})
    )


if __name__ == "__main__":
    main()
