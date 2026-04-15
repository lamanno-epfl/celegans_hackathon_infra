"""Degenerate baseline: random poses, constant embeddings. Should fail collapse check."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=os.environ.get("INPUT_DIR", "/input"))
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "/output"))
    args = p.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((input_dir / "manifest.json").read_text())

    rng = np.random.default_rng(0)
    poses = {}
    for name in manifest:
        A = rng.normal(size=(3, 3))
        Q, R = np.linalg.qr(A)
        Q = Q @ np.diag(np.sign(np.diag(R)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        poses[name] = {"rotation": Q.tolist(), "translation": [0.0, 0.0, 0.0]}

    embeddings = np.zeros((len(manifest), 128), dtype=np.float32)  # collapse
    (output_dir / "poses.json").write_text(json.dumps(poses))
    np.save(output_dir / "embeddings.npy", embeddings)
    (output_dir / "metadata.json").write_text(
        json.dumps({"embedding_dim": 128, "model_name": "degenerate"})
    )


if __name__ == "__main__":
    main()
