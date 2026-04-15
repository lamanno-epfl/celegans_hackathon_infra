"""Container output validation (section 4.2 of the spec)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import CONFIG


class ValidationError(Exception):
    pass


def validate_output(output_dir: Path, manifest: List[str]) -> Tuple[dict, np.ndarray, dict]:
    """Validate the output directory. Returns (poses, embeddings, metadata) on success."""
    output_dir = Path(output_dir)
    poses_path = output_dir / "poses.json"
    emb_path = output_dir / "embeddings.npy"
    meta_path = output_dir / "metadata.json"

    for p in (poses_path, emb_path, meta_path):
        if not p.exists():
            raise ValidationError(f"missing required output file: {p.name}")

    try:
        poses = json.loads(poses_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValidationError(f"poses.json is not valid JSON: {exc}")

    if not isinstance(poses, dict):
        raise ValidationError("poses.json must be a JSON object")

    missing = [f for f in manifest if f not in poses]
    if missing:
        raise ValidationError(f"poses.json missing entries for {len(missing)} files (first: {missing[0]})")

    for fname, pose in poses.items():
        if not isinstance(pose, dict) or "rotation" not in pose or "translation" not in pose:
            raise ValidationError(f"pose for {fname} must have rotation and translation")
        R = np.array(pose["rotation"], dtype=np.float64)
        t = np.array(pose["translation"], dtype=np.float64)
        if R.shape != (3, 3):
            raise ValidationError(f"rotation for {fname} must be 3x3, got {R.shape}")
        if t.shape not in ((3,), (3, 1), (1, 3)):
            raise ValidationError(f"translation for {fname} must be 3-vector, got {t.shape}")
        if not np.all(np.isfinite(R)) or not np.all(np.isfinite(t)):
            raise ValidationError(f"pose for {fname} contains NaN or Inf")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-2):
            raise ValidationError(f"rotation for {fname} is not orthogonal")
        det = float(np.linalg.det(R))
        if not np.isclose(det, 1.0, atol=1e-2):
            raise ValidationError(f"rotation for {fname} has det={det:.4f}, expected 1")

    try:
        embeddings = np.load(emb_path)
    except Exception as exc:
        raise ValidationError(f"embeddings.npy could not be loaded: {exc}")

    if embeddings.ndim != 2:
        raise ValidationError(f"embeddings.npy must be 2D, got shape {embeddings.shape}")
    n, d = embeddings.shape
    if n != len(manifest):
        raise ValidationError(f"embeddings has {n} rows, manifest has {len(manifest)}")
    dmin, dmax = CONFIG.scoring.min_embedding_dim, CONFIG.scoring.max_embedding_dim
    if not (dmin <= d <= dmax):
        raise ValidationError(f"embedding_dim {d} not in [{dmin}, {dmax}]")
    if not np.all(np.isfinite(embeddings)):
        raise ValidationError("embeddings contain NaN or Inf")

    try:
        metadata = json.loads(meta_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValidationError(f"metadata.json not valid JSON: {exc}")
    if "embedding_dim" not in metadata:
        raise ValidationError("metadata.json missing 'embedding_dim'")
    if int(metadata["embedding_dim"]) != d:
        raise ValidationError(
            f"metadata embedding_dim={metadata['embedding_dim']} != actual {d}"
        )

    return poses, embeddings, metadata
