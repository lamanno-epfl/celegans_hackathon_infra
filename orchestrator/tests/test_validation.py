import json

import numpy as np
import pytest

from orchestrator.validation import ValidationError, validate_output


def _write_valid(tmp_path, n=5, d=128):
    poses = {
        f"{i:05d}.npy": {
            "rotation": np.eye(3).tolist(),
            "translation": [0.0, 0.0, 0.0],
        }
        for i in range(n)
    }
    (tmp_path / "poses.json").write_text(json.dumps(poses))
    np.save(tmp_path / "embeddings.npy", np.random.RandomState(0).normal(size=(n, d)).astype(np.float32))
    (tmp_path / "metadata.json").write_text(json.dumps({"embedding_dim": d}))
    return [f"{i:05d}.npy" for i in range(n)]


def test_valid(tmp_path):
    manifest = _write_valid(tmp_path)
    poses, emb, meta = validate_output(tmp_path, manifest)
    assert len(poses) == len(manifest)
    assert emb.shape == (len(manifest), 128)


def test_missing_file(tmp_path):
    _write_valid(tmp_path)
    (tmp_path / "poses.json").unlink()
    with pytest.raises(ValidationError, match="missing"):
        validate_output(tmp_path, ["00000.npy"])


def test_bad_rotation(tmp_path):
    manifest = _write_valid(tmp_path)
    bad = {f: {"rotation": [[1, 1, 0], [0, 1, 0], [0, 0, 1]], "translation": [0, 0, 0]} for f in manifest}
    (tmp_path / "poses.json").write_text(json.dumps(bad))
    with pytest.raises(ValidationError, match="orthogonal"):
        validate_output(tmp_path, manifest)


def test_embedding_dim_out_of_range(tmp_path):
    manifest = _write_valid(tmp_path, d=32)
    with pytest.raises(ValidationError, match="embedding_dim"):
        validate_output(tmp_path, manifest)


def test_nan_embeddings(tmp_path):
    manifest = _write_valid(tmp_path)
    bad = np.full((len(manifest), 128), np.nan, dtype=np.float32)
    np.save(tmp_path / "embeddings.npy", bad)
    with pytest.raises(ValidationError, match="NaN"):
        validate_output(tmp_path, manifest)
