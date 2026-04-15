"""Zhou et al. (2019) continuous 6D rotation representation."""
from __future__ import annotations

import numpy as np
import torch


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrices via Gram-Schmidt.

    d6: shape (..., 6). Returns (..., 3, 3).
    """
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., 0, :], R[..., 1, :]], dim=-1)


def numpy_rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]

    def _norm(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True)
        return x / np.clip(n, 1e-9, None)

    b1 = _norm(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)
