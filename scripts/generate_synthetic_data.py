"""Generate a small synthetic 3D volume + simulated slices + fake 'real' slices.

Used to exercise the full pipeline without real microscopy data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import CONFIG  # noqa: E402


def _rand_rotation(rng: np.random.Generator) -> np.ndarray:
    # Uniform rotation via QR of a Gaussian matrix.
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def make_volume(shape: tuple[int, int, int], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make a synthetic volume: nuclei (blobs), membrane (edges of blobs), masks (integer labels)."""
    D, H, W = shape
    nuclei = np.zeros(shape, dtype=np.float32)
    masks = np.zeros(shape, dtype=np.int32)
    membrane = np.zeros(shape, dtype=np.float32)
    n_blobs = 20
    coords = rng.integers(low=[4, 4, 4], high=[D - 4, H - 4, W - 4], size=(n_blobs, 3))
    radii = rng.integers(3, 6, size=n_blobs)
    zz, yy, xx = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing="ij")
    for i, ((cz, cy, cx), r) in enumerate(zip(coords, radii), start=1):
        dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        inside = dist2 <= r**2
        shell = (dist2 <= (r + 1) ** 2) & (dist2 >= (r - 1) ** 2)
        nuclei[inside] = np.maximum(nuclei[inside], 1.0 - np.sqrt(dist2[inside]) / (r + 1))
        membrane[shell] = np.maximum(membrane[shell], 1.0)
        masks[inside] = i
    nuclei += rng.normal(0, 0.02, size=shape).astype(np.float32)
    membrane += rng.normal(0, 0.02, size=shape).astype(np.float32)
    return nuclei, membrane, masks


def sample_slice(volume: np.ndarray, R: np.ndarray, t: np.ndarray, plane_shape: tuple[int, int], order: int = 1) -> np.ndarray:
    """Sample a 2D slice through the 3D volume at pose (R, t).

    The plane is Z=0 in the rotated frame, centered at the volume center + t.
    """
    D, H, W = volume.shape
    H2, W2 = plane_shape
    vs = np.array([D, H, W]) / 2.0
    uu, vv = np.meshgrid(
        np.arange(H2) - H2 / 2,
        np.arange(W2) - W2 / 2,
        indexing="ij",
    )
    plane = np.stack([np.zeros_like(uu), uu, vv], axis=-1).astype(np.float32)  # (H2, W2, 3)
    # Rotate + translate into volume coordinates.
    rotated = plane @ R.T  # (H2, W2, 3)
    coords = rotated + vs + t  # (H2, W2, 3)
    coords_flat = coords.reshape(-1, 3).T  # (3, N)
    sampled = map_coordinates(volume, coords_flat, order=order, mode="constant", cval=0.0)
    return sampled.reshape(H2, W2).astype(np.float32)


def make_slice(volumes: dict, R: np.ndarray, t: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    nuc = sample_slice(volumes["nuclei"], R, t, shape, order=1)
    mem = sample_slice(volumes["membrane"], R, t, shape, order=1)
    mask = sample_slice(volumes["masks"].astype(np.float32), R, t, shape, order=0).astype(np.int32)
    img = np.stack([nuc, mem], axis=0)  # (2, H, W)
    return img, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-simulated", type=int, default=40)
    parser.add_argument("--n-real", type=int, default=20)
    parser.add_argument("--vol-shape", type=int, nargs=3, default=[48, 64, 64])
    parser.add_argument("--slice-shape", type=int, nargs=2, default=[48, 48])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.out) if args.out else CONFIG.data.root
    rng = np.random.default_rng(args.seed)
    (root / "reference_3d").mkdir(parents=True, exist_ok=True)
    (root / "simulated" / "_raw" / "images").mkdir(parents=True, exist_ok=True)
    (root / "simulated" / "_raw" / "masks").mkdir(parents=True, exist_ok=True)
    (root / "real" / "_raw" / "images").mkdir(parents=True, exist_ok=True)
    (root / "real" / "_raw" / "masks").mkdir(parents=True, exist_ok=True)

    print("generating 3D volume", tuple(args.vol_shape))
    nuc, mem, masks = make_volume(tuple(args.vol_shape), rng)
    np.save(root / "reference_3d" / "volume_nuclei.npy", nuc)
    np.save(root / "reference_3d" / "volume_membrane.npy", mem)
    np.save(root / "reference_3d" / "volume_masks.npy", masks)

    vols = {"nuclei": nuc, "membrane": mem, "masks": masks}
    poses = {}
    translation_scale = min(args.vol_shape) * 0.15
    for i in range(args.n_simulated):
        R = _rand_rotation(rng)
        t = rng.normal(size=3) * translation_scale
        img, mask = make_slice(vols, R, t, tuple(args.slice_shape))
        name = f"sim_{i:04d}.npy"
        np.save(root / "simulated" / "_raw" / "images" / name, img)
        np.save(root / "simulated" / "_raw" / "masks" / name, mask)
        poses[name] = {"rotation": R.tolist(), "translation": t.tolist()}
    (root / "simulated" / "_raw" / "poses.json").write_text(json.dumps(poses, indent=2))

    # Real: same mechanism but different appearance (blur + intensity shift + noise) to simulate domain gap.
    for i in range(args.n_real):
        R = _rand_rotation(rng)
        t = rng.normal(size=3) * translation_scale
        img, mask = make_slice(vols, R, t, tuple(args.slice_shape))
        # Domain gap: stronger noise, gamma shift, scale.
        img = np.sign(img) * (np.abs(img * 1.5) ** 1.3)
        img = img + rng.normal(0, 0.2, img.shape).astype(np.float32)
        name = f"real_{i:04d}.npy"
        np.save(root / "real" / "_raw" / "images" / name, img)
        np.save(root / "real" / "_raw" / "masks" / name, mask)
    print(f"wrote {args.n_simulated} simulated + {args.n_real} real slices to {root}")


if __name__ == "__main__":
    main()
