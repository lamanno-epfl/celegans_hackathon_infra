"""Generate public/held_out splits for simulated and real data.

Input layout (produced by scripts/generate_synthetic_data.py or real data sources):
  data/simulated/_raw/{images,masks,poses.json}
  data/real/_raw/{images,masks}

Output layout matches startingprompt.md §3.1.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

from config import CONFIG

log = logging.getLogger(__name__)


def _octant_bin(R: np.ndarray) -> int:
    # Use the direction of R @ [0,0,1] to bin into octants (8 bins).
    v = R @ np.array([0, 0, 1.0])
    b = (int(v[0] >= 0) << 2) | (int(v[1] >= 0) << 1) | int(v[2] >= 0)
    return b


def _stratified_split(items: List[str], bins: List[int], public_ratio: float, rng: np.random.Generator) -> tuple[list, list]:
    by_bin: Dict[int, List[str]] = {}
    for item, b in zip(items, bins):
        by_bin.setdefault(b, []).append(item)
    public, held = [], []
    for b, members in by_bin.items():
        shuffled = list(members)
        rng.shuffle(shuffled)
        k = max(1, int(round(len(shuffled) * public_ratio))) if len(shuffled) > 1 else len(shuffled)
        public.extend(shuffled[:k])
        held.extend(shuffled[k:])
    return public, held


def _move_files(names: List[str], src_root: Path, dst_root: Path, with_masks: bool = True):
    (dst_root / "images").mkdir(parents=True, exist_ok=True)
    if with_masks:
        (dst_root / "masks").mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copy(src_root / "images" / name, dst_root / "images" / name)
        if with_masks and (src_root / "masks" / name).exists():
            shutil.copy(src_root / "masks" / name, dst_root / "masks" / name)


def split_simulated(data_root: Path, public_ratio: float, rng: np.random.Generator) -> dict:
    raw = data_root / "simulated" / "_raw"
    poses = json.loads((raw / "poses.json").read_text())
    names = sorted(poses.keys())
    bins = [_octant_bin(np.array(poses[n]["rotation"])) for n in names]
    public, held = _stratified_split(names, bins, public_ratio, rng)

    _move_files(public, raw, data_root / "simulated" / "public")
    _move_files(held, raw, data_root / "simulated" / "held_out")
    (data_root / "simulated" / "public" / "poses.json").write_text(
        json.dumps({n: poses[n] for n in public}, indent=2)
    )
    (data_root / "simulated" / "held_out" / "poses.json").write_text(
        json.dumps({n: poses[n] for n in held}, indent=2)
    )
    return {"public": public, "held_out": held}


def split_real(data_root: Path, public_ratio: float, rng: np.random.Generator) -> dict:
    raw = data_root / "real" / "_raw"
    names = sorted(p.name for p in (raw / "images").glob("*.npy"))
    # Stratify by mean intensity bucket (3 bins) so held_out covers variability.
    intensities = [float(np.mean(np.load(raw / "images" / n))) for n in names]
    if intensities:
        q1, q2 = np.quantile(intensities, [1 / 3, 2 / 3])
        bins = [0 if i < q1 else (1 if i < q2 else 2) for i in intensities]
    else:
        bins = []
    public, held = _stratified_split(names, bins, public_ratio, rng)
    _move_files(public, raw, data_root / "real" / "public")
    _move_files(held, raw, data_root / "real" / "held_out")

    # Log variability stats.
    def _stats(nms):
        if not nms:
            return {}
        means = [float(np.mean(np.load(raw / "images" / n))) for n in nms]
        mask_counts = [
            int(np.unique(np.load(raw / "masks" / n)).size)
            if (raw / "masks" / n).exists() else 0
            for n in nms
        ]
        return {
            "n": len(nms),
            "mean_intensity": float(np.mean(means)),
            "std_intensity": float(np.std(means)),
            "mask_count_mean": float(np.mean(mask_counts)) if mask_counts else 0.0,
        }
    log.info("real public stats: %s", _stats(public))
    log.info("real held_out stats: %s", _stats(held))
    return {"public": public, "held_out": held}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    data_root = Path(args.data_root) if args.data_root else CONFIG.data.root
    rng = np.random.default_rng(args.seed)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    sim = split_simulated(data_root, CONFIG.data.simulated_public_ratio, rng)
    real = split_real(data_root, CONFIG.data.real_public_ratio, rng)

    splits = {
        "simulated": sim,
        "real": real,
        "config": {
            "simulated_public_ratio": CONFIG.data.simulated_public_ratio,
            "real_public_ratio": CONFIG.data.real_public_ratio,
        },
    }
    (data_root / "splits.json").write_text(json.dumps(splits, indent=2))
    print(
        f"splits written: simulated public={len(sim['public'])} held_out={len(sim['held_out'])}, "
        f"real public={len(real['public'])} held_out={len(real['held_out'])}"
    )


if __name__ == "__main__":
    main()
