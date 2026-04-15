"""v2 worker path: seg-in / seg-out contract.

Inputs at `/input/`: one `<sample_id>_seg.npy` per sample (Cellpose-style dict;
key "masks" = HxW int array of atlas cell IDs with dropout noise).

Outputs at `/output/`: one `<sample_id>_seg.npy` per input, same filename, with
predicted canonical atlas cell IDs per region.

Scoring: `scoring.seg_accuracy.score_directory(output_dir, gt_dir)` — majority
vote per GT region vs `ref_mask`.
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from scoring.seg_accuracy import score_directory


@dataclass
class PreparedInputV2:
    input_dir: Path
    output_dir: Path
    gt_dir: Path
    manifest: List[str]                  # sample ids, anonymized order
    mapping: Dict[str, str]              # anon_sample_id -> original_sample_id


def _mask_to_seg(mask: np.ndarray) -> dict:
    mask = mask.astype(np.int32)
    return {
        "masks": mask,
        "cell_ids": sorted(int(x) for x in np.unique(mask) if x > 0),
    }


def prepare_input_v2(
    eval_root: Path,
    work_dir: Path,
    rng: np.random.Generator,
) -> PreparedInputV2:
    """Stage evaluation inputs for v2.

    `eval_root` points at `evaluation_annotation_SEALED/` — expects `masks/*.npz`
    and `ground_truth_masks/*.npz`. Only samples that have BOTH a mask and a gt
    file are included (upload may be in progress).
    """
    eval_root = Path(eval_root)
    masks_dir = eval_root / "masks"
    gt_npz_dir = eval_root / "ground_truth_masks"

    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    gt_seg_dir = work_dir / "gt_seg"
    for d in (input_dir, output_dir, gt_seg_dir):
        d.mkdir(parents=True, exist_ok=True)

    mask_files = {p.stem: p for p in sorted(masks_dir.glob("sample_*.npz"))}
    gt_files = {p.stem: p for p in sorted(gt_npz_dir.glob("sample_*.npz"))}
    common = sorted(set(mask_files) & set(gt_files))
    if not common:
        raise RuntimeError(f"no overlapping samples in {masks_dir} and {gt_npz_dir}")

    indices = list(range(len(common)))
    rng.shuffle(indices)

    manifest: List[str] = []
    mapping: Dict[str, str] = {}

    for anon_idx, orig_idx in enumerate(indices):
        orig_stem = common[orig_idx]
        anon_stem = f"sample_{anon_idx:04d}"
        manifest.append(anon_stem)
        mapping[anon_stem] = orig_stem

        inp = np.load(mask_files[orig_stem])
        np.save(input_dir / f"{anon_stem}_seg.npy", _mask_to_seg(inp["masks"]))

        gt = np.load(gt_files[orig_stem], allow_pickle=True)
        np.save(gt_seg_dir / f"{anon_stem}_seg.npy", _mask_to_seg(gt["ref_mask"]))

    (input_dir / "manifest.json").write_text(
        "[" + ", ".join(f'"{s}"' for s in manifest) + "]"
    )

    return PreparedInputV2(
        input_dir=input_dir,
        output_dir=output_dir,
        gt_dir=gt_seg_dir,
        manifest=manifest,
        mapping=mapping,
    )


def run_container_v2(
    image_tag: str, input_dir: Path, output_dir: Path, timeout: int
) -> Tuple[int, str, str]:
    """Run a v2 container. Same sandbox as v1, but use the image's own CMD
    (no `/predict.sh` convention required)."""
    import os as _os
    _os.chmod(output_dir, 0o777)
    cmd = [
        "docker", "run", "--rm",
        "--network=none",
        "--memory=32g", "--memory-swap=32g",
        "--cpus=8",
        "--pids-limit=4096",
        "--read-only",
        "--tmpfs", "/tmp:size=10g",
        "--security-opt=no-new-privileges",
        "--cap-drop=ALL",
        "-v", f"{input_dir}:/input:ro",
        "-v", f"{output_dir}:/output",
        image_tag,
    ]
    if _os.environ.get("ENABLE_GPU", "0") == "1" and shutil.which("nvidia-smi"):
        cmd.insert(2, "--gpus")
        cmd.insert(3, "all")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", (exc.stderr or "") + f"\n[timeout after {timeout}s]"


def validate_output_v2(output_dir: Path, manifest: List[str]) -> None:
    """Each `<sample_id>_seg.npy` in manifest must exist, load, and be a
    dict with int "masks" of shape 554x554."""
    from .validation import ValidationError

    output_dir = Path(output_dir)
    missing = []
    for sid in manifest:
        p = output_dir / f"{sid}_seg.npy"
        if not p.exists():
            missing.append(p.name)
            continue
        try:
            obj = np.load(p, allow_pickle=True).item()
        except Exception as exc:
            raise ValidationError(f"{p.name} could not be loaded: {exc}")
        if not isinstance(obj, dict) or "masks" not in obj:
            raise ValidationError(f"{p.name} must be a dict with key 'masks'")
        m = np.asarray(obj["masks"])
        if m.ndim != 2:
            raise ValidationError(f"{p.name} masks must be 2D, got shape {m.shape}")
        if not np.issubdtype(m.dtype, np.integer):
            raise ValidationError(f"{p.name} masks must be integer, got {m.dtype}")
        if m.shape != (554, 554):
            raise ValidationError(f"{p.name} masks shape {m.shape}, expected (554, 554)")
    if missing:
        raise ValidationError(
            f"output missing {len(missing)} seg files (first: {missing[0]})"
        )


def score_submission_v2(prep: PreparedInputV2) -> Tuple[float, dict]:
    result = score_directory(prep.output_dir, prep.gt_dir)
    final = float(result.get("score", 0.0))
    return final, {
        "seg_accuracy": final,
        "n_scored": result.get("n_scored", 0),
        "n_missing": result.get("n_missing", 0),
        "note": result.get("note", ""),
    }
