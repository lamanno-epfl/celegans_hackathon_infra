"""v2 worker path: seg-in / seg-out contract + domain-adaptation embeddings.

Inputs at `/input/`:
  - `<sample_id>_seg.npy` — Cellpose-style dict per evaluation sample (sim).
    Key "masks" = (554, 554) int array of **instance labels 1..N** (shuffled
    per submission). Carries NO atlas / timepoint / pose information.
  - `real_manual/<LE003_*>_seg.npy` — real manually-annotated embryos, same
    shape, same dict layout, plain Cellpose instance IDs.
  - `manifest.json` — list of sim sample IDs in the shuffled/anonymized order.

Reference atlas mounted at `/atlas/` (read-only):
  - `reference.ome.zarr/` — 4D OME-Zarr v3, (T=255, Z=214, Y=356, X=256),
    `labels` int16 cell IDs, `membrane`/`nucleus` uint8 fluorescence.
  - `name_dictionary.csv` — cell ID → Sulston lineage name.
  Same atlas the scorer uses; participants must NOT bake it into their image.

Outputs required at `/output/`:
  - `<sample_id>_seg.npy` per sim input — same filename, dict with `masks`
    (int32, (554,554), per-region pixel values = predicted atlas IDs).
  - `embeddings.npz` — per-cell feature embeddings for BOTH sim and real,
    with a per-row domain label. Used to score domain adaptation.

Scoring: `final = 0.7 * seg_accuracy + 0.3 * integration_score`, where
  - seg_accuracy = `scoring.seg_accuracy.score_directory` (majority vote
    per gold region vs `ref_mask`),
  - integration_score = `scoring.integration.compute_integration_score`
    (1 − 2·|cv_acc − 0.5|; 1 = sim/real indistinguishable, 0 = perfect
    separability or collapsed embeddings).
"""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import CONFIG
from scoring.integration import compute_integration_score
from scoring.seg_accuracy import score_directory

# Final-score weights. Hardcoded per Luca's spec (2026-04-16):
# "70% seg accuracy + 30% domain-adaptation integration".
SEG_WEIGHT = 0.7
INTEGRATION_WEIGHT = 0.3
# Embedding-dimension guard-rails. D<2 breaks the LR classifier (need variance
# on >=2 features); D>512 is almost certainly a participant mistake (e.g. they
# forgot to pool and shipped a (H*W)-long vector) and would blow up the
# worker's StandardScaler memory.
EMB_MIN_DIM = 2
EMB_MAX_DIM = 512
EMB_MIN_ROWS_PER_DOMAIN = 3


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


def _relabel_to_instance_ids(
    mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Replace atlas IDs in `mask` with a shuffled 1..N instance labeling.

    Participants must NOT see atlas IDs in the input (that would trivialize the
    task — identity would score well above chance). Background (0) is preserved.
    Each submission gets a fresh shuffle so teams cannot cache an ID permutation
    across submissions.
    """
    orig_ids = np.unique(mask[mask > 0])
    if orig_ids.size == 0:
        return mask.astype(np.int32)
    perm = rng.permutation(orig_ids.size) + 1   # 1..N, reserve 0 for bg
    lut = np.zeros(int(mask.max()) + 1, dtype=np.int32)
    for old_id, new_id in zip(orig_ids, perm):
        lut[int(old_id)] = int(new_id)
    return lut[mask]


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
    real_manual_in = input_dir / "real_manual"
    for d in (input_dir, output_dir, gt_seg_dir, real_manual_in):
        d.mkdir(parents=True, exist_ok=True)

    # Stage real manual segs (domain-adaptation reference) as a sibling of the
    # main sim inputs. Participants can read these if they want a DA signal;
    # filenames retain the real LE003_*.npy naming so they can't be confused
    # with the anonymized sample_XXXX inputs.
    manual_root = (
        Path(CONFIG.data.root) / "real" / "held_out" / "05_manual_segmentation"
    )
    if manual_root.is_dir():
        for src in sorted(manual_root.glob("*_seg.npy")):
            shutil.copy(src, real_manual_in / src.name)

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
        masked = _relabel_to_instance_ids(inp["masks"], rng)
        np.save(input_dir / f"{anon_stem}_seg.npy", _mask_to_seg(masked))

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
    (no `/predict.sh` convention required). Mounts the 4D reference atlas
    read-only at `/atlas/` if `CONFIG.data.atlas_dir` exists on disk."""
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
    ]
    atlas_dir = Path(CONFIG.data.atlas_dir)
    if atlas_dir.is_dir() and (atlas_dir / "reference.ome.zarr").is_dir():
        cmd += ["-v", f"{atlas_dir.resolve()}:/atlas:ro"]
    cmd.append(image_tag)
    if _os.environ.get("ENABLE_GPU", "0") == "1" and shutil.which("nvidia-smi"):
        cmd.insert(2, "--gpus")
        cmd.insert(3, "all")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", (exc.stderr or "") + f"\n[timeout after {timeout}s]"


def validate_output_v2(output_dir: Path, manifest: List[str]) -> None:
    """Every manifest `<sample_id>_seg.npy` must exist, load, be a dict with
    int "masks" of shape (554,554). Additionally, a single
    `/output/embeddings.npz` must exist, with `embeddings` (N, D) float and
    `domain` (N,) string array of "sim"/"real" labels. D must be in
    [EMB_MIN_DIM, EMB_MAX_DIM] and both domains must have >= EMB_MIN_ROWS_PER_DOMAIN
    rows so the k-fold classifier is well-defined.
    """
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

    emb_path = output_dir / "embeddings.npz"
    if not emb_path.exists():
        raise ValidationError(
            "output missing embeddings.npz (required for the 30% domain-adaptation "
            "score; emit per-cell features for both sim and real samples — see "
            "quickstart Section 2c)"
        )
    try:
        emb_npz = np.load(emb_path, allow_pickle=False)
    except Exception as exc:
        raise ValidationError(f"embeddings.npz could not be loaded: {exc}")
    if "embeddings" not in emb_npz.files or "domain" not in emb_npz.files:
        raise ValidationError(
            f"embeddings.npz must contain arrays 'embeddings' and 'domain'; "
            f"got {list(emb_npz.files)}"
        )
    emb = np.asarray(emb_npz["embeddings"])
    dom = np.asarray(emb_npz["domain"])
    if emb.ndim != 2:
        raise ValidationError(f"embeddings must be 2D (N,D); got shape {emb.shape}")
    if not np.issubdtype(emb.dtype, np.floating):
        raise ValidationError(f"embeddings must be float; got dtype {emb.dtype}")
    n_rows, d = emb.shape
    if d < EMB_MIN_DIM or d > EMB_MAX_DIM:
        raise ValidationError(
            f"embedding dim D={d} not in [{EMB_MIN_DIM}, {EMB_MAX_DIM}]"
        )
    if dom.shape != (n_rows,):
        raise ValidationError(
            f"domain length {dom.shape} must match embeddings rows {n_rows}"
        )
    dom_str = np.asarray([str(x) for x in dom])
    allowed = {"sim", "real"}
    bad = set(dom_str.tolist()) - allowed
    if bad:
        raise ValidationError(
            f"domain values must be 'sim' or 'real'; saw {sorted(bad)}"
        )
    n_sim = int((dom_str == "sim").sum())
    n_real = int((dom_str == "real").sum())
    if n_sim < EMB_MIN_ROWS_PER_DOMAIN or n_real < EMB_MIN_ROWS_PER_DOMAIN:
        raise ValidationError(
            f"need >= {EMB_MIN_ROWS_PER_DOMAIN} embedding rows per domain; "
            f"got sim={n_sim}, real={n_real}"
        )
    if not np.all(np.isfinite(emb)):
        raise ValidationError("embeddings contain non-finite values (NaN/Inf)")


def _compute_integration_from_output(output_dir: Path) -> Tuple[float, dict]:
    """Load `embeddings.npz` from a validated output dir and compute the
    domain-adaptation integration score. Assumes validate_output_v2 has already
    ensured the file is well-formed.
    """
    emb_npz = np.load(output_dir / "embeddings.npz", allow_pickle=False)
    emb = np.asarray(emb_npz["embeddings"], dtype=np.float64)
    dom = np.asarray([str(x) for x in emb_npz["domain"]])
    labels = (dom == "real").astype(np.int32)  # 0 = sim, 1 = real
    score, details = compute_integration_score(emb, labels, n_folds=5)
    details["n_sim"] = int((dom == "sim").sum())
    details["n_real"] = int((dom == "real").sum())
    details["embedding_dim"] = int(emb.shape[1])
    return float(score), details


def score_submission_v2(prep: PreparedInputV2) -> Tuple[float, dict]:
    seg_result = score_directory(prep.output_dir, prep.gt_dir)
    seg_accuracy = float(seg_result.get("score", 0.0))
    integration_score, integration_details = _compute_integration_from_output(
        prep.output_dir
    )
    final = SEG_WEIGHT * seg_accuracy + INTEGRATION_WEIGHT * integration_score
    return float(final), {
        "seg_accuracy": seg_accuracy,
        "integration_score": integration_score,
        "seg_weight": SEG_WEIGHT,
        "integration_weight": INTEGRATION_WEIGHT,
        "n_scored": seg_result.get("n_scored", 0),
        "n_missing": seg_result.get("n_missing", 0),
        "integration_details": integration_details,
        "note": seg_result.get("note", ""),
    }
