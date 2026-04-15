"""Run a baseline locally (bypassing Docker) and score its outputs.

This is the end-to-end test that stitches everything together without needing Docker.
"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402
from orchestrator.validation import validate_output  # noqa: E402
from orchestrator.worker import prepare_input  # noqa: E402
from scoring.combined import compute_final_score  # noqa: E402


def run_predict_inline(baseline: str, input_dir: Path, output_dir: Path, weights: Path | None) -> None:
    """Run the baseline's predict.py in-process."""
    if baseline == "trivial":
        mod = importlib.import_module("baselines.trivial.predict")
    elif baseline == "domain_adapted":
        mod = importlib.import_module("baselines.domain_adapted.predict")
    elif baseline == "degenerate":
        mod = importlib.import_module("baselines.degenerate.predict")
    else:
        raise ValueError(f"unknown baseline: {baseline}")
    argv = ["--input-dir", str(input_dir), "--output-dir", str(output_dir)]
    if weights is not None:
        argv.extend(["--weights", str(weights)])
    old_argv = sys.argv
    try:
        sys.argv = ["predict.py"] + argv
        mod.main()
    finally:
        sys.argv = old_argv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, choices=["trivial", "domain_adapted", "degenerate"])
    p.add_argument("--data-root", default=str(CONFIG.data.root))
    p.add_argument("--weights", default=None)
    p.add_argument("--keep", action="store_true", help="keep working directory")
    args = p.parse_args()

    data_root = Path(args.data_root)
    sim_held = data_root / "simulated" / "held_out"
    real_held = data_root / "real" / "held_out"
    if not sim_held.exists() or not real_held.exists():
        print("error: held-out data not found. Run generate_synthetic_data.py then generate_splits.py", file=sys.stderr)
        sys.exit(1)

    work = Path(tempfile.mkdtemp(prefix="local-eval-"))
    try:
        shim = work / "held_out_shim"
        (shim / "simulated").mkdir(parents=True, exist_ok=True)
        (shim / "real").mkdir(parents=True, exist_ok=True)
        for sub in ("images", "masks"):
            src = sim_held / sub
            if src.exists():
                (shim / "simulated" / sub).symlink_to(src.resolve())
        (shim / "simulated" / "poses.json").symlink_to((sim_held / "poses.json").resolve())
        for sub in ("images", "masks"):
            src = real_held / sub
            if src.exists():
                (shim / "real" / sub).symlink_to(src.resolve())

        rng = np.random.default_rng(123)
        prep = prepare_input(shim, data_root / "reference_3d", work, rng)
        print(f"prepared {len(prep.manifest)} images (simulated={len(prep.simulated_filenames)}, real={len(prep.real_filenames)})")

        run_predict_inline(args.baseline, prep.input_dir, prep.output_dir, Path(args.weights) if args.weights else None)

        validate_output(prep.output_dir, prep.manifest)
        print("validation OK")

        poses_pred = json.loads((prep.output_dir / "poses.json").read_text())
        embeddings = np.load(prep.output_dir / "embeddings.npy")
        domain_labels = np.array([0 if prep.mapping[f]["kind"] == "simulated" else 1 for f in prep.manifest])

        final, details = compute_final_score(
            poses_pred=poses_pred,
            poses_gt=prep.poses_gt,
            embeddings=embeddings,
            domain_labels=domain_labels,
            volume_diameter=prep.volume_diameter,
            simulated_filenames=prep.simulated_filenames,
            filename_order=prep.manifest,
        )
        print(json.dumps({"final": final, "details": details}, indent=2, default=str))
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            print(f"kept work dir: {work}")


if __name__ == "__main__":
    main()
