"""Validate a built Docker image end-to-end against the container contract."""
from __future__ import annotations

import argparse
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
from orchestrator.worker import prepare_input, run_container  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--data-root", default=str(CONFIG.data.root))
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()

    data_root = Path(args.data_root)
    sim_held = data_root / "simulated" / "held_out"
    real_held = data_root / "real" / "held_out"

    work = Path(tempfile.mkdtemp(prefix="validate-"))
    try:
        shim = work / "shim"
        (shim / "simulated").mkdir(parents=True, exist_ok=True)
        (shim / "real").mkdir(parents=True, exist_ok=True)
        for sub in ("images", "masks"):
            if (sim_held / sub).exists():
                (shim / "simulated" / sub).symlink_to((sim_held / sub).resolve())
        if (sim_held / "poses.json").exists():
            (shim / "simulated" / "poses.json").symlink_to((sim_held / "poses.json").resolve())
        for sub in ("images", "masks"):
            if (real_held / sub).exists():
                (shim / "real" / sub).symlink_to((real_held / sub).resolve())

        prep = prepare_input(shim, data_root / "reference_3d", work, np.random.default_rng(0))
        rc, stdout, stderr = run_container(args.image, prep.input_dir, prep.output_dir, timeout=args.timeout)
        print("return code:", rc)
        if stdout:
            print("stdout:", stdout[-2000:])
        if stderr:
            print("stderr:", stderr[-2000:], file=sys.stderr)
        if rc != 0:
            sys.exit(rc)
        validate_output(prep.output_dir, prep.manifest)
        print("VALIDATION OK")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
