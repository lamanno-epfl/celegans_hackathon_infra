"""Evaluation worker: pulls submissions from queue and runs them end-to-end."""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy.orm import Session

from config import CONFIG
from scoring.combined import compute_final_score

from .email_service import send_email
from .leaderboard import write_leaderboard
from .models import EvaluationLog, Submission, Team, make_session_factory
from .queue import FileQueue
from .validation import ValidationError, validate_output
from .worker_v2 import (
    prepare_input_v2,
    run_container_v2,
    score_submission_v2,
    validate_output_v2,
)


def _v2_eval_root() -> Optional[Path]:
    """Return the v2 evaluation root iff it exists and has both masks and gt."""
    root = CONFIG.data.root / "real" / "held_out" / "evaluation_annotation_SEALED"
    if (root / "masks").is_dir() and (root / "ground_truth_masks").is_dir():
        return root
    return None

log = logging.getLogger(__name__)


@dataclass
class PreparedInput:
    input_dir: Path
    output_dir: Path
    mapping: Dict[str, dict]  # anon_name -> {original, kind}
    manifest: List[str]
    simulated_filenames: List[str]
    real_filenames: List[str]
    poses_gt: Dict[str, dict]
    volume_diameter: float


def _log(db: Session, submission_id: int, text: str) -> None:
    db.add(EvaluationLog(submission_id=submission_id, log_text=text))
    db.commit()
    log.info("sub=%s %s", submission_id, text)


def prepare_input(held_out_root: Path, reference_root: Path, work_dir: Path, rng: np.random.Generator) -> PreparedInput:
    """Copy held-out simulated+real into an anonymized input dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    (input_dir / "images").mkdir(parents=True, exist_ok=True)
    (input_dir / "masks").mkdir(parents=True, exist_ok=True)
    (input_dir / "reference_3d").mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fname in ("volume_nuclei.npy", "volume_membrane.npy", "volume_masks.npy"):
        shutil.copy(reference_root / fname, input_dir / "reference_3d" / fname)

    sim_dir = held_out_root / "simulated"
    real_dir = held_out_root / "real"
    sim_images = sorted((sim_dir / "images").glob("*.npy"))
    real_images = sorted((real_dir / "images").glob("*.npy"))

    sim_poses_src = json.loads((sim_dir / "poses.json").read_text())

    all_items = [(p, "simulated") for p in sim_images] + [(p, "real") for p in real_images]
    rng.shuffle(all_items)

    mapping: Dict[str, dict] = {}
    simulated_anon: List[str] = []
    real_anon: List[str] = []
    poses_gt: Dict[str, dict] = {}
    manifest: List[str] = []

    for idx, (src_img, kind) in enumerate(all_items):
        anon = f"{idx:05d}.npy"
        manifest.append(anon)
        shutil.copy(src_img, input_dir / "images" / anon)
        src_mask = src_img.parent.parent / "masks" / src_img.name
        if src_mask.exists():
            shutil.copy(src_mask, input_dir / "masks" / anon)
        original = src_img.name
        mapping[anon] = {"original": original, "kind": kind}
        if kind == "simulated":
            simulated_anon.append(anon)
            if original in sim_poses_src:
                poses_gt[anon] = sim_poses_src[original]
        else:
            real_anon.append(anon)

    (input_dir / "manifest.json").write_text(json.dumps(manifest))

    vol = np.load(input_dir / "reference_3d" / "volume_nuclei.npy", mmap_mode="r")
    volume_diameter = float(np.sqrt(sum(s**2 for s in vol.shape)))

    return PreparedInput(
        input_dir=input_dir,
        output_dir=output_dir,
        mapping=mapping,
        manifest=manifest,
        simulated_filenames=simulated_anon,
        real_filenames=real_anon,
        poses_gt=poses_gt,
        volume_diameter=volume_diameter,
    )


def run_container(image_tag: str, input_dir: Path, output_dir: Path, timeout: int) -> Tuple[int, str, str]:
    import os as _os
    # With --cap-drop=ALL the container loses CAP_DAC_OVERRIDE, so uid 0 inside
    # can no longer bypass host fs perms. Make the scratch output dir writable
    # for any uid the container might run as.
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
        "/predict.sh",
    ]
    # Use --gpus all only when explicitly enabled (nvidia toolkit + driver match required).
    if _os.environ.get("ENABLE_GPU", "0") == "1" and shutil.which("nvidia-smi"):
        cmd.insert(2, "--gpus")
        cmd.insert(3, "all")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", (exc.stderr or "") + f"\n[timeout after {timeout}s]"


def score_submission(prep: PreparedInput) -> Tuple[float, dict]:
    poses_raw = json.loads((prep.output_dir / "poses.json").read_text())
    embeddings = np.load(prep.output_dir / "embeddings.npy")

    # Build filename_order = manifest (already the order used to produce inputs).
    # Rows of embeddings.npy must correspond to manifest in that order.
    filename_order = prep.manifest
    domain_labels = np.array([0 if prep.mapping[f]["kind"] == "simulated" else 1 for f in filename_order])

    final, details = compute_final_score(
        poses_pred=poses_raw,
        poses_gt=prep.poses_gt,
        embeddings=embeddings,
        domain_labels=domain_labels,
        volume_diameter=prep.volume_diameter,
        simulated_filenames=prep.simulated_filenames,
        filename_order=filename_order,
    )
    return final, details


def evaluate_submission(submission_id: int, db: Session) -> None:
    """End-to-end evaluation for one submission ID. Never raises; always finalizes status."""
    sub: Optional[Submission] = db.get(Submission, submission_id)
    if sub is None:
        log.error("submission %s not found", submission_id)
        return
    team = db.get(Team, sub.team_id)
    work_root = CONFIG.orchestrator.work_dir / f"sub-{submission_id}-{uuid.uuid4().hex[:8]}"
    work_root.mkdir(parents=True, exist_ok=True)

    sub.status = "running"
    sub.started_at = datetime.utcnow()
    db.commit()
    _log(db, submission_id, f"starting evaluation for {sub.image_tag}")
    if team:
        send_email(team.email, "evaluation_started", {"submission_id": submission_id})

    try:
        # --- v2 dispatch ---------------------------------------------------
        v2_root = _v2_eval_root()
        if v2_root is not None:
            rng = np.random.default_rng(CONFIG.scoring.random_seed + submission_id)
            prep = prepare_input_v2(v2_root, work_root, rng)
            _log(db, submission_id, f"[v2] prepared {len(prep.manifest)} seg inputs from {v2_root.name}")

            image_tag = sub.image_tag
            inspect = subprocess.run(
                ["docker", "image", "inspect", image_tag],
                capture_output=True, text=True,
            )
            if inspect.returncode == 0:
                _log(db, submission_id, f"image {image_tag} present locally, skipping pull")
            else:
                _log(db, submission_id, f"pulling {image_tag}")
                pull = subprocess.run(["docker", "pull", image_tag], capture_output=True, text=True)
                if pull.returncode != 0:
                    _log(db, submission_id, f"docker pull failed: {pull.stderr[:500]}")

            _log(db, submission_id, "[v2] running container")
            rc, stdout, stderr = run_container_v2(
                image_tag, prep.input_dir, prep.output_dir,
                timeout=CONFIG.orchestrator.eval_timeout_seconds,
            )
            _log(db, submission_id, f"container exit={rc}")
            if stdout:
                _log(db, submission_id, f"stdout[-2000]: {stdout[-2000:]}")
            if stderr:
                _log(db, submission_id, f"stderr[-2000]: {stderr[-2000:]}")
            if rc != 0:
                raise RuntimeError(f"container exited with code {rc}: {stderr[-500:]}")

            try:
                validate_output_v2(prep.output_dir, prep.manifest)
            except ValidationError as vexc:
                sub.status = "validation_error"
                sub.error_message = str(vexc)
                sub.completed_at = datetime.utcnow()
                db.commit()
                _log(db, submission_id, f"[v2] validation error: {vexc}")
                if team:
                    send_email(team.email, "validation_error", {"details": str(vexc)})
                return

            final, details = score_submission_v2(prep)
            sub.registration_score = None
            sub.integration_score = None
            sub.final_score = final
            sub.status = "completed"
            sub.completed_at = datetime.utcnow()
            db.commit()
            _log(db, submission_id, f"[v2] completed: final={final:.4f} details={json.dumps(details)[:1000]}")

            if team:
                used = sum(1 for s in team.submissions
                           if s.status in ("queued", "running", "completed"))
                remaining = max(0, team.max_submissions - used)
                send_email(
                    team.email, "evaluation_complete",
                    {
                        "submission_id": submission_id,
                        "final_score": final,
                        "registration_score": 0.0,
                        "integration_score": 0.0,
                        "remaining": remaining,
                    },
                )
            return

        # --- v1 legacy path (synthetic data) -------------------------------
        data_root = CONFIG.data.root
        held_out_root = data_root
        # Build held-out directory structure expected by prepare_input.
        sim_held = data_root / "simulated" / "held_out"
        real_held = data_root / "real" / "held_out"
        if not sim_held.exists() or not real_held.exists():
            raise RuntimeError("held-out directories not found; run generate_splits.py first")

        # Fake a root with held_out/simulated and held_out/real layout.
        shim_root = work_root / "held_out_shim"
        (shim_root / "simulated").mkdir(parents=True, exist_ok=True)
        (shim_root / "real").mkdir(parents=True, exist_ok=True)
        for sub_name in ("images", "masks"):
            src = sim_held / sub_name
            if src.exists():
                (shim_root / "simulated" / sub_name).symlink_to(src.resolve())
        if (sim_held / "poses.json").exists():
            (shim_root / "simulated" / "poses.json").symlink_to((sim_held / "poses.json").resolve())
        for sub_name in ("images", "masks"):
            src = real_held / sub_name
            if src.exists():
                (shim_root / "real" / sub_name).symlink_to(src.resolve())

        rng = np.random.default_rng(CONFIG.scoring.random_seed + submission_id)
        prep = prepare_input(shim_root, data_root / "reference_3d", work_root, rng)
        _log(db, submission_id, f"prepared input: {len(prep.manifest)} images")

        # Pull + run. Skip pull if the image already exists locally (uploads
        # via /api/upload or scp-inbox docker-load the image before this runs).
        image_tag = sub.image_tag
        inspect = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            capture_output=True, text=True,
        )
        if inspect.returncode == 0:
            _log(db, submission_id, f"image {image_tag} present locally, skipping pull")
        else:
            _log(db, submission_id, f"pulling {image_tag}")
            pull = subprocess.run(["docker", "pull", image_tag], capture_output=True, text=True)
            if pull.returncode != 0:
                _log(db, submission_id, f"docker pull failed: {pull.stderr[:500]}")

        _log(db, submission_id, "running container")
        rc, stdout, stderr = run_container(
            image_tag, prep.input_dir, prep.output_dir, timeout=CONFIG.orchestrator.eval_timeout_seconds
        )
        _log(db, submission_id, f"container exit={rc}")
        if stdout:
            _log(db, submission_id, f"stdout[-2000]: {stdout[-2000:]}")
        if stderr:
            _log(db, submission_id, f"stderr[-2000]: {stderr[-2000:]}")
        if rc != 0:
            raise RuntimeError(f"container exited with code {rc}: {stderr[-500:]}")

        try:
            validate_output(prep.output_dir, prep.manifest)
        except ValidationError as vexc:
            sub.status = "validation_error"
            sub.error_message = str(vexc)
            sub.completed_at = datetime.utcnow()
            db.commit()
            _log(db, submission_id, f"validation error: {vexc}")
            if team:
                send_email(team.email, "validation_error", {"details": str(vexc)})
            return

        final, details = score_submission(prep)
        sub.registration_score = details["registration_accuracy"]
        sub.integration_score = details["integration_score"]
        sub.final_score = final
        sub.status = "completed"
        sub.completed_at = datetime.utcnow()
        db.commit()
        _log(db, submission_id, f"completed: final={final:.4f} details={json.dumps(details)[:1000]}")

        if team:
            used = sum(
                1 for s in team.submissions
                if s.status in ("queued", "running", "completed")
            )
            remaining = max(0, team.max_submissions - used)
            send_email(
                team.email,
                "evaluation_complete",
                {
                    "submission_id": submission_id,
                    "final_score": final,
                    "registration_score": sub.registration_score,
                    "integration_score": sub.integration_score,
                    "remaining": remaining,
                },
            )
    except Exception as exc:
        log.exception("evaluation failed for %s", submission_id)
        sub.status = "failed"
        sub.error_message = str(exc)[:2000]
        sub.completed_at = datetime.utcnow()
        db.commit()
        _log(db, submission_id, f"failed: {exc}")
        if team:
            send_email(team.email, "evaluation_failed", {"submission_id": submission_id, "error_message": str(exc)[:500]})
    finally:
        try:
            shutil.rmtree(work_root, ignore_errors=True)
        except Exception:
            pass
        try:
            write_leaderboard(db, CONFIG.orchestrator.work_dir.parent / "leaderboard")
        except Exception:
            log.exception("failed to write leaderboard")


def _reap_stuck_running(SessionLocal) -> None:
    """Mark any submissions left in 'running' from a prior crashed worker as failed.

    Without this, a mid-evaluation kill leaves a row in status=running forever,
    which both wastes a quota slot and makes the leaderboard misleading.
    """
    db = SessionLocal()
    try:
        stuck = db.query(Submission).filter(Submission.status == "running").all()
        for s in stuck:
            s.status = "failed"
            s.error_message = "worker restarted mid-evaluation"
            s.completed_at = datetime.utcnow()
        if stuck:
            db.commit()
            log.warning("reaped %d stuck submissions from previous run", len(stuck))
    finally:
        db.close()


def worker_loop(poll_interval: float = 2.0) -> None:
    SessionLocal = make_session_factory(CONFIG.orchestrator.database_url)
    _reap_stuck_running(SessionLocal)
    queue = FileQueue(CONFIG.orchestrator.queue_dir)
    log.info("worker loop started, polling %s", CONFIG.orchestrator.queue_dir)
    while True:
        sub_id = queue.dequeue()
        if sub_id is None:
            time.sleep(poll_interval)
            continue
        db = SessionLocal()
        try:
            evaluate_submission(sub_id, db)
        finally:
            db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    worker_loop()
