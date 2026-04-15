"""Watch runtime/inbox/ for *.tar submissions, docker-load them, enqueue.

This is an alternative to Harbor for small, trusted deployments.

Workflow:
  1. Team runs `docker save team/model:v1 | gzip > team_v1.tar.gz` on their box.
  2. Team `scp team_v1.tar.gz evaluator:~/celegans_hackathon_infra/runtime/inbox/`.
  3. This watcher picks it up, docker-loads, extracts the image tag, records a
     submission against the matching team (by harbor_project), moves the tar to
     runtime/inbox/processed/, and enqueues.
  4. Worker evaluates as usual.

Team matching: the first path segment of the loaded image tag must equal the
team's `harbor_project` field in the DB.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402
from orchestrator.models import Submission, Team, make_session_factory  # noqa: E402
from orchestrator.queue import FileQueue  # noqa: E402
from orchestrator.email_service import send_email  # noqa: E402
from orchestrator.leaderboard import write_leaderboard  # noqa: E402

log = logging.getLogger(__name__)

INBOX_DIR = Path(CONFIG.orchestrator.work_dir).parent / "inbox"
PROCESSED_DIR = INBOX_DIR / "processed"
REJECTED_DIR = INBOX_DIR / "rejected"

# Matches e.g. "Loaded image: fusar/mymodel:v1"
LOADED_RE = re.compile(r"Loaded image(?: ID)?:\s+(\S+)")


def docker_load(tar_path: Path) -> list[str]:
    """Return list of image tags loaded from the tar."""
    proc = subprocess.run(
        ["docker", "load", "-i", str(tar_path)], capture_output=True, text=True, timeout=600
    )
    if proc.returncode != 0:
        raise RuntimeError(f"docker load failed: {proc.stderr.strip()}")
    tags = []
    for line in proc.stdout.splitlines():
        m = LOADED_RE.search(line)
        if m and ":" in m.group(1):  # skip untagged sha256 layers
            tags.append(m.group(1))
    return tags


def process_one(tar_path: Path, db, queue: FileQueue) -> None:
    log.info("processing %s", tar_path.name)
    try:
        tags = docker_load(tar_path)
    except Exception as exc:
        log.error("docker load failed for %s: %s", tar_path.name, exc)
        REJECTED_DIR.mkdir(parents=True, exist_ok=True)
        tar_path.rename(REJECTED_DIR / tar_path.name)
        return

    if not tags:
        log.error("no image tags found in %s", tar_path.name)
        REJECTED_DIR.mkdir(parents=True, exist_ok=True)
        tar_path.rename(REJECTED_DIR / tar_path.name)
        return

    image_tag = tags[0]  # if several, pick first
    project = image_tag.split("/", 1)[0]
    team = db.query(Team).filter(Team.harbor_project == project).first()
    if team is None:
        log.error("unknown team (project=%s) in %s; rejecting", project, tar_path.name)
        REJECTED_DIR.mkdir(parents=True, exist_ok=True)
        tar_path.rename(REJECTED_DIR / tar_path.name)
        return

    # Quota check.
    used = (
        db.query(Submission)
        .filter(
            Submission.team_id == team.id,
            Submission.status.in_(["queued", "running", "completed"]),
        )
        .count()
    )
    if used >= team.max_submissions:
        log.warning("team %s over quota; rejecting %s", team.name, tar_path.name)
        send_email(team.email, "submission_limit_reached", {"max": team.max_submissions})
        REJECTED_DIR.mkdir(parents=True, exist_ok=True)
        tar_path.rename(REJECTED_DIR / tar_path.name)
        return

    sub = Submission(team_id=team.id, image_tag=image_tag, status="queued")
    db.add(sub)
    db.commit()
    db.refresh(sub)
    queue.enqueue(sub.id)
    remaining = team.max_submissions - used - 1
    send_email(
        team.email,
        "submission_received",
        {"submission_id": sub.id, "tag": image_tag, "remaining": remaining},
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    tar_path.rename(PROCESSED_DIR / tar_path.name)
    log.info("enqueued submission %s (%s) for team %s", sub.id, image_tag, team.name)
    write_leaderboard(db, CONFIG.orchestrator.work_dir.parent / "leaderboard")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    log.info("watching %s", INBOX_DIR)
    SessionLocal = make_session_factory(CONFIG.orchestrator.database_url)
    queue = FileQueue(CONFIG.orchestrator.queue_dir)
    while True:
        # Pick up any *.tar or *.tar.gz that is stable (not still being written).
        candidates = sorted(
            list(INBOX_DIR.glob("*.tar")) + list(INBOX_DIR.glob("*.tar.gz"))
        )
        for tar_path in candidates:
            # Skip if still being written (size changes between polls).
            size1 = tar_path.stat().st_size
            time.sleep(0.5)
            if not tar_path.exists():
                continue
            if tar_path.stat().st_size != size1:
                continue
            db = SessionLocal()
            try:
                process_one(tar_path, db, queue)
            except Exception:
                log.exception("unhandled error processing %s", tar_path.name)
            finally:
                db.close()
        time.sleep(2.0)


if __name__ == "__main__":
    main()
