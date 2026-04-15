"""FastAPI orchestrator."""
from __future__ import annotations

import logging
from typing import Optional

import shutil
import tempfile
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from sqlalchemy import func
from sqlalchemy.orm import Session

from config import CONFIG

from .email_service import send_email
from .models import EvaluationLog, Submission, Team, make_session_factory
from .queue import FileQueue

MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024  # 8 GB

log = logging.getLogger(__name__)
app = FastAPI(title="C. elegans Competition Orchestrator")

SessionLocal = make_session_factory(CONFIG.orchestrator.database_url)
queue = FileQueue(CONFIG.orchestrator.queue_dir)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    if x_api_key != CONFIG.orchestrator.api_key:
        raise HTTPException(status_code=401, detail="invalid api key")


@app.get("/api/health")
def health():
    return {"status": "ok", "queue_size": queue.size()}


def _extract_team_and_tag(payload: dict) -> tuple[str, str]:
    """Parse Harbor webhook payload."""
    event_data = payload.get("event_data", {})
    repo = event_data.get("repository", {}).get("repo_full_name", "")
    resources = event_data.get("resources") or [{}]
    tag = resources[0].get("tag", "")
    if not repo or not tag:
        raise HTTPException(status_code=400, detail="malformed harbor webhook")
    parts = repo.split("/")
    # Harbor repo_full_name is "{project}/{repository}"; project is the team.
    team_name = parts[0]
    return team_name, tag


@app.post("/webhook/harbor")
def harbor_webhook(payload: dict, db: Session = Depends(get_db)):
    team_name, tag = _extract_team_and_tag(payload)
    repo = payload["event_data"]["repository"]["repo_full_name"]

    team = db.query(Team).filter(Team.harbor_project == team_name).first()
    if team is None:
        log.warning("unknown team in webhook: %s", team_name)
        return {"error": "unknown team"}

    used = (
        db.query(func.count(Submission.id))
        .filter(
            Submission.team_id == team.id,
            Submission.status.in_(["queued", "running", "completed"]),
        )
        .scalar()
    )
    if used >= team.max_submissions:
        send_email(team.email, "submission_limit_reached", {"max": team.max_submissions})
        return {"error": "submission limit reached"}

    submission = Submission(team_id=team.id, image_tag=f"{repo}:{tag}", status="queued")
    db.add(submission)
    db.commit()
    db.refresh(submission)
    queue.enqueue(submission.id)

    remaining = team.max_submissions - used - 1
    send_email(
        team.email,
        "submission_received",
        {"submission_id": submission.id, "remaining": remaining, "tag": tag},
    )
    return {"status": "queued", "submission_id": submission.id}


@app.get("/api/teams/{team_name}/submissions", dependencies=[Depends(require_api_key)])
def list_team_submissions(team_name: str, db: Session = Depends(get_db)):
    team = db.query(Team).filter(Team.name == team_name).first()
    if not team:
        raise HTTPException(404, "team not found")
    subs = (
        db.query(Submission)
        .filter(Submission.team_id == team.id)
        .order_by(Submission.submitted_at.desc())
        .all()
    )
    return [
        {
            "id": s.id,
            "image_tag": s.image_tag,
            "status": s.status,
            "final_score": s.final_score,
            "registration_score": s.registration_score,
            "integration_score": s.integration_score,
            "submitted_at": s.submitted_at.isoformat() if s.submitted_at else None,
            "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            "error_message": s.error_message,
        }
        for s in subs
    ]


def _authenticate_team(authorization: Optional[str], db: Session) -> Team:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization.split(None, 1)[1].strip()
    team = db.query(Team).filter(Team.api_key == token).first()
    if team is None:
        raise HTTPException(status_code=401, detail="invalid token")
    return team


@app.post("/api/upload")
async def upload_submission(
    request: Request,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """Receive a docker-saved tar(.gz) from a team. Drops it into runtime/inbox."""
    team = _authenticate_team(authorization, db)

    inbox = CONFIG.orchestrator.work_dir.parent / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    # Filename encodes the team to make audit trivial; actual team is re-derived
    # from the loaded image tag by the inbox poller.
    tmp_path = inbox / f".uploading-{team.harbor_project}-{stamp}.tar.gz"
    final_path = inbox / f"{team.harbor_project}-{stamp}.tar.gz"

    total = 0
    try:
        with tmp_path.open("wb") as f:
            async for chunk in request.stream():
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="upload exceeds limit")
                f.write(chunk)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    tmp_path.rename(final_path)
    return {"status": "received", "bytes": total, "path": final_path.name}


@app.get("/api/leaderboard")
def leaderboard(db: Session = Depends(get_db)):
    rows = (
        db.query(Team.name, func.max(Submission.final_score).label("best"))
        .join(Submission, Submission.team_id == Team.id)
        .filter(Submission.status == "completed")
        .group_by(Team.id)
        .order_by(func.max(Submission.final_score).desc())
        .all()
    )
    return [{"team": name, "best_score": float(best) if best is not None else None} for name, best in rows]
