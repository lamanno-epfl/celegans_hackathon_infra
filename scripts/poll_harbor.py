"""Alternative to the Harbor webhook: poll Harbor for new image tags.

Use this instead of running `uvicorn orchestrator.app:app` when you prefer
no HTTP server on the evaluation host. Enqueues new (project, tag) pairs
into the same file queue the worker consumes.

Requires: HARBOR_URL, HARBOR_ADMIN_USER, HARBOR_ADMIN_PASSWORD env vars.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import requests
from requests.auth import HTTPBasicAuth

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import CONFIG  # noqa: E402
from orchestrator.models import Submission, Team, make_session_factory  # noqa: E402
from orchestrator.queue import FileQueue  # noqa: E402

log = logging.getLogger(__name__)


def _auth():
    return HTTPBasicAuth(CONFIG.harbor.admin_user, CONFIG.harbor.admin_password)


def _api(path: str) -> str:
    return CONFIG.harbor.url.rstrip("/") + "/api/v2.0" + path


def iter_artifacts(project: str) -> Iterable[dict]:
    page = 1
    while True:
        r = requests.get(
            _api(f"/projects/{project}/repositories"),
            auth=_auth(),
            params={"page": page, "page_size": 100},
            timeout=30,
        )
        r.raise_for_status()
        repos = r.json() or []
        if not repos:
            break
        for repo in repos:
            rname = repo["name"].split("/", 1)[1]  # project/repo -> repo
            ar = requests.get(
                _api(f"/projects/{project}/repositories/{rname}/artifacts"),
                auth=_auth(),
                params={"with_tag": "true", "page_size": 100},
                timeout=30,
            )
            ar.raise_for_status()
            for art in ar.json() or []:
                for tag in (art.get("tags") or []):
                    yield {"repo": f"{project}/{rname}", "tag": tag["name"], "pushed": tag.get("push_time")}
        if len(repos) < 100:
            break
        page += 1


def poll_once(db, queue: FileQueue) -> int:
    """Find new images in Harbor that we haven't seen; enqueue them."""
    new_count = 0
    teams = db.query(Team).all()
    seen_tags = {row.image_tag for row in db.query(Submission.image_tag).all()}
    for team in teams:
        try:
            for art in iter_artifacts(team.harbor_project):
                image_tag = f"{art['repo']}:{art['tag']}"
                if image_tag in seen_tags:
                    continue
                used = (
                    db.query(Submission)
                    .filter(
                        Submission.team_id == team.id,
                        Submission.status.in_(["queued", "running", "completed"]),
                    )
                    .count()
                )
                if used >= team.max_submissions:
                    log.info("skipping %s: team %s over quota", image_tag, team.name)
                    continue
                sub = Submission(team_id=team.id, image_tag=image_tag, status="queued")
                db.add(sub)
                db.commit()
                db.refresh(sub)
                queue.enqueue(sub.id)
                seen_tags.add(image_tag)
                new_count += 1
                log.info("enqueued new submission %s (%s)", sub.id, image_tag)
        except requests.RequestException as exc:
            log.warning("Harbor poll error for %s: %s", team.name, exc)
    return new_count


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    SessionLocal = make_session_factory(CONFIG.orchestrator.database_url)
    queue = FileQueue(CONFIG.orchestrator.queue_dir)
    interval = 30
    while True:
        db = SessionLocal()
        try:
            poll_once(db, queue)
        finally:
            db.close()
        time.sleep(interval)


if __name__ == "__main__":
    main()
