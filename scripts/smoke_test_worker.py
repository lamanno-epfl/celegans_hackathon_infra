"""End-to-end smoke test: seed a team, post a webhook, run the worker once.

Prerequisites:
  - Synthetic data + splits exist under data/ (run generate_synthetic_data.py + generate_splits.py).
  - A local Docker image built for the baseline (e.g. celegans/trivial:latest).

The 'team' harbor_project matches the first path segment of the image tag.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from orchestrator import app as app_mod  # noqa: E402
from orchestrator.models import Team, Submission  # noqa: E402
from orchestrator.worker import evaluate_submission  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="local image tag, e.g. celegans/trivial:latest")
    p.add_argument("--team-email", default="dev@example.org")
    args = p.parse_args()

    project, _, tag = args.image.replace(":", "/", 1).partition("/")
    # Actually parse: "celegans/trivial:latest" -> project=celegans, repo=trivial, tag=latest
    image_no_tag, _, tag_val = args.image.rpartition(":")
    project_name, _, repo_name = image_no_tag.partition("/")
    print(f"project={project_name} repo={repo_name} tag={tag_val}")

    # Seed team.
    db = app_mod.SessionLocal()
    try:
        team = db.query(Team).filter(Team.harbor_project == project_name).first()
        if team is None:
            team = Team(
                name=project_name,
                email=args.team_email,
                harbor_project=project_name,
                max_submissions=10,
            )
            db.add(team)
            db.commit()
            db.refresh(team)
            print(f"seeded team id={team.id}")
        else:
            print(f"using existing team id={team.id}")
    finally:
        db.close()

    # Webhook
    client = TestClient(app_mod.app)
    payload = {
        "event_data": {
            "repository": {"repo_full_name": f"{project_name}/{repo_name}"},
            "resources": [{"tag": tag_val}],
        }
    }
    r = client.post("/webhook/harbor", json=payload)
    print("webhook response:", r.json())
    sub_id = r.json()["submission_id"]

    # Drain the queue once: dequeue and run.
    from orchestrator.queue import FileQueue
    from config import CONFIG

    queue = FileQueue(CONFIG.orchestrator.queue_dir)
    start = time.time()
    while time.time() - start < 10:
        dequeued = queue.dequeue()
        if dequeued is not None:
            break
        time.sleep(0.2)
    if dequeued is None:
        print("ERROR: nothing in queue")
        sys.exit(1)
    print(f"dequeued submission {dequeued}")

    db = app_mod.SessionLocal()
    try:
        evaluate_submission(dequeued, db)
        sub = db.get(Submission, dequeued)
        print("==== RESULT ====")
        print(f"status: {sub.status}")
        print(f"final_score: {sub.final_score}")
        print(f"registration_score: {sub.registration_score}")
        print(f"integration_score: {sub.integration_score}")
        if sub.error_message:
            print(f"error: {sub.error_message}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
