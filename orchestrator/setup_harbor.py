"""Harbor API bootstrap: create project + robot account per team.

NOTE: Harbor must be installed separately. This script uses the Harbor REST API.
"""
from __future__ import annotations

import argparse
import json
import logging
import secrets
from pathlib import Path
from typing import List, Optional

import requests
from requests.auth import HTTPBasicAuth

from config import CONFIG
from .models import Team, make_session_factory

log = logging.getLogger(__name__)


def _auth() -> HTTPBasicAuth:
    return HTTPBasicAuth(CONFIG.harbor.admin_user, CONFIG.harbor.admin_password)


def _api(path: str) -> str:
    return CONFIG.harbor.url.rstrip("/") + "/api/v2.0" + path


def create_project(name: str) -> int:
    r = requests.post(
        _api("/projects"),
        auth=_auth(),
        json={"project_name": name, "metadata": {"public": "false"}},
        timeout=30,
    )
    if r.status_code not in (201, 409):
        r.raise_for_status()
    pr = requests.get(_api(f"/projects/{name}"), auth=_auth(), timeout=30)
    pr.raise_for_status()
    return int(pr.json()["project_id"])


def create_robot(team_name: str, project_name: str) -> dict:
    payload = {
        "name": f"push-{team_name}",
        "duration": -1,
        "level": "project",
        "permissions": [
            {
                "kind": "project",
                "namespace": project_name,
                "access": [
                    {"resource": "repository", "action": "push"},
                    {"resource": "repository", "action": "pull"},
                ],
            }
        ],
    }
    r = requests.post(
        _api(f"/projects/{project_name}/robots"),
        auth=_auth(),
        json=payload,
        timeout=30,
    )
    if r.status_code not in (201, 409):
        r.raise_for_status()
    return r.json() if r.content else {}


def setup_team(team_name: str, email: str, credentials_dir: Path, db_session) -> None:
    project = team_name.lower().replace(" ", "-")
    create_project(project)
    robot = create_robot(team_name, project)

    api_key = secrets.token_urlsafe(32)
    team = db_session.query(Team).filter(Team.name == team_name).first()
    if team is None:
        team = Team(
            name=team_name,
            email=email,
            harbor_project=project,
            max_submissions=CONFIG.orchestrator.max_submissions_default,
            api_key=api_key,
        )
        db_session.add(team)
    else:
        team.email = email
        team.harbor_project = project
        if not team.api_key:
            team.api_key = api_key
    db_session.commit()

    credentials_dir.mkdir(parents=True, exist_ok=True)
    creds_file = credentials_dir / f"{project}.json"
    creds_file.write_text(
        json.dumps(
            {
                "team": team_name,
                "harbor_url": CONFIG.harbor.url,
                "project": project,
                "robot_name": robot.get("name"),
                "robot_secret": robot.get("secret"),
                "orchestrator_api_key": team.api_key,
            },
            indent=2,
        )
    )
    log.info("team %s registered, credentials written to %s", team_name, creds_file)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teams", required=True, help="JSON file: [{name, email}]")
    parser.add_argument("--creds-dir", default="runtime/credentials")
    args = parser.parse_args(argv)

    teams = json.loads(Path(args.teams).read_text())
    SessionLocal = make_session_factory(CONFIG.orchestrator.database_url)
    db = SessionLocal()
    try:
        for t in teams:
            setup_team(t["name"], t["email"], Path(args.creds_dir), db)
    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
