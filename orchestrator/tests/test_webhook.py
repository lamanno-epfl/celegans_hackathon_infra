import os
import tempfile

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "t.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("QUEUE_DIR", str(tmp_path / "q"))
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "w"))
    monkeypatch.setenv("SMTP_DRY_RUN", "true")

    import importlib
    import config as cfg
    importlib.reload(cfg)
    import orchestrator.app as app_mod
    importlib.reload(app_mod)
    from orchestrator.models import Team
    db = app_mod.SessionLocal()
    db.add(Team(name="alpha", email="a@x.com", harbor_project="alpha", max_submissions=2))
    db.commit()
    db.close()
    return TestClient(app_mod.app), app_mod


def _payload(project, tag):
    return {
        "event_data": {
            "repository": {"repo_full_name": f"{project}/model"},
            "resources": [{"tag": tag}],
        }
    }


def test_health(client):
    c, _ = client
    r = c.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_webhook_queues(client):
    c, _ = client
    r = c.post("/webhook/harbor", json=_payload("alpha", "v1"))
    assert r.status_code == 200
    assert r.json()["status"] == "queued"


def test_webhook_limit(client):
    c, app_mod = client
    c.post("/webhook/harbor", json=_payload("alpha", "v1"))
    c.post("/webhook/harbor", json=_payload("alpha", "v2"))
    r = c.post("/webhook/harbor", json=_payload("alpha", "v3"))
    assert r.json().get("error") == "submission limit reached"


def test_webhook_unknown_team(client):
    c, _ = client
    r = c.post("/webhook/harbor", json=_payload("ghost", "v1"))
    assert r.json().get("error") == "unknown team"
