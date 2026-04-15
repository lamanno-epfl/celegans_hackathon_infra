# Operator Guide

How to run and administer the competition stack.

## Components

1. **Harbor** — Docker registry receiving team pushes. Installed separately (see
   [Harbor docs](https://goharbor.io/)).
2. **Orchestrator API** (`orchestrator/app.py`) — FastAPI process receiving the
   Harbor webhook and exposing the leaderboard + per-team endpoints.
3. **Worker** (`orchestrator/worker.py`) — long-running process that drains a file
   queue of submission IDs. For each, it prepares an anonymized input directory,
   runs the participant container, validates outputs, scores, and emails the team.
4. **Database** — SQLite by default (`runtime/orchestrator.db`). Swap in Postgres
   by setting `DATABASE_URL=postgresql+psycopg://...`.
5. **SMTP server** — any SMTP relay; set `SMTP_*` env vars and `SMTP_DRY_RUN=false`.

## One-time setup

```bash
# 1. Clone and install
git clone https://github.com/lamanno-epfl/celegans_hackathon_infra
cd celegans_hackathon_infra
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Drop in the reference 3D volume + simulated + real images
#    (see data layout below), then split
python generate_splits.py

# 3. Configure env (see "Environment variables" below)
cp .env.example .env   # create from the list below
source .env

# 4. Install Harbor separately. Once it's reachable at $HARBOR_URL, register teams:
python -m orchestrator.setup_harbor --teams teams.json --creds-dir runtime/credentials
# teams.json: [{"name": "alpha", "email": "alpha@uni.edu"}, ...]

# 5. Add Harbor webhook pointing at https://<orchestrator>/webhook/harbor

# 6. Start API + worker
uvicorn orchestrator.app:app --host 0.0.0.0 --port 8000 &
python -m orchestrator.worker &
```

## Data layout

After running `scripts/generate_synthetic_data.py` (or placing real data) and
`generate_splits.py`:

```
data/
├── reference_3d/{volume_nuclei,volume_membrane,volume_masks}.npy
├── simulated/
│   ├── public/   # given to participants
│   │   ├── images/*.npy, masks/*.npy, poses.json
│   └── held_out/ # never distributed, used for scoring
├── real/
│   ├── public/, held_out/ (no poses.json)
└── splits.json
```

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `REGISTRATION_THRESHOLD` | `0.3` | Below this, `final = registration only`. Calibrate with `scripts/calibrate_threshold.py`. |
| `REGISTRATION_WEIGHT` / `INTEGRATION_WEIGHT` | `0.8` / `0.2` | Weights in the combined formula. |
| `COLLAPSE_STD_THRESHOLD` | `1e-6` | Embeddings with std below this get integration=0. |
| `INTEGRATION_K_FOLDS` | `5` | Folds for the two-sample classifier. |
| `HARBOR_URL` / `HARBOR_ADMIN_USER` / `HARBOR_ADMIN_PASSWORD` | — | Harbor API admin credentials. |
| `HARBOR_WEBHOOK_SECRET` | `changeme` | Reserved (not currently validated — add if Harbor supports signed webhooks in your version). |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` | — | Outbound email. |
| `SMTP_FROM` | `competition@example.org` | From address. |
| `SMTP_TLS` | `false` | `true` to enable STARTTLS / SMTP_SSL. |
| `SMTP_DRY_RUN` | `true` | Dev default: log emails instead of sending. **Set to `false` in production.** |
| `DATABASE_URL` | `sqlite:///runtime/orchestrator.db` | SQLAlchemy URL. |
| `QUEUE_DIR` | `runtime/queue` | File queue directory. |
| `WORK_DIR` | `runtime/work` | Scratch for each evaluation. |
| `DATA_ROOT` | `./data` | Root for competition data. |
| `EVAL_TIMEOUT` | `3600` | Per-container wall-clock cap (seconds). |
| `MAX_SUBMISSIONS` | `10` | Default quota on team creation. |
| `ORCHESTRATOR_API_KEY` | `dev-api-key` | Required by the `X-API-Key` header on team endpoints. |
| `ENABLE_GPU` | `0` | Worker adds `--gpus all` to `docker run` when set to `1`. |

## Harbor setup

1. Follow the [official installer](https://goharbor.io/docs/latest/install-config/).
   Use HTTPS with a real cert — participant `docker login` requires it for remote use.
2. `harbor.yml`: set `hostname` and `harbor_admin_password`; leave the rest mostly
   default for a single-machine evaluator.
3. After Harbor is running, add a **webhook** (Admin → Webhook Policies) pointing at
   `https://<orchestrator-host>/webhook/harbor`, event type `PUSH_ARTIFACT`.
4. Register teams:

```bash
cat > teams.json <<EOF
[
  {"name": "alpha", "email": "a@lab.org"},
  {"name": "bravo", "email": "b@lab.org"}
]
EOF
python -m orchestrator.setup_harbor --teams teams.json --creds-dir runtime/credentials
# one JSON file per team appears in runtime/credentials/*.json — distribute these.
```

## Threshold calibration

Run all baselines after synthetic data is in place:

```bash
python baselines/trivial/train.py --public-dir data/simulated/public --epochs 5
python baselines/domain_adapted/train.py \
    --sim-dir data/simulated/public --real-dir data/real/public --epochs 5

python scripts/calibrate_threshold.py \
    --trivial-weights baselines/trivial/model.pt \
    --domain-weights baselines/domain_adapted/model.pt
# => suggested REGISTRATION_THRESHOLD: 0.xxx
```

Then export `REGISTRATION_THRESHOLD=...` (or bake into `config.py`).

## Operational runbook

- **Stuck submission:** the worker uses `try/finally` to always set terminal state.
  If a row is still `running` after an hour, the worker crashed; requeue with
  `echo '{"submission_id": N, "ts": 0}' > runtime/queue/restart-N.json`.
- **Disk filling up:** check `runtime/work/` — each eval gets a `sub-<id>-<hex>/`
  that should be cleaned on completion, but crashes leave residue. Safe to delete.
- **Image bloat:** run `docker image prune -af` periodically on the eval host.
- **Banning a team:** `UPDATE teams SET max_submissions = <current used> WHERE name=...;`
- **Leaderboard freeze:** stop the worker; new webhooks still queue but won't evaluate.

## Monitoring suggestions (not included, drop-in)

- Tail `evaluation_logs` table for per-submission step-by-step history.
- Export Prometheus metrics: queue depth, failure rate per team, mean eval duration.
- Wire Slack/Discord alerts on `failed` / `validation_error` counts.
