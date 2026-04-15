# Production Deployment — What You Actually Need Tomorrow

Concrete, prescriptive, no hand-waving. Done in order.

---

## 0. Machine(s)

**Minimum: one Linux server.** The orchestrator, worker, Harbor, and SMTP relay
can all coexist if you want. Split only if you run out of resources.

Required:

- Linux (tested on Ubuntu 22.04 / 24.04).
- Docker Engine ≥ 24 (`docker version` should print both client and server).
- ≥ 100 GB disk (participant images pile up quickly — each is 1–5 GB).
- 32 GB RAM minimum; 64 GB recommended if running Harbor on the same host.
- GPU optional. If you want to allow GPU submissions:
  - NVIDIA driver matching your torch/CUDA
  - `nvidia-container-toolkit` installed and configured (`nvidia-smi` inside
    `docker run --rm --gpus all nvidia/cuda:12.4.1-base nvidia-smi` must work)
  - Set `ENABLE_GPU=1` in the worker's env
- Outbound internet access (to pull base images for your own build, and for
  SMTP egress).

Not required: a domain name (unless you use Harbor — see §2).

---

## 1. SMTP — pick ONE and set env vars

You asked about "Sentry" — **Sentry is for error tracking, not email.** What you
want is an SMTP relay. Pick one:

| Provider | When to pick it | Signup time |
|---|---|---|
| **Your institution's relay** (e.g. EPFL `mail.epfl.ch`) | If IT gives you access — cheapest, no signup | minutes |
| **AWS SES** | You're on AWS already, want cheap + reliable | 30 min (domain verify) |
| **SendGrid** | You want a UI and free tier | 10 min (needs credit card) |
| **Mailgun** | Same tier, nicer API | 10 min |
| **Postmark** | Transactional-only, highest deliverability | 10 min |

**For tomorrow, pick your institution's relay.** Set:

```bash
export SMTP_HOST=smtp.epfl.ch          # or whatever
export SMTP_PORT=587
export SMTP_USER=your-service-account
export SMTP_PASSWORD=...
export SMTP_FROM=celegans-competition@your-institution.edu
export SMTP_TLS=true
export SMTP_DRY_RUN=false              # <-- important
```

Test once before going live:

```bash
SMTP_DRY_RUN=false python -c "
from orchestrator.email_service import send_email
send_email('you@your-institution.edu', 'evaluation_complete', {
    'submission_id': 999, 'final_score': 0.5, 'registration_score': 0.5,
    'integration_score': 0.5, 'remaining': 9
})
"
```

If that email arrives, you're done with SMTP.

---

## 2. Image delivery — pick A, B, or C

**A. Harbor + webhook** — the original design. Teams `docker push` to your
Harbor; Harbor fires a webhook; you evaluate.
Needs: domain name + TLS cert for Harbor + an HTTP server on the evaluator.

**B. Harbor + polling** — same Harbor, but the evaluator polls the Harbor API
every 30 s instead of listening for webhooks.
Needs: domain name + TLS cert for Harbor; no HTTP server on the evaluator.
**Use `scripts/poll_harbor.py` instead of `uvicorn`.**

**C. No Harbor, SCP drop** — teams `docker save` a tar and SCP it to a shared
inbox directory. A tiny watcher script picks it up, `docker load`s it, enqueues.
Needs: ssh keys per team; no domain, no TLS, no web UI.

### If A or B (Harbor)

1. Provision a DNS name (e.g. `registry.competition.epfl.ch`).
2. Install Harbor via the
   [official offline installer](https://goharbor.io/docs/latest/install-config/download-installer/).
   In `harbor.yml`:
   - `hostname: registry.competition.epfl.ch`
   - Set `harbor_admin_password`.
   - `https:` — point at cert files (use Let's Encrypt; `certbot certonly` works).
3. `./install.sh`
4. Add a webhook (Admin → Webhook Policies) pointing at
   - **(A)** `http://localhost:8000/webhook/harbor` if the orchestrator runs
     on the same host, or a firewalled private IP
   - (N/A for B — no webhook needed)
   Event type: `PUSH_ARTIFACT`.
5. Register teams: `python -m orchestrator.setup_harbor --teams teams.json`.
   This creates Harbor projects + robot accounts and writes one credentials
   JSON per team under `runtime/credentials/`. Email those to teams.

### If C (SCP drop)

Use `scripts/poll_harbor.py`'s cousin (not yet written — ping me and I'll add
`scripts/poll_scp_inbox.py`). Teams `docker save | gzip | ssh evaluator "cat > ~/inbox/<team>-<tag>.tar.gz"`,
the watcher `docker load`s and enqueues.

---

## 3. Processes to keep running

All long-running processes should be `systemd` units (or tmux if you want to be
really minimal). Example unit files:

```ini
# /etc/systemd/system/celegans-worker.service
[Unit]
Description=C. elegans competition worker
After=docker.service
Requires=docker.service

[Service]
User=celegans
WorkingDirectory=/opt/celegans_hackathon_infra
EnvironmentFile=/etc/celegans/env
ExecStart=/opt/celegans_hackathon_infra/.venv/bin/python -m orchestrator.worker
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/celegans-webhook.service   (only needed for option A)
[Unit]
Description=C. elegans webhook receiver
After=network.target

[Service]
User=celegans
WorkingDirectory=/opt/celegans_hackathon_infra
EnvironmentFile=/etc/celegans/env
ExecStart=/opt/celegans_hackathon_infra/.venv/bin/uvicorn orchestrator.app:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/celegans-poller.service   (only needed for option B)
[Unit]
Description=C. elegans Harbor poller
After=network.target

[Service]
User=celegans
WorkingDirectory=/opt/celegans_hackathon_infra
EnvironmentFile=/etc/celegans/env
ExecStart=/opt/celegans_hackathon_infra/.venv/bin/python scripts/poll_harbor.py
Restart=always

[Install]
WantedBy=multi-user.target
```

`/etc/celegans/env` collects all the `SMTP_*`, `HARBOR_*`, `DATA_ROOT`,
`REGISTRATION_THRESHOLD`, etc. settings (see `docs/operator_guide.md` for the
full list).

Enable: `systemctl daemon-reload && systemctl enable --now celegans-worker.service`
(and the matching webhook/poller one).

---

## 4. What counts as "the dashboard"

Instead of an HTTP UI, the worker writes **markdown files** to
`runtime/leaderboard/` after every completed submission:

- `leaderboard.md` — ranked best-score-per-team
- `submissions.md` — last 500 submissions with status + scores + errors

View via `cat`, `less`, or rsync to wherever you want. Easy to commit to a
public git repo periodically if you want a live public leaderboard:

```bash
# Crude public-leaderboard-via-git trick, run in cron or post-submission hook:
cd /opt/celegans-public-leaderboard
cp /opt/celegans_hackathon_infra/runtime/leaderboard/leaderboard.md .
git add leaderboard.md && git commit -m "update" -q && git push
```

---

## 5. Backups

Everything mutable is under `runtime/`:

- `runtime/orchestrator.db` — SQLite DB with teams, submissions, logs
- `runtime/queue/` — in-flight queue files (small, fine to lose in a pinch)
- `runtime/work/` — per-submission scratch (safe to lose)
- `runtime/leaderboard/` — generated markdown (regenerable from the DB)
- `runtime/credentials/` — team credentials **(sensitive — back up carefully)**

Simple backup via cron:

```bash
# /etc/cron.d/celegans-backup
0 * * * * celegans rsync -a /opt/celegans_hackathon_infra/runtime/ /mnt/backup/celegans-$(date +\%Y\%m\%d\%H)/
```

Keep at least one off-host copy of `orchestrator.db` and `credentials/`.

---

## 6. Firewall

If Harbor is on the same box:

- :80, :443 → Harbor (teams push here)
- :8000 → **localhost only** (webhook; do NOT expose)
- :25 / :587 outbound → SMTP relay
- :22 → you

If Harbor is separate: same but Harbor's own firewall rules.

---

## 7. One-time bring-up checklist

Run this in order, tomorrow morning, on the eval host.

```bash
# 0. System
sudo apt-get update && sudo apt-get install -y docker.io python3.11 python3.11-venv git rsync

# 1. Clone + install
sudo mkdir -p /opt && cd /opt
sudo git clone https://github.com/lamanno-epfl/celegans_hackathon_infra
sudo chown -R $USER /opt/celegans_hackathon_infra
cd celegans_hackathon_infra
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Put real data under data/ (or generate synthetic for a dry-run)
python scripts/generate_synthetic_data.py --n-simulated 80 --n-real 60   # for testing
python generate_splits.py

# 3. Config
sudo mkdir -p /etc/celegans
sudo tee /etc/celegans/env >/dev/null <<'EOF'
SMTP_HOST=smtp.your-relay
SMTP_PORT=587
SMTP_USER=...
SMTP_PASSWORD=...
SMTP_FROM=celegans@your-domain
SMTP_TLS=true
SMTP_DRY_RUN=false
HARBOR_URL=https://registry.competition.your-domain
HARBOR_ADMIN_USER=admin
HARBOR_ADMIN_PASSWORD=...
DATABASE_URL=sqlite:////opt/celegans_hackathon_infra/runtime/orchestrator.db
QUEUE_DIR=/opt/celegans_hackathon_infra/runtime/queue
WORK_DIR=/opt/celegans_hackathon_infra/runtime/work
DATA_ROOT=/opt/celegans_hackathon_infra/data
EVAL_TIMEOUT=3600
REGISTRATION_THRESHOLD=0.3
PYTHONPATH=/opt/celegans_hackathon_infra
EOF
sudo chmod 640 /etc/celegans/env
sudo chown root:celegans /etc/celegans/env

# 4. Harbor — install via official installer (if using options A/B)
#    See https://goharbor.io/docs/latest/install-config/

# 5. Teams
cat > teams.json <<'EOF'
[
  {"name": "alpha", "email": "a@lab.edu"},
  {"name": "bravo", "email": "b@lab.edu"}
]
EOF
python -m orchestrator.setup_harbor --teams teams.json --creds-dir runtime/credentials
# Email runtime/credentials/<team>.json to each team.

# 6. Services
sudo tee /etc/systemd/system/celegans-worker.service >/dev/null <<'EOF'
...  # paste unit file from §3
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now celegans-worker.service
# For option B:
sudo systemctl enable --now celegans-poller.service
# For option A:
sudo systemctl enable --now celegans-webhook.service

# 7. Smoke test
cd /opt/celegans_hackathon_infra
docker build -f baselines/degenerate/Dockerfile -t celegans/degenerate:latest .
source .venv/bin/activate
python scripts/smoke_test_worker.py --image celegans/degenerate:latest
# Expect: status=completed, emails sent.

# 8. Calibrate threshold
python baselines/trivial/train.py --public-dir data/simulated/public --epochs 5
python scripts/calibrate_threshold.py --trivial-weights baselines/trivial/model.pt
# Put the suggested value back in /etc/celegans/env as REGISTRATION_THRESHOLD.
# sudo systemctl restart celegans-worker celegans-poller

# 9. Backup cron
sudo tee /etc/cron.d/celegans-backup >/dev/null <<'EOF'
0 * * * * celegans rsync -a /opt/celegans_hackathon_infra/runtime/ /mnt/backup/celegans-$(date +\%Y\%m\%d\%H)/
EOF
```

---

## 8. What I'd still want you to decide before starting

1. **Harbor or no Harbor?** (§2) — big design fork.
2. **SMTP provider** (§1) — need credentials before first email.
3. **Domain + TLS cert** if you pick Harbor.
4. **GPU or CPU eval?** — CPU is fine for these baselines; real models may need GPU.
5. **How many teams, how many submissions per team?** — affects disk + quota defaults.
6. **Registration threshold** — run `scripts/calibrate_threshold.py` on real data
   before opening submissions.

Tell me which of these are unresolved and I'll give you concrete suggestions.
