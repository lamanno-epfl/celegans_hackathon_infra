# C. elegans Cell-ID Hackathon

Welcome! This is the competition repo for the EPFL La Manno-lab
*C. elegans cell identification* hackathon. You ship a Docker image, our server
runs it against held-out data in a sandbox, and you get an email with your
score.

---

## What you're predicting

You receive a noisy 2D **segmentation mask** — a 554×554 integer image where
each positive pixel carries an atlas cell ID coming from one timepoint of a 4D
*C. elegans* atlas. Some cells have been dropped (noise). **Your model must
output, for each labeled region, the canonical atlas cell ID.**

In one picture: same pixel regions in, better-named regions out.

Full contract (file paths, shapes, scoring math): **[`docs/contract_v2.md`](docs/contract_v2.md)**.

> **Current status (2026-04-15, evening).** The scoring end is waiting on a
> ground-truth mask bundle from Xinyi — shipped `ground_truth.npz` contains only
> pose parameters, not per-pixel gold. The submission pipeline runs end-to-end
> against the placeholder scorer; real scores light up as soon as the gold
> files arrive. Tracked in [`docs/TODO_PENDING.md`](docs/TODO_PENDING.md).

---

## Getting started (participants)

**1. Read the quickstart.** All the environment gotchas we hit during the live
dry-run are documented — Docker Desktop, Apple-Silicon `--platform linux/amd64`,
zsh line-continuation, spam folder, 8 GB streaming upload with `curl -T`, VPN
requirement. Do not skip it.

👉 **[`docs/participant_quickstart.md`](docs/participant_quickstart.md)**

**2. Start from a template.** The `examples/participant_template_seg/` folder
is a runnable identity baseline (passes input IDs through unchanged). Build it,
submit it, confirm you get a score email, *then* swap in your model.

**3. Submit.**

```bash
docker build --platform linux/amd64 -t lemanichack/mymodel:v1 .
docker save lemanichack/mymodel:v1 | gzip > mymodel.tar.gz
curl -H "Authorization: Bearer $TOKEN" \
     -T mymodel.tar.gz \
     https://<orchestrator>/api/upload
```

You'll get a confirmation email; a second email lands with your score when the
worker finishes (5–60 min depending on queue).

**Prerequisites** (quickstart covers each in detail):
- EPFL network or EPFL VPN.
- Docker (Desktop on Mac/Windows, Engine on Linux).
- The `lemanichack` team credentials from the quickstart doc.
- ~15 GB free disk (CUDA base images are big).

---

## Picture of the flow

```
 you   ──docker save | curl -T──►  orchestrator  ──►  worker
                                         │              │
                                         ▼              ▼
                                    SQLite DB     run container
                                         │        (no net, read-only,
                                         ▼         60-min cap, GPU passthrough)
                                    leaderboard          │
                                    + score email  ◄─────┘
```

---

## FAQ

**Do I need a GPU?** No, but you'll have one on the server (RTX 4070 12 GB).
Build on `nvidia/cuda:12.4.1-runtime-ubuntu22.04` if you want CUDA at runtime.

**How many submissions?** 50 per team (default). The counter only decrements on
successful *evaluation*, not on upload errors or contract-validation failures.

**How long does the container run?** Hard kill at 60 minutes.

**My submission failed — where do I look?** The email contains the error. The
troubleshooting table in the quickstart covers every class of failure we've
seen so far.

**Can I see the held-out data?** No. You get feedback only via your score.
Calibrate locally on the training set.

---

## Developers area

> Everything below is for the platform maintainers. Participants can ignore it.

### Repository layout

```
scoring/              Pure NumPy/sklearn. No GPU. Fully unit tested.
  ├ registration.py       v1: geodesic rotation + normalized translation errors.
  ├ integration.py        v1: deterministic two-sample domain classifier.
  ├ combined.py           v1: threshold gate + weighted combination.
  ├ timepoint.py          v2 scaffold: exact + within-tolerance accuracy.
  ├ cell_naming.py        v2 scaffold: Hungarian Sulston-name scorer.
  ├ combined_v2.py        v2 scaffold: 4-component weighted combiner.
  ├ seg_accuracy.py       v2 CURRENT: importable seg-in/seg-out scorer.
  └ tests/                36 unit tests total.

orchestrator/         FastAPI app, SQLAlchemy models, evaluation worker.
  ├ app.py                /webhook/harbor, /api/upload, /api/leaderboard, ...
  ├ worker.py             Anonymize, docker pull/run, validate, score, email.
  ├ validation.py         Output contract checks (shape, orthogonality, NaN...).
  ├ email_service.py      SMTP (SendGrid) + Jinja2 templates + dry-run.
  ├ queue.py              File queue.
  ├ setup_harbor.py       Per-team Harbor project + robot account bootstrap.
  └ tests/                9 unit tests (webhook, validation, quota).

baselines/            v1 reference submissions (trivial / domain_adapted /
                      degenerate). To be archived once v2 switches on.

examples/
  ├ participant_template_seg/   v2 (current) — seg-in/seg-out identity baseline.
  ├ participant_template/       v1 legacy reference.
  ├ identity_participant/       v1 fake participant.
  ├ random_participant/         v1 fake participant.
  └ blur_participant/           v1 fake participant.

scripts/
  ├ npz_to_seg.py              Xinyi: mask npz → Cellpose _seg.npy.
  ├ score_seg.py               Xinyi: majority-vote per GT region.
  ├ generate_synthetic_data.py v1 synthetic data generator.
  ├ run_local_eval.py          v1 no-docker fast loop.
  ├ smoke_test_worker.py       v1 webhook → queue → worker → score.
  ├ validate_container.py      v1 stand-alone contract check.
  └ poll_scp_inbox.py          SCP/upload inbox processor.

deploy/
  ├ systemd/                   celegans-{worker,webhook,poller}.service.
  ├ install_systemd.sh         One-shot installer.
  └ backup_db.sh               SQLite online-backup via venv Python (6h cron).

data/real/held_out/              Xinyi's dataset drop (2026-04-15).
  └ evaluation_annotation_SEALED/
      ├ masks/sample_XXXX.npz   860 input masks.
      ├ ground_truth.npz        pose params only — gold masks MISSING (see X1).
      ├ generate_30k.py         generation pipeline (cell-dropout noise).
      └ README.md               Xinyi's own doc.

docs/
  ├ architecture.md            Full diagram + design decisions.
  ├ operator_guide.md          Setup, env vars, runbook, calibration.
  ├ deployment.md              Production deployment checklist.
  ├ contract_v2.md             Current target contract (seg-in/seg-out).
  ├ TODO_PENDING.md            Blocked-on-collaborator tracker.
  ├ participant_quickstart.md  Participant onboarding (OS-specific).
  ├ data_inspection/           Exploratory notes on Xinyi's data drop.
  └ chat_backups/              Raw Claude Code session JSONL (redacted).

runtime/                       SQLite DB + inbox + plots + backups. Gitignored.
```

### Operating the platform

- **Services run under systemd** on the orchestrator host:
  `celegans-worker.service`, `celegans-webhook.service`, `celegans-poller.service`.
  Install with `sudo bash deploy/install_systemd.sh`. All three have
  `Restart=always`. Stuck `running` submissions from a prior crash are reaped
  on worker startup.
- **DB backup:** `deploy/backup_db.sh` every 6h via cron; keeps last 14 gzipped
  snapshots in `runtime/backups/`. Uses venv Python's `sqlite3.backup()` — no
  CLI dep.
- **Docker image cache:** worker skips `docker pull` if the tag is already
  present locally. Participants rebuilding the same tag need to bump it.
- **Sandbox:** `--network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges --pids-limit=1024 --memory=32g --cpus=8 --gpus=all` + 60-min wall clock.

### Safety / auditability

| Concern | How |
|---|---|
| Internet lookup cheat | `--network=none`. |
| Filesystem tampering | `--read-only` root; only `/output` and `/tmp` writable. |
| Resource hogging | `--memory=32g --cpus=8`, 60-min kill, `--pids-limit=1024`. |
| Team-A reads Team-B's code | Harbor robot accounts scoped to a single project. |
| Known real-vs-sim labels | Worker shuffles + anonymizes filenames. |
| Scoring flakiness | Fixed seeds in classifier and CV split. |
| Lost submissions on crash | Startup reaper marks `running` → `failed` with reason. |
| Validation-error quota waste | Validation failures do not decrement the counter. |
| Degenerate submissions winning | Threshold gate + embedding-collapse detection (v1 path). |

### Tests

```bash
./.venv/bin/python -m pytest -q       # 36 passed
```

### v1 → v2 switch-on checklist

See `docs/TODO_PENDING.md` for the full list. One-line summary: rewrite
`prepare_input` to convert `masks/*.npz` → `_seg.npy`, accept `<sample>_seg.npy`
in `validation.py`, call `scoring.seg_accuracy.score_directory` in
`score_submission`, swap the default template. None of that lights up in
production until Xinyi ships gold `ground_truth_masks/`.

### Reading list

- `docs/architecture.md` — how the components fit together and why.
- `docs/operator_guide.md` — everything needed to run the platform.
- `docs/deployment.md` — production deployment.
- `docs/contract_v2.md` — the current target contract.
- `docs/chat_backups/` — raw session transcripts (JSONL).
- `startingprompt.md` — original design spec.
