# C. elegans Cell-ID Hackathon

Welcome! This is the competition repo for the EPFL La Manno-lab
*C. elegans cell identification* hackathon. You ship a Docker image, our server
runs it against held-out data in a sandbox, and you get an email with your
score.

---

## What you're predicting

**In one sentence:** each input is a 2D segmentation mask cut from a 3D
*C. elegans* embryo at an unknown timepoint + unknown pose, with meaningless
pixel labels; your model must re-label each region with its canonical atlas
cell ID, using the 4D reference atlas baked into your Docker image.

**For the exact data contract** (file paths, dict keys, dtypes, a runnable
loader example), read **Section 2 of
[`docs/participant_quickstart.md`](docs/participant_quickstart.md#2-what-your-container-will-see-and-what-it-must-write)**.
That section is the single source of truth — the rest of this file summarizes.

Scoring: `final = 0.7 · seg_accuracy + 0.3 · integration_score`.

- **seg_accuracy** — per-region majority vote of predicted pixel IDs vs gold
  atlas IDs, micro-averaged across 857 held-out sim samples.
- **integration_score** — 5-fold logistic-regression classifier on the
  participant's `/output/embeddings.npz`, which must contain per-cell features
  for **both held-out sets** (sim `sample_*_seg.npy` + real `real_manual/*`).
  Score is `1 − 2·|cv_acc − 0.5|`: perfect sim/real mixing = 1.0, perfect
  separability = 0.0. Neither set is seen by participants locally; both are
  only mounted into the sandbox at eval time.

Identity baseline ≈ 0 on both terms. Full scoring math in
[`docs/contract_v2.md`](docs/contract_v2.md).

> **Status (2026-04-16):** pipeline live, scoring against the 857/860 SEALED
> samples. Inputs are re-labeled to instance IDs at submission time (each
> submission sees a fresh shuffle). Identity baseline ≈ 0. Reference 4D atlas
> at `data/reference_3d/` is a **placeholder** — a real atlas drop is pending
> from the La Manno lab; filename/path will stay the same.

---

## Getting started (participants)

**1. Read the quickstart.** All the environment gotchas we hit during the live
dry-run are documented — Docker Desktop, Apple-Silicon `--platform linux/amd64`,
zsh line-continuation, spam folder, 8 GB streaming upload with `curl -T`, VPN
requirement. Do not skip it.

👉 **[`docs/participant_quickstart.md`](docs/participant_quickstart.md)**

**2. Start from a template.** The `examples/participant_template_seg/` folder
is a runnable identity baseline — not a useful model, just a smoke test that
proves the Docker round-trip works. Build it, submit it, confirm you get a
score email, *then* swap in your real model.

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

**Do I need a GPU? → Yes, strongly recommended.** This is an ML competition
and the server runs your container with `--gpus all` on a **NVIDIA RTX 4070
(12 GB)**. CPU submissions will hit the 120-min wall clock on anything
non-trivial.

**What base image works out of the box?** The eval host is: driver
`570.x` (CUDA 12.8 cap), Docker `29.x`, nvidia-container-toolkit `1.18.x`,
RTX 4070 (Ada, compute capability `sm_89`). **Verified end-to-end:**

```dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
```

That is the exact image behind `examples/pytorch_baseline/` — Python 3.11,
torch 2.4.1 + CUDA 12.4 kernels preinstalled, no `apt-get` needed (builds
cleanly even behind EPFL VPN where `archive.ubuntu.com` is flaky), and
`torch.cuda.is_available()` returns `True` at eval time. If you roll your own
base, stay inside **CUDA runtime 11.8 – 12.8** (anything newer than the host's
12.8 cap will fail with `CUDA driver version is insufficient`) and use a
PyTorch wheel ≥ 2.1 (earlier wheels don't include `sm_89`).

**Quick sanity check at the top of your `predict.py`:**

```python
import torch
assert torch.cuda.is_available(), "GPU not visible — rebuild on a CUDA base image"
print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}")
```

If that print doesn't show `NVIDIA GeForce RTX 4070` in the container logs
(they land in the failure email), you shipped a CPU-only wheel.

**How many submissions?** 11 per team. The counter only decrements on
successful *evaluation*, not on upload errors or contract-validation failures.

**How long does the container run?** Hard kill at 120 minutes.

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
  ├ seg_accuracy.py       CURRENT: importable seg-in/seg-out scorer.
  ├ registration.py       legacy pose-regression scorer (not wired in).
  ├ integration.py        legacy domain-classifier scorer (not wired in).
  ├ combined.py           legacy combiner (not wired in).
  ├ timepoint.py          legacy scaffold (not wired in).
  ├ cell_naming.py        legacy Hungarian Sulston-name scorer (not wired in).
  ├ combined_v2.py        legacy 4-component combiner (not wired in).
  └ tests/                36 unit tests total.

orchestrator/         FastAPI app, SQLAlchemy models, evaluation worker.
  ├ app.py                /webhook/harbor, /api/upload, /api/leaderboard, ...
  ├ worker.py             Anonymize, docker pull/run, validate, score, email.
  ├ validation.py         Output contract checks (shape, orthogonality, NaN...).
  ├ email_service.py      SMTP (SendGrid) + Jinja2 templates + dry-run.
  ├ queue.py              File queue.
  ├ setup_harbor.py       Per-team Harbor project + robot account bootstrap.
  └ tests/                9 unit tests (webhook, validation, quota).

baselines/            Legacy reference submissions (kept for history).

examples/
  ├ participant_template_seg/   seg-in/seg-out identity baseline (NumPy).
  ├ pytorch_baseline/           seg-in/seg-out identity baseline (PyTorch + CUDA).
  ├ participant_template/       legacy reference (pre-seg contract).
  ├ identity_participant/       legacy fake participant.
  ├ random_participant/         legacy fake participant.
  └ blur_participant/           legacy fake participant.

scripts/
  ├ npz_to_seg.py              mask npz → Cellpose _seg.npy (Xinyi).
  ├ score_seg.py               majority-vote per GT region (Xinyi).
  ├ viz/plot_seg_overlay.py    input / GT / pred / correctness overlay plot.
  ├ viz/plot_domain_features.py sim-vs-real shape-feature UMAP + LR-CV score.
  ├ generate_synthetic_data.py legacy synthetic data generator.
  ├ run_local_eval.py          legacy no-docker fast loop.
  ├ smoke_test_worker.py       legacy webhook → queue → worker → score.
  ├ validate_container.py      legacy stand-alone contract check.
  └ poll_scp_inbox.py          SCP/upload inbox processor.

deploy/
  ├ systemd/                   celegans-{worker,webhook,poller}.service.
  ├ install_systemd.sh         One-shot installer.
  └ backup_db.sh               SQLite online-backup via venv Python (6h cron).

data/real/held_out/              Xinyi's dataset drop (2026-04-15).
  └ evaluation_annotation_SEALED/
      ├ masks/sample_XXXX.npz   860 input masks.
      ├ ground_truth.npz        pose params (timepoint, center, angles, u).
      ├ ground_truth_masks/     857+/860 per-sample ref_mask + cell_names (uploading).
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
- **Sandbox:** `--network=none --read-only --cap-drop=ALL --security-opt=no-new-privileges --pids-limit=4096 --memory=32g --cpus=8 --gpus=all` + **120-min wall clock** (`EVAL_TIMEOUT=7200` in `.env`).
- **GPU:** RTX 4070 12 GB, host driver `570.x` (CUDA cap 12.8), nvidia-container-toolkit 1.18.x. GPU passthrough requires `ENABLE_GPU=1` in `.env` (already set on prod) **plus** `nvidia-smi` on PATH. Compatible with any participant container up to CUDA runtime 12.8. Verified base: `pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime`.

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

### Legacy scoring modules

`scoring/{registration,integration,combined,timepoint,cell_naming,combined_v2}.py`
are from earlier task formulations (pose regression, domain integration,
Sulston-name Hungarian matching). They're not wired into the live worker but
are kept for reference in case we want to reinstate any of them. Tracked in
`docs/TODO_PENDING.md`.

### Pending from collaborators

- **Real 4D atlas.** `data/reference_3d/volume_masks.npy` is currently a small
  placeholder. A real timepoint-indexed atlas (matching the SEALED eval
  namespace) is expected from the La Manno lab. The worker doesn't depend on
  it directly — participants bake it into their images — but the quickstart
  and README both promise it exists under that path.

### Reading list

- `docs/architecture.md` — how the components fit together and why.
- `docs/operator_guide.md` — everything needed to run the platform.
- `docs/deployment.md` — production deployment.
- `docs/contract_v2.md` — the current target contract.
- `docs/chat_backups/` — raw session transcripts (JSONL).
- `startingprompt.md` — original design spec.
