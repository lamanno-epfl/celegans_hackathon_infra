# C. elegans Hackathon — Competition Infrastructure

End-to-end ML competition platform for the **3D → 2D registration + domain
adaptation** challenge. Participants submit a Docker image; our stack runs it in
an isolated sandbox, scores registration accuracy and cross-domain embedding
quality, and emails back the result. Built to be boring, auditable, and tamper-
resistant.

> **Status:** full pipeline working end-to-end against synthetic data. Three
> baseline models and three fake-participant examples execute through the Docker
> worker, validate, score, and land on the leaderboard. 30 unit tests pass.

---

## 1. The scientific problem in one paragraph

We have a high-quality **3D reference volume** (nuclei + membrane fluorescence,
with segmentation masks) from a reference lab. Two kinds of 2D slices exist:
*simulated* slices cut from the reference at known 6-DOF poses (ground truth
available), and *real* 2D sections acquired by a different lab (visually
different, no ground-truth poses). A successful model must (a) **register** a
2D slice into the 3D reference and (b) produce embeddings in which simulated
and real slices are **indistinguishable**, so the registration capability
actually transfers to real data.

## 2. Scoring in one paragraph

For each submission we compute:

```
registration_accuracy = mean over held-out simulated slices of
    0.5 · (1 − rotation_error_normalized)
  + 0.5 · (1 − translation_error_normalized)

integration_score = 1 − 2 · |classifier_accuracy − 0.5|
    where a 5-fold logistic regression tries to tell simulated from real
    embeddings; if embedding std < 1e-6 the score is forced to 0

final = registration_accuracy                                 if registration < 0.3
      = 0.8 · registration_accuracy + 0.2 · integration_score otherwise
```

Rotation error is the geodesic angle divided by π. Translation error is
Euclidean distance divided by the volume diameter. The threshold stops models
from gaming integration with random embeddings while registering poorly; the
collapse check stops the constant-embedding degenerate solution.

## 3. Pipeline in one picture

```
 team  ──docker push──►  Harbor ──webhook──►  Orchestrator ──queue──►  Worker
                                                   │                      │
                                                   ▼                      ▼
                                              SQLite DB             run container
                                                   │                 (read-only,
                                                   ▼                  no network,
                                             leaderboard              60-min cap)
                                                                          │
                                                                          ▼
                                                                     validate +
                                                                     score + email
```

See `docs/architecture.md` for the full version with design rationale.

## 4. Repository layout

```
scoring/          Pure Python/NumPy/sklearn. No GPU. Unit tested.
  ├ registration.py     Geodesic rotation + normalized translation errors.
  ├ integration.py      Deterministic two-sample domain classifier.
  ├ combined.py         Threshold gate + weighted combination.
  └ tests/              21 unit tests, incl. edge cases.

orchestrator/     FastAPI app, SQLAlchemy models, evaluation worker.
  ├ app.py              /webhook/harbor, /api/leaderboard, /api/teams/...
  ├ worker.py           Anonymize inputs, run container, validate, score.
  ├ validation.py       Output contract checks (shape, orthogonality, NaN...).
  ├ email_service.py    SMTP + Jinja2 templates; dry-run mode for dev.
  ├ queue.py            Simple file queue.
  ├ setup_harbor.py     Per-team Harbor project + robot account bootstrap.
  ├ templates/          6 email templates (text + HTML).
  └ tests/              9 unit tests covering webhook + validation.

baselines/        Reference submissions for calibration.
  ├ trivial/            ResNet-18 + 6D rotation, no domain adaptation.
  ├ domain_adapted/     + GRL + domain classifier.
  └ degenerate/         Random poses, constant embeddings (must score ≈ 0).

examples/         Fake-participant templates for end-to-end drills.
  ├ participant_template/     Copy this as a starting point.
  ├ random_participant/       Random poses + Gaussian embeddings.
  ├ identity_participant/     Identity pose + image-statistics embeddings.
  └ blur_participant/         Identity pose + blurred+normalized embeddings.

scripts/
  ├ generate_synthetic_data.py    Tiny 3D volume + simulated/real slices.
  ├ run_local_eval.py             Evaluate without Docker (fast loop).
  ├ smoke_test_worker.py          Full webhook → queue → worker → score path.
  ├ validate_container.py         Stand-alone container contract check.
  └ calibrate_threshold.py        Suggest REGISTRATION_THRESHOLD from baselines.

generate_splits.py    Deterministic 70/30 (sim) + 80/20 (real) stratified splits.
config.py             Central, env-overridable configuration.
Dockerfile            Orchestrator service image.
docker-compose.yml    Orchestrator + worker (Harbor installed separately).
docs/
  ├ architecture.md         Full diagram + design decisions.
  ├ operator_guide.md       Setup, env vars, runbook, calibration.
  └ participant_quickstart.md   The "hi I'm a participant" onboarding.
```

## 5. 60-second local demo

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_synthetic_data.py --n-simulated 80 --n-real 60
python generate_splits.py

# Fast inner-loop: scoring only, no Docker.
python baselines/trivial/train.py --public-dir data/simulated/public --epochs 3
python scripts/run_local_eval.py --baseline trivial --weights baselines/trivial/model.pt

# Full path with Docker:
docker build -f baselines/trivial/Dockerfile -t celegans/trivial:latest .
python scripts/validate_container.py --image celegans/trivial:latest
python scripts/smoke_test_worker.py --image celegans/trivial:latest
```

Expected from the last command:

```
project=celegans repo=trivial tag=latest
seeded team id=1
webhook response: {'status': 'queued', 'submission_id': 1}
dequeued submission 1
==== RESULT ====
status: completed
final_score: 0.47...
registration_score: 0.58...
integration_score: 0.06...
```

## 6. What each layer buys you

| Concern | How we handle it |
|---|---|
| **Cheating via internet lookup** | `docker run --network=none` — container is on a null network. |
| **Filesystem tampering** | `--read-only` root, only `/tmp` (10 GB tmpfs) and `/output` writable. |
| **Resource hogging** | Hard `--memory=32g --cpus=8` caps, 60-minute wall-clock kill. |
| **Team-A reads Team-B's code** | Harbor robot accounts scoped to a single project. |
| **Knowing which images are real/sim** | Worker shuffles + anonymizes filenames; mapping kept OOB. |
| **Scoring flakiness** | Fixed random seeds in both the classifier and the CV split. |
| **Lost submissions on crash** | Worker wraps evaluation in try/finally; terminal state always set. |
| **Validation-error waste of quota** | Validation failures do not decrement the team's counter. |
| **Degenerate submissions winning** | Threshold gate + embedding-collapse detection (std < 1e-6 ⇒ score 0). |

## 7. Baseline and fake-participant scores

Actually measured end-to-end on the synthetic dataset bundled here (60 real + 80
simulated slices, splits fixed at seed 0):

| Model | Registration | Integration | Final |
|---|---:|---:|---:|
| `baselines/degenerate` | 0.60 | **0.00** (collapse) | 0.48 |
| `baselines/trivial` | 0.58 | 0.06 | 0.47 |
| `baselines/domain_adapted` | 0.58 | 0.11 | 0.49 |
| `examples/random_participant` | 0.58 | 0.89 | 0.64 |
| `examples/identity_participant` | 0.59 | 0.00 | 0.47 |
| `examples/blur_participant` | 0.59 | 0.27 | 0.52 |

Two sanity observations to carry into real data:

- The degenerate baseline's integration score **is** 0 (collapse check fires), as
  required by the spec.
- On this synthetic dataset the threshold of 0.3 is well below typical
  registration accuracy, so calibration will really happen when real data lands.
  `scripts/calibrate_threshold.py` prints a suggested value from the baselines.

## 8. Tests

```bash
pytest -q          # 30 passed
```

Scoring has unit tests for the math (identity, 90°, 180° rotations; clipped and
scaled translations; collapse detection; deterministic classifier). Orchestrator
has tests for webhook routing, quota enforcement, and full output-validation
edge cases.

## 9. Next steps before going live

1. **Install Harbor** on the orchestrator host (official offline installer).
2. **Register teams** via `python -m orchestrator.setup_harbor --teams teams.json`.
3. **Configure SMTP** and set `SMTP_DRY_RUN=false`.
4. **Calibrate** `REGISTRATION_THRESHOLD` from baselines on the real data.
5. **Wire monitoring** — queue depth, failure rate, eval duration.

See `docs/operator_guide.md` for the full checklist with env-var reference and
runbook.

## 10. Reading list

- `docs/architecture.md` — how the components fit together and why.
- `docs/operator_guide.md` — everything needed to run the platform.
- `docs/participant_quickstart.md` — what a team has to do to submit.
- `startingprompt.md` — original design spec.
