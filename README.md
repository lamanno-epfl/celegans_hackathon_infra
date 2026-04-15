# C. elegans Hackathon — Competition Infrastructure

End-to-end ML competition infrastructure for a 3D → 2D registration + domain-adaptation
challenge. Scoring is split between **registration accuracy** (pose regression on
held-out simulated slices) and **integration score** (domain-classifier two-sample test
on embeddings).

See [`startingprompt.md`](startingprompt.md) for the full design spec.

## Layout

```
scoring/          registration / integration / combined scoring + unit tests
orchestrator/     FastAPI app, DB models, worker, email service, harbor setup
baselines/        trivial, domain_adapted, degenerate (each with Dockerfile)
scripts/          synthetic data, local eval, container validation, calibration
generate_splits.py   produces public / held_out splits under data/
config.py         central config (env-overridable)
docker-compose.yml   orchestrator + worker (Harbor installed separately)
```

## Quick start (local, no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Create a small synthetic dataset
python scripts/generate_synthetic_data.py --n-simulated 80 --n-real 60
python generate_splits.py

# Run the degenerate baseline end-to-end
python scripts/run_local_eval.py --baseline degenerate

# Train + evaluate trivial baseline
python baselines/trivial/train.py --public-dir data/simulated/public --epochs 5
python scripts/run_local_eval.py --baseline trivial --weights baselines/trivial/model.pt

# Train + evaluate domain-adapted baseline
python baselines/domain_adapted/train.py \
    --sim-dir data/simulated/public --real-dir data/real/public --epochs 5
python scripts/run_local_eval.py --baseline domain_adapted \
    --weights baselines/domain_adapted/model.pt

# Calibrate the registration threshold from baseline results
python scripts/calibrate_threshold.py \
    --trivial-weights baselines/trivial/model.pt \
    --domain-weights baselines/domain_adapted/model.pt
```

## Tests

```bash
pytest scoring/tests orchestrator/tests -q
```

## Orchestrator service

```bash
# Start API
uvicorn orchestrator.app:app --host 0.0.0.0 --port 8000

# In another process, start the worker
python -m orchestrator.worker
```

Or via docker-compose (Harbor must be installed separately):

```bash
docker compose up --build
```

### Endpoints

- `POST /webhook/harbor` — receives Harbor push webhooks, queues a submission.
- `GET  /api/health`
- `GET  /api/leaderboard`
- `GET  /api/teams/{team_name}/submissions` — requires `X-API-Key` header.

### Harbor bootstrap

`orchestrator/setup_harbor.py` creates a Harbor project + robot account per team.
Pass a JSON file with `[{"name": "...", "email": "..."}]`.

```bash
python -m orchestrator.setup_harbor --teams teams.json --creds-dir runtime/credentials
```

## Participant contract

A submission is a Docker image exposing `/predict.sh`. Inputs are mounted read-only at
`/input` (with `images/`, `masks/`, `reference_3d/`, and `manifest.json`); outputs must
be written to `/output/poses.json`, `/output/embeddings.npy`, `/output/metadata.json`.
Images in `manifest.json` mix simulated and real samples; the model does not know which
is which. See [`startingprompt.md` §4](startingprompt.md) for the full contract.

## Configuration

Everything is configurable via env vars — see `config.py` for the full list. Important
ones: `REGISTRATION_THRESHOLD`, `REGISTRATION_WEIGHT`, `INTEGRATION_WEIGHT`, `HARBOR_URL`,
`SMTP_*`, `DATABASE_URL`, `DATA_ROOT`, `EVAL_TIMEOUT`.

## Scoring formula

```
if registration_accuracy < REGISTRATION_THRESHOLD:
    final = registration_accuracy
else:
    final = 0.8 * registration_accuracy + 0.2 * integration_score
```

- `registration_accuracy = mean over held-out simulated slices of
    0.5 * (1 - norm_rotation_error) + 0.5 * (1 - norm_translation_error)`
- `integration_score = 1 - 2 * |classifier_accuracy - 0.5|` where the classifier is a
  5-fold logistic regression on embeddings with domain labels.
- Collapse detection: `integration_score = 0` if embedding std < 1e-6.
