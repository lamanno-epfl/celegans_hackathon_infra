# Architecture

```
                       ┌───────────────┐
                       │   Team        │
                       │  docker push  │
                       └───────┬───────┘
                               │
                               ▼
                       ┌───────────────┐
                       │    Harbor     │  (self-hosted registry,
                       │   Registry    │   per-team projects +
                       └───────┬───────┘    robot accounts)
                               │ PUSH_ARTIFACT webhook
                               ▼
 ┌──────────────────────────────────────────────────┐
 │                Orchestrator API                  │
 │               (FastAPI, SQLite)                  │
 │  /webhook/harbor → validate quota, enqueue       │
 │  /api/leaderboard, /api/teams/{name}/submissions │
 └───────────────┬──────────────────────────────────┘
                 │ file queue (runtime/queue/*.json)
                 ▼
 ┌──────────────────────────────────────────────────┐
 │                    Worker                        │
 │                                                  │
 │  1. docker pull <image>                          │
 │  2. Prepare anonymized /input:                   │
 │     - Copy held-out simulated + real images      │
 │     - Shuffle, rename to NNNNN.npy               │
 │     - Keep {anon → original, kind} map OOB       │
 │     - Copy reference 3D volume                   │
 │     - Write manifest.json                        │
 │  3. docker run --read-only --network=none ...    │
 │  4. Validate /output (shapes, NaN, orthogonality)│
 │  5. Score:                                       │
 │     - Registration on simulated held-out         │
 │     - Integration on all embeddings              │
 │     - Combine with threshold gating              │
 │  6. Update DB, email results                     │
 │  7. Clean up scratch dir                         │
 └──────────────────────────────────────────────────┘
                 │
                 ▼
         ┌──────────────┐
         │ SMTP server  │ ──► participant inbox
         └──────────────┘
```

## Why this shape

### Anonymized, mixed input
Participants see one big pile of images with no idea which are simulated vs. real.
This prevents trivial "if real, output domain-invariant embedding" hacks — the model
must produce stable embeddings regardless of which domain an image comes from.

### Two-part scoring with threshold gate
We want registration quality **and** domain-invariance, not either in isolation:

- **Integration alone** can be gamed with constant or random embeddings (maximally
  "mixed" because classifier cannot learn). The **collapse check** catches the
  constant case; the **registration threshold** ensures that a model with no actual
  registration capability doesn't score high just because its random embeddings
  happen to be inseparable.
- **Registration alone** ignores whether the method transfers to real data — the
  entire scientific point of the competition.

### Two-sample test for integration
A logistic-regression classifier on cross-validated folds is:

- **Deterministic:** same embeddings → same score (important for leaderboard stability).
- **Cheap:** fast enough to run on every submission.
- **Principled:** scores 0.5 (= perfect integration) when the two distributions are
  actually indistinguishable — unlike silhouette-style metrics that reward geometric
  features that may not correspond to domain mixing.

### File queue + SQLite
Intentionally boring. For a few hundred submissions over a hackathon weekend, we do
not need Redis/Celery/Postgres. Swap upward only when the workload demands it.

### Security model
- `--network=none`: containers can't phone home for labels or exfiltrate data.
- `--read-only`: container FS is immutable, `/tmp` is a bounded tmpfs.
- `--memory=32g --cpus=8`: hard caps prevent noisy-neighbor and cost-burn.
- Robot accounts have push-only to their own project: teams can't see each other's code.

## Data flow summary

| Step | Who | What |
|---|---|---|
| 1 | Organizer | Splits data, configures Harbor, registers teams. |
| 2 | Team | Builds Docker image, pushes to Harbor. |
| 3 | Harbor | Fires webhook. |
| 4 | Orchestrator | Validates quota, enqueues submission, acks email. |
| 5 | Worker | Pulls image, prepares anonymized input, runs container (60 min max), validates, scores, emails. |
| 6 | Team | Receives score email. Can query API for history. |
| 7 | Public | Checks leaderboard. |
