# Competition Infrastructure Implementation Prompt

You are implementing the full infrastructure for a machine learning competition. Read this entire document carefully before writing any code. This describes the scientific context, the evaluation design, the submission pipeline, and the orchestration system. Implement everything.

## 1. Scientific Context

This competition involves 3D biological imaging data from fluorescence microscopy. There are two channels: nuclei and membranes. The data includes:

- **A 3D reference volume** with segmentation masks (discrete labeled regions identifying individual cells/structures). This comes from a reference laboratory.
- **Simulated 2D slices** extracted from the 3D reference at known poses (rotation + translation). These have both fluorescence images and corresponding 2D segmentation masks. Ground truth poses are known.
- **Real 2D sections** from a different laboratory. These also have fluorescence images (nuclei + membranes) and segmentation masks, but **no ground truth poses** — we don't know where in the 3D reference they correspond. Visually, these look very different from the simulated data (domain gap).

### The Two Tasks

**Task 1 — Registration:** Given a 2D slice (simulated), predict its 6-DOF pose (3D rotation + 3D translation) relative to the 3D reference. This is supervised — ground truth is available for simulated slices.

**Task 2 — Domain Adaptation:** The model must produce representations where simulated and real data are indistinguishable. This ensures the registration capability transfers to real data, even though we can't directly evaluate registration on real data.

## 2. Evaluation Design

### 2.1 Score Formula

```
if registration_accuracy < THRESHOLD:
    final_score = registration_accuracy
else:
    final_score = 0.8 * registration_accuracy + 0.2 * integration_score
```

- `THRESHOLD` is a configurable minimum (start with 0.3, calibrate with baselines).
- `registration_accuracy` evaluates pose prediction on held-out simulated slices.
- `integration_score` evaluates domain mixing quality on embeddings.

### 2.2 Registration Accuracy (Part 1)

Participants' models receive 2D images and predict poses. Compare predicted vs. ground truth:

- **Rotation error:** geodesic distance between predicted and GT rotation matrices. Normalize to [0, 1] where 0 = 180° error, 1 = perfect.
- **Translation error:** Euclidean distance between predicted and GT translation vectors. Normalize by the diameter of the 3D volume to [0, 1].
- `registration_accuracy = 0.5 * (1 - norm_rotation_error) + 0.5 * (1 - norm_translation_error)`, averaged over all held-out slices.

Implement the geodesic distance for rotation matrices as: `arccos((trace(R_pred @ R_gt.T) - 1) / 2) / π`.

### 2.3 Integration Score (Part 2)

Uses a **domain classifier two-sample test**:

1. Collect embeddings from the participant's model for both simulated and real images.
2. Train a simple binary classifier (logistic regression or small MLP) to distinguish "simulated" vs. "real" embeddings. Use k-fold cross-validation (k=5) to get a robust accuracy estimate.
3. Compute: `integration_score = 1 - 2 * |classifier_accuracy - 0.5|`
   - Score = 1.0 when classifier accuracy ≈ 0.5 (can't distinguish → good integration)
   - Score = 0.0 when classifier accuracy ≈ 1.0 or 0.0 (perfect separation → poor integration)

**Important:** The classifier must be deterministic (fix random seeds) so the same embeddings always produce the same score. Use sklearn's LogisticRegression with fixed random_state and StratifiedKFold.

### 2.4 Degenerate Solution Prevention

- The threshold mechanism prevents trivially mixed but useless embeddings from scoring well.
- Additionally, check embedding variance: if the standard deviation of all embeddings (across all samples) is below a minimum threshold (e.g., 1e-6), set `integration_score = 0`. This catches constant-embedding collapse.

## 3. Data Splits and Held-Out Design

### 3.1 Directory Structure for Data

```
data/
├── reference_3d/                  # The 3D volume and its masks
│   ├── volume_nuclei.npy          # 3D fluorescence (nuclei channel)
│   ├── volume_membrane.npy        # 3D fluorescence (membrane channel)
│   └── volume_masks.npy           # 3D segmentation masks (integer labels)
├── simulated/
│   ├── public/                    # Given to participants
│   │   ├── images/                # 2D slices (nuclei + membrane as 2-channel images)
│   │   ├── masks/                 # Corresponding 2D masks
│   │   └── poses.json             # GT poses: {filename: {rotation: 3x3, translation: 3x1}}
│   └── held_out/                  # Never seen by participants
│       ├── images/
│       ├── masks/
│       └── poses.json
├── real/
│   ├── public/                    # Given to participants for domain adaptation training
│   │   ├── images/
│   │   └── masks/
│   └── held_out/                  # Never seen, used for integration scoring
│       ├── images/
│       └── masks/
└── splits.json                    # Records which files are public vs held-out
```

### 3.2 Split Generation Script

Create a script `generate_splits.py` that:

1. Takes the full data directories as input.
2. Splits simulated data: 70% public, 30% held-out. Stratify by pose orientation (bin rotations into octants of SO(3) and ensure proportional representation).
3. Splits real data: 80% public, 20% held-out. Use random stratified sampling; verify that held-out set covers the range of visual variability (log statistics: mean intensity, mask count distribution).
4. Writes `splits.json` recording the assignments.
5. Physically moves files into public/held_out subdirectories.

## 4. Participant Submission Interface

### 4.1 Docker Container Contract

Participants submit a Docker image that exposes a single script: `/predict.sh` (or `/predict.py`).

**The container is invoked as:**

```bash
docker run --gpus all --network=none \
    --memory=32g --memory-swap=32g \
    --cpus=8 \
    -v /path/to/input:/input:ro \
    -v /path/to/output:/output \
    --read-only --tmpfs /tmp:size=10g \
    participant-image:tag \
    /predict.sh
```

**Input directory structure (mounted read-only at /input):**

```
/input/
├── reference_3d/
│   ├── volume_nuclei.npy
│   ├── volume_membrane.npy
│   └── volume_masks.npy
├── images/          # Mix of simulated and real images, unlabeled
│   ├── 00001.npy    # 2-channel images (nuclei + membrane)
│   ├── 00002.npy
│   └── ...
├── masks/           # Corresponding masks
│   ├── 00001.npy
│   ├── 00002.npy
│   └── ...
└── manifest.json    # Lists all image filenames, nothing more
```

**Critical:** The input mixes simulated and real images together. The participant's model does NOT know which is which. File order is shuffled and filenames are anonymized.

**Output directory structure (writable at /output):**

```
/output/
├── poses.json       # {filename: {rotation: [[3x3]], translation: [3x1]}} for ALL images
│                    # (participants predict poses for everything, even real images)
├── embeddings.npy   # Shape: (N, D) where N = number of images, D = embedding dim
│                    # D must be between 64 and 2048
└── metadata.json    # {embedding_dim: D, model_name: "...", notes: "..."}
```

### 4.2 Container Validation

Before running the full evaluation, validate the container output:

1. Check `/output/poses.json` exists, is valid JSON, has entries for all filenames in manifest.
2. Check each pose has a 3x3 rotation matrix (verify orthogonality: R @ R.T ≈ I, det(R) ≈ 1) and a 3x1 translation vector.
3. Check `/output/embeddings.npy` exists, shape is (N, D) with correct N, D in [64, 2048].
4. Check `/output/metadata.json` exists and has `embedding_dim`.
5. Check no NaN or Inf values in poses or embeddings.

If validation fails, abort and report the specific error to the participant.

## 5. Container Registry and Submission Pipeline

### 5.1 Registry Setup

Use **Harbor** as a self-hosted Docker registry.

- Install Harbor on the orchestrator machine (can be separate from GPU machine).
- Create one Harbor **project** per team, with push access only for that team.
- Teams authenticate via robot accounts (generated at registration).
- Image naming convention: `registry.competition.org/{team_name}/model:{tag}`

Set up Harbor with these configurations:

```yaml
# harbor.yml adjustments
hostname: registry.competition.org
harbor_admin_password: <generate-secure-password>
database:
  password: <generate-secure-password>
```

Create a script `setup_harbor.py` that:

1. Uses Harbor's API to create a project per team.
2. Creates a robot account per team with push-only access to their project.
3. Stores credentials in a database.
4. Generates per-team credential files to distribute.

### 5.2 Orchestrator Service

Build a **FastAPI** application (`orchestrator/`) that manages the entire evaluation lifecycle.

**Database (SQLite via SQLAlchemy):**

```
teams:
    id, name, email, harbor_project, max_submissions (default=10), created_at

submissions:
    id, team_id, image_tag, status (queued/running/completed/failed/validation_error),
    registration_score, integration_score, final_score,
    error_message, submitted_at, started_at, completed_at

evaluation_logs:
    id, submission_id, log_text, created_at
```

**Endpoints:**

- `POST /webhook/harbor` — receives Harbor webhook when an image is pushed. Extracts team name from image path, creates a submission record with status=queued. Harbor webhook payload includes the repository name and tag.
- `GET /api/teams/{team_name}/submissions` — returns submission history for a team (authenticated via API key).
- `GET /api/leaderboard` — public leaderboard showing best score per team.
- `GET /api/health` — health check.

**Webhook handler logic:**

```python
@app.post("/webhook/harbor")
async def harbor_webhook(payload: dict):
    # 1. Extract team_name and image_tag from payload
    repo_name = payload["event_data"]["repository"]["repo_full_name"]
    tag = payload["event_data"]["resources"][0]["tag"]
    team_name = repo_name.split("/")[1]  # registry.competition.org/{project}/{repo}

    # 2. Look up team in DB
    team = db.query(Team).filter_by(harbor_project=team_name).first()
    if not team:
        return {"error": "Unknown team"}

    # 3. Count existing submissions (completed + queued + running)
    used = db.query(Submission).filter(
        Submission.team_id == team.id,
        Submission.status.in_(["queued", "running", "completed"])
    ).count()

    if used >= team.max_submissions:
        # Send email: no remaining submissions
        send_email(team.email, "submission_limit_reached", {})
        return {"error": "Submission limit reached"}

    # 4. Create submission record
    submission = Submission(
        team_id=team.id,
        image_tag=f"{repo_name}:{tag}",
        status="queued"
    )
    db.add(submission)
    db.commit()

    # 5. Queue evaluation job
    queue_evaluation(submission.id)

    # 6. Send confirmation email
    remaining = team.max_submissions - used - 1
    send_email(team.email, "submission_received", {
        "submission_id": submission.id,
        "remaining": remaining,
        "tag": tag
    })

    return {"status": "queued", "submission_id": submission.id}
```

### 5.3 Evaluation Worker

A background worker process (use `asyncio` with a simple queue, or Celery/RQ if you prefer; keep it simple with asyncio + a file-based queue for now) that:

1. **Pulls the image** from Harbor to the GPU machine. If the orchestrator and GPU machine are different hosts, use SSH + docker pull. For simplicity, assume same machine initially.

2. **Prepares the input directory:**
   - Copies held-out simulated images + real held-out images into a temporary directory.
   - Shuffles and renames to anonymized filenames (e.g., 00001.npy, 00002.npy, ...).
   - Keeps a mapping file (NOT in the input dir) recording which anonymized name is which original file and whether it's simulated or real.
   - Writes `manifest.json` listing all anonymized filenames.
   - Includes the 3D reference volume.

3. **Runs the container** with the security constraints from section 4.1. Sets a wall-clock timeout of 60 minutes. Captures stdout/stderr.

4. **Validates outputs** per section 4.2.

5. **Scores:**
   - De-anonymizes outputs using the mapping.
   - Computes `registration_accuracy` on simulated held-out samples.
   - Computes `integration_score` on all embeddings (using the simulated/real labels from the mapping).
   - Computes `final_score` per the formula.

6. **Updates DB** with scores and status.

7. **Sends results email** with: final_score, registration_accuracy, integration_score, remaining submissions. Does NOT include per-sample scores.

8. **Cleans up:** removes the temporary input/output directories and optionally prunes the Docker image.

Implement this as `orchestrator/worker.py`.

### 5.4 Email Service

Use a simple SMTP email sender. Create email templates (plain text + HTML) for:

- `submission_received`: "We received your submission. You have {remaining} submissions left."
- `evaluation_started`: "Your submission is now being evaluated."
- `evaluation_complete`: "Results: Final={final_score:.4f}, Registration={reg:.4f}, Integration={int:.4f}. Remaining: {remaining}."
- `evaluation_failed`: "Your submission failed: {error_message}. This does NOT count against your submission limit."
- `submission_limit_reached`: "You have used all {max} submissions."
- `validation_error`: "Your container output failed validation: {details}. This does NOT count against your submission limit."

**Important:** Failed submissions and validation errors do NOT decrement the counter. Only successful evaluations count.

Implement as `orchestrator/email_service.py`. Use Jinja2 for templates stored in `orchestrator/templates/`.

## 6. Scoring Implementation

### 6.1 File: `scoring/registration.py`

```python
import numpy as np
from typing import Dict, Tuple

def geodesic_rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """Geodesic distance between two rotation matrices, normalized to [0, 1]."""
    R_diff = R_pred @ R_gt.T
    trace_val = np.clip(np.trace(R_diff), -1.0, 3.0)
    angle = np.arccos(np.clip((trace_val - 1) / 2, -1.0, 1.0))
    return angle / np.pi  # 0 = identical, 1 = 180 degrees

def translation_error(t_pred: np.ndarray, t_gt: np.ndarray, volume_diameter: float) -> float:
    """Euclidean translation error, normalized by volume diameter to [0, 1]."""
    return np.clip(np.linalg.norm(t_pred - t_gt) / volume_diameter, 0.0, 1.0)

def compute_registration_accuracy(
    predictions: Dict[str, dict],
    ground_truth: Dict[str, dict],
    volume_diameter: float
) -> Tuple[float, dict]:
    """
    predictions: {filename: {rotation: 3x3 list, translation: 3x1 list}}
    ground_truth: same format
    Returns: (accuracy_score, per_sample_details)
    """
    scores = {}
    for fname, gt in ground_truth.items():
        if fname not in predictions:
            scores[fname] = 0.0
            continue
        pred = predictions[fname]
        R_pred = np.array(pred["rotation"])
        R_gt = np.array(gt["rotation"])
        t_pred = np.array(pred["translation"])
        t_gt = np.array(gt["translation"])

        rot_err = geodesic_rotation_error(R_pred, R_gt)
        trans_err = translation_error(t_pred, t_gt, volume_diameter)
        score = 0.5 * (1 - rot_err) + 0.5 * (1 - trans_err)
        scores[fname] = score

    accuracy = np.mean(list(scores.values()))
    return float(accuracy), scores
```

### 6.2 File: `scoring/integration.py`

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from typing import Tuple

RANDOM_SEED = 42

def check_embedding_collapse(embeddings: np.ndarray, threshold: float = 1e-6) -> bool:
    """Returns True if embeddings have collapsed (degenerate)."""
    return np.std(embeddings) < threshold

def domain_classifier_accuracy(
    embeddings: np.ndarray,
    labels: np.ndarray,  # 0 = simulated, 1 = real
    n_folds: int = 5
) -> float:
    """
    Train a logistic regression classifier to distinguish domains.
    Returns mean cross-validated accuracy.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    scaler = StandardScaler()
    accuracies = []

    for train_idx, test_idx in skf.split(embeddings, labels):
        X_train = scaler.fit_transform(embeddings[train_idx])
        X_test = scaler.transform(embeddings[test_idx])
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=1000,
            solver="lbfgs"
        )
        clf.fit(X_train, y_train)
        accuracies.append(clf.score(X_test, y_test))

    return float(np.mean(accuracies))

def compute_integration_score(
    embeddings: np.ndarray,
    domain_labels: np.ndarray
) -> Tuple[float, dict]:
    """
    Returns: (integration_score, details)
    """
    if check_embedding_collapse(embeddings):
        return 0.0, {"reason": "embedding_collapse", "classifier_accuracy": None}

    clf_acc = domain_classifier_accuracy(embeddings, domain_labels)
    score = 1.0 - 2.0 * abs(clf_acc - 0.5)

    return float(score), {
        "classifier_accuracy": clf_acc,
        "collapse_detected": False
    }
```

### 6.3 File: `scoring/combined.py`

```python
from scoring.registration import compute_registration_accuracy
from scoring.integration import compute_integration_score
import numpy as np
import json
from typing import Tuple

REGISTRATION_WEIGHT = 0.8
INTEGRATION_WEIGHT = 0.2
REGISTRATION_THRESHOLD = 0.3  # Calibrate with baselines

def compute_final_score(
    poses_pred: dict,
    poses_gt: dict,
    embeddings: np.ndarray,
    domain_labels: np.ndarray,
    volume_diameter: float,
    simulated_filenames: list,
    filename_order: list  # order matching embeddings rows
) -> Tuple[float, dict]:
    """
    Main scoring entry point.

    poses_pred: predicted poses for all images
    poses_gt: ground truth poses for simulated held-out images only
    embeddings: (N, D) array, one row per image in filename_order
    domain_labels: (N,) array, 0=simulated, 1=real, matching filename_order
    volume_diameter: normalization constant
    simulated_filenames: list of held-out simulated filenames (subset of filename_order)
    filename_order: list of all filenames in the order they appear in embeddings
    """
    # Part 1: Registration (only on simulated held-out)
    sim_preds = {f: poses_pred[f] for f in simulated_filenames if f in poses_pred}
    reg_accuracy, reg_details = compute_registration_accuracy(
        sim_preds, poses_gt, volume_diameter
    )

    # Part 2: Integration (on all embeddings)
    int_score, int_details = compute_integration_score(embeddings, domain_labels)

    # Combined score
    if reg_accuracy < REGISTRATION_THRESHOLD:
        final = reg_accuracy
        formula_used = "registration_only (below threshold)"
    else:
        final = REGISTRATION_WEIGHT * reg_accuracy + INTEGRATION_WEIGHT * int_score
        formula_used = "weighted_combination"

    details = {
        "final_score": final,
        "registration_accuracy": reg_accuracy,
        "integration_score": int_score,
        "formula_used": formula_used,
        "threshold": REGISTRATION_THRESHOLD,
        "weights": {"registration": REGISTRATION_WEIGHT, "integration": INTEGRATION_WEIGHT},
        "registration_details": reg_details,
        "integration_details": int_details
    }

    return final, details
```

## 7. Baseline Models

Implement three baselines to validate the scoring pipeline. These are simple and NOT meant to be competitive.

### 7.1 Trivial Baseline (`baselines/trivial/`)

- A ResNet-18 (pretrained on ImageNet, swap first conv to accept 2-channel input).
- Pose regression head: Global average pooling → Linear(512, 6) for rotation (6D continuous representation, convert to matrix via Gram-Schmidt) + Linear(512, 3) for translation.
- Embedding: the 512-dim vector after global average pooling.
- Train on public simulated data only, no domain adaptation.
- Expected: decent Part 1, poor Part 2.

### 7.2 Domain Adaptation Baseline (`baselines/domain_adapted/`)

- Same as trivial, but add a gradient reversal layer (GRL) + domain classifier head during training.
- The domain classifier tries to distinguish simulated vs. real embeddings; GRL reverses gradients so the encoder learns domain-invariant features.
- Train on public simulated (with pose labels) + public real (unlabeled, for domain adversarial loss only).
- Expected: slightly lower Part 1, much better Part 2.

### 7.3 Degenerate Baseline (`baselines/degenerate/`)

- Outputs random poses and constant embeddings (all zeros or all ones).
- Expected: terrible Part 1, integration_score = 0 (caught by collapse detection).

Each baseline should include a `Dockerfile` following the container contract in section 4.1, a `train.py` (except degenerate), and `predict.sh`.

## 8. Project Structure

```
competition/
├── README.md                      # Overview of the whole system
├── docker-compose.yml             # Harbor + orchestrator + worker
├── config.py                      # All configurable constants (thresholds, weights, paths, SMTP, etc.)
├── generate_splits.py             # Data split generation
├── scoring/
│   ├── __init__.py
│   ├── registration.py
│   ├── integration.py
│   ├── combined.py
│   └── tests/
│       ├── test_registration.py   # Unit tests with known inputs/outputs
│       ├── test_integration.py
│       └── test_combined.py
├── orchestrator/
│   ├── __init__.py
│   ├── app.py                     # FastAPI app
│   ├── models.py                  # SQLAlchemy models
│   ├── worker.py                  # Evaluation worker
│   ├── email_service.py
│   ├── templates/
│   │   ├── submission_received.html
│   │   ├── evaluation_complete.html
│   │   ├── evaluation_failed.html
│   │   ├── submission_limit_reached.html
│   │   └── validation_error.html
│   └── setup_harbor.py            # Harbor project/robot account setup
├── baselines/
│   ├── trivial/
│   │   ├── Dockerfile
│   │   ├── train.py
│   │   ├── model.py
│   │   └── predict.sh
│   ├── domain_adapted/
│   │   ├── Dockerfile
│   │   ├── train.py
│   │   ├── model.py
│   │   └── predict.sh
│   └── degenerate/
│       ├── Dockerfile
│       └── predict.sh
├── scripts/
│   ├── run_local_eval.py          # Run scoring locally for testing (no Docker)
│   ├── validate_container.py      # Test a container locally before deploying
│   └── calibrate_threshold.py     # Run all baselines, suggest threshold values
└── requirements.txt
```

## 9. Implementation Notes

- Use Python 3.11+. Type hints everywhere.
- Use PyTorch for baselines. NumPy + sklearn for scoring (keep scoring lightweight, no GPU needed).
- All configurable values (thresholds, weights, paths, SMTP settings, Harbor URL, etc.) go in `config.py` with environment variable overrides.
- Write unit tests for all scoring functions. Include edge cases: NaN embeddings, wrong shapes, non-orthogonal rotation matrices, missing filenames.
- The orchestrator should be robust: if the worker crashes, the submission status should be updated to "failed" (use try/finally). Never leave a submission stuck in "running."
- Log everything. Every evaluation step should write to the evaluation_logs table.
- For the 6D rotation representation in baselines, implement the Zhou et al. (2019) "On the Continuity of Rotation Representations" conversion (6D → 3x3 matrix via Gram-Schmidt).

## 10. What to Implement Now

Implement everything in this document. Start with:

1. Project structure and config
2. Scoring modules + tests
3. Orchestrator (FastAPI + DB + webhook + worker)
4. Email service
5. Container validation
6. Data split generation
7. Baselines (trivial first, then domain adapted, then degenerate)
8. Docker compose for the whole stack
9. Scripts for local testing and threshold calibration
10. README with setup and usage instructions

Do not use placeholder implementations. Every function should be complete and working. For Harbor setup, implement the API calls but note in comments that Harbor must be installed separately.

For anything requiring actual data files (which don't exist yet), use synthetic data generation in tests — create small random 3D volumes, generate slices at known poses, etc. This way the full pipeline can be tested end-to-end without real data.