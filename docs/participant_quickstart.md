# Participant Quickstart

This is everything you need to submit a model to the competition.

---

## 1. Register

Your organizer will email you a credentials JSON file that looks like:

```json
{
  "team": "myteam",
  "harbor_url": "https://registry.competition.org",
  "project": "myteam",
  "robot_name": "robot$myteam+push-myteam",
  "robot_secret": "<long secret>",
  "orchestrator_api_key": "<secret>"
}
```

Keep this file private.

## 2. Clone the template

```bash
cp -r examples/participant_template my_submission
cd my_submission
```

## 3. Implement your model

Open `predict.py`. The `predict_one(image, mask, reference)` function is called
once per input slice and must return:

- `rotation`: a 3×3 numpy array (orthogonal, det = +1)
- `translation`: a 3-vector (in voxel units of the 3D reference)
- `embedding`: a 1-D numpy array with dim between 64 and 2048

The `image` is a 2-channel array (nuclei, membrane). The `mask` is a 2-D integer
mask. `reference` gives you the full 3D volume (nuclei + membrane + masks) to
register against.

You can also rewrite `main()` if you need batch inference.

## 4. Smoke-test locally

Generate a tiny synthetic dataset once:

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py --n-simulated 60 --n-real 40
python generate_splits.py
```

Build + validate your image:

```bash
docker build -f my_submission/Dockerfile -t myteam/model:v1 .
python scripts/validate_container.py --image myteam/model:v1
# => VALIDATION OK
```

Run it through the full pipeline (scores included):

```bash
python scripts/smoke_test_worker.py --image myteam/model:v1
# => status=completed, final_score=..., registration_score=..., integration_score=...
```

## 5. Push to the competition registry

```bash
docker login <harbor_url> -u <robot_name> -p <robot_secret>
docker tag myteam/model:v1 <harbor_url>/<project>/model:v1
docker push         <harbor_url>/<project>/model:v1
```

The push fires a webhook to the orchestrator. You will receive:

1. `submission_received` email (with remaining quota).
2. `evaluation_started` email when your container starts running.
3. `evaluation_complete` email with the three scores, **or**
   `validation_error` / `evaluation_failed` if something went wrong.
   **Failures and validation errors do not count against your quota.**

## 6. Check your submission history

```bash
curl -H "X-API-Key: <orchestrator_api_key>" \
    https://orchestrator.competition.org/api/teams/myteam/submissions
```

Public leaderboard:

```bash
curl https://orchestrator.competition.org/api/leaderboard
```

---

## Container contract (reference)

- **Image entrypoint:** `/predict.sh` (make sure it's executable and at the image root).
- **Inputs (read-only at `/input`):**
  - `/input/images/*.npy` — 2-channel (nuclei, membrane) images
  - `/input/masks/*.npy` — integer segmentation masks, same filenames
  - `/input/reference_3d/volume_{nuclei,membrane,masks}.npy`
  - `/input/manifest.json` — list of image filenames, shuffled and anonymized
- **Outputs (writable at `/output`):**
  - `/output/poses.json` — `{filename: {rotation: 3x3, translation: 3}, ...}`
  - `/output/embeddings.npy` — `(N, D)` float array; row order = manifest order; D ∈ [64, 2048]
  - `/output/metadata.json` — must include `embedding_dim`
- **Resource limits:** 32 GB RAM, 8 CPUs, `--gpus all` if GPU node, no network,
  read-only FS with 10 GB tmpfs on `/tmp`, 60-minute wall-clock cap.
- **Important:** the input mixes simulated and real samples. Your model does **not**
  know which is which. Produce poses for every image regardless.

## Scoring summary

```
if registration_accuracy < 0.3:
    final = registration_accuracy
else:
    final = 0.8 * registration_accuracy + 0.2 * integration_score
```

- **Registration** is mean `0.5 * (1 - rotation_error) + 0.5 * (1 - translation_error)`
  on held-out simulated slices, where rotation error is the geodesic angle divided
  by π and translation error is the Euclidean distance divided by the volume diameter.
- **Integration** is `1 − 2 · |classifier_accuracy − 0.5|` where the classifier is
  a 5-fold cross-validated logistic regression on your embeddings with simulated
  vs. real domain labels. An embedding with std < 1e-6 scores 0 (collapse check).
