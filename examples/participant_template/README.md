# Participant Template

Minimal working submission. Copy this directory, edit `predict.py`, build, push.

## Contract recap

- Read: `/input/images/*.npy` (2-channel HxW), `/input/masks/*.npy`,
  `/input/reference_3d/{volume_nuclei,volume_membrane,volume_masks}.npy`,
  `/input/manifest.json` (list of filenames).
- Write: `/output/poses.json`, `/output/embeddings.npy`, `/output/metadata.json`.
- Embedding dim ∈ [64, 2048]. Matrix rotations must be orthogonal with det=+1.
- No internet access (`--network=none`), read-only FS, /tmp is 10 GB tmpfs.
- 60-min wall-clock limit.

## Build + smoke-test locally

```bash
# From the repo root:
docker build -f examples/participant_template/Dockerfile -t myteam/model:v1 .
python scripts/validate_container.py --image myteam/model:v1
```

## Submit

```bash
docker login registry.competition.org -u robot-name -p robot-secret
docker tag myteam/model:v1 registry.competition.org/myteam/model:v1
docker push registry.competition.org/myteam/model:v1
```

The Harbor webhook notifies the orchestrator. You'll get an email when evaluation
starts and when it finishes.
