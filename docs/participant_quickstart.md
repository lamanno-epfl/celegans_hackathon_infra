# Participant Quickstart

Everything you need to submit a model. Tested end-to-end on macOS (Intel + Apple Silicon) and Linux; Windows works via WSL2 (see section 0).

> **You must be on the EPFL network or EPFL VPN to reach the orchestrator.** Off-campus? Connect to EPFL VPN first (instructions at https://www.epfl.ch/campus/services/ip-network/vpn/). The orchestrator is only reachable from the EPFL internal network.

## Your credentials

- **Team name:** `lemanichack`
- **Bearer token:** `WmQstelLIxJLiFf1_59Z_TZfg73Resn-0EZH12utezw`
- **Orchestrator URL:** `http://128.178.188.212:8000`
- **Notification email:** `luca.fusarbassini@epfl.ch`

Keep the bearer token private. Anyone with it can submit as your team and burn your quota (50 submissions).

---

## 0. Prerequisites per OS

- **macOS:** install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or `brew install --cask docker`). **Launch it** (`open -a Docker`) and wait ~30 s until the whale icon in the menu bar stops animating. Verify: `docker info`.
- **Windows:** Docker Desktop requires WSL2. Install Docker Desktop for Windows (enable the WSL2 backend during setup), then **run all commands below from inside an Ubuntu WSL shell**, not PowerShell or cmd. Verify: `wsl -d Ubuntu -- docker info`. Pure PowerShell/Windows-native Docker is not supported by this guide; the `sed`, `curl --data-binary @file`, and line-editing conventions below assume a POSIX shell.
- **Linux:** install Docker Engine + the NVIDIA Container Toolkit if you plan to use GPUs locally. Your user must be in the `docker` group (`sudo usermod -aG docker $USER`, then re-login).

All OSes: need `git` and a POSIX shell (`bash` or `zsh`).

Sanity check the server is reachable (run this from your build machine — inside WSL on Windows):

```bash
curl http://128.178.188.212:8000/api/health
# expect: {"status":"ok","queue_size":0}
```

If that hangs or gives "connection refused": you're **off the EPFL network/VPN**, or a local firewall blocks port 8000. Connect to VPN and retry before doing anything else.

## 1. Get the repo and set up a submission

```bash
git clone https://github.com/lamanno-epfl/celegans_hackathon_infra
cd celegans_hackathon_infra
cp -r examples/participant_template my_submission
```

Patch the Dockerfile so it references `my_submission/` instead of the template path:

```bash
# Linux:
sed -i 's|examples/participant_template|my_submission|g' my_submission/Dockerfile
# macOS (the empty '' after -i is required):
sed -i '' 's|examples/participant_template|my_submission|g' my_submission/Dockerfile
```

## 2. Implement your model

Open `my_submission/predict.py`. `predict_one(image, mask, reference)` is called once per input slice and must return:

- `rotation`: 3×3 numpy array, orthogonal, det = +1
- `translation`: 3-vector (voxel units of the 3D reference)
- `embedding`: 1-D numpy array with `64 ≤ dim ≤ 2048`

`image` is a 2-channel `(nuclei, membrane)` array. `mask` is a 2-D integer mask. `reference` gives you the full 3D volume (`nuclei`, `membrane`, `masks`). You can rewrite `main()` if you prefer batch inference.

The template as-shipped already satisfies the output contract — you can submit it first to smoke-test the round-trip, then iterate.

## 3. Build the Docker image

The server runs **linux/amd64**. If you are on **Apple Silicon (M1/M2/M3) or any ARM machine**, you must pass `--platform linux/amd64` — otherwise the server will reject your image with `exec format error`.

```bash
# Apple Silicon / ARM:
docker build --platform linux/amd64 -f my_submission/Dockerfile -t lemanichack/mymodel:v1 .

# Intel Mac / Linux x86_64:
docker build -f my_submission/Dockerfile -t lemanichack/mymodel:v1 .
```

Two easy-to-miss details:

- **Trailing `.`** at the end (that's the build context — without it `docker build` refuses with "requires 1 argument").
- **`lemanichack/`** prefix in the tag must match your team exactly. The server identifies your submission by that first path segment.

### Using the GPU

The evaluation server has one **NVIDIA RTX 4070 (12 GB)** with the NVIDIA Container Toolkit installed. Your container runs with `--gpus all` when `ENABLE_GPU=1` is set on the server (currently: yes).

To use it:
- Base your image on a CUDA image, e.g. `nvidia/cuda:12.4.1-runtime-ubuntu22.04`.
- Install a CUDA-matching PyTorch wheel (e.g. `pip install --index-url https://download.pytorch.org/whl/cu124 torch`).
- Inside your code: `torch.cuda.is_available()` must be `True`.
- **Build with `--platform linux/amd64`** on Apple Silicon as usual; the GPU at runtime is on the server, not your laptop.

### (optional) Smoke-test locally before uploading

```bash
pip install -r requirements.txt
python scripts/generate_synthetic_data.py --n-simulated 60 --n-real 40
python generate_splits.py
python scripts/validate_container.py --image lemanichack/mymodel:v1
# => VALIDATION OK
```

## 4. Save + upload

```bash
docker save lemanichack/mymodel:v1 | gzip > mymodel.tar.gz
ls -lh mymodel.tar.gz    # sanity check: CUDA images are often 2–5 GB, plain Python models tens of MB

# Streaming upload (-T streams from disk; use this for anything over ~500 MB)
curl -X POST http://128.178.188.212:8000/api/upload -H "Authorization: Bearer WmQstelLIxJLiFf1_59Z_TZfg73Resn-0EZH12utezw" -H "Expect:" -T mymodel.tar.gz
# => {"status":"received","bytes":...,"path":"<team>-<timestamp>.tar.gz"}
```

**Two easy-to-hit gotchas here:**

- **Use `-T file`, not `--data-binary @file`.** `--data-binary` loads the whole tarball into RAM — for a 4 GB CUDA image curl will die with `option --data-binary: out of memory`. `-T` streams from disk and works for any size up to the 8 GB server cap.
- **Keep the whole command on one line.** zsh (macOS default) often breaks on `\` line continuations if there are trailing spaces — symptom: `command not found: -H` followed by `{"detail":"missing bearer token"}`.

Size cap: **8 GB** compressed.

## 5. Watch your inbox

Sender address: **`newsletter@paperboatch.com`**. **Check your spam folder the first time.** Mark it "not spam" / add to contacts so follow-ups reach you.

You'll receive, in order (total wall-clock ~1–5 minutes for a quick model):

1. `submission_received` — confirms the upload landed + remaining quota.
2. `evaluation_started` — the container is running.
3. One of:
   - `evaluation_complete` — with `final_score`, `registration_score`, `integration_score`.
   - `validation_error` — your outputs didn't match the contract. **Does not count against quota.**
   - `evaluation_failed` — container crashed, timed out, or returned non-zero. **Does not count against quota.**

If you hit your quota, you'll get `submission_limit_reached`. Quota is per team.

---

## Container contract (reference)

- **Entrypoint:** `/predict.sh` (executable, at image root — the template already handles this).
- **Inputs (read-only at `/input`):**
  - `/input/images/*.npy` — 2-channel `(nuclei, membrane)` images
  - `/input/masks/*.npy` — integer segmentation masks, same filenames
  - `/input/reference_3d/volume_{nuclei,membrane,masks}.npy`
  - `/input/manifest.json` — list of image filenames, shuffled and anonymized
- **Outputs (writable at `/output`):**
  - `/output/poses.json` — `{filename: {rotation: 3x3, translation: 3}, ...}`
  - `/output/embeddings.npy` — `(N, D)` float array; row order = manifest order; D ∈ [64, 2048]
  - `/output/metadata.json` — must include `embedding_dim`
- **Runtime resources:** 32 GB RAM, 8 CPUs, `--gpus all` if the node has a GPU, no network, read-only root FS with 10 GB tmpfs on `/tmp`, **60-minute wall-clock cap**.
- **Sandbox flags:** `--network=none`, `--read-only`, `--cap-drop=ALL`, `--security-opt=no-new-privileges`, `--pids-limit=4096`. **No internet at runtime** — download any model weights at build time and `COPY` them into the image.
- **Important:** the input mixes simulated and real slices. Your model does not know which is which. Produce poses for every image regardless.

## Scoring summary

```
if registration_accuracy < 0.3:
    final = registration_accuracy
else:
    final = 0.8 * registration_accuracy + 0.2 * integration_score
```

- **Registration** = mean of `0.5·(1 − rotation_error) + 0.5·(1 − translation_error)` on held-out simulated slices, where rotation error is the geodesic angle / π and translation error is Euclidean distance / volume diameter.
- **Integration** = `1 − 2·|classifier_accuracy − 0.5|`, where a 5-fold cross-validated logistic regression classifies embeddings as simulated vs real. Embedding with `std < 1e-6` is declared collapsed and scores 0.

---

## Troubleshooting — issues we've seen

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop not running | `open -a Docker`, wait ~30 s |
| `docker build ... requires 1 argument` | Missing trailing `.` | Add `.` at end of command |
| `command not found: -H` + `missing bearer token` | zsh broke your `\` line continuations | Put the whole `curl` on one line |
| `exec /predict.sh: exec format error` (container exit 255) | Built an ARM image on Apple Silicon | Rebuild with `--platform linux/amd64` |
| `413 upload exceeds limit` | Image > 8 GB | Trim the image (use a slimmer base, drop unused deps) |
| `curl: option --data-binary: out of memory` | Image too big to fit in RAM with `--data-binary` | Use `-T mymodel.tar.gz` (streams) instead |
| `{"detail":"invalid token"}` | Wrong/missing API key | Check the bearer token from your credentials |
| `connection refused` or hang on curl | Off the EPFL network/VPN | Connect to EPFL VPN and retry |
| `PermissionError: [Errno 13] '/output/...'` | Old bug on server; fixed | Re-submit (if you still see this, tell the organizer) |
| No email for 5+ min | Check spam; sender is `newsletter@paperboatch.com` | Mark as not spam |
