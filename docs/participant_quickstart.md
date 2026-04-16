# Participant Quickstart

Everything you need to submit a model. Tested end-to-end on macOS (Intel + Apple Silicon) and Linux; Windows works via WSL2 (see section 0).

> **You must be on the EPFL network or EPFL VPN to reach the orchestrator.** Off-campus? Connect to EPFL VPN first (instructions at https://www.epfl.ch/campus/services/ip-network/vpn/). The orchestrator is only reachable from the EPFL internal network.

## Your credentials

- **Team name:** `lemanichack`
- **Bearer token:** `WmQstelLIxJLiFf1_59Z_TZfg73Resn-0EZH12utezw`
- **Orchestrator URL:** `http://128.178.188.212:8000`
- **Notification email:** `luca.fusarbassini@epfl.ch`

Keep the bearer token private. Anyone with it can submit as your team and burn your quota (11 submissions).

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

## 1. Get the repo and pick a starting template

```bash
git clone https://github.com/lamanno-epfl/celegans_hackathon_infra
cd celegans_hackathon_infra
```

Two ready-to-use templates, both for the current **seg-in / seg-out** contract:

- `examples/participant_template_seg/` — pure-NumPy identity baseline (tiny image, fastest iteration).
- `examples/pytorch_baseline/` — PyTorch + CUDA base image; same identity output but runs a real `torch.nn.Module` inside the sandbox. Use this if you plan to ship a neural net.

```bash
cp -r examples/pytorch_baseline my_submission         # or participant_template_seg
```

## 2. Implement your model

Open `my_submission/predict.py`. `predict_one(mask, model)` (or `predict_ids(mask)` in the NumPy template) is called once per evaluation sample and must return an **HxW int mask with the same non-zero regions as the input**, whose pixel values are your predicted **canonical atlas cell IDs** for each region.

- **Input:** `(554, 554)` int mask. 0 = background. Each non-zero connected region is one segmented cell. **Pixel values inside each region are arbitrary instance labels (1..N in a random order).** They carry NO atlas, timepoint, or pose information. You cannot infer what cell a region is from its pixel value.
- **Output:** `(554, 554)` int32 mask. Same region layout. Pixel values = your predicted canonical atlas IDs (one ID per region; the scorer majority-votes inside each gold region).

**What the task actually is.** Each input mask is a 2D slice through a 3D *C. elegans* embryo. You don't know:
- the **developmental timepoint** (out of ~50 stages in the 4D atlas),
- the **orientation** of the 2D cut through the 3D embryo.

Your model has to use the shape/geometry of the segmented regions, compared against the **canonical 4D reference atlas** (shipped in `data/reference_3d/` in this repo — `COPY` it into your image at build time; there's no network at runtime), to infer timepoint + pose and then assign each region the atlas ID of the reference cell it spatially overlaps.

Both shipped templates are **identity baselines** that pass input labels straight through — they're only useful for confirming the Docker round-trip works. Expected identity score ≈ **0 (random)**, because input labels have no atlas meaning.

> **Reference atlas status (2026-04-16):** `data/reference_3d/volume_masks.npy` is a small placeholder — a real 4D reference is incoming from the La Manno lab and will drop into the same path with matching atlas-ID namespace. Pin your loader to the filename, not the current shape.

## 3. Build the Docker image

The server runs **linux/amd64**. If you are on **Apple Silicon (M1/M2/M3) or any ARM machine**, you must pass `--platform linux/amd64` — otherwise the server will reject your image with `exec format error`.

```bash
cd my_submission

# Apple Silicon / ARM:
docker build --platform linux/amd64 -t lemanichack/mymodel:v1 .

# Intel Mac / Linux x86_64:
docker build -t lemanichack/mymodel:v1 .
```

Two easy-to-miss details:

- **Trailing `.`** at the end (that's the build context — without it `docker build` refuses with "requires 1 argument").
- **`lemanichack/`** prefix in the tag must match your team exactly. The server identifies your submission by that first path segment.

### Using the GPU — strongly recommended

This is an ML competition and the evaluation server has one **NVIDIA RTX 4070 (12 GB)**. Containers run with `--gpus all`. CPU-only submissions will struggle to beat the 120-min wall-clock on anything real.

**Eval host as of 2026-04-16:** driver 570.x (max CUDA runtime 12.8), Docker 29.x, nvidia-container-toolkit 1.18.x, RTX 4070 (Ada, `sm_89`).

**Verified base image** (don't deviate unless you have a reason):

```dockerfile
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
```

Python 3.11 + torch 2.4.1 + CUDA 12.4 kernels preinstalled, no `apt-get` (builds reliably behind EPFL VPN), tested end-to-end — `torch.cuda.is_available() == True` inside the sandbox.

**Compatibility rules if you roll your own base:**
- CUDA runtime must be **≤ 12.8** (host driver cap). Anything newer → `CUDA driver version is insufficient`.
- PyTorch wheel must be **≥ 2.1** (earlier wheels don't include `sm_89` kernels → slow fallback or silent error).
- Do not pin a CPU-only torch wheel. If you used `pip install torch` without `--index-url https://download.pytorch.org/whl/cu124`, you probably got CPU torch on linux/amd64.
- **Build with `--platform linux/amd64`** on Apple Silicon; the GPU at runtime is on the server, not your laptop.

**Sanity check at the top of your `predict.py`:**

```python
import torch
assert torch.cuda.is_available(), "GPU not visible — rebuild on the CUDA base image"
print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__}", flush=True)
```

If that print doesn't show `NVIDIA GeForce RTX 4070` in the container logs (they're attached to any failure email), you shipped a CPU-only wheel by accident.

### (optional) Smoke-test locally before uploading

Stage a few input seg files from any mask `.npz` directory (e.g. the public
training masks, if you have them) and run the container against them:

```bash
mkdir -p /tmp/in /tmp/out
python scripts/npz_to_seg.py \
    /path/to/some/masks/ \
    -o /tmp/in --max-samples 5

docker run --rm --platform linux/amd64 \
    -v /tmp/in:/input:ro -v /tmp/out:/output \
    lemanichack/mymodel:v1

ls /tmp/out   # expect one *_seg.npy per input
```

## 4. Save + upload

```bash
docker save lemanichack/mymodel:v1 | gzip > mymodel.tar.gz
ls -lh mymodel.tar.gz    # sanity check: CUDA images are often 2–5 GB, plain Python models tens of MB
# Heads up: gzipping a 3–5 GB docker save can hang for 3–8 minutes on a Mac
# with no progress indicator — that's normal, don't ctrl-C it. Add `pv` in the
# middle if you want a progress bar:
#   docker save lemanichack/mymodel:v1 | pv | gzip > mymodel.tar.gz

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

- **Entrypoint:** whatever your Dockerfile's `CMD` is (e.g. `python3 /app/predict.py`). No `/predict.sh` convention needed.
- **Inputs (read-only at `/input/`):**
  - `/input/<sample_id>_seg.npy` — Cellpose-style dict (load with `np.load(path, allow_pickle=True).item()`); key `"masks"` is the `(554, 554)` int mask. Pixel values are **arbitrary instance labels 1..N (shuffled per submission)** — NOT atlas IDs. Your model must re-assign atlas IDs from scratch.
  - `/input/manifest.json` — list of sample ids in the shuffled/anonymized order used by the worker.
  - `/input/real_manual/<LE003_*>_seg.npy` — real manually-annotated embryos (from `data/real/held_out/05_manual_segmentation`). Same shape, plain Cellpose instance labels. Available as a **domain-adaptation reference** — your model may use them at inference time to calibrate features. Not scored against.
- **Outputs (writable at `/output/`):**
  - `/output/<sample_id>_seg.npy` — one per input, **same filename**. Dict with `"masks"` (`(554, 554)` int32) and `"cell_ids"` (sorted unique non-zero).
- **Runtime resources:** 32 GB RAM, 8 CPUs, `--gpus all` on the eval host, no network, read-only root FS with 10 GB tmpfs on `/tmp`, **120-minute wall-clock cap**.
- **Sandbox flags:** `--network=none`, `--read-only`, `--cap-drop=ALL`, `--security-opt=no-new-privileges`, `--pids-limit=4096`. **No internet at runtime** — download any model weights at build time and `COPY` them into the image.

## Scoring summary

```
per-sample accuracy = (#correctly-named regions) / (#regions in gold mask)
final score         = (total correct regions) / (total gold regions)   # micro-averaged
```

For each region in the gold `ref_mask`, the worker majority-votes the predicted pixel IDs inside that region and compares against the gold ID. A region is "correct" iff the majority pred ID equals the gold ID. See `scripts/score_seg.py` for the exact implementation.

**Reference numbers (identity baseline):** passing the input mask through unchanged scores ≈ **0** — input pixel values are arbitrary instance labels with no atlas information, so returning them as "predicted atlas IDs" gets essentially everything wrong. Every real attempt should beat this trivially; the interesting comparison is against a simple nearest-neighbor-to-reference baseline.

---

## Troubleshooting — issues we've seen

| Symptom | Cause | Fix |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker Desktop not running | `open -a Docker`, wait ~30 s |
| `docker build ... requires 1 argument` | Missing trailing `.` | Add `.` at end of command |
| `command not found: -H` + `missing bearer token` | zsh broke your `\` line continuations | Put the whole `curl` on one line |
| `exec format error` (container exit 255) | Built an ARM image on Apple Silicon | Rebuild with `--platform linux/amd64` |
| `missing required output file` / `output missing N seg files` | Your container didn't write `/output/<sample_id>_seg.npy` for every input | Check `/output` listing matches `/input` listing; use identical filenames |
| `413 upload exceeds limit` | Image > 8 GB | Trim the image (use a slimmer base, drop unused deps) |
| `curl: option --data-binary: out of memory` | Image too big to fit in RAM with `--data-binary` | Use `-T mymodel.tar.gz` (streams) instead |
| `{"detail":"invalid token"}` | Wrong/missing API key | Check the bearer token from your credentials |
| `connection refused` or hang on curl | Off the EPFL network/VPN | Connect to EPFL VPN and retry |
| `PermissionError: [Errno 13] '/output/...'` | Old bug on server; fixed | Re-submit (if you still see this, tell the organizer) |
| No email for 5+ min | Check spam; sender is `newsletter@paperboatch.com` | Mark as not spam |
