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

- Input: `(554, 554)` int mask. 0 = background. Non-zero pixel values are atlas IDs under some (unknown) timepoint, with cell-dropout noise applied.
- Output: `(554, 554)` int32 mask. Same region layout. Pixel values = your predicted canonical atlas IDs.

Both templates are **identity baselines** (pass input IDs through unchanged) — submit one first to verify the round-trip, then swap in your model. On the evaluation set the identity baseline scores ~0.7 because the noise only drops cells, it doesn't permute the surviving IDs; your model's job is to do better than that against the canonical reference frame.

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

### Using the GPU

The evaluation server has one **NVIDIA RTX 4070 (12 GB)** with the NVIDIA Container Toolkit installed. Your container runs with `--gpus all` when `ENABLE_GPU=1` is set on the server (currently: yes).

To use it:
- Base your image on a CUDA image, e.g. `nvidia/cuda:12.4.1-runtime-ubuntu22.04`.
- Install a CUDA-matching PyTorch wheel (e.g. `pip install --index-url https://download.pytorch.org/whl/cu124 torch`).
- Inside your code: `torch.cuda.is_available()` must be `True`.
- **Build with `--platform linux/amd64`** on Apple Silicon as usual; the GPU at runtime is on the server, not your laptop.

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

- **Entrypoint:** whatever your Dockerfile's `CMD` is (e.g. `python3 /app/predict.py`). No `/predict.sh` convention needed for v2.
- **Inputs (read-only at `/input/`):**
  - `/input/<sample_id>_seg.npy` — Cellpose-style dict (load with `np.load(path, allow_pickle=True).item()`); key `"masks"` is the `(554, 554)` int mask.
  - `/input/manifest.json` — list of sample ids in the shuffled/anonymized order used by the worker.
- **Outputs (writable at `/output/`):**
  - `/output/<sample_id>_seg.npy` — one per input, **same filename**. Dict with `"masks"` (`(554, 554)` int32) and `"cell_ids"` (sorted unique non-zero).
- **Runtime resources:** 32 GB RAM, 8 CPUs, `--gpus all` on the eval host, no network, read-only root FS with 10 GB tmpfs on `/tmp`, **60-minute wall-clock cap**.
- **Sandbox flags:** `--network=none`, `--read-only`, `--cap-drop=ALL`, `--security-opt=no-new-privileges`, `--pids-limit=4096`. **No internet at runtime** — download any model weights at build time and `COPY` them into the image.

## Scoring summary

```
per-sample accuracy = (#correctly-named regions) / (#regions in gold mask)
final score         = (total correct regions) / (total gold regions)   # micro-averaged
```

For each region in the gold `ref_mask`, the worker majority-votes the predicted pixel IDs inside that region and compares against the gold ID. A region is "correct" iff the majority pred ID equals the gold ID. See `scripts/score_seg.py` for the exact implementation.

**Reference numbers (identity baseline):** passing the input mask through unchanged scores ≈ 0.68 on a 5-sample smoke test — the "noise drops cells, keeps IDs" property means identity is a surprisingly strong floor. Beating it requires your model to recover cells the noise removed or to correct any IDs the generation pipeline perturbed.

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
