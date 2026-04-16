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

## 2. What your container will see (and what it must write)

**Read this carefully. Your `predict.py` I/O must match it exactly or the worker will reject your submission with `validation_error` — no score, but also no quota decrement.**

### 2a. Two held-out sets land in your sandbox

The server mounts **two** held-out test sets into `/input/`. You do **not** see either locally — they only appear at eval time inside the sandbox:

```
/input/                               read-only mount
  manifest.json                       ["sample_0000", "sample_0001", ..., "sample_NNNN"]
  sample_0000_seg.npy                 <-- simulated held-out embryos,
  sample_0001_seg.npy                     shuffled + anonymized per submission
  ...                                     (currently 857 files)
  sample_NNNN_seg.npy
  real_manual/
    LE003_*_seg.npy                   <-- real held-out embryos,
                                          manually annotated (ground-truth
                                          real data). Currently ~5 files.

/atlas/                               read-only mount, 4D reference atlas
  reference.ome.zarr/                 OME-Zarr v3 (T=255, Z=214, Y=356, X=256)
                                        labels   int16  cell IDs
                                        membrane uint8  fluorescence (0.18 µm iso)
                                        nucleus  uint8  raw confocal
  name_dictionary.csv                 cell ID -> Sulston lineage name (e.g. 5 -> ABalaa)
  README.md                           atlas provenance + structure
  view_reference.py                   napari viewer (for local inspection)

/output/                              read-write mount, empty at start
/tmp/                                 10 GB tmpfs, writable, wiped after run
/                                     everything else read-only
                                       NO network, GPU at /dev/nvidia*
```

Both held-out sets are **test data** — treat them symmetrically. Neither is a training signal; you process them both in the same inference run.

The atlas at `/atlas/` is the **same volume the scorer uses** (Bhogale et al. 2025, WT_Sample1, ground-truth-annotated). Do **not** bake it into your image — it's bind-mounted at runtime so every team uses the exact same reference. For local testing, mount your own copy with `-v /path/to/reference_4d:/atlas:ro` (download link in §1).

### 2b. What each `*_seg.npy` file contains

Every file in `/input/` (both sim and real) is a pickled Cellpose-style dict with two keys:

```python
import numpy as np
seg = np.load("/input/sample_0000_seg.npy", allow_pickle=True).item()
# seg is a dict:
seg["masks"]      # (554, 554) int32 ndarray. 0 = background.
                  # Each non-zero connected region = one segmented cell.
                  # Pixel values = arbitrary instance labels (1..N, shuffled
                  # per submission). NO atlas / timepoint / pose info.
seg["cell_ids"]   # sorted list[int] of unique non-zero labels in `masks`.
```

`/input/real_manual/LE003_*_seg.npy` has the **same dict shape**; labels there are plain Cellpose instance IDs from manual annotation.

### 2c. What you must write to `/output/`

Your container must produce **two things** in one run:

#### (i) Per-sim-sample predicted atlas IDs (scored → seg_accuracy, 70% of final)

For every `/input/sample_XXXX_seg.npy`, write `/output/sample_XXXX_seg.npy` — **same filename** — containing:

```python
{
    "masks":    np.int32 array of shape (554, 554),  # same non-zero footprint as
                                                     # the input; per-region pixel
                                                     # values = your predicted
                                                     # canonical atlas IDs.
    "cell_ids": sorted list[int],                    # unique non-zero atlas IDs used.
}
np.save("/output/sample_0000_seg.npy", the_dict_above)
```

The scorer majority-votes inside each gold region, so you do **not** need pixel-perfect boundaries — just a consistent predicted ID per region.

#### (ii) A single `embeddings.npz` — per-cell features for **both** sim and real (scored → integration_score, 30% of final)

For every non-zero region across **both** held-out sets (sim + real), emit one feature vector using the same encoder. Stack them all into one `/output/embeddings.npz` with a per-row domain label:

```python
np.savez(
    "/output/embeddings.npz",
    embeddings = np.ndarray of shape (N, D), dtype float32 or float64,
                                            # N = total cells across ALL sim + all real
                                            # D = your choice, 2 <= D <= 512
    domain     = np.ndarray of shape (N,), dtype "<Uxx",
                                            # each row: either "sim" or "real"
                                            # ordering within each domain is free
)
```

The server computes `integration_score = 1 − 2·|cv_acc − 0.5|` via a 5-fold logistic-regression classifier trying to separate sim from real in your feature space. The **more your sim and real embeddings mix** (classifier can't tell them apart), the higher the integration score. Near-constant embeddings ("collapsed") score 0.

**Validation rules (will cause `validation_error` if violated):**
- File exists, loads as `.npz` with arrays `embeddings` and `domain`.
- `embeddings.ndim == 2`, dtype float, all finite (no NaN/Inf).
- `2 <= D <= 512`.
- `len(domain) == N`, each value is exactly the string `"sim"` or `"real"`.
- At least 3 rows per domain.

### 2d. Minimal working loop

```python
from pathlib import Path
import numpy as np

INPUT_DIR  = Path("/input")
OUTPUT_DIR = Path("/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def cell_embedding(mask, cell_id):
    """Replace with your learned encoder. Must return a D-length float vector."""
    ys, xs = np.where(mask == cell_id)
    h, w = mask.shape
    return np.array([ys.mean()/h, xs.mean()/w, len(ys)/(h*w)], dtype=np.float32)

sim_embs, real_embs = [], []

# (1) sim held-out: predict atlas IDs + collect per-cell embeddings
for in_path in sorted(INPUT_DIR.glob("sample_*_seg.npy")):
    seg   = np.load(in_path, allow_pickle=True).item()
    mask  = seg["masks"]
    # ---- your classifier goes here ----
    mask_out = my_model.predict(mask)     # (554,554) int32 with predicted atlas IDs
    # -----------------------------------
    np.save(OUTPUT_DIR / in_path.name, {
        "masks":    mask_out.astype(np.int32),
        "cell_ids": sorted(int(x) for x in np.unique(mask_out) if x > 0),
    })
    for cid in (int(x) for x in np.unique(mask) if x > 0):
        sim_embs.append(cell_embedding(mask, cid))

# (2) real held-out: collect per-cell embeddings (no seg output expected)
for real_path in sorted((INPUT_DIR / "real_manual").glob("*_seg.npy")):
    seg  = np.load(real_path, allow_pickle=True).item()
    mask = seg["masks"]
    for cid in (int(x) for x in np.unique(mask) if x > 0):
        real_embs.append(cell_embedding(mask, cid))

# (3) single embeddings.npz with domain labels
all_embs = np.vstack(sim_embs + real_embs).astype(np.float32)
domain   = np.array(["sim"] * len(sim_embs) + ["real"] * len(real_embs))
np.savez(OUTPUT_DIR / "embeddings.npz", embeddings=all_embs, domain=domain)
```

Both shipped templates (`examples/participant_template_seg/` and `examples/pytorch_baseline/`) implement this loop end-to-end — start from one of them.

### 2e. The scientific task in one paragraph

Each input mask is a 2D slice through a 3D *C. elegans* embryo at an **unknown developmental timepoint** (one of 255 stages in the 4D atlas, T001–T255, 2 → 580 cells) and **unknown 3D orientation**. The pixel labels are arbitrary instance IDs — they tell you nothing. Your model must use the shape/geometry of the segmented regions, compared against the **canonical 4D reference atlas** mounted at `/atlas/reference.ome.zarr/`, to (i) infer the timepoint, (ii) infer the 3D pose, and (iii) assign each region the atlas ID of the reference cell it spatially overlaps. In parallel, your model must also produce per-cell features that align sim- and real-image cells in a shared space — that's the 30% integration term. Expected identity-baseline score ≈ **0** on seg and ≈ **0** on integration (shape-only features separate sim from real trivially with only ~5 reals).

> **Local atlas copy:** the atlas (~1.4 GB zip, 1.7 GB unpacked) is **not** in the git repo. Ask the organizers (`maxim.pavliv@gmail.com`) for the download link, then unpack it under any local path and mount with `-v /your/path/reference_4d:/atlas:ro` when testing locally. The atlas ID namespace matches the eval set 1-for-1 — no remapping needed.

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
   - `evaluation_complete` — with `final_score` (= seg accuracy) and `n_scored`. **Domain-adaptation is not currently part of the final score** — only per-sample segmentation-ID accuracy is scored.
   - `validation_error` — your outputs didn't match the contract. **Does not count against quota.**
   - `evaluation_failed` — container crashed, timed out, or returned non-zero. **Does not count against quota.**

If you hit your quota, you'll get `submission_limit_reached`. Quota is per team.

---

## Sandbox runtime (reference)

- **Entrypoint:** whatever your Dockerfile's `CMD` is (e.g. `python3 /app/predict.py`). No `/predict.sh` convention.
- **Inputs / outputs:** exact contract is in **Section 2 above**. Don't re-derive it from memory.
- **Runtime resources:** 32 GB RAM, 8 CPUs, `--gpus all` (RTX 4070 12 GB), no network, read-only root FS with 10 GB tmpfs on `/tmp`, **120-minute wall-clock cap**.
- **Sandbox flags:** `--network=none`, `--read-only`, `--cap-drop=ALL`, `--security-opt=no-new-privileges`, `--pids-limit=4096`. **No internet at runtime** — download any model weights at build time and `COPY` them into the image.

## Scoring summary

```
final = 0.7 * seg_accuracy + 0.3 * integration_score
```

**seg_accuracy** (70%): per-region majority vote of your predicted pixel IDs vs the gold atlas IDs, micro-averaged across all 857 held-out sim samples. Formally: `(total correct regions) / (total gold regions)`. A region is "correct" iff the majority predicted ID within that region equals the gold atlas ID. See `scripts/score_seg.py` for the implementation.

**integration_score** (30%): domain-adaptation term. A 5-fold stratified logistic-regression classifier is trained to separate sim vs real rows in your `embeddings.npz`. `integration_score = 1 − 2·|cv_acc − 0.5|` — so perfect mixing (classifier at chance) = 1.0 and perfect separability (classifier at 100%) = 0.0. Collapsed / non-finite embeddings → 0. See `scoring/integration.py`.

**Reference numbers (identity baseline + shape-only features):** seg ≈ 0, integration ≈ 0 (with only ~5 real samples, the sim/real shape distributions separate trivially). Any real model beats both floors; the interesting comparisons are a nearest-neighbor-to-reference-atlas baseline for seg and a cross-domain contrastive feature extractor for integration.

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
