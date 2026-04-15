# Seg-in / Seg-out participant template

Contract (see `docs/contract_v2.md`):

- **Input** (`/input/`): one `<sample>_seg.npy` per evaluation sample.
  Load with `np.load(path, allow_pickle=True).item()`; it's a dict with key
  `"masks"` = `(554, 554)` int mask. Pixel values are atlas cell IDs
  (0 = background), with cell-dropout noise applied.
- **Output** (`/output/`): one `<sample>_seg.npy` per input, same filename,
  same geometry (same pixel regions). Your model's job is to assign the
  **canonical atlas cell ID** to each region.

## Build and test locally

```bash
# Stage some inputs (convert shipped mask .npz files → _seg.npy)
mkdir -p /tmp/in /tmp/out
python scripts/npz_to_seg.py \
    data/real/held_out/evaluation_annotation_SEALED/masks/ \
    -o /tmp/in

docker build --platform linux/amd64 -t me/seg-baseline:v1 .
docker run --rm --platform linux/amd64 \
    -v /tmp/in:/input:ro -v /tmp/out:/output \
    me/seg-baseline:v1
ls /tmp/out | head
```

## Submit

Same upload path as v1 — see `docs/participant_quickstart.md`.
