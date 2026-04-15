# Container Contract v2 (post-Xinyi-data design)

> **Status:** specification only. The running v1 pipeline (registration + integration) is untouched. v2 will replace v1 once the atlas volumes and the cell-name mapping snippet arrive from Xinyi. Tracked in `docs/TODO_PENDING.md`.

## Why v2

Xinyi's `evaluation_annotation_SEALED` dataset reframes the task:

- Input is **a single noisy segmentation mask** (no nuclei/membrane channels for now).
- The model must predict (a) the cutting plane (rotation + translation + **timepoint** + u-value) and (b) **the Sulston cell name for each instance ID in the mask**.
- Domain adaptation moves from "simulated vs real images" to "simulated-mask pool vs ~5–9 real manual segmentations" (`05_manual_segmentation/`).

Image channels (nuclei/membrane) are **commented out, not removed** — Luca may revisit.

## Inputs (read-only at `/input`)

```
/input/
├── manifest.json                      # list of sample-ids, shuffled + anonymized
├── masks/
│   └── <sample_id>.npz                # keys: "masks" (HxW int), "n_cells" (scalar int)
├── atlas/                             # PENDING from Xinyi
│   └── timepoint_<T>/volume.npy       # one 3D volume per timepoint
└── manual_seg/                        # for domain adaptation; PENDING final swap from Xinyi
    ├── <name>.tif                     # 2-channel raw image (uint16, 554x554)
    └── <name>_seg.npy                 # Cellpose seg dict; key "masks" is the mask
```

The participant's container reads these, infers, writes outputs.

## Outputs (writable at `/output`)

```
/output/
├── poses.json
├── cell_predictions.json
├── embeddings.npy
└── metadata.json
```

### `poses.json`
```json
{
  "<sample_id>": {
    "timepoint": <int>,           // predicted reference timepoint (1-based)
    "rotation": [[..3x3..]],      // 3x3 orthogonal, det=+1
    "translation": [z, y, x],     // in voxel units of the predicted timepoint volume
    "u_value": <float>            // optional plane orientation, can be omitted
  }, ...
}
```

### `cell_predictions.json`
```json
{
  "<sample_id>": {
    "<instance_id>": "<sulston_name>",   // e.g. "ABala", "MSpaap"
    ...
  }, ...
}
```
Instance IDs are arbitrary integer labels from `masks` array; participants must **assign a Sulston name to each**. Background (id=0) is excluded.

### `embeddings.npy`
- `(N, D)` float32, row order matches `manifest.json`.
- `D ∈ [64, 2048]`. Used for domain-integration scoring against the manual-seg pool.

### `metadata.json`
- Must include `embedding_dim` (int) and `model_name` (str).

## Scoring (v2)

| Component | Status | Module |
|---|---|---|
| `registration_score` | math unchanged from v1 (geodesic rotation + normalized translation) | `scoring/registration.py` |
| `timepoint_accuracy` | new: fraction with correct timepoint | `scoring/timepoint.py` |
| `cell_naming_accuracy` | **placeholder** until Xinyi's `(timepoint, center, angles, u) → per-instance Sulston names` snippet lands | `scoring/cell_naming.py` |
| `integration_score` | unchanged in spirit; embeddings labeled simulated-mask vs manual-seg | `scoring/integration.py` |
| `final` | weighted combination, exact weights TBD with Luca | `scoring/combined_v2.py` |

Default weights (placeholder, easy to retune):
```
final = 0.30 * registration_score
      + 0.20 * timepoint_accuracy
      + 0.30 * cell_naming_accuracy
      + 0.20 * integration_score
```
Same registration-threshold gate idea applies (if registration < 0.3, model is too bad to fairly weight cell naming; collapse into registration-only).
