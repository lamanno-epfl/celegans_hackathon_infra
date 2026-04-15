# Container Contract v2 (seg-in / seg-out, 2026-04-15)

> **Status:** this is the **current target** contract per Xinyi's 2026-04-15
> clarification. The running v1 pipeline (registration + integration on image
> slices) is untouched in the worker. v2 switches on once the gold seg files
> land. Tracked in `docs/TODO_PENDING.md`.

## The task in one paragraph

Given a noisy 2D segmentation mask whose pixel values are **integer atlas IDs
from one timepoint of a 4D C. elegans atlas**, assign the **canonical atlas ID**
to each labeled region. Scoring is per-region majority vote vs a gold mask with
the same geometry (Xinyi's `scripts/score_seg.py`). The simulation-to-real
domain-adaptation piece and the pose/rotation/timepoint pieces from the earlier
v2 draft are **deferred** — they may come back but are not in-scope for the
near-term evaluation.

## Inputs (read-only at `/input/`)

```
/input/
└── <sample_id>_seg.npy       # Cellpose-style dict; key "masks" = (H, W) int
```

- Produced from the shipped `evaluation_annotation_SEALED/masks/*.npz` by
  `scripts/npz_to_seg.py` (Xinyi) — the worker runs this conversion, so
  participants just see `/input/*_seg.npy`.
- Pixel values = atlas IDs under the sample's (unknown) timepoint, after
  cell-dropout noise (some cells removed; remaining IDs unperturbed).
- Background = 0.

## Outputs (writable at `/output/`)

```
/output/
└── <sample_id>_seg.npy       # same filename as input, same geometry
```

Each output `_seg.npy` is a dict with:
- `masks` — `(H, W)` int32, **must match the input's non-zero region layout
  exactly** (score_seg.py majority-votes per GT region; divergent geometry will
  just score lower but not fail).
- `cell_ids` — sorted list of unique non-zero IDs in `masks`.

## Scoring

`scoring/seg_accuracy.py::score_directory(pred_dir, gt_dir)` — thin importable
wrapper around `scripts/score_seg.py::score_single`. Per sample:

```
accuracy = n_correct_regions / n_gt_regions
```

Overall score = total_correct / total_gt across all samples.

**Placeholder until gold files land:** the scorer returns `0.0` with a
`note` if `gt_dir` is empty/missing.

## What is NOT in this contract (by design)

- No pose outputs (`rotation`, `translation`, `timepoint`, `u_value`).
- No `embeddings.npy` — the simulated-vs-real integration score is deferred.
- No nuclei/membrane image channels — commented out of the plan, not deleted.
  Luca may revisit once the core scorer stabilizes.

Old v2 material covering those components has been moved to
`docs/contract_v2_pose_archive.md` for reference.

## Known blockers

| ID | What | Why it blocks the switch-on |
|---|---|---|
| **X1** | Gold `ground_truth_masks/<sample_id>_seg.npy` (or the atlas-ID → canonical-ID lookup to derive them) | `score_seg.py` has no target to compare against. `ground_truth.npz` shipped 2026-04-15 contains only pose parameters (26 KB total; a per-pixel gold would be ~1 GB). |
| X3 | Held-out set size (860; Luca wants ≥2000) | Statistical reliability of per-timepoint scores. |
| X4 | Final `05_manual_segmentation/` swap | Only matters if integration scoring comes back. |

See `docs/TODO_PENDING.md` for the full list.
