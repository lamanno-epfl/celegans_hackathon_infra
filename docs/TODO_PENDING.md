# Pending items — blocked on inputs from collaborators

Last updated: 2026-04-15.

## Blocked on Xinyi

| # | Item | Blocks |
|---|---|---|
| X1 | **Atlas → Sulston-name mapping snippet.** Python that takes `(timepoint, center, angles, u_value)` and returns `{instance_id: sulston_name}` for the gold cells in that cutting plane. | Implementing real `scoring/cell_naming.py` (currently a placeholder returning 0.0). |
| X2 | **Atlas 3D volumes.** One per timepoint. To be mounted into the container at `/input/atlas/timepoint_<T>/volume.npy`. | Letting participants project / register against the right reference. Worker `prepare_input` cannot stage what doesn't exist yet. |
| X3 | **Larger held-out set.** Currently 860 (≈3% of 30k, ≈3 per timepoint). Luca asked for ≥2000 stratified. | Statistical reliability of per-timepoint scores. Pipeline runs fine at any size. |
| X4 | **Final `05_manual_segmentation/` swap.** Current 9 manual segs are visible to participants; the *real* held-out manual segs (lab data, same format) replace them tomorrow. | Domain-adaptation integration score. Format already known; just swap files in place. |
| X5 | **Mask noise distribution doc.** "moderate" vs "heavy" labels in `ground_truth.npz` are unexplained — what fraction of cells dropped/added/perturbed for each? | Calibrating expectations for participants; not blocking infra. |

## Blocked on Luca

| # | Item |
|---|---|
| L1 | Final scoring weights for `combined_v2`. Current placeholder: `0.30·reg + 0.20·tp + 0.30·cell + 0.20·integration`. |
| L2 | Decision on whether to keep nuclei/membrane image channels available in the container in addition to masks (currently disabled per Luca's instruction). |
| L3 | Decision on whether to allow `u_value` as a separate scored dimension or fold it into rotation. |

## Infra changes deferred until X1 + X2 land

(These will be done together; doing them piecemeal forces participants to re-target a moving spec.)

- `orchestrator/worker.py::prepare_input` — switch from `images/*.npy + masks/*.npy + reference_3d/` to `masks/*.npz + atlas/timepoint_<T>/volume.npy + manual_seg/` layout (per `docs/contract_v2.md`).
- `orchestrator/validation.py` — accept the v2 output set: `poses.json` (with `timepoint`), `cell_predictions.json`, `embeddings.npy`, `metadata.json`.
- `orchestrator/worker.py::score_submission` — call `combined_v2` with the four components.
- `examples/participant_template/` — rewrite `predict.py` to read `masks/*.npz`, emit v2 outputs.
- `baselines/*` — rewrite trivial baseline against the v2 contract; degenerate baseline becomes "predict random Sulston names + identity rotation"; domain-adapted becomes a real GRL on the embedding head.
- `README.md` + `docs/participant_quickstart.md` — update the "scientific problem" + "container contract" sections.
- Synthetic-data generator — produce v2-shaped masks + dummy atlas + a few "real-style" segmentations, for end-to-end smoke testing without real data.

## Done now (v2 scaffolding that does NOT touch the running v1 pipeline)

- `docs/contract_v2.md` — full spec.
- `scoring/timepoint.py` — accuracy + within-tolerance scorer.
- `scoring/cell_naming.py` — placeholder Hungarian scorer + interface for the eventual mapping function.
- `scoring/combined_v2.py` — weighted combiner with the four components.
- `scoring/tests/test_v2.py` — unit tests for the new modules.
