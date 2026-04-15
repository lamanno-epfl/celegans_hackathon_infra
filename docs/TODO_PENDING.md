# Pending items — blocked on inputs from collaborators

Last updated: 2026-04-15 (evening).

## Blocked on Xinyi — **critical path**

| # | Item | Blocks |
|---|---|---|
| **X1** | **Gold `ground_truth_masks/<sample_id>_seg.npy`** (or the atlas-ID → canonical-ID lookup so we can derive them ourselves). Referenced in her own `scripts/npz_to_seg.py` usage docstring but not shipped. `evaluation_annotation_SEALED/ground_truth.npz` contains pose params only (26 KB). | **Turning on v2 scoring.** Without it `scoring/seg_accuracy.py` stays at placeholder 0.0 and the worker cannot score seg-in/seg-out submissions. |
| X3 | **Larger held-out set.** Currently 860 (≈3% of 30k, ≈3 per timepoint). Luca asked for ≥2000 stratified. | Statistical reliability of per-timepoint scores. Pipeline runs fine at any size. |
| X4 | **Final `05_manual_segmentation/` swap.** Deferred-relevance: only matters if the integration-score component comes back into the v2 contract. | Domain-adaptation integration score. |
| X5 | **Mask noise distribution doc.** "moderate" vs "heavy" labels in `ground_truth.npz` are unexplained — from inspection it looks like cell dropout (e.g. 92→75), but fraction/distribution aren't documented. | Calibrating expectations for participants. Not blocking infra. |

## Blocked on Luca

| # | Item |
|---|---|
| L1 | Final scoring weights once v2 components are all live. Current scaffold: `combined_v2` with `0.30·reg + 0.20·tp + 0.30·cell + 0.20·integration`. If the near-term contract is seg-in/seg-out only, this collapses to just the seg accuracy. |
| L2 | Decision on whether nuclei/membrane image channels come back into the container inputs. Currently out per instruction. |
| L3 | Decision on whether `u_value`, pose outputs, and integration score return as scored dimensions (currently deferred — see `docs/contract_v2.md`). |

## Worker switch-on (deferred until X1 lands)

These changes are scaffolded but not wired into the running v1 worker — doing so
before gold files arrive would just force participants to retarget a moving spec.

- `orchestrator/worker.py::prepare_input` — convert shipped
  `masks/sample_XXXX.npz` → `/input/<sample>_seg.npy` via
  `scripts/npz_to_seg.py`.
- `orchestrator/validation.py` — accept output `<sample>_seg.npy` per input
  (filename match + dict shape with `"masks"` int array).
- `orchestrator/worker.py::score_submission` — call
  `scoring.seg_accuracy.score_directory(pred_dir, gt_dir)`.
- Swap the example `examples/participant_template/` → `participant_template_seg/`
  as the default reference for new teams.
- Retire / archive the old trivial-baseline and image-based fake participants
  (move under `baselines/_v1_archive/`).

## Done now (v2 scaffolding that does NOT touch the running v1 pipeline)

- `docs/contract_v2.md` — full seg-in/seg-out spec, with the old pose/integration
  material called out as deferred.
- `scoring/seg_accuracy.py` — importable wrapper around `scripts/score_seg.py`,
  placeholder-aware (returns 0.0 + note when gold dir missing).
- `scoring/tests/test_seg_accuracy.py` — 6 unit tests covering perfect match,
  majority vote, placeholder mode, end-to-end tmpdir scoring.
- `scoring/timepoint.py`, `scoring/cell_naming.py`, `scoring/combined_v2.py`
  — kept in case pose/name components come back; all under test.
- `examples/participant_template_seg/` — identity baseline + Dockerfile +
  README demonstrating the new contract.
- `docs/participant_quickstart.md` — OS prereqs, VPN, credentials for
  `lemanichack`, troubleshooting table (unchanged in this pass).
