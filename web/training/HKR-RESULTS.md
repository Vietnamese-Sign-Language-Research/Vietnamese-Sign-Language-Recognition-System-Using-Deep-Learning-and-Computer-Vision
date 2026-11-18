# H/K/R retraining results

**Status: user-approved demo release of the retrained model. Full production
quality checks remain failed for H/K/R; those outputs are marked experimental.**

## Final corrected candidate

The national-chart skeleton CNN was retrained on 23 base letters. Conflicting
supplemental H/R recordings and old-corpus H/K/R positives were excluded.
Reviewed raw K recordings and camera-angle augmentation supplement K. A small
K specialist was also trained; it is not approved for production.

| Reserved classifier split | Correct | Total |
| --- | ---: | ---: |
| H | 119 | 120 |
| K | 177 | 181 |
| R | 120 | 120 |

Overall classifier accuracy: 98.79%.
These source-block tests are not a signer-independent real-world benchmark.

## Raw-input checks (complete website inference pipeline)

- Held-out K recordings: 97.22% label-correct, 70/72 planned frames.
- Separate-source K photos: 12/20. Still fails release gate.
- Separate-source R photos: 20/20.
- Bundled Y and X demo samples: both recognized.
- Blank input: rejected.
- User-supplied H reference: incorrectly recognized as K.
  The original model returns no sign. The photo was NOT used for training.
- K specialist validation recall: 75%, despite higher test-frame recall.
- Regression suite for experimental code: 31 tests passed. Passing software
  tests does not override failed recognition quality checks.

## Comparison with the original website model

On the same 434 photos covering the 22 common labels, the candidate
pipeline scores 86.87%, versus 32.26% for the original
YOLO. The candidate includes a restricted original-model fallback.

This is a mixed-label development diagnostic: its H labels conflict with the
selected national-chart form, the samples contain correlated video frames,
identity overlap is unknown, and the diagnostics informed revisions. Therefore
this aggregate cannot establish that the retrained model is better for real
VSL users. The supplied H photo is a concrete counterexample to promotion.

## Reproducibility and release boundary

- Detailed evaluation: `artifacts/alphabet-audit/vsl-corrected/skeleton-v4/final_release_evaluation.json`.
- Candidate: `artifacts/alphabet-audit/vsl-corrected/skeleton-v4/letter_skeleton.pt`.
- Candidate SHA256: `0c1065a3023e58a048f699d5b5b6820b9557961249b123a4b86c990bccb2930d`.
- Source review and exclusions: `training/AUDIT.md`.
- H reference source: https://tudienngonngukyhieu.com/bang-chu-cai-ngon-ngu-ky-hieu?chu-cai=h
- Original source videos include regional entries; inspect the specific clip
  rather than assume every H variant has one identical handshape.

The user clarified that the retrained model should be deployed for the demo,
with H/K/R improvements deferred. The release checkpoint preserves all trained
tensors, records the failed checks, sets approved_for_demo=True and
approved_for_production=False, and marks H/K/R results as uncertain. The original
YOLO weights remain available for fallback and rollback. The user reference
photograph remains private, evaluation-only, and excluded from the repository.

On the 374 common-label diagnostic photos excluding H/K/R, the retrained pipeline
classified 345 correctly (92.25%) versus 140 (37.43%) for the original YOLO.
This is development evidence, not a guarantee for a new signer.
