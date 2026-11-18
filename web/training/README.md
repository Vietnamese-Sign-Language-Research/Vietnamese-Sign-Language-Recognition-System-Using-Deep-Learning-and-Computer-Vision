# Vietnamese alphabet retraining

This is an experimental replacement pipeline. A dataset title alone does not
establish that its gestures match a particular Vietnamese signing standard.

## National H/K/R candidate and demo release

The original geometry experiment below is retained for reproducibility. Its
H/K/R outputs are incompatible with the national chart and are blocked by the
new adapter. The corrected candidate uses `letter_skeleton.py`, a 64×64 CNN,
with national-chart H/K/R skeletons and reviewed raw K recordings. It is still
subject to release checks; do not place an unapproved candidate in `weights/`.

After preparing the initial skeleton corpus and blended diagnostic data:

```powershell
python training/prepare_national_skeletons.py
python training/train_skeleton.py artifacts/alphabet-audit/vsl-corrected/cuong-vsl-v2.zip artifacts/alphabet-audit/vsl-corrected/skeleton-v4 --init artifacts/alphabet-audit/vsl-corrected/skeleton/letter_skeleton.pt --learning-rate .00035 --rotation 1.1 --epochs 10 --focus-hkr
python training/check_letter_release.py artifacts/alphabet-audit/vsl-corrected/skeleton-v4/letter_skeleton.pt
```

`--approve` refuses promotion when any release check fails, including missing
matching raw H evaluation. A good internal score cannot substitute for this.
See `AUDIT.md` for the H curation correction, excluded sources, archive hashes,
and the limits of the validation diagnostics. Supplemental H/R clips are not
used in the final national candidate. No third-party raw recordings are
bundled with the website.

## Sources and scope

- Training candidate: Mạc Phương Nga, DATASET_VSL, Kaggle, version 2,
  https://www.kaggle.com/datasets/mcphngnga/dataset-vsl (Apache 2.0).
- Use 23 base-letter folders. Map `DD` to `Đ`. Exclude `AA`, `AAA`, and `OOO`:
  their accent semantics require separate review; do not silently map them.
- Independent-source evaluation: TuanKiet1774/SignLanguage_Vietnamese, commit
  `2e44477b2939f14be2fbfd222191af9e617448f4`, `data/` images only.
  https://github.com/TuanKiet1774/SignLanguage_Vietnamese
  Its license is unspecified. Use these public images for evaluation only;
  do not redistribute them with the app or include them in training.
- Reference standard: Circular 17/2020/TT-BGDĐT, gazette page 51,
  https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2020/6/31603/31697-1-2020671-67217-2020-tt-bgddt.pdf#page=6
- Primary comparison of VSL and ASL handshapes: Tran et al. (2025),
  https://jst.vn/index.php/ssad/article/download/56/19, Figure 4.

The original Roboflow dataset contains demonstrably Vietnamese references
(including a Hanoi training-center H clip and the downward-fingers M).
There is insufficient evidence to call the entire dataset ASL. However, the
app's national-chart H and sampled training H are visually inconsistent.
The candidate data also contains that Hanoi-style H, so retraining alone
does not establish compatibility with every national-chart or regional variant.

## Reproduce the candidate

Download and retain the pinned training archive and its SHA-256 separately.
Run from the web workspace with its Python dependencies installed:

```powershell
python training/prepare_letters.py path/to/dataset-vsl-v2.zip artifacts/alphabet-audit/prepared --per-class 400 --workers 4
python training/train_letters.py artifacts/alphabet-audit/prepared
python training/prepare_letters.py path/to/external-source.zip artifacts/alphabet-audit/external/prepared --per-class 20 --workers 2 --external
python training/evaluate_letters.py artifacts/alphabet-audit/prepared/letter_candidate.pt artifacts/alphabet-audit/external/prepared --legacy weights/alphabet_model.pt
```

The last command expects the pinned external archive at
`artifacts/alphabet-audit/external/source.zip`.

Only MediaPipe-tracked hands enter the classifier. The report separately records
tracking failures. Frames rejected by tracking must not be counted as successes.
The data preparation removes exact duplicates and raises on conflicting labels.
Filename blocks are separated across train/validation/test to reduce adjacent-
frame overlap; this is **not a signer-independent split**. The separate-source
collection also contains correlated video frames, and its signer identities
relative to training are not established. Neither score is a deployment
accuracy guarantee.

Training writes a small PyTorch MLP candidate with `approved_for_production=False`.
No script replaces the live checkpoint. Before promotion, review handshapes and
evaluate unfamiliar signers, both hands, camera orientations, backgrounds,
especially the user's failing H sign. Do not change production just because
training or internal validation accuracy is high.

## Demo release decision

The user explicitly approved deploying the retrained model while deferring
H/K/R improvements. Its release checkpoint uses approved_for_demo=True and
approved_for_production=False. The site opts into demo checkpoints explicitly;
failed quality checks remain recorded, and H/K/R outputs are uncertain.
See HKR-RESULTS.md for the comparison and limits.
