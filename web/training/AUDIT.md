# Latest status

The user explicitly approved deploying the corrected model as a **demo**,
deferring H/K/R improvements. Full production quality remains unapproved;
failed checks are preserved. See `HKR-RESULTS.md` for the release decision.
Earlier MLP and mixed-form scores are historical diagnostics, not national-H proof.

# Letter-model audit and retraining outcome

Audit date: 2026-11-18. Production was not changed.

## What the evidence establishes

The original training notebook references Ho Chi Minh University of Technology's
Roboflow `vietnam-sign-language` project, version 7. Its name alone cannot prove
that every frame has the correct Vietnamese handshape or label.

Directly inspected original examples:

- H: `youtube-28.jpg`, image `0M6oEbM1qGZPTgAhff6A`, tagged
  `youtube:ggQY-g4aQp8`, visibly from Trung tâm đào tạo Ngôn ngữ ký hiệu Hà Nội.
  https://universe.roboflow.com/ho-chi-minh-university-of-technology-clmwp/vietnam-sign-language/images/0M6oEbM1qGZPTgAhff6A
- M: `23.jpg`, image `0MWzGUEUlycSrnKpbmHk`, shows three fingers pointing down,
  consistent with the Vietnamese reference rather than the ASL closed-fist M.
  https://universe.roboflow.com/ho-chi-minh-university-of-technology-clmwp/vietnam-sign-language/images/0MWzGUEUlycSrnKpbmHk
- The image used by the app as sample letter D is labeled **Đ** by the original
  source. The old checkpoint itself has no Đ class. This is a semantic mismatch
  that a successful pipeline smoke test cannot resolve.
  https://universe.roboflow.com/ho-chi-minh-university-of-technology-clmwp/vietnam-sign-language/images/0Bfya8NVK6Qm9WXXu28l

There is therefore insufficient evidence to call the entire original dataset
ASL. However, it is also incorrect to certify its training labels merely from
the dataset title. The national-chart H displayed by the app is visually
inconsistent with sampled H teaching/training photographs. Regional variants,
transitional frames, labeling mistakes, and hand-tracker errors remain possible
explanations; a fluent VSL signer should resolve those distinctions.

Reference: Ministry of Education and Training, Circular 17/2020/TT-BGDĐT,
gazette page 51 (PDF page 6):
https://congbaocdn.chinhphu.vn/CongBaoCP/VanBan/2020/6/31603/31697-1-2020671-67217-2020-tt-bgddt.pdf#page=6

Tran et al. (2025), Figure 4, directly compares VSL and ASL handshapes and
identifies differences including H, K, M, N, R, S, and T:
https://jst.vn/index.php/ssad/article/download/56/19

## Actual retraining performed

Downloaded Mạc Phương Nga's public DATASET_VSL version 2 (Apache 2.0):
https://www.kaggle.com/datasets/mcphngnga/dataset-vsl

Archive SHA-256:
`5d12d6743d4dda307e141777b47a3bab3d56db6ab5a4b50a7158fe925e0269e0`

- Archive contains 26,000 images across 26 folders.
- Selected 23 base-letter classes; explicitly mapped `DD` to `Đ`.
- Excluded `AA`, `AAA`, `OOO`; their accent meanings need separate review.
- Sampled 400 images per class, 9,200 total, seed 20261118.
- Extracted usable MediaPipe landmarks from 4,305 unique images. Failed tracking
  is recorded; those images were not assigned invented landmarks or predictions.
- Final filename-group split: 3,110 training, 555 validation, 640 test.
- Train/validation/test groups are disjoint and every class has evaluation
  coverage. This is **not** a signer-independent split.
- Trained an MLP with coordinates, pairwise 2D/3D distances, and finger-joint
  angles, including mirror and modest 3D rotation augmentation. CPU only.
- Final checkpoint: `artifacts/alphabet-audit/prepared/letter_candidate.pt`.
  Its `approved_for_production` field is **false**.

The candidate's dataset also includes a Hanoi-style H. It has not been
established as a complete, uniformly correct national-standard dataset.

## Evaluation

The external collection is a separate public repository, pinned to commit
`2e44477b2939f14be2fbfd222191af9e617448f4`:
https://github.com/TuanKiet1774/SignLanguage_Vietnamese

Sampled 460 images (20 per class); after tracking failures and duplicate removal,
454 remain in classifier comparison. Images are evaluation-only; this source's
license is not specified. Do not redistribute its photos with the app.

| Check | Existing YOLO | Retrained candidate v2 |
| --- | ---: | ---: |
| Separate-source accuracy, 23-class scope | 30.84% | 86.56% |
| Same 22 letters supported by both | 32.26% | 87.10% |
| H correctly classified, 20 tracked examples | 0/20 | 10/20 |

The old model has no Đ output, so the 22-letter row is the fairer comparison.
Candidate internal test accuracy is 88.13%; internal validation is 89.01%.
The final candidate still scores 0/20 on K and 0/20 on R in the external
collection. External H errors are classified as T. Some apparent errors may
reflect inconsistent sign labels or variants rather than model errors; they
require visual ground-truth review before further training or relabeling.

These percentages are diagnostic results, not population accuracy estimates:
the external source contains correlated video frames; signer overlap with the
training source is unknown; the external results informed feature revision
and therefore no longer constitute an untouched final test.

Detailed JSON reports, preparation manifests, and the experimental checkpoint
are saved under `artifacts/alphabet-audit/`. Training scripts are in `training/`.

## Decision and next required input

**Do not replace the production checkpoint with this candidate.** A better
aggregate score does not compensate for failing the user's reported H case or
unresolved signing-label discrepancies.

Obtain a photo/video of the H the user actually performs. Check it against a
fluent VSL signer or a clearly identified Vietnamese signing reference. Establish
the intended variety/standard, curate consistent H/K/R examples and the D/Đ
mapping, then retrain and evaluate unfamiliar signers on an untouched test set.
Do not fix this by renaming ASL labels or by forcing a hand-pose rule to output H.


## Follow-up H/K/R audit — 18 November 2026

The earlier geometry candidate is not the final corrected model. Closer visual
review found its K/R labels incompatible with the national chart. Its H label
and the supplemental Hauuto/Ngan H videos also do not match the national-chart
H (the supplemental clips raise three fingers with thumb/finger contact).
They must not be used as proof that national H recognition succeeds.

Additional sources inspected:

- cuongnk9104/vsl-v2, version 1, 58,439 skeleton JPEGs, 25 folders. The national
  base-letter subset has 23 classes. The publisher supplies no explicit license.
  Archive SHA256: f6af87c8bde415c3b7514c183698581b47ef959804e6bff6b2c0b24bdc82f353.
- hauuto/vietnamese-sign-language-alphabet, version 2. Reviewed raw K recordings
  from Hau/Tai are training data, Khoi K validation, Vy K diagnostic evaluation.
  Archive SHA256: 4998dd42d3ecd825c4aa885c5b387829dd27678c91035a1a96dda07debda40c7.
  All raw H and R from this source are excluded from the national candidate.
- ngannguyen65/vietnamese-sign-language. Reviewed K clips 0/40 train, 19/79
  diagnostic evaluation. Its H and crossed-finger R are excluded.
- HGM-4 paper Table 2 was checked visually. Despite its Vietnamese-language
  description, its displayed H/R handshapes do not match the chosen national
  chart. Do not treat its dataset title as certification or use those labels.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7229479/

The first skeleton CNN scored 99.67% internally but failed raw K (1/20 on the
external diagnostic). Mixed candidate v3 scored 99.17% internally and 80.6%
on a second set of K recordings, while regressing the existing X thumbnail.
Its apparent 100% supplemental H result tested the incompatible H form and
must not be cited as national-H accuracy. Neither candidate is approved.

Candidate v4 removes ALL supplemental H. H/R training comes from the reviewed
national-chart skeleton collection. K adds reviewed raw recordings and camera
rotation augmentation, generated only from training clips. Non-H/K/R poses
from the earlier corpus supplement the remaining letters. The auxiliary
geometry model is prohibited from producing H/K/R; the original YOLO fallback
is also prohibited from producing H/K/R or the ambiguous D mapping.

These source collections have unknown signer overlap. Adjacent video frames
are correlated. Several diagnostic sets informed revisions, so they are now
validation diagnostics rather than untouched test sets. No supplied success
percentage establishes performance on unseen VSL users.

**Promotion requires matching real-person national-H examples, raw K/R checks,
other-letter regressions, and blank-input rejection. Missing raw H evaluation
must fail the release gate. No model is copied into active weights automatically.**


### User-supplied H reference

The user supplied a new H photograph after the training runs. It was copied
only to ignored local evaluation artifacts and was NOT included in training.
The current original detector reports no sign. Candidate v4 with the K
specialist outputs K (score about 0.973), exposing a serious H-to-K confusion.
The standalone CNN is also uncertain (top D about 0.284); the K specialist
converts that uncertainty into a wrong confident answer. This is a failed
release candidate. The original video/source has been requested to establish
clearer finger positions and obtain further verified examples. Do not promote
it or treat the nominal 98.8% internal score as real H performance.

The K specialist uses a validation-selected operating threshold (maximum K
recall with at most 0.5% validation negatives marked K). It reaches only 75%
validation K recall despite 98.6% on tracked held-out K frames. That discrepancy
is another release failure, not a reason to certify the model from test alone.


### User-authorized demo release

After reviewing the H failure, the user clarified: deploy the retrained model
as-is for demonstrations of other letters, and defer H/K/R work. The release
records this separately as approved_for_demo=True, approved_for_production=False,
with failed checks intact. H/K/R predictions are displayed as experimental
possible matches. The original YOLO is retained for rollback. No additional
training was performed on the user H photograph.
