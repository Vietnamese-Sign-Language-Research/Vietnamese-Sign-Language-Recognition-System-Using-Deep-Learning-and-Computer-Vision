# VSL Studio

A redesigned local workspace for Vietnamese Sign Language recognition. The
interface keeps the original navy, cyan, and violet palette, with self-hosted
Manrope typography, responsive layouts, explicit upload confirmation, native
camera preview, and optional recognition details.

## Run on Windows

For an existing local environment:

```powershell
.\.venv\Scripts\python.exe test.py
```

Open **http://127.0.0.1:8000**. The original `/home.html`, `/upload.html`, and
`/webcam.html` routes open separate Home, Upload, and Live Webcam pages. The
navigation stays visible, and recognition pages focus on their own input and
translation controls. For a fresh environment,
use Python 3.10 or 3.11, create `.venv`, and install `requirements.txt`.

In the GitHub repository, this app lives in `web/`. Run `cd web` first, then
follow the commands below. The app also finds the repository's existing
alphabet and sentence checkpoints in the parent `weights/` folder.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Recognition modes

| Mode | Input | Model and status |
| --- | --- | --- |
| Letters | JPG / PNG / WebP or held webcam sign | Retrained skeleton/geometry ensemble; 23 base letters; demo release, H/K/R experimental |
| Single words | Image or held webcam sign | YOLO adapter implemented; trained word weights are not present |
| Sentences | MP4 / WebM / MOV / AVI or moving webcam sign | Existing ResNet1D checkpoint; 60 phrases plus idle |

### Uploads

1. Select the recognition level.
2. Choose or drop a file, or select a bundled alphabet sample.
3. Review the preview, then press **Translate sign**.

Selection never runs inference. Sentence mode requires a video rather than a
still photo. Files are limited to 30 MB, clips to 45 seconds, and images to
20 megapixels. Unreadable files receive actionable errors; no detected sign
produces a retry state rather than an invented translation.

### Webcam

Enable the camera, press **Start a sign**, sign naturally, then press
**Finish sign** to stop recording and request a prediction. There is no
stillness requirement or automatic six-second countdown. The camera preview
stays available for **Start another sign**. **Turn off camera** cancels a
recording without translating.

The client stops sending new frames when Finish is pressed and waits for the
last in-flight frame before calling the finish endpoint. Letter/word inference
uses the latest clear hand frame, so lowering a hand to click Finish does not
replace the pose with an empty image. Sentence inference receives all valid
frames in the user-selected sequence. Repeated Finish requests reuse the result.

The video preview runs natively in the browser, targeting 30 fps. Recognition
samples unmirrored JPEG frames at at most 6 fps and 640 pixels on the longest
edge, with one request in flight. Slow inference cannot queue camera frames.
Each session owns a MediaPipe tracker. The visible preview is mirrored;
handedness is corrected to the dataset's anatomical left/right convention.
Camera tracks stop on source/model changes, tab hiding, or page exit.

The 45-second capture limit stops recording with a retry message and does not
run inference automatically. It does not split a continuous recording into
fixed prediction windows. Blank/expired
sessions are removed on new-session creation.

### What the sentence model can do

The supplied checkpoint was trained on **individual 126-value vectors**:
21 landmarks × 3 coordinates × 2 hands. It does not learn temporal motion
across frames. The new app preserves that input, collects a complete sign,
uses all valid frames until Finish is pressed, and averages per-frame probability
evidence. Weak or conflicting evidence is marked as a possible match.

This changes segmentation and evidence collection; it does **not** create an
open-vocabulary translator or claim a new accuracy result. Reliable temporal
recognition needs a separately trained sequence model and an evaluated dataset.
Raw weights contain no normalization or class-name metadata, so the app uses
the original project label order and identity normalization. Scores are not
calibrated probabilities of correctness.

### Watch a supported phrase

Select **Sentences → Explore supported signs**, then choose a phrase. The
**With videos** filter lists the three verified public references: **Xin chào.**,
**Cảm ơn.**, and **Bạn tên là gì?** Playback controls support replay and speed
changes; **Try this sign** returns to the recognition controls. Videos stop
when the dialog closes or you return to the phrase list. Other phrases show
an unavailable state until a verified recording is added.

These learning references may use different regional signs from the training
data. Attribution and instructions for extending the catalog are in
`assets/signs/SOURCES.md`. No third-party video is copied or redistributed.

### Connect word weights

Place your project's trained YOLO detection or classification checkpoint at
`weights/word_model.pt`, then restart the server. Alternatively:

```powershell
$env:VSL_WORD_MODEL = 'C:\path\to\your\word_model.pt'
.\.venv\Scripts\python.exe test.py
```

Its class names become the supported word vocabulary automatically. Letter
and sentence paths can also be overridden with `VSL_LETTER_MODEL` and
`VSL_SENTENCE_MODEL`. Use checkpoints from your own trusted training pipeline.

The old root `model.pt` is an HTML Google Drive viewer page saved with a `.pt`
extension, not PyTorch weights. It is preserved but is not used. Alphabet
weights were downloaded from this project's public repository into
`weights/alphabet_model.pt`.

For a repository clone, the demo letter checkpoint is `../weights/letter_skeleton.pt`.
The original YOLO remains at `../weights/alphabet_model.pt` for scoped fallback and rollback.
Set `VSL_LETTER_ENGINE=legacy` to restore the original 22-letter engine.
Other default model paths are `../weights/alphabet_model.pt`
and `../weights/sentence_resnet_model.pt`. Environment overrides take priority;
otherwise the app looks for local checkpoints before the parent weights folder.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite exercises real image inference, video decoding, live tracking,
session lifecycle, left/right ordering, corruption/empty-file rejection,
blank-input rejection, manual sequence boundaries, tracking loss, cancellation,
and absence of predictions before Finish is pressed. The bundled source-dataset
samples recognize **C, B and Y** from the project owner’s photographs. The former D sample was removed because its
source annotation is Đ, outside the checkpoint's vocabulary. These are pipeline checks, not an independent
accuracy benchmark. Physical camera smoothness still depends on the device,
browser, and lighting.

## Files and sources

- `home/home.html`: introduction and project information.
- `upload/upload.html`, `webcam/webcam.html`: dedicated recognition pages.
- `assets/studio.css` and `studio.js`: shared design and recognition controls.
- `test.py`: FastAPI routes, upload decoding, and session lifecycle.
- `recognition.py`: model adapters, phrase evidence, and capture state machine.
- `predict_realtime.py`: original model architecture and label ordering.
- `assets/samples/SOURCES.md`: owner-provided C/B/Y demo photos and historical dataset attribution.
- `assets/fonts/OFL.txt`: Manrope's SIL Open Font License. Fonts load locally.

Project: https://github.com/MortyPham/Vietnamese-Sign-Language-Recognition-System-Using-Deep-Learning-and-Computer-Vision

Research paper: https://docs.google.com/document/d/1FpAOx5w2dIDbhCpRAPXXTfhFj6V8ickP_8FBTIwYfiU/edit

Training dataset: https://universe.roboflow.com/ho-chi-minh-university-of-technology-clmwp/vietnam-sign-language

Sentence extraction reference: https://github.com/khooinguyeen/Vietnamese-Sign-Language-Translation

## Retrained letter demo release

The user-approved retrained model is deployed as a demo release. It supports
23 base letters, including Đ; H, K and R remain experimental and return a
possible-match state. The original model remains available for rollback.

Normal letter recognition loads only the retrained model. A missing or uncertain
match returns a retry/possible-match result directly; it no longer invokes a
second YOLO model, whose additional inference memory could restart Render's
512 MB free instance. Set `VSL_LETTER_ENGINE=legacy` only for an explicit rollback.
Client retry controls also recover from interrupted or incomplete responses
without waiting for session cleanup to finish.

Deployment requirements explicitly select PyTorch 2.8.0 CPU and the matching
torchvision 0.23.0 CPU build from the official PyTorch wheel index. Inference
uses one thread to fit the free host's CPU and memory allocation. Local tests
use the same CPU runtime. The pretrained model tensors are unchanged.

Render start command: `sh web/render-start.sh`. This sets glibc arena/mapping
limits before Python starts. Finished/cancelled recordings explicitly collect
and trim freed native buffers; upload requests also reclaim their temporary
image allocations. One active webcam recording is allowed by default on the
free instance (`VSL_MAX_ACTIVE_CAPTURES` can raise this on larger instances).
Other visitors can still browse and upload; a concurrent recording receives a
retry message instead of allocating another tracking graph. Abandoned captures
expire after the recording limit plus a short grace period.

The checkpoint records approved_for_demo=True and approved_for_production=False.
Failed H/K/R quality checks remain recorded rather than being presented as
passed. See [training/HKR-RESULTS.md](training/HKR-RESULTS.md) and
[training/demo-evaluation.json](training/demo-evaluation.json) for the measured
comparison, source review, and limitations. No user reference photo or raw
third-party dataset is included in the deployed application.

Client capture/race checks (Node.js, no camera required):

```powershell
node --test tests/manual-finish-client.test.cjs
```

Live protocol: POST /api/live/session, stream frames to /api/live/{id}/frame,
then POST /api/live/{id}/finish. DELETE /api/live/{id} cancels/cleans up.
