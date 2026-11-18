"""Model adapters and user-finished capture, independent of the web interface."""
from __future__ import annotations

import os
import time
from collections import deque
from pathlib import Path
from threading import Lock

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(BASE_DIR / ".cache" / "ultralytics"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)

import cv2
import mediapipe as mp
import numpy as np
import torch
from predict_realtime import CLASS_NAMES_FALLBACK, ResNet1D, load_any_checkpoint, mp_frame_to_126
from memory_runtime import release_native_memory

# The deployed host has a fraction of one CPU. Extra inference threads retain
# additional workspaces and compete for the same small allocation.
torch.set_num_threads(1)
MAX_SEQUENCE_SECONDS = 45.0
SAMPLE_FPS = 6
PHRASES = [
    "Bạn đang làm gì?", "Bạn đi đâu thế?", "Bạn hiểu ngôn ngữ ký hiệu không?",
    "Bạn học lớp mấy?", "Bạn khỏe không?", "Bạn muộn giờ rồi.", "Bạn phải cảnh giác.",
    "Bạn tên là gì?", "Bạn tiến bộ đấy.", "Bạn trông cáu có thế.",
    "Bố mẹ tôi cũng là người Điếc.", "Cái này bao nhiêu tiền?", "Cái này là cái gì?",
    "Cảm ơn.", "Cấp cứu!", "Chúc mừng!",
    "Chúng tôi giao tiếp với nhau bằng ngôn ngữ ký hiệu.", "Con yêu mẹ.",
    "Công việc của bạn là gì?", "Hẹn gặp lại các bạn.", "Món này không ngon.",
    "Tôi bị chóng mặt.", "Tôi bị cướp.", "Tôi bị đau đầu.", "Tôi bị đau họng.",
    "Tôi bị kẹt xe.", "Tôi bị lạc.", "Tôi bị phân biệt đối xử.",
    "Tôi cảm thấy rất hồi hộp.", "Tôi cảm thấy rất vui.", "Tôi cần ăn sáng.",
    "Tôi cần đi vệ sinh.", "Tôi cần gặp bác sĩ.", "Tôi cần phiên dịch.", "Tôi cần thuốc.",
    "Tôi đang ăn sáng.", "Tôi đang buồn.", "Tôi đang ở bến xe.", "Tôi đang ở công viên.",
    "Tôi đang phải cách ly.", "Tôi đang phân vân.", "Tôi đi siêu thị.", "Tôi đi tới Hà Nội.",
    "Tôi đọc kém.", "Tôi khỏi bệnh rồi.", "Tôi không đem theo tiền.", "Tôi không hiểu.",
    "Tôi không quan tâm.", "Tôi là học sinh.", "Tôi là người Điếc.", "Tôi là thợ thêu.",
    "Tôi làm việc ở cửa hàng.", "Tôi nhầm địa chỉ.", "Tôi sống ở Hà Nội.",
    "Tôi thấy đói bụng.", "Tôi thấy nhớ bạn.", "Tôi thích ăn mì.", "Tôi thích phim truyện.",
    "Tôi viết kém.", "Xin chào.", "idle",
]
DISPLAY_LABELS = dict(zip(CLASS_NAMES_FALLBACK, PHRASES))


def model_path(environment_variable, *candidates):
    """Support the standalone interface folder and the repository's web/ layout."""
    override = os.environ.get(environment_variable)
    if override:
        return Path(override)
    return next((path for path in candidates if path.is_file()), candidates[0])


def hand_tracker(static=False):
    return mp.solutions.hands.Hands(static_image_mode=static, max_num_hands=2,
        model_complexity=0, min_detection_confidence=0.55, min_tracking_confidence=0.55)


def tracking_image(rgb):
    height, width = rgb.shape[:2]
    if max(height, width) > 640:
        ratio = 640 / max(height, width)
        rgb = cv2.resize(rgb, (round(width * ratio), round(height * ratio)))
    return np.ascontiguousarray(rgb)


class SequenceCapture:
    """Collect the sequence until Finish sign; motion never ends a capture."""
    def __init__(self, started_at=None):
        self.samples = deque(maxlen=int(MAX_SEQUENCE_SECONDS * SAMPLE_FPS) + 20)
        self.anchor = None
        self.started_at = started_at
        self.movement_count = 0
        self.done = False

    def observe(self, vector, now):
        if self.done:
            return {"phase": "complete"}
        if self.started_at is None:
            self.started_at = now
        elapsed = max(0.0, now - self.started_at)
        state = {"elapsed": round(min(elapsed, MAX_SEQUENCE_SECONDS), 1),
                 "max_seconds": MAX_SEQUENCE_SECONDS}
        if elapsed >= MAX_SEQUENCE_SECONDS:
            self.done = True
            return dict(state, phase="timeout")
        if vector is None or not np.any(vector):
            self.anchor = None
            return dict(state, phase="waiting_hands")
        points = np.asarray(vector, dtype=np.float32).reshape(2, 21, 3)[:, :, :2]
        self.samples.append((now, np.asarray(vector, dtype=np.float32).copy()))
        moving = self.anchor is None
        if self.anchor is not None:
            present = np.any(points != 0, axis=(1, 2))
            old_present = np.any(self.anchor != 0, axis=(1, 2))
            moving = bool(np.any(present != old_present))
            if np.any(present & old_present):
                distances = np.linalg.norm(points[present & old_present] - self.anchor[present & old_present], axis=2)
                moving = moving or float(np.mean(distances)) > 0.018
        if moving:
            self.anchor = points.copy()
            self.movement_count += 1
        return dict(state, phase="capturing")

    def finish(self):
        self.done = True

    def active_vectors(self):
        return [v for _, v in self.samples]


class ModelRegistry:
    def __init__(self):
        self.detectors, self.errors = {}, {}
        self.lock = Lock()
        self.sentence = None
        self.letters = None
        self.class_names = []
        self.mu = np.zeros(126, dtype=np.float32)
        self.sd = np.ones(126, dtype=np.float32)

    def load(self):
        try:
            path = model_path("VSL_SENTENCE_MODEL", BASE_DIR / "tinyresnet1d.pt",
                BASE_DIR / "weights" / "sentence_resnet_model.pt",
                BASE_DIR.parent / "weights" / "sentence_resnet_model.pt")
            ckpt = load_any_checkpoint(path)
            state = ckpt.get("state_dict", ckpt)
            self.class_names = ckpt.get("class_names") or CLASS_NAMES_FALLBACK
            self.sentence = ResNet1D(len(self.class_names), in_ch=1)
            self.sentence.load_state_dict(state, strict=True)
            self.sentence.eval()
            if ckpt.get("mu") is not None and ckpt.get("sd") is not None:
                self.mu = np.asarray(ckpt["mu"], dtype=np.float32).reshape(126)
                self.sd = np.maximum(np.asarray(ckpt["sd"], dtype=np.float32).reshape(126), 1e-8)
        except Exception as exc:
            self.sentence = None
            self.errors["sentences"] = f"Sentence model could not load: {type(exc).__name__}."
        for mode, env, filename in [("letters", "VSL_LETTER_MODEL", "alphabet_model.pt"),
                                    ("words", "VSL_WORD_MODEL", "word_model.pt")]:
            if mode == "letters" and os.environ.get("VSL_LETTER_ENGINE", "auto") != "legacy" and not os.environ.get("VSL_LETTER_MODEL"):
                skeleton_path = model_path("VSL_LETTER_SKELETON", BASE_DIR / "weights" / "letter_skeleton.pt",
                                           BASE_DIR.parent / "weights" / "letter_skeleton.pt")
                if skeleton_path.is_file():
                    try:
                        from letter_skeleton import SkeletonRecognizer
                        self.letters = SkeletonRecognizer(skeleton_path, allow_demo=True)
                        # A rejected pose is a normal result, not a reason to
                        # load/run a second photo model. Its first inference
                        # can exceed the free host's memory limit. The original
                        # detector remains available via VSL_LETTER_ENGINE=legacy.
                        continue
                    except Exception as exc:
                        self.errors[mode] = f"Vietnamese letter model could not load: {type(exc).__name__}."
                        continue
            path = model_path(env, BASE_DIR / "weights" / filename, BASE_DIR.parent / "weights" / filename)
            if not path.is_file():
                if mode != "letters" or self.letters is None:
                    self.errors[mode] = "This model's trained weights have not been added yet."
                continue
            try:
                from ultralytics import YOLO
                detector = YOLO(str(path))
                if detector.task not in {"detect", "classify"}:
                    raise ValueError("Expected a detection or classification checkpoint")
                self.detectors[mode] = detector
            except Exception as exc:
                if mode != "letters" or self.letters is None:
                    self.errors[mode] = f"Model could not load: {type(exc).__name__}."

    def metadata(self):
        result = []
        for mode, title in [("letters", "Letters"), ("words", "Single words"), ("sentences", "Sentences")]:
            detector = self.detectors.get(mode)
            available = detector is not None if mode != "sentences" else self.sentence is not None
            labels = list(detector.names.values()) if detector else [DISPLAY_LABELS.get(x, x) for x in self.class_names if x != "idle"] if mode == "sentences" else []
            if mode == "letters" and self.letters is not None:
                available, labels = True, list(self.letters.labels)
            result.append({"id": mode, "title": title, "available": available, "labels": labels,
                "count": len(labels), "reason": self.errors.get(mode, ""),
                "engine": self.letters.engine if mode == "letters" and self.letters else "resnet1d" if mode == "sentences" else "yolo",
                "release_channel": getattr(self.letters, "release_channel", "production") if mode == "letters" else "production",
                "experimental_labels": getattr(self.letters, "experimental_labels", []) if mode == "letters" else [],
                "accepts": ["video"] if mode == "sentences" else ["image"],
                "description": "Recognize a supported letter from a clear hand photo." if mode == "letters" else
                    "Recognize an isolated sign from a photo." if mode == "words" else
                    "Match a complete sign clip to one of the supported phrases.",
                "note": f"Matches a vocabulary of {len(labels)} learned phrases. Results are suggestions; please confirm their meaning." if mode == "sentences" else
                    f"Recognizes {len(labels)} trained Vietnamese base letters. H, K and R are experimental. Accent and tone marks are not included." if mode == "letters" and self.letters and getattr(self.letters, "experimental_labels", []) else
                    f"Recognizes {len(labels)} trained Vietnamese base letters. Accent and tone marks are not included." if mode == "letters" and self.letters else
                    f"Recognizes the {len(labels)} letters included in the trained checkpoint." if mode == "letters" else
                    "Add a trained single-word YOLO checkpoint to enable recognition."})
        return result

    def available(self, mode):
        return mode in self.detectors or (mode == "letters" and self.letters is not None) or (mode == "sentences" and self.sentence is not None)

    def close(self):
        if self.letters is not None:
            self.letters.close()
            self.letters = None

    def image_result(self, rgb, mode):
        if mode == "letters" and self.letters is not None:
            with self.lock:
                return self.letters.image_result(rgb)
        return self._detector_image_result(rgb, mode)

    def _detector_image_result(self, rgb, mode):
        with self.lock:
            prediction = self.detectors[mode].predict(source=np.ascontiguousarray(rgb[:, :, ::-1]),
                imgsz=640, conf=0.35, verbose=False, device="cpu", max_det=4)[0]
        boxes = []
        if prediction.probs is not None:
            index, score = int(prediction.probs.top1), float(prediction.probs.top1conf)
        elif prediction.boxes is not None and len(prediction.boxes):
            best = int(prediction.boxes.conf.argmax())
            index, score = int(prediction.boxes.cls[best]), float(prediction.boxes.conf[best])
            height, width = rgb.shape[:2]
            for box in prediction.boxes:
                boxes.append({"label": prediction.names[int(box.cls[0])], "score": round(float(box.conf[0]), 4),
                    "bounds": (box.xyxy[0].cpu().numpy() / [width, height, width, height]).tolist()})
        else:
            return {"status": "no_sign", "label": None, "message": "No supported sign found. Try a clearer photo with the whole hand visible."}
        uncertain = score < 0.5
        return {"status": "uncertain" if uncertain else "recognized", "label": prediction.names[index],
            "message": "Possible match. Try again with better lighting to confirm." if uncertain else "Sign recognized.",
            "details": {"model": "YOLO · " + mode, "score": round(score, 4), "boxes": boxes}}

    def sequence_result(self, vectors, movement_count=None):
        if len(vectors) < 4 or (movement_count is not None and movement_count < 3):
            return {"status": "no_sign", "label": None, "message": "A sentence needs a complete moving sign with several clear frames. Record the full phrase and try again."}
        data = np.asarray(vectors, dtype=np.float32)
        if len(data) > 180:
            data = data[np.linspace(0, len(data) - 1, 180).astype(int)]
        data = (data - self.mu) / self.sd
        with self.lock, torch.inference_mode():
            probabilities = torch.softmax(self.sentence(torch.from_numpy(data).unsqueeze(1)), dim=-1).numpy()
        mean = probabilities.mean(axis=0)
        order = np.argsort(mean)[::-1]
        index, score = int(order[0]), float(mean[order[0]])
        label = self.class_names[index]
        agreement = float(np.mean(probabilities.argmax(axis=1) == index))
        details = {"model": "ResNet1D · phrase evidence", "score": round(score, 4), "frames": len(data),
            "agreement": round(agreement, 3), "alternatives": [
                {"label": DISPLAY_LABELS.get(self.class_names[int(i)], self.class_names[int(i)]), "score": round(float(mean[i]), 4)} for i in order[:3]],
            "note": "Model scores are not calibrated accuracy. This checkpoint evaluates hand poses independently; clip evidence is averaged."}
        if label == "idle":
            return {"status": "no_sign", "label": None, "message": "The model detected an idle pose. Try signing the complete phrase again.", "details": details}
        uncertain = score < 0.4 or agreement < 0.4 or score - float(mean[order[1]]) < 0.08
        return {"status": "uncertain" if uncertain else "recognized", "label": DISPLAY_LABELS.get(label, label),
            "message": "Possible phrase match. Please confirm the meaning." if uncertain else "Closest supported phrase.", "details": details}


class LiveSession:
    def __init__(self, mode):
        self.mode = mode
        self.last_seen = time.monotonic()
        self.capture = SequenceCapture(started_at=self.last_seen)
        self.tracker = hand_tracker()
        self.lock = Lock()
        self.result = None
        self.last_frame = self.last_hand_frame = None
        self.closed = False

    def process(self, rgb, registry):
        with self.lock:
            now = time.monotonic()
            self.last_seen = now
            if self.closed:
                return {"phase": "cancelled"}
            if self.capture.done:
                return {"phase": "complete", "result": self.result}
            frame = tracking_image(rgb)
            tracked = self.tracker.process(frame)
            vector = mp_frame_to_126(tracked) if tracked.multi_hand_landmarks else None
            state = self.capture.observe(vector, now)
            state["hands"] = len(tracked.multi_hand_landmarks or [])
            if state["phase"] == "timeout":
                self.result = {"status": "no_sign", "label": None, "message": "Recording reached 45 seconds. Start again and select Finish sign before the limit."}
                state["result"] = self.result
                self._release_capture()
            else:
                # Retain only two bounded-size frames. The last clear hand is
                # useful when a user lowers their signing hand to click Finish.
                self.last_frame = frame.copy()
                if tracked.multi_hand_landmarks:
                    self.last_hand_frame = self.last_frame
            return state

    def finish(self, registry):
        with self.lock:
            self.last_seen = time.monotonic()
            if self.closed:
                return {"phase": "cancelled"}
            if self.result is None:
                self.capture.finish()
                # Recording is over. Release its MediaPipe graph before model
                # inference, without waiting for a browser DELETE request.
                self._close_tracker()
                try:
                    if self.mode == "sentences":
                        self.result = registry.sequence_result(self.capture.active_vectors(), self.capture.movement_count)
                    else:
                        frame = self.last_hand_frame if self.last_hand_frame is not None else self.last_frame
                        self.result = registry.image_result(frame, self.mode) if frame is not None else {
                            "status": "no_sign", "label": None,
                            "message": "No camera frames were recorded. Start a sign, keep your hand in view, then select Finish sign."}
                finally:
                    self._release_capture()
            return {"phase": "complete", "result": self.result}

    def _close_tracker(self):
        if self.tracker is not None:
            tracker = self.tracker
            self.tracker = None
            try:
                tracker.close()
            finally:
                del tracker
                release_native_memory()

    def _release_capture(self):
        self._close_tracker()
        self.last_frame = self.last_hand_frame = None
        self.capture.samples.clear()
        release_native_memory()

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self._release_capture()
