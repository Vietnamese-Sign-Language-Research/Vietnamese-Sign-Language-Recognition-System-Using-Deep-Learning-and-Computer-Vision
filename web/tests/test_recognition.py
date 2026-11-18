"""Behavioral regression tests for inference contracts and sign boundaries."""
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from recognition import LiveSession, SequenceCapture
from predict_realtime import mp_frame_to_126
from test import app, decode_image, registry


def hand_vector(offset=0):
    vector = np.zeros((2, 21, 3), dtype=np.float32)
    vector[0, :, 0] = 0.3 + offset
    vector[0, :, 1] = 0.4
    return vector.flatten()


class CaptureTests(unittest.TestCase):
    def test_stillness_never_finishes_capture(self):
        capture = SequenceCapture()
        for timestamp in np.arange(0, 20, .2):
            self.assertEqual(capture.observe(hand_vector(), float(timestamp))["phase"], "capturing")
        self.assertFalse(capture.done)
        capture.finish()
        self.assertTrue(capture.done)

    def test_motion_tracking_loss_and_network_gaps_do_not_finish(self):
        capture = SequenceCapture()
        for timestamp, vector in [(0, hand_vector()), (1, None), (12, hand_vector(.1)), (30, hand_vector(.2))]:
            state = capture.observe(vector, timestamp)
            self.assertNotEqual(state["phase"], "complete")
        self.assertFalse(capture.done)
        self.assertEqual(len(capture.active_vectors()), 3)

    def test_manual_capture_keeps_all_valid_frames_including_tail(self):
        capture = SequenceCapture()
        for index in range(20):
            capture.observe(hand_vector(.08 * (index % 2)) if index < 8 else hand_vector(), index / 6)
        capture.finish()
        self.assertEqual(len(capture.active_vectors()), 20)
        np.testing.assert_equal(capture.active_vectors()[-1], hand_vector())

    def test_capture_limit_does_not_depend_on_detecting_hands(self):
        capture = SequenceCapture()
        capture.observe(None, 0)
        self.assertEqual(capture.observe(None, 45)["phase"], "timeout")
        self.assertTrue(capture.done)

    def test_unmirrored_handedness_matches_dataset_order(self):
        points = [SimpleNamespace(x=.2, y=.4, z=.01) for _ in range(21)]
        results = SimpleNamespace(multi_hand_landmarks=[SimpleNamespace(landmark=points)],
            multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label="Right")])])
        raw = mp_frame_to_126(results)
        selfie = mp_frame_to_126(results, mirrored=True)
        self.assertTrue(np.any(raw[:63]))
        self.assertFalse(np.any(raw[63:]))
        self.assertFalse(np.any(selfie[:63]))
        self.assertTrue(np.any(selfie[63:]))

    def test_letter_prediction_waits_for_finish_and_runs_once(self):
        tracker = Mock()
        tracker.process.return_value = SimpleNamespace(multi_hand_landmarks=[object()])
        model = Mock()
        model.image_result.return_value = {"status": "recognized", "label": "A"}
        with patch("recognition.hand_tracker", return_value=tracker), patch("recognition.mp_frame_to_126", return_value=hand_vector()), patch("recognition.time.monotonic", return_value=0):
            session = LiveSession("letters")
            frame = np.zeros((32, 32, 3), dtype=np.uint8)
            for timestamp in np.arange(0, 12, .2):
                with patch("recognition.time.monotonic", return_value=float(timestamp)):
                    response = session.process(frame, model)
                self.assertNotIn("result", response)
                model.image_result.assert_not_called()
            result = session.finish(model)
            self.assertEqual(result["result"]["label"], "A")
            self.assertEqual(session.finish(model), result)
            self.assertEqual(session.process(frame, model), result)
            model.image_result.assert_called_once()
            session.close()

    def test_finish_uses_last_hand_frame_when_hand_is_lowered_to_click(self):
        tracker = Mock()
        tracker.process.side_effect = [SimpleNamespace(multi_hand_landmarks=[object()]), SimpleNamespace(multi_hand_landmarks=None)]
        model = Mock()
        with patch("recognition.hand_tracker", return_value=tracker), patch("recognition.mp_frame_to_126", return_value=hand_vector()):
            session = LiveSession("letters")
            sign = np.full((32,32,3), 180, dtype=np.uint8)
            session.process(sign, model)
            session.process(np.zeros_like(sign), model)
            session.finish(model)
            np.testing.assert_equal(model.image_result.call_args.args[0], sign)
            session.close()

    def test_sentence_finishes_only_on_request_with_the_complete_sequence(self):
        tracker = Mock()
        tracker.process.return_value = SimpleNamespace(multi_hand_landmarks=[object()])
        model = Mock()
        model.sequence_result.return_value = {"status":"recognized", "label":"Example phrase"}
        with patch("recognition.hand_tracker", return_value=tracker), patch("recognition.mp_frame_to_126", side_effect=[hand_vector(.08 * (i % 2)) for i in range(15)]):
            session = LiveSession("sentences")
            for _ in range(15):
                response = session.process(np.zeros((32,32,3),dtype=np.uint8), model)
                self.assertNotIn("result", response)
            model.sequence_result.assert_not_called()
            result = session.finish(model)
            self.assertEqual(result["result"]["label"], "Example phrase")
            vectors, motion_count = model.sequence_result.call_args.args
            self.assertEqual(len(vectors), 15)
            self.assertGreaterEqual(motion_count, 3)
            model.image_result.assert_not_called()
            self.assertEqual(session.finish(model), result)
            model.sequence_result.assert_called_once()
            session.close()

    def test_finish_empty_capture_does_not_invent_a_letter(self):
        with patch("recognition.hand_tracker"):
            session = LiveSession("letters")
            model = Mock()
            self.assertEqual(session.finish(model)["result"]["status"], "no_sign")
            model.image_result.assert_not_called()
            session.close()

    def test_safety_limit_stops_without_running_inference(self):
        with patch("recognition.hand_tracker") as tracker, patch("recognition.time.monotonic", return_value=0):
            tracker.return_value.process.return_value = SimpleNamespace(multi_hand_landmarks=None)
            session = LiveSession("letters")
            model = Mock()
            with patch("recognition.time.monotonic", return_value=46):
                result = session.process(np.zeros((32,32,3),dtype=np.uint8), model)
            self.assertEqual(result["phase"], "timeout")
            self.assertEqual(session.finish(model)["result"]["status"], "no_sign")
            model.image_result.assert_not_called()
            session.close()

    def test_cancel_clears_evidence_without_a_prediction(self):
        with patch("recognition.hand_tracker"):
            session = LiveSession("letters")
            model = Mock()
            session.close()
            self.assertEqual(session.finish(model)["phase"], "cancelled")
            model.image_result.assert_not_called()


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = TestClient(app)
        cls.client = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_loaded_models_and_real_dataset_images(self):
        modes = {item["id"]: item for item in self.client.get("/api/models").json()["models"]}
        # Unapproved candidates must leave the existing 22-letter model active.
        corrected = modes["letters"]["engine"] == "vsl-corrected-ensemble-v1"
        self.assertEqual(modes["letters"]["count"], 23 if corrected else 22)
        if corrected:
            self.assertIn("Đ", modes["letters"]["labels"])
            self.assertEqual(modes["letters"]["release_channel"], "demo")
            self.assertEqual(modes["letters"]["experimental_labels"], ["H", "K", "R"])
        self.assertTrue(modes["sentences"]["available"])
        for name, expected in [("sample-c.jpg", "C"), ("sample-b.jpg", "B"), ("sample-y.jpg", "Y")]:
            content = (Path(__file__).resolve().parents[1] / "assets" / "samples" / name).read_bytes()
            response = self.client.post("/api/upload", data={"mode": "letters"}, files={"file": (name, content, "image/jpeg")})
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["label"], expected)

    def test_sentence_photo_has_actionable_error(self):
        response = self.client.post("/api/upload", data={"mode": "sentences"}, files={"file": ("photo.jpg", b"data", "image/jpeg")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("video", response.json()["detail"])

    def test_demonstrations_match_supported_sentence_labels(self):
        from urllib.parse import urlparse
        response = self.client.get("/assets/signs/sentence-videos.json")
        self.assertEqual(response.status_code, 200)
        videos = response.json()["demonstrations"]
        sentence_model = next(model for model in self.client.get("/api/models").json()["models"] if model["id"] == "sentences")
        labels = [video["label"] for video in videos]
        self.assertGreater(len(labels), 0)
        self.assertEqual(len(labels), len(set(labels)))
        self.assertTrue(set(labels).issubset(sentence_model["labels"]))
        for video in videos:
            self.assertEqual(video["provider"], "youtube")
            self.assertRegex(video["video_id"], r"^[A-Za-z0-9_-]{11}$")
            self.assertEqual(urlparse(video["source_url"]).scheme, "https")
            self.assertIn(urlparse(video["source_url"]).hostname, {"tokyolife.vn", "talkingdictionary.swarthmore.edu"})
            self.assertEqual(video["watch_url"], "https://www.youtube.com/watch?v=" + video["video_id"])

    def test_corrupt_image_rejected(self):
        response = self.client.post("/api/upload", data={"mode": "letters"}, files={"file": ("photo.jpg", b"not an image", "image/jpeg")})
        self.assertEqual(response.status_code, 400)

    def test_empty_upload_rejected(self):
        response = self.client.post("/api/upload", files={"file": ("photo.png", b"", "image/png")})
        self.assertEqual(response.status_code, 400)

    def test_blank_image_does_not_invent_translation(self):
        content = io.BytesIO()
        Image.new("RGB", (200, 200), "black").save(content, format="PNG")
        response = self.client.post("/api/upload", files={"file": ("blank.png", content.getvalue(), "image/png")})
        self.assertEqual(response.json()["status"], "no_sign")
        self.assertIsNone(response.json()["label"])

    def test_actual_video_decode_with_no_hands(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blank.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (128, 128))
            self.assertTrue(writer.isOpened())
            for _ in range(24):
                writer.write(np.zeros((128, 128, 3), dtype=np.uint8))
            writer.release()
            response = self.client.post("/api/upload", data={"mode": "sentences"}, files={"file": ("blank.mp4", path.read_bytes(), "video/mp4")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["status"], "no_sign")

    def test_session_lifecycle_and_blank_frame(self):
        response = self.client.post("/api/live/session", data={"mode": "letters"})
        self.assertEqual(response.status_code, 200)
        key = response.json()["id"]
        content = io.BytesIO()
        Image.new("RGB", (128, 128), "black").save(content, format="JPEG")
        response = self.client.post(f"/api/live/{key}/frame", files={"file": ("frame.jpg", content.getvalue(), "image/jpeg")})
        self.assertEqual(response.json()["phase"], "waiting_hands")
        self.assertNotIn("result", response.json())
        self.assertEqual(self.client.delete(f"/api/live/{key}").status_code, 200)
        self.assertEqual(self.client.post(f"/api/live/{key}/frame", files={"file": ("frame.jpg", content.getvalue(), "image/jpeg")}).status_code, 404)

    def test_real_camera_frames_are_predicted_only_after_manual_finish(self):
        content = (Path(__file__).resolve().parents[1] / "assets" / "samples" / "sample-y.jpg").read_bytes()
        response = self.client.post("/api/live/session", data={"mode":"letters"})
        self.assertEqual(response.json()["capture_mode"], "manual")
        key = response.json()["id"]
        try:
            for _ in range(5):
                response = self.client.post(f"/api/live/{key}/frame", files={"file":("frame.jpg",content,"image/jpeg")})
                self.assertEqual(response.status_code, 200)
                self.assertNotIn("result", response.json())
            result = self.client.post(f"/api/live/{key}/finish")
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual(result.json()["result"]["label"], "Y")
            self.assertEqual(self.client.post(f"/api/live/{key}/finish").json(), result.json())
        finally:
            self.client.delete(f"/api/live/{key}")
        self.assertEqual(self.client.post(f"/api/live/{key}/finish").status_code,404)

    def test_finishing_blank_sentence_capture_returns_retry_instead_of_a_phrase(self):
        key=self.client.post("/api/live/session",data={"mode":"sentences"}).json()["id"]
        try:
            result=self.client.post(f"/api/live/{key}/finish")
            self.assertEqual(result.status_code,200)
            self.assertEqual(result.json()["result"]["status"],"no_sign")
            self.assertIsNone(result.json()["result"]["label"])
        finally:
            self.client.delete(f"/api/live/{key}")


    def test_navigation_opens_distinct_home_upload_and_webcam_pages(self):
        for path, page in [("/", "home"), ("/home.html", "home"), ("/upload.html", "upload"), ("/webcam.html", "webcam")]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertIn("VSL Studio", response.text)
            self.assertNotIn("iframe", response.text)
            self.assertIn(f'data-page="{page}"', response.text)
            if page == "home":
                self.assertNotIn('id="translate-button"', response.text)
            else:
                self.assertNotIn('id="about"', response.text)


if __name__ == "__main__":
    unittest.main()
