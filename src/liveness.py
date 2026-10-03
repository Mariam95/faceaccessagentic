"""Lightweight MiniFASNetV2 ONNX face anti-spoofing.

Follows the published Silent-Face / yakhyo inference procedure: the face box is
expanded by up to 2.7x (shifted, NOT clipped, when it hits the frame border),
resized to 80x80, passed as float32 BGR/CHW (no mean/std), softmax, class 1 = Real.
"""
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import config


class LivenessDetector:
    """MiniFASNetV2 anti-spoofing gate using ONNX Runtime."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path) if model_path else config.LIVENESS_MODEL_PATH
        self.session = None
        self.input_name = None
        self.output_name = None
        self.input_size = (80, 80)
        self.scale = 2.7
        self.demo_mode = True

        if self.model_path.exists():
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"])
                inp = self.session.get_inputs()[0]
                out = self.session.get_outputs()[0]
                self.input_name = inp.name
                self.output_name = out.name
                shape = inp.shape
                if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                    self.input_size = (int(shape[3]), int(shape[2]))
                self.demo_mode = False
            except Exception as e:
                print(f"[LivenessDetector] Failed to load MiniFASNetV2: {e}")
                self.session = None
                self.demo_mode = True
        else:
            print(f"[LivenessDetector] MiniFASNetV2 model not found: {self.model_path}")

    @property
    def available(self) -> bool:
        return self.session is not None

    def _crop_face(self, frame: np.ndarray, box) -> np.ndarray:
        """Reference crop: scale the box, then SHIFT it back inside the frame.

        Clipping instead of shifting (the old behaviour) produced a distorted,
        off-centre crop whenever the face was near an edge, which made real faces
        look like spoofs.
        """
        h, w = frame.shape[:2]
        bx, by = float(box.x), float(box.y)
        bw, bh = float(max(1, box.w)), float(max(1, box.h))

        scale = min((h - 1) / bh, (w - 1) / bw, self.scale)
        nw, nh = bw * scale, bh * scale
        cx, cy = bx + bw / 2.0, by + bh / 2.0

        x1, y1 = cx - nw / 2.0, cy - nh / 2.0
        x2, y2 = cx + nw / 2.0, cy + nh / 2.0

        if x1 < 0:
            x2 -= x1
            x1 = 0.0
        if y1 < 0:
            y2 -= y1
            y1 = 0.0
        if x2 > w - 1:
            x1 -= x2 - (w - 1)
            x2 = float(w - 1)
        if y2 > h - 1:
            y1 -= y2 - (h - 1)
            y2 = float(h - 1)

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w - 1, int(x2)), min(h - 1, int(y2))
        crop = frame[y1:y2 + 1, x1:x2 + 1]
        if crop.size == 0:
            raise ValueError("MiniFASNet received an empty face crop.")
        return cv2.resize(crop, self.input_size)

    @staticmethod
    def _to_probs(logits: np.ndarray) -> np.ndarray:
        """Softmax the output unless the model already returned probabilities."""
        arr = np.asarray(logits, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        sums = arr.sum(axis=1)
        if np.all(arr >= 0.0) and np.all(arr <= 1.0) and np.allclose(sums, 1.0, atol=1e-3):
            return arr  # already probabilities; a second softmax would flatten them
        arr = arr - np.max(arr, axis=1, keepdims=True)
        exp = np.exp(arr)
        return exp / np.sum(exp, axis=1, keepdims=True)

    @staticmethod
    def _make_tensor(crop: np.ndarray, mode: str) -> np.ndarray:
        img = crop
        if mode.startswith("rgb"):
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        if mode.endswith("01"):
            img = img / 255.0
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None, ...])

    def probabilities(self, frame: np.ndarray, face_box, mode: Optional[str] = None) -> np.ndarray:
        """Raw class probabilities for one face (used by predict_detailed and diagnose.py)."""
        crop = self._crop_face(frame, face_box)
        tensor = self._make_tensor(crop, mode or config.LIVENESS_INPUT_MODE)
        logits = self.session.run([self.output_name], {self.input_name: tensor})[0]
        probs = self._to_probs(logits)
        if probs.shape[1] < 2:
            raise ValueError(f"Unexpected MiniFASNetV2 output shape: {probs.shape}")
        return probs[0]

    def predict_detailed(self, frame: np.ndarray, face_box) -> Tuple[str, float, list]:
        """Return (LIVE/SPOOF, confidence, class_probabilities). Fails closed if unavailable."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("A valid camera frame is required for liveness detection.")
        if face_box is None:
            raise ValueError("A detected face is required for liveness detection.")
        if self.session is None:
            return "SPOOF", 0.0, []
        probs = self.probabilities(frame, face_box)
        predicted_class = int(np.argmax(probs))
        live_prob = float(probs[1])
        is_live = predicted_class == 1 and live_prob >= config.LIVENESS_THRESHOLD
        label = "LIVE" if is_live else "SPOOF"
        confidence = live_prob if is_live else float(probs[predicted_class])
        return label, confidence, [round(float(p), 3) for p in probs]

    def predict(self, frame: np.ndarray, face_box) -> Tuple[str, float]:
        label, confidence, _ = self.predict_detailed(frame, face_box)
        return label, confidence
