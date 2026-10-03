"""Lightweight YuNet face detector with 5-point landmarks.

YuNet is used instead of Haar because face recognition needs stable facial
landmarks for alignment. The detector returns a FaceBox compatible with the
rest of the application.
"""
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import cv2
import numpy as np
import config

@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    landmarks: Optional[np.ndarray] = None
    score: float = 0.0
    raw: Optional[np.ndarray] = None

    @property
    def area(self) -> int:
        return self.w * self.h

class FaceDetector:
    def __init__(self, min_face_size: int = None):
        self.min_face_size = min_face_size or config.MIN_FACE_SIZE
        self.model_path = Path(config.YUNET_MODEL_PATH)
        self.detector = None
        if self.model_path.exists():
            self.detector = cv2.FaceDetectorYN.create(
                str(self.model_path), "", (320, 320),
                config.FACE_DETECTION_THRESHOLD,
                config.FACE_NMS_THRESHOLD,
                config.FACE_TOP_K,
            )
        else:
            print(f"[FaceDetector] YuNet model not found: {self.model_path}")

    @property
    def available(self) -> bool:
        return self.detector is not None

    def detect_faces(self, frame: np.ndarray) -> List[FaceBox]:
        if frame is None or frame.size == 0 or self.detector is None:
            return []
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, detections = self.detector.detect(frame)
        if detections is None:
            return []
        boxes = []
        for row in detections:
            x, y, bw, bh = map(int, row[:4])
            score = float(row[14])
            if bw < self.min_face_size or bh < self.min_face_size:
                continue
            landmarks = np.asarray(row[4:14], dtype=np.float32).reshape(5, 2)
            boxes.append(FaceBox(x, y, bw, bh, landmarks, score, row.astype(np.float32)))
        boxes.sort(key=lambda b: b.area, reverse=True)
        return boxes

def crop_face(frame: np.ndarray, box: FaceBox, margin: float = 0.15) -> Optional[np.ndarray]:
    if frame is None or box is None:
        return None
    h, w = frame.shape[:2]
    mx, my = int(box.w * margin), int(box.h * margin)
    x1, y1 = max(0, box.x-mx), max(0, box.y-my)
    x2, y2 = min(w, box.x+box.w+mx), min(h, box.y+box.h+my)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()

def draw_faces(frame, boxes: List[FaceBox], label: str = "Detected") -> np.ndarray:
    out = frame.copy()
    for box in boxes:
        cv2.rectangle(out, (box.x, box.y), (box.x+box.w, box.y+box.h), (0, 200, 0), 2)
        if box.landmarks is not None:
            for px, py in box.landmarks.astype(int):
                cv2.circle(out, (px, py), 2, (255, 180, 0), -1)
        cv2.putText(out, label, (box.x, max(0, box.y-10)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,200,0), 2)
    return out
