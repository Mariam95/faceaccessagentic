"""Vision toolkit: the repo's existing models, wrapped as plain functions.

Nothing here is new AI. It reuses YuNet (face detection), MiniFASNetV2 (liveness)
and SFace (recognition) from `src/`, and adds one small measurement: head turn,
computed from the 5 face landmarks. The agent's tools call into this class.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

import config
from src.face_detection import FaceDetector, FaceBox, crop_face
from src.liveness import LivenessDetector
from src.face_recognition import EmbeddingExtractor, match_embedding
from src.database import UserDatabase, DatabaseError


@dataclass
class FrameEvidence:
    """Everything the models can say about one camera frame."""
    faces: int = 0
    fully_in_frame: bool = False
    detector_score: float = 0.0
    face_size_px: int = 0
    liveness_label: Optional[str] = None      # LIVE / SPOOF / None (no face)
    live_probability: float = 0.0             # probability of the "real" class
    liveness_probs: List[float] = field(default_factory=list)
    identity: Optional[str] = None            # best match above threshold, else None
    match_score: float = 0.0                  # best cosine score (even if below threshold)
    head_turn: Optional[float] = None         # 0.5 = facing camera, <0.4 / >0.6 = turned
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "faces": self.faces,
            "fully_in_frame": self.fully_in_frame,
            "detector_score": round(self.detector_score, 3),
            "face_size_px": self.face_size_px,
            "liveness_label": self.liveness_label,
            "live_probability": round(self.live_probability, 3),
            "liveness_threshold": config.LIVENESS_THRESHOLD,
            "identity": self.identity,
            "match_score": round(self.match_score, 3),
            "match_threshold": config.FACE_RECOGNITION_THRESHOLD,
            "head_turn": None if self.head_turn is None else round(self.head_turn, 3),
            "error": self.error,
        }


def head_turn_ratio(box: FaceBox) -> Optional[float]:
    """Where the nose sits between the two eyes, from 0 to 1.

    About 0.5 when looking straight at the camera; it moves towards 0 or 1 as the
    head turns. A flat photo cannot change this value: tilting or bending a printed
    picture squeezes the eyes and the nose by the same amount, so the ratio stays put.
    That is why a head-turn challenge is a useful extra liveness test.
    """
    if box is None or box.landmarks is None:
        return None
    lm = box.landmarks
    eye_a_x, eye_b_x = sorted([float(lm[0][0]), float(lm[1][0])])
    span = eye_b_x - eye_a_x
    if span < 1.0:
        return None
    return float((float(lm[2][0]) - eye_a_x) / span)


class VisionToolkit:
    def __init__(self):
        self.detector = FaceDetector()
        self.liveness = LivenessDetector()
        self.embedder = EmbeddingExtractor()
        self.users = UserDatabase()

    @property
    def ready(self) -> Tuple[bool, List[str]]:
        missing = []
        if not self.detector.available:
            missing.append(config.YUNET_MODEL_PATH.name)
        if not self.liveness.available:
            missing.append(config.LIVENESS_MODEL_PATH.name)
        if not self.embedder.available:
            missing.append(config.SFACE_MODEL_PATH.name)
        return (not missing), missing

    # ---------- single-frame analysis ----------
    def primary_face(self, frame) -> Tuple[List[FaceBox], Optional[FaceBox]]:
        boxes = self.detector.detect_faces(frame)
        return boxes, (boxes[0] if boxes else None)

    def analyze(self, frame) -> Tuple[FrameEvidence, Optional[FaceBox], Optional[np.ndarray]]:
        """Run detection, liveness and recognition on one frame.

        Returns (evidence, face_box, embedding). Recognition runs on every frame so the
        agent can see the score, but the access policy only accepts it with liveness.
        """
        ev = FrameEvidence()
        try:
            boxes, face = self.primary_face(frame)
            ev.faces = len(boxes)
            if face is None:
                return ev, None, None
            h, w = frame.shape[:2]
            tx, ty = face.w * config.FACE_EDGE_TOLERANCE, face.h * config.FACE_EDGE_TOLERANCE
            ev.fully_in_frame = not (face.x < -tx or face.y < -ty or
                                     face.x + face.w > w + tx or face.y + face.h > h + ty)
            ev.detector_score = float(face.score)
            ev.face_size_px = int(min(face.w, face.h))
            ev.head_turn = head_turn_ratio(face)

            if self.liveness.available:
                label, _, probs = self.liveness.predict_detailed(frame, face)
                ev.liveness_label = label
                ev.liveness_probs = probs
                ev.live_probability = float(probs[1]) if len(probs) > 1 else 0.0
            else:
                ev.liveness_label = "SPOOF"  # fail closed

            emb = self.embedder.embed(frame, face)
            if emb is not None:
                name, score = match_embedding(emb, self.users.get_all_active_embeddings(), self.embedder)
                ev.identity, ev.match_score = name, float(max(score, 0.0))
            return ev, face, emb
        except (DatabaseError, Exception) as exc:  # never crash the app on one bad frame
            ev.error = f"{type(exc).__name__}: {exc}"
            return ev, None, None

    def same_person(self, emb_a, emb_b) -> float:
        if emb_a is None or emb_b is None:
            return 0.0
        return float(self.embedder.similarity(emb_a, emb_b))

    # ---------- enrollment ----------
    def enrollment_embedding(self, frame) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Same checks as the original /api/register, for one frame."""
        boxes, face = self.primary_face(frame)
        if not boxes:
            return None, "No face found. Face the camera in good light."
        if len(boxes) > 1:
            return None, "More than one face in view. Only one person should be visible."
        ev, face, emb = self.analyze(frame)
        if ev.error:
            return None, ev.error
        if not ev.fully_in_frame:
            return None, "Face is cut off by the edge of the picture. Move to the centre."
        if ev.detector_score < config.FACE_DETECTION_THRESHOLD:
            return None, "Face is not clear enough. Add light or move closer."
        if config.REGISTRATION_REQUIRE_LIVENESS and ev.liveness_label != "LIVE":
            return None, (f"Liveness check said SPOOF (real score {ev.live_probability:.2f}). "
                          "Use even front lighting, no window behind you, 40-80 cm away.")
        if emb is None:
            return None, "Could not compute the face template."
        return emb, None


def decode_image(data: bytes):
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("The picture could not be read.")
    return frame


def annotate(frame, box: Optional[FaceBox], text: str, ok: Optional[bool]):
    """Draw the face box with a coloured verdict, returned as RGB for display."""
    out = frame.copy()
    if box is not None:
        color = (60, 170, 60) if ok else (60, 60, 210) if ok is False else (0, 170, 230)
        cv2.rectangle(out, (box.x, box.y), (box.x + box.w, box.y + box.h), color, 3)
        if box.landmarks is not None:
            for px, py in box.landmarks.astype(int):
                cv2.circle(out, (int(px), int(py)), 3, (255, 200, 0), -1)
        cv2.putText(out, text, (box.x, max(20, box.y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


__all__ = ["VisionToolkit", "FrameEvidence", "head_turn_ratio", "decode_image", "annotate", "crop_face"]
