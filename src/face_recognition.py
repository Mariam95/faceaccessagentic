"""SFace face recognition using OpenCV Zoo ONNX models.

Pipeline: YuNet landmarks -> SFace 5-point alignment -> 128-D embedding
-> cosine similarity against multiple enrollment templates.
"""
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import config

class EmbeddingExtractor:
    EMBEDDING_SIZE = 128
    def __init__(self):
        self.model_path = Path(config.SFACE_MODEL_PATH)
        self.available = False
        self.recognizer = None
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"SFace model not found: {self.model_path}")
            self.recognizer = cv2.FaceRecognizerSF.create(
                str(self.model_path), "", cv2.dnn.DNN_BACKEND_OPENCV, cv2.dnn.DNN_TARGET_CPU
            )
            self.available = True
        except Exception as e:
            print(f"[EmbeddingExtractor] {e}")

    def embed(self, frame_bgr: np.ndarray, face) -> Optional[np.ndarray]:
        if not self.available or frame_bgr is None or face is None or face.raw is None:
            return None
        try:
            aligned = self.recognizer.alignCrop(frame_bgr, face.raw)
            feature = self.recognizer.feature(aligned)
            vec = np.asarray(feature, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception as e:
            print(f"[EmbeddingExtractor] embed failed: {e}")
            return None

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return -1.0
        return float(self.recognizer.match(a.reshape(1,-1), b.reshape(1,-1), cv2.FaceRecognizerSF_FR_COSINE))

def match_embedding(query_embedding: np.ndarray, known_embeddings: dict, extractor: EmbeddingExtractor, threshold: float = None):
    threshold = config.FACE_RECOGNITION_THRESHOLD if threshold is None else threshold
    if query_embedding is None or not known_embeddings:
        return None, 0.0
    best_name, best_score = None, -1.0
    for name, templates in known_embeddings.items():
        if isinstance(templates, np.ndarray):
            templates = [templates]
        scores = [extractor.similarity(query_embedding, t) for t in templates]
        score = max(scores) if scores else -1.0
        if score > best_score:
            best_name, best_score = name, score
    return (best_name, best_score) if best_score >= threshold else (None, best_score)
