import base64
import io
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi import Request
from fastapi.staticfiles import StaticFiles

import config
from src.face_detection import FaceDetector, crop_face, draw_faces
from src.liveness import LivenessDetector
from src.face_recognition import EmbeddingExtractor, match_embedding
from src.database import UserDatabase, AccessLogDatabase, DatabaseError
from src.access_control import decide_access

app = FastAPI(title="AI Access Control System", version="1.0.0")

# Static frontend. Vercel can serve static FastAPI files through its CDN.
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# Models are initialized once per warm function instance.
detector = FaceDetector()
liveness_detector = LivenessDetector()
embedding_extractor = EmbeddingExtractor()
user_db = UserDatabase()
access_log_db = AccessLogDatabase()


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    # Always return JSON so the browser can display the real backend error.
    print(f"[ERROR] {request.method} {request.url.path}: {type(exc).__name__}: {exc}")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {str(exc)}"})


def decode_image(data: bytes):
    if not data:
        raise HTTPException(400, "No image was received.")
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "The uploaded image could not be decoded.")
    return frame


def process_frame(frame, write_log=True):
    boxes = detector.detect_faces(frame)
    if not boxes:
        return {
            "face_detected": False, "faces": 0, "liveness": None,
            "liveness_confidence": 0, "identity": None,
            "recognition_confidence": 0, "access": "DENIED",
            "reason": "No face detected", "image": None,
            "demo_mode": liveness_detector.demo_mode,
            "liveness_available": liveness_detector.available,
        }

    primary = boxes[0]
    face_crop = crop_face(frame, primary)
    label, confidence = None, 0.0
    if liveness_detector.available:
        try:
            label, confidence = liveness_detector.predict(frame, primary)
        except Exception as exc:
            print(f"[LIVENESS ERROR] {type(exc).__name__}: {exc}")
            label, confidence = "ERROR", 0.0
    else:
        label, confidence = "SPOOF", 0.0

    recognized_name, match_score = None, 0.0
    if label == "LIVE" and embedding_extractor.available and face_crop is not None:
        try:
            query_embedding = embedding_extractor.embed(frame, primary)
            recognized_name, match_score = match_embedding(
                query_embedding, user_db.get_all_active_embeddings(), embedding_extractor)
        except (DatabaseError, Exception) as exc:
            print(f"[RECOGNITION ERROR] {type(exc).__name__}: {exc}")
            recognized_name, match_score = None, 0.0

    decision = decide_access(label, confidence, recognized_name, match_score)
    if write_log:
        try:
            access_log_db.log_access(
                decision.user_name, decision.liveness_result,
                decision.liveness_confidence, decision.recognition_result,
                decision.recognition_confidence, decision.access_result)
        except DatabaseError:
            pass

    box_label = label or "Detected"
    if label == "LIVE":
        box_label = recognized_name or "LIVE - Unknown"
    annotated = draw_faces(frame, boxes, label=box_label)
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
    image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii") if ok else None

    return {
        "face_detected": True, "faces": len(boxes),
        "liveness": decision.liveness_result,
        "liveness_confidence": round(float(decision.liveness_confidence), 4),
        "identity": decision.user_name,
        "recognition_confidence": round(float(decision.recognition_confidence), 4),
        "access": decision.access_result, "reason": decision.reason,
        "image": "data:image/jpeg;base64," + image_b64 if image_b64 else None,
        "demo_mode": liveness_detector.demo_mode,
            "liveness_available": liveness_detector.available,
    }

@app.get("/")
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.get("/api/health")
def health():
    missing = [p.name for p in (config.YUNET_MODEL_PATH, config.SFACE_MODEL_PATH, config.LIVENESS_MODEL_PATH)
               if not Path(p).exists()]
    return {"ok": not missing and detector.available and embedding_extractor.available and liveness_detector.available,
            "liveness_model": liveness_detector.available,
            "liveness_model_name": "MiniFASNetV2",
            "face_recognition": embedding_extractor.available, "face_detector": detector.available,
            "recognition_model": "SFace", "detector_model": "YuNet",
            "missing_models": missing}

@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    return process_frame(decode_image(await file.read()))

def check_enrollment_sample(frame):
    """Run every enrollment check on one frame.

    Returns (embedding, None) on success or (None, human_readable_reason) on failure.
    """
    h, w = frame.shape[:2]
    boxes = detector.detect_faces(frame)
    if len(boxes) == 0:
        return None, "No face found (move to the centre of the camera, face the camera, add light)"
    if len(boxes) > 1:
        return None, "More than one face in view (only one person should be visible)"
    face = boxes[0]

    tol_x, tol_y = face.w * config.FACE_EDGE_TOLERANCE, face.h * config.FACE_EDGE_TOLERANCE
    if face.x < -tol_x or face.y < -tol_y or face.x + face.w > w + tol_x or face.y + face.h > h + tol_y:
        return None, "Face is cut off by the edge of the camera view (move to the centre of the frame)"

    if face.score < config.FACE_DETECTION_THRESHOLD:
        return None, f"Face not clear enough (detector score {face.score:.2f})"
    if crop_face(frame, face) is None:
        return None, "Could not crop the face"

    if not liveness_detector.available:
        return None, "Liveness model (MiniFASNetV2.onnx) is not loaded"
    if config.REGISTRATION_REQUIRE_LIVENESS:
        try:
            live_label, live_conf, probs = liveness_detector.predict_detailed(frame, face)
        except Exception as exc:
            print(f"[REGISTRATION LIVENESS ERROR] {type(exc).__name__}: {exc}")
            return None, f"Liveness check crashed: {type(exc).__name__}: {exc}"
        print(f"[REGISTRATION] liveness={live_label} conf={live_conf:.2f} probs={probs}")
        if live_label != "LIVE":
            return None, (f"Liveness said SPOOF (class probabilities {probs}, real needs index 1 >= "
                          f"{config.LIVENESS_THRESHOLD}). Use even front lighting, avoid backlight/window glare, "
                          "sit 40-80 cm away, no photo/screen. Run `python diagnose.py --camera` to test")

    emb = embedding_extractor.embed(frame, face)
    if emb is None:
        return None, "Could not compute the face embedding"
    return emb, None


@app.post("/api/register")
async def register(name: str = Form(...), files: list[UploadFile] = File(...)):
    """Enroll several live captures and store multiple face templates."""
    name = name.strip()
    if not name:
        raise HTTPException(400, "Name is required.")
    if len(name) > 100:
        raise HTTPException(400, "Name is too long (max 100 characters).")
    if len(files) < config.ENROLLMENT_MIN_SAMPLES:
        raise HTTPException(400, f"Please provide at least {config.ENROLLMENT_MIN_SAMPLES} face samples.")

    missing = []
    if not detector.available:
        missing.append(config.YUNET_MODEL_PATH.name)
    if not embedding_extractor.available:
        missing.append(config.SFACE_MODEL_PATH.name)
    if not liveness_detector.available:
        missing.append(config.LIVENESS_MODEL_PATH.name)
    if missing:
        raise HTTPException(
            503, "Model file(s) missing or failed to load: " + ", ".join(missing) +
            ". Run `python download_models.py`, then restart the server.")

    embeddings, reasons = [], []
    for idx, upload in enumerate(files[:config.ENROLLMENT_MAX_SAMPLES], start=1):
        try:
            frame = decode_image(await upload.read())
        except HTTPException as exc:
            reasons.append(f"Sample {idx}: {exc.detail}")
            continue
        emb, reason = check_enrollment_sample(frame)
        if emb is not None:
            embeddings.append(emb)
        else:
            reasons.append(f"Sample {idx}: {reason}")

    if len(embeddings) < config.ENROLLMENT_MIN_SAMPLES:
        # Show each distinct reason once so the message stays readable.
        distinct = list(dict.fromkeys(r.split(": ", 1)[1] for r in reasons))
        raise HTTPException(
            400,
            f"Registration needs {config.ENROLLMENT_MIN_SAMPLES} good samples but only "
            f"{len(embeddings)} passed. Why: " + "; ".join(distinct[:3]) + ".")

    try:
        user_id = user_db.add_user(name, embeddings)
    except (DatabaseError, ValueError) as e:
        raise HTTPException(500, str(e))
    return {"success": True, "user_id": user_id, "name": name, "samples": len(embeddings),
            "rejected": len(reasons)}

@app.get("/api/users")
def users():
    return user_db.list_users()

@app.get("/api/logs")
def logs(limit: int = 50):
    return access_log_db.get_recent(limit)
