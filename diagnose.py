"""Step-by-step pipeline diagnosis, no server needed.

    python diagnose.py --camera          # grab a frame from your webcam (press nothing, waits 2 s)
    python diagnose.py photo.jpg         # test a saved image

Shows what each stage decides, and how the liveness model scores your face under every
supported input format, so you can see exactly why registration fails.
"""
import sys
import time
import cv2
import numpy as np
import config
from src.face_detection import FaceDetector
from src.liveness import LivenessDetector
from src.face_recognition import EmbeddingExtractor


def load_frame(argv):
    if "--camera" in argv:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("[FAIL] Could not open the webcam (is another app/browser tab using it?).")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Look at the camera... capturing in 2 seconds")
        end = time.time() + 2
        frame = None
        while time.time() < end:
            ok, f = cap.read()
            if ok:
                frame = f
        cap.release()
        if frame is None:
            sys.exit("[FAIL] Camera opened but returned no frames.")
        cv2.imwrite("diagnose_capture.jpg", frame)
        print("Saved the captured frame as diagnose_capture.jpg (open it - is your whole face visible and well lit?)")
        return frame
    paths = [a for a in argv if not a.startswith("--")]
    if not paths:
        sys.exit(__doc__)
    frame = cv2.imread(paths[0])
    if frame is None:
        sys.exit(f"[FAIL] Cannot read image: {paths[0]}")
    return frame


def main():
    frame = load_frame(sys.argv[1:])
    h, w = frame.shape[:2]
    print(f"\nFrame: {w}x{h}")

    det, live, emb = FaceDetector(), LivenessDetector(), EmbeddingExtractor()
    print(f"YuNet loaded: {det.available} | MiniFASNetV2 loaded: {live.available} | SFace loaded: {emb.available}")
    if not (det.available and live.available and emb.available):
        sys.exit("[FAIL] A model is missing. Run: python download_models.py")

    boxes = det.detect_faces(frame)
    print(f"Faces found: {len(boxes)} (min size {config.MIN_FACE_SIZE}px, detector threshold {config.FACE_DETECTION_THRESHOLD})")
    if not boxes:
        sys.exit("[FAIL] No face detected. Face the camera, add light, move closer/centre.")
    b = boxes[0]
    cut = b.x < 0 or b.y < 0 or b.x + b.w > w or b.y + b.h > h
    print(f"Face box x={b.x} y={b.y} w={b.w} h={b.h} score={b.score:.2f}  -> {'CUT OFF by frame edge!' if cut else 'fully inside frame'}")

    print(f"\nLiveness (class 1 = Real, LIVE needs class 1 to win and be >= {config.LIVENESS_THRESHOLD}):")
    best = None
    for mode in ("bgr255", "rgb255", "bgr01", "rgb01"):
        p = live.probabilities(frame, b, mode)
        verdict = "LIVE" if int(np.argmax(p)) == 1 and p[1] >= config.LIVENESS_THRESHOLD else "SPOOF"
        tag = "  <- current setting" if mode == config.LIVENESS_INPUT_MODE else ""
        print(f"  {mode}: probs={np.round(p, 3).tolist()} -> {verdict}{tag}")
        if best is None or p[1] > best[1]:
            best = (mode, float(p[1]))
    print(f"Highest real-probability: {best[0]} ({best[1]:.2f}).")
    print("If your REAL face is SPOOF in the current mode but clearly LIVE in another, set e.g.\n"
          f"  set LIVENESS_INPUT_MODE={best[0]}   (Windows cmd)   /   export LIVENESS_INPUT_MODE={best[0]}   (Linux/mac)\n"
          "If it is SPOOF in all modes: fix lighting/distance (no backlight, 40-80 cm, even light on the face).")

    e = emb.embed(frame, b)
    print(f"\nEmbedding: {'OK (128-D)' if e is not None else 'FAILED'}")


if __name__ == "__main__":
    main()
