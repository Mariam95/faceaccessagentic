"""Preflight check for the AI access-control project."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "YuNet detector": ROOT / "models" / "face_detection_yunet_2023mar.onnx",
    "SFace recognizer": ROOT / "models" / "face_recognition_sface_2021dec_int8.onnx",
    "MiniFASNetV2 liveness": ROOT / "models" / "MiniFASNetV2.onnx",
}

def fail(msg):
    print(f"[FAIL] {msg}")
    return False

ok = True
print("=== AI Access Control Preflight Check ===")
print(f"Python: {sys.version.split()[0]}")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
    for attr in ("FaceDetectorYN", "FaceRecognizerSF"):
        if not hasattr(cv2, attr):
            ok = fail(f"OpenCV is missing cv2.{attr}. Install opencv-contrib-python-headless only.") or ok
        else:
            print(f"[OK] cv2.{attr} available")
except Exception as e:
    ok = fail(f"Could not import OpenCV: {e}") or ok

try:
    from importlib import metadata as _md
    _cv = sorted({d.metadata["Name"] for d in _md.distributions() if d.metadata["Name"] and d.metadata["Name"].lower().startswith("opencv")})
    print("Installed OpenCV packages:", ", ".join(_cv) or "none")
    if len(_cv) > 1:
        print("[WARN] Several OpenCV packages are installed; they overwrite each other's cv2 and often break FaceDetectorYN.\n"
              "       Fix:  pip uninstall -y " + " ".join(_cv) + "   then   pip install opencv-contrib-python-headless")
except Exception:
    pass

try:
    import onnxruntime as ort
    print(f"ONNX Runtime: {ort.__version__}")
except Exception as e:
    ok = fail(f"Could not import onnxruntime: {e}") or ok
    ort = None

for label, path in MODEL_FILES.items():
    if not path.exists():
        ok = fail(f"{label} model is missing: {path}") or ok
    elif path.stat().st_size < 100_000:
        ok = fail(f"{label} model looks incomplete ({path.stat().st_size} bytes): {path}") or ok
    else:
        print(f"[OK] {label}: {path.name} ({path.stat().st_size / 1024 / 1024:.2f} MB)")

if ort:
    for label, path in MODEL_FILES.items():
        if not path.exists() or path.stat().st_size < 100_000:
            continue
        try:
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inp = session.get_inputs()[0]
            out = session.get_outputs()[0]
            print(f"[OK] {label} loads: input={inp.shape}, output={out.shape}")
            if label == "MiniFASNetV2 liveness":
                if len(out.shape) != 2 or not isinstance(out.shape[-1], int) or out.shape[-1] < 2:
                    ok = fail(f"MiniFASNetV2 output shape is unexpected: {out.shape}") or ok
                else:
                    print("[OK] MiniFASNetV2 has the expected multi-class output; class index 1 is treated as Real/Live.")
        except Exception as e:
            ok = fail(f"{label} cannot be loaded by ONNX Runtime: {e}") or ok

print("\nRESULT:", "READY" if ok else "NOT READY")
sys.exit(0 if ok else 1)
