"""Download the face models used by the project.
Run: python download_models.py
"""
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

ROOT = Path(__file__).resolve().parent / "models"
ROOT.mkdir(exist_ok=True)
FILES = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        100_000,
    ),
    "face_recognition_sface_2021dec_int8.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8.onnx",
        1_000_000,
    ),
    "MiniFASNetV2.onnx": (
        "https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",
        1_000_000,
    ),
}

for name, (url, min_size) in FILES.items():
    out = ROOT / name
    if out.exists() and out.stat().st_size >= min_size:
        print(f"[OK] {name} ({out.stat().st_size / 1024 / 1024:.1f} MB)")
        continue
    print(f"[DOWNLOAD] {name}")
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = out.with_suffix(out.suffix + ".part")
    try:
        with urlopen(req, timeout=120) as response, open(tmp, "wb") as f:
            f.write(response.read())
        size = tmp.stat().st_size
        if size < min_size:
            raise RuntimeError(f"Downloaded file is only {size} bytes; expected at least {min_size} bytes. The download may have returned an error page.")
        tmp.replace(out)
        print(f"[OK] {out} ({size / 1024 / 1024:.1f} MB)")
    except (HTTPError, URLError, TimeoutError, OSError, RuntimeError) as e:
        if tmp.exists():
            tmp.unlink()
        raise SystemExit(
            f"\n[ERROR] Could not download {name}.\n{e}\n\n"
            f"Check your internet connection and run this command again:\n"
            f"    python download_models.py\n"
        )

print("All required models are ready.")
