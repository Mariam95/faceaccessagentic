# Model files

Required runtime models:

- `face_detection_yunet_2023mar.onnx` — YuNet face detector + 5 landmarks
- `face_recognition_sface_2021dec_int8.onnx` — SFace face-recognition embedding model
- `MiniFASNetV2.onnx` — MiniFASNetV2 face anti-spoofing model

Run `python download_models.py` from the project root to download all three.
