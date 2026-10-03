import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Serverless (Vercel) only allows writing to /tmp. Locally, keep data next to the project
# so it works on Windows too (a hard-coded "/tmp" becomes C:\tmp there).
_DEFAULT_DATA_DIR = Path("/tmp") if os.getenv("VERCEL") else BASE_DIR / "data"
DATA_DIR = Path(os.getenv("DATA_DIR", str(_DEFAULT_DATA_DIR)))

LIVENESS_MODEL_PATH = Path(os.getenv("LIVENESS_MODEL_PATH", str(BASE_DIR / "models" / "MiniFASNetV2.onnx")))
YUNET_MODEL_PATH = Path(os.getenv("YUNET_MODEL_PATH", str(BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx")))
SFACE_MODEL_PATH = Path(os.getenv("SFACE_MODEL_PATH", str(BASE_DIR / "models" / "face_recognition_sface_2021dec_int8.onnx")))
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", str(DATA_DIR / "face_access_users.db")))
ACCESS_LOG_PATH = Path(os.getenv("ACCESS_LOG_PATH", str(DATA_DIR / "face_access_logs.db")))

MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "70"))
LIVENESS_THRESHOLD = float(os.getenv("LIVENESS_THRESHOLD", "0.50"))
FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.42"))
FACE_DETECTION_THRESHOLD = float(os.getenv("FACE_DETECTION_THRESHOLD", "0.85"))
FACE_NMS_THRESHOLD = float(os.getenv("FACE_NMS_THRESHOLD", "0.30"))
FACE_TOP_K = int(os.getenv("FACE_TOP_K", "5000"))

# Enrollment: the browser captures several frames; at least MIN of them must pass
# every check (one face, fully inside the frame, LIVE, embedding OK).
ENROLLMENT_MIN_SAMPLES = int(os.getenv("ENROLLMENT_MIN_SAMPLES", "3"))
ENROLLMENT_MAX_SAMPLES = int(os.getenv("ENROLLMENT_MAX_SAMPLES", "8"))
# How much of the face box may stick out of the frame before a sample is rejected (fraction).
FACE_EDGE_TOLERANCE = float(os.getenv("FACE_EDGE_TOLERANCE", "0.03"))

# Liveness model input format. Default matches the reference MiniFASNetV2 ONNX (raw 0-255 BGR).
# If `python diagnose.py --camera` shows REAL faces scoring SPOOF in this mode but LIVE in another,
# set LIVENESS_INPUT_MODE to that one. Options: bgr255, rgb255, bgr01, rgb01.
LIVENESS_INPUT_MODE = os.getenv("LIVENESS_INPUT_MODE", "bgr255").lower()
# Debug switch for enrollment ONLY. Leave "1" in any real deployment: with "0" anyone (or a photo)
# can be enrolled. Daily access checks still always require LIVE.
REGISTRATION_REQUIRE_LIVENESS = os.getenv("REGISTRATION_REQUIRE_LIVENESS", "1") not in ("0", "false", "False")

# ---------------------------------------------------------------------------
# Agentic pipeline (CrewAI) settings
# ---------------------------------------------------------------------------
# Which LLM the security agent uses is picked from the API key you set, in this order:
# GROQ_API_KEY -> Groq (llama-3.3-70b-versatile, free tier; falls back to llama-3.1-8b-instant
# when the free daily limit is reached), XAI_API_KEY -> Grok (grok-4.3),
# ANTHROPIC_API_KEY -> Claude Haiku 4.5, OPENAI_API_KEY -> GPT-4o mini.
# Change the Grok or Groq model with XAI_MODEL / GROQ_MODEL; LLM_MODEL overrides any provider's model name.
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()
# Fast path: clear cases are decided instantly without calling the LLM.
FAST_GRANT_LIVE = float(os.getenv("FAST_GRANT_LIVE", "0.85"))     # real-face probability for instant grant
FAST_GRANT_MATCH = float(os.getenv("FAST_GRANT_MATCH", "0.50"))   # match score for instant grant
FAST_DENY_SPOOF = float(os.getenv("FAST_DENY_SPOOF", "0.10"))     # real-face probability below this = obvious fake
FAST_DENY_UNKNOWN = float(os.getenv("FAST_DENY_UNKNOWN", "0.25")) # live face matching nobody this well = stranger
# Challenges the agent may ask for in one access attempt before it must decide.
MAX_CHALLENGES = int(os.getenv("MAX_CHALLENGES", "2"))
# How much the nose-between-eyes ratio must change to count as a real head turn.
HEAD_TURN_MIN_CHANGE = float(os.getenv("HEAD_TURN_MIN_CHANGE", "0.12"))
AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))
