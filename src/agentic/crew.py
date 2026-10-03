"""The CrewAI security agent.

One agent ("Access Security Officer") gets the facts of an unclear access attempt and
decides what to do by calling tools that wrap the existing vision models:

    inspect_face          -> detection + liveness + identity + head turn for one photo
    verify_head_turn      -> did the person really turn their head between two photos?
    check_access_history  -> is this user blocked, and have there been recent spoof attempts?

It ends with one of three decisions: GRANTED, DENIED, or CHALLENGE (ask the person to do
something, e.g. turn their head, and take another photo). The pipeline then checks that
decision against hard safety rules, so the LLM can never open the door on its own.
"""
import json
import os
import threading
from typing import Literal, Optional

from pydantic import BaseModel, Field

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

from crewai import Agent, Crew, LLM, Process, Task  # noqa: E402
from crewai.tools import tool  # noqa: E402

import config  # noqa: E402
from src.agentic.case import AccessCase  # noqa: E402

# ---------------------------------------------------------------------------
# Shared state for tools. CrewAI tools are plain functions, so they read the case
# being investigated from here. A lock makes sure only one case is investigated at a time.
# ---------------------------------------------------------------------------
_RUN_LOCK = threading.Lock()
_ACTIVE: dict = {"case": None, "toolkit": None, "logs": None}


def _trace(name: str, args: dict, result: dict) -> str:
    case: AccessCase = _ACTIVE["case"]
    if case is not None:
        case.trace.append({"tool": name, "input": args, "output": result})
    return json.dumps(result)


@tool("inspect_face")
def inspect_face(frame_id: str) -> str:
    """Look at one photo from this access attempt. Returns: number of faces, whether the
    face is fully inside the picture, live_probability (how likely the face is real rather
    than a photo or screen; LIVE needs >= liveness_threshold), identity (registered user
    matched, or null) with match_score vs match_threshold, and head_turn (0.5 = facing the
    camera; values far from 0.5 mean the head is turned). Input: a frame_id like 'ab12cd34-f1'."""
    case: AccessCase = _ACTIVE["case"]
    rec = case.frame(frame_id.strip()) if case else None
    if rec is None:
        known = [f.frame_id for f in case.frames] if case else []
        return _trace("inspect_face", {"frame_id": frame_id}, {"error": f"Unknown frame_id. Valid ids: {known}"})
    out = rec.evidence.to_dict()
    out["frame_id"] = rec.frame_id
    out["role"] = rec.role
    return _trace("inspect_face", {"frame_id": frame_id}, out)


@tool("verify_head_turn")
def verify_head_turn(before_frame_id: str, after_frame_id: str) -> str:
    """Check a head-turn challenge. Compares the photo taken before the challenge with the
    photo taken after it. Passes only if (1) the head-turn value changed enough, which a
    flat printed photo or a still image on a screen cannot do, and (2) both photos show the
    same person. Use it whenever a challenge response photo exists."""
    case: AccessCase = _ACTIVE["case"]
    tk = _ACTIVE["toolkit"]
    args = {"before_frame_id": before_frame_id, "after_frame_id": after_frame_id}
    before = case.frame(before_frame_id.strip()) if case else None
    after = case.frame(after_frame_id.strip()) if case else None
    if before is None or after is None:
        known = [f.frame_id for f in case.frames] if case else []
        return _trace("verify_head_turn", args, {"error": f"Unknown frame_id. Valid ids: {known}"})

    b, a = before.evidence, after.evidence
    problems = []
    if a.faces != 1:
        problems.append(f"challenge photo shows {a.faces} faces, expected exactly 1")
    change = None
    if b.head_turn is None or a.head_turn is None:
        problems.append("head position could not be measured in one of the photos")
    else:
        change = abs(a.head_turn - b.head_turn)
        if change < config.HEAD_TURN_MIN_CHANGE:
            problems.append(f"head barely moved (change {change:.2f}, need {config.HEAD_TURN_MIN_CHANGE})")
    same = tk.same_person(before.embedding, after.embedding) if tk else 0.0
    if same < config.FACE_RECOGNITION_THRESHOLD:
        problems.append(f"the two photos may not be the same person (similarity {same:.2f})")
    if a.live_probability < config.FAST_DENY_SPOOF:
        problems.append(f"challenge photo looks fake (real-face probability {a.live_probability:.2f})")

    result = {
        "passed": not problems,
        "head_turn_before": None if b.head_turn is None else round(b.head_turn, 3),
        "head_turn_after": None if a.head_turn is None else round(a.head_turn, 3),
        "head_turn_change": None if change is None else round(change, 3),
        "required_change": config.HEAD_TURN_MIN_CHANGE,
        "same_person_similarity": round(same, 3),
        "identity": a.identity or b.identity,
        "problems": problems,
    }
    for ch in case.challenges:
        if ch.before_frame_id == before.frame_id and ch.after_frame_id == after.frame_id:
            ch.result = result
    return _trace("verify_head_turn", args, result)


@tool("check_access_history")
def check_access_history(name: str) -> str:
    """Look up a registered user by exact name. Returns whether their account is active or
    blocked, and their most recent access attempts (including spoof attempts). Blocked users
    must never be granted access."""
    tk = _ACTIVE["toolkit"]
    logs = _ACTIVE["logs"]
    name = (name or "").strip()
    status = tk.users.get_status(name) if tk else None
    recent = []
    if logs is not None:
        recent = [r for r in logs.get_recent(200) if r.get("user_name") == name][:5]
    result = {
        "name": name,
        "registered": status is not None,
        "status": status,
        "recent_attempts": [{"time": r["timestamp"], "liveness": r["liveness_result"],
                             "access": r["access_result"]} for r in recent],
    }
    return _trace("check_access_history", {"name": name}, result)


class AgentDecision(BaseModel):
    decision: Literal["GRANTED", "DENIED", "CHALLENGE"]
    person: Optional[str] = Field(None, description="Registered user's exact name if identified, else null")
    challenge_kind: Optional[Literal["head_turn", "retake"]] = Field(
        None, description="Only for CHALLENGE: head_turn to test liveness, retake for a better photo")
    challenge_instruction: Optional[str] = Field(
        None, description="Only for CHALLENGE: one short instruction shown on screen, max 15 words")
    reasoning: str = Field(..., description="1-3 plain-English sentences explaining the decision")


# Providers the agent can use, in the order they are checked. xAI (Grok) and Groq both
# speak the OpenAI API format, so they go through CrewAI's OpenAI client with their own URL.
PROVIDERS = [
    # (key env var, label, default model, model override env var, base URL)
    ("GROQ_API_KEY", "Groq", "openai/gpt-oss-120b", "GROQ_MODEL", "https://api.groq.com/openai/v1"),
    ("XAI_API_KEY", "Grok (xAI)", "grok-4.3", "XAI_MODEL", "https://api.x.ai/v1"),
    ("ANTHROPIC_API_KEY", "Claude", "claude-haiku-4-5-20251001", None, None),
    ("OPENAI_API_KEY", "OpenAI", "gpt-4o-mini", None, None),
]


def _llm_settings() -> Optional[dict]:
    """Pick the LLM from whichever API key is set. Returns LLM() kwargs plus a label."""
    for key_env, label, default_model, model_env, base_url in PROVIDERS:
        key = os.getenv(key_env, "").strip()
        if not key:
            continue
        model = (os.getenv(model_env, "").strip() if model_env else "") or config.LLM_MODEL or default_model
        if base_url:  # OpenAI-compatible provider (Groq, Grok)
            out = {"label": f"{label}: {model}", "model": f"openai/{model}", "base_url": base_url, "api_key": key}
            if key_env == "GROQ_API_KEY":
                fb = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b").strip()
                if fb and fb != model:
                    out["fallback_model"] = fb
            return out
        prefix = "anthropic" if key_env == "ANTHROPIC_API_KEY" else "openai"
        return {"label": f"{label}: {model}", "model": f"{prefix}/{model}", "api_key": key}
    return None


def resolve_model() -> Optional[str]:
    """Human-readable name of the model in use, or None if no key is set."""
    settings = _llm_settings()
    return settings["label"] if settings else None


def agent_available() -> bool:
    return _llm_settings() is not None


BACKSTORY = (
    "You guard a door that opens with a face scan. Fast automatic checks already handled the "
    "clear cases; you only see the unclear ones. You are careful, never guess, and always base "
    "decisions on what your tools report. Letting the wrong person in is far worse than asking "
    "a real user for one more photo."
)

POLICY = """Decision rules:
- GRANTED only if the person is identified as a registered, active user AND liveness is proven:
  either a photo of that person scored LIVE, or a head-turn challenge for that person passed.
- CHALLENGE (head_turn) when the face might be real but liveness is unsure, or the match score is
  close to the threshold. Tell them to turn their head to the left or right and take the photo while turned.
- CHALLENGE (retake) when the photo itself is the problem: several faces, face cut off, or too small/dark.
- No more than {max_challenges} challenges per attempt; {challenges_left} remaining now. With none left you
  must choose GRANTED or DENIED.
- DENIED when liveness clearly fails, a head-turn challenge failed, the person is unknown, or the user is blocked."""


def _case_brief(case: AccessCase) -> str:
    lines = [f"Access attempt {case.case_id}. Why it needs you: {case.escalation_reason}", "Photos:"]
    for f in case.frames:
        lines.append(f"- {f.frame_id} ({f.role})")
    if case.challenges:
        lines.append("Challenges issued so far:")
        for i, ch in enumerate(case.challenges, 1):
            after = ch.after_frame_id or "not answered yet"
            lines.append(f"{i}. [{ch.kind}] \"{ch.instruction}\" before={ch.before_frame_id} after={after}")
    return "\n".join(lines)


def _is_rate_limit(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(k in text for k in ("429", "rate limit", "rate_limit", "ratelimit", "too many requests", "tokens per"))


def _kickoff(case: AccessCase, llm_kwargs: dict, challenges_left: int):
    officer = Agent(
        role="Access Security Officer",
        goal="Decide whether to open the door for the person in the photos, or what they must do to prove they are real.",
        backstory=BACKSTORY,
        tools=[inspect_face, verify_head_turn, check_access_history],
        llm=LLM(temperature=0, max_tokens=2000, **llm_kwargs),  # room for the model's reasoning
        allow_delegation=False,
        max_iter=6,
        max_execution_time=config.AGENT_TIMEOUT_SECONDS,
        verbose=False,
    )
    task = Task(
        description=(
            _case_brief(case) + "\n\n"
            "Investigate with your tools: inspect the newest photo (and older ones if useful); if a "
            "challenge response photo exists, run verify_head_turn on it (for head_turn challenges); "
            "if someone is identified, check their access history.\n\n"
            + POLICY.format(max_challenges=config.MAX_CHALLENGES, challenges_left=challenges_left)
        ),
        expected_output="The decision, the person (or null), the challenge details if any, and short reasoning.",
        agent=officer,
        output_pydantic=AgentDecision,
    )
    return Crew(agents=[officer], tasks=[task], process=Process.sequential, verbose=False).kickoff()


def run_agent(case: AccessCase, toolkit, access_logs) -> AgentDecision:
    """Investigate one case with the CrewAI agent and return its decision.

    On Groq's free tier the main model has a small daily token allowance. If it is used up
    (HTTP 429), the agent retries once with the smaller fallback model, which has a much
    bigger allowance. The safety rules still check whatever it decides.
    """
    settings = _llm_settings()
    if settings is None:
        raise RuntimeError("No LLM key set. Add GROQ_API_KEY (free) or XAI_API_KEY (Grok).")
    fallback = settings.pop("fallback_model", None)
    llm_kwargs = {k: v for k, v in settings.items() if k != "label"}
    challenges_left = max(0, config.MAX_CHALLENGES - len(case.challenges))

    with _RUN_LOCK:
        _ACTIVE.update(case=case, toolkit=toolkit, logs=access_logs)
        trace_start = len(case.trace)
        try:
            try:
                result = _kickoff(case, llm_kwargs, challenges_left)
            except Exception as exc:
                if not (fallback and _is_rate_limit(exc)):
                    raise
                del case.trace[trace_start:]  # drop half-finished tool calls from the failed try
                case.trace.append({"tool": "note", "input": {},
                                   "output": {"message": f"Main model hit its rate limit; retried with {fallback}."}})
                result = _kickoff(case, {**llm_kwargs, "model": f"openai/{fallback}"}, challenges_left)
        finally:
            _ACTIVE.update(case=None, toolkit=None, logs=None)

    decision = getattr(result, "pydantic", None)
    if isinstance(decision, AgentDecision):
        return decision
    # Fallback: the model answered in text that still contains JSON.
    raw = getattr(result, "raw", str(result))
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        return AgentDecision.model_validate_json(raw[start:end + 1])
    raise ValueError(f"Agent returned no usable decision: {raw[:200]}")
