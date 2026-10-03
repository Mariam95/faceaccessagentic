"""The agentic access pipeline.

    photo -> vision models -> clear case?  yes -> fast decision (no LLM, instant)
                                           no  -> CrewAI agent investigates with tools
                                                    -> GRANTED / DENIED / CHALLENGE
                                                    -> safety rules check the decision
                                                    -> log

Modes:
    "unclear"  agent handles only unclear cases (recommended)
    "always"   agent handles every photo (best for showing the agent in a demo)
    "off"      the original fixed checklist from src/access_control.py
"""
from dataclasses import dataclass, field
from typing import List, Optional

import config
from src.access_control import decide_access
from src.database import AccessLogDatabase, DatabaseError
from src.agentic.case import AccessCase, CaseRegistry, Challenge, DecisionAudit, FrameRecord, new_frame_id
from src.agentic.vision import VisionToolkit, annotate
from src.agentic import crew as agent_crew

DEFAULT_TURN = "Turn your head to the left, then take the photo while turned."
DEFAULT_RETAKE = "Make sure only you are in view, face centred, then take the photo."


@dataclass
class Outcome:
    status: str                      # GRANTED, DENIED, CHALLENGE, RETRY
    message: str
    decided_by: str                  # fast path, agent, safety rule, fixed rules, -
    case_id: Optional[str] = None
    person: Optional[str] = None
    instruction: Optional[str] = None
    reasoning: str = ""
    evidence: dict = field(default_factory=dict)
    trace: List[dict] = field(default_factory=list)
    image: Optional[object] = None   # annotated RGB array
    notes: List[str] = field(default_factory=list)


class AccessPipeline:
    def __init__(self):
        self.vision = VisionToolkit()
        self.logs = AccessLogDatabase()
        self.audit = DecisionAudit()
        self.cases = CaseRegistry()

    # ------------------------------------------------------------------ entry points
    def start(self, frame, mode: str = "unclear") -> Outcome:
        """A new person takes a photo at the door."""
        case = self.cases.new_case()
        rec = self._add_frame(case, frame, "first photo")
        ev = rec.evidence

        if ev.error:
            return self._retry(case, rec, f"Could not process the photo ({ev.error}).")
        if ev.faces == 0:
            return self._retry(case, rec, "No face found. Look at the camera in good light and try again.")

        if mode == "off" or not agent_crew.agent_available():
            out = self._fixed_rules(case, rec)
            if mode != "off":
                out.notes.append("Agent is off because no LLM key is set, so the fixed checklist decided.")
            return out

        if mode == "unclear":
            fast = self._fast_path(case, rec)
            if fast is not None:
                return fast
        else:
            case.escalation_reason = "Agent mode is set to review every attempt."
        return self._ask_agent(case)

    def answer_challenge(self, case_id: str, frame) -> Outcome:
        """The person did what the agent asked and took another photo."""
        case = self.cases.get(case_id)
        if case is None or case.closed or case.pending_challenge is None:
            return Outcome("RETRY", "That attempt has expired. Please start again.", "-")
        challenge = case.pending_challenge
        rec = self._add_frame(case, frame, f"challenge response ({challenge.kind})")
        challenge.after_frame_id = rec.frame_id
        if not agent_crew.agent_available():
            return self._fixed_rules(case, rec)
        return self._ask_agent(case)

    # ------------------------------------------------------------------ deciders
    def _fast_path(self, case: AccessCase, rec: FrameRecord) -> Optional[Outcome]:
        ev = rec.evidence
        if ev.faces != 1 or not ev.fully_in_frame:
            case.escalation_reason = "Photo has several faces or the face is cut off."
            return None
        if (ev.liveness_label == "LIVE" and ev.live_probability >= config.FAST_GRANT_LIVE
                and ev.identity and ev.match_score >= config.FAST_GRANT_MATCH):
            return self._close(case, rec, "GRANTED", ev.identity, "fast path",
                               f"Clearly real face ({ev.live_probability:.0%}) and strong match to {ev.identity}.")
        if ev.live_probability < config.FAST_DENY_SPOOF:
            return self._close(case, rec, "DENIED", None, "fast path",
                               f"Obvious fake: real-face probability only {ev.live_probability:.0%}.")
        if ev.liveness_label == "LIVE" and not ev.identity and ev.match_score < config.FAST_DENY_UNKNOWN:
            return self._close(case, rec, "DENIED", None, "fast path",
                               "Real face, but it does not match any active registered user.")
        parts = []
        if ev.liveness_label != "LIVE" or ev.live_probability < config.FAST_GRANT_LIVE:
            parts.append(f"liveness is unsure ({ev.live_probability:.0%} real)")
        if not ev.identity or ev.match_score < config.FAST_GRANT_MATCH:
            parts.append(f"identity match is borderline (score {ev.match_score:.2f})")
        case.escalation_reason = "; ".join(parts) or "unclear evidence"
        return None

    def _fixed_rules(self, case: AccessCase, rec: FrameRecord) -> Outcome:
        ev = rec.evidence
        d = decide_access(ev.liveness_label, ev.live_probability, ev.identity, ev.match_score)
        return self._close(case, rec, d.access_result, d.user_name, "fixed rules", d.reason)

    def _ask_agent(self, case: AccessCase) -> Outcome:
        rec = case.latest
        try:
            decision = agent_crew.run_agent(case, self.vision, self.logs)
        except Exception as exc:  # LLM down, bad key, timeout: fail safely to the fixed rules
            out = self._fixed_rules(case, rec)
            out.notes.append(f"Agent unavailable ({type(exc).__name__}: {str(exc)[:160]}). Fixed checklist decided.")
            return out

        status, person, why_override = self._enforce_policy(case, decision)
        reasoning = decision.reasoning
        decided_by = "agent"
        if why_override:
            decided_by = "safety rule"
            reasoning = f"Agent said {decision.decision}, but {why_override} Agent's reasoning: {decision.reasoning}"

        if status == "CHALLENGE":
            kind = "head_turn" if why_override else (decision.challenge_kind or "head_turn")
            instruction = (None if why_override else decision.challenge_instruction) or \
                (DEFAULT_TURN if kind == "head_turn" else DEFAULT_RETAKE)
            case.challenges.append(Challenge(kind=kind, instruction=instruction, before_frame_id=rec.frame_id))
            self._audit(case, decided_by, "CHALLENGE", person, f"{reasoning} Asked: {instruction}")
            return Outcome("CHALLENGE", instruction, decided_by, case.case_id, person, instruction, reasoning,
                           rec.evidence.to_dict(), list(case.trace),
                           annotate(rec.frame, rec.box, "Checking...", None))
        return self._close(case, rec, status, person, decided_by, reasoning)

    def _enforce_policy(self, case: AccessCase, d) -> tuple:
        """Hard rules the LLM cannot override. Returns (status, person, override_reason or None)."""
        challenges_left = config.MAX_CHALLENGES - len(case.challenges)

        def fallback(reason):
            return ("CHALLENGE", None, reason) if challenges_left > 0 else ("DENIED", None, reason)

        if d.decision == "CHALLENGE":
            if challenges_left <= 0:
                return "DENIED", None, "no challenges are left for this attempt."
            return "CHALLENGE", None, None
        if d.decision == "DENIED":
            return "DENIED", None, None

        # GRANTED: check every condition against the models' own evidence, not the LLM's words.
        name = case.latest.evidence.identity
        if not name:
            return fallback("the latest photo is not matched to any registered user.")
        if d.person and d.person.strip() != name:
            return fallback(f"the agent named '{d.person}' but the face matches '{name}'.")
        if self.vision.users.get_status(name) != "active":
            return "DENIED", None, f"{name}'s account is blocked."
        live_proven = any(f.evidence.liveness_label == "LIVE" and f.evidence.identity == name for f in case.frames)
        turn_proven = any(c.result and c.result.get("passed") and c.result.get("identity") == name
                          for c in case.challenges)
        if not (live_proven or turn_proven):
            return fallback("liveness was never proven (no LIVE photo and no passed head-turn challenge).")
        return "GRANTED", name, None

    # ------------------------------------------------------------------ helpers
    def _add_frame(self, case: AccessCase, frame, role: str) -> FrameRecord:
        ev, box, emb = self.vision.analyze(frame)
        rec = FrameRecord(new_frame_id(case), role, frame, ev, box, emb)
        case.frames.append(rec)
        return rec

    def _retry(self, case: AccessCase, rec: FrameRecord, msg: str) -> Outcome:
        case.closed = True
        return Outcome("RETRY", msg, "-", case.case_id, evidence=rec.evidence.to_dict(),
                       image=annotate(rec.frame, rec.box, "", None))

    def _close(self, case, rec, status, person, decided_by, reasoning) -> Outcome:
        case.closed = True
        ev = rec.evidence
        try:
            self.logs.log_access(
                person, ev.liveness_label or "UNKNOWN", ev.live_probability,
                "MATCH" if person else ("UNKNOWN" if ev.liveness_label == "LIVE" else "N/A"),
                ev.match_score, status)
        except DatabaseError:
            pass
        self._audit(case, decided_by, status, person, reasoning)
        msg = f"Welcome, {person}." if status == "GRANTED" else "Access denied."
        label = person if status == "GRANTED" else "DENIED"
        return Outcome(status, msg, decided_by, case.case_id, person, None, reasoning, ev.to_dict(),
                       list(case.trace), annotate(rec.frame, rec.box, label, status == "GRANTED"))

    def _audit(self, case, decided_by, decision, person, reasoning):
        try:
            self.audit.record(case.case_id, decided_by, decision, person, reasoning, case.trace)
        except DatabaseError:
            pass
