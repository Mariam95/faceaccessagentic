"""State for one access attempt ("case") and an audit log of every decision step.

A case starts when someone takes a photo at the door. If the agent asks for a
challenge, the next photo is added to the same case, so the agent can compare them.
"""
import json
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

import config
from src.database import _connect
from src.face_detection import FaceBox
from src.agentic.vision import FrameEvidence


@dataclass
class FrameRecord:
    frame_id: str
    role: str                       # "first photo" or "challenge response"
    frame: np.ndarray
    evidence: FrameEvidence
    box: Optional[FaceBox]
    embedding: Optional[np.ndarray]


@dataclass
class Challenge:
    kind: str                       # "head_turn" or "retake"
    instruction: str
    before_frame_id: str
    after_frame_id: Optional[str] = None
    result: Optional[dict] = None   # filled by the verify_head_turn tool


@dataclass
class AccessCase:
    case_id: str
    escalation_reason: str = ""
    frames: List[FrameRecord] = field(default_factory=list)
    challenges: List[Challenge] = field(default_factory=list)
    trace: List[dict] = field(default_factory=list)   # tool calls, shown in the UI
    closed: bool = False

    def frame(self, frame_id: str) -> Optional[FrameRecord]:
        return next((f for f in self.frames if f.frame_id == frame_id), None)

    @property
    def latest(self) -> FrameRecord:
        return self.frames[-1]

    @property
    def pending_challenge(self) -> Optional[Challenge]:
        if self.challenges and self.challenges[-1].after_frame_id is None:
            return self.challenges[-1]
        return None


class CaseRegistry:
    """In-memory store of open cases. Old cases are dropped automatically."""

    def __init__(self, max_cases: int = 200):
        self._cases: "OrderedDict[str, AccessCase]" = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_cases

    def new_case(self) -> AccessCase:
        case = AccessCase(case_id=uuid.uuid4().hex[:8])
        with self._lock:
            self._cases[case.case_id] = case
            while len(self._cases) > self._max:
                self._cases.popitem(last=False)
        return case

    def get(self, case_id: str) -> Optional[AccessCase]:
        with self._lock:
            return self._cases.get(case_id)


def new_frame_id(case: AccessCase) -> str:
    return f"{case.case_id}-f{len(case.frames) + 1}"


class DecisionAudit:
    """One row per decision step: who decided (fast path, agent, safety rule) and why."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or config.ACCESS_LOG_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with _connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                case_id TEXT NOT NULL, decided_by TEXT NOT NULL, decision TEXT NOT NULL,
                person TEXT, reasoning TEXT, tool_calls TEXT)""")

    def record(self, case_id, decided_by, decision, person, reasoning, tool_calls):
        with _connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO agent_decisions(timestamp,case_id,decided_by,decision,person,reasoning,tool_calls)"
                " VALUES(?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), case_id, decided_by, decision, person,
                 reasoning, json.dumps(tool_calls, default=str)))

    def recent(self, limit: int = 50):
        with _connect(self.db_path) as conn:
            rows = conn.execute("SELECT * FROM agent_decisions ORDER BY id DESC LIMIT ?",
                                (max(1, min(int(limit), 200)),)).fetchall()
            return [dict(r) for r in rows]
