import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import config

class DatabaseError(Exception): pass

@contextmanager
def _connect(db_path: Path):
    conn = None
    try:
        conn = sqlite3.connect(str(db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    except sqlite3.Error as e:
        if conn: conn.rollback()
        raise DatabaseError(f"Database operation failed: {e}") from e
    finally:
        if conn: conn.close()

class UserDatabase:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or config.DATABASE_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with _connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active')""")

    def add_user(self, name: str, embeddings) -> int:
        if not name or not name.strip(): raise ValueError("Name is required.")
        if embeddings is None or len(embeddings) == 0: raise ValueError("At least one embedding is required.")
        payload = [np.asarray(e, dtype=np.float32).tolist() for e in embeddings]
        with _connect(self.db_path) as conn:
            existing = conn.execute("SELECT id FROM users WHERE name=? ORDER BY id LIMIT 1", (name.strip(),)).fetchone()
            if existing:
                conn.execute("UPDATE users SET embedding=?, created_at=? WHERE id=?", (json.dumps(payload), datetime.now(timezone.utc).isoformat(), int(existing['id'])))
                return int(existing['id'])
            cur = conn.execute("INSERT INTO users(name,embedding,created_at,status) VALUES(?,?,?,'active')",
                (name.strip(), json.dumps(payload), datetime.now(timezone.utc).isoformat()))
            return int(cur.lastrowid)

    def get_all_active_embeddings(self) -> Dict[str, list]:
        result = {}
        with _connect(self.db_path) as conn:
            rows = conn.execute("SELECT name, embedding FROM users WHERE status='active' ORDER BY id").fetchall()
            for row in rows:
                raw = json.loads(row['embedding'])
                # Backward compatible with the old single-vector format.
                if raw and isinstance(raw[0], (int, float)): raw = [raw]
                valid = [np.array(v, dtype=np.float32) for v in raw if isinstance(v, list) and len(v) == 128]
                if valid: result[row['name']] = valid
        return result

    def name_exists(self, name: str) -> bool:
        with _connect(self.db_path) as conn:
            return conn.execute("SELECT 1 FROM users WHERE name=? LIMIT 1", (name.strip(),)).fetchone() is not None

    def get_status(self, name: str) -> Optional[str]:
        with _connect(self.db_path) as conn:
            row = conn.execute("SELECT status FROM users WHERE name=? ORDER BY id LIMIT 1", (name.strip(),)).fetchone()
            return row["status"] if row else None

    def set_status(self, user_id: int, status: str) -> None:
        """Block ('blocked') or restore ('active') a user. Blocked users are never matched."""
        if status not in ("active", "blocked"):
            raise ValueError("status must be 'active' or 'blocked'")
        with _connect(self.db_path) as conn:
            conn.execute("UPDATE users SET status=? WHERE id=?", (status, int(user_id)))

    def list_users(self) -> List[dict]:
        with _connect(self.db_path) as conn:
            return [dict(r) for r in conn.execute("SELECT id,name,created_at,status FROM users ORDER BY id DESC").fetchall()]

class AccessLogDatabase:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or config.ACCESS_LOG_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with _connect(self.db_path) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, user_name TEXT,
                liveness_result TEXT NOT NULL, liveness_confidence REAL,
                recognition_result TEXT NOT NULL, recognition_confidence REAL,
                access_result TEXT NOT NULL)""")
    def log_access(self, user_name, liveness_result, liveness_confidence, recognition_result, recognition_confidence, access_result):
        with _connect(self.db_path) as conn:
            cur = conn.execute("INSERT INTO access_log(timestamp,user_name,liveness_result,liveness_confidence,recognition_result,recognition_confidence,access_result) VALUES(?,?,?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), user_name, liveness_result, float(liveness_confidence), recognition_result, float(recognition_confidence), access_result))
            return int(cur.lastrowid)
    def get_recent(self, limit=50):
        with _connect(self.db_path) as conn:
            limit=max(1,min(int(limit),200)); return [dict(r) for r in conn.execute("SELECT * FROM access_log ORDER BY id DESC LIMIT ?",(limit,)).fetchall()]
