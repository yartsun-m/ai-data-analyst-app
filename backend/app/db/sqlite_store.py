from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.config import settings


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteDatabase:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.sqlite_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    raw_path TEXT NOT NULL,
                    target_column TEXT,
                    task_type TEXT,
                    model_path TEXT,
                    column_types TEXT,
                    profile TEXT,
                    cleaning_report TEXT,
                    eda TEXT,
                    ml_results TEXT,
                    dashboard TEXT,
                    chat_history TEXT,
                    validation_report TEXT,
                    clustering TEXT,
                    anomaly TEXT,
                    rag_chunks TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    progress REAL DEFAULT 0,
                    result TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._ensure_column(conn, "sessions", "clustering", "TEXT")
            self._ensure_column(conn, "sessions", "anomaly", "TEXT")
            self._ensure_column(conn, "sessions", "rag_chunks", "TEXT")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")

    def upsert_session(self, row: dict[str, Any]) -> None:
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, filename, raw_path, target_column, task_type, model_path,
                    column_types, profile, cleaning_report, eda, ml_results, dashboard,
                    chat_history, validation_report, clustering, anomaly, rag_chunks,
                    created_at, updated_at
                ) VALUES (
                    :session_id, :filename, :raw_path, :target_column, :task_type, :model_path,
                    :column_types, :profile, :cleaning_report, :eda, :ml_results, :dashboard,
                    :chat_history, :validation_report, :clustering, :anomaly, :rag_chunks,
                    :created_at, :updated_at
                )
                ON CONFLICT(session_id) DO UPDATE SET
                    filename=excluded.filename,
                    raw_path=excluded.raw_path,
                    target_column=excluded.target_column,
                    task_type=excluded.task_type,
                    model_path=excluded.model_path,
                    column_types=excluded.column_types,
                    profile=excluded.profile,
                    cleaning_report=excluded.cleaning_report,
                    eda=excluded.eda,
                    ml_results=excluded.ml_results,
                    dashboard=excluded.dashboard,
                    chat_history=excluded.chat_history,
                    validation_report=excluded.validation_report,
                    clustering=excluded.clustering,
                    anomaly=excluded.anomaly,
                    rag_chunks=excluded.rag_chunks,
                    updated_at=excluded.updated_at
                """,
                {
                    "session_id": row["session_id"],
                    "filename": row["filename"],
                    "raw_path": row["raw_path"],
                    "target_column": row.get("target_column"),
                    "task_type": row.get("task_type"),
                    "model_path": row.get("model_path"),
                    "column_types": _json_or_none(row.get("column_types")),
                    "profile": _json_or_none(row.get("profile")),
                    "cleaning_report": _json_or_none(row.get("cleaning_report")),
                    "eda": _json_or_none(row.get("eda")),
                    "ml_results": _json_or_none(row.get("ml_results")),
                    "dashboard": _json_or_none(row.get("dashboard")),
                    "chat_history": _json_or_none(row.get("chat_history")),
                    "validation_report": _json_or_none(row.get("validation_report")),
                    "clustering": _json_or_none(row.get("clustering")),
                    "anomaly": _json_or_none(row.get("anomaly")),
                    "rag_chunks": _json_or_none(row.get("rag_chunks")),
                    "created_at": row.get("created_at") or now,
                    "updated_at": now,
                },
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return _row_to_session_dict(row)

    def create_job(self, job_id: str, session_id: str, job_type: str) -> None:
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (job_id, session_id, job_type, status, progress, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', 0, ?, ?)
                """,
                (job_id, session_id, job_type, now, now),
            )

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: float | None = None,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        fields: list[str] = ["updated_at = ?"]
        values: list[Any] = [_utc_now()]
        if status is not None:
            fields.append("status = ?")
            values.append(status)
        if progress is not None:
            fields.append("progress = ?")
            values.append(progress)
        if result is not None:
            fields.append("result = ?")
            values.append(json.dumps(result, default=str))
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        values.append(job_id)
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE job_id = ?", values)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        if data.get("result"):
            data["result"] = json.loads(data["result"])
        return data


def _json_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, default=str)


def _row_to_session_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in (
        "column_types",
        "profile",
        "cleaning_report",
        "eda",
        "ml_results",
        "dashboard",
        "chat_history",
        "validation_report",
        "clustering",
        "anomaly",
        "rag_chunks",
    ):
        if data.get(key):
            data[key] = json.loads(data[key])
    return data


db = SQLiteDatabase()
