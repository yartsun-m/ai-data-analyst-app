from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import settings
from app.db.sqlite_store import db
from app.utils.data_loader import load_tabular_file


@dataclass
class DatasetSession:
    session_id: str
    filename: str
    raw_path: Path
    raw_df: pd.DataFrame | None = None
    cleaned_df: pd.DataFrame | None = None
    column_types: dict[str, str] = field(default_factory=dict)
    target_column: str | None = None
    task_type: str | None = None
    profile: dict[str, Any] | None = None
    cleaning_report: dict[str, Any] | None = None
    eda: dict[str, Any] | None = None
    ml_results: dict[str, Any] | None = None
    dashboard: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None
    chat_history: list[dict[str, str]] = field(default_factory=list)
    model_path: Path | None = None


class SessionStore:
    def __init__(self) -> None:
        self._cache: dict[str, DatasetSession] = {}
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        settings.processed_dir.mkdir(parents=True, exist_ok=True)
        settings.models_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, filename: str, file_path: Path) -> DatasetSession:
        session_id = str(uuid.uuid4())
        session = DatasetSession(session_id=session_id, filename=filename, raw_path=file_path)
        self._cache[session_id] = session
        self.persist(session)
        return session

    def get(self, session_id: str) -> DatasetSession:
        if session_id in self._cache:
            return self._cache[session_id]
        row = db.get_session(session_id)
        if row is None:
            raise KeyError(f"Session {session_id} not found")
        session = DatasetSession(
            session_id=row["session_id"],
            filename=row["filename"],
            raw_path=Path(row["raw_path"]),
            target_column=row.get("target_column"),
            task_type=row.get("task_type"),
            column_types=row.get("column_types") or {},
            profile=row.get("profile"),
            cleaning_report=row.get("cleaning_report"),
            eda=row.get("eda"),
            ml_results=row.get("ml_results"),
            dashboard=row.get("dashboard"),
            validation_report=row.get("validation_report"),
            chat_history=row.get("chat_history") or [],
            model_path=Path(row["model_path"]) if row.get("model_path") else None,
        )
        self._cache[session_id] = session
        return session

    def persist(self, session: DatasetSession) -> None:
        db.upsert_session(
            {
                "session_id": session.session_id,
                "filename": session.filename,
                "raw_path": str(session.raw_path),
                "target_column": session.target_column,
                "task_type": session.task_type,
                "model_path": str(session.model_path) if session.model_path else None,
                "column_types": session.column_types or None,
                "profile": session.profile,
                "cleaning_report": session.cleaning_report,
                "eda": session.eda,
                "ml_results": session.ml_results,
                "dashboard": session.dashboard,
                "validation_report": session.validation_report,
                "chat_history": session.chat_history,
            }
        )

    def ensure_raw_df(self, session: DatasetSession) -> pd.DataFrame:
        if session.raw_df is None:
            session.raw_df = load_tabular_file(session.raw_path)
        return session.raw_df

    def get_active_df(self, session: DatasetSession) -> pd.DataFrame:
        if session.cleaned_df is not None:
            return session.cleaned_df
        return self.ensure_raw_df(session)

    def get_df_for_training(self, session: DatasetSession, target_column: str) -> pd.DataFrame:
        if session.cleaned_df is not None and target_column in session.cleaned_df.columns:
            return session.cleaned_df
        raw = self.ensure_raw_df(session)
        if target_column not in raw.columns:
            raise KeyError(f"Target column '{target_column}' not found")
        return raw

    def save_json(self, session_id: str, name: str, payload: dict[str, Any]) -> Path:
        path = settings.processed_dir / f"{session_id}_{name}.json"
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")
        return path

    def append_chat(self, session: DatasetSession, role: str, content: str) -> None:
        session.chat_history.append({"role": role, "content": content})
        limit = settings.chat_history_limit
        if len(session.chat_history) > limit * 2:
            session.chat_history = session.chat_history[-limit * 2 :]
        self.persist(session)


session_store = SessionStore()
