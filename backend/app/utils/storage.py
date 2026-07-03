from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import settings
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


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, DatasetSession] = {}
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        settings.processed_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, filename: str, file_path: Path) -> DatasetSession:
        session_id = str(uuid.uuid4())
        session = DatasetSession(session_id=session_id, filename=filename, raw_path=file_path)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> DatasetSession:
        if session_id not in self._sessions:
            raise KeyError(f"Session {session_id} not found")
        return self._sessions[session_id]

    def ensure_raw_df(self, session: DatasetSession) -> pd.DataFrame:
        if session.raw_df is None:
            session.raw_df = load_tabular_file(session.raw_path)
        return session.raw_df

    def get_active_df(self, session: DatasetSession) -> pd.DataFrame:
        if session.cleaned_df is not None:
            return session.cleaned_df
        return self.ensure_raw_df(session)

    def get_df_for_training(self, session: DatasetSession, target_column: str) -> pd.DataFrame:
        """Return a dataframe that still contains the chosen target column."""
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


session_store = SessionStore()
