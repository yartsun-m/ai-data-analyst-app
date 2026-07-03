from __future__ import annotations

from typing import Any

import pandas as pd

from app.llm.client import get_llm_client
from app.llm.context_builder import SYSTEM_PROMPT, build_llm_context
from app.services.cleaning_service import clean_dataframe
from app.services.dashboard_service import build_dashboard
from app.services.eda_service import generate_eda
from app.services.ml_service import train_models
from app.services.profiling_service import profile_dataframe
from app.utils.json_utils import to_json_safe
from app.utils.storage import DatasetSession, session_store


class AnalysisOrchestrator:
    def profile_session(self, session: DatasetSession, target_column: str | None = None) -> dict[str, Any]:
        df = session_store.ensure_raw_df(session)
        session.target_column = target_column or session.target_column
        profile = profile_dataframe(df, target_column=session.target_column)
        session.profile = profile
        session.column_types = profile["column_types"]
        session.task_type = profile.get("task_type")
        return profile

    def clean_session(self, session: DatasetSession) -> dict[str, Any]:
        df = session_store.ensure_raw_df(session)
        cleaned, report = clean_dataframe(df, target_column=session.target_column)
        session.cleaned_df = cleaned
        session.cleaning_report = report
        return report

    def eda_session(self, session: DatasetSession) -> dict[str, Any]:
        df = session_store.get_active_df(session)
        column_types = session.column_types or session.profile.get("column_types", {}) if session.profile else {}
        eda = to_json_safe(generate_eda(df, column_types))
        session.eda = eda
        return eda

    def train_session(self, session: DatasetSession, target_column: str | None = None) -> dict[str, Any]:
        if target_column:
            session.target_column = target_column
        if not session.target_column:
            raise ValueError("Target column is required for training")

        try:
            df = session_store.get_df_for_training(session, session.target_column)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
        if session.profile is None:
            self.profile_session(session, target_column=session.target_column)

        task_type = session.task_type or (session.profile or {}).get("task_type")
        ml_results = train_models(df, session.target_column, task_type=task_type)
        session.ml_results = to_json_safe(ml_results)
        session.ml_results = ml_results
        session.task_type = ml_results["task_type"]
        return ml_results

    async def ask_session(self, session: DatasetSession, question: str) -> dict[str, Any]:
        df = session_store.get_active_df(session)
        explainability = (session.ml_results or {}).get("explainability")
        context = build_llm_context(
            df=df,
            profile=session.profile,
            cleaning_report=session.cleaning_report,
            eda=session.eda,
            ml_results=session.ml_results,
            explainability=explainability,
            question=question,
        )
        client = get_llm_client()
        llm_result = await client.chat(SYSTEM_PROMPT, context)
        return {
            "question": question,
            "answer": llm_result["answer"],
            "model_used": llm_result.get("model_used", "unknown"),
            "context_preview": context[:1500],
        }

    def dashboard_session(self, session: DatasetSession) -> dict[str, Any]:
        if session.eda is None:
            self.eda_session(session)
        df = session_store.get_active_df(session)
        dashboard = to_json_safe(
            build_dashboard(
                df=df,
                profile=session.profile,
                cleaning_report=session.cleaning_report,
                eda=session.eda,
                ml_results=session.ml_results,
            )
        )
        session.dashboard = dashboard
        return dashboard


analysis_orchestrator = AnalysisOrchestrator()
