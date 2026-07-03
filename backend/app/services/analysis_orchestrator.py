from __future__ import annotations

from typing import Any

import pandas as pd

from app.config import settings
from app.llm.client import get_llm_client
from app.llm.context_builder import SYSTEM_PROMPT, build_llm_context
from app.llm.tools import execute_tools, format_tool_results
from app.services.anomaly_service import detect_anomalies
from app.services.cleaning_service import clean_dataframe
from app.services.clustering_service import run_clustering
from app.services.dashboard_service import build_dashboard
from app.services.eda_service import generate_custom_chart, generate_eda
from app.services.ml_service import train_models
from app.services.mlflow_service import log_training_run
from app.services.model_store import save_model_pipeline
from app.services.profiling_service import profile_dataframe
from app.services.rag_service import build_artifact_chunks, format_rag_context, retrieve_relevant_chunks
from app.services.validation_service import validate_dataframe
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
        session.validation_report = validate_dataframe(df, profile["column_types"])
        profile["validation_report"] = session.validation_report
        self._refresh_rag_index(session)
        session_store.persist(session)
        return profile

    def clean_session(
        self,
        session: DatasetSession,
        outlier_strategy: str | None = None,
    ) -> dict[str, Any]:
        df = session_store.ensure_raw_df(session)
        strategy = outlier_strategy or settings.default_outlier_strategy
        cleaned, report = clean_dataframe(
            df,
            target_column=session.target_column,
            outlier_strategy=strategy,  # type: ignore[arg-type]
        )
        session.cleaned_df = cleaned
        session.cleaning_report = report
        self._refresh_rag_index(session)
        session_store.persist(session)
        return report

    def eda_session(self, session: DatasetSession) -> dict[str, Any]:
        df = session_store.get_active_df(session)
        column_types = session.column_types or session.profile.get("column_types", {}) if session.profile else {}
        eda = to_json_safe(generate_eda(df, column_types))
        session.eda = eda
        self._refresh_rag_index(session)
        session_store.persist(session)
        return eda

    def custom_eda_session(
        self,
        session: DatasetSession,
        x_column: str,
        y_column: str | None = None,
        chart_type: str = "scatter",
    ) -> dict[str, Any]:
        df = session_store.get_active_df(session)
        chart = to_json_safe(generate_custom_chart(df, x_column, y_column, chart_type))
        if session.eda is None:
            self.eda_session(session)
        custom = session.eda.setdefault("custom_charts", [])
        custom.append(chart)
        session_store.persist(session)
        return chart

    def clustering_session(
        self,
        session: DatasetSession,
        n_clusters: int | None = None,
    ) -> dict[str, Any]:
        df = session_store.get_active_df(session)
        result = to_json_safe(run_clustering(df, n_clusters=n_clusters))
        session.clustering = result
        self._refresh_rag_index(session)
        session_store.persist(session)
        return result

    def anomaly_session(
        self,
        session: DatasetSession,
        contamination: float = 0.05,
    ) -> dict[str, Any]:
        df = session_store.get_active_df(session)
        result = to_json_safe(detect_anomalies(df, contamination=contamination))
        session.anomaly = result
        self._refresh_rag_index(session)
        session_store.persist(session)
        return result

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

        pipeline = ml_results.pop("_best_pipeline", None)
        model_path = None
        if pipeline is not None:
            model_path = save_model_pipeline(session.session_id, pipeline)
            session.model_path = model_path
            ml_results["model_saved"] = True
            ml_results["model_path"] = str(model_path)

        run_id = log_training_run(
            session.session_id,
            session.target_column,
            ml_results,
            model_path=str(model_path) if model_path else None,
        )
        if run_id:
            ml_results["mlflow_run_id"] = run_id

        session.ml_results = to_json_safe(ml_results)
        session.task_type = ml_results["task_type"]
        self._refresh_rag_index(session)
        session_store.persist(session)
        return ml_results

    async def ask_session(self, session: DatasetSession, question: str) -> dict[str, Any]:
        context = self._build_enriched_context(session, question)
        client = get_llm_client()
        llm_result = await client.chat(SYSTEM_PROMPT, context)
        session_store.append_chat(session, "user", question)
        session_store.append_chat(session, "assistant", llm_result["answer"])
        return {
            "question": question,
            "answer": llm_result["answer"],
            "model_used": llm_result.get("model_used", "unknown"),
            "context_preview": context[:1500],
            "tools_used": [t["tool"] for t in execute_tools(session, question)],
        }

    def build_ask_context(self, session: DatasetSession, question: str) -> str:
        return self._build_enriched_context(session, question)

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
        session_store.persist(session)
        return dashboard

    def _build_enriched_context(self, session: DatasetSession, question: str) -> str:
        df = session_store.get_active_df(session)
        explainability = (session.ml_results or {}).get("explainability")
        tool_results = execute_tools(session, question)
        rag_chunks = retrieve_relevant_chunks(session.rag_chunks or [], question)
        base = build_llm_context(
            df=df,
            profile=session.profile,
            cleaning_report=session.cleaning_report,
            eda=session.eda,
            ml_results=session.ml_results,
            explainability=explainability,
            question=question,
            chat_history=session.chat_history,
        )
        extras = [format_tool_results(tool_results), format_rag_context(rag_chunks)]
        return "\n\n".join(part for part in [base, *extras] if part)

    def _refresh_rag_index(self, session: DatasetSession) -> None:
        session.rag_chunks = build_artifact_chunks(
            {
                "profile": session.profile,
                "validation_report": session.validation_report,
                "cleaning_report": session.cleaning_report,
                "eda": session.eda,
                "ml_results": session.ml_results,
                "clustering": session.clustering,
                "anomaly": session.anomaly,
            }
        )


analysis_orchestrator = AnalysisOrchestrator()
