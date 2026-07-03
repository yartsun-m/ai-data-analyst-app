from __future__ import annotations

from typing import Any

from app.config import settings


def log_training_run(
    session_id: str,
    target_column: str,
    ml_results: dict[str, Any],
    model_path: str | None = None,
) -> str | None:
    """Log ML training to local MLflow; returns run_id or None if disabled."""
    if not settings.mlflow_enabled:
        return None
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name=f"{session_id[:8]}-{target_column}") as run:
            mlflow.log_param("session_id", session_id)
            mlflow.log_param("target_column", target_column)
            mlflow.log_param("task_type", ml_results.get("task_type"))
            mlflow.log_param("best_model", ml_results.get("best_model"))

            metrics = ml_results.get("best_metrics") or {}
            for key, val in metrics.items():
                if isinstance(val, (int, float)) and key != "primary_score":
                    mlflow.log_metric(key, float(val))

            cv = ml_results.get("cross_validation") or {}
            if cv.get("mean") is not None:
                mlflow.log_metric("cv_mean", float(cv["mean"]))
            if cv.get("std") is not None:
                mlflow.log_metric("cv_std", float(cv["std"]))

            if model_path:
                mlflow.log_artifact(model_path)

            return run.info.run_id
    except Exception:
        return None
