from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None  # type: ignore
    XGBRegressor = None  # type: ignore

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def build_explainability(
    pipeline: Pipeline | None,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    feature_importance: list[dict[str, Any]] | None = None,
    sample_size: int = 150,
) -> dict[str, Any]:
    """Explain predictions using the same fitted pipeline that was trained."""
    result: dict[str, Any] = {
        "method": "feature_importance",
        "top_features": feature_importance or [],
        "shap_available": HAS_SHAP,
    }

    if pipeline is None or X.empty:
        return result

    try:
        explained = _explain_pipeline(pipeline, X, y, task_type, sample_size)
        if explained:
            result["method"] = explained["method"]
            result["top_features"] = explained["top_features"]
    except Exception as exc:
        result["explain_error"] = str(exc)
        if feature_importance:
            result["top_features"] = _normalize_feature_list(feature_importance)

    return result


def _explain_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    sample_size: int,
) -> dict[str, Any] | None:
    preprocessor = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")
    if preprocessor is None or model is None:
        return None

    sample_n = min(sample_size, len(X))
    X_sample = X.sample(n=sample_n, random_state=42)
    y_sample = y.loc[X_sample.index]

    X_transformed = preprocessor.transform(X_sample)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()

    feature_names = _get_transformed_feature_names(preprocessor)
    if len(feature_names) != X_transformed.shape[1]:
        feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    if HAS_SHAP and _supports_tree_shap(model):
        top = _shap_tree_importance(model, X_transformed, feature_names)
        if top:
            return {"method": "shap", "top_features": top}

    coef_top = _coefficient_importance(model, feature_names)
    if coef_top:
        return {"method": "coefficients", "top_features": coef_top}

    perm_top = _permutation_importance(pipeline, X_sample, y_sample, task_type)
    if perm_top:
        return {"method": "permutation", "top_features": perm_top}

    if feature_importance := _model_native_importance(model, feature_names):
        return {"method": "feature_importance", "top_features": feature_importance}

    return None


def _supports_tree_shap(model: Any) -> bool:
    tree_types = (RandomForestClassifier, RandomForestRegressor)
    if XGBClassifier is not None:
        tree_types = tree_types + (XGBClassifier, XGBRegressor)  # type: ignore
    return isinstance(model, tree_types)


def _shap_tree_importance(
    model: Any,
    X_transformed: np.ndarray,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        # Multiclass: average absolute SHAP across classes and samples.
        values = np.abs(np.array(shap_values)).mean(axis=(0, 1))
    else:
        values = np.abs(shap_values).mean(axis=0)

    return _scores_to_feature_list(feature_names, np.asarray(values))


def _coefficient_importance(model: Any, feature_names: list[str]) -> list[dict[str, Any]] | None:
    if not hasattr(model, "coef_"):
        return None
    coef = model.coef_
    values = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    return _scores_to_feature_list(feature_names, values)


def _model_native_importance(model: Any, feature_names: list[str]) -> list[dict[str, Any]] | None:
    if not hasattr(model, "feature_importances_"):
        return None
    return _scores_to_feature_list(feature_names, model.feature_importances_)


def _permutation_importance(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
    task_type: str,
) -> list[dict[str, Any]] | None:
    scoring = "r2" if task_type == "regression" else "accuracy"
    try:
        result = permutation_importance(
            pipeline,
            X_sample,
            y_sample,
            n_repeats=3,
            random_state=42,
            scoring=scoring,
            n_jobs=1,
        )
    except Exception:
        return None
    return _scores_to_feature_list(X_sample.columns.tolist(), result.importances_mean)


def _scores_to_feature_list(names: list[str], scores: np.ndarray) -> list[dict[str, Any]]:
    scores = np.asarray(scores, dtype=np.float64)
    scores = np.nan_to_num(scores, nan=0.0)
    total = scores.sum()
    if total > 0:
        scores = scores / total

    pairs = sorted(zip(names, scores), key=lambda x: x[1], reverse=True)
    return [
        {"feature": _clean_feature_name(str(name)), "importance": round(float(val), 4)}
        for name, val in pairs[:15]
        if val > 0
    ]


def _clean_feature_name(name: str) -> str:
    for prefix in ("num__", "cat__", "freq__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    if name.endswith("_freq"):
        name = name[: -len("_freq")]
    return name.replace("_", " ")


def _normalize_feature_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = np.array([float(i.get("importance", 0) or 0) for i in items], dtype=np.float64)
    total = scores.sum()
    if total > 0:
        scores = scores / total
    return [
        {"feature": item["feature"], "importance": round(float(score), 4)}
        for item, score in zip(items, scores)
    ]


def _get_transformed_feature_names(preprocessor: Any) -> list[str]:
    if hasattr(preprocessor, "get_feature_names_out"):
        try:
            return preprocessor.get_feature_names_out().tolist()
        except Exception:
            pass
    return []
