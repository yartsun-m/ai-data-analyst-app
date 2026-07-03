from __future__ import annotations

import platform
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from app.config import settings
from app.ml.encoders import FrequencyEncoder, MeanEncoder
from app.ml.explainability import build_explainability
from app.services.profiling_service import detect_task_type
from app.utils.column_roles import (
    detect_column_roles,
    get_ml_excluded_columns,
    onehot_max_categories,
    validate_classification_target,
)
from app.utils.ml_features import sanitize_features_for_ml


def train_models(
    df: pd.DataFrame,
    target_column: str,
    task_type: str | None = None,
    test_size: float = 0.2,
    random_state: int | None = None,
) -> dict[str, Any]:
    random_state = random_state if random_state is not None else settings.random_state

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    column_roles = detect_column_roles(df)
    excluded = get_ml_excluded_columns(df, column_roles, target_column=target_column)

    work = df.dropna(subset=[target_column]).copy()
    y_raw = work[target_column]
    feature_cols = [col for col in work.columns if col != target_column and col not in excluded]
    if not feature_cols:
        feature_cols = [col for col in work.columns if col != target_column]
    if not feature_cols:
        raise ValueError(
            "This dataset has only one column, so there are no features to train on. "
            "Upload a dataset with at least two columns."
        )

    X = work[feature_cols]
    inferred = task_type or detect_task_type(y_raw, _infer_type(y_raw))
    if inferred == "time_series":
        inferred = "regression"
    if inferred == "unsuitable":
        inferred = "regression" if pd.api.types.is_numeric_dtype(y_raw) else "classification"

    metadata = {
        "excluded_feature_columns": excluded,
        "feature_columns": feature_cols,
        "warnings": [],
        "reproducibility": {
            "random_state": random_state,
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "rows_used": int(len(work)),
            "cv_folds": settings.cv_folds,
        },
    }

    if inferred == "classification":
        target_info = validate_classification_target(y_raw)
        metadata["warnings"].extend(target_info.get("warnings", []))
        result = _train_classification(X, y_raw, test_size, random_state)
    elif inferred == "regression":
        result = _train_regression(X, y_raw, test_size, random_state)
    else:
        raise ValueError(f"Unsupported task type: {inferred}")

    result.update(metadata)

    if result.get("task_type") == "regression":
        r2 = result.get("best_metrics", {}).get("r2")
        if r2 is not None and r2 < 0:
            result["warnings"].append(
                f"R² is {r2:.4f} (negative) — the model performs worse than a baseline that always "
                "predicts the mean target value. Treat metrics and feature importance as unreliable."
            )

    best_pipeline = result.pop("_best_pipeline", None)
    y_for_explain = y_raw
    if result.get("task_type") == "classification" and result.get("label_mapping"):
        y_for_explain = y_raw.astype(str)
    result["explainability"] = build_explainability(
        pipeline=best_pipeline,
        X=X,
        y=y_for_explain,
        task_type=result.get("task_type", inferred),
        feature_importance=result.get("feature_importance"),
    )
    if best_pipeline is not None:
        result["_best_pipeline"] = best_pipeline
    return result


def _infer_type(series: pd.Series) -> str:
    n = max(len(series), 1)
    low_cardinality_limit = max(2, min(50, int(n * 0.05)))
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique(dropna=True) <= low_cardinality_limit:
            return "categorical"
        return "numeric"
    return "categorical"


def _prepare_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    work = sanitize_features_for_ml(X)
    ohe_limit = onehot_max_categories(len(work))
    numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
    onehot_cols: list[str] = []
    frequency_cols: list[str] = []
    for col in work.columns:
        if col in numeric_cols:
            continue
        nunique = work[col].nunique(dropna=True)
        if nunique <= ohe_limit:
            onehot_cols.append(col)
        else:
            frequency_cols.append(col)
    return work, numeric_cols, onehot_cols, frequency_cols


def _build_preprocessor(
    numeric_cols: list[str],
    onehot_cols: list[str],
    frequency_cols: list[str],
    row_count: int,
    task_type: str = "regression",
) -> ColumnTransformer:
    ohe_limit = onehot_max_categories(row_count)
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            )
        )
    if onehot_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=ohe_limit)),
                ]),
                onehot_cols,
            )
        )
    if frequency_cols:
        if task_type == "regression":
            transformers.append(("meanenc", MeanEncoder(), frequency_cols))
        else:
            transformers.append(("freq", FrequencyEncoder(), frequency_cols))
    if not transformers:
        raise ValueError("No usable feature columns remain for model training.")
    return ColumnTransformer(transformers=transformers)


def _split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    stratify: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if stratify and y.nunique() > 1:
        try:
            return train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            pass
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _tune_random_forest(pipeline: Pipeline, X_train, y_train, task_type: str, random_state: int) -> Pipeline:
    if not settings.enable_hyperparameter_tuning:
        pipeline.fit(X_train, y_train)
        return pipeline

    param_dist = {
        "model__n_estimators": [50, 100, 150],
        "model__max_depth": [None, 8, 16],
        "model__min_samples_leaf": [1, 2, 4],
    }
    scoring = "r2" if task_type == "regression" else "f1_weighted"
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=6,
        cv=min(settings.cv_folds, 3),
        random_state=random_state,
        n_jobs=1,
        scoring=scoring,
        error_score="raise",
    )
    try:
        search.fit(X_train, y_train)
        return search.best_estimator_
    except Exception:
        pipeline.fit(X_train, y_train)
        return pipeline


def _cross_validate(pipeline: Pipeline, X, y, task_type: str, random_state: int) -> dict[str, Any]:
    scoring = "r2" if task_type == "regression" else "f1_weighted"
    folds = min(settings.cv_folds, max(2, int(len(y) // 10)))
    try:
        scores = cross_val_score(pipeline, X, y, cv=folds, scoring=scoring, n_jobs=1)
        return {
            "folds": folds,
            "scoring": scoring,
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "scores": [float(s) for s in scores],
        }
    except Exception as exc:
        return {"folds": folds, "scoring": scoring, "error": str(exc)}


def _regression_diagnostics(y_true, y_pred) -> dict[str, Any]:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    sample_n = min(50, len(residuals))
    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "baseline_note": "A mean predictor always has R²=0; negative R² means worse than baseline.",
        "actual_vs_predicted": [
            {"actual": float(y_true[i]), "predicted": float(y_pred[i])}
            for i in range(sample_n)
        ],
    }


def _classification_diagnostics(y_true, y_pred, label_classes: list | None) -> dict[str, Any]:
    labels = label_classes or sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))) if label_classes else None)
    return {
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def _train_regression(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> dict[str, Any]:
    y = pd.to_numeric(y, errors="coerce")
    mask = y.notna()
    X, y = X.loc[mask], y.loc[mask]

    X, numeric_cols, onehot_cols, frequency_cols = _prepare_features(X)
    preprocessor = _build_preprocessor(
        numeric_cols, onehot_cols, frequency_cols, len(X), task_type="regression"
    )

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
    }
    if HAS_XGBOOST:
        models["xgboost"] = XGBRegressor(
            n_estimators=100,
            random_state=random_state,
            verbosity=0,
            objective="reg:squarederror",
        )

    return _evaluate_models(models, preprocessor, X, y, test_size, random_state, task_type="regression")


def _train_classification(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int) -> dict[str, Any]:
    label_encoder = LabelEncoder()
    y_encoded = pd.Series(label_encoder.fit_transform(y.astype(str)))

    X, numeric_cols, onehot_cols, frequency_cols = _prepare_features(X)
    preprocessor = _build_preprocessor(
        numeric_cols, onehot_cols, frequency_cols, len(X), task_type="classification"
    )

    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
    }
    if HAS_XGBOOST:
        models["xgboost"] = XGBClassifier(
            n_estimators=100,
            random_state=random_state,
            verbosity=0,
            eval_metric="mlogloss" if y_encoded.nunique() > 2 else "logloss",
        )

    result = _evaluate_models(
        models,
        preprocessor,
        X,
        y_encoded,
        test_size,
        random_state,
        task_type="classification",
        label_classes=label_encoder.classes_.tolist(),
        stratify=True,
    )
    result["label_mapping"] = {str(i): str(c) for i, c in enumerate(label_encoder.classes_)}
    return result


def _evaluate_models(
    models: dict[str, Any],
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
    task_type: str,
    label_classes: list | None = None,
    stratify: bool = False,
) -> dict[str, Any]:
    X_train, X_test, y_train, y_test = _split_data(X, y, test_size, random_state, stratify=stratify)

    leaderboard: list[dict[str, Any]] = []
    best_name = None
    best_score = float("-inf")
    best_model = None
    best_metrics: dict[str, Any] = {}

    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", clone(preprocessor)), ("model", estimator)])
        try:
            if name == "random_forest":
                pipeline = _tune_random_forest(pipeline, X_train, y_train, task_type, random_state)
            else:
                pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            metrics = _compute_metrics(task_type, y_test, preds, pipeline, X_test)
            score = metrics.get("primary_score", float("-inf"))
            leaderboard.append({"model": name, "metrics": metrics})
            if score > best_score:
                best_score = score
                best_name = name
                best_model = pipeline
                best_metrics = metrics
        except Exception as exc:
            leaderboard.append({"model": name, "error": str(exc)})

    cv_results = _cross_validate(best_model, X, y, task_type, random_state) if best_model else {}
    diagnostics: dict[str, Any] = {}
    if best_model is not None:
        preds = best_model.predict(X_test)
        if task_type == "regression":
            diagnostics = _regression_diagnostics(y_test, preds)
        else:
            diagnostics = _classification_diagnostics(y_test, preds, label_classes)

    feature_importance = _extract_feature_importance(best_model, X.columns.tolist()) if best_model else []

    return {
        "task_type": task_type,
        "best_model": best_name,
        "best_metrics": best_metrics,
        "leaderboard": leaderboard,
        "feature_importance": feature_importance,
        "cross_validation": cv_results,
        "diagnostics": diagnostics,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "label_classes": label_classes,
        "_best_pipeline": best_model,
    }


def _compute_metrics(task_type: str, y_true, y_pred, pipeline, X_test) -> dict[str, Any]:
    if task_type == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2, "primary_score": r2}
    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics = {"accuracy": accuracy, "f1_weighted": f1, "primary_score": f1}
    try:
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            proba = pipeline.predict_proba(X_test)
            if proba.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
    except Exception:
        pass
    return metrics


def _extract_feature_importance(pipeline, original_columns: list[str]) -> list[dict[str, Any]]:
    from app.ml.explainability import _scores_to_feature_list

    model = pipeline.named_steps.get("model")
    if model is None:
        return []

    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

    if importances is None:
        return []

    preprocessor = pipeline.named_steps.get("preprocessor")
    names = original_columns
    if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
        try:
            names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            pass
    return _scores_to_feature_list(names, importances)
