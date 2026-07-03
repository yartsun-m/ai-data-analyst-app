from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.utils.column_roles import detect_column_roles
from app.utils.json_utils import plotly_figure_to_dict


def generate_eda(df: pd.DataFrame, column_types: dict[str, str]) -> dict[str, Any]:
    charts: list[dict[str, Any]] = []
    insights: list[str] = []
    column_roles = detect_column_roles(df)
    skip_roles = {"identifier", "phone", "email", "url"}

    numeric_cols = [
        c
        for c, t in column_types.items()
        if t == "numeric" and c in df.columns and column_roles.get(c) not in skip_roles
    ]
    categorical_cols = [c for c, t in column_types.items() if t in {"categorical", "boolean"} and c in df.columns]
    datetime_cols = [c for c, t in column_types.items() if t == "datetime" and c in df.columns]

    for col in numeric_cols[:8]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        fig = px.histogram(series, x=col, nbins=30, title=f"Distribution: {col}")
        charts.append(_chart_payload(f"histogram_{col}", "histogram", col, fig))

    if len(numeric_cols) >= 2:
        numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        corr = numeric_df.corr(numeric_only=True)
        if not corr.empty:
            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                title="Correlation Matrix",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
            )
            charts.append(_chart_payload("correlation_matrix", "heatmap", "correlation", fig))
            strong = _strong_correlations(corr)
            if strong:
                insights.append(f"Strong correlations detected: {', '.join(strong[:5])}")

    for col in numeric_cols[:6]:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        fig = px.box(y=series, title=f"Outliers: {col}", labels={"y": col})
        charts.append(_chart_payload(f"boxplot_{col}", "boxplot", col, fig))

    for col in categorical_cols[:6]:
        counts = df[col].astype(str).value_counts().head(15)
        if counts.empty:
            continue
        fig = px.bar(x=counts.index.astype(str), y=counts.values, title=f"Category Distribution: {col}", labels={"x": col, "y": "count"})
        charts.append(_chart_payload(f"category_{col}", "bar", col, fig))

    for col in datetime_cols[:2]:
        parsed = pd.to_datetime(df[col], errors="coerce")
        temp = df.copy()
        temp["_dt"] = parsed
        temp = temp.dropna(subset=["_dt"]).sort_values("_dt")
        if temp.empty:
            continue
        value_col = numeric_cols[0] if numeric_cols else None
        if value_col and value_col in temp.columns:
            fig = px.line(temp, x="_dt", y=value_col, title=f"Time Series: {value_col} over {col}")
            charts.append(_chart_payload(f"timeseries_{col}_{value_col}", "line", f"{col}/{value_col}", fig))
        else:
            counts = temp.groupby(temp["_dt"].dt.date).size().reset_index(name="count")
            fig = px.line(counts, x="_dt", y="count", title=f"Record Count over {col}")
            charts.append(_chart_payload(f"timeseries_count_{col}", "line", col, fig))

    return {
        "chart_count": len(charts),
        "charts": charts,
        "insights": insights,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "datetime_columns": datetime_cols,
    }


def _chart_payload(chart_id: str, chart_type: str, column: str, fig: go.Figure) -> dict[str, Any]:
    return {
        "id": chart_id,
        "type": chart_type,
        "column": column,
        "figure": plotly_figure_to_dict(fig),
    }


def _strong_correlations(corr: pd.DataFrame, threshold: float = 0.7) -> list[str]:
    pairs: list[str] = []
    cols = corr.columns.tolist()
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append(f"{col_a} ↔ {col_b} ({value:.2f})")
    return pairs


def generate_custom_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str | None = None,
    chart_type: str = "scatter",
) -> dict[str, Any]:
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found.")
    if y_column and y_column not in df.columns:
        raise ValueError(f"Column '{y_column}' not found.")

    if chart_type == "histogram" or y_column is None:
        series = df[x_column]
        if pd.api.types.is_numeric_dtype(series):
            fig = px.histogram(series.dropna(), x=x_column, nbins=30, title=f"Histogram: {x_column}")
        else:
            counts = series.astype(str).value_counts().head(20)
            fig = px.bar(x=counts.index, y=counts.values, title=f"Bar: {x_column}")
        return _chart_payload(f"custom_{x_column}", chart_type, x_column, fig)

    x_series = df[x_column]
    y_series = df[y_column]
    if chart_type == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    elif chart_type == "line":
        fig = px.line(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    elif chart_type == "box":
        fig = px.box(df, x=x_column, y=y_column, title=f"Box: {y_column} by {x_column}")
    else:
        fig = px.scatter(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    return _chart_payload(f"custom_{x_column}_{y_column}", chart_type, f"{x_column}/{y_column}", fig)
