from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio

from app.services.dashboard_service import build_dashboard


def build_html_report(
    df,
    profile: dict[str, Any] | None,
    cleaning_report: dict[str, Any] | None,
    eda: dict[str, Any] | None,
    ml_results: dict[str, Any] | None,
    filename: str | None = None,
) -> str:
    dashboard = build_dashboard(df, profile, cleaning_report, eda, ml_results)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    title = dashboard.get("title", "AI Data Analyst Report")
    source = html.escape(filename or "dataset")

    parts: list[str] = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='utf-8'/>",
        f"<title>{html.escape(title)}</title>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'/>",
        _report_styles(),
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        f"<p class='meta'>Generated {generated_at} · Source: {source}</p>",
        _section_kpis(dashboard.get("kpis", [])),
        _section_report_blocks(dashboard.get("report_sections", [])),
        _section_profile(dashboard.get("profile_summary", {})),
        _section_cleaning(dashboard.get("cleaning_summary", {})),
        _section_eda_insights(eda),
        _section_ml(dashboard.get("ml_summary", {})),
        "<h2>Charts</h2>",
        _section_charts(dashboard.get("charts", [])),
        "<footer><p>AI Data Analyst — automated data report</p></footer>",
        "</body>",
        "</html>",
    ]
    return "\n".join(parts)


def _report_styles() -> str:
    return """
<style>
  body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin: 2rem; color: #111; line-height: 1.5; }
  h1 { margin-bottom: 0.25rem; }
  h2 { margin-top: 2rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.35rem; }
  .meta { color: #6b7280; font-size: 0.9rem; }
  .kpi-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 1rem; margin: 1.5rem 0; }
  .kpi { border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem; background: #fafafa; }
  .kpi-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.03em; }
  .kpi-value { font-size: 1.35rem; font-weight: 600; margin-top: 0.25rem; }
  .card { border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0; background: #fff; }
  table { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 0.75rem 0; }
  th, td { border: 1px solid #e5e7eb; padding: 0.45rem 0.65rem; text-align: left; }
  th { background: #f3f4f6; }
  .chart-block { margin: 2rem 0; page-break-inside: avoid; }
  .chart-title { font-weight: 600; margin-bottom: 0.25rem; }
  .chart-type { color: #6b7280; font-size: 0.85rem; margin-bottom: 0.75rem; }
  ul { padding-left: 1.25rem; }
  footer { margin-top: 3rem; color: #9ca3af; font-size: 0.8rem; }
</style>
"""


def _section_kpis(kpis: list[dict[str, Any]]) -> str:
    if not kpis:
        return ""
    cells = "".join(
        f"<div class='kpi'><div class='kpi-label'>{html.escape(str(k['label']))}</div>"
        f"<div class='kpi-value'>{html.escape(str(k['value']))}</div></div>"
        for k in kpis
    )
    return f"<h2>Key metrics</h2><div class='kpi-grid'>{cells}</div>"


def _section_report_blocks(sections: list[dict[str, str]]) -> str:
    if not sections:
        return ""
    blocks = "".join(
        f"<div class='card'><h3>{html.escape(s['title'])}</h3><p>{html.escape(s['content'])}</p></div>"
        for s in sections
    )
    return f"<h2>Summary</h2>{blocks}"


def _section_profile(profile_summary: dict[str, Any]) -> str:
    if not profile_summary:
        return ""
    rows = ""
    column_types = profile_summary.get("column_types") or {}
    for col, col_type in column_types.items():
        rows += f"<tr><td>{html.escape(str(col))}</td><td>{html.escape(str(col_type))}</td></tr>"
    targets = profile_summary.get("suggested_targets") or []
    target_text = ", ".join(html.escape(str(t)) for t in targets) if targets else "—"
    shape = profile_summary.get("shape") or {}
    shape_text = f"{shape.get('rows', '?')} rows × {shape.get('columns', '?')} columns"
    return (
        "<h2>Dataset profile</h2>"
        f"<p>{html.escape(shape_text)} · Suggested targets: {target_text}</p>"
        f"<table><thead><tr><th>Column</th><th>Type</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def _section_cleaning(cleaning_summary: dict[str, Any]) -> str:
    if not cleaning_summary or not cleaning_summary.get("steps"):
        return ""
    steps = "".join(f"<li>{html.escape(str(step))}</li>" for step in cleaning_summary["steps"])
    return f"<h2>Data cleaning</h2><ul>{steps}</ul>"


def _section_eda_insights(eda: dict[str, Any] | None) -> str:
    if not eda or not eda.get("insights"):
        return ""
    items = "".join(f"<li>{html.escape(str(i))}</li>" for i in eda["insights"])
    return f"<h2>EDA insights</h2><ul>{items}</ul>"


def _section_ml(ml_summary: dict[str, Any]) -> str:
    if not ml_summary or not ml_summary.get("best_model"):
        return ""
    metrics = ml_summary.get("best_metrics") or {}
    metrics_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in metrics.items()
        if k != "primary_score"
    )
    leaderboard = ml_summary.get("leaderboard") or []
    lb_rows = ""
    for entry in leaderboard:
        name = html.escape(str(entry.get("model", "?")))
        if entry.get("metrics"):
            detail = html.escape(json.dumps(entry["metrics"]))
        else:
            detail = html.escape(str(entry.get("error", "—")))
        lb_rows += f"<tr><td>{name}</td><td>{detail}</td></tr>"

    importance = ml_summary.get("feature_importance") or []
    fi_rows = "".join(
        f"<tr><td>{html.escape(str(item.get('feature', '')))}</td>"
        f"<td>{html.escape(str(item.get('importance', '')))}</td></tr>"
        for item in importance[:15]
    )

    parts = [
        "<h2>Machine learning</h2>",
        f"<p>Task: <strong>{html.escape(str(ml_summary.get('task_type', '')))}</strong> · "
        f"Best model: <strong>{html.escape(str(ml_summary.get('best_model', '')))}</strong></p>",
    ]
    if metrics_rows:
        parts.append(f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{metrics_rows}</tbody></table>")
    if lb_rows:
        parts.append(
            f"<h3>Model leaderboard</h3>"
            f"<table><thead><tr><th>Model</th><th>Result</th></tr></thead><tbody>{lb_rows}</tbody></table>"
        )
    if fi_rows:
        parts.append(
            f"<h3>Feature importance</h3>"
            f"<table><thead><tr><th>Feature</th><th>Importance</th></tr></thead><tbody>{fi_rows}</tbody></table>"
        )
    return "".join(parts)


def _section_charts(charts: list[dict[str, Any]]) -> str:
    if not charts:
        return "<p>No charts were generated for this dataset.</p>"

    blocks: list[str] = []
    for idx, chart in enumerate(charts):
        column = html.escape(str(chart.get("column", f"Chart {idx + 1}")))
        chart_type = html.escape(str(chart.get("type", "")))
        figure_dict = chart.get("figure") or {}
        try:
            fig = go.Figure(figure_dict)
            plotly_fragment = pio.to_html(
                fig,
                full_html=False,
                include_plotlyjs="cdn" if idx == 0 else False,
                config={"responsive": True, "displayModeBar": False},
            )
        except Exception as exc:
            plotly_fragment = f"<p>Could not render chart: {html.escape(str(exc))}</p>"

        blocks.append(
            f"<div class='chart-block'>"
            f"<div class='chart-title'>{column}</div>"
            f"<div class='chart-type'>{chart_type}</div>"
            f"{plotly_fragment}"
            f"</div>"
        )
    return "".join(blocks)
