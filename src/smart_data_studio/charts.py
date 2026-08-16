"""Validated chart specifications rendered with Plotly Express."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

# Warm accent first so single-series charts pick up the app's own colour.
PALETTE = ["#d96b45", "#2f6f62", "#c9a227", "#7c5c8f", "#4a7fa5", "#8f5230", "#5b8c7b"]


@dataclass(frozen=True)
class ChartSpec:
    kind: str
    x: str
    y: str | None = None
    color: str | None = None
    title: str | None = None


def make_figure(frame: pd.DataFrame, spec: ChartSpec) -> Figure:
    kind = spec.kind.lower()
    supported = {"bar", "line", "scatter", "area", "histogram", "box", "pie"}
    if kind not in supported:
        raise ValueError(f"Unsupported chart kind: {spec.kind}. Choose from {sorted(supported)}")

    required = [spec.x, spec.y, spec.color]
    missing = [column for column in required if column and column not in frame.columns]
    if missing:
        raise ValueError(f"Chart column(s) not found: {', '.join(missing)}")
    if kind in {"bar", "line", "scatter", "area", "pie"} and not spec.y:
        raise ValueError(f"A y column is required for a {kind} chart")

    common = {"data_frame": frame, "x": spec.x, "title": spec.title}
    if kind == "histogram":
        figure = px.histogram(**common, color=spec.color)
    elif kind == "box":
        figure = px.box(**common, y=spec.y, color=spec.color)
    elif kind == "pie":
        figure = px.pie(frame, names=spec.x, values=spec.y, color=spec.color, title=spec.title)
    else:
        chart = getattr(px, kind)
        figure = chart(**common, y=spec.y, color=spec.color)
    figure.update_layout(
        template="plotly_white",
        colorway=PALETTE,
        margin=dict(l=8, r=8, t=52 if spec.title else 16, b=8),
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0),
        title=dict(font=dict(size=17), x=0, xanchor="left"),
        font=dict(size=13, color="#17211d"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="#17211d", font=dict(color="#f6f1e7", size=12)),
    )
    figure.update_xaxes(showgrid=False, linecolor="#d9ded8", ticks="outside", tickcolor="#d9ded8")
    figure.update_yaxes(gridcolor="#e7eae5", zerolinecolor="#d9ded8", linecolor="rgba(0,0,0,0)")
    return figure
