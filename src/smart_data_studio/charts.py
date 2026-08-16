"""Validated chart specifications rendered with Plotly Express."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure

# Warm accent first so single-series charts pick up the app's own colour.
PALETTE = ["#d96b45", "#2f6f62", "#c9a227", "#7c5c8f", "#4a7fa5", "#8f5230", "#5b8c7b"]
INK = "#17211d"
MUTED = "#6b7671"
GRID = "#e7eae5"
AXIS = "#c9d0ca"
# Above this, axis ticks read better as 22M than as 22,056,432.
SI_THRESHOLD = 10_000
MARKER_LIMIT = 30
SORTABLE = {"bar", "pie"}


@dataclass(frozen=True)
class ChartSpec:
    kind: str
    x: str
    y: str | None = None
    color: str | None = None
    title: str | None = None


def label(name: str) -> str:
    """total_theo_win -> Total Theo Win, avgBet -> Avg Bet."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).replace("_", " ")
    return " ".join(word.capitalize() if word.islower() else word for word in spaced.split())


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

    frame = _ordered(frame, spec, kind)
    labels = {column: label(column) for column in frame.columns}
    # Express fixes trace colours at construction, so a layout colorway alone
    # leaves single-series charts in Plotly's default blue.
    common = {
        "data_frame": frame,
        "x": spec.x,
        "title": spec.title,
        "labels": labels,
        "color_discrete_sequence": PALETTE,
    }

    if kind == "histogram":
        figure = px.histogram(**common, color=spec.color)
    elif kind == "box":
        figure = px.box(**common, y=spec.y, color=spec.color)
    elif kind == "pie":
        figure = px.pie(
            frame,
            names=spec.x,
            values=spec.y,
            color=spec.color,
            title=spec.title,
            hole=0.45,
            color_discrete_sequence=PALETTE,
        )
    else:
        figure = getattr(px, kind)(**common, y=spec.y, color=spec.color)

    _style(figure, frame, spec, kind)
    return figure


def _ordered(frame: pd.DataFrame, spec: ChartSpec, kind: str) -> pd.DataFrame:
    """Rank categories by size, unless the axis is a date, where order carries meaning."""
    if kind not in SORTABLE or not spec.y or spec.color:
        return frame
    if pd.api.types.is_datetime64_any_dtype(frame[spec.x]):
        return frame
    if not pd.api.types.is_numeric_dtype(frame[spec.y]):
        return frame
    return frame.sort_values(spec.y, ascending=False)


def _style(figure: Figure, frame: pd.DataFrame, spec: ChartSpec, kind: str) -> None:
    figure.update_layout(
        template="plotly_white",
        colorway=PALETTE,
        margin=dict(l=8, r=8, t=58 if spec.title else 18, b=8),
        title=dict(font=dict(size=18, color=INK), x=0, xanchor="left", y=0.96, yanchor="top"),
        font=dict(size=13, color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor=INK, font=dict(color="#f6f1e7", size=12), bordercolor=INK),
        bargap=0.28,
    )
    if kind == "pie":
        figure.update_traces(textposition="inside", textinfo="percent+label")
        return

    # Only the value axis carries gridlines; a category axis gains nothing from them.
    figure.update_xaxes(
        showgrid=False,
        linecolor=AXIS,
        ticks="outside",
        tickcolor=AXIS,
        title_font=dict(size=12, color=MUTED),
        tickfont=dict(size=11.5),
    )
    figure.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=AXIS,
        linecolor="rgba(0,0,0,0)",
        ticks="",
        title_font=dict(size=12, color=MUTED),
        tickfont=dict(size=11.5),
    )

    value_column = spec.y if kind not in {"histogram"} else None
    if value_column and pd.api.types.is_numeric_dtype(frame[value_column]):
        largest = frame[value_column].abs().max()
        if pd.notna(largest) and largest >= SI_THRESHOLD:
            figure.update_yaxes(tickformat="~s")

    if kind in {"line", "area"} and len(frame) <= MARKER_LIMIT:
        figure.update_traces(mode="lines+markers", marker=dict(size=6))
