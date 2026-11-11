"""Графики Plotly. Цвет учебника закреплён за учебником и не зависит от места в рейтинге.

Подписи берутся из core/i18n.py на текущем языке интерфейса.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.analysis import ALL, WEAK_MASTERY
from core.i18n import t
from core.report import fmt

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
BLUE_SOFT = "#b7d3f6"
RED = "#e34948"
BOOK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
CONFIG = {"displayModeBar": False, "responsive": True}
PP = " п.п."


def status_label(status: str) -> str:
    return t(f"status_{status}")


def book_color(books: list[str], book: str) -> str:
    return BOOK_COLORS[books.index(book) % len(BOOK_COLORS)]


def _layout(figure: go.Figure, height: int, left: int = 10, showlegend: bool = False) -> go.Figure:
    figure.update_layout(
        height=height,
        margin={"l": left, "r": 24, "t": 12, "b": 40},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, 'Segoe UI', sans-serif", "color": INK_2, "size": 13},
        hoverlabel={"bgcolor": SURFACE, "bordercolor": AXIS, "font": {"color": INK, "size": 13}},
        showlegend=showlegend,
        legend={"orientation": "h", "y": -0.10, "x": 0, "font": {"color": INK_2}},
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont={"color": MUTED})
    figure.update_yaxes(showgrid=False, zeroline=False, linecolor=AXIS, tickfont={"color": INK_2}, ticks="")
    return figure


def topic_mastery(table: pd.DataFrame, weak: float = WEAK_MASTERY) -> go.Figure:
    data = table.iloc[::-1]
    colors = [BLUE if s != "ok" else BLUE_SOFT for s in data["status"]]
    labels = [
        f"{fmt(c, 0)}%" + ("" if s == "ok" else f" · {status_label(s)}")
        for c, s in zip(data["clean"], data["status"])
    ]
    clean_label, raw_label = t("legend_clean"), t("legend_raw")
    figure = go.Figure(
        go.Bar(
            x=data["clean"],
            y=data["name"],
            orientation="h",
            marker={"color": colors, "cornerradius": 4},
            text=labels,
            textposition="outside",
            textfont={"color": INK_2, "size": 13},
            cliponaxis=False,
            name=clean_label,
            hovertemplate=f"%{{y}}<br>{clean_label}: %{{x:.1f}}%<extra></extra>",
        )
    )
    with_bad = data[data["n_bad"] > 0]
    if not with_bad.empty:
        figure.add_trace(
            go.Scatter(
                x=with_bad["raw"],
                y=with_bad["name"],
                mode="markers",
                marker={"symbol": "line-ns", "size": 18, "line": {"color": INK, "width": 2.5}},
                name=raw_label,
                hovertemplate=f"%{{y}}<br>{raw_label}: %{{x:.1f}}%<extra></extra>",
            )
        )
    figure.add_vline(x=weak, line={"color": MUTED, "width": 1, "dash": "dot"})
    figure.update_xaxes(range=[0, 118], tickvals=[0, 20, 40, 60, 80, 100], ticksuffix="%")
    figure.update_layout(bargap=0.32)
    return _layout(figure, 520, showlegend=not with_bad.empty)


def textbook_ranking(means: pd.DataFrame, books: list[str], adjusted: bool) -> go.Figure:
    overall = means[means["scope"] == ALL].set_index("textbook")
    values = pd.concat([overall["naive"], overall["adjusted"]])
    x_range = [np.floor(values.min()) - 2, np.ceil(values.max()) + 3]
    labels = [t("book", b=b) for b in books]
    value_label = t("adjusted_hover") if adjusted else t("mean_hover")
    figure = go.Figure()
    for book, label in zip(books, labels):
        color = book_color(books, book)
        naive, fair = overall.loc[book, "naive"], overall.loc[book, "adjusted"]
        value = fair if adjusted else naive
        if adjusted:
            figure.add_trace(
                go.Scatter(
                    x=[naive, fair], y=[label, label], mode="lines",
                    line={"color": AXIS, "width": 2}, hoverinfo="skip", showlegend=False,
                )
            )
            figure.add_trace(
                go.Scatter(
                    x=[naive], y=[label], mode="markers",
                    marker={"size": 13, "color": SURFACE, "line": {"color": MUTED, "width": 2}},
                    hovertemplate=f"{label}<br>{t('naive_hover')}: {fmt(naive)}%<extra></extra>",
                    showlegend=False,
                )
            )
        figure.add_trace(
            go.Scatter(
                x=[value], y=[label], mode="markers+text",
                marker={"size": 20, "color": color, "line": {"color": SURFACE, "width": 2}},
                text=[f"<b>{fmt(value)}%</b>"],
                textposition="top center",
                textfont={"color": INK, "size": 15},
                cliponaxis=False,
                hovertemplate=f"{label}<br>{value_label}: {fmt(value)}%<extra></extra>",
                showlegend=False,
            )
        )
    figure.update_xaxes(range=x_range, ticksuffix="%", showgrid=True)
    figure.update_yaxes(autorange="reversed", tickfont={"size": 15, "color": INK})
    _layout(figure, 300)
    figure.update_layout(margin={"t": 34})
    return figure


def _interval_hover(prefix: str, value: str = "%{x:+.1f}") -> str:
    return (
        f"%{{y}}<br>{prefix}: {value}{PP}<br>{t('interval_hover')}: "
        "%{customdata[0]:.1f} … %{customdata[1]:.1f}"
    )


def contrast_forest(table: pd.DataFrame, names: dict[str, str], first: str, second: str, books: list[str]) -> go.Figure:
    data = table.iloc[::-1].copy()
    data["name"] = data["scope"].map(names)
    data["group"] = np.select(
        [data["proven"] & (data["diff"] > 0), data["proven"] & (data["diff"] < 0)],
        [first, second],
        default="none",
    )
    figure = go.Figure()
    series = [
        (first, book_color(books, first), t("proven_stronger", x=first)),
        (second, book_color(books, second), t("proven_stronger", x=second)),
        ("none", AXIS, t("not_proven")),
    ]
    for group, color, label in series:
        part = data[data["group"] == group]
        if part.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=part["diff"],
                y=part["name"],
                mode="markers",
                marker={"size": 12, "color": color, "line": {"color": SURFACE, "width": 2}},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": part["high"] - part["diff"],
                    "arrayminus": part["diff"] - part["low"],
                    "color": color,
                    "thickness": 2,
                    "width": 0,
                },
                name=label,
                customdata=np.column_stack([part["low"], part["high"]]),
                hovertemplate=_interval_hover(t("diff_hover")) + "<extra></extra>",
            )
        )
    figure.add_vline(x=0, line={"color": MUTED, "width": 1})
    figure.update_xaxes(ticksuffix=PP)
    figure.update_yaxes(categoryorder="array", categoryarray=list(data["name"]))
    return _layout(figure, 500, showlegend=True)


def hours_plan(table: pd.DataFrame, names: dict[str, str]) -> go.Figure:
    data = table.iloc[::-1].copy()
    data["name"] = data["topic_id"].map(names)
    data["delta"] = data["new"] - data["planned"]
    figure = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.05,
        subplot_titles=(t("hours_change"), t("topic_forecast")),
    )
    figure.add_trace(
        go.Bar(
            x=data["delta"],
            y=data["name"],
            orientation="h",
            marker={"color": [BLUE if d > 0 else RED for d in data["delta"]], "cornerradius": 4},
            text=[f"{d:+d}".replace("-", "−") if d else "" for d in data["delta"]],
            textposition="outside",
            textfont={"color": INK_2},
            cliponaxis=False,
            hovertemplate=f"%{{y}}<br>{t('change_hover')}: %{{x:+d}} {t('hours_unit')}<extra></extra>",
        ),
        row=1, col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=data["gain"],
            y=data["name"],
            mode="markers",
            marker={
                "size": 12,
                "color": [BLUE if g > 0 else RED for g in data["gain"]],
                "line": {"color": SURFACE, "width": 2},
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": data["high"] - data["gain"],
                "arrayminus": data["gain"] - data["low"],
                "color": AXIS,
                "thickness": 2,
                "width": 0,
            },
            customdata=np.column_stack([data["low"], data["high"]]),
            hovertemplate=_interval_hover(t("forecast_hover")) + "<extra></extra>",
        ),
        row=1, col=2,
    )
    spread = max(abs(data["delta"]).max(), 1) + 1
    figure.update_xaxes(range=[-spread, spread], dtick=1, row=1, col=1)
    figure.add_vline(x=0, line={"color": MUTED, "width": 1}, row=1, col=1)
    figure.add_vline(x=0, line={"color": MUTED, "width": 1}, row=1, col=2)
    figure.update_annotations(font={"size": 14, "color": INK_2})
    figure.update_layout(bargap=0.32)
    _layout(figure, 500)
    figure.update_layout(margin={"t": 36})
    return figure


def group_gaps(table: pd.DataFrame, names: dict[str, str], factor: str, first: str, second: str) -> go.Figure:
    """Разница групп по темам при равных условиях, с 95% интервалом. first/second — подписи групп."""
    data = table[(table["factor"] == factor) & (table["scope"] != ALL)].iloc[::-1].copy()
    data["name"] = data["scope"].map(names)
    colors = [BOOK_COLORS[0] if d > 0 else BOOK_COLORS[1] for d in data["diff"]]
    colors = [c if p else AXIS for c, p in zip(colors, data["proven"])]
    figure = go.Figure(
        go.Scatter(
            x=data["diff"],
            y=data["name"],
            mode="markers",
            marker={"size": 12, "color": colors, "line": {"color": SURFACE, "width": 2}},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": data["high"] - data["diff"],
                "arrayminus": data["diff"] - data["low"],
                "color": AXIS,
                "thickness": 2,
                "width": 0,
            },
            customdata=np.column_stack([data["low"], data["high"], data["naive_first"], data["naive_second"]]),
            hovertemplate=(
                _interval_hover(t("equal_hover"))
                + f"<br>{t('naive_hover')}: {first} %{{customdata[2]:.1f}}%, {second} %{{customdata[3]:.1f}}%<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line={"color": MUTED, "width": 1})
    figure.update_xaxes(ticksuffix=PP)
    return _layout(figure, 420)


def truth_scatter(table: pd.DataFrame) -> go.Figure:
    limit = float(max(table["diff"].abs().max(), table["truth"].abs().max())) + 1
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[-limit, limit], y=[-limit, limit], mode="lines",
            line={"color": MUTED, "width": 1, "dash": "dot"}, hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table["truth"], y=table["diff"], mode="markers",
            marker={"size": 10, "color": BLUE, "line": {"color": SURFACE, "width": 2}},
            hovertemplate=(
                f"{t('truth_hover_x')}: %{{x:+.1f}}{PP}<br>{t('truth_hover_y')}: %{{y:+.1f}}{PP}<extra></extra>"
            ),
        )
    )
    figure.update_xaxes(range=[-limit, limit], title={"text": t("truth_x"), "font": {"color": MUTED}})
    figure.update_yaxes(
        range=[-limit, limit], showgrid=True, gridcolor=GRID,
        title={"text": t("truth_y"), "font": {"color": MUTED}},
    )
    return _layout(figure, 380, left=60)
