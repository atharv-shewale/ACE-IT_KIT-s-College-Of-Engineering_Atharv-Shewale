"""
src/visuals.py
─────────────────────────────────────────────────────────────────────────────
Plotly visualisation layer for the Decision Reversal Pattern Detector.

All functions are pure: they accept DataFrames and return a plotly Figure.
No global state, no side effects.

Public API
──────────
  timeline_chart(events_df, reversals_df, user_id)
      Line chart of choices over time, per decision.
      Reversal events are overlaid as red markers + labels.

  stage_chart(reversals_df)
      Horizontal bar chart of reversal counts per funnel stage (stage order).

  flow_chart(events_df, user_id, decision_id)
      Sankey diagram of choice-to-choice transitions, making loops visible.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from src.config import STAGE_ORDER, HEATMAP_COLOR_SCALE

# ── Shared visual constants ───────────────────────────────────────────────────
_FONT_FAMILY  = "Inter, Arial, sans-serif"
_BG_COLOR     = "#0f1117"           # dark background (matches Streamlit dark theme)
_PAPER_COLOR  = "#0f1117"
_GRID_COLOR   = "#2a2d3a"
_TEXT_COLOR   = "#e0e0e0"
_REVERSAL_RED = "#ff4b4b"           # colour for reversal markers / bars
_PALETTE      = px.colors.qualitative.Bold   # up to 11 distinct colours

# Map each stage to its index for consistent x-axis ordering
_STAGE_INDEX: dict[str, int] = {s: i for i, s in enumerate(STAGE_ORDER)}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Timeline chart
# ─────────────────────────────────────────────────────────────────────────────

def timeline_chart(
    events_df: pd.DataFrame,
    reversals_df: pd.DataFrame,
    user_id: str,
) -> go.Figure:
    """
    Plot a user's choice history over time, one line per decision.

    Each line shows how the chosen value evolved chronologically.
    Reversal events are overlaid as red circle markers with text labels
    showing "from → to".

    Parameters
    ----------
    events_df    : clean output of loader.load()
    reversals_df : output of detector.detect()
    user_id      : str – user to visualise

    Returns
    -------
    go.Figure
    """
    # ── Filter to the target user ──────────────────────────────────────────
    u_events = (
        events_df[events_df["user_id"] == user_id]
        .sort_values("timestamp")
        .copy()
    )
    u_revs = (
        reversals_df[reversals_df["user_id"] == user_id].copy()
        if not reversals_df.empty
        else reversals_df
    )

    if u_events.empty:
        return _empty_figure(f"No events found for user {user_id}")

    # Encode choice values as numeric so they can be plotted on a y-axis.
    # Each decision gets its own y-category mapping.
    decision_ids = sorted(u_events["decision_id"].unique())
    fig = go.Figure()

    for idx, did in enumerate(decision_ids):
        color = _PALETTE[idx % len(_PALETTE)]
        dec_events = u_events[u_events["decision_id"] == did].copy()

        # Build a clean label for the legend (strip the user-prefix)
        dec_label = _decision_label(did)

        # All unique choices for this decision → numeric rank
        choice_order = list(dict.fromkeys(dec_events["choice"].tolist()))
        choice_rank  = {c: i for i, c in enumerate(choice_order)}
        dec_events["y_val"]  = dec_events["choice"].map(choice_rank)
        dec_events["y_tick"] = dec_events["choice"]

        # ── Main choice-over-time line ─────────────────────────────────────
        fig.add_trace(go.Scatter(
            x          = dec_events["timestamp"],
            y          = dec_events["y_val"],
            mode       = "lines+markers",
            name       = dec_label,
            line       = dict(color=color, width=2),
            marker     = dict(color=color, size=7),
            text       = dec_events["choice"],
            hovertemplate = (
                f"<b>{dec_label}</b><br>"
                "Time: %{x}<br>"
                "Choice: %{text}<extra></extra>"
            ),
        ))

        # ── Reversal markers (red) ─────────────────────────────────────────
        dec_revs = u_revs[u_revs["decision_id"] == did]
        if not dec_revs.empty:
            # Match each reversal to the y-position of its to_choice
            rev_y     = dec_revs["to_choice"].map(choice_rank)
            rev_label = dec_revs["from_choice"] + " → " + dec_revs["to_choice"]
            flip_back_flag = dec_revs["is_flip_back"].apply(
                lambda fb: " ↩ flip-back" if fb else ""
            )

            fig.add_trace(go.Scatter(
                x    = dec_revs["timestamp"],
                y    = rev_y,
                mode = "markers+text",
                name = f"{dec_label} reversals",
                marker = dict(
                    color   = _REVERSAL_RED,
                    size    = 14,
                    symbol  = "circle-open",
                    line    = dict(width=2.5, color=_REVERSAL_RED),
                ),
                text          = rev_label + flip_back_flag,
                textposition  = "top center",
                textfont      = dict(color=_REVERSAL_RED, size=10),
                hovertemplate = (
                    f"<b>Reversal – {dec_label}</b><br>"
                    "Time: %{x}<br>"
                    "Change: %{text}<extra></extra>"
                ),
                showlegend = False,
            ))

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title = dict(
            text     = f"Choice Timeline — {user_id}",
            font     = dict(size=16, color=_TEXT_COLOR, family=_FONT_FAMILY),
            x        = 0.5,
            xanchor  = "center",
        ),
        xaxis = dict(
            title      = "Time",
            showgrid   = True,
            gridcolor  = _GRID_COLOR,
            tickfont   = dict(color=_TEXT_COLOR),
            title_font = dict(color=_TEXT_COLOR),
        ),
        yaxis = dict(
            title      = "Choice value (per decision, 0-indexed)",
            showgrid   = True,
            gridcolor  = _GRID_COLOR,
            tickfont   = dict(color=_TEXT_COLOR),
            title_font = dict(color=_TEXT_COLOR),
        ),
        legend = dict(
            font      = dict(color=_TEXT_COLOR),
            bgcolor   = "rgba(0,0,0,0)",
            bordercolor = _GRID_COLOR,
        ),
        plot_bgcolor  = _BG_COLOR,
        paper_bgcolor = _PAPER_COLOR,
        font          = dict(family=_FONT_FAMILY, color=_TEXT_COLOR),
        hovermode     = "x unified",
        margin        = dict(l=60, r=30, t=60, b=60),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stage bar chart
# ─────────────────────────────────────────────────────────────────────────────

def stage_chart(
    reversals_df: pd.DataFrame,
    stage_order: list[str] | None = None,
) -> go.Figure:
    """
    Horizontal bar chart of reversal counts per funnel stage.

    Stages are displayed in the active funnel order,
    even if some stages have zero reversals.

    Parameters
    ----------
    reversals_df : output of detector.detect()
    stage_order  : list[str] | None – custom stages in order (auto-detected if None)

    Returns
    -------
    go.Figure
    """
    if reversals_df.empty:
        return _empty_figure("No reversals detected")

    if stage_order is None:
        stage_order = reversals_df.attrs.get("stage_order")
        if stage_order is None:
            unique_stages = list(dict.fromkeys(reversals_df["stage"].dropna().tolist()))
            canon_subset = [s for s in STAGE_ORDER if s in unique_stages]
            other_stages = [s for s in unique_stages if s not in STAGE_ORDER]
            stage_order = (canon_subset + other_stages) if (canon_subset or other_stages) else STAGE_ORDER

    # Count reversals per stage and re-index to ensure all stages present
    counts_series = reversals_df["stage"].value_counts()
    counts = {stage: int(counts_series.get(stage, 0)) for stage in stage_order}

    stages  = stage_order                       # x-axis (displayed vertically)
    values  = [counts[s] for s in stages]       # reversal counts
    colours = _stage_colour_scale(values)       # shade bars by intensity

    fig = go.Figure(go.Bar(
        x             = values,
        y             = stages,
        orientation   = "h",                    # horizontal so labels are readable
        marker_color  = colours,
        text          = values,
        textposition  = "outside",
        textfont      = dict(color=_TEXT_COLOR, size=12),
        hovertemplate = "<b>%{y}</b><br>Reversals: %{x}<extra></extra>",
    ))

    fig.update_layout(
        title = dict(
            text    = "Reversal Count by Funnel Stage",
            font    = dict(size=16, color=_TEXT_COLOR, family=_FONT_FAMILY),
            x       = 0.5,
            xanchor = "center",
        ),
        xaxis = dict(
            title     = "Number of Reversals",
            showgrid  = True,
            gridcolor = _GRID_COLOR,
            tickfont  = dict(color=_TEXT_COLOR),
            title_font = dict(color=_TEXT_COLOR),
        ),
        yaxis = dict(
            tickfont    = dict(color=_TEXT_COLOR, size=12),
            # Reverse so Browsing is at the top (funnel order)
            autorange   = "reversed",
            categoryorder = "array",
            categoryarray = STAGE_ORDER,
        ),
        plot_bgcolor  = _BG_COLOR,
        paper_bgcolor = _PAPER_COLOR,
        font          = dict(family=_FONT_FAMILY, color=_TEXT_COLOR),
        margin        = dict(l=120, r=60, t=60, b=60),
        showlegend    = False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sankey flow chart
# ─────────────────────────────────────────────────────────────────────────────

def flow_chart(
    events_df: pd.DataFrame,
    user_id: str,
    decision_id: str,
) -> go.Figure:
    """
    Sankey diagram of sequential choice-to-choice transitions for one
    (user, decision) pair.

    Each unique choice string becomes a node.  Each consecutive change in
    choice creates a directed link.  When the same choice appears at multiple
    positions in the sequence (e.g. A→B→A), it is disambiguated by appending
    a position index so the loop is visually clear.

    Parameters
    ----------
    events_df   : clean output of loader.load()
    user_id     : str
    decision_id : str

    Returns
    -------
    go.Figure
    """
    dec_events = (
        events_df[
            (events_df["user_id"]     == user_id) &
            (events_df["decision_id"] == decision_id)
        ]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if dec_events.empty:
        return _empty_figure(
            f"No events for user {user_id}, decision {decision_id}"
        )

    choices = dec_events["choice"].tolist()

    if len(choices) < 2:
        return _empty_figure(
            f"{user_id} / {_decision_label(decision_id)}: only one event, "
            "no transitions to show."
        )

    # ── Build disambiguated node labels ───────────────────────────────────
    # Each position in the sequence gets its own node label "<choice>#<pos>",
    # so revisiting a choice creates a visible loop rather than collapsing it.
    node_labels: list[str] = [f"{c}" for c in choices]  # position-unique labels
    node_display: list[str] = choices                    # human-readable (no suffix)

    # ── Build link (source→target) pairs from consecutive events ──────────
    link_map: dict[tuple[int, int], int] = {}   # (src_idx, tgt_idx) → flow count
    for i in range(len(choices) - 1):
        key = (i, i + 1)
        link_map[key] = link_map.get(key, 0) + 1

    sources = [k[0] for k in link_map]
    targets = [k[1] for k in link_map]
    values  = list(link_map.values())

    # ── Colour nodes by position in the palette ────────────────────────────
    node_colours = [
        _PALETTE[i % len(_PALETTE)] for i in range(len(node_labels))
    ]
    link_colours = [
        "rgba(255,75,75,0.35)" for _ in sources   # semi-transparent red links
    ]

    dec_label = _decision_label(decision_id)

    fig = go.Figure(go.Sankey(
        arrangement = "snap",
        node = dict(
            pad        = 24,
            thickness  = 20,
            line       = dict(color=_GRID_COLOR, width=0.5),
            label      = node_display,
            color      = node_colours,
            hovertemplate = "Choice: <b>%{label}</b><extra></extra>",
        ),
        link = dict(
            source        = sources,
            target        = targets,
            value         = values,
            color         = link_colours,
            hovertemplate = (
                "Transition: <b>%{source.label}</b> → "
                "<b>%{target.label}</b><extra></extra>"
            ),
        ),
    ))

    fig.update_layout(
        title = dict(
            text    = f"Choice Flow — {user_id} / {dec_label}",
            font    = dict(size=16, color=_TEXT_COLOR, family=_FONT_FAMILY),
            x       = 0.5,
            xanchor = "center",
        ),
        paper_bgcolor = _PAPER_COLOR,
        font          = dict(family=_FONT_FAMILY, color=_TEXT_COLOR, size=13),
        margin        = dict(l=30, r=30, t=60, b=30),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Internal utilities
# ─────────────────────────────────────────────────────────────────────────────

def _empty_figure(message: str) -> go.Figure:
    """Return a blank figure with a centred message (instead of crashing)."""
    fig = go.Figure()
    fig.add_annotation(
        text      = message,
        x         = 0.5,
        y         = 0.5,
        xref      = "paper",
        yref      = "paper",
        showarrow = False,
        font      = dict(size=14, color=_TEXT_COLOR, family=_FONT_FAMILY),
    )
    fig.update_layout(
        plot_bgcolor  = _BG_COLOR,
        paper_bgcolor = _PAPER_COLOR,
        xaxis = dict(visible=False),
        yaxis = dict(visible=False),
        margin = dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _decision_label(decision_id: str) -> str:
    """Clean label for display: 'U001_payment_method' → 'payment_method'."""
    s = str(decision_id)
    parts = s.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("U") and parts[0][1:].isdigit():
        return parts[1]
    return s


def _stage_colour_scale(values: list[int]) -> list[str]:
    """
    Map stage bar heights to a colour gradient from muted to vibrant red,
    so higher-reversal stages stand out visually.
    """
    max_v = max(values) if max(values) > 0 else 1
    colours = []
    for v in values:
        # Linearly interpolate alpha: low count → muted, high count → vivid
        intensity = v / max_v          # 0.0 – 1.0
        r = 255
        g = int(75  + (1 - intensity) * 100)   # 75 (vivid) → 175 (muted)
        b = int(75  + (1 - intensity) * 100)
        colours.append(f"rgb({r},{g},{b})")
    return colours
