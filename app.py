"""
app.py
─────────────────────────────────────────────────────────────────────────────
Streamlit dashboard for the Decision Reversal Pattern Detector (CEREBRO).

Supports any arbitrary event dataset with dynamic column mapping,
auto-detection, custom funnel stages, and multiple industry presets:
  • Preset 1: SaaS Subscription Checkout (200 users, 5 funnel stages)
  • Preset 2: E-Commerce Shopping Cart (Orders, Shipping, Payment)
  • Preset 3: Loan / Insurance Application (Eligibility, Quote, Signature)
  • Custom Upload: Upload any CSV from any domain with auto-schema mapping

Run locally:
  streamlit run app.py
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import json
import pathlib
import pandas as pd
import streamlit as st

from src.loader import load
from src.detector import detect
from src.metrics import compute
from src.classifier import (
    classify,
    PATTERN_DECISIVE,
    PATTERN_SINGLE_CORRECTOR,
    PATTERN_FLIP_FLOPPER,
    PATTERN_LATE_REVERSER,
    PATTERN_CHRONIC_INDECISIVE,
    PATTERN_MIXED,
)
from src.explain import explain_all
from src.visuals import timeline_chart, stage_chart, flow_chart
from src.config import (
    APP_TITLE,
    CSV_COLUMNS,
    STAGE_ORDER,
    LATE_STAGES,
    auto_detect_columns,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATASET_PRESETS = {
    "🚀 SaaS Funnel (200 users, balanced)": pathlib.Path("data/sample_events.csv"),
    "🌐 Real-World Online Retail (UCI / Databricks, 98 customers)": pathlib.Path("data/real_retail_events.csv"),
    "🏥 Healthcare & Telehealth (Triage & Slots, 50 patients)": pathlib.Path("data/healthcare_appointments.csv"),
    "🏨 Luxury Hotel & Travel (54 guests, Room & Dining)": pathlib.Path("data/hotel_travel_booking.csv"),
    "☁️ Cloud DevOps Infrastructure (50 dev specs, Instances)": pathlib.Path("data/cloud_infra_provisioning.csv"),
    "🎓 EdTech Masterclass Enrollment (50 students, Majors)": pathlib.Path("data/edtech_course_enrollment.csv"),
    "⚔️ RPG Character Loadout (54 gamers, Weapons & Perks)": pathlib.Path("data/gaming_character_loadout.csv"),
    "🛍️ E-Commerce Checkout (Orders & Shipping)": pathlib.Path("data/ecommerce_events.csv"),
    "🏦 Insurance & Loan Funnel (Dynamic Stages)": pathlib.Path("data/loan_application_events.csv"),
    "📂 Upload Custom CSV (Any Schema)": None,
}

# Emoji badge per pattern
PATTERN_BADGE: dict[str, str] = {
    PATTERN_DECISIVE:           "✅ Decisive",
    PATTERN_SINGLE_CORRECTOR:   "🔧 Single Corrector",
    PATTERN_FLIP_FLOPPER:       "🔀 Flip-Flopper",
    PATTERN_LATE_REVERSER:      "⏰ Late Reverser",
    PATTERN_CHRONIC_INDECISIVE: "🌀 Chronic Indecisive",
    PATTERN_MIXED:              "🔄 Mixed",
}

# Colour tag per pattern (for st.markdown coloured pills)
PATTERN_COLOUR: dict[str, str] = {
    PATTERN_DECISIVE:           "#2ecc71",
    PATTERN_SINGLE_CORRECTOR:   "#3498db",
    PATTERN_FLIP_FLOPPER:       "#e74c3c",
    PATTERN_LATE_REVERSER:      "#e67e22",
    PATTERN_CHRONIC_INDECISIVE: "#9b59b6",
    PATTERN_MIXED:              "#95a5a6",
}


# ─────────────────────────────────────────────────────────────────────────────
# Cached pipeline (re-runs only when raw bytes or config change)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Running CEREBRO analytical pipeline…")
def run_pipeline(
    csv_bytes: bytes,
    column_mapping_json: str | None = None,
    stage_order_json: str | None = None,
    late_stages_json: str | None = None,
    allow_custom_stages: bool = True,
) -> tuple[
    pd.DataFrame,   # events
    pd.DataFrame,   # reversals
    pd.DataFrame,   # classified metrics + explanation
]:
    """
    Load → detect → metrics → classify → explain.
    Adapts dynamically to any custom schema, custom stages, and late stages.
    """
    column_mapping = json.loads(column_mapping_json) if column_mapping_json else None
    stage_order = json.loads(stage_order_json) if stage_order_json else None
    late_stages = json.loads(late_stages_json) if late_stages_json else None

    events_df = load(
        io.BytesIO(csv_bytes),
        column_mapping=column_mapping,
        stage_order=stage_order,
        allow_custom_stages=allow_custom_stages,
        auto_map_columns=True,
    )
    reversals_df = detect(events_df)
    active_stages = events_df.attrs.get("stage_order")
    active_late_stages = late_stages or events_df.attrs.get("late_stages")

    metrics_df   = compute(events_df, reversals_df, stage_order=active_stages)
    classified   = classify(metrics_df, reversals_df, late_stages=active_late_stages)
    full_df      = explain_all(metrics_df, reversals_df, classified, late_stages=active_late_stages)

    # Attach stage metadata to full_df so tabs have full access
    full_df.attrs["stage_order"] = active_stages
    full_df.attrs["late_stages"] = active_late_stages
    reversals_df.attrs["stage_order"] = active_stages

    return events_df, reversals_df, full_df


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – data source & dynamic configuration
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[bytes | None, str | None, str | None, str | None]:
    """
    Render sidebar controls with dataset selection and auto-mapping UI.
    Returns (csv_bytes, mapping_json, stage_order_json, late_stages_json).
    """
    with st.sidebar:
        st.markdown(
            """
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;'>
                <span style='font-size:2.2rem;'>🧠</span>
                <div>
                    <h2 style='margin:0;font-size:1.4rem;'>CEREBRO</h2>
                    <p style='margin:0;font-size:0.75rem;color:#888;'>Universal Reversal Detector</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        preset_choice = st.selectbox(
            "Dataset Source",
            options=list(DATASET_PRESETS.keys()),
            index=0,
            help="Choose an industry preset or upload any CSV from your machine.",
        )

        csv_bytes: bytes | None = None

        if preset_choice == "📂 Upload Custom CSV (Any Schema)":
            uploaded = st.file_uploader(
                "Upload Event Log CSV",
                type=["csv"],
                help="Upload any CSV file. Columns will be auto-detected.",
            )
            if uploaded is not None:
                csv_bytes = uploaded.read()
            else:
                st.info("Upload any CSV file to begin analysis.")
                return None, None, None, None
        else:
            preset_path = DATASET_PRESETS[preset_choice]
            if preset_path and preset_path.exists():
                csv_bytes = preset_path.read_bytes()
                st.success(f"Loaded preset: `{preset_path.name}`")
            else:
                st.error(f"Preset file not found: `{preset_path}`. Run generation scripts first.")
                return None, None, None, None

        if not csv_bytes:
            return None, None, None, None

        # Peek at columns for smart auto-detection
        try:
            peek_df = pd.read_csv(io.BytesIO(csv_bytes), nrows=10)
            avail_cols = list(peek_df.columns)
        except Exception as e:
            st.error(f"Failed to inspect CSV headers: {e}")
            return None, None, None, None

        detected = auto_detect_columns(avail_cols)

        # ── Dynamic Schema Expander ───────────────────────────────────────
        with st.expander("⚙️ Schema & Funnel Mapping", expanded=(preset_choice == "📂 Upload Custom CSV (Any Schema)")):
            st.caption("CEREBRO auto-maps your columns. Adjust if needed:")

            def _get_default_idx(field: str, fallback_idx: int = 0) -> int:
                det = detected.get(field)
                if det and det in avail_cols:
                    return avail_cols.index(det)
                return min(fallback_idx, len(avail_cols) - 1)

            col_user = st.selectbox("User ID", avail_cols, index=_get_default_idx("user_id", 0))
            col_dec  = st.selectbox("Decision ID", avail_cols, index=_get_default_idx("decision_id", 1))
            col_ts   = st.selectbox("Timestamp", avail_cols, index=_get_default_idx("timestamp", 2))
            col_val  = st.selectbox("Choice / Option", avail_cols, index=_get_default_idx("choice", min(4, len(avail_cols)-1)))

            # Stage column can be optional
            stage_options = ["— Auto-generate single stage —"] + avail_cols
            default_stage_idx = 0
            if "stage" in detected and detected["stage"] in avail_cols:
                default_stage_idx = stage_options.index(detected["stage"])

            col_stg = st.selectbox("Stage (Optional)", stage_options, index=default_stage_idx)

            column_mapping = {
                "user_id": col_user,
                "decision_id": col_dec,
                "timestamp": col_ts,
                "choice": col_val,
            }
            if col_stg != "— Auto-generate single stage —":
                column_mapping["stage"] = col_stg

            # Peek unique stages from dataset if stage column chosen
            detected_stages: list[str] = []
            if col_stg != "— Auto-generate single stage —" and col_stg in peek_df.columns:
                try:
                    full_preview = pd.read_csv(io.BytesIO(csv_bytes), usecols=[col_stg])
                    detected_stages = list(dict.fromkeys(full_preview[col_stg].dropna().astype(str).tolist()))
                except Exception:
                    detected_stages = list(dict.fromkeys(peek_df[col_stg].dropna().astype(str).tolist()))

            late_stages = []
            if detected_stages:
                # Default late stages = last 40% of stages
                n_late = max(1, round(len(detected_stages) * 0.4))
                default_late = detected_stages[-n_late:]
                late_stages = st.multiselect(
                    "Late Stages (for cold-feet rule)",
                    options=detected_stages,
                    default=default_late,
                    help="Users making ≥60% of reversals at these stages are flagged as Late Reversers.",
                )

        mapping_json = json.dumps(column_mapping)
        stage_order_json = json.dumps(detected_stages) if detected_stages else None
        late_stages_json = json.dumps(late_stages) if late_stages else None

        st.caption(f"📊 Columns: {len(avail_cols)} | Decisions: {peek_df[col_dec].nunique() if col_dec in peek_df else '—'}")
        st.divider()

        return csv_bytes, mapping_json, stage_order_json, late_stages_json


# ─────────────────────────────────────────────────────────────────────────────
# Helper widgets
# ─────────────────────────────────────────────────────────────────────────────

def _pattern_pill(pattern: str) -> str:
    """Return an HTML coloured pill span for a pattern label."""
    label  = PATTERN_BADGE.get(pattern, pattern)
    colour = PATTERN_COLOUR.get(pattern, "#888")
    return (
        f'<span style="background:{colour};color:#fff;padding:3px 10px;'
        f'border-radius:12px;font-size:0.85rem;font-weight:600;">'
        f'{label}</span>'
    )


def _metric_card(label: str, value: str, delta: str | None = None) -> None:
    """Thin wrapper around st.metric for consistent styling."""
    st.metric(label=label, value=value, delta=delta)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 – Overview
# ─────────────────────────────────────────────────────────────────────────────

def render_overview(
    full_df: pd.DataFrame,
    reversals_df: pd.DataFrame,
) -> None:
    st.header("Overview — All Users")

    # ── Summary KPIs ─────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _metric_card("Total users", str(len(full_df)))
    with col2:
        n_rev = int(full_df["total_reversals"].sum())
        _metric_card("Total reversals", f"{n_rev:,}")
    with col3:
        n_fb = int(full_df["flip_back_count"].sum())
        _metric_card("Total flip-backs", f"{n_fb:,}")
    with col4:
        avg_rate = full_df["reversal_rate"].mean() if len(full_df) > 0 else 0.0
        _metric_card("Avg reversals / decision", f"{avg_rate:.2f}")

    st.divider()

    # ── User ranking table ────────────────────────────────────────────────
    st.subheader("User decision reversal ranking")

    display_cols = [
        "user_id", "pattern", "total_reversals", "flip_back_count",
        "decisions_with_reversals", "reversal_rate", "max_share_on_one_decision",
        "avg_hesitation_seconds", "most_common_reversal_stage",
    ]
    table = (
        full_df[display_cols]
        .sort_values("reversal_rate", ascending=False)
        .reset_index(drop=True)
    )

    table.insert(
        1,
        "Pattern",
        table["pattern"].map(lambda p: PATTERN_BADGE.get(p, p)),
    )
    table = table.drop(columns=["pattern"])
    table.columns = [
        "User ID", "Pattern", "Reversals", "Flip-backs",
        "Decisions changed", "Reversal rate", "Max decision share",
        "Avg hesitation (s)", "Most-reversed stage",
    ]

    st.dataframe(
        table.style
            .format({
                "Reversal rate":      "{:.3f}",
                "Max decision share": lambda v: f"{v:.1%}" if pd.notna(v) else "—",
                "Avg hesitation (s)": lambda v: f"{v:.1f}" if pd.notna(v) else "—",
            })
            .background_gradient(
                subset=["Reversal rate"],
                cmap="YlOrRd",
            ),
        use_container_width=True,
        height=min(600, 38 * len(table) + 40),
    )

    # ── Pattern distribution & Funnel stage chart ─────────────────────────
    st.subheader("Pattern distribution & Funnel analysis")
    pattern_counts = full_df["pattern"].value_counts().reset_index()
    pattern_counts.columns = ["pattern", "count"]
    pattern_counts["label"] = pattern_counts["pattern"].map(
        lambda p: PATTERN_BADGE.get(p, p)
    )

    col_left, col_right = st.columns([1, 2])
    with col_left:
        for _, row in pattern_counts.iterrows():
            pct = row["count"] / len(full_df) * 100 if len(full_df) > 0 else 0
            st.markdown(
                f'{_pattern_pill(row["pattern"])} &nbsp; '
                f'<span style="color:#ccc">{row["count"]} users ({pct:.0f}%)</span>',
                unsafe_allow_html=True,
            )
            st.progress(int(pct))

    with col_right:
        active_stages = full_df.attrs.get("stage_order")
        st.plotly_chart(
            stage_chart(reversals_df, stage_order=active_stages),
            use_container_width=True,
            key="stage_chart_overview",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 – User Deep-Dive
# ─────────────────────────────────────────────────────────────────────────────

def render_user_tab(
    events_df: pd.DataFrame,
    reversals_df: pd.DataFrame,
    full_df: pd.DataFrame,
) -> None:
    st.header("User Deep-Dive")

    all_users = sorted(full_df["user_id"].unique())
    if not all_users:
        st.warning("No users found.")
        return

    # User selector
    col_sel, _ = st.columns([2, 3])
    with col_sel:
        user_id: str = st.selectbox(
            "Select a user to inspect:",
            options=all_users,
            index=0,
            help="Pick any user to inspect their decision trajectory, reversals, and charts.",
        )

    user_row  = full_df[full_df["user_id"] == user_id].iloc[0]
    user_revs = (
        reversals_df[reversals_df["user_id"] == user_id]
        if not reversals_df.empty
        else reversals_df
    )
    user_evts = events_df[events_df["user_id"] == user_id]

    pattern = user_row["pattern"]

    # Pattern banner
    badge  = PATTERN_BADGE.get(pattern, pattern)
    colour = PATTERN_COLOUR.get(pattern, "#888")
    st.markdown(
        f"""
        <div style="background:{colour}22; border-left: 5px solid {colour};
                    padding: 12px 18px; border-radius: 6px; margin: 12px 0;">
            <span style="font-size:1.3rem; font-weight:700; color:{colour};">{badge}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(f"💬 {user_row['explanation']}")
    st.divider()

    # Key metrics row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        _metric_card("Total reversals",  str(int(user_row["total_reversals"])))
    with c2:
        _metric_card("Flip-backs",       str(int(user_row["flip_back_count"])))
    with c3:
        _metric_card("Reversal rate",    f"{user_row['reversal_rate']:.3f}")
    with c4:
        share = user_row.get("max_share_on_one_decision", 0.0)
        share_str = f"{share:.0%}" if user_row["total_reversals"] > 0 else "—"
        _metric_card("Max decision share", share_str)
    with c5:
        hes = user_row["avg_hesitation_seconds"]
        hes_str = f"{hes:.1f} s" if pd.notna(hes) else "—"
        _metric_card("Avg hesitation",   hes_str)
    with c6:
        _metric_card(
            "Most-reversed stage",
            str(user_row["most_common_reversal_stage"] or "—"),
        )

    st.divider()

    # Raw reversal table
    with st.expander(
        f"Reversal log ({len(user_revs)} reversal(s))", expanded=False
    ):
        if user_revs.empty:
            st.success("No reversals — this user made no changes.")
        else:
            display_revs = user_revs.copy()
            display_revs["seconds_since_previous"] = (
                display_revs["seconds_since_previous"].round(1)
            )
            st.dataframe(display_revs, use_container_width=True)

    st.divider()

    # Timeline chart
    st.subheader("Choice timeline")
    st.plotly_chart(
        timeline_chart(events_df, user_id),
        use_container_width=True,
        key=f"timeline_{user_id}",
    )

    # Per-user stage distribution
    if not user_revs.empty:
        st.subheader("Reversals by stage for this user")
        active_stages = full_df.attrs.get("stage_order")
        st.plotly_chart(
            stage_chart(user_revs, stage_order=active_stages),
            use_container_width=True,
            key=f"user_stage_{user_id}",
        )

    # Sankey flow chart
    st.subheader("Choice transition flow (Sankey)")
    user_decisions = sorted(user_evts["decision_id"].unique())

    if not user_decisions:
        st.info("No decision events found for this user.")
    else:
        selected_dec = st.selectbox(
            "Select decision flow to view:",
            options=user_decisions,
            index=0,
            key=f"sankey_dec_{user_id}",
        )
        st.plotly_chart(
            flow_chart(events_df, user_id, selected_dec),
            use_container_width=True,
            key=f"flow_{user_id}_{selected_dec}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown(
        f"""
        <h1 style='text-align:center;font-family:Inter,Arial,sans-serif;
                   font-size:2rem;margin-bottom:0'>
            🧠 {APP_TITLE}
        </h1>
        <p style='text-align:center;color:#888;margin-top:4px;font-size:0.9rem'>
            Universal detection of decision reversals, hesitation, and churn across any funnel or schema
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # Sidebar controls
    csv_bytes, mapping_json, stage_order_json, late_stages_json = render_sidebar()
    if csv_bytes is None:
        st.info("👈 Select a preset dataset or upload your own CSV in the sidebar.")
        return

    # Run the pipeline
    try:
        events_df, reversals_df, full_df = run_pipeline(
            csv_bytes,
            column_mapping_json=mapping_json,
            stage_order_json=stage_order_json,
            late_stages_json=late_stages_json,
            allow_custom_stages=True,
        )
    except FileNotFoundError as exc:
        st.error(f"File not found: {exc}")
        return
    except ValueError as exc:
        st.error(f"Schema / Data error: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected processing error: {exc}")
        st.exception(exc)
        return

    # Render primary views
    tab_overview, tab_user = st.tabs(["📊 Overview", "🔍 User Deep-Dive"])

    with tab_overview:
        render_overview(full_df, reversals_df)

    with tab_user:
        render_user_tab(events_df, reversals_df, full_df)


if __name__ == "__main__":
    main()
