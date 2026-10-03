"""
src/metrics.py
─────────────────────────────────────────────────────────────────────────────
Per-user metric aggregation layer.

Takes the raw events DataFrame (from loader) and the reversals DataFrame
(from detector) and returns one summary row per user. Dynamically adapts
to any dataset's stage definitions and decision structures.

Computed columns
────────────────
  user_id                  – user identifier
  total_reversals          – total reversal events for this user
  flip_back_count          – reversals where is_flip_back == True
  decisions_with_reversals – number of distinct decision_ids that had ≥1 reversal
  reversal_rate            – total_reversals / number of distinct decisions made
  max_share_on_one_decision – largest single-decision share of user's reversals (0.0 if no reversals)
  avg_hesitation_seconds   – mean seconds_since_previous across all reversals
                             (NaN when user has no reversals)
  most_common_reversal_stage – stage name with the highest reversal count
                               (None / empty string when no reversals)
  stage_distribution       – dict {stage: count} for reversal counts per stage
                             (empty dict when no reversals)
  reversals_per_decision   – dict {decision_id: count} (empty dict when none)

Public API
──────────
  compute(events_df, reversals_df, stage_order=None) -> pd.DataFrame
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from src.config import DECISION_TYPES, STAGE_ORDER


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute(
    events_df: pd.DataFrame,
    reversals_df: pd.DataFrame,
    stage_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-user metrics from the event log and reversal table.

    Parameters
    ----------
    events_df   : pd.DataFrame  – clean output of loader.load()
    reversals_df: pd.DataFrame  – output of detector.detect()
    stage_order : list[str] | None – custom stages in order (auto-detected if None)

    Returns
    -------
    pd.DataFrame
        One row per user found in events_df, sorted by user_id.
        Users with zero reversals are included with safe zero / empty values.
    """
    # Auto-detect or inherit stage order
    if stage_order is None:
        stage_order = events_df.attrs.get("stage_order") or reversals_df.attrs.get("stage_order")
        if stage_order is None:
            unique_stages = list(dict.fromkeys(events_df["stage"].dropna().tolist())) if not events_df.empty else []
            canon_subset = [s for s in STAGE_ORDER if s in unique_stages]
            other_stages = [s for s in unique_stages if s not in STAGE_ORDER]
            stage_order = (canon_subset + other_stages) if (canon_subset or other_stages) else STAGE_ORDER

    # All users that appear in the event log (the ground truth for user list)
    all_users: list[str] = sorted(events_df["user_id"].unique()) if not events_df.empty else []

    rows: list[dict] = []
    for user_id in all_users:
        # Events for this user (to count distinct decisions)
        user_events = events_df[events_df["user_id"] == user_id]

        # Reversals for this user (may be empty)
        user_revs = (
            reversals_df[reversals_df["user_id"] == user_id]
            if not reversals_df.empty
            else reversals_df
        )

        rows.append(_user_metrics(user_id, user_events, user_revs, stage_order))

    result_df = pd.DataFrame(rows)
    result_df.attrs["stage_order"] = stage_order
    if "late_stages" in events_df.attrs:
        result_df.attrs["late_stages"] = events_df.attrs["late_stages"]
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Per-user metric calculation
# ─────────────────────────────────────────────────────────────────────────────

def _user_metrics(
    user_id: str,
    user_events: pd.DataFrame,
    user_revs: pd.DataFrame,
    stage_order: list[str],
) -> dict:
    """Build the metric dict for a single user."""

    # ── How many distinct decisions did this user make? ────────────────────
    n_decisions = user_events["decision_id"].nunique()

    # ── Basic reversal counts ──────────────────────────────────────────────
    total_reversals = len(user_revs)
    flip_back_count = int(user_revs["is_flip_back"].sum()) if total_reversals > 0 else 0

    # ── Reversal rate ──────────────────────────────────────────────────────
    reversal_rate = (total_reversals / n_decisions) if n_decisions > 0 else 0.0

    # ── Average hesitation ─────────────────────────────────────────────────
    if total_reversals > 0:
        avg_hesitation_seconds = float(user_revs["seconds_since_previous"].mean())
    else:
        avg_hesitation_seconds = float("nan")

    # ── Stage distribution ─────────────────────────────────────────────────
    if total_reversals > 0:
        stage_counts_series = user_revs["stage"].value_counts()
        stage_distribution = {
            stage: int(stage_counts_series.get(stage, 0))
            for stage in stage_order
        }
        most_common_reversal_stage = stage_counts_series.idxmax()
    else:
        stage_distribution = {}
        most_common_reversal_stage = None

    # ── Per-decision reversal counts ───────────────────────────────────────
    if total_reversals > 0:
        rev_per_dec_series = user_revs.groupby("decision_id").size()
        reversals_per_decision = rev_per_dec_series.to_dict()
        decisions_with_reversals = len(reversals_per_decision)
        max_on_one = max(reversals_per_decision.values())
        max_share_on_one_decision = round(max_on_one / total_reversals, 4)
    else:
        reversals_per_decision = {}
        decisions_with_reversals = 0
        max_share_on_one_decision = 0.0

    return {
        "user_id":                   user_id,
        "total_reversals":           total_reversals,
        "flip_back_count":           flip_back_count,
        "decisions_with_reversals":  decisions_with_reversals,
        "reversal_rate":             round(reversal_rate, 4),
        "max_share_on_one_decision": max_share_on_one_decision,
        "avg_hesitation_seconds":    avg_hesitation_seconds,
        "most_common_reversal_stage": most_common_reversal_stage,
        "stage_distribution":        stage_distribution,
        "reversals_per_decision":    reversals_per_decision,
    }
