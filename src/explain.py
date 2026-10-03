"""
src/explain.py
─────────────────────────────────────────────────────────────────────────────
Human-readable evidence sentence generator.

For each user, `explain_user()` assembles one plain English sentence from
their metrics and classified pattern — using string templates only, zero LLMs.
Dynamically handles arbitrary decision names and funnel stages.

Example outputs
───────────────
  decisive:
    "U007 made no changes to any decision."

  single_corrector:
    "U023 made a quick correction to 'plan' within 30 s and did not look back."

  flip_flopper:
    "U045 changed 'plan' 4 times, 2 of them back to a previous choice,
     mostly at the Review stage."

  late_reverser:
    "U061 reversed 3 decision(s) late in the funnel: 2 at Confirmation and
     1 at Post-commit."

  chronic_indecisive:
    "U082 changed decisions across 3 area(s) with an overall reversal rate
     of 1.33 changes per decision."

  mixed:
    "U099 made 2 reversal(s) across 1 decision(s) with 0 flip-back(s)."

Public API
──────────
  explain_user(user_id, user_metrics_row, user_revs, pattern, late_stages=None) -> str
  explain_all(metrics_df, reversals_df, classified_df, late_stages=None) -> pd.DataFrame
      Returns classified_df with a new 'explanation' column.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pandas as pd

from src.classifier import (
    PATTERN_DECISIVE,
    PATTERN_SINGLE_CORRECTOR,
    PATTERN_FLIP_FLOPPER,
    PATTERN_LATE_REVERSER,
    PATTERN_CHRONIC_INDECISIVE,
    PATTERN_MIXED,
)
from src.config import LATE_STAGES


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def explain_user(
    user_id: str,
    user_metrics_row: pd.Series,
    user_revs: pd.DataFrame,
    pattern: str,
    late_stages: list[str] | None = None,
) -> str:
    """
    Build a one-sentence evidence string for a single user.

    Parameters
    ----------
    user_id          : str
    user_metrics_row : pd.Series  – one row from classify() output
    user_revs        : pd.DataFrame – reversal rows for this user only
    pattern          : str         – classified pattern label
    late_stages      : list[str] | None – custom late stages

    Returns
    -------
    str – plain English sentence, no newlines.
    """
    if late_stages is None:
        late_stages = LATE_STAGES

    builders = {
        PATTERN_DECISIVE:           _explain_decisive,
        PATTERN_SINGLE_CORRECTOR:   _explain_single_corrector,
        PATTERN_FLIP_FLOPPER:       _explain_flip_flopper,
        PATTERN_LATE_REVERSER:      lambda uid, r, revs: _explain_late_reverser(uid, r, revs, late_stages=late_stages),
        PATTERN_CHRONIC_INDECISIVE: _explain_chronic_indecisive,
        PATTERN_MIXED:              _explain_mixed,
    }
    builder = builders.get(pattern, _explain_mixed)
    return builder(user_id, user_metrics_row, user_revs)


def explain_all(
    metrics_df: pd.DataFrame,
    reversals_df: pd.DataFrame,
    classified_df: pd.DataFrame,
    late_stages: list[str] | None = None,
) -> pd.DataFrame:
    """
    Append an 'explanation' column to classified_df.

    Parameters
    ----------
    metrics_df    : output of metrics.compute()
    reversals_df  : output of detector.detect()
    classified_df : output of classifier.classify() (has 'pattern' column)
    late_stages   : list[str] | None – custom late stages (inferred from attrs if None)

    Returns
    -------
    pd.DataFrame – classified_df with new 'explanation' column (str).
    """
    result = classified_df.copy()
    if late_stages is None:
        late_stages = classified_df.attrs.get("late_stages") or LATE_STAGES

    explanations: list[str] = []

    for _, row in result.iterrows():
        uid = row["user_id"]
        user_revs = (
            reversals_df[reversals_df["user_id"] == uid]
            if not reversals_df.empty
            else reversals_df
        )
        explanations.append(
            explain_user(uid, row, user_revs, row["pattern"], late_stages=late_stages)
        )

    result["explanation"] = explanations
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-pattern sentence builders  (each returns a str)
# ─────────────────────────────────────────────────────────────────────────────

def _explain_decisive(
    user_id: str,
    row: pd.Series,
    user_revs: pd.DataFrame,
) -> str:
    return f"{user_id} made no changes to any decision."


def _explain_single_corrector(
    user_id: str,
    row: pd.Series,
    user_revs: pd.DataFrame,
) -> str:
    rev       = user_revs.iloc[0]
    dec_label = _decision_label(rev["decision_id"])
    gap       = int(rev["seconds_since_previous"])
    return (
        f"{user_id} made a quick correction to '{dec_label}' within {gap} s "
        f"(changed from '{rev['from_choice']}' to '{rev['to_choice']}') "
        f"and did not look back."
    )


def _explain_flip_flopper(
    user_id: str,
    row: pd.Series,
    user_revs: pd.DataFrame,
) -> str:
    per_dec: dict[str, int] = row["reversals_per_decision"]
    most_churned_id    = max(per_dec, key=per_dec.get) if per_dec else "choice"
    most_churned_count = per_dec[most_churned_id] if per_dec else int(row["total_reversals"])
    dec_label          = _decision_label(str(most_churned_id))

    flip_backs = int(row["flip_back_count"])
    top_stage = row["most_common_reversal_stage"] or "various stages"

    return (
        f"{user_id} changed '{dec_label}' {most_churned_count} time(s), "
        f"{flip_backs} of them back to a previous choice, "
        f"mostly at the {top_stage} stage."
    )


def _explain_late_reverser(
    user_id: str,
    row: pd.Series,
    user_revs: pd.DataFrame,
    late_stages: list[str] | None = None,
) -> str:
    total = int(row["total_reversals"])
    if late_stages is None:
        late_stages = LATE_STAGES

    late_parts: list[str] = []
    for stage in late_stages:
        count = int((user_revs["stage"] == stage).sum())
        if count > 0:
            late_parts.append(f"{count} at {stage}")

    # Fallback to whatever stages appeared if none in late_stages
    if not late_parts and not user_revs.empty:
        for stage, count in user_revs["stage"].value_counts().items():
            late_parts.append(f"{count} at {stage}")

    late_summary = " and ".join(late_parts) if late_parts else "late stages"
    n_decisions  = int(row["decisions_with_reversals"])

    return (
        f"{user_id} reversed {n_decisions} decision(s) late in the funnel: "
        f"{late_summary} (out of {total} reversal(s) total)."
    )


def _explain_chronic_indecisive(
    user_id: str,
    row: pd.Series,
    user_revs: pd.DataFrame,
) -> str:
    n_decisions = int(row["decisions_with_reversals"])
    rate        = float(row["reversal_rate"])
    total       = int(row["total_reversals"])

    return (
        f"{user_id} changed decisions across {n_decisions} area(s) "
        f"with an overall reversal rate of {rate:.2f} changes per decision "
        f"({total} reversal(s) in total)."
    )


def _explain_mixed(
    user_id: str,
    row: pd.Series,
    user_revs: pd.DataFrame,
) -> str:
    total      = int(row["total_reversals"])
    n_dec      = int(row["decisions_with_reversals"])
    flip_backs = int(row["flip_back_count"])

    return (
        f"{user_id} made {total} reversal(s) across {n_dec} decision(s) "
        f"with {flip_backs} flip-back(s)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _decision_label(decision_id: str) -> str:
    """
    Extract a human-readable decision type from a decision_id.
    e.g. 'U001_plan' -> 'plan', 'U042_payment_method' -> 'payment_method'
    Falls back to the full ID if no underscore is found.
    """
    parts = str(decision_id).split("_", 1)
    if len(parts) == 2 and parts[0].startswith("U") and parts[0][1:].isdigit():
        return str(decision_id).rsplit("_", 1)[-1]
    return str(decision_id)
