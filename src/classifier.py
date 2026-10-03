"""
src/classifier.py
─────────────────────────────────────────────────────────────────────────────
Rule-based persona classifier.

Rules are evaluated in strict priority order (first match wins):

  Priority  Label                Condition
  ────────  ───────────────────  ─────────────────────────────────────────────
  1         decisive             total_reversals == 0
  2         single_corrector     total_reversals == 1
                                  AND is_flip_back == False
                                  AND seconds_since_previous <= CORRECTION_MAX_SECONDS
  3         chronic_indecisive   reversal_rate >= CHRONIC_INDECISION_RATE_THRESHOLD
                                  AND decisions_with_reversals >= CHRONIC_INDECISION_MIN_DECISIONS
                                  AND max_share_on_one_decision < CHRONIC_INDECISION_MAX_SHARE_ON_ONE_DECISION
                                  AND not late_reverser
                                  (broad churn across multiple decisions)
  4         flip_flopper         (reversals on ONE decision >= FLIPFLOPPER_MIN_REVERSALS_ON_ONE_DECISION
                                  OR flip_back_count >= FLIPFLOPPER_MIN_FLIP_BACKS)
                                  AND max_share_on_one_decision >= FLIPFLOPPER_MIN_SHARE_ON_ONE_DECISION
                                  (concentrated churn on a single decision)
  5         late_reverser        fraction of reversals at late stages >= LATE_REVERSAL_STAGE_RATIO
                                  (requires at least 1 reversal)
  6         mixed                catch-all

All thresholds are read from src/config.py, with support for dynamically configured
late stages across diverse dataset domains.

Public API
──────────
  classify(metrics_df, reversals_df, late_stages=None) -> pd.DataFrame
      Returns metrics_df with a new "pattern" column appended.

  classify_user(user_row, user_revs, late_stages=None) -> str
      Classify a single user; exposed for testing and explain.py.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pandas as pd

from src.config import (
    CORRECTION_MAX_SECONDS,
    FLIPFLOPPER_MIN_REVERSALS_ON_ONE_DECISION,
    FLIPFLOPPER_MIN_FLIP_BACKS,
    FLIPFLOPPER_MIN_SHARE_ON_ONE_DECISION,
    LATE_REVERSAL_STAGE_RATIO,
    LATE_STAGES,
    CHRONIC_INDECISION_RATE_THRESHOLD,
    CHRONIC_INDECISION_MIN_DECISIONS,
    CHRONIC_INDECISION_MAX_SHARE_ON_ONE_DECISION,
)

# Canonical pattern label strings (used as constants to avoid typos)
PATTERN_DECISIVE            = "decisive"
PATTERN_SINGLE_CORRECTOR    = "single_corrector"
PATTERN_FLIP_FLOPPER        = "flip_flopper"
PATTERN_LATE_REVERSER       = "late_reverser"
PATTERN_CHRONIC_INDECISIVE  = "chronic_indecisive"
PATTERN_MIXED               = "mixed"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def classify(
    metrics_df: pd.DataFrame,
    reversals_df: pd.DataFrame,
    late_stages: list[str] | None = None,
) -> pd.DataFrame:
    """
    Append a 'pattern' column to metrics_df by classifying each user.

    Parameters
    ----------
    metrics_df   : pd.DataFrame – output of metrics.compute()
    reversals_df : pd.DataFrame – output of detector.detect()
    late_stages  : list[str] | None – stages considered late in the funnel

    Returns
    -------
    pd.DataFrame – metrics_df with a new 'pattern' column (str).
    """
    result = metrics_df.copy()

    # Resolve late stages from parameter, DataFrame attrs, or config default
    if late_stages is None:
        late_stages = metrics_df.attrs.get("late_stages") or reversals_df.attrs.get("late_stages")
        if late_stages is None:
            stage_order = metrics_df.attrs.get("stage_order") or reversals_df.attrs.get("stage_order")
            if stage_order and len(stage_order) >= 2:
                n_late = max(1, round(len(stage_order) * 0.4))
                late_stages = stage_order[-n_late:]
            else:
                late_stages = LATE_STAGES

    patterns: list[str] = []
    for _, user_row in result.iterrows():
        uid = user_row["user_id"]
        # Slice the reversal rows that belong to this user
        user_revs = (
            reversals_df[reversals_df["user_id"] == uid]
            if not reversals_df.empty
            else reversals_df
        )
        patterns.append(classify_user(user_row, user_revs, late_stages=late_stages))

    result["pattern"] = patterns
    result.attrs["late_stages"] = late_stages
    if "stage_order" in metrics_df.attrs:
        result.attrs["stage_order"] = metrics_df.attrs["stage_order"]
    return result


def classify_user(
    user_row: pd.Series,
    user_revs: pd.DataFrame,
    late_stages: list[str] | None = None,
) -> str:
    """
    Classify a single user using the priority-ordered rule chain.

    Parameters
    ----------
    user_row    : pd.Series     – one row from metrics.compute() output
    user_revs   : pd.DataFrame  – reversal rows for this user only
    late_stages : list[str] | None – late funnel stages

    Returns
    -------
    str – one of the six PATTERN_* constants
    """
    if late_stages is None:
        late_stages = LATE_STAGES

    # ── Rule 1: Decisive ──────────────────────────────────────────────────
    if user_row["total_reversals"] == 0:
        return PATTERN_DECISIVE

    # ── Rule 2: Single corrector ──────────────────────────────────────────
    if _is_single_corrector(user_row, user_revs):
        return PATTERN_SINGLE_CORRECTOR

    # ── Rule 3: Chronic indecisive (broad churn across multiple decisions) ─
    if _is_chronic_indecisive(user_row, user_revs, late_stages=late_stages):
        return PATTERN_CHRONIC_INDECISIVE

    # ── Rule 4: Flip-flopper (concentrated churn on a single decision) ──────
    if _is_flip_flopper(user_row, user_revs):
        return PATTERN_FLIP_FLOPPER

    # ── Rule 5: Late reverser ─────────────────────────────────────────────
    if _is_late_reverser(user_row, user_revs, late_stages=late_stages):
        return PATTERN_LATE_REVERSER

    # ── Rule 6: Catch-all ─────────────────────────────────────────────────
    return PATTERN_MIXED


# ─────────────────────────────────────────────────────────────────────────────
# Individual rule predicates  (each returns True / False)
# ─────────────────────────────────────────────────────────────────────────────

def _is_single_corrector(user_row: pd.Series, user_revs: pd.DataFrame) -> bool:
    """
    Rule 2: exactly one reversal, no flip-back, gap ≤ CORRECTION_MAX_SECONDS.
    """
    if user_row["total_reversals"] != 1:
        return False

    rev = user_revs.iloc[0]   # the one and only reversal row

    no_flip_back  = not bool(rev["is_flip_back"])
    within_window = float(rev["seconds_since_previous"]) <= CORRECTION_MAX_SECONDS

    return no_flip_back and within_window


def _is_chronic_indecisive(
    user_row: pd.Series,
    user_revs: pd.DataFrame,
    late_stages: list[str] | None = None,
) -> bool:
    """
    Rule 3 (broad): reversal_rate >= CHRONIC_INDECISION_RATE_THRESHOLD
                   AND decisions_with_reversals >= CHRONIC_INDECISION_MIN_DECISIONS
                   AND max_share_on_one_decision < CHRONIC_INDECISION_MAX_SHARE_ON_ONE_DECISION
                   AND not late_reverser.
    """
    high_rate = float(user_row["reversal_rate"]) >= CHRONIC_INDECISION_RATE_THRESHOLD
    spread    = int(user_row["decisions_with_reversals"]) >= CHRONIC_INDECISION_MIN_DECISIONS
    max_share = float(user_row.get("max_share_on_one_decision", 0.0))
    broad     = max_share < CHRONIC_INDECISION_MAX_SHARE_ON_ONE_DECISION
    not_late  = not _is_late_reverser(user_row, user_revs, late_stages=late_stages)
    return high_rate and spread and broad and not_late


def _is_flip_flopper(user_row: pd.Series, user_revs: pd.DataFrame) -> bool:
    """
    Rule 4 (concentrated):
      (>= FLIPFLOPPER_MIN_REVERSALS_ON_ONE_DECISION on a single decision
       OR >= FLIPFLOPPER_MIN_FLIP_BACKS total flip-backs)
      AND max_share_on_one_decision >= FLIPFLOPPER_MIN_SHARE_ON_ONE_DECISION.
    """
    max_share = float(user_row.get("max_share_on_one_decision", 0.0))
    if max_share < FLIPFLOPPER_MIN_SHARE_ON_ONE_DECISION:
        return False

    # Check flip-back count
    if user_row["flip_back_count"] >= FLIPFLOPPER_MIN_FLIP_BACKS:
        return True

    # Check per-decision reversal counts
    per_decision: dict[str, int] = user_row.get("reversals_per_decision", {})
    if per_decision:
        max_on_one = max(per_decision.values())
        if max_on_one >= FLIPFLOPPER_MIN_REVERSALS_ON_ONE_DECISION:
            return True

    return False


def _is_late_reverser(
    user_row: pd.Series,
    user_revs: pd.DataFrame,
    late_stages: list[str] | None = None,
) -> bool:
    """
    Rule 5: at least LATE_REVERSAL_STAGE_RATIO of reversals occur at
            one of the late_stages.
    """
    if late_stages is None:
        late_stages = LATE_STAGES

    total = user_row["total_reversals"]
    if total == 0:
        return False

    late_count = int(user_revs["stage"].isin(late_stages).sum())
    return (late_count / total) >= LATE_REVERSAL_STAGE_RATIO
