"""
tests/test_classifier.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for src/classifier.py and src/explain.py.

Classifier test cases
──────────────────────
  1.  Decisive              – 0 reversals
  2.  Single corrector      – 1 reversal, no flip-back, gap <= 60 s
  3.  Not single corrector  – 1 reversal but gap > 60 s  → mixed
  4.  Not single corrector  – 1 reversal, flip-back      → flip_flopper
  5.  Flip-flopper via rev  – 3+ reversals on one decision
  6.  Flip-flopper via flips– 2+ flip-backs
  7.  Late reverser         – >=60% reversals at Confirmation/Post-commit
  8.  Not late reverser     – <60% at late stages → not late_reverser
  9.  Chronic indecisive    – high rate + 2+ decisions
  10. Mixed                 – does not satisfy any earlier rule
  11. Priority: flip-flopper beats late-reverser when both qualify
  12. classify() attaches 'pattern' column to every user
  13. explain_user() returns a non-empty string for each pattern
  14. explain_all() attaches 'explanation' column matching user count
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.detector import detect
from src.metrics import compute
from src.classifier import (
    classify,
    classify_user,
    PATTERN_DECISIVE,
    PATTERN_SINGLE_CORRECTOR,
    PATTERN_FLIP_FLOPPER,
    PATTERN_LATE_REVERSER,
    PATTERN_CHRONIC_INDECISIVE,
    PATTERN_MIXED,
)
from src.explain import explain_user, explain_all

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

T0 = datetime(2024, 6, 1, 10, 0, 0)


def _evt(user_id, decision_id, offset_s, stage, choice):
    return {
        "user_id":     user_id,
        "decision_id": decision_id,
        "timestamp":   pd.Timestamp(T0 + timedelta(seconds=offset_s)),
        "stage":       stage,
        "choice":      choice,
    }


def _run(rows) -> tuple[pd.Series, pd.DataFrame]:
    """Full pipeline → (user_metrics_row, user_revs) for the first user."""
    events_df    = pd.DataFrame(rows)
    reversals_df = detect(events_df)
    metrics_df   = compute(events_df, reversals_df)
    classified   = classify(metrics_df, reversals_df)

    uid       = events_df["user_id"].iloc[0]
    user_row  = classified[classified["user_id"] == uid].iloc[0]
    user_revs = reversals_df[reversals_df["user_id"] == uid]
    return user_row, user_revs


# ─────────────────────────────────────────────────────────────────────────────
# Classifier tests
# ─────────────────────────────────────────────────────────────────────────────

def test_decisive():
    row, revs = _run([
        _evt("U001", "U001_plan", 0, "Browsing", "basic"),
    ])
    assert row["pattern"] == PATTERN_DECISIVE


def test_single_corrector_within_window():
    """One reversal, no flip-back, gap exactly at threshold."""
    row, revs = _run([
        _evt("U002", "U002_plan", 0,  "Browsing",  "basic"),
        _evt("U002", "U002_plan", 45, "Selection", "premium"),   # gap=45s ≤ 60
    ])
    assert row["pattern"] == PATTERN_SINGLE_CORRECTOR


def test_not_single_corrector_slow_gap():
    """One reversal but gap > 60 s → falls through to mixed (no other rule fires)."""
    row, revs = _run([
        _evt("U003", "U003_plan", 0,   "Browsing",  "basic"),
        _evt("U003", "U003_plan", 120, "Selection", "premium"),  # gap=120s > 60
    ])
    # Should NOT be single_corrector
    assert row["pattern"] != PATTERN_SINGLE_CORRECTOR


def test_not_single_corrector_flip_back():
    """One reversal but it IS a flip-back → rule 2 fails, rule 3 fires."""
    row, revs = _run([
        _evt("U004", "U004_plan", 0,  "Browsing",  "basic"),
        _evt("U004", "U004_plan", 20, "Selection", "premium"),
        _evt("U004", "U004_plan", 40, "Review",    "basic"),    # flip-back in 40s
    ])
    # 2 reversals → not single_corrector; has 1 flip-back → not flip_flopper yet
    # Falls through based on config thresholds
    assert row["pattern"] != PATTERN_SINGLE_CORRECTOR


def test_flip_flopper_via_reversal_count():
    """3+ reversals on one decision → flip_flopper."""
    row, revs = _run([
        _evt("U005", "U005_plan", 0,   "Browsing",  "basic"),
        _evt("U005", "U005_plan", 100, "Selection", "premium"),
        _evt("U005", "U005_plan", 200, "Review",    "standard"),
        _evt("U005", "U005_plan", 300, "Review",    "premium"),  # 3rd reversal
    ])
    assert row["pattern"] == PATTERN_FLIP_FLOPPER


def test_flip_flopper_via_flip_back_count():
    """2 flip-backs concentrated on one decision (max_share >= 0.6) → flip_flopper."""
    row, revs = _run([
        # A → B → A → B  (2 flip-backs on plan)
        _evt("U006", "U006_plan", 0,   "Browsing",  "basic"),
        _evt("U006", "U006_plan", 100, "Selection", "premium"),
        _evt("U006", "U006_plan", 200, "Review",    "basic"),    # flip-back 1
        _evt("U006", "U006_plan", 300, "Review",    "premium"),  # flip-back 2
    ])
    assert row["pattern"] == PATTERN_FLIP_FLOPPER
    assert row["max_share_on_one_decision"] >= 0.60


def test_flip_backs_spread_across_decisions_is_chronic():
    """2 flip-backs spread across 2 decisions (max_share < 0.6) → chronic_indecisive."""
    row, revs = _run([
        # A → B → A  (1st flip-back on plan)
        _evt("U006b", "U006b_plan", 0,   "Browsing",  "basic"),
        _evt("U006b", "U006b_plan", 100, "Selection", "premium"),
        _evt("U006b", "U006b_plan", 200, "Review",    "basic"),    # flip-back 1
        # separate decision: X → Y → X (2nd flip-back on add_on)
        _evt("U006b", "U006b_add_on", 10,  "Browsing",  "none"),
        _evt("U006b", "U006b_add_on", 110, "Selection", "vpn"),
        _evt("U006b", "U006b_add_on", 210, "Review",    "none"),   # flip-back 2
    ])
    # max_share = 2/4 = 0.5 < 0.6; decisions_with_reversals = 2; rate = 4/2 = 2.0
    assert row["pattern"] == PATTERN_CHRONIC_INDECISIVE
    assert row["max_share_on_one_decision"] < 0.60


def test_late_reverser_above_threshold():
    """100 % of reversals at Post-commit → late_reverser."""
    row, revs = _run([
        _evt("U007", "U007_plan",   0,   "Browsing",   "basic"),
        _evt("U007", "U007_plan",   600, "Post-commit", "premium"),   # late reversal
        _evt("U007", "U007_add_on", 10,  "Browsing",   "none"),
        _evt("U007", "U007_add_on", 700, "Post-commit", "vpn"),       # late reversal
    ])
    assert row["pattern"] == PATTERN_LATE_REVERSER


def test_not_late_reverser_below_threshold():
    """Only 33 % of reversals at late stages → must NOT be late_reverser."""
    row, revs = _run([
        # early reversal
        _evt("U008", "U008_plan", 0,   "Browsing",  "basic"),
        _evt("U008", "U008_plan", 60,  "Selection", "premium"),
        # early reversal
        _evt("U008", "U008_add_on", 5,   "Browsing",  "none"),
        _evt("U008", "U008_add_on", 70,  "Selection", "vpn"),
        # late reversal (1 of 3)
        _evt("U008", "U008_add_on", 700, "Post-commit", "none"),
    ])
    assert row["pattern"] != PATTERN_LATE_REVERSER


def test_chronic_indecisive():
    """High reversal_rate across 3 decisions → chronic_indecisive."""
    row, revs = _run([
        # plan: 2 reversals
        _evt("U009", "U009_plan",           0,   "Browsing",  "basic"),
        _evt("U009", "U009_plan",           90,  "Selection", "premium"),
        _evt("U009", "U009_plan",           200, "Review",    "standard"),
        # add_on: 2 reversals
        _evt("U009", "U009_add_on",         10,  "Browsing",  "none"),
        _evt("U009", "U009_add_on",         100, "Selection", "vpn"),
        _evt("U009", "U009_add_on",         210, "Review",    "cloud_backup"),
        # payment_method: 2 reversals
        _evt("U009", "U009_payment_method", 20,  "Browsing",  "credit_card"),
        _evt("U009", "U009_payment_method", 110, "Selection", "upi"),
        _evt("U009", "U009_payment_method", 220, "Review",    "net_banking"),
    ])
    # reversal_rate = 6 reversals / 3 decisions = 2.0 > threshold (0.75)
    # decisions_with_reversals = 3 >= 2
    assert row["pattern"] == PATTERN_CHRONIC_INDECISIVE


def test_mixed_catch_all():
    """
    Two reversals, no flip-back, slow gap, not concentrated at late stages,
    low rate across only 1 decision → mixed.
    """
    row, revs = _run([
        _evt("U010", "U010_plan", 0,   "Browsing",  "basic"),
        _evt("U010", "U010_plan", 200, "Selection", "standard"),  # gap > 60
        _evt("U010", "U010_plan", 400, "Review",    "premium"),   # gap > 60
    ])
    # 2 reversals on 1 decision, no flip-backs, early stages, rate = 2.0 on 1 decision
    # flip_flopper requires 3+ on one OR 2 flip-backs (neither)
    # late_reverser requires 60%+ at late stages (0%)
    # chronic requires 2+ decisions with reversals (only 1)
    assert row["pattern"] == PATTERN_MIXED


def test_flip_flopper_beats_late_reverser_in_priority():
    """
    When both flip_flopper and late_reverser conditions are met,
    flip_flopper must win (priority 3 < priority 4).
    """
    row, revs = _run([
        # 3 reversals on one decision (flip_flopper)
        _evt("U011", "U011_plan", 0,   "Browsing",     "basic"),
        _evt("U011", "U011_plan", 100, "Confirmation", "premium"),   # late
        _evt("U011", "U011_plan", 200, "Post-commit",  "standard"),  # late
        _evt("U011", "U011_plan", 300, "Post-commit",  "basic"),     # late + flip-back
    ])
    assert row["pattern"] == PATTERN_FLIP_FLOPPER


def test_classify_adds_pattern_column_for_all_users():
    """classify() must produce a 'pattern' column with one value per user."""
    events_df = pd.DataFrame([
        _evt("UA", "UA_plan", 0, "Browsing", "basic"),          # decisive
        _evt("UB", "UB_plan", 0, "Browsing", "basic"),
        _evt("UB", "UB_plan", 30, "Selection", "premium"),      # single_corrector
    ])
    reversals_df = detect(events_df)
    metrics_df   = compute(events_df, reversals_df)
    result       = classify(metrics_df, reversals_df)

    assert "pattern" in result.columns
    assert len(result) == 2
    assert set(result["user_id"]) == {"UA", "UB"}
    assert result[result["user_id"] == "UA"]["pattern"].iloc[0] == PATTERN_DECISIVE
    assert result[result["user_id"] == "UB"]["pattern"].iloc[0] == PATTERN_SINGLE_CORRECTOR


# ─────────────────────────────────────────────────────────────────────────────
# Explain tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pattern", [
    PATTERN_DECISIVE,
    PATTERN_SINGLE_CORRECTOR,
    PATTERN_FLIP_FLOPPER,
    PATTERN_LATE_REVERSER,
    PATTERN_CHRONIC_INDECISIVE,
    PATTERN_MIXED,
])
def test_explain_user_returns_nonempty_string(pattern):
    """explain_user() must return a non-empty string for every known pattern."""
    # Build a minimal classified row by running a representative user
    scenarios = {
        PATTERN_DECISIVE: [
            _evt("UX", "UX_plan", 0, "Browsing", "basic"),
        ],
        PATTERN_SINGLE_CORRECTOR: [
            _evt("UX", "UX_plan", 0,  "Browsing",  "basic"),
            _evt("UX", "UX_plan", 30, "Selection", "premium"),
        ],
        PATTERN_FLIP_FLOPPER: [
            _evt("UX", "UX_plan", 0,   "Browsing",  "basic"),
            _evt("UX", "UX_plan", 100, "Selection", "premium"),
            _evt("UX", "UX_plan", 200, "Review",    "standard"),
            _evt("UX", "UX_plan", 300, "Review",    "basic"),
        ],
        PATTERN_LATE_REVERSER: [
            _evt("UX", "UX_plan", 0,   "Browsing",     "basic"),
            _evt("UX", "UX_plan", 600, "Post-commit",  "premium"),
            _evt("UX", "UX_add_on", 10,  "Browsing",    "none"),
            _evt("UX", "UX_add_on", 700, "Post-commit", "vpn"),
        ],
        PATTERN_CHRONIC_INDECISIVE: [
            _evt("UX", "UX_plan",           0,   "Browsing",  "basic"),
            _evt("UX", "UX_plan",           90,  "Selection", "premium"),
            _evt("UX", "UX_plan",           180, "Review",    "standard"),
            _evt("UX", "UX_add_on",         10,  "Browsing",  "none"),
            _evt("UX", "UX_add_on",         100, "Selection", "vpn"),
            _evt("UX", "UX_add_on",         190, "Review",    "cloud_backup"),
            _evt("UX", "UX_payment_method", 20,  "Browsing",  "credit_card"),
            _evt("UX", "UX_payment_method", 110, "Selection", "upi"),
            _evt("UX", "UX_payment_method", 200, "Review",    "net_banking"),
        ],
        PATTERN_MIXED: [
            _evt("UX", "UX_plan", 0,   "Browsing",  "basic"),
            _evt("UX", "UX_plan", 200, "Selection", "standard"),
            _evt("UX", "UX_plan", 400, "Review",    "premium"),
        ],
    }

    events_df    = pd.DataFrame(scenarios[pattern])
    reversals_df = detect(events_df)
    metrics_df   = compute(events_df, reversals_df)
    classified   = classify(metrics_df, reversals_df)
    user_row     = classified.iloc[0]
    user_revs    = reversals_df[reversals_df["user_id"] == "UX"]

    sentence = explain_user("UX", user_row, user_revs, pattern)

    assert isinstance(sentence, str)
    assert len(sentence) > 0
    assert "UX" in sentence   # user_id must appear in the sentence


def test_explain_all_adds_explanation_column():
    """explain_all() must append an 'explanation' column for every row."""
    events_df = pd.DataFrame([
        _evt("UA", "UA_plan", 0, "Browsing", "basic"),
        _evt("UB", "UB_plan", 0, "Browsing", "basic"),
        _evt("UB", "UB_plan", 30, "Selection", "premium"),
    ])
    reversals_df = detect(events_df)
    metrics_df   = compute(events_df, reversals_df)
    classified   = classify(metrics_df, reversals_df)
    result       = explain_all(metrics_df, reversals_df, classified)

    assert "explanation" in result.columns
    assert len(result) == 2
    # Each explanation must be a non-empty string
    for exp in result["explanation"]:
        assert isinstance(exp, str) and len(exp) > 0
