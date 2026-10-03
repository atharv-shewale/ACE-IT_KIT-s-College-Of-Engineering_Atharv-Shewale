"""
tests/test_metrics.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for src/metrics.py.

Test cases
──────────
  1.  Zero-reversal user  – all counts zero, NaN hesitation, empty dicts
  2.  Single reversal     – totals, rate, hesitation, stage all correct
  3.  Flip-back count     – flip_back_count increments correctly
  4.  Reversal rate       – rate = reversals / distinct decisions
  5.  Stage distribution  – correct stage bucketing; most_common_reversal_stage
  6.  Avg hesitation      – mean of seconds_since_previous
  7.  Multi-user output   – output has one row per user, sorted by user_id
  8.  reversals_per_decision – keyed by decision_id with correct counts
  9.  decisions_with_reversals – counts only decisions that had ≥1 reversal
─────────────────────────────────────────────────────────────────────────────
"""

import math
import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.detector import detect
from src.metrics import compute

# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

T0 = datetime(2024, 6, 1, 10, 0, 0)


def _evt(user_id, decision_id, offset_s, stage, choice):
    """Build a single event row dict."""
    return {
        "user_id":     user_id,
        "decision_id": decision_id,
        "timestamp":   pd.Timestamp(T0 + timedelta(seconds=offset_s)),
        "stage":       stage,
        "choice":      choice,
    }


def _make_events(*rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _pipeline(rows) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run events → detect → compute and return all three DataFrames."""
    events_df = _make_events(*rows)
    reversals_df = detect(events_df)
    metrics_df = compute(events_df, reversals_df)
    return events_df, reversals_df, metrics_df


def _user_row(metrics_df: pd.DataFrame, user_id: str) -> pd.Series:
    """Extract the single metrics row for a given user."""
    rows = metrics_df[metrics_df["user_id"] == user_id]
    assert len(rows) == 1, f"Expected 1 row for {user_id}, got {len(rows)}"
    return rows.iloc[0]


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 – Zero-reversal user
# ─────────────────────────────────────────────────────────────────────────────

def test_zero_reversal_user():
    """A decisive user (no changes) must produce safe zero / empty / NaN values."""
    _, _, metrics_df = _pipeline([
        _evt("U001", "U001_plan",   0, "Browsing",  "basic"),
        _evt("U001", "U001_add_on", 60, "Selection", "none"),
    ])

    row = _user_row(metrics_df, "U001")

    assert row["total_reversals"]          == 0
    assert row["flip_back_count"]          == 0
    assert row["decisions_with_reversals"] == 0
    assert row["reversal_rate"]            == 0.0
    assert row["max_share_on_one_decision"] == 0.0
    assert row["reversals_per_decision"]   == {}
    assert row["stage_distribution"]       == {}
    assert row["most_common_reversal_stage"] is None
    # avg_hesitation_seconds must be NaN (not 0) to signal "no data"
    assert math.isnan(row["avg_hesitation_seconds"])


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 – Single reversal: totals, rate, hesitation, stage
# ─────────────────────────────────────────────────────────────────────────────

def test_single_reversal_metrics():
    """One reversal on one of two decisions."""
    _, _, metrics_df = _pipeline([
        _evt("U001", "U001_plan",   0,   "Browsing",  "basic"),
        _evt("U001", "U001_plan",   90,  "Selection", "premium"),  # reversal
        _evt("U001", "U001_add_on", 200, "Browsing",  "none"),     # no change
    ])

    row = _user_row(metrics_df, "U001")

    assert row["total_reversals"] == 1
    # rate = 1 reversal / 2 decisions
    assert row["reversal_rate"] == pytest.approx(0.5, abs=1e-4)
    assert row["avg_hesitation_seconds"] == pytest.approx(90.0)
    assert row["most_common_reversal_stage"] == "Selection"
    assert row["stage_distribution"]["Selection"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 – Flip-back count
# ─────────────────────────────────────────────────────────────────────────────

def test_flip_back_count():
    """A→B→A: flip_back_count must be 1."""
    _, _, metrics_df = _pipeline([
        _evt("U002", "U002_plan", 0,   "Browsing",  "basic"),
        _evt("U002", "U002_plan", 60,  "Selection", "premium"),
        _evt("U002", "U002_plan", 120, "Review",    "basic"),   # flip-back
    ])

    row = _user_row(metrics_df, "U002")

    assert row["total_reversals"] == 2
    assert row["flip_back_count"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 – Reversal rate with multiple decisions
# ─────────────────────────────────────────────────────────────────────────────

def test_reversal_rate():
    """3 decisions, 2 with reversals → rate = 2/3."""
    _, _, metrics_df = _pipeline([
        # plan: 1 reversal
        _evt("U003", "U003_plan",           0,   "Browsing",  "basic"),
        _evt("U003", "U003_plan",           90,  "Selection", "premium"),
        # add_on: 1 reversal
        _evt("U003", "U003_add_on",         10,  "Browsing",  "none"),
        _evt("U003", "U003_add_on",         100, "Selection", "vpn"),
        # payment_method: no change
        _evt("U003", "U003_payment_method", 20,  "Browsing",  "credit_card"),
    ])

    row = _user_row(metrics_df, "U003")

    assert row["total_reversals"]          == 2
    assert row["decisions_with_reversals"] == 2
    assert row["reversal_rate"] == pytest.approx(2 / 3, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 – Stage distribution and most_common_reversal_stage
# ─────────────────────────────────────────────────────────────────────────────

def test_stage_distribution():
    """Reversals at different stages are bucketed correctly."""
    _, _, metrics_df = _pipeline([
        _evt("U004", "U004_plan",   0,    "Browsing",     "basic"),
        _evt("U004", "U004_plan",   300,  "Confirmation", "premium"),   # reversal
        _evt("U004", "U004_add_on", 0,    "Browsing",     "none"),
        _evt("U004", "U004_add_on", 200,  "Post-commit",  "vpn"),       # reversal
        _evt("U004", "U004_add_on", 500,  "Post-commit",  "cloud_backup"), # reversal
    ])

    row = _user_row(metrics_df, "U004")

    dist = row["stage_distribution"]
    assert dist["Confirmation"] == 1
    assert dist["Post-commit"]  == 2
    assert dist.get("Browsing", 0) == 0

    # Post-commit has the most reversals (2)
    assert row["most_common_reversal_stage"] == "Post-commit"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 – Average hesitation (mean of seconds_since_previous)
# ─────────────────────────────────────────────────────────────────────────────

def test_avg_hesitation_seconds():
    """avg_hesitation_seconds must be the mean gap across all reversals."""
    # Two reversals with gaps of 100 s and 200 s → mean = 150 s
    _, _, metrics_df = _pipeline([
        _evt("U005", "U005_plan", 0,   "Browsing",  "basic"),
        _evt("U005", "U005_plan", 100, "Selection", "standard"),  # gap = 100
        _evt("U005", "U005_plan", 300, "Review",    "premium"),   # gap = 200
    ])

    row = _user_row(metrics_df, "U005")

    assert row["total_reversals"] == 2
    assert row["avg_hesitation_seconds"] == pytest.approx(150.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test 7 – Multi-user output has one row per user, sorted
# ─────────────────────────────────────────────────────────────────────────────

def test_multi_user_one_row_each_sorted():
    """Output DataFrame must have exactly one row per user, sorted by user_id."""
    _, _, metrics_df = _pipeline([
        _evt("U010", "U010_plan", 0,  "Browsing", "basic"),
        _evt("U005", "U005_plan", 10, "Browsing", "premium"),
        _evt("U001", "U001_plan", 20, "Browsing", "standard"),
    ])

    assert list(metrics_df["user_id"]) == ["U001", "U005", "U010"]
    assert len(metrics_df) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Test 8 – reversals_per_decision keyed correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_reversals_per_decision_dict():
    """reversals_per_decision must map each decision_id → reversal count."""
    _, _, metrics_df = _pipeline([
        # plan: 2 reversals
        _evt("U006", "U006_plan",   0,   "Browsing",  "basic"),
        _evt("U006", "U006_plan",   60,  "Selection", "premium"),
        _evt("U006", "U006_plan",   120, "Review",    "standard"),
        # add_on: 1 reversal
        _evt("U006", "U006_add_on", 10,  "Browsing",  "none"),
        _evt("U006", "U006_add_on", 200, "Selection", "vpn"),
    ])

    row = _user_row(metrics_df, "U006")
    rpd = row["reversals_per_decision"]

    assert rpd["U006_plan"]   == 2
    assert rpd["U006_add_on"] == 1
    assert "U006_payment_method" not in rpd  # no events for this decision


# ─────────────────────────────────────────────────────────────────────────────
# Test 9 – decisions_with_reversals counts only changed decisions
# ─────────────────────────────────────────────────────────────────────────────

def test_decisions_with_reversals_count():
    """
    decisions_with_reversals must count distinct decision_ids that had ≥1
    reversal, not all decisions the user touched.
    """
    _, _, metrics_df = _pipeline([
        # plan: 1 reversal
        _evt("U007", "U007_plan",           0,   "Browsing", "basic"),
        _evt("U007", "U007_plan",           90,  "Review",   "premium"),
        # add_on: stable
        _evt("U007", "U007_add_on",         5,   "Browsing", "none"),
        # payment_method: stable
        _evt("U007", "U007_payment_method", 10,  "Browsing", "credit_card"),
    ])

    row = _user_row(metrics_df, "U007")

    # Only plan changed → 1 decision with reversals out of 3
    assert row["decisions_with_reversals"] == 1
    assert row["total_reversals"]          == 1
    assert row["max_share_on_one_decision"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 10 – max_share_on_one_decision computed accurately
# ─────────────────────────────────────────────────────────────────────────────

def test_max_share_on_one_decision():
    """
    max_share_on_one_decision must be the largest single-decision share
    of total reversals.
    """
    # 3 on plan, 1 on add_on → total 4 reversals, max_share = 3/4 = 0.75
    _, _, metrics_df = _pipeline([
        _evt("U008", "U008_plan",   0,   "Browsing",  "basic"),
        _evt("U008", "U008_plan",   60,  "Selection", "premium"),
        _evt("U008", "U008_plan",   120, "Review",    "standard"),
        _evt("U008", "U008_plan",   180, "Review",    "basic"),
        _evt("U008", "U008_add_on", 10,  "Browsing",  "none"),
        _evt("U008", "U008_add_on", 90,  "Selection", "vpn"),
    ])

    row = _user_row(metrics_df, "U008")
    assert row["total_reversals"] == 4
    assert row["reversals_per_decision"]["U008_plan"] == 3
    assert row["reversals_per_decision"]["U008_add_on"] == 1
    assert row["max_share_on_one_decision"] == pytest.approx(0.75, abs=1e-4)
