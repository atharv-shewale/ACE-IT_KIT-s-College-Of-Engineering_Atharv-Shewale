"""
tests/test_detector.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for src/loader.py and src/detector.py.

Hand-crafted test cases
────────────────────────
  1.  No change          – single event, zero reversals
  2.  Single change      – A→B, one reversal, is_flip_back=False
  3.  Flip-back          – A→B→A, second reversal is a flip-back
  4.  No flip-back chain – A→B→C, two reversals, neither is a flip-back
  5.  Multi-stage        – reversals at Confirmation and Post-commit
  6.  Two decisions      – reversals isolated to their own decision
  7.  Quick correction   – gap ≤ 60 s (seconds_since_previous ≤ 60)
  8.  Slow reversal      – gap > 60 s
  9.  Empty DataFrame    – detector returns empty output with correct columns
  10. Loader validation  – missing column, bad stage, null value all raise ValueError
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from src.loader import load
from src.detector import detect, _empty_reversal_df
from src.config import CSV_COLUMNS, STAGE_ORDER


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build a minimal clean DataFrame from a list of tuples
# ─────────────────────────────────────────────────────────────────────────────

T0 = datetime(2024, 6, 1, 10, 0, 0)   # anchor timestamp


def _make_df(rows: list[tuple]) -> pd.DataFrame:
    """
    Build a loader-clean DataFrame from a list of
    (user_id, decision_id, timestamp_offset_seconds, stage, choice) tuples.
    Timestamps are computed as T0 + offset.
    """
    records = []
    for user_id, decision_id, offset_s, stage, choice in rows:
        records.append({
            "user_id":     user_id,
            "decision_id": decision_id,
            "timestamp":   pd.Timestamp(T0 + timedelta(seconds=offset_s)),
            "stage":       stage,
            "choice":      choice,
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# Case 1 – No change (decisive user)
# ─────────────────────────────────────────────────────────────────────────────

def test_no_change_yields_no_reversals():
    """A single event per (user, decision) should produce zero reversals."""
    df = _make_df([
        ("U001", "U001_plan", 0, "Selection", "basic"),
    ])
    result = detect(df)
    assert len(result) == 0
    # Columns must still be present even when empty
    assert set(_empty_reversal_df().columns).issubset(result.columns)


# ─────────────────────────────────────────────────────────────────────────────
# Case 2 – Single change A → B (not a flip-back)
# ─────────────────────────────────────────────────────────────────────────────

def test_single_change_not_flip_back():
    """A→B: one reversal, is_flip_back must be False."""
    df = _make_df([
        ("U001", "U001_plan", 0,   "Browsing",  "basic"),
        ("U001", "U001_plan", 120, "Selection", "premium"),
    ])
    result = detect(df)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["from_choice"] == "basic"
    assert row["to_choice"]   == "premium"
    assert row["is_flip_back"] == False
    assert row["seconds_since_previous"] == pytest.approx(120.0)


# ─────────────────────────────────────────────────────────────────────────────
# Case 3 – Flip-back A → B → A
# ─────────────────────────────────────────────────────────────────────────────

def test_flip_back_aba():
    """A→B→A: second reversal must have is_flip_back=True because A was seen before."""
    df = _make_df([
        ("U001", "U001_plan", 0,   "Browsing",  "basic"),
        ("U001", "U001_plan", 100, "Selection", "premium"),
        ("U001", "U001_plan", 250, "Review",    "basic"),   # back to first choice
    ])
    result = detect(df)

    assert len(result) == 2

    first_rev  = result.iloc[0]
    second_rev = result.iloc[1]

    # A→B: not a flip-back (B was never seen)
    assert first_rev["from_choice"]  == "basic"
    assert first_rev["to_choice"]    == "premium"
    assert first_rev["is_flip_back"] == False

    # B→A: flip-back (A was seen at index 0)
    assert second_rev["from_choice"]  == "premium"
    assert second_rev["to_choice"]    == "basic"
    assert second_rev["is_flip_back"] == True


# ─────────────────────────────────────────────────────────────────────────────
# Case 4 – A → B → C: two reversals, neither is a flip-back
# ─────────────────────────────────────────────────────────────────────────────

def test_abc_no_flip_back():
    """A→B→C: both reversals are non-flip-backs (each new choice is brand new)."""
    df = _make_df([
        ("U001", "U001_plan", 0,   "Browsing",  "basic"),
        ("U001", "U001_plan", 90,  "Selection", "standard"),
        ("U001", "U001_plan", 200, "Review",    "premium"),
    ])
    result = detect(df)

    assert len(result) == 2
    assert result["is_flip_back"].tolist() == [False, False]
    assert result["from_choice"].tolist()  == ["basic", "standard"]
    assert result["to_choice"].tolist()    == ["standard", "premium"]


# ─────────────────────────────────────────────────────────────────────────────
# Case 5 – Changes at different (late) stages
# ─────────────────────────────────────────────────────────────────────────────

def test_reversals_capture_correct_stage():
    """Stage recorded on the reversal row must be the stage of the *new* event."""
    df = _make_df([
        ("U002", "U002_add_on", 0,    "Browsing",      "none"),
        ("U002", "U002_add_on", 300,  "Confirmation",  "vpn"),
        ("U002", "U002_add_on", 600,  "Post-commit",   "cloud_backup"),
    ])
    result = detect(df)

    assert len(result) == 2
    assert result["stage"].tolist() == ["Confirmation", "Post-commit"]


# ─────────────────────────────────────────────────────────────────────────────
# Case 6 – Two independent decisions, reversals isolated correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_reversals_isolated_per_decision():
    """
    Changes on decision_A must not influence flip-back detection on decision_B,
    and vice-versa.
    """
    df = _make_df([
        # decision plan: A→B (not flip-back)
        ("U003", "U003_plan",   0,   "Browsing",  "basic"),
        ("U003", "U003_plan",   90,  "Selection", "premium"),
        # decision add_on: X→Y→X (second is flip-back)
        ("U003", "U003_add_on", 10,  "Browsing",  "none"),
        ("U003", "U003_add_on", 120, "Selection", "vpn"),
        ("U003", "U003_add_on", 240, "Review",    "none"),
    ])
    result = detect(df)

    plan_revs   = result[result["decision_id"] == "U003_plan"]
    add_on_revs = result[result["decision_id"] == "U003_add_on"]

    assert len(plan_revs)   == 1
    assert len(add_on_revs) == 2

    # plan reversal: not a flip-back
    assert plan_revs.iloc[0]["is_flip_back"] == False

    # add_on first reversal: not a flip-back; second: flip-back
    assert add_on_revs.iloc[0]["is_flip_back"] == False
    assert add_on_revs.iloc[1]["is_flip_back"] == True


# ─────────────────────────────────────────────────────────────────────────────
# Case 7 – Quick reversal (within correction window)
# ─────────────────────────────────────────────────────────────────────────────

def test_quick_reversal_seconds_within_threshold():
    """seconds_since_previous ≤ 60 for a rapid change."""
    from src.config import CORRECTION_MAX_SECONDS

    gap = 45   # well within correction window
    df = _make_df([
        ("U004", "U004_plan", 0,   "Review", "basic"),
        ("U004", "U004_plan", gap, "Review", "premium"),
    ])
    result = detect(df)

    assert len(result) == 1
    assert result.iloc[0]["seconds_since_previous"] == pytest.approx(float(gap))
    assert result.iloc[0]["seconds_since_previous"] <= CORRECTION_MAX_SECONDS


# ─────────────────────────────────────────────────────────────────────────────
# Case 8 – Slow reversal (beyond correction window)
# ─────────────────────────────────────────────────────────────────────────────

def test_slow_reversal_seconds_beyond_threshold():
    """seconds_since_previous > 60 for a deliberate late change."""
    from src.config import CORRECTION_MAX_SECONDS

    gap = 300   # 5 minutes — deliberate
    df = _make_df([
        ("U005", "U005_plan", 0,   "Selection", "standard"),
        ("U005", "U005_plan", gap, "Review",    "premium"),
    ])
    result = detect(df)

    assert len(result) == 1
    assert result.iloc[0]["seconds_since_previous"] == pytest.approx(float(gap))
    assert result.iloc[0]["seconds_since_previous"] > CORRECTION_MAX_SECONDS


# ─────────────────────────────────────────────────────────────────────────────
# Case 9 – Empty input DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_dataframe_returns_empty_with_correct_columns():
    """detect() on an empty input must return an empty DataFrame with all columns."""
    empty_df = _make_df([])
    result = detect(empty_df)

    expected_cols = {
        "user_id", "decision_id", "from_choice", "to_choice",
        "stage", "timestamp", "seconds_since_previous", "is_flip_back",
    }
    assert len(result) == 0
    assert expected_cols.issubset(set(result.columns))


# ─────────────────────────────────────────────────────────────────────────────
# Case 10 – Loader validation
# ─────────────────────────────────────────────────────────────────────────────

def test_loader_raises_on_missing_column():
    """A DataFrame lacking a required column must raise ValueError."""
    bad_df = pd.DataFrame({
        "user_id":  ["U001"],
        "stage":    ["Browsing"],
        "choice":   ["basic"],
        # missing: decision_id, timestamp
    })
    with pytest.raises(ValueError, match="Missing required column"):
        load(bad_df)


def test_loader_raises_on_unknown_stage():
    """A DataFrame with an unrecognised stage value must raise ValueError."""
    bad_df = pd.DataFrame({
        "user_id":     ["U001"],
        "decision_id": ["U001_plan"],
        "timestamp":   ["2024-06-01T10:00:00"],
        "stage":       ["Checkout"],          # not in STAGE_ORDER
        "choice":      ["basic"],
    })
    with pytest.raises(ValueError, match="Unknown stage"):
        load(bad_df)


def test_loader_raises_on_null_value():
    """A null in any required column must raise ValueError."""
    bad_df = pd.DataFrame({
        "user_id":     ["U001"],
        "decision_id": ["U001_plan"],
        "timestamp":   [None],                # null timestamp
        "stage":       ["Browsing"],
        "choice":      ["basic"],
    })
    with pytest.raises(ValueError, match="Null values"):
        load(bad_df)


def test_loader_sorts_output():
    """Loader must return rows sorted by (user_id, decision_id, timestamp)."""
    df = pd.DataFrame({
        "user_id":     ["U002", "U001", "U001"],
        "decision_id": ["U002_plan", "U001_plan", "U001_plan"],
        "timestamp":   [
            "2024-06-01T10:05:00",
            "2024-06-01T10:02:00",
            "2024-06-01T10:00:00",
        ],
        "stage":  ["Review", "Selection", "Browsing"],
        "choice": ["standard", "premium", "basic"],
    })
    result = load(df)
    # After sort: U001 (earlier timestamp first), then U002
    assert result.iloc[0]["user_id"]  == "U001"
    assert result.iloc[0]["stage"]    == "Browsing"
    assert result.iloc[-1]["user_id"] == "U002"


def test_loader_accepts_file_not_found():
    """A non-existent path must raise FileNotFoundError."""
    import pathlib
    with pytest.raises(FileNotFoundError):
        load(pathlib.Path("data/does_not_exist.csv"))


def test_loader_auto_detects_column_names():
    """Loader auto-detects aliases like customer_id, feature, created_at, selection."""
    custom_df = pd.DataFrame({
        "customer_id": ["C100", "C100"],
        "feature":     ["tier", "tier"],
        "created_at":  ["2024-06-01T10:00:00", "2024-06-01T10:01:00"],
        "phase":       ["Browsing", "Selection"],
        "selection":   ["silver", "gold"],
    })
    result = load(custom_df, auto_map_columns=True)
    assert list(result.columns) == ["user_id", "decision_id", "timestamp", "stage", "choice"]
    assert result.iloc[0]["user_id"] == "C100"
    assert result.iloc[1]["choice"] == "gold"


def test_loader_accepts_custom_stages():
    """With allow_custom_stages=True, any arbitrary funnel stages are accepted."""
    custom_df = pd.DataFrame({
        "user_id":     ["U100", "U100"],
        "decision_id": ["plan", "plan"],
        "timestamp":   ["2024-06-01T10:00:00", "2024-06-01T10:02:00"],
        "stage":       ["Cart", "Payment"],
        "choice":      ["basic", "pro"],
    })
    result = load(custom_df, allow_custom_stages=True)
    assert result.attrs["stage_order"] == ["Cart", "Payment"]
    assert "Payment" in result.attrs["late_stages"]


def test_loader_handles_missing_stage_column():
    """If a dataset has no stage column, a default stage is cleanly synthesized."""
    df_no_stage = pd.DataFrame({
        "user_id":     ["U200", "U200"],
        "decision_id": ["theme", "theme"],
        "timestamp":   ["2024-06-01T10:00:00", "2024-06-01T10:01:00"],
        "choice":      ["dark", "light"],
    })
    result = load(df_no_stage)
    assert "stage" in result.columns
    assert (result["stage"] == "General").all()
