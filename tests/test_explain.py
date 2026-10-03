"""
tests/test_explain.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for src/explain.py: deterministic human-readable evidence explanation
generation for all personas without LLMs.
─────────────────────────────────────────────────────────────────────────────
"""

import pytest
import pandas as pd

from src.explain import explain_user, explain_all
from src.classifier import (
    PATTERN_DECISIVE,
    PATTERN_SINGLE_CORRECTOR,
    PATTERN_FLIP_FLOPPER,
    PATTERN_LATE_REVERSER,
    PATTERN_CHRONIC_INDECISIVE,
    PATTERN_MIXED,
)


def test_explain_decisive():
    metrics_row = pd.Series({"total_reversals": 0, "reversal_rate": 0.0})
    revs = pd.DataFrame()
    explanation = explain_user("U001", metrics_row, revs, PATTERN_DECISIVE)
    assert "U001" in explanation
    assert "no changes" in explanation.lower()


def test_explain_single_corrector():
    metrics_row = pd.Series({"total_reversals": 1})
    revs = pd.DataFrame([{
        "user_id": "U002",
        "decision_id": "plan",
        "seconds_since_previous": 25.0,
        "from_choice": "basic",
        "to_choice": "premium",
    }])
    explanation = explain_user("U002", metrics_row, revs, PATTERN_SINGLE_CORRECTOR)
    assert "U002" in explanation
    assert "plan" in explanation
    assert "quick correction" in explanation.lower() or "within" in explanation.lower()


def test_explain_flip_flopper():
    metrics_row = pd.Series({
        "total_reversals": 3,
        "reversals_per_decision": {"plan": 3},
        "flip_back_count": 2,
        "most_common_reversal_stage": "Review",
    })
    revs = pd.DataFrame([
        {"user_id": "U003", "decision_id": "plan", "is_flip_back": True, "stage": "Review"},
        {"user_id": "U003", "decision_id": "plan", "is_flip_back": True, "stage": "Review"},
        {"user_id": "U003", "decision_id": "plan", "is_flip_back": False, "stage": "Selection"},
    ])
    explanation = explain_user("U003", metrics_row, revs, PATTERN_FLIP_FLOPPER)
    assert "U003" in explanation
    assert "plan" in explanation
    assert "back to a previous choice" in explanation.lower()


def test_explain_late_reverser():
    metrics_row = pd.Series({
        "total_reversals": 2,
        "decisions_with_reversals": 1,
        "late_reversal_ratio": 1.0,
    })
    revs = pd.DataFrame([
        {"user_id": "U004", "decision_id": "plan", "stage": "Confirmation"},
        {"user_id": "U004", "decision_id": "plan", "stage": "Post-commit"},
    ])
    explanation = explain_user("U004", metrics_row, revs, PATTERN_LATE_REVERSER)
    assert "U004" in explanation
    assert "late" in explanation.lower()


def test_explain_chronic_indecisive():
    metrics_row = pd.Series({
        "total_reversals": 4,
        "reversal_rate": 1.33,
        "decisions_with_reversals": 3,
    })
    revs = pd.DataFrame([
        {"user_id": "U005", "decision_id": "plan"},
        {"user_id": "U005", "decision_id": "seat"},
        {"user_id": "U005", "decision_id": "payment"},
    ])
    explanation = explain_user("U005", metrics_row, revs, PATTERN_CHRONIC_INDECISIVE)
    assert "U005" in explanation
    assert "3" in explanation or "area" in explanation.lower()


def test_explain_all_integration():
    classified_df = pd.DataFrame([
        {"user_id": "U1", "pattern": PATTERN_DECISIVE, "total_reversals": 0, "reversal_rate": 0.0},
        {
            "user_id": "U2",
            "pattern": PATTERN_SINGLE_CORRECTOR,
            "total_reversals": 1,
            "reversals_per_decision": {"plan": 1},
            "flip_back_count": 0,
            "most_common_reversal_stage": "Selection",
        },
    ])
    metrics_df = classified_df.copy()
    reversals_df = pd.DataFrame([
        {
            "user_id": "U2",
            "decision_id": "plan",
            "seconds_since_previous": 20.0,
            "from_choice": "basic",
            "to_choice": "standard",
            "is_flip_back": False,
            "stage": "Selection",
        }
    ])

    result = explain_all(metrics_df, reversals_df, classified_df)
    assert "explanation" in result.columns
    assert len(result) == 2
    assert len(result.iloc[0]["explanation"]) > 0
    assert len(result.iloc[1]["explanation"]) > 0
