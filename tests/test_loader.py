"""
tests/test_loader.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for src/loader.py: validation, ingestion streams, dynamic column
mapping, missing stage synthesis, and ordering.
─────────────────────────────────────────────────────────────────────────────
"""

import io
import pytest
import pandas as pd
from datetime import datetime

from src.loader import load
from src.config import STAGE_ORDER


def test_load_from_valid_dataframe():
    raw_df = pd.DataFrame([
        {"user_id": "U1", "decision_id": "plan", "timestamp": "2024-06-01T10:00:00", "stage": "Selection", "choice": "basic"},
        {"user_id": "U1", "decision_id": "plan", "timestamp": "2024-06-01T10:05:00", "stage": "Review", "choice": "premium"},
    ])
    df = load(raw_df)
    assert len(df) == 2
    assert list(df.columns) == ["user_id", "decision_id", "timestamp", "stage", "choice"]
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_load_from_string_io():
    csv_text = (
        "user_id,decision_id,timestamp,stage,choice\n"
        "U1,plan,2024-06-01 10:00:00,Selection,basic\n"
        "U1,plan,2024-06-01 10:05:00,Review,standard\n"
    )
    df = load(io.StringIO(csv_text))
    assert len(df) == 2
    assert df.iloc[0]["choice"] == "basic"
    assert df.iloc[1]["stage"] == "Review"


def test_load_auto_detect_synonyms():
    csv_text = (
        "client_id,decision_type,recorded_at,funnel_step,option_selected\n"
        "C01,subscription,2024-06-01 10:00:00,Selection,tier_1\n"
        "C01,subscription,2024-06-01 10:02:00,Review,tier_2\n"
    )
    df = load(io.StringIO(csv_text), auto_map_columns=True)
    assert "user_id" in df.columns
    assert "decision_id" in df.columns
    assert "timestamp" in df.columns
    assert df.iloc[0]["user_id"] == "C01"
    assert df.iloc[1]["choice"] == "tier_2"


def test_load_missing_stage_synthesizes_general():
    csv_text = (
        "user_id,decision_id,timestamp,choice\n"
        "U1,seat,2024-06-01 10:00:00,12A\n"
        "U1,seat,2024-06-01 10:01:00,14C\n"
    )
    df = load(io.StringIO(csv_text))
    assert "stage" in df.columns
    assert (df["stage"] == "General").all()


def test_load_custom_stage_sequence():
    csv_text = (
        "user_id,decision_id,timestamp,stage,choice\n"
        "U1,role,2024-06-01 10:00:00,Phase_Alpha,DPS\n"
        "U1,role,2024-06-01 10:05:00,Phase_Beta,Tank\n"
    )
    custom_stages = ["Phase_Alpha", "Phase_Beta", "Phase_Omega"]
    df = load(io.StringIO(csv_text), stage_order=custom_stages, allow_custom_stages=True)
    assert len(df) == 2
    assert df.iloc[0]["stage"] == "Phase_Alpha"


def test_load_raises_on_null_values():
    csv_text = (
        "user_id,decision_id,timestamp,stage,choice\n"
        "U1,,2024-06-01 10:00:00,Selection,basic\n"
    )
    with pytest.raises(ValueError, match="Null values found"):
        load(io.StringIO(csv_text))
