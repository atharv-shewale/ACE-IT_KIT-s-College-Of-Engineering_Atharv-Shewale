"""
tests/test_multi_dataset_pipeline.py
─────────────────────────────────────────────────────────────────────────────
End-to-end integration tests verifying dynamic multi-dataset compatibility.
Ensures CEREBRO processes arbitrary schemas, missing stages, and custom funnels.
─────────────────────────────────────────────────────────────────────────────
"""

from io import StringIO
import pandas as pd
import pytest

from src.loader import load
from src.detector import detect
from src.metrics import compute
from src.classifier import classify
from src.explain import explain_all
from src.visuals import stage_chart, timeline_chart, flow_chart


def test_pipeline_ecommerce():
    """Verify end-to-end pipeline on non-standard e-commerce checkout dataset."""
    events = load("data/ecommerce_events.csv", allow_custom_stages=True, auto_map_columns=True)
    assert "user_id" in events.columns
    assert "decision_id" in events.columns
    assert "stage" in events.columns

    reversals = detect(events)
    stage_order = events.attrs.get("stage_order")
    late_stages = events.attrs.get("late_stages")
    assert stage_order == ["Cart", "Shipping", "Payment_Info", "Order_Review"]
    assert "Order_Review" in late_stages

    metrics_df = compute(events, reversals, stage_order=stage_order)
    assert len(metrics_df) == 60
    assert "reversal_rate" in metrics_df.columns

    labeled_df = classify(metrics_df, reversals, late_stages=late_stages)
    assert "pattern" in labeled_df.columns
    assert set(labeled_df["pattern"].unique()).issubset({
        "decisive", "single_corrector", "flip_flopper", "late_reverser", "chronic_indecisive", "mixed"
    })

    explained_df = explain_all(metrics_df, reversals, labeled_df, late_stages=late_stages)
    assert "explanation" in explained_df.columns
    assert len(explained_df["explanation"].iloc[0]) > 10

    # Visuals generation without exceptions
    fig_stage = stage_chart(reversals, stage_order=stage_order)
    first_user = labeled_df.iloc[0]["user_id"]
    fig_timeline = timeline_chart(events, reversals, first_user)
    first_decision = events[events["user_id"] == first_user]["decision_id"].iloc[0]
    fig_flow = flow_chart(events, first_user, first_decision)
    assert fig_stage is not None and fig_timeline is not None and fig_flow is not None


def test_pipeline_loan_application():
    """Verify end-to-end pipeline on fintech / loan application dataset."""
    events = load("data/loan_application_events.csv", allow_custom_stages=True, auto_map_columns=True)
    reversals = detect(events)
    stage_order = events.attrs.get("stage_order")
    late_stages = events.attrs.get("late_stages")
    assert "Final_Signature" in stage_order

    metrics_df = compute(events, reversals, stage_order=stage_order)
    labeled_df = classify(metrics_df, reversals, late_stages=late_stages)
    assert len(labeled_df) == 60

    explained_df = explain_all(metrics_df, reversals, labeled_df, late_stages=late_stages)
    assert not explained_df["explanation"].isna().any()


def test_pipeline_arbitrary_unstructured_csv_no_stage():
    """Verify handling of raw clickstream data with completely custom aliases and NO stage column."""
    csv_data = """visitor_id,action_type,timestamp_utc,variant_chosen
user_A,button_color,2026-10-01 10:00:00,blue
user_A,button_color,2026-10-01 10:00:15,red
user_B,button_color,2026-10-01 10:00:00,green
user_B,font_size,2026-10-01 10:01:00,14px
user_B,font_size,2026-10-01 10:02:00,16px
user_B,font_size,2026-10-01 10:03:00,14px
"""
    df = load(StringIO(csv_data), allow_custom_stages=True, auto_map_columns=True)
    assert "stage" in df.columns
    assert df["stage"].unique().tolist() == ["General"]

    revs = detect(df)
    assert len(revs) >= 2  # user_A (1) + user_B (2)

    metrics = compute(df, revs)
    classified = classify(metrics, revs)
    explained = explain_all(metrics, revs, classified)

    assert len(classified) == 2
    assert "pattern" in classified.columns
    assert "explanation" in explained.columns


def test_pipeline_healthcare():
    """Verify pipeline on healthcare appointments dataset."""
    events = load("data/healthcare_appointments.csv", allow_custom_stages=True, auto_map_columns=True)
    revs = detect(events)
    stage_order = events.attrs.get("stage_order")
    late_stages = events.attrs.get("late_stages")
    metrics = compute(events, revs, stage_order=stage_order)
    classified = classify(metrics, revs, late_stages=late_stages)
    assert len(classified) == 50
    assert "Insurance_Verification" in stage_order


def test_pipeline_hotel_travel():
    """Verify pipeline on hotel & travel booking dataset."""
    events = load("data/hotel_travel_booking.csv", allow_custom_stages=True, auto_map_columns=True)
    revs = detect(events)
    stage_order = events.attrs.get("stage_order")
    late_stages = events.attrs.get("late_stages")
    metrics = compute(events, revs, stage_order=stage_order)
    classified = classify(metrics, revs, late_stages=late_stages)
    assert len(classified) == 54
    assert "Final_Confirmation" in stage_order


def test_pipeline_cloud_infra():
    """Verify pipeline on cloud devops provisioning dataset."""
    events = load("data/cloud_infra_provisioning.csv", allow_custom_stages=True, auto_map_columns=True)
    revs = detect(events)
    stage_order = events.attrs.get("stage_order")
    late_stages = events.attrs.get("late_stages")
    metrics = compute(events, revs, stage_order=stage_order)
    classified = classify(metrics, revs, late_stages=late_stages)
    assert len(classified) == 50
    assert "Review_Deploy" in stage_order


def test_pipeline_edtech():
    """Verify pipeline on edtech course enrollment dataset."""
    events = load("data/edtech_course_enrollment.csv", allow_custom_stages=True, auto_map_columns=True)
    revs = detect(events)
    stage_order = events.attrs.get("stage_order")
    late_stages = events.attrs.get("late_stages")
    metrics = compute(events, revs, stage_order=stage_order)
    classified = classify(metrics, revs, late_stages=late_stages)
    assert len(classified) == 50
    assert "Tuition_Plan" in stage_order


def test_pipeline_gaming():
    """Verify pipeline on gaming character loadout dataset."""
    events = load("data/gaming_character_loadout.csv", allow_custom_stages=True, auto_map_columns=True)
    revs = detect(events)
    stage_order = events.attrs.get("stage_order")
    late_stages = events.attrs.get("late_stages")
    metrics = compute(events, revs, stage_order=stage_order)
    classified = classify(metrics, revs, late_stages=late_stages)
    assert len(classified) == 54
    assert "Confirm_Battle_Loadout" in stage_order


def test_pipeline_real_retail_dataset():
    """Verify pipeline on real-world UCI / Databricks retail transaction dataset."""
    events = load("data/real_retail_events.csv", allow_custom_stages=True, auto_map_columns=True)
    revs = detect(events)
    assert len(revs) > 0
    metrics = compute(events, revs)
    classified = classify(metrics, revs)
    assert len(classified) == 98
    explained = explain_all(metrics, revs, classified)
    assert len(explained) == 98
