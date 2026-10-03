# src/config.py
# ─────────────────────────────────────────────────────────────────────────────
# Central configuration for the Decision Reversal Pattern Detector.
# All thresholds live here so they can be tuned without touching any logic.
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# Input schema
# ---------------------------------------------------------------------------
# Expected columns in the CSV uploaded by the user.
CSV_COLUMNS = ["user_id", "decision_id", "timestamp", "stage", "choice"]

# The column used to parse timestamps (ISO-8601 strings are fine).
TIMESTAMP_COL = "timestamp"

# ---------------------------------------------------------------------------
# Dynamic schema / column alias auto-detection
# ---------------------------------------------------------------------------
COLUMN_ALIASES: dict[str, list[str]] = {
    "user_id": [
        "user_id", "userid", "user", "customer_id", "customer", "client_id",
        "client", "account_id", "account", "visitor_id", "visitor", "id", "uid",
        "applicant_id", "shopper_id", "member_id", "patient_id", "patient",
        "guest_id", "guest", "student_id", "student", "learner_id",
        "developer_id", "dev_id", "engineer_id", "player_id", "player", "gamer_tag", "gamer"
    ],
    "decision_id": [
        "decision_id", "decisionid", "decision", "decision_type", "feature",
        "feature_id", "item_id", "item", "item_type", "product_id", "product", "attribute",
        "setting", "component", "target", "step_id", "field", "field_name", "parameter",
        "action_type", "action_name", "interaction", "element", "event_type", "variable",
        "stockcode", "stock_code", "sku", "code", "consultation_item", "package_param",
        "infra_parameter", "curriculum_field", "loadout_slot", "slot"
    ],
    "timestamp": [
        "timestamp", "time", "datetime", "date", "event_time", "created_at",
        "recorded_at", "ts", "event_timestamp", "action_time", "timestamp_utc", "invoicedate",
        "scheduled_at", "booking_time", "config_time", "action_timestamp"
    ],
    "stage": [
        "stage", "funnel_stage", "step", "funnel_step", "checkout_step", "phase", "state",
        "screen", "page", "status", "milestone", "triage_step", "checkout_milestone",
        "deployment_phase", "enrollment_step", "inventory_screen"
    ],
    "choice": [
        "choice", "value", "option", "selection", "selected_option", "action", "variant",
        "variant_chosen", "selected", "decision_value", "choice_value", "selected_value", "picked_option",
        "quantity", "qty", "amount", "count", "rating", "option_selected", "spec_value",
        "chosen_option", "equipped_gear", "gear"
    ],
}


def auto_detect_columns(available_columns: list[str]) -> dict[str, str]:
    """
    Given a list of column names in an arbitrary CSV, auto-detect the best
    matching column for each standard field ('user_id', 'decision_id', etc.).
    Returns a dict mapping standard_field -> uploaded_column_name.
    """
    mapping: dict[str, str] = {}
    normalized = {c.strip().lower().replace("-", "_").replace(" ", "_"): c for c in available_columns}

    # Pass 1: exact match with standard field name
    for std_field in COLUMN_ALIASES:
        if std_field in normalized:
            mapping[std_field] = normalized[std_field]

    # Pass 2: match against configured aliases
    for std_field, aliases in COLUMN_ALIASES.items():
        if std_field in mapping:
            continue
        for alias in aliases:
            if alias in normalized:
                orig_col = normalized[alias]
                if orig_col not in mapping.values():
                    mapping[std_field] = orig_col
                    break

    # Pass 3: substring / keyword heuristic fallback
    keywords = {
        "user_id": ["user", "cust", "client", "member", "applicant", "account", "patient", "guest", "student", "player", "gamer", "dev", "tag"],
        "decision_id": ["decision", "item", "feature", "field", "product", "attr", "param", "action", "element", "stock", "sku", "slot"],
        "timestamp": ["time", "date", "ts", "sched"],
        "stage": ["stage", "step", "phase", "screen", "page", "milestone", "triage"],
        "choice": ["choice", "option", "select", "val", "variant", "qty", "quantity", "rating", "gear", "spec"],
    }
    for std_field, kw_list in keywords.items():
        if std_field in mapping:
            continue
        for norm_col, orig_col in normalized.items():
            if orig_col in mapping.values():
                continue
            if any(kw in norm_col for kw in kw_list):
                mapping[std_field] = orig_col
                break

    return mapping

# ---------------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------------
# Defines the canonical funnel order. Used for ordering and for computing
# which stage a reversal happens at.
STAGE_ORDER = [
    "Browsing",
    "Selection",
    "Review",
    "Confirmation",
    "Post-commit",
]

# ---------------------------------------------------------------------------
# Decision types recognised in the domain
# ---------------------------------------------------------------------------
DECISION_TYPES = ["plan", "add_on", "payment_method"]

# ---------------------------------------------------------------------------
# Reversal detection thresholds
# ---------------------------------------------------------------------------

# Maximum time (in seconds) between two consecutive choices for the pair to
# be classified as a "correction" (quick fix) rather than a deliberate reversal.
CORRECTION_MAX_SECONDS: int = 60

# Minimum number of reversals on a single (user, decision) to flag that
# decision as "high-churn" in the metrics summary.
HIGH_CHURN_REVERSAL_THRESHOLD: int = 3

# ---------------------------------------------------------------------------
# Classifier thresholds
# ---------------------------------------------------------------------------

# Rule 3 – Chronic indecision (broad across decisions)
# Minimum reversal_rate to qualify as chronically indecisive.
CHRONIC_INDECISION_RATE_THRESHOLD: float = 0.75
# Minimum number of distinct decisions that must show reversals.
CHRONIC_INDECISION_MIN_DECISIONS: int = 2
# Maximum share of reversals on a single decision (must be broad, not concentrated).
CHRONIC_INDECISION_MAX_SHARE_ON_ONE_DECISION: float = 0.60

# Rule 4 – Repeated flip-flopper (concentrated on one decision)
# Minimum reversals on ONE decision to qualify as flip-flopper via reversal count.
FLIPFLOPPER_MIN_REVERSALS_ON_ONE_DECISION: int = 3
# Minimum total flip-backs to qualify as flip-flopper via flip-back count.
FLIPFLOPPER_MIN_FLIP_BACKS: int = 2
# Minimum share of reversals concentrated on a single decision.
FLIPFLOPPER_MIN_SHARE_ON_ONE_DECISION: float = 0.60

# Rule 5 – Late reverser
# Minimum fraction of reversals that must occur at Confirmation or Post-commit.
LATE_REVERSAL_STAGE_RATIO: float = 0.60
# Stage names considered "late" for the late-reverser rule.
LATE_STAGES: list[str] = ["Confirmation", "Post-commit"]

# ---------------------------------------------------------------------------
# Pattern / heatmap display settings
# ---------------------------------------------------------------------------

# Colour scale used in the reversal-frequency heatmap (Plotly colour scale name).
HEATMAP_COLOR_SCALE: str = "YlOrRd"

# Minimum reversal count to label a cell in the heatmap (avoids clutter).
HEATMAP_LABEL_MIN_COUNT: int = 1

# ---------------------------------------------------------------------------
# App / UI settings
# ---------------------------------------------------------------------------

# Default number of rows shown in the raw-data preview table.
PREVIEW_ROWS: int = 10

# Page title shown in the Streamlit browser tab.
APP_TITLE: str = "CEREBRO – Decision Reversal Pattern Detector"
