"""
data/generate_data.py
─────────────────────────────────────────────────────────────────────────────
Synthetic event-log generator for the Decision Reversal Pattern Detector.

Produces:
  data/sample_events.csv  – one row per choice event (the model input)
  data/ground_truth.csv   – one row per user with their hidden persona

Personas
────────
decisive           – never changes a choice
single_corrector   – exactly one quick change (≤60 s) on one decision, no flip-back
flip_flopper       – 3+ changes on one decision, ≥2 flip-backs
late_reverser      – changes concentrated at Confirmation / Post-commit stages
chronic_indecisive – changes spread across 2+ decisions, high overall change rate

Run:
  python data/generate_data.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── make sure project root is on the path so we can import config ─────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import STAGE_ORDER, DECISION_TYPES, CORRECTION_MAX_SECONDS

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Constants / domain vocabulary
# ─────────────────────────────────────────────────────────────────────────────
N_USERS = 200
BASE_DATE = datetime(2024, 6, 1, 8, 0, 0)   # session start anchor

# Possible choices per decision type
CHOICES: dict[str, list[str]] = {
    "plan":           ["basic", "standard", "premium"],
    "add_on":         ["none", "cloud_backup", "vpn", "antivirus"],
    "payment_method": ["credit_card", "debit_card", "upi", "net_banking"],
}

# Persona distribution (must sum to N_USERS)
PERSONA_COUNTS: dict[str, int] = {
    "decisive":           40,
    "single_corrector":   40,
    "flip_flopper":       40,
    "late_reverser":      40,
    "chronic_indecisive": 40,
}

assert sum(PERSONA_COUNTS.values()) == N_USERS, "Persona counts must sum to N_USERS"

# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _decision_id(user_id: str, decision_type: str) -> str:
    """Stable decision identifier for a (user, decision_type) pair."""
    return f"{user_id}_{decision_type}"


def _other_choice(current: str, dtype: str) -> str:
    """Pick a different choice from the same decision type."""
    options = [c for c in CHOICES[dtype] if c != current]
    return random.choice(options)


def _earlier_choice(history: list[str], current: str, dtype: str) -> str:
    """
    Return a choice that was used before in `history` AND is different from
    `current`.  Falls back to a random different choice if history has no
    earlier value to flip back to.
    """
    earlier = [c for c in history if c != current]
    if earlier:
        return random.choice(earlier)
    return _other_choice(current, dtype)


def _make_event(
    user_id: str,
    dtype: str,
    stage: str,
    choice: str,
    ts: datetime,
) -> dict:
    """Build a single event row dictionary."""
    return {
        "user_id":     user_id,
        "decision_id": _decision_id(user_id, dtype),
        "timestamp":   ts.isoformat(timespec="seconds"),
        "stage":       stage,
        "choice":      choice,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-persona session builders
# Each builder returns a list of event dicts for one user session.
# ─────────────────────────────────────────────────────────────────────────────

def _build_decisive(user_id: str, session_start: datetime) -> list[dict]:
    """
    Decisive: picks one choice per decision at a sensible stage and never
    revisits it.  Moves cleanly through the funnel.
    """
    events = []
    ts = session_start

    # Assign one stage per decision (spread across early funnel stages)
    stage_slots = random.sample(STAGE_ORDER[:4], k=3)  # picks from first 4 stages

    for dtype, stage in zip(DECISION_TYPES, sorted(stage_slots, key=STAGE_ORDER.index)):
        choice = random.choice(CHOICES[dtype])
        ts += timedelta(seconds=random.randint(30, 120))
        events.append(_make_event(user_id, dtype, stage, choice, ts))

    return events


def _build_single_corrector(user_id: str, session_start: datetime) -> list[dict]:
    """
    Single corrector: one decision has exactly one reversal within ≤60 s,
    no flip-back (the new value was never seen before); other decisions are stable.
    """
    events = []
    ts = session_start

    # Pick which decision gets the quick correction
    corrected_dtype = random.choice(DECISION_TYPES)

    for dtype in DECISION_TYPES:
        stage = random.choice(STAGE_ORDER[:3])   # early-to-mid funnel
        choice = random.choice(CHOICES[dtype])
        ts += timedelta(seconds=random.randint(30, 90))
        events.append(_make_event(user_id, dtype, stage, choice, ts))

        if dtype == corrected_dtype:
            # Quick correction: gap ≤ CORRECTION_MAX_SECONDS, new choice != old
            gap = random.randint(5, CORRECTION_MAX_SECONDS)
            ts += timedelta(seconds=gap)
            # Ensure the new choice was never seen before (no flip-back)
            new_choice = _other_choice(choice, dtype)
            events.append(_make_event(user_id, dtype, stage, new_choice, ts))

    return events


def _build_flip_flopper(user_id: str, session_start: datetime) -> list[dict]:
    """
    Flip-flopper: one decision has ≥3 changes with ≥2 flip-backs.
    The other decisions may be stable or have one small change.
    """
    events = []
    ts = session_start

    # The "turbulent" decision
    flop_dtype = random.choice(DECISION_TYPES)
    history: list[str] = []

    # Initial choice at Browsing / Selection
    initial_stage = random.choice(["Browsing", "Selection"])
    first_choice = random.choice(CHOICES[flop_dtype])
    ts += timedelta(seconds=random.randint(15, 60))
    events.append(_make_event(user_id, flop_dtype, initial_stage, first_choice, ts))
    history.append(first_choice)
    current = first_choice

    # Generate at least 3 changes; at least 2 must be flip-backs
    n_changes = random.randint(3, 5)
    flip_back_quota = 2   # minimum flip-backs required

    stage_progression = iter(STAGE_ORDER[1:])  # start from Selection onward
    next_stage = next(stage_progression, "Review")

    for i in range(n_changes):
        ts += timedelta(seconds=random.randint(30, 180))

        # Use up flip-back quota first, then randomise
        if flip_back_quota > 0 and len(history) >= 2:
            new_choice = _earlier_choice(history, current, flop_dtype)
            flip_back_quota -= 1
        else:
            new_choice = _other_choice(current, flop_dtype)

        # Advance stage occasionally
        if random.random() < 0.6:
            next_stage = next(stage_progression, STAGE_ORDER[-1])

        events.append(_make_event(user_id, flop_dtype, next_stage, new_choice, ts))
        history.append(new_choice)
        current = new_choice

    # Other decisions: mostly stable, occasionally one small change
    for dtype in DECISION_TYPES:
        if dtype == flop_dtype:
            continue
        stage = random.choice(STAGE_ORDER[:2])
        choice = random.choice(CHOICES[dtype])
        ts += timedelta(seconds=random.randint(20, 60))
        events.append(_make_event(user_id, dtype, stage, choice, ts))

        if random.random() < 0.3:   # 30 % chance of one extra change
            ts += timedelta(seconds=random.randint(60, 200))
            events.append(_make_event(user_id, dtype, "Review",
                                      _other_choice(choice, dtype), ts))

    return events


def _build_late_reverser(user_id: str, session_start: datetime) -> list[dict]:
    """
    Late reverser: makes initial choices early in the funnel, then changes
    at least one decision at Confirmation or Post-commit (late stages).
    """
    events = []
    ts = session_start

    # Pick 1–2 decisions to reverse late
    n_late = random.randint(1, 2)
    late_dtypes = random.sample(DECISION_TYPES, k=n_late)

    # Initial choices – all made in the first three stages
    initial_choices: dict[str, str] = {}
    for dtype in DECISION_TYPES:
        stage = random.choice(STAGE_ORDER[:3])
        choice = random.choice(CHOICES[dtype])
        ts += timedelta(seconds=random.randint(30, 120))
        events.append(_make_event(user_id, dtype, stage, choice, ts))
        initial_choices[dtype] = choice

    # Fast-forward to late funnel stages
    ts += timedelta(seconds=random.randint(300, 900))   # several minutes pass

    # Late reversals at Confirmation / Post-commit
    for dtype in late_dtypes:
        late_stage = random.choice(["Confirmation", "Post-commit"])
        new_choice = _other_choice(initial_choices[dtype], dtype)
        ts += timedelta(seconds=random.randint(30, 120))
        events.append(_make_event(user_id, dtype, late_stage, new_choice, ts))

    return events


def _build_chronic_indecisive(user_id: str, session_start: datetime) -> list[dict]:
    """
    Chronically indecisive: changes occur across ≥2 decisions, with a high
    overall change rate (multiple changes per decision, spread across stages).
    """
    events = []
    ts = session_start

    # All three decisions will be changed; at least 2 will have multiple changes
    for dtype in DECISION_TYPES:
        n_changes = random.randint(2, 4)
        stage_iter = iter(STAGE_ORDER)
        current_stage = next(stage_iter, "Browsing")

        choice = random.choice(CHOICES[dtype])
        ts += timedelta(seconds=random.randint(20, 60))
        events.append(_make_event(user_id, dtype, current_stage, choice, ts))

        for _ in range(n_changes):
            ts += timedelta(seconds=random.randint(60, 300))
            # Advance stage occasionally
            if random.random() < 0.5:
                current_stage = next(stage_iter, STAGE_ORDER[-1])
            new_choice = _other_choice(choice, dtype)
            events.append(_make_event(user_id, dtype, current_stage, new_choice, ts))
            choice = new_choice

    return events


# ─────────────────────────────────────────────────────────────────────────────
# Persona → builder mapping
# ─────────────────────────────────────────────────────────────────────────────
BUILDERS = {
    "decisive":           _build_decisive,
    "single_corrector":   _build_single_corrector,
    "flip_flopper":       _build_flip_flopper,
    "late_reverser":      _build_late_reverser,
    "chronic_indecisive": _build_chronic_indecisive,
}

# ─────────────────────────────────────────────────────────────────────────────
# Main generation routine
# ─────────────────────────────────────────────────────────────────────────────

def generate(n_users: int = N_USERS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic events for `n_users` users.

    Returns
    -------
    events_df      : DataFrame with columns per CSV_COLUMNS spec
    ground_truth_df: DataFrame with columns [user_id, persona]
    """
    # Build ordered persona list (40 of each)
    personas: list[str] = []
    for persona, count in PERSONA_COUNTS.items():
        personas.extend([persona] * count)
    random.shuffle(personas)   # randomise user assignment

    all_events: list[dict] = []
    truth_rows: list[dict] = []

    for i, persona in enumerate(personas):
        user_id = f"U{i + 1:04d}"
        # Spread session starts across a 30-day window for realism
        offset_hours = random.uniform(0, 30 * 24)
        session_start = BASE_DATE + timedelta(hours=offset_hours)

        builder = BUILDERS[persona]
        user_events = builder(user_id, session_start)
        all_events.extend(user_events)
        truth_rows.append({"user_id": user_id, "persona": persona})

    events_df = pd.DataFrame(all_events)
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    events_df = events_df.sort_values("timestamp").reset_index(drop=True)

    ground_truth_df = pd.DataFrame(truth_rows)
    return events_df, ground_truth_df


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generating synthetic event log …")
    events_df, gt_df = generate()

    events_path = os.path.join(out_dir, "sample_events.csv")
    truth_path  = os.path.join(out_dir, "ground_truth.csv")

    events_df.to_csv(events_path, index=False)
    gt_df.to_csv(truth_path, index=False)

    print(f"\n[OK] sample_events.csv  -> {len(events_df):,} rows  ({events_path})")
    print(f"[OK] ground_truth.csv   -> {len(gt_df):,} rows  ({truth_path})")

    print("\nPersona distribution:")
    counts = gt_df["persona"].value_counts()
    for persona, count in counts.items():
        bar = "#" * count
        print(f"  {persona:<22} {count:>3}  {bar}")
