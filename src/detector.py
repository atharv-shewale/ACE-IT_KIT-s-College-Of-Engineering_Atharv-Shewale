"""
src/detector.py
─────────────────────────────────────────────────────────────────────────────
Core reversal detection engine.

Definitions (from spec)
────────────────────────
  Reversal   : a choice that differs from the immediately preceding choice
               for the same (user_id, decision_id).

  Flip-back  : a reversal where the new choice equals ANY earlier choice
               recorded for that (user_id, decision_id) — i.e. the user
               is returning to something they already tried.

  Correction : exactly one reversal on a (user_id, decision_id), made
               within CORRECTION_MAX_SECONDS of the previous choice.
               (Classified downstream in classifier.py; the seconds
               field is computed here so the classifier can use it.)

Public API
──────────
  detect(df: pd.DataFrame) -> pd.DataFrame
      Input  : clean DataFrame from loader.load()
      Output : one row per reversal with columns:
                 user_id, decision_id, from_choice, to_choice,
                 stage, timestamp, seconds_since_previous, is_flip_back
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import pandas as pd

from src.config import TIMESTAMP_COL


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def detect(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect reversals in a clean event DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Output of loader.load() — sorted by (user_id, decision_id, timestamp).

    Returns
    -------
    pd.DataFrame
        One row per reversal event, with columns:
          user_id, decision_id, from_choice, to_choice,
          stage, timestamp, seconds_since_previous, is_flip_back

        Returns an empty DataFrame (with correct columns) when no reversals
        are found.
    """
    if df.empty:
        return _empty_reversal_df()

    # Process each (user, decision) group independently
    reversal_rows: list[dict] = []

    groups = df.groupby(["user_id", "decision_id"], sort=False)
    for (user_id, decision_id), group in groups:
        rows = _detect_in_group(user_id, decision_id, group)
        reversal_rows.extend(rows)

    if not reversal_rows:
        return _empty_reversal_df()

    result = pd.DataFrame(reversal_rows)
    # Preserve chronological order across all users
    result = result.sort_values(TIMESTAMP_COL).reset_index(drop=True)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-group detection
# ─────────────────────────────────────────────────────────────────────────────

def _detect_in_group(
    user_id: str,
    decision_id: str,
    group: pd.DataFrame,
) -> list[dict]:
    """
    Walk through the chronologically ordered events for one (user, decision)
    pair and emit one dict per reversal found.

    Parameters
    ----------
    user_id      : str
    decision_id  : str
    group        : sub-DataFrame already sorted by timestamp for this pair

    Returns
    -------
    list[dict]   : zero or more reversal records
    """
    # Sort defensively (loader already sorts, but groups may arrive unsorted)
    group = group.sort_values(TIMESTAMP_COL)
    events = group.to_dict("records")  # list of row dicts — fast to iterate

    reversals: list[dict] = []
    seen_choices: list[str] = []       # ordered history of all choices seen so far

    for i, event in enumerate(events):
        current_choice = event["choice"]

        if i == 0:
            # First event for this group — nothing to compare against
            seen_choices.append(current_choice)
            continue

        prev_event  = events[i - 1]
        prev_choice = prev_event["choice"]
        prev_ts     = prev_event[TIMESTAMP_COL]
        curr_ts     = event[TIMESTAMP_COL]

        # ── Is this a reversal? ────────────────────────────────────────────
        if current_choice == prev_choice:
            # Same choice repeated — not a reversal
            seen_choices.append(current_choice)
            continue

        # ── Compute time gap ──────────────────────────────────────────────
        seconds_since_prev = _seconds_between(prev_ts, curr_ts)

        # ── Is this a flip-back? ──────────────────────────────────────────
        # True when the new choice matches ANY choice from the history
        # (excluding the immediately previous one, which we already know
        #  is different — so we check the full seen list).
        is_flip_back = current_choice in seen_choices

        reversals.append({
            "user_id":               user_id,
            "decision_id":           decision_id,
            "from_choice":           prev_choice,
            "to_choice":             current_choice,
            "stage":                 event["stage"],
            "timestamp":             curr_ts,
            "seconds_since_previous": seconds_since_prev,
            "is_flip_back":          is_flip_back,
        })

        seen_choices.append(current_choice)

    return reversals


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _seconds_between(t1: pd.Timestamp, t2: pd.Timestamp) -> float:
    """Return the number of seconds between two Timestamps (t2 − t1)."""
    delta = t2 - t1
    return delta.total_seconds()


def _empty_reversal_df() -> pd.DataFrame:
    """Return a correctly-columned empty DataFrame when there are no reversals."""
    return pd.DataFrame(columns=[
        "user_id",
        "decision_id",
        "from_choice",
        "to_choice",
        "stage",
        "timestamp",
        "seconds_since_previous",
        "is_flip_back",
    ])
