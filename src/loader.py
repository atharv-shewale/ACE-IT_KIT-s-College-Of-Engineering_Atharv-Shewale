"""
src/loader.py
─────────────────────────────────────────────────────────────────────────────
Responsible for reading, auto-mapping, and validating event logs from any CSV.

Public API
──────────
  load(source, column_mapping=None, stage_order=None, allow_custom_stages=False, auto_map_columns=True) -> pd.DataFrame
      Accepts a file path (str | Path), an in-memory DataFrame, or BytesIO.
      Supports automatic column discovery, custom schema mapping, and arbitrary stages.
      Returns a clean, sorted DataFrame ready for the detector.

Validation performed
────────────────────
  * Required columns present (or mapped via column_mapping / auto-detection)
  * No null values in required columns
  * Stage values validated (either against STAGE_ORDER or custom stage list)
  * Timestamp column is parseable as datetime
  * Sorted by (user_id, decision_id, timestamp) ascending
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import io
import pathlib
from typing import Union

import pandas as pd

from src.config import CSV_COLUMNS, STAGE_ORDER, TIMESTAMP_COL, auto_detect_columns

# Type alias: caller may pass a file path, a ready DataFrame, or a byte stream
PathOrDF = Union[str, pathlib.Path, pd.DataFrame, io.BytesIO, io.StringIO]


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def load(
    source: PathOrDF,
    column_mapping: dict[str, str] | None = None,
    stage_order: list[str] | None = None,
    allow_custom_stages: bool = False,
    auto_map_columns: bool = True,
) -> pd.DataFrame:
    """
    Load, map, and validate an event log.

    Parameters
    ----------
    source : str | Path | pd.DataFrame | io.BytesIO | io.StringIO
        Path to a CSV file *or* an already-in-memory DataFrame or stream.
    column_mapping : dict[str, str] | None
        Optional dictionary mapping standard field names to uploaded column names
        (e.g. {"user_id": "customer_id", "choice": "option_selected"}), or vice versa.
    stage_order : list[str] | None
        Custom ordered list of funnel stages. If provided, stages are validated
        against this sequence instead of the default STAGE_ORDER.
    allow_custom_stages : bool
        If True, permits any arbitrary stage names in the dataset, dynamically
        inferring their chronological order of appearance if stage_order is None.
    auto_map_columns : bool
        If True and standard columns are missing, automatically searches for common
        column aliases (e.g. 'userId', 'event_time', 'selection').

    Returns
    -------
    pd.DataFrame
        Validated, sorted DataFrame with standardized column names and parsed `timestamp`.
        Metadata attached to `df.attrs['stage_order']` and `df.attrs['late_stages']`.

    Raises
    ------
    FileNotFoundError   – path does not exist
    ValueError          – schema or data content is invalid
    """
    df = _read(source)
    df, active_stages, late_stages = _validate(
        df,
        column_mapping=column_mapping,
        stage_order=stage_order,
        allow_custom_stages=allow_custom_stages,
        auto_map_columns=auto_map_columns,
    )
    df = _sort(df)

    # Attach dynamic metadata for downstream pipeline steps
    df.attrs["stage_order"] = active_stages
    df.attrs["late_stages"] = late_stages
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read(source: PathOrDF) -> pd.DataFrame:
    """Load from a CSV path, byte/string stream, or pass a DataFrame through."""
    if isinstance(source, pd.DataFrame):
        return source.copy()

    if isinstance(source, (io.BytesIO, io.StringIO)):
        source.seek(0)
        return pd.read_csv(source)

    path = pathlib.Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Event log not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a .csv file, got: {path.suffix!r}")

    return pd.read_csv(path)


def _validate(
    df: pd.DataFrame,
    column_mapping: dict[str, str] | None = None,
    stage_order: list[str] | None = None,
    allow_custom_stages: bool = False,
    auto_map_columns: bool = True,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Check schema correctness, apply column mappings, and parse timestamps.
    Returns (cleaned_df, active_stage_order, late_stages).
    """
    # ── 1. Apply column mapping if provided ───────────────────────────────
    if column_mapping:
        rename_dict = {}
        for k, v in column_mapping.items():
            if k in CSV_COLUMNS and v in df.columns:
                rename_dict[v] = k
            elif v in CSV_COLUMNS and k in df.columns:
                rename_dict[k] = v
        df = df.rename(columns=rename_dict)
    elif auto_map_columns:
        # Check if any standard columns are missing before auto-mapping
        missing_std = [c for c in CSV_COLUMNS if c not in df.columns]
        if missing_std:
            detected = auto_detect_columns(list(df.columns))
            rename_dict = {orig: std for std, orig in detected.items() if orig in df.columns and std not in df.columns}
            if rename_dict:
                df = df.rename(columns=rename_dict)

    # ── 2. Handle optional 'stage' column ─────────────────────────────────
    had_stage_col = "stage" in df.columns
    if not had_stage_col:
        df["stage"] = "General"

    # ── 3. Check for required columns ─────────────────────────────────────
    missing = [col for col in CSV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}. "
            f"Expected: {CSV_COLUMNS}"
        )

    # Work only with required columns
    df = df[CSV_COLUMNS].copy()

    # ── 4. No nulls in required columns ───────────────────────────────────
    null_counts = df.isnull().sum()
    bad_cols = null_counts[null_counts > 0].to_dict()
    if bad_cols:
        raise ValueError(
            f"Null values found in column(s): {bad_cols}. "
            "All required columns must be fully populated."
        )

    # ── 5. Parse timestamps ───────────────────────────────────────────────
    try:
        parsed_ts = pd.to_datetime(df[TIMESTAMP_COL], utc=False, errors="coerce")
        if parsed_ts.isnull().any():
            bad_samples = df[TIMESTAMP_COL][parsed_ts.isnull()].head(3).tolist()
            raise ValueError(f"Unparseable datetime samples: {bad_samples}")
        df[TIMESTAMP_COL] = parsed_ts
    except Exception as exc:
        raise ValueError(
            f"Could not parse '{TIMESTAMP_COL}' column as datetime. "
            f"Use ISO-8601 format (e.g. 2024-06-01T09:04:25). "
            f"Original error: {exc}"
        ) from exc

    # ── 6. Ensure clean string representations ────────────────────────────
    df["user_id"] = df["user_id"].astype(str).str.strip()
    df["decision_id"] = df["decision_id"].astype(str).str.strip()
    df["choice"] = df["choice"].astype(str).str.strip()
    df["stage"] = df["stage"].astype(str).str.strip()

    # ── 7. Validate and determine Stage Ordering ──────────────────────────
    unique_stages = list(dict.fromkeys(df["stage"].tolist()))

    if not had_stage_col and stage_order is None:
        active_stages = ["General"]
    elif stage_order is not None:
        active_stages = list(stage_order)
        # Ensure all stages present in dataset are in stage_order
        unknown = set(unique_stages) - set(active_stages)
        if unknown and not allow_custom_stages:
            raise ValueError(
                f"Unknown stage value(s): {unknown}. "
                f"Allowed stages: {active_stages}"
            )
        # Add any unlisted stages to the end if custom stages allowed
        for u in unique_stages:
            if u not in active_stages:
                active_stages.append(u)
    else:
        # Check if stages are a subset of the default STAGE_ORDER
        is_canonical_subset = set(unique_stages).issubset(set(STAGE_ORDER))
        if is_canonical_subset:
            # Preserve standard funnel order
            active_stages = [s for s in STAGE_ORDER if s in unique_stages] or STAGE_ORDER
        elif allow_custom_stages:
            # Order stages dynamically by first chronological appearance
            first_seen = df.groupby("stage")[TIMESTAMP_COL].min().sort_values()
            active_stages = list(first_seen.index)
        else:
            unknown_stages = set(unique_stages) - set(STAGE_ORDER)
            raise ValueError(
                f"Unknown stage value(s): {unknown_stages}. "
                f"Allowed stages: {STAGE_ORDER}"
            )

    # ── 8. Determine Late Stages dynamically ──────────────────────────────
    if len(active_stages) >= 3:
        # Last 35-40% of stages are considered 'late' (minimum 1, default last 2)
        n_late = max(1, round(len(active_stages) * 0.4))
        late_stages = active_stages[-n_late:]
    elif len(active_stages) == 2:
        late_stages = [active_stages[-1]]
    else:
        late_stages = active_stages

    return df, active_stages, late_stages


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    """Sort events into chronological order within each (user, decision) group."""
    return (
        df
        .sort_values(["user_id", "decision_id", TIMESTAMP_COL])
        .reset_index(drop=True)
    )
