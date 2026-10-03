"""
evaluate.py
─────────────────────────────────────────────────────────────────────────────
Evaluation script for the Decision Reversal Pattern Detector.

Runs the full pipeline on data/sample_events.csv, compares predicted pattern
labels against data/ground_truth.csv, and prints:
  • Overall accuracy
  • Confusion matrix (rows = true persona, columns = predicted pattern)
  • Per-class precision, recall, F1, and support

Usage
─────
  python evaluate.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys
import pathlib
import pandas as pd
import numpy as np

# ── Pipeline imports ──────────────────────────────────────────────────────────
from src.loader     import load
from src.detector   import detect
from src.metrics    import compute
from src.classifier import classify

# ── Paths ─────────────────────────────────────────────────────────────────────
EVENTS_PATH = pathlib.Path("data/sample_events.csv")
GT_PATH     = pathlib.Path("data/ground_truth.csv")

# ── Display width ──────────────────────────────────────────────────────────────
COL_W = 20   # width for each cell in printed tables


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hline(cols: list[str], col_w: int = COL_W) -> str:
    """Return a horizontal separator line sized for the given columns."""
    return "+" + "+".join("-" * (col_w + 2) for _ in cols) + "+"


def _row(cells: list[str], col_w: int = COL_W) -> str:
    """Format one table row."""
    return "| " + " | ".join(str(c).ljust(col_w) for c in cells) + " |"


def print_confusion_matrix(
    true_labels: pd.Series,
    pred_labels: pd.Series,
    classes: list[str],
) -> None:
    """Pretty-print a confusion matrix with true labels as rows."""
    cm = pd.crosstab(
        true_labels.rename("True \\ Pred"),
        pred_labels.rename(""),
        dropna=False,
    ).reindex(index=classes, columns=classes, fill_value=0)

    header_cols = ["True \\ Pred"] + classes
    print(_hline(header_cols))
    print(_row(header_cols))
    print(_hline(header_cols))
    for true_cls in classes:
        row_vals = [true_cls] + [str(cm.loc[true_cls, c]) if c in cm.columns else "0"
                                  for c in classes]
        print(_row(row_vals))
    print(_hline(header_cols))


def print_per_class_report(
    true_labels: pd.Series,
    pred_labels: pd.Series,
    classes: list[str],
) -> None:
    """
    Compute and print per-class precision, recall, F1, and support.
    Uses the standard TP / (TP+FP) and TP / (TP+FN) formulae.
    """
    rows: list[dict] = []
    for cls in classes:
        tp = int(((true_labels == cls) & (pred_labels == cls)).sum())
        fp = int(((true_labels != cls) & (pred_labels == cls)).sum())
        fn = int(((true_labels == cls) & (pred_labels != cls)).sum())
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        rows.append({
            "class":     cls,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "support":   support,
        })

    # ── Print table ───────────────────────────────────────────────────────
    W = COL_W
    header = ["Class", "Precision", "Recall", "F1", "Support"]
    print(_hline(header))
    print(_row(header))
    print(_hline(header))
    for r in rows:
        print(_row([
            r["class"],
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
            str(r["support"]),
        ]))
    print(_hline(header))

    # ── Macro averages ────────────────────────────────────────────────────
    macro_p  = np.mean([r["precision"] for r in rows])
    macro_r  = np.mean([r["recall"]    for r in rows])
    macro_f1 = np.mean([r["f1"]        for r in rows])
    print(_row(["macro avg",
                f"{macro_p:.4f}", f"{macro_r:.4f}", f"{macro_f1:.4f}", ""]))
    print(_hline(header))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Guard: input files must exist ─────────────────────────────────────
    for p in [EVENTS_PATH, GT_PATH]:
        if not p.exists():
            print(f"[ERROR] Required file not found: {p}")
            print("Run `python data/generate_data.py` first.")
            sys.exit(1)

    print("=" * 70)
    print("  CEREBRO -- Decision Reversal Pattern Detector | Evaluation")
    print("=" * 70)

    # ── 1. Run pipeline ───────────────────────────────────────────────────
    print("\n[1/3] Running pipeline on", EVENTS_PATH, "...")
    events_df    = load(str(EVENTS_PATH))
    reversals_df = detect(events_df)
    metrics_df   = compute(events_df, reversals_df)
    classified   = classify(metrics_df, reversals_df)
    print(f"      {len(events_df):,} events | {len(reversals_df):,} reversals | "
          f"{len(classified)} users classified")

    # ── 2. Load ground truth & merge ──────────────────────────────────────
    print("\n[2/3] Loading ground truth from", GT_PATH, "...")
    gt = pd.read_csv(GT_PATH)
    merged = classified.merge(gt, on="user_id", how="inner")

    if len(merged) == 0:
        print("[ERROR] No matching user_ids between pipeline output and ground truth.")
        sys.exit(1)

    true_labels = merged["persona"]    # ground-truth column from ground_truth.csv
    pred_labels = merged["pattern"]    # predicted column from classifier

    # Canonical class order (alphabetical for readability)
    all_classes = sorted(set(true_labels.unique()) | set(pred_labels.unique()))

    # ── 3. Print results ──────────────────────────────────────────────────
    print("\n[3/3] Results")
    print("-" * 70)

    # Overall accuracy
    accuracy = (true_labels == pred_labels).mean()
    n_correct = (true_labels == pred_labels).sum()
    print(f"\n  Overall accuracy : {accuracy:.4f}  ({n_correct}/{len(merged)} users correct)")

    # Confusion matrix
    print("\n  Confusion matrix (rows = true persona, cols = predicted pattern):\n")
    print_confusion_matrix(true_labels, pred_labels, all_classes)

    # Per-class report
    print("\n  Per-class precision / recall / F1:\n")
    print_per_class_report(true_labels, pred_labels, all_classes)

    # Pattern distribution summary
    print("\n  Predicted pattern distribution:")
    dist = pred_labels.value_counts().sort_index()
    max_bar = 40
    max_count = dist.max()
    for label, count in dist.items():
        bar = "#" * int(count / max_count * max_bar)
        print(f"    {label:<25} {count:>4}  {bar}")

    # Save outputs
    out_dir = pathlib.Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    classified.to_csv(out_dir / "user_metrics_and_patterns.csv", index=False)
    reversals_df.to_csv(out_dir / "detected_reversals.csv", index=False)
    print(f"  Artifacts saved to {out_dir}/ (user_metrics_and_patterns.csv, detected_reversals.csv)")

    print("\n" + "=" * 70)
    print("  Evaluation complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
