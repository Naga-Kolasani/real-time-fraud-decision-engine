"""
Threshold sweep on the validation split.

This script loads the saved model artifact, scores the validation set, and shows how precision and recall move as the decision threshold changes.

The goal here is not to auto-pick the final thresholds.
It's just a quick way to narrow down reasonable values for T1 (approve/review) and T2 (review/block) before wiring them into infer.py.
"""

import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from src.data.load_data import load_raw_data
from src.data.preprocess import train_val_test_split

MODEL_PATH = "artifacts/models/model_v1.pkl"
TARGET_COL = "Class"
PLOT_PATH = "artifacts/metrics/threshold_sweep.png"

# Starting targets for the sweep.
# T1 is the lower boundary (approve vs. review).
# T2 is the upper boundary (review vs. block).
MIN_RECALL_FOR_T1 = 0.98
MIN_PRECISION_FOR_T2 = 0.90


def sweep_thresholds(y_true, y_scores, step: float = 0.01) -> pd.DataFrame:
    """
    Building a simple threshold sweep table on the validation scores.

    At each threshold, treat `risk_score >= threshold` as the positive class and record the resulting precision and recall.
    This is just for analysis here.
    The actual inference path still uses two thresholds, not one.
    """
    thresholds = np.arange(0.0, 1.0 + step, step)
    rows = []
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        # skip degenerate thresholds where nothing (or everything) is predicted
        # positive - precision is undefined (0/0) at the extremes
        if y_pred.sum() == 0 and t > 0:
            precision = 1.0  # nothing flagged, no false positives by definition
        else:
            precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        rows.append({"threshold": round(t, 2), "precision": precision, "recall": recall})

    return pd.DataFrame(rows)


def suggest_thresholds(sweep_df: pd.DataFrame) -> tuple[float, float]:
    """
    Suggests a starting T1/T2 pair from the sweep table.

    T1 is the lower cutoff between approve and review.
    T2 is the upper cutoff between review and block.

    This is only meant to narrow things down.
    The final choice still depends on how much review volume feels reasonable and how strict I want the auto-block boundary to be.
    """
    t1_candidates = sweep_df[sweep_df["recall"] >= MIN_RECALL_FOR_T1]
    t1 = t1_candidates["threshold"].max() if not t1_candidates.empty else 0.0

    t2_candidates = sweep_df[sweep_df["precision"] >= MIN_PRECISION_FOR_T2]
    t2 = t2_candidates["threshold"].min() if not t2_candidates.empty else 1.0

    if t1 >= t2:
        print(
            f"Warning: T1 ({t1:.2f}) is not below T2 ({t2:.2f}).  "
            f"With the current targets, there isn't really a usable review band."
        )

    return t1, t2


def plot_sweep(sweep_df: pd.DataFrame, t1: float, t2: float, path: str = PLOT_PATH) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision")
    ax.plot(sweep_df["threshold"], sweep_df["recall"], label="Recall")
    ax.axvline(t1, color="green", linestyle="--", label=f"T1 = {t1:.2f}")
    ax.axvline(t2, color="red", linestyle="--", label=f"T2 = {t2:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall vs. Threshold (validation set)")
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    print(f"Saved plot to {path}")
    plt.show()


def main():
    print("Loading validation split...")
    df = load_raw_data()
    # Same random_state/split as train.py, so X_val here matches what was held out during training - the model has never seen these rows.
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, target_col=TARGET_COL
    )

    print(f"Loading model artifact: {MODEL_PATH}")
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    model = artifact["model"]

    print("Scoring validation set...")
    X_val_t = pipeline.transform(X_val)
    y_scores = model.predict_proba(X_val_t)[:, 1]

    print("Running threshold sweep...")
    sweep_df = sweep_thresholds(y_val, y_scores)

    t1, t2 = suggest_thresholds(sweep_df)
    print(f"\nSuggested T1 (approve/review) = {t1:.2f}")
    print(f"Suggested T2 (review/block)   = {t2:.2f}")

    review_band_pct = (
        (y_scores >= t1) & (y_scores < t2)
    ).sum() / len(y_scores) * 100
    print(f"Review band size: ~ {review_band_pct:.2f}% of validation rows")

    print("\nSweep table around the suggested thresholds:")
    nearby = sweep_df[
        (sweep_df["threshold"] >= max(0, t1 - 0.05))
        & (sweep_df["threshold"] <= min(1, t2 + 0.05))
    ]
    print(nearby.to_string(index=False))

    plot_sweep(sweep_df, t1, t2)


if __name__ == "__main__":
    main()