"""
Compare recent logged transaction features with the training reference.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.load_data import load_raw_data

PREDICTION_LOG_PATH = Path("artifacts/logs/predictions.jsonl")
REPORT_PATH = Path("artifacts/metrics/drift_report.json")

FEATURES_TO_CHECK = ["Amount", "V1", "V2", "V3", "V4"]
N_BINS = 10
PSI_DRIFT_THRESHOLD = 0.20


def load_logged_transactions(log_path: Path = PREDICTION_LOG_PATH) -> pd.DataFrame:
    """ Load transaction values from the JSONL prediction log. """
    if not log_path.exists():
        raise FileNotFoundError(
            f"No prediction log found at {log_path}. "
            "Score transactions before running a drift check."
        )

    records: list[dict] = []
    with log_path.open(encoding="utf-8") as log_file:
        for line in log_file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError("Prediction log is empty.")

    return pd.DataFrame([record["transaction"] for record in records])


def population_stability_index(
        reference: pd.Series,
        current: pd.Series,
        n_bins: int = N_BINS,
) -> float:
    """ Calculate PSI using quantile bins based on the reference distribution. """
    reference_values = reference.dropna().to_numpy()
    current_values = current.dropna().to_numpy()

    if len(reference_values) == 0 or len(current_values) == 0:
        return float("nan")

    bin_edges = np.unique(
        np.quantile(reference_values, np.linspace(0, 1, n_bins + 1))
    )

    if len(bin_edges) < 2:
        return 0.0

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference_values, bins=bin_edges)
    current_counts, _ = np.histogram(current_values, bins=bin_edges)

    reference_share = np.clip(reference_counts / reference_counts.sum(), 1e-6, None)
    current_share = np.clip(current_counts / current_counts.sum(), 1e-6, None)

    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def build_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> dict:
    """ Build feature-level PSI and mean-comparison results. """
    feature_results: dict[str, dict] = {}

    for feature in FEATURES_TO_CHECK:
        if feature not in reference_df or feature not in current_df:
            continue

        psi = population_stability_index(reference_df[feature], current_df[feature])

        feature_results[feature] = {
            "psi": round(psi, 6),
            "reference_mean": round(float(reference_df[feature].mean()), 6),
            "current_mean": round(float(current_df[feature].mean()), 6),
            "drift_detected": bool(psi >= PSI_DRIFT_THRESHOLD),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_rows": int(len(reference_df)),
        "current_rows": int(len(current_df)),
        "psi_drift_threshold": PSI_DRIFT_THRESHOLD,
        "features": feature_results,
    }


def main() -> None:
    """ Run the local drift check and save its JSON report. """
    reference_df = load_raw_data().drop(columns=["Class"])
    current_df = load_logged_transactions()
    report = build_drift_report(reference_df, current_df)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Drift report saved to {REPORT_PATH}")
    print(
        f"Compared {report['current_rows']} logged transactions against "
        f"{report['reference_rows']} reference rows."
    )

    for feature, result in report["features"].items():
        status = "DRIFT" if result["drift_detected"] else "OK"
        print(
            f"{feature}: PSI={result['psi']:.4f} "
            f"(reference_mean={result['reference_mean']:.4f}, "
            f"current_mean={result['current_mean']:.4f}) [{status}]"
        )


if __name__ == "__main__":
    main()