import json
import os

import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from src.data.load_data import load_raw_data
from src.data.preprocess import (
    train_val_test_split,
    build_preprocessing_pipeline,
    apply_pipeline,
)

TARGET_COL = "Class"
RESULTS_PATH = "artifacts/metrics/xgboost_tuning_results.json"

CANDIDATES = [
    {"n_estimators": 150, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9, "colsample_bytree": 0.8},
    {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.9},
    {"n_estimators": 250, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "subsample": 0.9, "colsample_bytree": 0.9},
    {"n_estimators": 400, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
    {"n_estimators": 250, "max_depth": 4, "learning_rate": 0.08, "subsample": 0.9, "colsample_bytree": 0.8},
]


def main():
    df = load_raw_data()
    X_train, X_val, _, y_train, y_val, _ = train_val_test_split(
        df, target_col=TARGET_COL
    )

    pipeline = build_preprocessing_pipeline(X_train.columns)
    X_train_t, X_val_t, _ = apply_pipeline(pipeline, X_train, X_val, X_val)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    results = []
    for index, candidate in enumerate(CANDIDATES, start=1):
        params = {
            **candidate,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = XGBClassifier(**params)
        model.fit(X_train_t, y_train)
        pr_auc = average_precision_score(y_val, model.predict_proba(X_val_t)[:, 1])

        result = {
            "candidate": index,
            **candidate,
            "scale_pos_weight": scale_pos_weight,
            "validation_pr_auc": pr_auc,
        }
        results.append(result)
        print(
            f"Candidate {index}/{len(CANDIDATES)}: "
            f"validation PR-AUC = {pr_auc:.4f} | {candidate}"
        )

    results.sort(key=lambda item: item["validation_pr_auc"], reverse=True)
    best = results[0]

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as file:
        json.dump({"best": best, "results": results}, file, indent=2)

    print("\nBest validation configuration:")
    print(json.dumps(best, indent=2))
    print(f"\nSaved results to {RESULTS_PATH}")

    print("\nAll candidates:")
    print(
        pd.DataFrame(results)
        .drop(columns=["scale_pos_weight"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()