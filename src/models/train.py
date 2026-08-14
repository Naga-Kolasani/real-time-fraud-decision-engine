"""
Trains:
  - LogisticRegression (class_weight="balanced" to deal with the imbalance)
  - XGBoost (reasonable default params - not tuned)

Both get evaluated on the same held-out test set. Whichever has the better PR-AUC gets saved to disk along with the fitted preprocessing pipeline, since both together are needed to actually score a new transaction later.

MLflow tracking is local for now: params, metrics, confusion matrices, and the winning model artifact.
"""

import json
import os
from datetime import datetime

import joblib
import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

from src.data.load_data import load_raw_data
from src.data.preprocess import (
    train_val_test_split,
    build_preprocessing_pipeline,
    apply_pipeline,
)

TARGET_COL = "Class"
MODEL_DIR = "artifacts/models"
MODEL_PATH = os.path.join(MODEL_DIR, "model_v1.pkl")
METRICS_DIR = "artifacts/metrics"

MLFLOW_EXPERIMENT_NAME = "fraud-decision-engine"


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Scores a fitted model on the test set. Returns a dict of metrics so results from multiple models can be compared side by side.

    Using a plain 0.5 threshold here for precision/recall/F1 - actual decision thresholds (T1/T2 for approve/review/block) are for later.
    """
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_scores),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm

    return metrics


def print_metrics_table(all_metrics: list[dict]) -> None:
    """ Printing a simple side-by-side comparison table of all trained models. """
    summary = pd.DataFrame(
        [
            {
                "model": m["model"],
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "pr_auc": round(m["pr_auc"], 4),
            }
            for m in all_metrics
        ]
    )
    print("\n=== Model comparison (test set) ===")
    print(summary.to_string(index=False))

    for m in all_metrics:
        print(f"\nConfusion matrix - {m['model']}")
        print("              pred_non_fraud  pred_fraud")
        cm = m["confusion_matrix"]
        print(f"actual_non_fraud   {cm[0][0]:>10}   {cm[0][1]:>10}")
        print(f"actual_fraud       {cm[1][0]:>10}   {cm[1][1]:>10}")


def save_confusion_matrix_json(model_name: str, cm) -> str:
    """ Dumps a confusion matrix to JSON so it can be logged as an MLflow artifact. """
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, f"{model_name}_confusion_matrix.json")
    with open(path, "w") as f:
        json.dump(cm.tolist(), f, indent=2)
    return path


def train_logistic_regression(X_train, y_train):
    # params kept in one dict so what's used to build the model and what gets logged to MLflow can't drift apart.
    params = {
        "model_name": "LogisticRegression",
        "class_weight": "balanced",
        "max_iter": 1000,
        "random_state": 42
    }
    model = LogisticRegression(
        class_weight=params["class_weight"],
        max_iter=params["max_iter"],
        random_state=params["random_state"],
    )
    model.fit(X_train, y_train)
    return model, params


def train_xgboost(X_train, y_train):
    # scale_pos_weight roughly balances classes for XGBoost - ratio of negative to positive examples in the training set.
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    params = {
        "model_name": "XGBoost",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "random_state": 42,
    }
    model = XGBClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        scale_pos_weight=params["scale_pos_weight"],
        eval_metric="aucpr",
        random_state=params["random_state"],
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, params


def main():
    print("Loading data...")
    df = load_raw_data()

    print("Splitting train/val/test...")
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, target_col=TARGET_COL
    )

    print("Building + fitting preprocessing pipeline...")
    pipeline = build_preprocessing_pipeline(X_train.columns)
    X_train_t, X_val_t, X_test_t = apply_pipeline(pipeline, X_train, X_val, X_test)

    print("Training LogisticRegression...")
    log_reg, log_reg_params = train_logistic_regression(X_train_t, y_train)

    print("Training XGBoost...")
    xgb_model, xgb_params = train_xgboost(X_train_t, y_train)

    print("Evaluating on test set...")
    all_metrics = [
        evaluate_model(log_reg, X_test_t, y_test, "LogisticRegression"),
        evaluate_model(xgb_model, X_test_t, y_test, "XGBoost"),
    ]
    print_metrics_table(all_metrics)

    # Pick the winner by PR-AUC - the right metric here given how imbalanced the classes are (plain accuracy or ROC-AUC would be misleading).
    # Deciding this now, before any MLflow run opens, so the winner's nested run can log the joblib artifact without ever needing to reopen a run.
    models_by_name = {"LogisticRegression": log_reg, "XGBoost": xgb_model}
    params_by_name = {"LogisticRegression": log_reg_params, "XGBoost": xgb_params}
    best = max(all_metrics, key=lambda m: m["pr_auc"])
    best_model = models_by_name[best["model"]]

    print(f"\nBest model: {best['model']} (PR-AUC = {best['pr_auc']:.4f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    artifact = {
        "pipeline": pipeline,
        "model": best_model,
        "model_name": best["model"],
        "feature_columns": list(X_train.columns),
        "trained_at": datetime.now().isoformat(),
        "test_metrics": {
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "pr_auc": best["pr_auc"],
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved best model + pipeline to {MODEL_PATH}")

    # MLflow logging - doesn't touch the artifact above or any of the prints, just records what happened.
    # One nested run per model, both logged the same way; the winner's run additionally gets the joblib artifact while it's still open.
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="train_models"):
        for model_name, metrics in zip(
            ("LogisticRegression", "XGBoost"), all_metrics
        ):
            params = params_by_name[model_name]

            with mlflow.start_run(run_name=model_name, nested=True):
                mlflow.log_params(params)
                mlflow.log_metrics(
                    {
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "pr_auc": metrics["pr_auc"],
                    }
                )

                cm_path = save_confusion_matrix_json(
                    model_name, metrics["confusion_matrix"]
                )
                mlflow.log_artifact(cm_path)

                if model_name == best["model"]:
                    mlflow.log_artifact(MODEL_PATH)


if __name__ == "__main__":
    main()