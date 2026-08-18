"""
Inference helpers for the saved fraud model.

Loads the model artifact, scores one transaction, and maps the fraud probability to approve / review / block using two thresholds.
"""

from datetime import datetime, timezone

import joblib
import pandas as pd

MODEL_PATH = "artifacts/models/model_v1.pkl"

# Working thresholds chosen from the validation sweep.
# risk_score < T1          -> approve
# T1 <= risk_score < T2    -> review
# risk_score >= T2         -> block
T1 = 0.10
T2 = 0.89

# Keeping versioning simple for now.
MODEL_VERSION = "v1"

_artifact = None    # Simple cache so the artifact is only loaded once per process


def load_model_artifact(path: str = MODEL_PATH) -> dict:
    """
    Load the saved pipeline + model artifact.

    This is cached after the first call so the API doesn't keep reloading the same file from disk on every request.
    """
    global _artifact
    if _artifact is None:
        _artifact = joblib.load(path)
    return _artifact


def _build_feature_row(transaction: dict, feature_columns: list) -> pd.DataFrame:
    """
    Builds a one-row DataFrame in the same column order used at training time.

    If any required feature is missing, fails early with a clear error instead of letting sklearn blow up later.
    """
    missing = [col for col in feature_columns if col not in transaction]
    if missing:
        raise ValueError(
            f"Transaction is missing required feature(s): {missing}. "
            f"Expected all of: {feature_columns}"
        )

    row = {col: transaction[col] for col in feature_columns}
    return pd.DataFrame([row], columns=feature_columns)


def get_risk_score(transaction: dict, artifact: dict = None) -> float:
    """
    Run one transaction through the saved pipeline and return the risk score.

    This returns the model's positive-class probability, which is what the approve / review / block logic uses downstream.
    """
    if artifact is None:
        artifact = load_model_artifact()

    row_df = _build_feature_row(transaction, artifact["feature_columns"])
    transformed = artifact["pipeline"].transform(row_df)
    risk_score = artifact["model"].predict_proba(transformed)[:, 1][0]
    return float(risk_score)


def decide(risk_score: float, t1: float = T1, t2: float = T2) -> str:
    """
    Map a risk score to approve, review, or block.
    """
    if risk_score < t1:
        return "approve"
    elif risk_score < t2:
        return "review"
    else:
        return "block"


def score_transaction(transaction: dict) -> dict:
    """
    Scores one transaction end to end.

    Returns the raw risk score, the 3-way decision, and a few basic metadata fields that the API can pass through directly.
    """
    artifact = load_model_artifact()

    risk_score = get_risk_score(transaction, artifact=artifact)
    decision = decide(risk_score)

    return {
        "risk_score": round(risk_score, 6),
        "decision": decision,
        "model_name": artifact["model_name"],
        "model_version": MODEL_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_top_features(n: int = 5, artifact: dict = None) -> list[dict]:
    """
    Return the top N global feature importances from the selected tree model.

    These values rank features across the model as a whole. They are not transaction-specific explanations or SHAP contributions.
    """
    if n < 1:
        raise ValueError("n must be at least 1")

    if artifact is None:
        artifact = load_model_artifact()

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            f"Model {artifact['model_name']} does not expose feature_importances_."
        )

    importances = model.feature_importances_
    ranked = sorted(
        zip(feature_columns, importances),
        key=lambda item: item[1],
        reverse=True,
    )

    return [
        {"feature": feature, "importance": round(float(importance), 6)}
        for feature, importance in ranked[:n]
    ]


if __name__ == "__main__":
    # Quick manual check: `python -m src.models.infer`
    from src.data.load_data import load_raw_data

    df = load_raw_data()
    sample_row = df.drop(columns=["Class"]).iloc[0].to_dict()

    result = score_transaction(sample_row)
    print("Sample score:")
    for key, value in result.items():
        print(f"  {key}: {value}")