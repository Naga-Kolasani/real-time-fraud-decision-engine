"""
Preprocessing: train/val/test split + the scaling pipeline.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.load_data import load_raw_data

# Columns that need scaling. Everything else (V1 - V28) is passed through unchanged.
SCALE_COLS = ["Amount", "Time"]

def train_val_test_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """
    Stratified split into train/val/test sets.

    Splitting happens in two steps: first pull off the test set, then pull the
    validation set out of what's left. Both splits are stratified on the target
    column, since fraud is a small minority class and a plain random split risks
    ending up with very few (or zero) fraud rows in val/test.

    Args:
        df: Full dataset, including the target column.
        target_col: Name of the target/label column (e.g. "Class").
        test_size: Fraction of the *full* dataset to hold out for the test set.
        val_size: Fraction of the *full* dataset to hold out for the validation set.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' not found in dataframe columns")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Step 1: split off the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Step 2: split val out of the remaining data.
    # val_size is expressed as a fraction of the *original* dataset, so converting it to a fraction of what's left (X_temp) before splitting again.
    remaining_fraction = 1 - test_size
    val_fraction_of_remaining = val_size / remaining_fraction

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_fraction_of_remaining,
        stratify=y_temp,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessing_pipeline(feature_columns) -> Pipeline:
    """
    Building the preprocessing pipeline: StandardScaler on Amount/Time, everything else is passed through unchanged.

    Args:
        feature_columns: Column names of the feature matrix (i.e., X_train.columns).
            Used to figure out which columns are "everything else" (V1 - V28, or whatever isn't in SCALE_COLS).

    Returns:
        An unfit sklearn Pipeline.
    """
    missing = [c for c in SCALE_COLS if c not in feature_columns]
    if missing:
        raise ValueError(
            f"Expected columns {missing} not found in feature_columns. "
            f"Update SCALE_COLS if your column names differ."
        )

    passthrough_cols = [c for c in feature_columns if c not in SCALE_COLS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), SCALE_COLS),
            ("passthrough", "passthrough", passthrough_cols),
        ]
    )

    pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

    # Keeping output as a DataFrame with real column names instead of a bare NumPy array, makes debugging and SHAP (later) much less annoying.
    pipeline.set_output(transform="pandas")

    return pipeline

def apply_pipeline(pipeline: Pipeline, X_train, X_val, X_test):
    """
    Fitting the pipeline on X_train and using it to transform train/val/test.

    Fitting only happens on X_train - val and test should never influence the scaler, or else information from those sets could leak into the training.

    Args:
        pipeline: Output of build_preprocessing_pipeline(), unfit.
        X_train, X_val, X_test: Feature matrices from train_val_test_split().

    Returns:
        X_train_t, X_val_t, X_test_t - transformed versions of each input.
        The same `pipeline`, now fit, is mutated in place (fit_transform fits it).
    """
    X_train_t = pipeline.fit_transform(X_train)
    X_val_t = pipeline.transform(X_val)
    X_test_t = pipeline.transform(X_test)

    return X_train_t, X_val_t, X_test_t

if __name__ == "__main__":
    # Quick manual check: `python -m src.data.preprocess`
    df = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df, target_col="Class"
    )

    print(f"Train: {X_train.shape[0]} rows, fraud rate {y_train.mean():.4%}")
    print(f"Val:   {X_val.shape[0]} rows, fraud rate {y_val.mean():.4%}")
    print(f"Test:  {X_test.shape[0]} rows, fraud rate {y_test.mean():.4%}")