"""
Loads the raw fraud dataset from disk.

Nothing fancy here on purpose - just a single function that reads the CSV and gives a
clear error if the file isn't where it's expected. Preprocessing (splitting, scaling,
encoding) lives in preprocess.py, not here.
"""

import os
import pandas as pd

# Default path to the raw CSV.
DATA_PATH = "data/creditcard.csv"


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw transaction dataset from a CSV file.

    Args:
        path: Path to the CSV file. Defaults to DATA_PATH.

    Returns:
        DataFrame with the raw, unprocessed data.

    Raises:
        FileNotFoundError: If the file doesn't exist at the given path, with a message
            that points to the likely fix.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find dataset at '{path}'. "
            f"Download it from Kaggle (mlg-ulb/creditcardfraud) and place it there, "
            f"or pass a different path to load_raw_data()."
        )

    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    # Quick manual check: `python src/data/load_data.py`
    data = load_raw_data()
    print(f"Loaded {data.shape[0]} rows, {data.shape[1]} columns from {DATA_PATH}")