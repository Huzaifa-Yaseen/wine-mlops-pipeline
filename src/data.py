"""Data ingestion, validation, and splitting module for Wine dataset."""
from typing import Tuple
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
EXPECTED_FEATURE_COUNT = 13


def load_and_validate_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load Wine dataset and validate shape, columns, and missing values."""
    wine_bunch = load_wine(as_frame=True)
    df = wine_bunch.frame

    features = df.drop(columns=["target"])
    target = df["target"]

    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, found {features.shape[1]}"
        )

    if features.isnull().any().any():
        raise ValueError("Missing/null values detected in feature matrix.")

    if target.isnull().any():
        raise ValueError("Missing/null values detected in target vector.")

    return features, target


def get_train_test_data(
    test_size: float = 0.20, random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified 80/20 train-test split with fixed random seed."""
    x_data, y_data = load_and_validate_data()
    return train_test_split(
        x_data,
        y_data,
        test_size=test_size,
        stratify=y_data,
        random_state=random_state,
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_data()
    print("Data loaded and validated successfully.")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
