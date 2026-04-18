from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.preprocessing import StandardScaler


def subset_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Return only the selected feature columns."""
    return df[feature_cols].copy()


def drop_missing_for_columns(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """Drop rows with missing values in required columns."""
    return df.dropna(subset=required_cols).reset_index(drop=True)


def standardize_features(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize features to mean 0 and std 1.

    Returns a DataFrame with the same columns/index and the fitted scaler.
    """
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(
        scaled_array,
        columns=df.columns,
        index=df.index,
    )
    return scaled_df, scaler


def to_numpy(df: pd.DataFrame) -> Any:
    """Convert a DataFrame to a NumPy array."""
    return df.to_numpy()


def build_analysis_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
    standardize: bool = True,
) -> dict[str, Any]:
    """
    Build the matrix used for PCA/UMAP.

    Returns:
        {
            "X": pd.DataFrame,
            "feature_names": list[str],
            "scaler": StandardScaler | None,
        }
    """
    X = subset_features(df, feature_cols)

    scaler = None
    if standardize:
        X, scaler = standardize_features(X)

    return {
        "X": X,
        "feature_names": list(X.columns),
        "scaler": scaler,
    }