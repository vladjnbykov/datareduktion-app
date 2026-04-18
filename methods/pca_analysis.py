from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.decomposition import PCA

from utils.preprocessing import build_analysis_matrix


def build_pca_scores_df(
    scores,
    target_series: pd.Series | None = None,
    target_name: str = "target",
) -> pd.DataFrame:
    """
    Build a DataFrame from PCA scores.

    Parameters
    ----------
    scores : array-like
        PCA-transformed matrix of shape (n_samples, n_components).
    target_series : pd.Series | None
        Optional target labels to append.
    target_name : str
        Name of the target column in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns PC1, PC2, ..., and optionally target column.
    """
    n_components = scores.shape[1]
    columns = [f"PC{i + 1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, columns=columns)

    if target_series is not None:
        scores_df[target_name] = target_series.reset_index(drop=True)

    return scores_df


def compute_loadings(pca_model: PCA, feature_names: list[str]) -> pd.DataFrame:
    """
    Compute PCA loadings.

    Parameters
    ----------
    pca_model : PCA
        Fitted PCA model.
    feature_names : list[str]
        Names of original features.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by feature name with columns PC1, PC2, ...
    """
    component_names = [f"PC{i + 1}" for i in range(pca_model.components_.shape[0])]
    loadings_df = pd.DataFrame(
        pca_model.components_.T,
        index=feature_names,
        columns=component_names,
    )
    loadings_df.index.name = "feature"
    return loadings_df.reset_index()


def compute_explained_variance(pca_model: PCA) -> tuple[list[float], list[float]]:
    """
    Return explained variance ratio and cumulative explained variance.

    Parameters
    ----------
    pca_model : PCA
        Fitted PCA model.

    Returns
    -------
    tuple[list[float], list[float]]
        Explained variance ratio per component, cumulative explained variance.
    """
    explained_variance_ratio = pca_model.explained_variance_ratio_.tolist()
    cumulative_variance = (
        pd.Series(explained_variance_ratio).cumsum().tolist()
    )
    return explained_variance_ratio, cumulative_variance


def run_pca(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_components: int = 2,
    standardize: bool = True,
    target_series: pd.Series | None = None,
    target_name: str = "target",
) -> dict[str, Any]:
    """
    Run PCA on selected features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_cols : list[str]
        Selected feature columns.
    n_components : int
        Number of PCA components.
    standardize : bool
        Whether to standardize features before PCA.
    target_series : pd.Series | None
        Optional target labels to append to scores_df.
    target_name : str
        Name of the target column in scores_df.

    Returns
    -------
    dict[str, Any]
        Dictionary containing processed matrix, scores, explained variance,
        cumulative variance, loadings, feature names, and fitted PCA model.
    """
    matrix_payload = build_analysis_matrix(
        df=df,
        feature_cols=feature_cols,
        standardize=standardize,
    )

    X_processed = matrix_payload["X"]
    feature_names = matrix_payload["feature_names"]

    pca_model = PCA(n_components=n_components)
    scores = pca_model.fit_transform(X_processed)

    explained_variance_ratio, cumulative_variance = compute_explained_variance(
        pca_model
    )
    loadings_df = compute_loadings(pca_model, feature_names)
    scores_df = build_pca_scores_df(
        scores=scores,
        target_series=target_series,
        target_name=target_name,
    )

    return {
        "X_processed": X_processed,
        "scores": scores,
        "scores_df": scores_df,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_variance": cumulative_variance,
        "components_": pca_model.components_,
        "loadings_df": loadings_df,
        "feature_names": feature_names,
        "pca_model": pca_model,
    }