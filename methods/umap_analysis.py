from __future__ import annotations

from typing import Any
import warnings

import pandas as pd
import umap

from utils.preprocessing import build_analysis_matrix


def build_umap_embedding_df(
    embedding,
    target_series: pd.Series | None = None,
    target_name: str = "target",
) -> pd.DataFrame:
    """
    Build a DataFrame from UMAP embedding.

    Parameters
    ----------
    embedding : array-like
        UMAP embedding of shape (n_samples, n_components).
    target_series : pd.Series | None
        Optional target labels to append.
    target_name : str
        Name of the target column in the returned DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns UMAP1, UMAP2, ..., and optionally target column.
    """
    n_components = embedding.shape[1]
    columns = [f"UMAP{i + 1}" for i in range(n_components)]
    embedding_df = pd.DataFrame(embedding, columns=columns)

    if target_series is not None:
        embedding_df[target_name] = target_series.reset_index(drop=True)

    return embedding_df


def get_umap_params_summary(
    standardize: bool,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    n_components: int,
    random_state: int,
) -> dict[str, Any]:
    """
    Return a summary of UMAP parameters used.
    """
    return {
        "standardize": standardize,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "n_components": n_components,
        "random_state": random_state,
    }


def run_umap(
    df: pd.DataFrame,
    feature_cols: list[str],
    standardize: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    n_components: int = 2,
    random_state: int = 42,
    target_series: pd.Series | None = None,
    target_name: str = "target",
) -> dict[str, Any]:
    """
    Run UMAP on selected features.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_cols : list[str]
        Selected feature columns.
    standardize : bool
        Whether to standardize features before UMAP.
    n_neighbors : int
        Number of neighbors in local neighborhood graph.
    min_dist : float
        Minimum distance between embedded points.
    metric : str
        Distance metric used by UMAP.
    n_components : int
        Number of output dimensions.
    random_state : int
        Random seed for reproducibility.
    target_series : pd.Series | None
        Optional target labels to append to embedding_df.
    target_name : str
        Name of the target column in embedding_df.

    Returns
    -------
    dict[str, Any]
        Dictionary containing processed matrix, embedding, embedding_df,
        feature names, parameters, and fitted UMAP model.
    """
    matrix_payload = build_analysis_matrix(
        df=df,
        feature_cols=feature_cols,
        standardize=standardize,
    )

    X_processed = matrix_payload["X"]
    feature_names = matrix_payload["feature_names"]

    umap_model = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
    )

    
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="n_jobs value 1 overridden to 1 by setting random_state.*",
        )
        embedding = umap_model.fit_transform(X_processed)

    embedding_df = build_umap_embedding_df(
        embedding=embedding,
        target_series=target_series,
        target_name=target_name,
    )

    params = get_umap_params_summary(
        standardize=standardize,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        random_state=random_state,
    )

    return {
        "X_processed": X_processed,
        "embedding": embedding,
        "embedding_df": embedding_df,
        "feature_names": feature_names,
        "params": params,
        "umap_model": umap_model,
    }