from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from methods.pca_analysis import run_pca
from utils.preprocessing import build_analysis_matrix


def run_kmeans(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 3,
    standardize: bool = True,
    random_state: int = 42,
    use_pca: bool = False,
    pca_n_components: int | None = None,
) -> dict[str, Any]:
    matrix_payload = build_analysis_matrix(
        df=df,
        feature_cols=feature_cols,
        standardize=standardize,
    )

    X_processed = matrix_payload["X"]
    pca_model = None

    if use_pca:
        from sklearn.decomposition import PCA

        if pca_n_components is None:
            pca_n_components = 2

        pca_model = PCA(n_components=pca_n_components)
        X_processed = pca_model.fit_transform(X_processed)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )

    labels = model.fit_predict(X_processed)

    result_df = df.copy()
    result_df["cluster"] = labels.astype(str)

    sil_score = None
    if n_clusters > 1 and n_clusters < len(X_processed):
        sil_score = silhouette_score(X_processed, labels)

    return {
        "X_processed": X_processed,
        "labels": labels,
        "result_df": result_df,
        "model": model,
        "inertia": model.inertia_,
        "silhouette_score": sil_score,
        "feature_names": feature_cols,
        "used_pca": use_pca,
        "pca_n_components": pca_n_components,
        "pca_model": pca_model,
    }


def compute_elbow_and_silhouette(
    df: pd.DataFrame,
    feature_cols: list[str],
    k_values: list[int],
    standardize: bool = True,
    random_state: int = 42,
    use_pca: bool = False,
    pca_n_components: int | None = None,
) -> pd.DataFrame:
    from sklearn.decomposition import PCA

    rows = []

    matrix_payload = build_analysis_matrix(
        df=df,
        feature_cols=feature_cols,
        standardize=standardize,
    )
    X_processed = matrix_payload["X"]

    if use_pca:
        if pca_n_components is None:
            pca_n_components = 2

        pca_model = PCA(n_components=pca_n_components)
        X_processed = pca_model.fit_transform(X_processed)

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=10,
        )
        labels = model.fit_predict(X_processed)

        sil = None
        if k > 1 and k < len(X_processed):
            sil = silhouette_score(X_processed, labels)

        rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette_score": sil,
            }
        )

    return pd.DataFrame(rows)


def run_pca_for_kmeans_visualization(
    df: pd.DataFrame,
    feature_cols: list[str],
    cluster_labels,
    standardize: bool = True,
) -> pd.DataFrame:
    pca_results = run_pca(
        df=df,
        feature_cols=feature_cols,
        n_components=2,
        standardize=standardize,
    )

    scores_df = pca_results["scores_df"].copy()
    scores_df["cluster"] = cluster_labels.astype(str)

    return scores_df