from __future__ import annotations
import warnings
import umap

from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from methods.pca_analysis import run_pca
from utils.preprocessing import build_analysis_matrix
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score


def run_kmeans(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 3,
    standardize: bool = True,
    random_state: int = 42,
    embedding_method: str = "original",
    pca_n_components: int | None = None,
    umap_n_neighbors: int | None = None,
    umap_min_dist: float | None = None,
    umap_metric: str | None = None,
) -> dict[str, Any]:
    matrix_payload = build_analysis_matrix(
        df=df,
        feature_cols=feature_cols,
        standardize=standardize,
    )

    X_processed = matrix_payload["X"]

    pca_model = None
    umap_model = None
    pca_scores_df = None
    umap_scores_df = None

    if embedding_method == "pca":
        from sklearn.decomposition import PCA

        if pca_n_components is None:
            pca_n_components = 2

        pca_model = PCA(n_components=pca_n_components)
        X_pca = pca_model.fit_transform(X_processed)
        X_processed = X_pca

        pca_scores_df = pd.DataFrame(
            X_pca[:, :2],
            columns=["PC1", "PC2"],
        )

    elif embedding_method == "umap":
        if umap_n_neighbors is None:
            umap_n_neighbors = 15
        if umap_min_dist is None:
            umap_min_dist = 0.1
        if umap_metric is None:
            umap_metric = "euclidean"

        umap_model = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            n_components=2,
            random_state=random_state,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="n_jobs value 1 overridden to 1 by setting random_state.*",
            )
            X_umap = umap_model.fit_transform(X_processed)

        X_processed = X_umap

        umap_scores_df = pd.DataFrame(
            X_umap,
            columns=["UMAP1", "UMAP2"],
        )

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
    )

    labels = model.fit_predict(X_processed)

    result_df = df.copy()
    result_df["cluster"] = labels.astype(str)

    if pca_scores_df is not None:
        pca_scores_df["cluster"] = labels.astype(str)

    if umap_scores_df is not None:
        umap_scores_df["cluster"] = labels.astype(str)

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
        "embedding_method": embedding_method,
        "used_pca": embedding_method == "pca",
        "used_umap": embedding_method == "umap",
        "pca_n_components": pca_n_components,
        "pca_model": pca_model,
        "pca_scores_df": pca_scores_df,
        "umap_model": umap_model,
        "umap_scores_df": umap_scores_df,
        "umap_n_neighbors": umap_n_neighbors,
        "umap_min_dist": umap_min_dist,
        "umap_metric": umap_metric,
    }


def compute_elbow_and_silhouette(
    df: pd.DataFrame,
    feature_cols: list[str],
    k_values: list[int],
    standardize: bool = True,
    random_state: int = 42,
    use_pca: bool = False,
    embedding_method: str = "original",
    pca_n_components: int | None = None,
    umap_n_neighbors: int | None = None,
    umap_min_dist: float | None = None,
    umap_metric: str | None = None,
) -> pd.DataFrame:
    from sklearn.decomposition import PCA

    rows = []

    matrix_payload = build_analysis_matrix(
        df=df,
        feature_cols=feature_cols,
        standardize=standardize,
    )
    X_processed = matrix_payload["X"]

    if embedding_method == "pca":
        from sklearn.decomposition import PCA

        if pca_n_components is None:
            pca_n_components = 2

        pca_model = PCA(n_components=pca_n_components)
        X_processed = pca_model.fit_transform(X_processed)

    elif embedding_method == "umap":
        if umap_n_neighbors is None:
            umap_n_neighbors = 15
        if umap_min_dist is None:
            umap_min_dist = 0.1
        if umap_metric is None:
            umap_metric = "euclidean"

        umap_model = umap.UMAP(
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric=umap_metric,
            n_components=2,
            random_state=random_state,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="n_jobs value 1 overridden to 1 by setting random_state.*",
            )
            X_processed = umap_model.fit_transform(X_processed)

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

def compute_hierarchical_linkage(
    df: pd.DataFrame,
    feature_cols: list[str],
    standardize: bool = True,
    linkage_method: str = "ward",
):
    """
    Compute linkage matrix for hierarchical clustering.
    """
    matrix_payload = build_analysis_matrix(
        df=df,
        feature_cols=feature_cols,
        standardize=standardize,
    )

    X_processed = matrix_payload["X"]

    Z = linkage(
        X_processed,
        method=linkage_method,
        metric="euclidean",
    )

    return {
        "linkage_matrix": Z,
        "X_processed": X_processed,
        "feature_names": feature_cols,
    }

def get_cut_height(linkage_matrix, n_clusters: int) -> float | None:
    """
    Estimate a cut height that produces approximately n_clusters.
    """
    distances = linkage_matrix[:, 2]

    if n_clusters < 2:
        return None

    if n_clusters >= len(distances) + 1:
        return None

    lower_idx = len(distances) - n_clusters
    upper_idx = len(distances) - n_clusters + 1

    if lower_idx < 0 or upper_idx >= len(distances):
        return float(distances[0])

    return float((distances[lower_idx] + distances[upper_idx]) / 2)

def run_hierarchical_clustering(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int = 3,
    standardize: bool = True,
    linkage_method: str = "ward",
) -> dict[str, Any]:
    """
    Run hierarchical clustering and assign cluster labels.
    """

    linkage_payload = compute_hierarchical_linkage(
        df=df,
        feature_cols=feature_cols,
        standardize=standardize,
        linkage_method=linkage_method,
    )

    Z = linkage_payload["linkage_matrix"]
    X_processed = linkage_payload["X_processed"]

    labels = fcluster(
        Z,
        t=n_clusters,
        criterion="maxclust",
    )

    # Gör labels till 0,1,2 istället för 1,2,3
    labels = labels - 1

    result_df = df.copy()
    result_df["cluster"] = labels.astype(str)

    sil_score = None
    if n_clusters > 1 and n_clusters < len(X_processed):
        sil_score = silhouette_score(X_processed, labels)
    
    cut_height = get_cut_height(Z, n_clusters)

    return {
        "linkage_matrix": Z,
        "labels": labels,
        "result_df": result_df,
        "silhouette_score": sil_score,
        "feature_names": feature_cols,
        "linkage_method": linkage_method,
        "n_clusters": n_clusters,
        "cut_height": cut_height,
    }