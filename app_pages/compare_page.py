from __future__ import annotations

import streamlit as st

from components.controls import render_compare_controls
from components.dataset_view import (
    render_data_preview,
    render_dataset_summary,
    render_feature_list,
)
from components.interpretation import (
    render_compare_reflection_questions,
    render_compare_summary_box,
)
from components.intro import (
    render_dataset_intro,
    render_note_box,
    render_page_title,
)
from components.plots import (
    plot_pca_scatter,
    plot_umap_2d,
)
from data.loaders import load_penguins_data
from methods.pca_analysis import run_pca
from methods.umap_analysis import run_umap
from utils.validators import (
    validate_min_selected_features,
)


def render_compare_page() -> None:
    """
    Render side-by-side comparison of PCA and UMAP on Penguins.
    """
    render_page_title("Jämför PCA och UMAP")
    st.write(
        """
Här används samma dataset i två olika metoder.
Målet är inte att hitta en absolut vinnare, utan att förstå skillnader
i struktur, tolkbarhet och parameterkänslighet.
        """
    )

    payload = load_penguins_data()
    df = payload["df"]
    feature_columns = payload["feature_columns"]
    target_column = payload["target_column"]
    description = payload["description"]

    render_dataset_intro("Penguins", description)
    render_dataset_summary(df)
    render_feature_list(feature_columns)
    render_data_preview(df[feature_columns + [target_column]])

    controls = render_compare_controls(feature_columns)

    selected_features = controls["selected_features"]
    standardize = controls["standardize"]
    n_neighbors = controls["n_neighbors"]
    min_dist = controls["min_dist"]
    metric = controls["metric"]
    random_state = controls["random_state"]

    try:
        validate_min_selected_features(selected_features, min_required=2)
    except ValueError as exc:
        render_note_box(str(exc), kind="error")
        return

    pca_results = run_pca(
        df=df,
        feature_cols=selected_features,
        n_components=2,
        standardize=standardize,
        target_series=df[target_column],
        target_name=target_column,
    )

    umap_results = run_umap(
        df=df,
        feature_cols=selected_features,
        standardize=standardize,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=2,
        random_state=random_state,
        target_series=df[target_column],
        target_name=target_column,
    )

    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("## PCA")
        plot_pca_scatter(
            pca_results["scores_df"],
            target_col=target_column,
            pc1_label="PC1",
            pc2_label="PC2",
        )

    with right_col:
        st.markdown("## UMAP")
        plot_umap_2d(
            umap_results["embedding_df"],
            target_col=target_column,
        )

    render_compare_summary_box()
    render_compare_reflection_questions()