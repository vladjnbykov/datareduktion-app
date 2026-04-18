from __future__ import annotations

import streamlit as st

from components.controls import render_pca_controls
from components.dataset_view import (
    render_class_distribution,
    render_data_preview,
    render_dataset_summary,
    render_feature_list,
)
from components.interpretation import (
    render_pca_interpretation,
    render_reflection_questions,
)
from components.intro import (
    render_dataset_intro,
    render_method_intro,
    render_note_box,
    render_page_title,
)
from components.plots import (
    plot_correlation_heatmap,
    plot_cumulative_variance,
    plot_loadings_bar,
    plot_pca_scatter,
    plot_scree,
)
from config import METHOD_DESCRIPTIONS, PCA_DATASET_NAME
from data.loaders import load_wine_data
from methods.pca_analysis import run_pca
from utils.validators import (
    validate_components_against_features,
    validate_min_selected_features,
)


def render_pca_page() -> None:
    """
    Render the PCA page.
    """
    render_page_title("PCA – Principal Component Analysis")
    render_method_intro("PCA", METHOD_DESCRIPTIONS["PCA"])

    payload = load_wine_data()
    df = payload["df"]
    feature_columns = payload["feature_columns"]
    target_column = payload["target_name_column"]
    description = payload["description"]

    render_dataset_intro(PCA_DATASET_NAME, description)
    render_dataset_summary(df)
    render_feature_list(feature_columns)
    render_data_preview(df[feature_columns + [target_column]])
    render_class_distribution(df, target_column)

    left_col, right_col = st.columns([1, 2])

    with left_col:
        controls = render_pca_controls(feature_columns)

    selected_features = controls["selected_features"]
    standardize = controls["standardize"]
    n_components = controls["n_components"]
    show_loadings = controls["show_loadings"]
    show_corr_matrix = controls["show_corr_matrix"]

    try:
        validate_min_selected_features(selected_features, min_required=2)
        validate_components_against_features(n_components, len(selected_features))
    except ValueError as exc:
        with right_col:
            render_note_box(str(exc), kind="error")
        return

    results = run_pca(
        df=df,
        feature_cols=selected_features,
        n_components=n_components,
        standardize=standardize,
        target_series=df[target_column],
        target_name=target_column,
    )

    pc1_var = results["explained_variance_ratio"][0] * 100
    pc2_var = (
        results["explained_variance_ratio"][1] * 100
        if len(results["explained_variance_ratio"]) > 1
        else 0.0
    )

    with right_col:
        st.markdown("## Resultat")

        plot_pca_scatter(
            results["scores_df"],
            target_col=target_column,
            pc1_label=f"PC1 ({pc1_var:.2f}%)",
            pc2_label=f"PC2 ({pc2_var:.2f}%)",
        )

        plot_scree(results["explained_variance_ratio"])
        plot_cumulative_variance(results["cumulative_variance"])

        if show_loadings:
            plot_loadings_bar(results["loadings_df"], pcs=("PC1", "PC2"))
            st.dataframe(results["loadings_df"], width="stretch")

        if show_corr_matrix:
            plot_correlation_heatmap(df[selected_features])

    render_pca_interpretation(results, standardize=standardize)
    render_reflection_questions("PCA")