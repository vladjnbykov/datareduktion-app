from __future__ import annotations

import pandas as pd
import streamlit as st

from components.controls import render_umap_controls
from components.dataset_view import (
    render_class_distribution,
    render_data_preview,
    render_dataset_summary,
    render_feature_list,
)
from components.interpretation import (
    render_reflection_questions,
    render_umap_interpretation,
)
from components.intro import (
    render_dataset_intro,
    render_method_intro,
    render_note_box,
    render_page_title,
)
from components.plots import (
    plot_umap_2d,
    plot_umap_3d,
)
from config import METHOD_DESCRIPTIONS, UMAP_ALLOWED_METRICS, UMAP_DATASET_NAME
from data.loaders import load_penguins_data
from methods.umap_analysis import run_umap_cached
from utils.validators import (
    validate_components_against_features,
    validate_metric,
    validate_min_selected_features,
)


def render_umap_page() -> None:
    """
    Render the UMAP page.
    """
    render_page_title("UMAP – Uniform Manifold Approximation and Projection")
    render_method_intro("UMAP", METHOD_DESCRIPTIONS["UMAP"])

    payload = load_penguins_data()
    df = payload["df"]
    feature_columns = payload["feature_columns"]
    target_column = payload["target_column"]
    description = payload["description"]

    render_dataset_intro(UMAP_DATASET_NAME, description)
    render_dataset_summary(df)
    render_feature_list(feature_columns)
    render_data_preview(df[feature_columns + [target_column]])
    render_class_distribution(df, target_column)

    left_col, right_col = st.columns([1, 2])

    with left_col:
        controls = render_umap_controls(feature_columns)
        st.caption("Resultatet uppdateras först när du klickar på 'Kör UMAP'.")
        run_clicked = st.button("Kör UMAP", type="primary")

    selected_features = controls["selected_features"]
    standardize = controls["standardize"]
    n_neighbors = controls["n_neighbors"]
    min_dist = controls["min_dist"]
    metric = controls["metric"]
    n_components = controls["n_components"]
    random_state = controls["random_state"]

    try:
        validate_min_selected_features(selected_features, min_required=2)
        validate_components_against_features(n_components, len(selected_features))
        validate_metric(metric, UMAP_ALLOWED_METRICS)
    except ValueError as exc:
        with right_col:
            render_note_box(str(exc), kind="error")
        return

    if run_clicked:
        results = run_umap_cached(
            df=df,
            feature_cols=selected_features,
            standardize=standardize,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_components=n_components,
            random_state=random_state,
            target_series=df[target_column],
            target_name=target_column,
        )
        st.session_state["umap_results"] = results

    stored_results = st.session_state.get("umap_results")

    if stored_results is not None:
        with right_col:
            st.markdown("## Resultat")

            if stored_results["params"]["n_components"] == 2:
                plot_umap_2d(
                    stored_results["embedding_df"],
                    target_col=target_column,
                )
            else:
                plot_umap_3d(
                    stored_results["embedding_df"],
                    target_col=target_column,
                )

            st.markdown("### Använda parametrar")
            params_df = pd.DataFrame(
                {
                    "Parameter": list(stored_results["params"].keys()),
                    "Värde": [
                        f"{v:.2f}" if isinstance(v, float) else str(v)
                        for v in stored_results["params"].values()
                    ],
                }
            )
            st.dataframe(params_df, width="stretch")

        render_umap_interpretation(stored_results["params"])
        render_reflection_questions("UMAP")
    else:
        with right_col:
            render_note_box(
                "Välj parametrar och klicka på **Kör UMAP** för att skapa visualiseringen.",
                kind="info",
            )