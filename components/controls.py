from __future__ import annotations

import streamlit as st

from config import (
    CA_DEFAULT_N_COMPONENTS,
    CA_DEFAULT_VIEW_MODE,
    CA_SHOW_INERTIA_DEFAULT,
    CA_SHOW_PROFILES_DEFAULT,
    PCA_DEFAULT_N_COMPONENTS,
    PCA_DEFAULT_STANDARDIZE,
    PCA_SHOW_CORR_DEFAULT,
    PCA_SHOW_LOADINGS_DEFAULT,
    UMAP_ALLOWED_METRICS,
    UMAP_DEFAULT_METRIC,
    UMAP_DEFAULT_MIN_DIST,
    UMAP_DEFAULT_N_COMPONENTS,
    UMAP_DEFAULT_N_NEIGHBORS,
    UMAP_DEFAULT_RANDOM_STATE,
    UMAP_DEFAULT_STANDARDIZE,
)


def render_pca_controls(feature_cols: list[str]) -> dict[str, object]:
    """
    Render PCA controls and return selected settings.
    """
    st.markdown("## Inställningar")

    selected_features = st.multiselect(
        "Välj features",
        options=feature_cols,
        default=feature_cols,
    )

    standardize = st.checkbox(
        "Standardisera data före PCA",
        value=PCA_DEFAULT_STANDARDIZE,
    )

    max_components = max(len(selected_features), 2)
    n_components = st.slider(
        "Antal komponenter",
        min_value=2,
        max_value=max_components,
        value=min(PCA_DEFAULT_N_COMPONENTS, max_components),
    )

    show_loadings = st.checkbox(
        "Visa variabelbidrag (loadings)",
        value=PCA_SHOW_LOADINGS_DEFAULT,
    )

    show_corr_matrix = st.checkbox(
        "Visa korrelationsmatris",
        value=PCA_SHOW_CORR_DEFAULT,
    )

    return {
        "selected_features": selected_features,
        "standardize": standardize,
        "n_components": n_components,
        "show_loadings": show_loadings,
        "show_corr_matrix": show_corr_matrix,
    }


def render_umap_controls(feature_cols: list[str]) -> dict[str, object]:
    """
    Render UMAP controls and return selected settings.
    """
    st.markdown("## Inställningar")

    selected_features = st.multiselect(
        "Välj features",
        options=feature_cols,
        default=feature_cols,
    )

    standardize = st.checkbox(
        "Standardisera data före UMAP",
        value=UMAP_DEFAULT_STANDARDIZE,
    )

    n_neighbors = st.slider(
        "n_neighbors",
        min_value=2,
        max_value=50,
        value=UMAP_DEFAULT_N_NEIGHBORS,
    )

    min_dist = st.slider(
        "min_dist",
        min_value=0.0,
        max_value=0.99,
        value=float(UMAP_DEFAULT_MIN_DIST),
        step=0.01,
    )

    metric = st.selectbox(
        "metric",
        options=UMAP_ALLOWED_METRICS,
        index=UMAP_ALLOWED_METRICS.index(UMAP_DEFAULT_METRIC),
    )

    n_components = st.selectbox(
        "Antal dimensioner",
        options=[2, 3],
        index=[2, 3].index(UMAP_DEFAULT_N_COMPONENTS),
    )

    random_state = st.number_input(
        "random_state",
        min_value=0,
        value=UMAP_DEFAULT_RANDOM_STATE,
        step=1,
    )

    return {
        "selected_features": selected_features,
        "standardize": standardize,
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": metric,
        "n_components": int(n_components),
        "random_state": int(random_state),
    }


def render_ca_controls() -> dict[str, object]:
    """
    Render CA controls and return selected settings.
    """
    st.markdown("## Inställningar")

    view_mode = st.radio(
        "Visa tabellen som",
        options=["Frekvenser", "Radproportioner", "Kolumnproportioner"],
        index=["Frekvenser", "Radproportioner", "Kolumnproportioner"].index(
            CA_DEFAULT_VIEW_MODE
        ),
    )

    show_inertia = st.checkbox(
        "Visa inertia",
        value=CA_SHOW_INERTIA_DEFAULT,
    )

    show_profiles = st.checkbox(
        "Visa profiler",
        value=CA_SHOW_PROFILES_DEFAULT,
    )

    n_components = st.selectbox(
        "Antal komponenter",
        options=[2],
        index= 0,
    )

    return {
        "view_mode": view_mode,
        "show_inertia": show_inertia,
        "show_profiles": show_profiles,
        "n_components": int(n_components),
    }


def render_compare_controls(feature_cols: list[str]) -> dict[str, object]:
    """
    Render controls for comparison page (PCA vs UMAP on same dataset).
    """
    st.markdown("## Inställningar")

    selected_features = st.multiselect(
        "Välj features",
        options=feature_cols,
        default=feature_cols,
    )

    standardize = st.checkbox(
        "Standardisera data",
        value=True,
    )

    st.markdown("### UMAP-parametrar")

    n_neighbors = st.slider(
        "n_neighbors",
        min_value=2,
        max_value=50,
        value=UMAP_DEFAULT_N_NEIGHBORS,
    )

    min_dist = st.slider(
        "min_dist",
        min_value=0.0,
        max_value=0.99,
        value=float(UMAP_DEFAULT_MIN_DIST),
        step=0.01,
    )

    metric = st.selectbox(
        "metric",
        options=UMAP_ALLOWED_METRICS,
        index=UMAP_ALLOWED_METRICS.index(UMAP_DEFAULT_METRIC),
    )

    random_state = st.number_input(
        "random_state",
        min_value=0,
        value=UMAP_DEFAULT_RANDOM_STATE,
        step=1,
    )

    return {
        "selected_features": selected_features,
        "standardize": standardize,
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "metric": metric,
        "random_state": int(random_state),
    }