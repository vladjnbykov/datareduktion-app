from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import streamlit as st


def plot_pca_scatter(
    scores_df: pd.DataFrame,
    target_col: str,
    pc1_label: str = "PC1",
    pc2_label: str = "PC2",
) -> None:
    """
    Render a 2D PCA scatter plot using Plotly.
    """
    fig = px.scatter(
        scores_df,
        x="PC1",
        y="PC2",
        color=target_col,
        title="PCA: PC1 vs PC2",
        labels={
            "PC1": pc1_label,
            "PC2": pc2_label,
            target_col: target_col,
        },
    )
    fig.update_layout(legend_title_text=target_col)
    st.plotly_chart(fig, width="stretch")


def plot_scree(explained_variance_ratio: Sequence[float]) -> None:
    """
    Render a scree plot for PCA explained variance.
    """
    components = list(range(1, len(explained_variance_ratio) + 1))
    values = [v * 100 for v in explained_variance_ratio]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(components, values, marker="o")
    ax.set_title("Scree Plot")
    ax.set_xlabel("Komponent")
    ax.set_ylabel("Förklarad varians (%)")
    ax.set_xticks(components)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)


def plot_cumulative_variance(cumulative_variance: Sequence[float]) -> None:
    """
    Render cumulative explained variance plot.
    """
    components = list(range(1, len(cumulative_variance) + 1))
    values = [v * 100 for v in cumulative_variance]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(components, values, marker="o")
    ax.set_title("Kumulativ förklarad varians")
    ax.set_xlabel("Antal komponenter")
    ax.set_ylabel("Kumulativ varians (%)")
    ax.set_xticks(components)
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)


def plot_loadings_bar(
    loadings_df: pd.DataFrame,
    pcs: tuple[str, str] = ("PC1", "PC2"),
) -> None:
    """
    Render PCA loadings as grouped bar chart.
    """
    cols_needed = ["feature", *pcs]
    plot_df = loadings_df[cols_needed].copy()

    long_df = plot_df.melt(
        id_vars="feature",
        value_vars=list(pcs),
        var_name="Komponent",
        value_name="Loading",
    )

    fig = px.bar(
        long_df,
        x="feature",
        y="Loading",
        color="Komponent",
        barmode="group",
        title="Variabelbidrag (loadings)",
    )
    fig.update_layout(xaxis_title="Feature", yaxis_title="Loading")
    st.plotly_chart(fig, width="stretch")


def plot_correlation_heatmap(df_features: pd.DataFrame) -> None:
    """
    Render a correlation heatmap using matplotlib.
    """
    corr = df_features.corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, aspect="auto")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Korrelationsmatris")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)


def plot_umap_2d(embedding_df: pd.DataFrame, target_col: str) -> None:
    """
    Render a 2D UMAP plot using Plotly.
    """
    fig = px.scatter(
        embedding_df,
        x="UMAP1",
        y="UMAP2",
        color=target_col,
        title="UMAP: 2D-representation",
        labels={
            "UMAP1": "UMAP1",
            "UMAP2": "UMAP2",
            target_col: target_col,
        },
    )
    fig.update_layout(legend_title_text=target_col)
    st.plotly_chart(fig, width="stretch")


def plot_umap_3d(embedding_df: pd.DataFrame, target_col: str) -> None:
    """
    Render a 3D UMAP plot using Plotly.
    """
    fig = px.scatter_3d(
        embedding_df,
        x="UMAP1",
        y="UMAP2",
        z="UMAP3",
        color=target_col,
        title="UMAP: 3D-representation",
        labels={
            "UMAP1": "UMAP1",
            "UMAP2": "UMAP2",
            "UMAP3": "UMAP3",
            target_col: target_col,
        },
    )
    fig.update_layout(legend_title_text=target_col)
    st.plotly_chart(fig, width="stretch")


def plot_ca_biplot(row_coords: pd.DataFrame, col_coords: pd.DataFrame) -> None:
    """
    Render a CA biplot with row and column categories.
    Expects DataFrames with columns:
    - row_coords: row_category, Dim1, Dim2
    - col_coords: column_category, Dim1, Dim2
    """
    fig = px.scatter(
        title="Korrespondensanalys: biplot"
    )

    if {"row_category", "Dim1", "Dim2"}.issubset(row_coords.columns):
        fig.add_scatter(
            x=row_coords["Dim1"],
            y=row_coords["Dim2"],
            mode="markers+text",
            text=row_coords["row_category"],
            textposition="top center",
            name="Rader",
            marker=dict(
                size=10,
                symbol="circle",
                color="blue"
            ),
)

    if {"column_category", "Dim1", "Dim2"}.issubset(col_coords.columns):
        fig.add_scatter(
            x=col_coords["Dim1"],
            y=col_coords["Dim2"],
            mode="markers+text",
            text=col_coords["column_category"],
            textposition="top center",
            name="Kolumner",
            marker=dict(
                size=12,
                symbol="diamond",
                color="red"
            ),
)
    fig.update_layout(
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
    )
    fig.add_hline(y=0, line_dash="dash", opacity=0.5)
    fig.add_vline(x=0, line_dash="dash", opacity=0.5)
    st.plotly_chart(fig, width="stretch")


def plot_inertia_bar(explained_inertia: Sequence[float]) -> None:
    """
    Render explained inertia per CA dimension.
    """
    components = [f"Dim{i + 1}" for i in range(len(explained_inertia))]
    values = [v * 100 for v in explained_inertia]

    fig = px.bar(
        x=components,
        y=values,
        title="Förklarad inertia per dimension",
        labels={"x": "Dimension", "y": "Inertia (%)"},
    )
    st.plotly_chart(fig, width="stretch")