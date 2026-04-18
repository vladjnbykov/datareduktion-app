from __future__ import annotations

import pandas as pd
import streamlit as st

from data.loaders import get_dataset_summary
from utils.formatting import round_df


def render_dataset_summary(df: pd.DataFrame, title: str = "Datasetöversikt") -> None:
    """
    Render a compact dataset summary block.
    """
    summary = get_dataset_summary(df)

    st.markdown(f"### {title}")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Antal rader", summary["n_rows"])
    with col2:
        st.metric("Antal kolumner", summary["n_columns"])
    with col3:
        st.metric("Saknade värden", summary["missing_values_total"])


def render_feature_list(feature_cols: list[str], title: str = "Valbara features") -> None:
    """
    Render a simple feature list.
    """
    st.markdown(f"### {title}")
    st.write(", ".join(feature_cols))


def render_data_preview(
    df: pd.DataFrame,
    n_rows: int = 5,
    title: str = "Förhandsvisning av data",
    decimals: int = 3,
) -> None:
    """
    Render a preview of the first n rows of the dataset.
    """
    st.markdown(f"### {title}")
    st.dataframe(round_df(df.head(n_rows), decimals=decimals), width="stretch")


def render_contingency_table(
    table: pd.DataFrame,
    title: str = "Kontingenstabell",
    decimals: int = 3,
) -> None:
    """
    Render a contingency table for CA.
    """
    st.markdown(f"### {title}")
    st.dataframe(round_df(table, decimals=decimals), width="stretch")


def render_class_distribution(
    df: pd.DataFrame,
    target_col: str,
    title: str = "Fördelning av klasser",
) -> None:
    """
    Render class distribution as a small table.
    """
    st.markdown(f"### {title}")
    distribution = (
        df[target_col]
        .value_counts(dropna=False)
        .rename_axis(target_col)
        .reset_index(name="count")
    )
    st.dataframe(distribution, width="stretch")


def render_profiles_table(
    df: pd.DataFrame,
    title: str,
    decimals: int = 3,
) -> None:
    """
    Render row or column profile table.
    """
    st.markdown(f"### {title}")
    st.dataframe(round_df(df, decimals=decimals), width="stretch")