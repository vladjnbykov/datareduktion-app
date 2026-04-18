from __future__ import annotations

from typing import Any

import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine

from config import PENGUINS_FEATURE_COLUMNS, PENGUINS_TARGET_COLUMN, WINE_TARGET_NAMES


def load_wine_data() -> dict[str, Any]:
    """Load the Wine dataset and return a structured payload."""
    wine = load_wine(as_frame=True)
    df = wine.frame.copy()

    feature_columns = list(wine.feature_names)
    target_column = "target"

    df["target_name"] = df[target_column].map(WINE_TARGET_NAMES)

    description = (
        "Wine-datasetet innehåller tre olika typer av vin (Wine A, B och C), "
        "beskrivna med hjälp av kemiska egenskaper. "
        "Målet är att se om PCA kan separera dessa grupper."
    )

    return {
        "df": df,
        "feature_columns": feature_columns,
        "target_column": target_column,
        "target_name_column": "target_name",
        "target_names": list(WINE_TARGET_NAMES.values()),
        "description": description,
    }


def load_penguins_data() -> dict[str, Any]:
    """Load the Penguins dataset, keep relevant columns, and drop missing rows."""
    df = sns.load_dataset("penguins").copy()

    required_columns = PENGUINS_FEATURE_COLUMNS + [PENGUINS_TARGET_COLUMN]
    df = df[required_columns].dropna().reset_index(drop=True)

    description = (
        "Penguins-datasetet innehåller kroppsmått för pingviner. "
        "Det är tabulär numerisk data med tydliga grupper och passar bra för UMAP."
    )

    return {
        "df": df,
        "feature_columns": PENGUINS_FEATURE_COLUMNS,
        "target_column": PENGUINS_TARGET_COLUMN,
        "description": description,
    }


def get_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return a compact dataset summary."""
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values_total": int(df.isna().sum().sum()),
    }