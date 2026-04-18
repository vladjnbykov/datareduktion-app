from __future__ import annotations

from typing import Any

import pandas as pd
import prince


def compute_row_profiles(table: pd.DataFrame) -> pd.DataFrame:
    """
    Compute row profiles by dividing each row by its row sum.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table.

    Returns
    -------
    pd.DataFrame
        Row profile table.
    """
    row_sums = table.sum(axis=1)
    return table.div(row_sums, axis=0)


def compute_col_profiles(table: pd.DataFrame) -> pd.DataFrame:
    """
    Compute column profiles by dividing each column by its column sum.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table.

    Returns
    -------
    pd.DataFrame
        Column profile table.
    """
    col_sums = table.sum(axis=0)
    return table.div(col_sums, axis=1)


def extract_ca_coordinates(
    ca_model: prince.CA,
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract row and column coordinates from fitted CA model.

    Parameters
    ----------
    ca_model : prince.CA
        Fitted correspondence analysis model.
    table : pd.DataFrame
        Original contingency table.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Row coordinates and column coordinates.
    """
    row_coords = ca_model.row_coordinates(table).copy()
    col_coords = ca_model.column_coordinates(table).copy()

    row_coords.columns = [f"Dim{i + 1}" for i in range(row_coords.shape[1])]
    col_coords.columns = [f"Dim{i + 1}" for i in range(col_coords.shape[1])]

    row_coords.index.name = "row_category"
    col_coords.index.name = "column_category"

    return row_coords.reset_index(), col_coords.reset_index()


def extract_ca_inertia(ca_model: prince.CA) -> list[float]:
    """
    Extract explained inertia from the fitted CA model.

    Parameters
    ----------
    ca_model : prince.CA
        Fitted correspondence analysis model.

    Returns
    -------
    list[float]
        Percentage of variance / inertia per dimension as decimals.
    """
    # prince percentage_of_variance_ is typically in percent values like [80.2, 19.8]
    variance = getattr(ca_model, "percentage_of_variance_", None)
    if variance is None:
        return []

    return [v / 100 for v in variance]


def run_ca(table: pd.DataFrame, n_components: int = 2) -> dict[str, Any]:
    """
    Run correspondence analysis on a contingency table.

    Parameters
    ----------
    table : pd.DataFrame
        Contingency table.
    n_components : int
        Number of CA dimensions to extract.

    Returns
    -------
    dict[str, Any]
        Dictionary containing model, row/column coordinates, explained inertia,
        row profiles, column profiles, and the original table.
    """
    ca_model = prince.CA(
        n_components=n_components,
        n_iter=10,
        copy=True,
        check_input=True,
        engine="sklearn",
        random_state=42,
    )

    ca_model = ca_model.fit(table)

    row_profiles = compute_row_profiles(table)
    col_profiles = compute_col_profiles(table)
    row_coords, col_coords = extract_ca_coordinates(ca_model, table)
    explained_inertia = extract_ca_inertia(ca_model)

    return {
        "ca_model": ca_model,
        "row_coords": row_coords,
        "col_coords": col_coords,
        "explained_inertia": explained_inertia,
        "row_profiles": row_profiles,
        "col_profiles": col_profiles,
        "table": table,
    }