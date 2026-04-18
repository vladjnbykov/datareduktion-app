from __future__ import annotations

import pandas as pd


def format_percent(value: float, decimals: int = 2) -> str:
    """Format a float as a percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_percent_list(values: list[float], decimals: int = 2) -> list[str]:
    """Format a list of floats as percentage strings."""
    return [format_percent(v, decimals=decimals) for v in values]


def round_df(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Return a rounded copy of a DataFrame."""
    return df.round(decimals)


def pretty_label(text: str) -> str:
    """Simple label prettifier."""
    return text.replace("_", " ").strip().title()