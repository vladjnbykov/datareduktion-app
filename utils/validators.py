from __future__ import annotations

import pandas as pd


def validate_min_selected_features(feature_cols: list[str], min_required: int = 2) -> None:
    """Ensure that at least a minimum number of features are selected."""
    if len(feature_cols) < min_required:
        raise ValueError(
            f"Minst {min_required} features måste väljas för att köra analysen."
        )


def validate_components_against_features(n_components: int, n_features: int) -> None:
    """Ensure n_components does not exceed the number of selected features."""
    if n_components < 1:
        raise ValueError("Antal komponenter måste vara minst 1.")
    if n_components > n_features:
        raise ValueError(
            "Antal komponenter kan inte vara större än antalet valda features."
        )


def validate_ca_table(table: pd.DataFrame) -> None:
    """Validate a CA contingency table."""
    if table.empty:
        raise ValueError("Kontingenstabellen är tom.")

    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError(
            "Kontingenstabellen måste ha minst två rader och två kolumner."
        )

    if (table < 0).any().any():
        raise ValueError("Kontingenstabellen får inte innehålla negativa värden.")

    if table.sum().sum() == 0:
        raise ValueError("Kontingenstabellen får inte bestå enbart av nollor.")


def validate_metric(metric: str, allowed_metrics: list[str]) -> None:
    """Validate a chosen UMAP metric."""
    if metric not in allowed_metrics:
        raise ValueError(
            f"Ogiltigt metric-val: {metric}. Tillåtna värden: {allowed_metrics}"
        )