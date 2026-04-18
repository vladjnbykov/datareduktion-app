from __future__ import annotations

import streamlit as st

from components.controls import render_ca_controls
from components.dataset_view import (
    render_contingency_table,
    render_profiles_table,
)
from components.interpretation import (
    render_ca_interpretation,
    render_reflection_questions,
)
from components.intro import (
    render_dataset_intro,
    render_method_intro,
    render_note_box,
    render_page_title,
)
from components.plots import (
    plot_ca_biplot,
    plot_inertia_bar,
)
from config import CA_DATASET_NAME, METHOD_DESCRIPTIONS
from data.synthetic_data import get_ai_tool_preference_table
from methods.ca_analysis import (
    compute_col_profiles,
    compute_row_profiles,
    run_ca,
)
from utils.validators import validate_ca_table


def render_ca_page() -> None:
    """
    Render the correspondence analysis page.
    """
    render_page_title("Korrespondensanalys (CA)")
    render_method_intro("CA", METHOD_DESCRIPTIONS["CA"])

    table = get_ai_tool_preference_table()

    render_dataset_intro(
        CA_DATASET_NAME,
        "Detta är en liten syntetisk kontingenstabell där rader är studieformer "
        "och kolumner är favoritverktyg inom AI.",
    )

    left_col, right_col = st.columns([1, 2])

    with left_col:
        controls = render_ca_controls()

    view_mode = controls["view_mode"]
    show_inertia = controls["show_inertia"]
    show_profiles = controls["show_profiles"]
    n_components = controls["n_components"]

    try:
        validate_ca_table(table)
    except ValueError as exc:
        with right_col:
            render_note_box(str(exc), kind="error")
        return

    if view_mode == "Frekvenser":
        table_to_show = table.copy()
    elif view_mode == "Radproportioner":
        table_to_show = compute_row_profiles(table)
    else:
        table_to_show = compute_col_profiles(table)

    results = run_ca(table=table, n_components=n_components)

    with right_col:
        render_contingency_table(table_to_show)

        st.markdown("## Resultat")
        plot_ca_biplot(results["row_coords"], results["col_coords"])


        if show_inertia and results["explained_inertia"]:
            plot_inertia_bar(results["explained_inertia"])

        if show_profiles:
            prof_col1, prof_col2 = st.columns(2)

            with prof_col1:
                render_profiles_table(results["row_profiles"], "Radprofiler")

            with prof_col2:
                render_profiles_table(results["col_profiles"], "Kolumnprofiler")

    render_ca_interpretation(
        explained_inertia=results["explained_inertia"],
        row_coords=results["row_coords"],
        col_coords=results["col_coords"],
    )
    render_reflection_questions("CA")