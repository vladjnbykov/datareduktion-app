from __future__ import annotations

import streamlit as st

from utils.formatting import format_percent


def render_pca_interpretation(results: dict, standardize: bool) -> None:
    """
    Render an interpretation block for PCA results.
    """
    explained = results.get("explained_variance_ratio", [])
    cumulative = results.get("cumulative_variance", [])

    if not explained:
        st.warning("Ingen PCA-tolkning kunde genereras.")
        return

    pc1_text = format_percent(explained[0])
    pc2_text = format_percent(explained[1]) if len(explained) > 1 else "N/A"
    cum2_text = format_percent(cumulative[1]) if len(cumulative) > 1 else pc1_text

    scaling_text = (
        "Data standardiserades före PCA, vilket ofta är viktigt när variablerna har olika skala."
        if standardize
        else "Data standardiserades inte. Då kan variabler med större skala dominera komponenterna."
    )

    st.markdown("### Tolkning")
    st.info(
        f"""
- **PC1** förklarar **{pc1_text}** av variationen.
- **PC2** förklarar **{pc2_text}** av variationen.
- Tillsammans förklarar de två första komponenterna **{cum2_text}**.
- {scaling_text}
        """
    )


def render_umap_interpretation(params: dict) -> None:
    """
    Render an interpretation block for UMAP settings.
    """
    n_neighbors = params.get("n_neighbors")
    min_dist = params.get("min_dist")
    metric = params.get("metric")
    n_components = params.get("n_components")

    st.markdown("### Tolkning")
    st.info(
        f"""
- **n_neighbors = {n_neighbors}** styr hur mycket lokal respektive mer global struktur som betonas.
- **min_dist = {min_dist}** påverkar hur tätt punkter kan ligga i den reducerade representationen.
- **metric = {metric}** avgör hur likhet/avstånd mellan observationer beräknas.
- Här visas datan i **{n_components} dimension(er)** efter UMAP-reduktion.
        """
    )


def render_ca_interpretation(
    explained_inertia: list[float],
    row_coords,
    col_coords,
) -> None:
    """
    Render an interpretation block for CA results.
    """
    if explained_inertia:
        first_dim = format_percent(explained_inertia[0])
        second_dim = (
            format_percent(explained_inertia[1])
            if len(explained_inertia) > 1
            else "N/A"
        )
        inertia_text = (
            f"Första dimensionen förklarar **{first_dim}** och andra dimensionen "
            f"förklarar **{second_dim}** av strukturen i tabellen."
        )
    else:
        inertia_text = "Inertia kunde inte läsas ut från modellen."

    st.markdown("### Tolkning")
    st.info(
        f"""
- {inertia_text}
- Kategorier som ligger nära varandra i biplotten tenderar att ha liknande profiler.
- Närhet mellan en radkategori och en kolumnkategori kan antyda association. Campus och Copilot ligger nära → dem har liknande preferensmönster
- Tolkningen bör alltid göras försiktigt och ses som en explorativ analys.
        """
    )


def render_reflection_questions(method_name: str) -> None:
    """
    Render method-specific reflection questions.
    """
    st.markdown("### Reflektionsfrågor")

    method_name = method_name.upper()

    if method_name == "PCA":
        st.markdown(
            """
1. Vilka variabler verkar bidra mest till PC1?
2. Hur mycket information verkar gå förlorad om du bara använder två komponenter?
3. Hur ändras resultatet om du stänger av standardisering?
            """
        )
    elif method_name == "UMAP":
        st.markdown(
            """
1. Hur ändras strukturen när du ökar `n_neighbors`?
2. Blir grupperna tätare när du minskar `min_dist`?
3. Hur påverkar valet av `metric` resultatet?
            """
        )
    elif method_name == "CA":
        st.markdown(
            """
1. Vilka kategorier verkar ligga nära varandra?
2. Finns det någon kategori som verkar avvika tydligt?
3. Hur mycket av strukturen fångas av de två första dimensionerna?
            """
        )
    else:
        st.markdown(
            """
1. Vad visar visualiseringen?
2. Vilka val påverkade resultatet mest?
3. Hur säker är du på din tolkning?
            """
        )


def render_compare_reflection_questions() -> None:
    """
    Render reflection questions for comparison page.
    """
    st.markdown("### Reflektionsfrågor")
    st.markdown(
        """
1. Vilken metod separerar grupperna tydligast?
2. Vilken metod känns mest tolkbar?
3. Hur påverkas UMAP av parameterändringar jämfört med PCA?
4. När skulle du välja PCA även om UMAP ser mer visuellt tydlig ut?
        """
    )


def render_compare_summary_box() -> None:
    """
    Render a short summary box for PCA vs UMAP comparison.
    """
    st.info(
        """
**PCA** är en linjär metod och är ofta lättare att tolka.  
**UMAP** är en icke-linjär metod och kan ofta ge tydligare visuella grupper.  
En snygg separation betyder dock inte automatiskt att metoden är bäst för alla syften.
        """
    )