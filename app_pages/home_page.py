from __future__ import annotations

import pandas as pd
import streamlit as st

from components.intro import (
    render_note_box,
    render_page_title,
)


def render_home_page() -> None:
    """
    Render the home page of the app.
    """
    render_page_title(
        "Datareduktion Playground",
        "En pedagogisk app för att utforska PCA, UMAP och korrespondensanalys före egna projekt.",
    )

    st.write(
        """
Den här appen är byggd som ett **utforskande stöd** för att hjälpa dig förstå
hur olika datareduktionsmetoder fungerar, när de passar, och hur parametrar
påverkar resultatet.
        """
    )

    st.markdown("## Metoder i appen")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### PCA")
        st.write(
            """
- Numerisk data
- Linjär metod
- Bra för varians, komponenter och tolkning
            """
        )

    with col2:
        st.markdown("### UMAP")
        st.write(
            """
- Numerisk data
- Icke-linjär metod
- Bra för struktur, kluster och parameterutforskning
            """
        )

    with col3:
        st.markdown("### Korrespondensanalys")
        st.write(
            """
- Kategorisk data / kontingenstabell
- Visar samband mellan kategorier
- Bra för rad- och kolumnrelationer
            """
        )

    st.markdown("## Jämförelse mellan metoder")

    compare_df = pd.DataFrame(
        {
            "Metod": ["PCA", "UMAP", "CA"],
            "Datatyp": [
                "Numerisk",
                "Numerisk",
                "Kategorisk / frekvenstabell",
            ],
            "Typ av struktur": [
                "Linjär",
                "Ofta icke-linjär",
                "Associationer mellan kategorier",
            ],
            "Vanligt användningsområde": [
                "Variansreduktion, tolkning",
                "Visualisering, kluster",
                "Kontingenstabeller, kategoriska samband",
            ],
        }
    )
    st.dataframe(compare_df, width="stretch")

    st.markdown("## Så använder du appen")
    st.markdown(
        """
1. Välj en metod i menyn.
2. Titta på datasetet och dess variabler.
3. Ändra parametrar och kör analysen.
4. Studera visualiseringen.
5. Läs tolkningen och reflektera över vad som händer.
6. Jämför gärna PCA och UMAP på samma dataset.
        """
    )

    render_note_box(
        """
En snygg plot betyder inte automatiskt att metoden är bäst.
Fråga alltid:
- Vilken datatyp har jag?
- Är metoden tolkbar?
- Hur känsligt verkar resultatet vara för parametrar?
        """,
        kind="warning",
    )