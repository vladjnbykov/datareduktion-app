from __future__ import annotations

import streamlit as st

from config import APP_ICON, APP_LAYOUT, APP_TITLE
from app_pages.ca_page import render_ca_page
from app_pages.compare_page import render_compare_page
from app_pages.home_page import render_home_page
from app_pages.pca_page import render_pca_page
from app_pages.umap_page import render_umap_page


def main() -> None:
    """
    Main app entry point.
    """
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout=APP_LAYOUT,
    )

    with st.sidebar:
        st.title(APP_TITLE)

        page = st.radio(
            "Navigera",
            options=[
                "Hem",
                "PCA",
                "UMAP",
                "Korrespondensanalys",
                "Jämför metoder",
            ],
        )

    if page == "Hem":
        render_home_page()
    elif page == "PCA":
        render_pca_page()
    elif page == "UMAP":
        render_umap_page()
    elif page == "Korrespondensanalys":
        render_ca_page()
    elif page == "Jämför metoder":
        render_compare_page()


if __name__ == "__main__":
    main()