from __future__ import annotations

import streamlit as st

from config import APP_ICON, APP_LAYOUT, APP_TITLE
from app_pages.ca_page import render_ca_page
from app_pages.compare_page import render_compare_page
from app_pages.home_page import render_home_page
from app_pages.pca_page import render_pca_page
from app_pages.umap_page import render_umap_page
from app_pages.clustering_page import render_clustering_page
from app_pages.hierarchical_page import render_hierarchical_page


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

        if "active_page" not in st.session_state:
            st.session_state["active_page"] = "Hem"

        with st.expander("📊 Datareduktion", expanded=True):
            if st.button("Hem", width="stretch"):
                st.session_state["active_page"] = "Hem"

            if st.button("PCA", width="stretch"):
                st.session_state["active_page"] = "PCA"

            if st.button("UMAP", width="stretch"):
                st.session_state["active_page"] = "UMAP"

            if st.button("Korrespondensanalys", width="stretch"):
                st.session_state["active_page"] = "Korrespondensanalys"

            if st.button("Jämför metoder", width="stretch"):
                st.session_state["active_page"] = "Jämför metoder"
             
        with st.expander("🧠 Klustring", expanded=True):
            if st.button("K-means", width="stretch"):
                st.session_state["active_page"] = "K-means"
            if st.button("Hierarkisk klustring", width="stretch"):
                st.session_state["active_page"] = "Hierarkisk klustring"      

        page = st.session_state["active_page"]

        st.markdown("---")
        st.caption(f"Aktiv sida: **{page}**")

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
    elif page == "K-means":
        render_clustering_page()
    elif page == "Hierarkisk klustring":
        render_hierarchical_page()


if __name__ == "__main__":
    main()