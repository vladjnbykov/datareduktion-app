from __future__ import annotations

import streamlit as st


def render_page_title(title: str, subtitle: str | None = None) -> None:
    """
    Render a consistent page title and optional subtitle.
    """
    st.title(title)
    if subtitle:
        st.caption(subtitle)


def render_method_intro(method_name: str, description: str) -> None:
    """
    Render a styled introduction box for a method.
    """
    st.info(f"**{method_name}**\n\n{description}")


def render_dataset_intro(dataset_name: str, description: str) -> None:
    """
    Render a compact dataset description box.
    """
    st.markdown(f"### Dataset: {dataset_name}")
    st.write(description)


def render_note_box(text: str, kind: str = "info") -> None:
    """
    Render a note box with different visual styles.

    Parameters
    ----------
    text : str
        Text to display.
    kind : str
        One of: 'info', 'warning', 'success', 'error'
    """
    kind = kind.lower()

    if kind == "info":
        st.info(text)
    elif kind == "warning":
        st.warning(text)
    elif kind == "success":
        st.success(text)
    elif kind == "error":
        st.error(text)
    else:
        st.write(text)