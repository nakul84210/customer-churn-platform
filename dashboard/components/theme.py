from __future__ import annotations

from pathlib import Path
import streamlit as st


def apply_theme() -> None:
    # Dark mode toggle
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = True

    with st.sidebar:
        st.toggle("Dark mode", key="dark_mode")

    css_path = Path(__file__).resolve().parent.parent / "assets" / "style.css"
    css = css_path.read_text(encoding="utf-8")

    if st.session_state.dark_mode:
        st.markdown(
            """
<style>
:root { color-scheme: dark; }
</style>
""",
            unsafe_allow_html=True,
        )

    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)