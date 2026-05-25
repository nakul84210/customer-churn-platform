from __future__ import annotations

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def require_login() -> None:
    if st.session_state.get("auth_ok"):
        return

    username = os.getenv("APP_USERNAME", "admin")
    password = os.getenv("APP_PASSWORD", "admin123")

    with st.sidebar:
        st.markdown("### Login")
        u = st.text_input("Username", key="auth_user")
        p = st.text_input("Password", type="password", key="auth_pass")
        if st.button("Sign in"):
            if u == username and p == password:
                st.session_state.auth_ok = True
                st.success("Logged in.")
                st.rerun()
            else:
                st.error("Invalid credentials.")

    st.stop()