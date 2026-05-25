from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.components.auth import require_login
from dashboard.components.theme import apply_theme
from utils.data_loader import load_csv, ensure_total_charges_numeric
from utils.preprocessing import clean_dataframe

APP_ROOT = Path(__file__).resolve().parent.parent


st.set_page_config(
    page_title="Customer Churn Platform",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
require_login()

st.title("Customer Churn Prediction & Business Analytics Platform")
st.caption("FastAPI + Streamlit • ML (LogReg/RF/XGBoost) • SHAP • Segmentation • Reports")

with st.sidebar:
    st.markdown("### Quick links")
    st.page_link("dashboard/streamlit_app.py", label="Home", icon="🏠")
    st.page_link("dashboard/pages/1_📤_Upload_Dataset.py", label="Upload Dataset", icon="📤")
    st.page_link("dashboard/pages/3_🧠_Model_Training.py", label="Model Training", icon="🧠")
    st.page_link("dashboard/pages/4_🔮_Prediction.py", label="Prediction", icon="🔮")
    st.page_link("dashboard/pages/8_🧪_API_Tester.py", label="API Tester", icon="🧪")

st.markdown("## What this app does")
c1, c2, c3 = st.columns(3)
c1.metric("Models", "3", "LogReg / RF / XGB")
c2.metric("Explainability", "SHAP", "local explanations")
c3.metric("Segmentation", "KMeans", "customer clusters")

st.markdown("---")
st.markdown("## Get started")
st.markdown(
    """
1. Upload a Telco-style dataset (or use sample) in **Upload Dataset**.
2. Train the best model in **Model Training** (saved to `model/` using joblib).
3. Use **Prediction** (single customer or batch).
4. Explore EDA, Segmentation, SHAP, and generate PDF reports.
"""
)

with st.expander("Load sample dataset preview"):
    sample_path = APP_ROOT / "data" / "sample_telco_churn.csv"
    df = load_csv(sample_path)
    df = ensure_total_charges_numeric(df)
    df = clean_dataframe(df)
    st.dataframe(df.head(20), use_container_width=True)