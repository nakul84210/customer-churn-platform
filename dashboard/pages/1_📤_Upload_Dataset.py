from __future__ import annotations

import streamlit as st
import pandas as pd

from utils.data_loader import ensure_total_charges_numeric
from utils.preprocessing import clean_dataframe

st.title("📤 Upload Dataset")

st.write("Upload a Telco-style churn dataset CSV (must include `Churn` column).")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
use_sample = st.button("Use sample dataset (data/sample_telco_churn.csv)")

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample:
    df = pd.read_csv("data/sample_telco_churn.csv")

if df is not None:
    df = ensure_total_charges_numeric(df)
    df = clean_dataframe(df)
    st.session_state["dataset_df"] = df
    st.success(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    st.dataframe(df.head(50), use_container_width=True)
else:
    st.info("Upload a file or click 'Use sample dataset'.")