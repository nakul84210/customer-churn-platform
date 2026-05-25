from __future__ import annotations

import streamlit as st
import pandas as pd

from utils import eda

st.title("📊 EDA Dashboard")

df: pd.DataFrame | None = st.session_state.get("dataset_df")
if df is None:
    st.warning("No dataset loaded. Go to **Upload Dataset** first.")
    st.stop()

target_col = st.selectbox("Target column", options=[c for c in df.columns], index=df.columns.get_loc("Churn") if "Churn" in df.columns else 0)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(eda.churn_distribution(df, target_col=target_col), use_container_width=True)
with c2:
    try:
        st.plotly_chart(eda.monthly_charges_vs_churn(df, target_col=target_col), use_container_width=True)
    except Exception as e:
        st.info(str(e))

c3, c4 = st.columns(2)
with c3:
    try:
        st.plotly_chart(eda.contract_vs_churn(df, target_col=target_col), use_container_width=True)
    except Exception as e:
        st.info(str(e))
with c4:
    try:
        st.plotly_chart(eda.tenure_analysis(df, target_col=target_col), use_container_width=True)
    except Exception as e:
        st.info(str(e))

st.markdown("### Correlations")
try:
    st.plotly_chart(eda.correlation_heatmap(df), use_container_width=True)
except Exception as e:
    st.info(str(e))