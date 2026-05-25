from __future__ import annotations

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.segmentation import run_kmeans_segmentation

st.title("🧩 Customer Segmentation (KMeans)")

df: pd.DataFrame | None = st.session_state.get("dataset_df")
if df is None:
    st.warning("No dataset loaded. Go to **Upload Dataset** first.")
    st.stop()

n_clusters = st.slider("Clusters", 2, 8, 4)

features = st.multiselect(
    "Numeric features",
    options=list(df.columns),
    default=[c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns],
)

if st.button("Run Segmentation"):
    res = run_kmeans_segmentation(df=df, n_clusters=int(n_clusters), features=features)
    out = res.df_with_clusters
    st.session_state["segmented_df"] = out

    st.success("Segmentation complete.")
    st.dataframe(out.head(30), use_container_width=True)

    if len(features) >= 2:
        fig = px.scatter(out, x=features[0], y=features[1], color="Cluster", title="Clusters")
        st.plotly_chart(fig, use_container_width=True)