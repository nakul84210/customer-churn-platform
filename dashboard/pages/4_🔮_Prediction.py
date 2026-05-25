from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

from utils.io import df_to_csv_bytes

st.title("🔮 Prediction")

pipe_path = Path("model/churn_pipeline.joblib")
if not pipe_path.exists():
    st.warning("No trained model found. Train it in **Model Training** first.")
    st.stop()

pipeline = joblib.load(pipe_path)

st.markdown("### Single customer prediction")
default_json_path = Path("data/dummy_customer.json")
default = json.loads(default_json_path.read_text(encoding="utf-8"))

payload = st.text_area("Customer JSON", value=json.dumps(default, indent=2), height=320)

if st.button("Predict (single)"):
    row = json.loads(payload)
    df = pd.DataFrame([row])
    proba = float(pipeline.predict_proba(df)[0][1])
    label = "Likely to Churn" if proba >= 0.5 else "Not Likely to Churn"
    st.success(label)
    st.metric("Churn probability", f"{proba:.3f}")

st.markdown("---")
st.markdown("### Batch prediction (CSV)")
uploaded = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

if uploaded is not None:
    bdf = pd.read_csv(uploaded)
    # drop known target column if included
    if "Churn" in bdf.columns:
        bdf = bdf.drop(columns=["Churn"])
    if "customerID" in bdf.columns:
        customer_ids = bdf["customerID"].astype(str)
        bdf2 = bdf.drop(columns=["customerID"])
    else:
        customer_ids = None
        bdf2 = bdf

    probs = pipeline.predict_proba(bdf2)[:, 1]
    out = bdf.copy()
    out["churn_probability"] = probs
    out["prediction"] = ["Likely to Churn" if p >= 0.5 else "Not Likely to Churn" for p in probs]

    st.dataframe(out.head(50), use_container_width=True)
    st.download_button(
        "Download predictions CSV",
        data=df_to_csv_bytes(out),
        file_name="predictions.csv",
        mime="text/csv",
    )