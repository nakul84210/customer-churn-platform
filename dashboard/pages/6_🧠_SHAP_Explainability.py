from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

from utils.shap_utils import explain_single

st.title("🧠 SHAP Explainability")

pipe_path = Path("model/churn_pipeline.joblib")
if not pipe_path.exists():
    st.warning("No trained model found. Train it in **Model Training** first.")
    st.stop()

pipeline = joblib.load(pipe_path)

st.write("SHAP explanations can be compute-heavy. This page provides a basic local explanation for a single record.")

default_json_path = Path("data/dummy_customer.json")
default = json.loads(default_json_path.read_text(encoding="utf-8"))
payload = st.text_area("Customer JSON", value=json.dumps(default, indent=2), height=320)

if st.button("Explain with SHAP"):
    row = json.loads(payload)
    row_df = pd.DataFrame([row])

    with st.spinner("Computing SHAP..."):
        res = explain_single(pipeline=pipeline, row_df=row_df)

    st.success("Computed SHAP values (raw).")
    st.write({"base_value": str(res.base_value)})
    st.write(res.shap_values)
    st.info(
        "Tip: For production-grade SHAP plots, we’d persist explainer artifacts and build feature-name mapping "
        "for one-hot encoded features. This starter keeps it robust and beginner-friendly."
    )