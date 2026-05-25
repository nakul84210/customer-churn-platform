from __future__ import annotations

import json
from pathlib import Path
import streamlit as st
import pandas as pd

from utils.modeling import train_and_select_best_model
from dashboard.components.kpi import kpi_row

st.title("🧠 Model Training")

df: pd.DataFrame | None = st.session_state.get("dataset_df")
if df is None:
    st.warning("No dataset loaded. Go to **Upload Dataset** first.")
    st.stop()

target_col = st.selectbox("Target column", options=list(df.columns), index=df.columns.get_loc("Churn") if "Churn" in df.columns else 0)
random_state = st.number_input("Random state", min_value=0, value=42, step=1)

st.write("This will train 3 models with cross-validation + tuning + SMOTE and select the best by CV F1.")

if st.button("Train & Select Best Model"):
    tmp_csv = Path("data/_uploaded_tmp.csv")
    tmp_csv.parent.mkdir(exist_ok=True, parents=True)
    df.to_csv(tmp_csv, index=False)

    with st.spinner("Training..."):
        res = train_and_select_best_model(
            csv_path=tmp_csv,
            target_col=target_col,
            model_dir=Path("model"),
            random_state=int(random_state),
        )
    st.success(f"Best model: {res.best_model_name}")

    kpi_row(
        [
            ("Accuracy", f"{res.metrics['accuracy']:.3f}", None),
            ("Precision", f"{res.metrics['precision']:.3f}", None),
            ("Recall", f"{res.metrics['recall']:.3f}", None),
            ("F1", f"{res.metrics['f1']:.3f}", None),
        ]
    )

    st.markdown("### Confusion matrix")
    st.write(res.metrics["confusion_matrix"])

    st.markdown("### Classification report")
    st.code(res.metrics["classification_report"])

    st.markdown("### Saved artifacts")
    st.code(str(res.pipeline_path))
    st.code(str(res.metadata_path))

    st.session_state["trained_metadata"] = res.metrics