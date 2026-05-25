from __future__ import annotations

from pathlib import Path
import streamlit as st
import pandas as pd

from utils.pdf_report import generate_pdf_report

st.title("🧾 Reports")

df: pd.DataFrame | None = st.session_state.get("dataset_df")
meta_path = Path("model/churn_metadata.json")

if df is None:
    st.warning("No dataset loaded. Go to **Upload Dataset** first.")
    st.stop()

kpis = {
    "Rows": int(df.shape[0]),
    "Columns": int(df.shape[1]),
    "Churn rate (if available)": f"{(df['Churn'].astype(str).str.title()=='Yes').mean():.2%}" if "Churn" in df.columns else "N/A",
}

notes = st.text_area("Notes to include in the report", value="Executive summary: ...", height=120)

if st.button("Generate PDF report"):
    out = generate_pdf_report(
        out_path=Path("reports/churn_report.pdf"),
        kpis=kpis,
        model_metadata_path=meta_path if meta_path.exists() else None,
        notes=notes,
    )
    st.success(f"Generated: {out}")

    pdf_bytes = out.read_bytes()
    st.download_button("Download PDF", data=pdf_bytes, file_name=out.name, mime="application/pdf")