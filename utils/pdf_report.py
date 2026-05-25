from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from fpdf import FPDF


@dataclass(frozen=True)
class PDFReportConfig:
    title: str = "Customer Churn Report"
    author: str = "Customer Churn Platform"


def generate_pdf_report(
    out_path: str | Path,
    kpis: dict,
    model_metadata_path: str | Path | None = None,
    notes: str | None = None,
    cfg: PDFReportConfig = PDFReportConfig(),
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, cfg.title, ln=True)

    pdf.set_font("Helvetica", size=11)
    pdf.cell(0, 8, f"Generated: {datetime.now().isoformat(timespec='seconds')}", ln=True)
    pdf.cell(0, 8, f"Author: {cfg.author}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "KPIs", ln=True)
    pdf.set_font("Helvetica", size=11)
    for k, v in kpis.items():
        pdf.multi_cell(0, 6, f"- {k}: {v}")
    pdf.ln(2)

    if model_metadata_path is not None and Path(model_metadata_path).exists():
        meta = json.loads(Path(model_metadata_path).read_text(encoding="utf-8"))
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Model Summary", ln=True)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, f"Best model: {meta.get('best_model')}")
        pdf.multi_cell(0, 6, f"Holdout F1: {meta.get('f1')}")
        pdf.multi_cell(0, 6, f"Holdout Accuracy: {meta.get('accuracy')}")
        pdf.ln(2)

    if notes:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Notes", ln=True)
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 6, notes)

    pdf.output(str(out_path))
    return out_path