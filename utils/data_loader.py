from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)


def ensure_total_charges_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Telco datasets sometimes store TotalCharges as string with blanks
    if "TotalCharges" in df.columns:
        df = df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df