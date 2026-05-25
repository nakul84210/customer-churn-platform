from __future__ import annotations

import pandas as pd
import plotly.express as px


def churn_distribution(df: pd.DataFrame, target_col: str = "Churn"):
    return px.histogram(df, x=target_col, color=target_col, title="Churn Distribution")


def monthly_charges_vs_churn(df: pd.DataFrame, target_col: str = "Churn"):
    if "MonthlyCharges" not in df.columns:
        raise ValueError("MonthlyCharges column missing.")
    return px.box(df, x=target_col, y="MonthlyCharges", color=target_col, title="MonthlyCharges vs Churn")


def contract_vs_churn(df: pd.DataFrame, target_col: str = "Churn"):
    if "Contract" not in df.columns:
        raise ValueError("Contract column missing.")
    return px.histogram(df, x="Contract", color=target_col, barmode="group", title="Contract vs Churn")


def tenure_analysis(df: pd.DataFrame, target_col: str = "Churn"):
    if "tenure" not in df.columns:
        raise ValueError("tenure column missing.")
    return px.histogram(df, x="tenure", color=target_col, nbins=30, title="Tenure Distribution by Churn")


def correlation_heatmap(df: pd.DataFrame):
    # numeric only
    num = df.select_dtypes(include="number")
    if num.shape[1] < 2:
        raise ValueError("Not enough numeric columns for correlation.")
    corr = num.corr(numeric_only=True)
    return px.imshow(corr, text_auto=True, title="Correlation Heatmap (Numeric)")