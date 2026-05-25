from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap


@dataclass(frozen=True)
class ShapResult:
    base_value: Any
    shap_values: Any


def explain_single(
    pipeline,
    row_df: pd.DataFrame,
    positive_class_index: int = 1,
) -> ShapResult:
    """
    SHAP for a single row.

    We try TreeExplainer for tree models and fall back to KernelExplainer.
    Note: Pipelines + OneHot can be complex; we explain using the model's predict_proba
    wrapper (KernelExplainer) when needed.
    """
    # Prefer fast explainers if possible
    try:
        model = pipeline.named_steps.get("clf", None)
        if model is not None and hasattr(model, "get_booster"):
            # XGBoost
            explainer = shap.TreeExplainer(model)
            # Need transformed data
            X_trans = pipeline.named_steps["pre"].transform(row_df)
            sv = explainer.shap_values(X_trans)
            # For binary xgb, sv might be (n, m) or list
            return ShapResult(base_value=explainer.expected_value, shap_values=sv)
    except Exception:
        pass

    # Robust fallback: KernelExplainer on pipeline predict_proba
    def f(x: np.ndarray) -> np.ndarray:
        # x is numpy, convert to dataframe with same columns
        df = pd.DataFrame(x, columns=row_df.columns)
        return pipeline.predict_proba(df)

    bg = row_df.sample(n=1, replace=True, random_state=0)
    explainer = shap.KernelExplainer(f, bg)
    sv = explainer.shap_values(row_df, nsamples=100)
    return ShapResult(base_value=explainer.expected_value, shap_values=sv)