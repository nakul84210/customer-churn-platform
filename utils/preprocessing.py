from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class FeatureSpec:
    numeric: list[str]
    categorical: list[str]


def infer_feature_spec(df: pd.DataFrame, target_col: str) -> FeatureSpec:
    cols = [c for c in df.columns if c != target_col]
    numeric = []
    categorical = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)
    return FeatureSpec(numeric=numeric, categorical=categorical)


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, spec.numeric),
            ("cat", categorical_pipe, spec.categorical),
        ],
        remainder="drop",
    )


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    # Normalize common churn values
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].astype(str).str.strip().str.title()
        df["Churn"] = df["Churn"].replace({"True": "Yes", "False": "No"})
    return df