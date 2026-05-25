from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SegmentationResult:
    df_with_clusters: pd.DataFrame
    model: KMeans


def run_kmeans_segmentation(
    df: pd.DataFrame,
    n_clusters: int = 4,
    features: list[str] | None = None,
    random_state: int = 42,
) -> SegmentationResult:
    df = df.copy()

    if features is None:
        # sensible defaults for telco-like data
        features = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in df.columns]

    if len(features) < 2:
        raise ValueError("Need at least 2 numeric features for clustering (e.g., tenure, MonthlyCharges).")

    X = df[features].copy()
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    clusters = km.fit_predict(Xs)

    df["Cluster"] = clusters.astype(int)
    return SegmentationResult(df_with_clusters=df, model=km)