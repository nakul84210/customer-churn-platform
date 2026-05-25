from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from utils.data_loader import ensure_total_charges_numeric, load_csv
from utils.preprocessing import build_preprocessor, clean_dataframe, infer_feature_spec


@dataclass(frozen=True)
class TrainingResult:
    best_model_name: str
    pipeline_path: Path
    metadata_path: Path
    metrics: dict[str, Any]


def _make_models(random_state: int) -> dict[str, Any]:
    return {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "rf": RandomForestClassifier(
            n_estimators=400, random_state=random_state, class_weight="balanced", n_jobs=-1
        ),
        "xgb": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            eval_metric="logloss",
        ),
    }


def _param_grids() -> dict[str, dict[str, list[Any]]]:
    return {
        "logreg": {
            "clf__C": [0.1, 1.0, 3.0],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
        "rf": {
            "clf__n_estimators": [300, 500],
            "clf__max_depth": [None, 8, 16],
            "clf__min_samples_split": [2, 5],
        },
        "xgb": {
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.03, 0.08],
            "clf__subsample": [0.8, 1.0],
        },
    }


def _safe_n_splits(y_train: pd.Series, max_splits: int = 5) -> int:
    """
    Pick a safe number of StratifiedKFold splits so we don't error on tiny datasets.

    Constraint: n_splits <= min(class_counts)
    """
    vc = y_train.value_counts()
    if vc.empty:
        return 2
    min_class = int(vc.min())
    return max(2, min(max_splits, min_class))


def _should_use_smote(y_train: pd.Series) -> bool:
    """
    SMOTE requires enough minority samples. For small datasets, it may fail.
    This conservative check avoids runtime errors.
    """
    vc = y_train.value_counts()
    if len(vc) < 2:
        return False
    return int(vc.min()) >= 6


def train_and_select_best_model(
    csv_path: str | Path,
    target_col: str,
    model_dir: str | Path,
    random_state: int = 42,
) -> TrainingResult:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_csv(csv_path)
    df = ensure_total_charges_numeric(df)
    df = clean_dataframe(df)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)}")

    # Basic label normalization
    y_raw = df[target_col].astype(str).str.strip().str.title()
    y = (y_raw == "Yes").astype(int)  # 1 = churn

    X = df.drop(columns=[target_col])

    # Drop obvious ID columns if present
    for id_col in ["customerID", "CustomerID", "customer_id"]:
        if id_col in X.columns:
            X = X.drop(columns=[id_col])

    # Feature spec + preprocessor
    spec = infer_feature_spec(pd.concat([X, y], axis=1), target_col=target_col)
    preprocessor = build_preprocessor(spec)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    # Safe CV + SMOTE decisions for small data
    n_splits = _safe_n_splits(y_train, max_splits=5)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    use_smote = _should_use_smote(y_train)

    models = _make_models(random_state=random_state)
    grids = _param_grids()

    best_name = ""
    best_est = None
    best_score = -1.0
    results: dict[str, Any] = {"candidates": {}}

    for name, clf in models.items():
        steps: list[tuple[str, Any]] = [("pre", preprocessor)]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=random_state)))
        steps.append(("clf", clf))

        pipe = ImbPipeline(steps=steps)

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=grids[name],
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        results["candidates"][name] = {
            "best_params": grid.best_params_,
            "cv_best_f1": float(grid.best_score_),
            "cv_n_splits": int(n_splits),
            "smote_used": bool(use_smote),
        }

        if float(grid.best_score_) > best_score:
            best_score = float(grid.best_score_)
            best_name = name
            best_est = grid.best_estimator_

    assert best_est is not None

    # Evaluate on holdout
    y_pred = best_est.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "best_model": best_name,
        "cv_best_f1": float(best_score),
        "cv_n_splits": int(n_splits),
        "smote_used": bool(use_smote),
        "feature_spec": {"numeric": spec.numeric, "categorical": spec.categorical},
        "grid_results": results,
    }

    import joblib

    pipeline_path = model_dir / "churn_pipeline.joblib"
    metadata_path = model_dir / "churn_metadata.json"
    joblib.dump(best_est, pipeline_path)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return TrainingResult(
        best_model_name=best_name,
        pipeline_path=pipeline_path,
        metadata_path=metadata_path,
        metrics=metrics,
    )