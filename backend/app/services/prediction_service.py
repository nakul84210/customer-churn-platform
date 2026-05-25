from __future__ import annotations

from dataclasses import dataclass
import pandas as pd

from backend.app.core.model_registry import load_model
from backend.app.schemas.predict import PredictRequest, PredictResponse, ModelInfoResponse


@dataclass
class PredictionService:
    _loaded: object | None = None

    def _get_loaded(self):
        if self._loaded is None:
            self._loaded = load_model()
        return self._loaded

    def get_model_info(self) -> ModelInfoResponse:
        loaded = self._get_loaded()
        meta = loaded.metadata or {}
        return ModelInfoResponse(
            best_model=meta.get("best_model"),
            metrics=meta,
        )

    def predict(self, req: PredictRequest) -> PredictResponse:
        loaded = self._get_loaded()
        pipeline = loaded.pipeline

        row = req.customer.model_dump()
        df = pd.DataFrame([row])

        proba = float(pipeline.predict_proba(df)[0][1])
        label = "Likely to Churn" if proba >= 0.5 else "Not Likely to Churn"

        return PredictResponse(
            label=label,
            churn_probability=proba,
            raw=row,
        )