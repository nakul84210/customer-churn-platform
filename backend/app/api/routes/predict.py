from __future__ import annotations

import logging
from fastapi import APIRouter, HTTPException
from backend.app.schemas.predict import PredictRequest, PredictResponse, ModelInfoResponse
from backend.app.services.prediction_service import PredictionService
from backend.app.services.telemetry_service import TelemetryService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    svc = PredictionService()
    return svc.get_model_info()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        svc = PredictionService()
        result = svc.predict(req)
        # optional MySQL telemetry
        TelemetryService().log_prediction(req=req, resp=result)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal Server Error") from e