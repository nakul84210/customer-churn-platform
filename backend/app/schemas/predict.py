from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    # Telco-like schema (allow extras so custom CSVs can still work)
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    tenure: Optional[float] = None
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[float] = None

    model_config = {"extra": "allow"}


class PredictRequest(BaseModel):
    customer: CustomerFeatures = Field(..., description="Single customer features")


class PredictResponse(BaseModel):
    label: str
    churn_probability: float
    raw: Dict[str, Any]


class ModelInfoResponse(BaseModel):
    best_model: str | None = None
    metrics: Dict[str, Any] = {}