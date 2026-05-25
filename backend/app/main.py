from __future__ import annotations

from fastapi import FastAPI
from backend.app.core.logging import setup_logging
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.predict import router as predict_router

setup_logging()

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="FastAPI backend for churn predictions + model metadata.",
)

app.include_router(health_router, tags=["health"])
app.include_router(predict_router, prefix="", tags=["prediction"])