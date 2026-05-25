from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from sqlalchemy import text

from backend.app.core.mysql import get_mysql_client
from backend.app.schemas.predict import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)


@dataclass
class TelemetryService:
    def log_prediction(self, req: PredictRequest, resp: PredictResponse) -> None:
        client = get_mysql_client()
        if client is None:
            return
        try:
            with client.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO prediction_logs (input_json, prediction_label, churn_probability)
                        VALUES (:input_json, :label, :prob)
                        """
                    ),
                    {
                        "input_json": json.dumps(req.model_dump()),
                        "label": resp.label,
                        "prob": resp.churn_probability,
                    },
                )
        except Exception:
            logger.exception("Failed to write prediction log to MySQL")