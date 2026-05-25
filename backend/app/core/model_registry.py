from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import joblib

from backend.app.core.config import settings


@dataclass(frozen=True)
class LoadedModel:
    pipeline: object
    metadata: dict


def load_model() -> LoadedModel:
    model_dir = Path(settings.MODEL_DIR)
    pipe_path = model_dir / settings.MODEL_PIPELINE_FILENAME
    meta_path = model_dir / settings.MODEL_METADATA_FILENAME

    if not pipe_path.exists():
        raise ValueError(
            f"Model pipeline not found at {pipe_path}. Train first: python main.py train --csv data/sample_telco_churn.csv"
        )
    pipeline = joblib.load(pipe_path)

    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    return LoadedModel(pipeline=pipeline, metadata=metadata)