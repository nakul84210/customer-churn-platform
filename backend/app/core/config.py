from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MODEL_DIR: str = "./model"
    MODEL_PIPELINE_FILENAME: str = "churn_pipeline.joblib"
    MODEL_METADATA_FILENAME: str = "churn_metadata.json"

    MYSQL_ENABLED: bool = False
    MYSQL_HOST: str = "localhost"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    MYSQL_PASSWORD: str = "password"
    MYSQL_DATABASE: str = "churn_platform"


settings = Settings()