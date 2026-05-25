from __future__ import annotations

from dataclasses import dataclass
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from backend.app.core.config import settings


@dataclass(frozen=True)
class MySQLClient:
    engine: Engine

    def init_schema(self) -> None:
        # Minimal schema for prediction logs
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_logs (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        input_json JSON,
                        prediction_label VARCHAR(64),
                        churn_probability DOUBLE
                    );
                    """
                )
            )


def get_mysql_client() -> MySQLClient | None:
    if not settings.MYSQL_ENABLED:
        return None

    url = (
        f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
        f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
    )
    engine = create_engine(url, pool_pre_ping=True)
    client = MySQLClient(engine=engine)
    client.init_schema()
    return client