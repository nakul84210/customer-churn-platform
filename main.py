"""
Project launcher helpers.

Usage (recommended for local dev):
  1) Train model:
      python main.py train --csv data/sample_telco_churn.csv

  2) Run backend:
      python main.py backend

  3) Run dashboard:
      python main.py dashboard

Notes:
- Backend serves FastAPI on http://localhost:8000
- Dashboard serves Streamlit on http://localhost:8501
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def cmd_train(args: argparse.Namespace) -> int:
    from utils.modeling import train_and_select_best_model

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 2

    out_dir = PROJECT_ROOT / "model"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = train_and_select_best_model(
        csv_path=csv_path,
        target_col=args.target,
        model_dir=out_dir,
        random_state=args.random_state,
    )
    print("\nTraining complete.")
    print("Best model:", result.best_model_name)
    print("Saved pipeline:", result.pipeline_path)
    print("Saved metadata:", result.metadata_path)
    return 0


def cmd_backend(_: argparse.Namespace) -> int:
    # Use uvicorn to run backend/app/main.py:app
    return run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ]
    )


def cmd_dashboard(_: argparse.Namespace) -> int:
    return run([sys.executable, "-m", "streamlit", "run", "dashboard/streamlit_app.py"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Customer Churn Platform")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train model and save to /model")
    p_train.add_argument("--csv", required=True, help="Path to churn dataset CSV")
    p_train.add_argument("--target", default="Churn", help="Target column (default: Churn)")
    p_train.add_argument("--random-state", type=int, default=42)
    p_train.set_defaults(func=cmd_train)

    p_backend = sub.add_parser("backend", help="Run FastAPI backend")
    p_backend.set_defaults(func=cmd_backend)

    p_dash = sub.add_parser("dashboard", help="Run Streamlit dashboard")
    p_dash.set_defaults(func=cmd_dashboard)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())