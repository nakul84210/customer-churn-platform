# Customer Churn Prediction & Business Analytics Platform

A production-ready starter for a full-stack ML application:
- **FastAPI** backend (`/predict`, `/health`, `/model-info`)
- **Streamlit** dashboard (EDA, training, predictions, segmentation, SHAP, reports)
- ML pipelines with **SMOTE**, CV + hyperparameter tuning, and model persistence via **joblib**
- Optional **MySQL** telemetry for prediction logs

---

## Architecture

**Training (offline / interactive)**
- `utils/modeling.py` trains 3 candidates:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Uses:
  - `ColumnTransformer` preprocessing (impute + OneHot + scaling)
  - `imblearn` pipeline with **SMOTE**
  - `GridSearchCV` with `StratifiedKFold`
- Saves:
  - `model/churn_pipeline.joblib`
  - `model/churn_metadata.json`

**Serving**
- FastAPI loads the pipeline from `model/` and serves predictions.

**Dashboard**
- Streamlit multipage app with a modern layout, dark mode toggle, and API tester.

---

## Setup

### 1) Create a virtual environment
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\\Scripts\\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Configure environment variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Update:
- `APP_USERNAME`, `APP_PASSWORD` (Streamlit login)
- `MYSQL_*` (optional)
- `BACKEND_URL` (dashboard → backend)

---

## Run locally

### A) Train the model
```bash
python main.py train --csv data/sample_telco_churn.csv
```

### B) Start the backend (FastAPI)
```bash
python main.py backend
```
Backend: `http://localhost:8000`

### C) Start the dashboard (Streamlit)
```bash
python main.py dashboard
```
Dashboard: `http://localhost:8501`

---

## API Documentation

### `GET /health`
Returns service status.

### `GET /model-info`
Returns saved training metrics and the selected best model.

### `POST /predict`
Request body:
```json
{
  "customer": {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 3,
    "MonthlyCharges": 89.2,
    "TotalCharges": 267.6,
    "Contract": "Month-to-month"
  }
}
```

Response:
```json
{
  "label": "Likely to Churn",
  "churn_probability": 0.73,
  "raw": { "...": "..." }
}
```

---

## Optional MySQL telemetry

If `MYSQL_ENABLED=true`, backend will create a table `prediction_logs` and store basic prediction logs.

Recommended local MySQL:
- Run via Docker:
```bash
docker run --name churn-mysql -e MYSQL_ROOT_PASSWORD=password -e MYSQL_DATABASE=churn_platform -p 3306:3306 -d mysql:8
```

Then set in `.env`:
```
MYSQL_ENABLED=true
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=password
MYSQL_DATABASE=churn_platform
```

---

## Deployment

### Docker (basic)
Build:
```bash
docker build -t churn-platform:latest .
```

Run backend:
```bash
docker run -p 8000:8000 --env-file .env churn-platform:latest python main.py backend
```

Run dashboard:
```bash
docker run -p 8501:8501 --env-file .env churn-platform:latest python -m streamlit run dashboard/streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

### Streamlit Cloud
- Deploy `dashboard/streamlit_app.py`
- Add secrets (username/password + BACKEND_URL)
- Backend can be deployed separately on Render/Railway.

### Render / Railway
- Deploy FastAPI with `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
- Ensure `model/` artifacts are included (or train during build, or mount persistent storage).

---

## Collaboration workflow

Recommended:
- Use feature branches + PRs
- Keep `.env` local; commit `.env.example`
- Store models in `model/` **locally** or in artifact storage for production. For git, avoid committing large binaries.

---

## Screenshots

Add screenshots to `static/` and reference them here:
- `static/home.png`
- `static/eda.png`
- `static/prediction.png`
- `static/reports.png`

---

## Future enhancements

- Better SHAP visualization for one-hot encoded features (feature name mapping)
- Batch `/predict` endpoint and async inference
- Role-based auth and proper user management
- Experiment tracking (MLflow)
- Scheduled retraining + model registry
- More robust schema validation for varied datasets