from __future__ import annotations

import json
import os
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("🧪 Real-time API Tester")

backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
st.text_input("Backend URL", value=backend_url, key="backend_url")

col1, col2 = st.columns(2)

with col1:
    if st.button("GET /health"):
        r = requests.get(f"{st.session_state.backend_url}/health", timeout=10)
        st.write(r.status_code)
        st.json(r.json())

with col2:
    if st.button("GET /model-info"):
        r = requests.get(f"{st.session_state.backend_url}/model-info", timeout=10)
        st.write(r.status_code)
        st.json(r.json())

st.markdown("---")
st.markdown("### POST /predict")

default_json_path = Path("data/dummy_customer.json")
default = json.loads(default_json_path.read_text(encoding="utf-8"))

payload = st.text_area("Request JSON", value=json.dumps({"customer": default}, indent=2), height=320)

if st.button("POST /predict"):
    r = requests.post(
        f"{st.session_state.backend_url}/predict",
        json=json.loads(payload),
        timeout=30,
    )
    st.write(r.status_code)
    try:
        st.json(r.json())
    except Exception:
        st.text(r.text)