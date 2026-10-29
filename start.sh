#!/bin/bash
# 1. Start FastAPI backend on internal port 8000 in background
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --app-dir backend &

# 2. Immediately launch Streamlit on Render's assigned port
PORT="${PORT:-8501}"
exec streamlit run frontend/app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
