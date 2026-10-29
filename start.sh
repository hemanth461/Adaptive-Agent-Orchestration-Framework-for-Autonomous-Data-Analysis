#!/bin/bash
# 1. Start FastAPI backend in background on internal port 8000
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --app-dir backend &

# 2. Give the backend a moment to initialize
sleep 3

# 3. Start Streamlit frontend on the public port assigned by cloud provider ($PORT or 8501)
PORT="${PORT:-8501}"
streamlit run frontend/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
