#!/bin/bash
# 1. Start FastAPI backend in background on internal port 8000
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --app-dir backend &

# 2. Wait for backend to be ready
for i in {1..10}; do
  if curl -s http://127.0.0.1:8000/health > /dev/null; then
    echo "Backend is ready!"
    break
  fi
  echo "Waiting for backend..."
  sleep 1
done

# 3. Start Streamlit frontend with CORS and XSRF disabled for reverse proxies like Render
PORT="${PORT:-8501}"
streamlit run frontend/app.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false
