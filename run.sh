#!/usr/bin/env bash
set -e

echo "Starting API on http://localhost:8001"
uvicorn src.api.main:app --host 0.0.0.0 --port 8001 --reload --reload-exclude "src/retrieval/pipeline.py" &
API_PID=$!

echo "Starting dashboard on http://localhost:8501"
API_BASE="http://localhost:8001" python src/dashboard/app.py &
DASHBOARD_PID=$!

cleanup() {
  echo "Stopping processes..."
  kill $API_PID $DASHBOARD_PID 2>/dev/null || true
}

trap cleanup EXIT INT TERM

echo "AutoRAG stack is running."
echo "API:       http://localhost:8001"
echo "Dashboard: http://localhost:8501"
echo "Press Ctrl+C to stop."

wait
