#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${API_PORT:-8001}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8501}"

SKIP_SYNC=0
SKIP_DATA_CHECK=0
USE_DOCKER=0

usage() {
  cat <<'EOF'
Run the full AutoRAG stack (API + Flask dashboard).

Usage:
  scripts/run_all.sh [options]

Options:
  --docker            Run via docker compose (build + up)
  --skip-sync         Skip `uv sync`
  --skip-data-check   Skip data presence validation
  -h, --help          Show this help

Environment overrides:
  API_PORT            FastAPI port (default: 8001)
  DASHBOARD_PORT      Flask dashboard port (default: 8501)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --docker)
      USE_DOCKER=1
      shift
      ;;
    --skip-sync)
      SKIP_SYNC=1
      shift
      ;;
    --skip-data-check)
      SKIP_DATA_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo
      usage
      exit 1
      ;;
  esac
done

cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    cp ".env.example" ".env"
    echo "Created .env from .env.example"
    echo "Please set GOOGLE_API_KEY in .env before running real experiments."
  else
    echo "Missing .env and .env.example"
    exit 1
  fi
fi

if [[ "$USE_DOCKER" -eq 1 ]]; then
  echo "Starting stack with docker compose..."
  docker compose up --build
  exit 0
fi

if [[ "$SKIP_SYNC" -eq 0 ]]; then
  echo "Installing/syncing dependencies with uv..."
  uv sync
  uv pip install -r requirements.txt
fi

if [[ "$SKIP_DATA_CHECK" -eq 0 ]]; then
  if ! uv run python -c "from pathlib import Path; import sys; from src.benchmark.loader import EnterpriseRagBenchLoader as L; sys.exit(0 if L(Path('./data')).benchmark_exists() else 1)"; then
    echo "Benchmark data missing under ./data (expect data/docs/ and data/bench/questions_subset.jsonl or questions.jsonl)."
    echo "Downloading official questions only (no full corpus). Add documents under data/docs/ or run download with documents."
    uv run python -c "from pathlib import Path; from src.benchmark.loader import EnterpriseRagBenchLoader as L; L(Path('./data')).download_release_files(include_all_documents=False)"
  fi

  if ! uv run python -c "from pathlib import Path; import sys; from src.benchmark.loader import EnterpriseRagBenchLoader as L; sys.exit(0 if L(Path('./data')).benchmark_exists() else 1)"; then
    echo "Benchmark files are still not usable under ./data"
    exit 1
  fi
fi

api_status_code() {
  local url="$1"
  uv run python - "$url" <<'PYCODE'
import sys
import urllib.error
import urllib.request

url = sys.argv[1]
try:
    with urllib.request.urlopen(url, timeout=2) as resp:
        print(resp.status)
except urllib.error.HTTPError as exc:
    print(exc.code)
except Exception:
    print("000")
PYCODE
}

wait_for_api_ready() {
  local attempts=20
  local status="000"
  while [[ "$attempts" -gt 0 ]]; do
    status="$(api_status_code "http://localhost:${API_PORT}/research/status")"
    if [[ "$status" == "200" ]]; then
      return 0
    fi
    sleep 0.5
    attempts=$((attempts - 1))
  done

  echo "API did not become ready on http://localhost:${API_PORT}/research/status (last status: ${status})."
  return 1
}

cleanup() {
  echo
  echo "Stopping services..."
  if [[ -n "${API_PID:-}" ]]; then kill "$API_PID" 2>/dev/null || true; fi
  if [[ -n "${DASH_PID:-}" ]]; then kill "$DASH_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

API_PID=""
if lsof -nP -iTCP:"${API_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Port ${API_PORT} is already in use. Validating existing service..."
  existing_status="$(api_status_code "http://localhost:${API_PORT}/research/status")"
  if [[ "$existing_status" != "200" ]]; then
    echo "Another process is listening on port ${API_PORT}, but it is not this API (status ${existing_status} on /research/status)."
    echo "Stop that process or run with a different port, e.g. API_PORT=8002 ./scripts/run_all.sh"
    exit 1
  fi
  echo "Reusing existing API on http://localhost:${API_PORT}"
else
  echo "Starting API on http://localhost:${API_PORT}"
  uv run uvicorn src.api.main:app --host 0.0.0.0 --port "$API_PORT" --reload --reload-exclude "src/retrieval/pipeline.py" &
  API_PID=$!

  if ! wait_for_api_ready; then
    if [[ -n "$API_PID" ]]; then
      kill "$API_PID" 2>/dev/null || true
      API_PID=""
    fi
    exit 1
  fi
fi

echo "Starting Flask dashboard on http://localhost:${DASHBOARD_PORT}"
API_BASE="http://localhost:${API_PORT}" DASHBOARD_PORT="$DASHBOARD_PORT" uv run python src/dashboard/app.py &
DASH_PID=$!

echo "AutoRAG stack is running."
echo "API:       http://localhost:${API_PORT}"
echo "Dashboard: http://localhost:${DASHBOARD_PORT}"
echo "Press Ctrl+C to stop."

if [[ -n "${API_PID:-}" ]]; then
  wait "$API_PID" "$DASH_PID"
else
  wait "$DASH_PID"
fi
