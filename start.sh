#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start Flask backend (in its own process group so we can kill the reloader child too)
echo "Starting Flask API on http://localhost:5000 ..."
setsid env PYTHONPATH="$SCRIPT_DIR/src" FLASK_APP="$SCRIPT_DIR/src/aoa/api.py" \
  poetry run python -m flask run --port 5000 --no-reload &
FLASK_PID=$!

# Start Vite dev server (also in its own process group)
echo "Starting Vite frontend on http://localhost:3000 ..."
setsid npm --prefix "$SCRIPT_DIR/frontend" run dev &
VITE_PID=$!

# Cleanup on exit — kill entire process groups
cleanup() {
  echo ""
  echo "Shutting down..."
  kill -TERM -"$FLASK_PID" 2>/dev/null
  kill -TERM -"$VITE_PID" 2>/dev/null
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

echo ""
echo "Open http://localhost:3000 in your browser."
echo "Press Ctrl+C to stop both servers."
echo ""

wait
