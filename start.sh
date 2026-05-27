#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start Flask backend
echo "Starting Flask API on http://localhost:5000 ..."
PYTHONPATH="$SCRIPT_DIR/src" python -m aoa.api &
FLASK_PID=$!

# Start Vite dev server
echo "Starting Vite frontend on http://localhost:3000 ..."
cd "$SCRIPT_DIR/frontend" && npm run dev &
VITE_PID=$!

# Cleanup on exit
cleanup() {
  echo ""
  echo "Shutting down..."
  kill $FLASK_PID 2>/dev/null
  kill $VITE_PID 2>/dev/null
  wait
}
trap cleanup EXIT INT TERM

echo ""
echo "Open http://localhost:3000 in your browser."
echo "Press Ctrl+C to stop both servers."
echo ""

wait
