#!/bin/sh
# auto-restart wrapper
cd "$(dirname "$0")"
python3 -c "import websocket" 2>/dev/null || pip install -q websocket-client 2>/dev/null
while true; do
  python3 app.py
  echo "[run.sh] app exited ($?) — restarting in 3s..."
  sleep 3
done
