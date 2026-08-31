#!/bin/bash
# morning_start.sh — live drive mode: iPhone (Iriun) camera + road context on.
cd "$HOME/Projects/mobile_tracker"
.venv/bin/python - <<'PY'
s = open("config.yaml").read()
import re
s = re.sub(r'^  source:.*$', '  source: 0', s, count=1, flags=re.M)
open("config.yaml","w").write(s)
PY
SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())") nohup .venv/bin/python main.py > /tmp/mt_run.log 2>&1 &
disown
sleep 12
curl -s http://127.0.0.1:8500/api/stats | .venv/bin/python -c "import json,sys; s=json.load(sys.stdin); print('LIVE | fps:', s['fps'], '| classifier:', s['classifier']['available'])" 2>/dev/null || tail -5 /tmp/mt_run.log
