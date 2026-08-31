#!/bin/bash
# monitor_small.sh — watch pod ConvNeXt-Small v2, fetch+compare+deploy on finish.
# Runs on the Mac. Safe to leave unattended.
set -u
cd "$HOME/Projects/mobile_tracker"
LOG=/tmp/small_verdict.txt
SSL="SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi; print(certifi.where())')"
POD="root@213.192.2.94"
PORT="40047"
KEY="$HOME/.ssh/id_ed25519"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT -i $KEY $POD"

{
  echo "[$(date +%H:%M)] watcher started; waiting for ConvNeXt-Small v2 to finish..."
} | tee "$LOG"

# wait until training done (epoch 10/10 line + process gone)
while $SSH "pgrep -f train_classifier.py > /dev/null" 2>/dev/null; do
  sleep 90
done
sleep 30   # allow final checkpoint flush
{
  echo "[$(date +%H:%M)] pod training done. last log:"
  $SSH "grep -aE '^epoch|done' /tmp/train_small2.log | tail -3" 2>/dev/null
} | tee -a "$LOG"

# fetch + compare + conditional deploy
scp -o StrictHostKeyChecking=no -P "$PORT" -i "$KEY" "$POD:/root/mt/models/convnext_small_v2.pt" models/ >> "$LOG" 2>&1
{
  echo "[$(date +%H:%M)] comparison:"
  eval "$SSL .venv/bin/python compare_ckpts.py --data storage/dataset/dashcam_val \
    --ckpt models/convnext_v3.pt models/convnext_small_v2.pt" 2>&1
} >> "$LOG"

v13=$(grep -A3 "convnext_v3.pt" "$LOG" | grep -o "top-1 0\.[0-9]*" | head -1 | awk '{print $2}')
small=$(grep -A3 "convnext_small_v2.pt" "$LOG" | grep -o "top-1 0\.[0-9]*" | head -1 | awk '{print $2}')
DECISION="KEPT v13"
if [ -n "${v13:-}" ] && [ -n "${small:-}" ]; then
  if [ "$(.venv/bin/python -c "print(1 if $small >= $v13 + 0.02 else 0)")" = "1" ]; then
    cp models/r50_dashcam.pt models/convnext_v13_keep.pt
    cp models/convnext_small_v2.pt models/r50_dashcam.pt
    DECISION="DEPLOYED convnext_small_v2 ($small vs $v13)"
  else
    DECISION="KEPT v13 ($small vs $v13)"
  fi
fi
echo "DECISION: $DECISION" | tee -a "$LOG"

# auto-terminate the pod (RUNPOD_API_KEY in .env, pod id known) so no wasted $
API_KEY=$(grep -E '^RUNPOD_API_KEY=' .env | cut -d= -f2-)
if [ -n "$API_KEY" ]; then
  term=$(.venv/bin/python - <<'EOF'
import json, ssl, certifi, urllib.request
key = [l.split("=",1)[1].strip() for l in open(".env") if l.startswith("RUNPOD_API_KEY=")][0]
ctx = ssl.create_default_context(cafile=certifi.where())
req = urllib.request.Request("https://api.runpod.io/v2/pods/zj2x0cn2zsvjf8",
    method="DELETE", headers={"Authorization": f"Bearer {key}", "User-Agent":"Mozilla/5.0"})
try:
    with urllib.request.urlopen(req, timeout=30, context=ctx) as r:
        return r.status
except urllib.error.HTTPError as e:
    return e.code
EOF
)
  echo "POD TERMINATION REQUESTED (HTTP $term)" | tee -a "$LOG"
else
  echo "no RUNPOD_API_KEY — terminate pod zj2x0cn2zsvjf8 manually" | tee -a "$LOG"
fi
