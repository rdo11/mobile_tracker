#!/bin/bash
# auto_shutdown.sh — watch pod training, fetch checkpoints, then terminate pod.
# Runs on the Mac. Polls the pod every 90s; when the chain is done (or hard
# deadline hits), downloads final models, verifies them, terminates the pod.
set -u

POD=9bl8vjzt2khtfe
HOST=root@157.157.221.29
PORT=57145
KEY=$HOME/.ssh/RunPod
DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$DIR/models"
LOG=/tmp/auto_shutdown.log
DEADLINE=${1:-"01:30"}   # hard stop: terminate by this time no matter what

log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

API_KEY=$(grep RUNPOD_API_KEY "$DIR/.env" | cut -d= -f2)
if [ -z "$API_KEY" ]; then
  log "RUNPOD_API_KEY is empty in .env — refusing to run (terminate would silently fail)"
  exit 1
fi

terminate() {
  log "TERMINATING pod $POD"
  curl -s -X POST https://api.runpod.io/graphql \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $API_KEY" \
    -d "{\"query\":\"mutation { podTerminate(input: {podId: \\\"$POD\\\"}) { id desiredStatus } }\"}" >> "$LOG" 2>&1
  log "terminate request sent"
  exit 0
}

while true; do
  now=$(date +%H:%M)
  if [[ "$now" > "$DEADLINE" ]]; then
    log "HARD DEADLINE reached ($DEADLINE) — shutting down without download"
    terminate
  fi
  done_marker=$(ssh -o BatchMode=yes -o ConnectTimeout=15 -i "$KEY" "$HOST" -p "$PORT" \
    "grep -c CHAIN-DONE /tmp/chain.log 2>/dev/null" 2>/dev/null)
  if [[ "$done_marker" == "1" ]]; then
    log "chain done — downloading checkpoints"
    scp -P "$PORT" -i "$KEY" "$HOST:/root/train/models/r18_filt_60.pt" "$MODELS_DIR/" >> "$LOG" 2>&1
    scp -P "$PORT" -i "$KEY" "$HOST:/root/train/models/r50_filt_60.pt" "$MODELS_DIR/" >> "$LOG" 2>&1
    ssh -o BatchMode=yes -i "$KEY" "$HOST" -p "$PORT" \
      "grep -A3 '=== eval ===' /tmp/chain.log | tail -6" > /tmp/eval_results.txt 2>/dev/null
    # verify checkpoints load
    v18=$("$DIR/.venv/bin/python" -c "
import torch; c = torch.load('$MODELS_DIR/r18_filt_60.pt', map_location='cpu')
print(len(c['classes']), 'classes,', len(c['state_dict']), 'keys')" 2>/dev/null)
    v50=$("$DIR/.venv/bin/python" -c "
import torch; c = torch.load('$MODELS_DIR/r50_filt_60.pt', map_location='cpu')
print(len(c['classes']), 'classes,', len(c['state_dict']), 'keys')" 2>/dev/null)
    log "verify r18: $v18"
    log "verify r50: $v50"
    if [[ "$v18" == *"classes"* && "$v50" == *"classes"* ]]; then
      log "VERIFIED — safe to terminate"
      terminate
    else
      log "VERIFICATION FAILED — NOT terminating, need human"
      exit 1
    fi
  else
    log "still training (marker=$done_marker) — sleeping 90s"
    sleep 90
  fi
done