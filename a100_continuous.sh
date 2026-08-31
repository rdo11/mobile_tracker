#!/bin/bash
# a100_continuous.sh — download the Large checkpoint EVERY time it updates
# (best-val saved each epoch), so even a mid-run pod death loses nothing.
# Then compare + auto-terminate when training fully completes.
LOG=/tmp/a100_continuous.log
say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p 50283 -i $HOME/.ssh/id_ed25519 root@204.12.201.58"
SCP="scp -o StrictHostKeyChecking=no -P 50283 -i $HOME/.ssh/id_ed25519"
CD="$HOME/Projects/mobile_tracker"
last_size=0
for cycle in $(seq 1 400); do
  running=$($SSH "pgrep -fc train_classifier.py" 2>/dev/null)
  size=$($SSH "stat -c%s /root/mt/models/convnext_large_a100.pt 2>/dev/null" 2>/dev/null)
  ep=$($SSH "grep -acE '^epoch' /root/job2_large.log 2>/dev/null" 2>/dev/null)
  say "cycle $cycle: running=${running} epochs=${ep:-0} ckpt_size=${size:-none}"
  # download whenever the checkpoint grew (best-val re-saved)
  if [ -n "${size:-}" ] && [ "$size" != "$last_size" ] && [ "$size" -gt 100000000 ]; then
    say "  checkpoint updated (${size} bytes) — downloading"
    $SCP "root@204.12.201.58:/root/mt/models/convnext_large_a100.pt" "$CD/models/convnext_large_a100.pt" >> "$LOG" 2>&1
    say "  downloaded"
    last_size="$size"
  fi
  # done: no trainer running AND checkpoint present -> final compare + terminate
  if [ "${running:-1}" -eq 0 ] && [ -n "${size:-}" ]; then
    say "TRAINING COMPLETE — final compare"
    SSL_CERT_FILE=$($CD/.venv/bin/python -c "import certifi; print(certifi.where())") \
      $CD/.venv/bin/python $CD/compare_ckpts.py --data $CD/storage/dataset/dashcam_val \
      --ckpt $CD/models/r50_dashcam.pt $CD/models/convnext_large_a100.pt \
      > /tmp/large_compare.txt 2>&1
    say "compare -> /tmp/large_compare.txt"
    tail -12 /tmp/large_compare.txt >> "$LOG"
    say "AUTO-TERMINATING pod via API"
    SSL_CERT_FILE=$($CD/.venv/bin/python -c "import certifi; print(certifi.where())") \
      $CD/.venv/bin/python - <<'PY' >> "$LOG" 2>&1
import json, ssl, certifi, urllib.request as U
key = [l.split("=",1)[1].strip() for l in open(""$(dirname "$0")/.env"") if l.startswith("RUNPOD_API_KEY=")][0]
ctx = ssl.create_default_context(cafile=certifi.where())
r = U.Request("https://api.runpod.io/v2/pods", headers={"Authorization": f"Bearer {key}", "User-Agent":"Mozilla/5.0"})
with U.urlopen(r, timeout=30, context=ctx) as resp:
    d = json.loads(resp.read().decode())
target=None
for p in d.get("pods", []):
    rt=p.get("runtime") or {}
    for pt in (rt.get("ports") or []):
        if pt.get("publicIp")=="204.12.201.58": target=p["id"]
    if target: break
if not target and d.get("pods"): target=d["pods"][0]["id"]
if target:
    r = U.Request(f"https://api.runpod.io/v2/pods/{target}", method="DELETE",
                  headers={"Authorization": f"Bearer {key}", "User-Agent":"Mozilla/5.0"})
    try:
        with U.urlopen(r, timeout=30, context=ctx) as resp: print("TERMINATED", target, resp.status)
    except U.HTTPError as e: print("term HTTP", e.code)
else: print("no pod found")
PY
    echo "ALL_DONE $(date)" >> "$LOG"
    exit 0
  fi
  sleep 240
done
say "fetcher timed out"
