#!/bin/bash
# a100_fetch2.sh — wait for convnext_large checkpoint to exist & be stable,
# then download it + run compare locally. Robust to job1/job2 chaining.
LOG=/tmp/a100_fetch.log
say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p 50283 -i $HOME/.ssh/id_ed25519 root@204.12.201.58"
SCP="scp -o StrictHostKeyChecking=no -P 50283 -i $HOME/.ssh/id_ed25519"
CD="$HOME/Projects/mobile_tracker"
for cycle in $(seq 1 200); do
  running=$($SSH "pgrep -fc train_classifier.py" 2>/dev/null)
  size=$($SSH "stat -c%s /root/mt/models/convnext_large_a100.pt 2>/dev/null" 2>/dev/null)
  say "cycle $cycle: running=${running} size=${size:-none}"
  # done when no trainer running AND checkpoint exists
  if [ "${running:-1}" -eq 0 ] && [ -n "${size:-}" ] && [ "$size" -gt 100000000 ]; then
    say "TRAINING DONE — fetching checkpoint"
    $SCP "root@204.12.201.58:/root/mt/models/convnext_large_a100.pt" "$CD/models/" >> "$LOG" 2>&1
    say "fetched large checkpoint"
    say "COMPARING vs deployed locally..."
    SSL_CERT_FILE=$($CD/.venv/bin/python -c "import certifi; print(certifi.where())") \
      $CD/.venv/bin/python $CD/compare_ckpts.py --data $CD/storage/dataset/dashcam_val \
      --ckpt $CD/models/r50_dashcam.pt $CD/models/convnext_large_a100.pt \
      > /tmp/large_compare.txt 2>&1
    say "COMPARE -> /tmp/large_compare.txt"
    tail -12 /tmp/large_compare.txt >> "$LOG"
    echo "SAFE_TO_TERMINATE $(date)" >> "$LOG"
    exit 0
  fi
  sleep 300
done
say "fetcher timed out"

# --- auto-terminate the pod after successful fetch+compare (save credits) ---
say "AUTO-TERMINATING pod via API (artifacts already downloaded + compared)"
SSL_CERT_FILE=$(.venv/bin/python -c "import certifi; print(certifi.where())") \
  .venv/bin/python - <<'PY' >> "$LOG" 2>&1
import json, ssl, certifi, urllib.request
key = [l.split("=",1)[1].strip() for l in open(".env") if l.startswith("RUNPOD_API_KEY=")][0]
ctx = ssl.create_default_context(cafile=certifi.where())
# find pod by IP (204.12.201.58) and terminate it
import urllib.request as U
r = U.Request("https://api.runpod.io/v2/pods", headers={"Authorization": f"Bearer {key}", "User-Agent":"Mozilla/5.0"})
with U.urlopen(r, timeout=30, context=ctx) as resp:
    d = json.loads(resp.read().decode())
target = None
for p in d.get("pods", []):
    rt = p.get("runtime") or {}
    for pt in (rt.get("ports") or []):
        if pt.get("publicIp") == "204.12.201.58":
            target = p["id"]; break
    if target: break
if not target and d.get("pods"):
    target = d["pods"][0]["id"]
if target:
    r = U.Request(f"https://api.runpod.io/v2/pods/{target}", method="DELETE",
                  headers={"Authorization": f"Bearer {key}", "User-Agent":"Mozilla/5.0"})
    try:
        with U.urlopen(r, timeout=30, context=ctx) as resp:
            print("POD TERMINATED:", target, resp.status)
    except U.HTTPError as e:
        print("terminate HTTP", e.code)
else:
    print("no pod found to terminate")
PY
say "POD TERMINATION PROCESS DONE"
