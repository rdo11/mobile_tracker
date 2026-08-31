#!/bin/bash
# after_v13.sh — overnight: extract batch4 -> deepseek label -> wait for
# gemini quota reset -> cross-check -> rebuild -> train v13 -> deploy if +2pts.
set -u
cd "$HOME/Projects/mobile_tracker"
LOG=/tmp/v13_pipeline.log
SRC=storage/driving_videos_src4
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
SSL="SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi; print(certifi.where())')"

# 1) parallel extraction of batch 4 (3 workers, round-robin over videos)
say "extracting batch 4 (3 workers)"
i=0
for v in "$SRC"/*.mp4; do
  g=$((i % 3))
  echo "$v" >> /tmp/v13_group_$g.txt
  i=$((i+1))
done
pids=()
for g in 0 1 2; do
  [ -s /tmp/v13_group_$g.txt ] || continue
  vids=$(sed 's|^|"|; s|$|"|; s|" "| " |g' /tmp/v13_group_$g.txt | tr '\n' ' ')
  nohup bash -c "$SSL exec .venv/bin/python -u extract_crops.py $vids >> /tmp/extract5_g$g.log 2>&1" < /dev/null > /dev/null 2>&1 &
  pids+=($!)
done
for p in "${pids[@]:-}"; do wait "$p" 2>/dev/null || true; done
while pgrep -f extract_crops.py > /dev/null; do sleep 60; done
say "extraction done"

# 2) deepseek labeling until nothing left (or only rejects)
for round in 1 2 3 4 5 6 7 8; do
  remaining=$(.venv/bin/python - <<'EOF'
from pathlib import Path
yt = Path("storage/dataset/dashcam_youtube"); raw = Path("storage/dataset/dashcam_raw")
labeled = {p.name for p in raw.rglob("*.jpg")}
print(sum(1 for p in yt.rglob("*.jpg") if p.name not in labeled))
EOF
)
  [ "$remaining" = "0" ] && break
  say "labeling round $round: $remaining unlabeled"
  eval "$SSL .venv/bin/python label_crops.py --provider deepseek --batch 20 --max-requests 45 --max-images 9999 >> $LOG 2>&1"
  sleep 30
done
say "labeling done: $(find storage/dataset/dashcam_raw -name '*.jpg' | wc -l | tr -d ' ') crops total"

# 3) wait for Gemini daily quota reset (~09:00 CEST), then cross-check BOTH models
target_epoch=$(date -v+1H +%s)   # fallback wake time
while [ "$(date +%H)" -lt 8 ]; do sleep 300; done
sleep 600                         # 08:10 — quota window open
say "gemini cross-check starting"
for m in gemini-3.6-flash gemini-3.5-flash-lite; do
  eval "$SSL .venv/bin/python cross_check_labels.py --provider gemini --model $m --max-requests 40 >> $LOG 2>&1" || true
done
say "cross-check done: $(find storage/dataset/quarantine -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ') quarantined total"

# 4) rebuild on verified data
eval "$SSL .venv/bin/python build_merged.py --allow-new --oversample 30 >> $LOG 2>&1"
say "train set rebuilt"

# 5) train v13 from deployed ConvNeXt
nohup bash -c "$SSL exec .venv/bin/python -u train_classifier.py --data storage/dataset/train_v6 --arch convnext_tiny --epochs 10 --batch 64 --lr 4e-4 --init models/r50_dashcam.pt --out models/convnext_v3.pt > /tmp/train_v13.log 2>&1" < /dev/null > /dev/null 2>&1 &
disown
say "v13 training started (init=deployed convnext)"
sleep 90

# 6) wait + compare vs deployed + deploy if +2pts
while pgrep -f train_classifier.py > /dev/null; do sleep 180; done
{
  echo "=== v13 finished $(date) ==="
  tail -12 /tmp/train_v13.log; echo
  eval "$SSL .venv/bin/python compare_ckpts.py --data storage/dataset/dashcam_val \
    --ckpt models/convnext_v1_backup.pt models/convnext_v3.pt"
} > /tmp/v13_compare.txt 2>&1

c1=$(grep -A3 "convnext_v1_backup.pt" /tmp/v13_compare.txt | grep -o "top-1 0\.[0-9]*" | head -1 | awk '{print $2}')
c3=$(grep -A3 "convnext_v3.pt" /tmp/v13_compare.txt | grep -o "top-1 0\.[0-9]*" | head -1 | awk '{print $2}')
DECISION="KEPT current"
if [ -n "${c1:-}" ] && [ -n "${c3:-}" ]; then
  if [ "$(.venv/bin/python -c "print(1 if $c3 >= $c1 + 0.02 else 0)")" = "1" ]; then
    cp models/r50_dashcam.pt models/convnext_v2_kept_backup.pt
    cp models/convnext_v3.pt models/r50_dashcam.pt
    DECISION="DEPLOYED v13 ($c3 vs $c1)"
  else
    DECISION="KEPT current ($c3 vs $c1)"
  fi
fi
say "DECISION: $DECISION"

cat >> PROJECT_NOTES.md <<EOF

### 2026-08-25 overnight — batch 4 flywheel (automated)
- Batch 4: Madrid / rainy Paris / Rome night / Amsterdam / Zurich.
- after_v13.sh ran the full loop incl. morning Gemini verification BEFORE
  training (v12 lesson applied). Outcome: SEE /tmp/v13_compare.txt -> $DECISION
EOF
echo "PIPELINE COMPLETE $(date)" | tee -a "$LOG"
pmset sleepnow
