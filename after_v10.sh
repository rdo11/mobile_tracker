#!/bin/bash
# after_v10.sh — full evening pipeline for batch #2. Runs unattended.
set -u
cd "$HOME/Projects/mobile_tracker"
SRC=storage/driving_videos_src2
LOG=/tmp/v10_pipeline.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
SSL="SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi; print(certifi.where())')"

# 1) wait for the copy to finish
while pgrep -f "cp /Volumes/4TB/Driving videos/second batch" > /dev/null; do sleep 20; done
say "copy done: $(du -sh $SRC | cut -f1)"

# 2) extract multi-crops
say "extraction starting"
nohup bash -c "$SSL exec .venv/bin/python -u extract_crops.py $SRC/*.mp4 > /tmp/extract3.log 2>&1" < /dev/null > /dev/null 2>&1 &
disown
# robust wait: poll the real process, not a subshell pid
while pgrep -f extract_crops.py > /dev/null; do sleep 30; done
say "extraction done: $(tail -1 /tmp/extract3.log)"

# 3) label everything new via DeepSeek (retry loop; parse hiccups are transient)
for round in 1 2 3 4; do
  remaining=$(.venv/bin/python - <<'EOF'
from pathlib import Path
yt = Path("storage/dataset/dashcam_youtube"); raw = Path("storage/dataset/dashcam_raw")
labeled = {p.name for p in raw.rglob("*.jpg")}
print(sum(1 for p in yt.rglob("*.jpg") if p.name not in labeled))
EOF
)
  [ "$remaining" = "0" ] && break
  say "labeling round $round: $remaining unlabeled"
  eval "$SSL .venv/bin/python label_crops.py --provider deepseek --max-requests 15 --max-images 9999 >> $LOG 2>&1"
  sleep 30
done
say "labeling done: $(find storage/dataset/dashcam_raw -name '*.jpg' | wc -l | tr -d ' ') crops total"

# 4) rebuild train set (group-aware holdout inside)
eval "$SSL .venv/bin/python build_merged.py --allow-new --oversample 30 >> $LOG 2>&1"
say "merged: $(grep -c '^' storage/dataset/train_v6 2>/dev/null || echo ok) class dirs"

# 5) train v10 from v9
nohup bash -c "$SSL exec .venv/bin/python -u train_classifier.py --data storage/dataset/train_v6 --arch resnet50 --epochs 8 --batch 128 --lr 3e-4 --init models/r50_dashcam_v9.pt --out models/r50_dashcam_v10.pt > /tmp/train_v10.log 2>&1" < /dev/null > /dev/null 2>&1 &
disown
TRAIN_PID=$!
say "v10 training started (pid $TRAIN_PID)"
caffeinate -w $TRAIN_PID &
sleep 60

# 6) wait, compare, deploy if clearly better (+2pts top-1), notes
while pgrep -f train_classifier.py > /dev/null; do sleep 120; done
{
  echo "=== v10 finished $(date) ==="
  tail -12 /tmp/train_v10.log; echo
  eval "$SSL .venv/bin/python compare_ckpts.py --data storage/dataset/dashcam_val \
    --ckpt models/r50_dashcam_v9.pt models/r50_dashcam_v10.pt"
} > /tmp/v10_compare.txt 2>&1

v9=$(grep -A3 "r50_dashcam_v9.pt" /tmp/v10_compare.txt | grep -o "top-1 0\.[0-9]*" | head -1 | awk '{print $2}')
v10=$(grep -A3 "r50_dashcam_v10.pt" /tmp/v10_compare.txt | grep -o "top-1 0\.[0-9]*" | head -1 | awk '{print $2}')
DECISION="KEPT v9"
if [ -n "${v9:-}" ] && [ -n "${v10:-}" ]; then
  if [ "$(.venv/bin/python -c "print(1 if $v10 >= $v9 + 0.02 else 0)")" = "1" ]; then
    cp models/r50_dashcam_v10.pt models/r50_dashcam.pt
    DECISION="DEPLOYED v10 ($v10 vs $v9)"
  else
    DECISION="KEPT v9 ($v10 vs $v9)"
  fi
fi
say "DECISION: $DECISION"

cat >> PROJECT_NOTES.md <<EOF

### 2026-08-23 evening — batch #2 flywheel (automated)
- Second YT batch: Brussels/Paris/Prague/Frankfurt/Vienna/+1 (290 min).
- Extraction -> DeepSeek labeling -> rebuild -> v10 (from v9 init), all run
  by after_v10.sh. Outcome: SEE /tmp/v10_compare.txt -> $DECISION
- Gemini cross-check of tonight's labels: PENDING (free quota resets next
  morning) — run cross_check_labels.py --provider gemini before trusting v11.
EOF
echo "PIPELINE COMPLETE $(date)" | tee -a "$LOG"
# user asked: keep Mac awake (no pmset sleepnow here)
