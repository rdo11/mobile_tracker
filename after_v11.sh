#!/bin/bash
# after_v11.sh — batch-3 overnight mega-pipeline.
# copy -> [wait v10] -> extract -> deepseek-label -> rebuild -> v11 -> compare/deploy -> notes -> sleep
set -u
cd "$HOME/Projects/mobile_tracker"
SRC="/Volumes/4TB/Driving videos/third batch"
DST=storage/driving_videos_src3
LOG=/tmp/v11_pipeline.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
SSL="SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi; print(certifi.where())')"

# 1) copy batch 3 (I/O only — safe during v10 training).
# Robust: wait for an ALREADY-RUNNING copy first, then verify every source
# file exists locally with size >= source before declaring done.
say "syncing batch 3..."
mkdir -p "$DST"
if ! pgrep -f "Driving videos/third batch" > /dev/null; then
  nohup cp "$SRC"/*.mp4 "$DST/" > /tmp/vidcopy3.log 2>&1 &
fi
sleep 5   # let any fresh cp spawn before polling (race-condition lesson)
while pgrep -f "cp /Volumes/4TB/Driving videos/third batch" > /dev/null; do sleep 20; done
# verify: same file count and none smaller than source
missing=0
for f in "$SRC"/*.mp4; do
  b=$(basename "$f")
  s_src=$(stat -f%z "$f" 2>/dev/null || echo 0)
  s_dst=$(stat -f%z "$DST/$b" 2>/dev/null || echo 0)
  [ "$s_dst" -lt "$s_src" ] && missing=1 && say "incomplete: $b ($s_dst < $s_src)"
done
if [ "$missing" = "1" ]; then
  say "re-copying incomplete files"
  for f in "$SRC"/*.mp4; do
    b=$(basename "$f")
    s_src=$(stat -f%z "$f"); s_dst=$(stat -f%z "$DST/$b" 2>/dev/null || echo 0)
    if [ "$s_dst" -lt "$s_src" ]; then cp "$f" "$DST/$b"; fi
  done
fi
say "COPY VERIFIED: $(du -sh $DST | cut -f1) — HDD CAN BE DISCONNECTED NOW"

# 2) wait for v10 training + any cross-check to finish (GPU free)
while pgrep -f train_classifier.py > /dev/null; do sleep 60; done
while pgrep -f cross_check_labels.py > /dev/null; do sleep 60; done
say "GPU free — extracting batch 3"

# 3) extraction (multi-crop harvester)
nohup bash -c "$SSL exec .venv/bin/python -u extract_crops.py $DST/*.mp4 > /tmp/extract4.log 2>&1" < /dev/null > /dev/null 2>&1 &
disown
while pgrep -f extract_crops.py > /dev/null; do sleep 60; done
say "extraction done: $(tail -1 /tmp/extract4.log)"

# 4) deepseek labeling (batch 20 + generous max_tokens; retry rounds)
for round in 1 2 3 4 5 6; do
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
  sleep 45
done
say "labeling done: $(find storage/dataset/dashcam_raw -name '*.jpg' | wc -l | tr -d ' ') crops total"

# 5) rebuild train set
eval "$SSL .venv/bin/python build_merged.py --allow-new --oversample 30 >> $LOG 2>&1"
say "train set rebuilt"

# 6) v11 from currently-deployed best
DEPLOYED=$(ls -la models/r50_dashcam.pt | awk '{print $NF}')
nohup bash -c "$SSL exec .venv/bin/python -u train_classifier.py --data storage/dataset/train_v6 --arch resnet50 --epochs 8 --batch 128 --lr 3e-4 --init models/r50_dashcam.pt --out models/r50_dashcam_v11.pt > /tmp/train_v11.log 2>&1" < /dev/null > /dev/null 2>&1 &
disown
say "v11 training started (base=$DEPLOYED)"
sleep 90

# 7) wait + compare vs deployed
while pgrep -f train_classifier.py > /dev/null; do sleep 120; done
{
  echo "=== v11 finished $(date) ==="
  tail -12 /tmp/train_v11.log; echo
  eval "$SSL .venv/bin/python compare_ckpts.py --data storage/dataset/dashcam_val \
    --ckpt models/r50_dashcam_v9.pt models/r50_dashcam_v10.pt models/r50_dashcam_v11.pt"
} > /tmp/v11_compare.txt 2>&1

v10=$(grep -A3 "r50_dashcam_v10.pt" /tmp/v11_compare.txt | grep -o "top-1 0\.[0-9]*" | head -1 | awk '{print $2}')
v11=$(grep -A3 "r50_dashcam_v11.pt" /tmp/v11_compare.txt | grep -o "top-1 0\.[0-9]*" | head -1 | awk '{print $2}')
DECISION="KEPT current (v11 gain < 2pts or parse issue)"
if [ -n "${v10:-}" ] && [ -n "${v11:-}" ]; then
  if [ "$(.venv/bin/python -c "print(1 if $v11 >= $v10 + 0.02 else 0)")" = "1" ]; then
    cp models/r50_dashcam_v11.pt models/r50_dashcam.pt
    DECISION="DEPLOYED v11 ($v11 vs $v10)"
  else
    DECISION="KEPT v10 ($v11 vs $v10)"
  fi
fi
say "DECISION: $DECISION"

cat >> PROJECT_NOTES.md <<EOF

### 2026-08-24 overnight — batch #3 mega-run (automated)
- Batch 3: 17 videos / 26GB — DK x3, PL x2, IT, HU, HR, SK, PT, LU, BE, AT,
  UK-night. Full flywheel run by after_v11.sh.
- Labels are DeepSeek-only tonight; Gemini cross-check STILL PENDING — run
  cross_check_labels.py --provider gemini (cycle models: 3.5-flash-lite /
  3.6-flash) before trusting v12 training on these.
- Outcome: SEE /tmp/v11_compare.txt -> $DECISION
EOF
echo "PIPELINE COMPLETE $(date)" | tee -a "$LOG"
pmset sleepnow
