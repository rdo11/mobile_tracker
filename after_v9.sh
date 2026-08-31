#!/bin/bash
# after_v9.sh — overnight finisher. Runs AFTER training exits:
#   1. compare v5/v6/v9 on the real-crop holdout  -> /tmp/v9_compare.txt
#   2. deploy v9 ONLY if it beats v6 by >= +2 pts top-1 (safety margin)
#   3. append outcome to PROJECT_NOTES.md
#   4. put the Mac to sleep
set -u
cd "$HOME/Projects/mobile_tracker"
OUT=/tmp/v9_compare.txt

# 1) wait for training to end (never signals it)
while pgrep -f train_classifier.py > /dev/null; do sleep 60; done

{
  echo "=== v9 finished $(date) ==="
  tail -12 /tmp/train_v9.log
  echo
  SSL_CERT_FILE="$(.venv/bin/python -c 'import certifi; print(certifi.where())')" \
    .venv/bin/python compare_ckpts.py --data storage/dataset/dashcam_val \
      --ckpt models/r50_dashcam_v5_backup.pt models/r50_dashcam_v6.pt models/r50_dashcam_v9.pt
} > "$OUT" 2>&1

# 2) conditional deploy: parse top-1 of v6 vs v9 lines from the report
v6=$(grep -A3 "r50_dashcam_v6.pt" "$OUT" | grep -o "top-1 0\.[0-9]*" | head -1 | awk '{print $2}')
v9=$(grep -A3 "r50_dashcam_v9.pt" "$OUT" | grep -o "top-1 0\.[0-9]*" | head -1 | awk '{print $2}')
decision="KEPT v6 (v9 not clearly better or parse failed)"
if [ -n "${v6:-}" ] && [ -n "${v9:-}" ]; then
  better=$(.venv/bin/python -c "print(1 if $v9 >= $v6 + 0.02 else 0)")
  if [ "$better" = "1" ]; then
    cp models/r50_dashcam_v9.pt models/r50_dashcam.pt
    decision="DEPLOYED v9 (holdout $v9 vs $v6)"
  else
    decision="KEPT v6 (holdout $v9 vs $v6 — gain < 2 pts)"
  fi
fi
echo "DECISION: $decision" >> "$OUT"

# 3) notes
cat >> PROJECT_NOTES.md <<EOF

## 15. OVERNIGHT SESSION (2026-08-22 -> 23)

- Migrated to M5 MacBook Pro 24GB; env rebuilt; MPS training VALIDATED here
  (2-class protocol: loss drops normally). M5 = ~5-6x faster inference than
  M2 Air; r50 epochs ~10 min locally; RunPod only worth it for big jobs.
- THE BIG BUG: AsyncClassifier froze classifier.available at construction
  (before model load) -> local labels NEVER appeared live. Fixed with a live
  property. Land Rover outputs had been Grok masking this.
- v6 retrained on M5 (285 cl): holdout 31% -> 53.3% top-1. Deployed;
  r50_dashcam_v5_backup.pt kept. Zero LR bias in EU-drive replay.
- Pedestrian/coarse labels fixed (person/bicycle were excluded before).
- label_crops.py: --provider gemini|deepseek; DeepSeek needs
  "thinking":{"type":"disabled"} (else content comes back empty); skips
  already-labeled crops; folds accents at save time. 139 unlabeled crops
  labeled via DeepSeek tonight (+51 earlier via Gemini) -> dashcam_raw=578.
- build_merged.py (reproducible canon/variant-fold/oversample/holdout):
  305 classes, ~25.8k imgs, 56-crop holdout in dashcam_val.
- v9 (305 cl, all new data) trained overnight; result + deploy decision:
  SEE /tmp/v9_compare.txt -> $decision
- Recordings have boxes burned in (recorder writes annotated buffer) — use
  clean YouTube footage for crop mining. 3 long raw YT drives NOT yet on the
  4TB HDD (searched; user will re-download).
- Report rewritten for the M5 era: PROJECT_REPORT.txt.

### Next steps
1. Read /tmp/v9_compare.txt; if KEPT v6, consider more epochs/data next time
2. Label more crops when Gemini quota resets (free) or via DeepSeek
3. Restart dashboard when wanted: SSL_CERT_FILE=\$(.venv/bin/python -c "import certifi; print(certifi.where())") .venv/bin/python main.py &
4. RunPod one-command launcher still to be built (needs fresh RUNPOD_API_KEY)
5. git init; HF dataset upload; blur_plates=true before sharing
EOF

# 4) sleep the Mac (safe; nothing running anymore)
sleep 20
pmset sleepnow 2>/dev/null || osascript -e 'tell application "System Events" to shut down' 2>/dev/null || true
