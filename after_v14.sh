#!/bin/bash
# after_v14.sh — morning: verify batch-4 + tail labels with Gemini, rebuild,
# then prepare a pod-ready dataset bundle for the user to train on vast.ai/runpod.
set -u
cd "$HOME/Projects/mobile_tracker"
LOG=/tmp/v14_pipeline.log
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
SSL="SSL_CERT_FILE=$(.venv/bin/python -c 'import certifi; print(certifi.where())')"

# wait until the pod training for today is confirmed done (user-driven; just wait
# until local GPU is quiet too so we don't contend)
while pgrep -f "train_classifier|extract_crops|label_crops" > /dev/null; do sleep 60; done

# 1) wait for Gemini quota reset (~08:10 CEST) then verify batch-4 + tail labels
while [ "$(date +%H)" -lt 8 ]; do sleep 300; done
sleep 600
say "gemini cross-check (batch-4 + tail labels)"
for m in gemini-3.6-flash gemini-3.5-flash-lite; do
  eval "$SSL .venv/bin/python cross_check_labels.py --provider gemini --model $m --max-requests 45 >> $LOG 2>&1" || true
done
say "cross-check done; quarantine now: $(find storage/dataset/quarantine -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')"

# 2) rebuild train set on the verified pool
eval "$SSL .venv/bin/python build_merged.py --allow-new --oversample 30 >> $LOG 2>&1"
say "rebuilt: $(find storage/dataset/train_v6 -name '*.jpg' | wc -l | tr -d ' ') train imgs, $(find storage/dataset/dashcam_val -name '*.jpg' | wc -l | tr -d ' ') holdout"

# 3) package a pod-ready bundle (dataset tar + scripts + deployed init)
cd storage/dataset && tar -cf /tmp/v14_bundle.tar train_v6 dashcam_val 2>/dev/null
cd ~/Projects/mobile_tracker && cp train_classifier.py compare_ckpts.py models/r50_dashcam.pt /tmp/
say "POD-READY BUNDLE: /tmp/v14_bundle.tar (upload + train v14 on pod)"
echo "V14_READY $(date)" | tee -a "$LOG"
