#!/bin/bash
# daily_verify.sh — run Gemini verification daily (quota resets), resume-safe,
# saves results, and after each pass checks if enough is verified to build the
# cleaned dataset.
cd /Users/radovanhloska/Projects/mobile_tracker

echo "=== DAILY VERIFY pass $(date) ==="
.venv/bin/python -u verify_all_labels.py --provider gemini 2>&1 | tail -5

echo "=== progress check ==="
DONE=$(cat verify_results/agree.txt verify_results/disagree.txt verify_results/lowconf.txt 2>/dev/null | wc -l | tr -d ' ')
echo "verified so far: $DONE / 21284 crops"

# when >= 40% verified, build the cleaned subset
if [ "$DONE" -ge 8500 ]; then
  echo "=== enough verified — building cleaned dataset ==="
  .venv/bin/python -u build_clean_dataset.py 2>&1 | tail -8
fi
echo "=== pass complete $(date) ==="
