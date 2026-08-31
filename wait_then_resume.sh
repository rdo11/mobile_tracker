#!/bin/bash
# wait_then_resume.sh — waits for the current wikimedia scraper to finish,
# then launches a resume pass (skips already-done models, scrapes only new
# entries like the added EV models).
DIR="$(cd "$(dirname "$0")" && pwd)"
while pgrep -f "wikimedia_scraper\\.py" >/dev/null; do
  sleep 120
done
sleep 5
cd "$DIR"
nohup .venv/bin/python -u wikimedia_scraper.py --all --resume >> /tmp/wm_scrape.log 2>&1 &
