#!/bin/bash
# hdd_download.sh — download new dashcam videos from links_todo.txt to the 4TB HDD.
# 1080p mp4, 8 parallel fragments, resumable, skips existing files.
set -u
cd "$(dirname "$0")"
OUT="/Volumes/4TB/Driving videos/seventh batch"
mkdir -p "$OUT"
LOG=/tmp/hdd_download.log

echo "=== HDD DOWNLOAD started $(date) ===" > "$LOG"
echo "output: $OUT" >> "$LOG"
echo "links: $(wc -l < /tmp/links_todo.txt)" >> "$LOG"

.venv/bin/python -m yt_dlp -a /tmp/links_todo.txt \
  -f "bv[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]" \
  --merge-output-format mp4 \
  -N 8 --retries 10 --fragment-retries 10 \
  --no-overwrites --continue \
  --progress \
  -o "$OUT/%(title).80s_%(id)s_1080p.%(ext)s" \
  >> "$LOG" 2>&1

echo "=== HDD DOWNLOAD DONE $(date) ===" >> "$LOG"
