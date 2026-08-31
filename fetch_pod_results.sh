#!/bin/bash
# fetch_pod_results.sh — pull tonight's artifacts from the pod, deploy winner.
set -e
cd "$HOME/Projects/mobile_tracker"
ssh -o StrictHostKeyChecking=no root@157.157.221.29 -p 56898 -i ~/.ssh/id_ed25519 \
  "cat /tmp/pod_compare.txt" > pod_compare_v12.txt
cat pod_compare_v12.txt
scp -o StrictHostKeyChecking=no -P 56898 -i ~/.ssh/id_ed25519 \
  "root@157.157.221.29:/root/mt/models/convnext_v2.pt" models/ 2>/dev/null
v1=$(grep -A3 "convnext_v1.pt" pod_compare_v12.txt | grep -o "top-1 0\.[0-9]*" | head -1 | awk '{print $2}')
v2=$(grep -A3 "convnext_v2.pt" pod_compare_v12.txt | grep -o "top-1 0\.[0-9]*" | head -1 | awk '{print $2}')
if [ -n "${v2:-}" ] && [ -n "${v1:-}" ] && [ "$(.venv/bin/python -c "print(1 if $v2 >= $v1 + 0.02 else 0)")" = "1" ]; then
  cp models/r50_dashcam.pt models/convnext_v1_backup.pt
  cp models/convnext_v2.pt models/r50_dashcam.pt
  echo "DEPLOYED convnext_v2 ($v2 vs $v1)"
else
  echo "KEPT current model ($v2 vs $v1)"
fi
