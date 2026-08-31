#!/bin/bash
cd /root
mkdir -p mt/storage/dataset mt/models
tar -xf train_v6.tar -C mt/storage/dataset/
mv r50_dashcam.pt train_classifier.py compare_ckpts.py mt/
pip install --break-system-packages -q opencv-python-headless pillow tqdm
cd mt
rm -f train.log
setsid nohup python -u train_classifier.py --data storage/dataset/train_v6 --arch convnext_base --epochs 12 --batch 32 --size 336 --lr 0.0004 --out models/convnext_base_336.pt > /root/train.log 2>&1 < /dev/null &
echo LAUNCHED
