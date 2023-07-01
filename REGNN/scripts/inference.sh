#!/bin/sh

python train_final.py --test --model-pth "new_logs/1e3-sd1-50-t3-noUDIVA-all-lip3-layer2-st25/mhp-epoch95-seed1.pth" \
                      --neighbor-pattern 'nearest' --seed 1