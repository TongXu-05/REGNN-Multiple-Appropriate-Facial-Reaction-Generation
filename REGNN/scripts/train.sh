#!/bin/sh

python train.py --logs-dir '/root/autodl-tmp/Action-Generation/new_logs/f50-s25-e8-k6-cog_ind-lip5-nodet-mid'\
                --lr 0.0001 --gamma 0.1 --warmup-factor 0.01 --milestones 9 --batch-size 128 --layers 2\
                --act 'ELU' --seed 1 --train-iters 100 --norm --neighbor-pattern 'all'\
                --convert-type 'direct' --loss-mid