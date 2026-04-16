#!/bin/bash

python infra/launch_training.py \
  --group stage2 \
  --training-mode top_n \
  --n-layers 2 \
  --lr 5e-6 \
  --epochs 5

python infra/launch_training.py \
  --group stage2 \
  --training-mode top_n \
  --n-layers 4 \
  --lr 5e-6 \
  --epochs 5

python infra/launch_training.py \
  --group stage2 \
  --training-mode top_n \
  --n-layers 2 \
  --lr 1e-5 \
  --epochs 5
