#!/bin/bash
python scripts/run_training.py \
  --run-name stage1-lr1e-3 --group stage1 \
  --training-mode head_only --lr 1e-3 \
  --use-class-weights --epochs 5 \
  --seed 42

python scripts/run_training.py \
  --run-name stage1-lr5e-4 --group stage1 \
  --training-mode head_only --lr 5e-4 \
  --use-class-weights --epochs 5 \
  --seed 42

python scripts/run_training.py \
  --run-name stage1-lr1e-4 --group stage1 \
  --training-mode head_only --lr 1e-4 \
  --use-class-weights --epochs 5 \
  --seed 42
