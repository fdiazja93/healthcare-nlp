#!/bin/bash

python scripts/run_training.py \
  --run-name biobert-stage1-lr1e-3 --group biobert-stage1 \
  --training-mode head_only --lr 1e-3 \
  --use-class-weights --epochs 5 \
  --seed 42 \
  --model-endpoint dmis-lab/biobert-base-cased-v1.2

python scripts/run_training.py \
  --run-name biobert-stage1-lr5e-4 --group biobert-stage1 \
  --training-mode head_only --lr 5e-4 \
  --use-class-weights --epochs 5 \
  --seed 42 \
  --model-endpoint dmis-lab/biobert-base-cased-v1.2

python scripts/run_training.py \
  --run-name biobert-stage1-lr1e-4 --group biobert-stage1 \
  --training-mode head_only --lr 1e-4 \
  --use-class-weights --epochs 5 \
  --seed 42 \
  --model-endpoint dmis-lab/biobert-base-cased-v1.2
