#!/bin/bash
# Train the RP90-Motion model (single production architecture).
# Point DATA_ROOT at the directory of *_dataset_90s*.h5 files, then:
#   bash cloud_training/run_rp90_motion.sh
# Single-GPU:  python scripts/train_rp90_motion.py --config configs/train_rp90_motion.json
set -e
export DATA_ROOT="${DATA_ROOT:-/path/to/input_h5_90sec}"
export TORCH_COMPILE_DISABLE=1
NPROC="${NPROC:-4}"

torchrun --nproc_per_node="${NPROC}" \
  scripts/train_rp90_motion.py \
  --config configs/train_rp90_motion.json \
  --num-workers 4
