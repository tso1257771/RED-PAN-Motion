#!/usr/bin/env bash
# Multi-GPU launcher for RP90-Motion XL: the MTAN R2U-Net of redpan_motion widened
# from nb_filters [8,16,24,32,40] to [16,32,48,64,80] (2,071,048 parameters), trained
# from scratch with the sampling of configs/train_rp90_motion_v53.json and more
# regularization (dropout 0.20, weight_decay 3e-4). No checkpoint from this
# configuration is shipped.
#
#   DATA_ROOT=/your/h5/root bash cloud_training/run_rp90_motion_xlarge.sh
#
# Environment overrides (all optional):
#   DATA_ROOT     root of the 90 s H5 files     default: /home/$USER/input_h5_90sec
#   NPROC         GPUs, one DDP rank each       default: all visible devices
#   NUM_WORKERS   dataloader workers per rank   default: 4
#   CONFIG        config path                   default: configs/train_rp90_motion_xlarge.json
#
# batch_size in the config is per rank, so the global batch is batch_size x NPROC
# (48 x 4 = 192 for the shipped checkpoints). On GPUs without bf16 support, such as
# the V100, amp_dtype "auto" resolves to fp16 with a GradScaler.
#
# To resume an interrupted run, set "resume" in the config to the newest
# experiment/rp90_motion_xlarge/epoch_NNNN.pt and set "resume_optimizer" and "resume_scheduler" to true.
# "pretrained_weights" loads weights only and restarts the schedule.
#
# Each run writes best.pt (selected on the picker and detector validation losses),
# best_total.pt (selected on the total loss) and epoch_NNNN.pt every 5 epochs. The
# validation loss is not a reliable guide to picking skill, so compare candidates
# on held-out data, for example with scripts/benchmarks/benchmark_stead.py.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

: "${DATA_ROOT:=/home/${USER}/input_h5_90sec}"
: "${NUM_WORKERS:=4}"
: "${CONFIG:=configs/train_rp90_motion_xlarge.json}"

if [[ -z "${NPROC:-}" ]]; then
    NPROC="$(python3 -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)"
fi
if [[ "${NPROC}" -lt 1 ]]; then
    echo "ERROR: no usable CUDA device. Diagnose: nvidia-smi ; python3 -c 'import torch;print(torch.cuda.device_count())'" >&2
    exit 1
fi
if [[ "${NPROC}" -ne 4 ]]; then
    echo "[run_rp90_motion_xlarge] WARNING: NPROC=${NPROC} (not 4); global batch = 48 x ${NPROC} = $((48 * NPROC)) vs the ladder's 192." >&2
fi

export DATA_ROOT
export TORCH_COMPILE_DISABLE=1

[[ -f "${CONFIG}" ]] || { echo "ERROR: config not found: ${CONFIG}" >&2; exit 1; }
[[ -d "${DATA_ROOT}" ]] || { echo "ERROR: DATA_ROOT not found: ${DATA_ROOT} (override: DATA_ROOT=/path bash $0)" >&2; exit 1; }

# XL trains from scratch at a new width, so a configured warm start is an error.
WARM_START="$(python3 -c "import json;print(json.load(open('${CONFIG}')).get('pretrained_weights') or '')")"
if [[ -n "${WARM_START}" ]]; then
    echo "ERROR: ${CONFIG} sets pretrained_weights='${WARM_START}', but XL is from-scratch at a new width." >&2
    echo "       Set it to null, or use 'resume' for a mid-run restart." >&2
    exit 1
fi

OUTPUT_DIR="$(python3 -c "import json;print(json.load(open('${CONFIG}'))['output_dir'])")"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/console_$(date +%Y%m%d_%H%M%S).log"
ulimit -n 65536 || true

cat <<EOF
[run_rp90_motion_xlarge] =====================================================
[run_rp90_motion_xlarge] CONFIG:      ${CONFIG}   (nb_filters [16,32,48,64,80], 2.07M params, FROM SCRATCH)
[run_rp90_motion_xlarge] DATA_ROOT:   ${DATA_ROOT}
[run_rp90_motion_xlarge] OUTPUT_DIR:  ${OUTPUT_DIR}
[run_rp90_motion_xlarge] GPUs:        ${NPROC}   workers/rank: ${NUM_WORKERS}
[run_rp90_motion_xlarge] Log:         ${LOG_FILE}
[run_rp90_motion_xlarge] =====================================================
EOF

torchrun --standalone --nproc_per_node="${NPROC}" \
  scripts/train_rp90_motion.py \
  --config "${CONFIG}" \
  --num-workers "${NUM_WORKERS}" \
  2>&1 | tee "${LOG_FILE}"
