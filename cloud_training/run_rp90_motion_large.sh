#!/usr/bin/env bash
# Multi-GPU launcher for RP90-Motion large: the MTAN R2U-Net of redpan_motion widened
# from nb_filters [8,16,24,32,40] to [12,24,36,48,60] (1,183,404 parameters), trained
# from scratch with the square-root tempered S-P sampling of
# configs/train_rp90_motion_v53.json and more regularization (dropout 0.15,
# weight_decay 2e-4). No checkpoint from this configuration is shipped.
#
#   DATA_ROOT=/your/h5/root bash cloud_training/run_rp90_motion_large.sh
#
# Environment overrides (all optional):
#   DATA_ROOT     root of the 90 s H5 files     default: /home/$USER/input_h5_90sec
#   NPROC         GPUs, one DDP rank each       default: all visible devices
#   NUM_WORKERS   dataloader workers per rank   default: 4
#   CONFIG        config path                   default: configs/train_rp90_motion_large.json
#
# batch_size in the config is per rank, so the global batch is batch_size x NPROC
# (48 x 4 = 192 for the shipped checkpoints). On GPUs without bf16 support, such as
# the V100, amp_dtype "auto" resolves to fp16 with a GradScaler.
#
# To resume an interrupted run, set "resume" in the config to the newest
# experiment/rp90_motion_large/epoch_NNNN.pt and set "resume_optimizer" and "resume_scheduler" to true.
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
: "${CONFIG:=configs/train_rp90_motion_large.json}"

if [[ -z "${NPROC:-}" ]]; then
    NPROC="$(python3 -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)"
fi
if [[ "${NPROC}" -lt 1 ]]; then
    echo "ERROR: no usable CUDA device. Diagnose with:" >&2
    echo "  nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv" >&2
    echo "  python3 -c 'import torch;print(torch.cuda.is_available(), torch.cuda.device_count())'" >&2
    exit 1
fi
if [[ "${NPROC}" -ne 4 ]]; then
    echo "[run_rp90_motion_large] WARNING: NPROC=${NPROC}, not 4." >&2
    echo "[run_rp90_motion_large]          global batch = batch_size x NPROC = 48 x ${NPROC} = $((48 * NPROC))" >&2
    echo "[run_rp90_motion_large]          v49/v53 trained at 192; a different global batch is a recipe change." >&2
fi

export DATA_ROOT
export TORCH_COMPILE_DISABLE=1   # no benefit on V100 (CC 7.0), costs startup time

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config not found: ${CONFIG}" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}" >&2
    echo "       Override with: DATA_ROOT=/your/path bash $0" >&2
    exit 1
fi

# The large model trains from scratch at a new width, so a configured warm start is
# an error: a redpan_motion checkpoint has the wrong width and would not load.
WARM_START="$(python3 -c "import json;print(json.load(open('${CONFIG}')).get('pretrained_weights') or '')")"
if [[ -n "${WARM_START}" ]]; then
    echo "ERROR: ${CONFIG} sets pretrained_weights='${WARM_START}', but LARGE is from-scratch" >&2
    echo "       at a new width. Set it to null, or (for a resume) use 'resume' instead." >&2
    exit 1
fi

OUTPUT_DIR="$(python3 -c "import json;print(json.load(open('${CONFIG}'))['output_dir'])")"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/console_$(date +%Y%m%d_%H%M%S).log"

ulimit -n 65536 || true

cat <<EOF
[run_rp90_motion_large] =====================================================
[run_rp90_motion_large] Repo:        ${REPO_ROOT}
[run_rp90_motion_large] CONFIG:      ${CONFIG}   (nb_filters [12,24,36,48,60], 1.18M params, FROM SCRATCH)
[run_rp90_motion_large] DATA_ROOT:   ${DATA_ROOT}
[run_rp90_motion_large] OUTPUT_DIR:  ${OUTPUT_DIR}
[run_rp90_motion_large] GPUs:        ${NPROC}   workers/rank: ${NUM_WORKERS}
[run_rp90_motion_large] Log:         ${LOG_FILE}
[run_rp90_motion_large] =====================================================
EOF

torchrun --standalone --nproc_per_node="${NPROC}" \
  scripts/train_rp90_motion.py \
  --config "${CONFIG}" \
  --num-workers "${NUM_WORKERS}" \
  2>&1 | tee "${LOG_FILE}"
