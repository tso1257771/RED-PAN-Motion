#!/usr/bin/env bash
# Multi-GPU launcher for EdgeRP90 (edge_rp90_v1), the architecture of the shipped
# edge_rp90 checkpoint: a PConv encoder, a dilated-TCN neck, a decoder with additive
# skips, and sign-split polarity (308,904 parameters, 0.205 GMAC per window). It is
# trained from scratch with the data and recipe of configs/train_rp90_motion_v53.json.
#
#   DATA_ROOT=/your/h5/root bash cloud_training/run_edge_rp90.sh
#
# Environment overrides (all optional):
#   DATA_ROOT     root of the 90 s H5 files     default: /path/to/data/input_h5_90sec
#   NPROC         GPUs, one DDP rank each       default: all visible devices
#   NUM_WORKERS   dataloader workers per rank   default: 4
#   CONFIG        config path                   default: configs/train_edge_rp90.json
#
# batch_size in the config is per rank, so the global batch is batch_size x NPROC
# (48 x 4 = 192 for the shipped checkpoints). On GPUs without bf16 support, such as
# the V100, amp_dtype "auto" resolves to fp16 with a GradScaler.
#
# To resume an interrupted run, set "resume" in the config to the newest
# experiment/edge_rp90/epoch_NNNN.pt and set "resume_optimizer" and "resume_scheduler" to true.
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

: "${DATA_ROOT:=/path/to/data/input_h5_90sec}"
: "${NUM_WORKERS:=4}"
: "${CONFIG:=configs/train_edge_rp90.json}"

# Default NPROC to the GPUs this container can actually open, not a hardcoded 4.
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
    echo "[run_edge_rp90] WARNING: NPROC=${NPROC}, not 4." >&2
    echo "[run_edge_rp90]          global batch = batch_size x NPROC = 48 x ${NPROC} = $((48 * NPROC))" >&2
    echo "[run_edge_rp90]          v49/v53 trained at 192; a different NPROC changes the effective batch." >&2
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

# EdgeRP90 trains from scratch: redpan_motion weights do not fit its layers, so a
# configured warm start is an error.
WARM_START="$(python3 -c "import json;print(json.load(open('${CONFIG}')).get('pretrained_weights') or '')")"
if [[ -n "${WARM_START}" ]]; then
    echo "ERROR: ${CONFIG} sets pretrained_weights='${WARM_START}', but EdgeRP90 is a new" >&2
    echo "       architecture that cannot warm-start from v49. Set pretrained_weights: null" >&2
    echo "       (use 'resume' instead to continue an interrupted edge run)." >&2
    exit 1
fi

MODEL_TYPE="$(python3 -c "import json;print(json.load(open('${CONFIG}')).get('model_type',''))")"
if [[ "${MODEL_TYPE}" != "edge_rp90_v1" ]]; then
    echo "ERROR: ${CONFIG} has model_type='${MODEL_TYPE}', expected 'edge_rp90_v1'." >&2
    exit 1
fi

OUTPUT_DIR="$(python3 -c "import json;print(json.load(open('${CONFIG}'))['output_dir'])")"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/console_$(date +%Y%m%d_%H%M%S).log"

ulimit -n 65536 || true

cat <<EOF
[run_edge_rp90] =====================================================
[run_edge_rp90] Repo:        ${REPO_ROOT}
[run_edge_rp90] CONFIG:      ${CONFIG}   (model_type=edge_rp90_v1, FROM SCRATCH)
[run_edge_rp90] DATA_ROOT:   ${DATA_ROOT}
[run_edge_rp90] OUTPUT_DIR:  ${OUTPUT_DIR}
[run_edge_rp90] GPUs:        ${NPROC}   workers/rank: ${NUM_WORKERS}
[run_edge_rp90] Log:         ${LOG_FILE}
[run_edge_rp90] =====================================================
EOF

torchrun --standalone --nproc_per_node="${NPROC}" \
  scripts/train_rp90_motion.py \
  --config "${CONFIG}" \
  --num-workers "${NUM_WORKERS}" \
  2>&1 | tee "${LOG_FILE}"
