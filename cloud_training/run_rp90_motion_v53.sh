#!/usr/bin/env bash
# Multi-GPU launcher for RP90-Motion v53: a warm start from the shipped redpan_motion
# checkpoint (checkpoints/redpan_motion/best.pt) for a second 200-epoch cosine cycle
# at the same width. The five single-event S-P bins are sampled in proportion to the
# square root of each bin's pool size, and best.pt is selected on the picker and
# detector validation losses. No checkpoint from this configuration is shipped.
#
#   DATA_ROOT=/your/h5/root bash cloud_training/run_rp90_motion_v53.sh
#
# Environment overrides (all optional):
#   DATA_ROOT     root of the 90 s H5 files     default: /home/$USER/input_h5_90sec
#   NPROC         GPUs, one DDP rank each       default: all visible devices
#   NUM_WORKERS   dataloader workers per rank   default: 4
#   CONFIG        config path                   default: configs/train_rp90_motion_v53.json
#
# batch_size in the config is per rank, so the global batch is batch_size x NPROC
# (48 x 4 = 192 for the shipped checkpoints). On GPUs without bf16 support, such as
# the V100, amp_dtype "auto" resolves to fp16 with a GradScaler.
#
# To resume an interrupted run, set "resume" in the config to the newest
# experiment/rp90_motion_v53/epoch_NNNN.pt and set "resume_optimizer" and "resume_scheduler" to true.
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
: "${CONFIG:=configs/train_rp90_motion_v53.json}"

# Default NPROC to the GPUs this container can actually open, not a hardcoded 4.
# torchrun spawns one rank per NPROC and each rank calls torch.cuda.set_device;
# asking for more devices than the flavor allocated fails late and unhelpfully
# ("CUDA-capable device(s) is/are busy or unavailable") only on the extra ranks.
if [[ -z "${NPROC:-}" ]]; then
    NPROC="$(python3 -c "import torch;print(torch.cuda.device_count())" 2>/dev/null || echo 0)"
fi
if [[ "${NPROC}" -lt 1 ]]; then
    echo "ERROR: no usable CUDA device. Diagnose with:" >&2
    echo "  nvidia-smi --query-gpu=index,name,memory.used,compute_mode --format=csv" >&2
    echo "  python3 -c 'import torch;print(torch.cuda.is_available(), torch.cuda.device_count())'" >&2
    echo "  echo \"CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-unset}\"" >&2
    exit 1
fi
if [[ "${NPROC}" -ne 4 ]]; then
    # batch_size in the config is PER RANK (it goes straight to each rank's
    # DataLoader alongside a DistributedSampler), so the global batch is
    # batch_size x NPROC. v49 trained at 48 x 4 = 192; a different NPROC changes
    # the effective batch and makes v53 no longer a clean comparison to v49.
    echo "[run_rp90_motion_v53] WARNING: NPROC=${NPROC}, not 4." >&2
    echo "[run_rp90_motion_v53]          global batch = batch_size x NPROC = 48 x ${NPROC} = $((48 * NPROC))" >&2
    echo "[run_rp90_motion_v53]          v49 trained at 192. Results will not be directly comparable." >&2
fi

export DATA_ROOT
# torch.compile gives no benefit on V100 (CC 7.0) and costs startup time.
export TORCH_COMPILE_DISABLE=1

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: config not found: ${CONFIG}" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "ERROR: DATA_ROOT not found: ${DATA_ROOT}" >&2
    echo "       Override with: DATA_ROOT=/your/path bash $0" >&2
    exit 1
fi

# v53 is a warm start, so a missing pretrained_weights is an error rather than a
# silent run from random weights.
WARM_START="$(python3 -c "import json;print(json.load(open('${CONFIG}')).get('pretrained_weights') or '')")"
if [[ -z "${WARM_START}" ]]; then
    echo "ERROR: ${CONFIG} has no pretrained_weights — v53 is a warm start from v49." >&2
    exit 1
fi
if [[ ! -f "${WARM_START}" ]]; then
    echo "ERROR: warm-start checkpoint not found: ${WARM_START}" >&2
    echo "       checkpoints/ is tracked in git; re-sync the repo." >&2
    exit 1
fi

OUTPUT_DIR="$(python3 -c "import json;print(json.load(open('${CONFIG}'))['output_dir'])")"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/console_$(date +%Y%m%d_%H%M%S).log"

# Many H5 files across 4 ranks x 4 workers need many open file handles.
ulimit -n 65536 || true

cat <<EOF
[run_rp90_motion_v53] =====================================================
[run_rp90_motion_v53] Repo:        ${REPO_ROOT}
[run_rp90_motion_v53] CONFIG:      ${CONFIG}
[run_rp90_motion_v53] DATA_ROOT:   ${DATA_ROOT}
[run_rp90_motion_v53] Warm start:  ${WARM_START}
[run_rp90_motion_v53] OUTPUT_DIR:  ${OUTPUT_DIR}
[run_rp90_motion_v53] GPUs:        ${NPROC}   workers/rank: ${NUM_WORKERS}
[run_rp90_motion_v53] Log:         ${LOG_FILE}
[run_rp90_motion_v53] =====================================================
EOF

torchrun --standalone --nproc_per_node="${NPROC}" \
  scripts/train_rp90_motion.py \
  --config "${CONFIG}" \
  --num-workers "${NUM_WORKERS}" \
  2>&1 | tee "${LOG_FILE}"
