#!/bin/bash
# Full sweep: every model x thread count, one process per configuration so that
# peak-RSS is measured without allocator carry-over between sessions.
#
#   ./run_sweep.sh [tag] [reps] ["thread counts"]
#   ./run_sweep.sh clean 60 "1 2 3 4"
#
# Honours RPM_ONNX_DIR / RPM_RESULTS_DIR / RPM_CKPT_DIR. Reads board power if the
# INA3221 rails are readable (see README section 3), otherwise omits energy.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG=${1:-clean}; REPS=${2:-60}; THREADS=${3:-"1 2 3 4"}
PY=${PYTHON:-python}
RAIL=/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/in_power0_input

IDLE=""
if [ -r "$RAIL" ]; then
  s=0; for _ in $(seq 20); do s=$((s + $(cat "$RAIL"))); sleep 0.1; done
  IDLE="--idle-power $((s / 20))"; echo "idle VDD_IN: $((s / 20)) mW"
else
  echo "power rails not readable (root-only) -- energy columns omitted"
fi

echo "=== RED-PAN device benchmark | tag=$TAG reps=$REPS ==="
echo "before: $(uptime | sed 's/.*load/load/') | mem_avail $(awk '/MemAvailable/{printf "%.0f MB", $2/1024}' /proc/meminfo)"
first=--header
for m in redpan_60s redpan_motion edge_rp90 edge_rp90_int8; do
  for t in $THREADS; do
    $PY "$HERE/bench_models.py" --only "${m}_t${t}" --reps "$REPS" --tag "$TAG" $IDLE $first
    first=""
  done
done
echo "after : $(uptime | sed 's/.*load/load/') | clocks $(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | tr '\n' ' ')"
