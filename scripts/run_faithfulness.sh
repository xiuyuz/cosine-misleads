#!/usr/bin/env bash
# Driver for the faithfulness corruption sweep (PRISM Axis 2).
#
# Runs `interpretability/faithfulness_corrupt.py` on the 5 must-have ckpts
# × 3 benchmarks = 15 jobs. Each job runs 6 corruption modes.
#
# Total wall-clock: ~2-3 hours on 2 GPUs.

set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON_BIN:-$(which python)}
OUT_DIR=${OUT_DIR:-${WORKSPACE}/interpretability_results/faithfulness_20260515}
STEPS=${STEPS:-8}
PLVR_STEPS=${PLVR_STEPS:-16}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-1}
DONOR_POOL=${DONOR_POOL:-64}
LIMIT=${LIMIT:-0}

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

declare -a VARIANTS=(
    "lvr_baseline ${WORKSPACE}/checkpoints/stage1/checkpoint-2500"
    "nlvr         ${WORKSPACE}/nlvr/checkpoints/stage1/checkpoint-2500"
    "dlvr_a       ${WORKSPACE}/dlvr/checkpoints/dlvr_a/checkpoint-2500"
    "plvr2        ${WORKSPACE}/plvr2/checkpoints/stage1/checkpoint-2500"
    "plvr3        ${WORKSPACE}/plvr3/checkpoints/stage1/checkpoint-2500"
)
BENCHMARKS=(vstar MMVP blink)

LIMIT_ARG=""
if [ "$LIMIT" != "0" ]; then
    LIMIT_ARG="--limit $LIMIT"
fi

for entry in "${VARIANTS[@]}"; do
    variant=$(echo "$entry" | awk '{print $1}')
    ckpt=$(echo "$entry" | awk '{print $2}')
    if [ -n "${ONLY_VARIANT:-}" ] && [ "$variant" != "$ONLY_VARIANT" ]; then continue; fi
    if [ ! -d "$ckpt" ]; then
        echo "[run_faith] WARN: missing checkpoint $ckpt for $variant; skipping"
        continue
    fi
    for bench in "${BENCHMARKS[@]}"; do
        if [ -n "${ONLY_BENCH:-}" ] && [ "$bench" != "$ONLY_BENCH" ]; then continue; fi
        out_file="$OUT_DIR/faith_${variant}_${bench}_seed${SEED}.json"
        if [ -f "$out_file" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "[run_faith] SKIP existing: $out_file (set FORCE=1 to overwrite)"
            continue
        fi
        run_steps="$STEPS"
        case "$variant" in
            plvr2|plvr3) run_steps="$PLVR_STEPS" ;;
        esac
        log="$LOG_DIR/${variant}_${bench}.log"
        echo "[run_faith] $(date +%H:%M:%S) variant=$variant bench=$bench steps=$run_steps -> $out_file"
        "$PY" interpretability/faithfulness_corrupt.py \
            --ckpt "$ckpt" \
            --variant "$variant" \
            --benchmark "$bench" \
            --steps "$run_steps" \
            --seed "$SEED" \
            --batch-size "$BATCH_SIZE" \
            --donor-pool-size "$DONOR_POOL" \
            --out-dir "$OUT_DIR" \
            $LIMIT_ARG 2>&1 | tee "$log"
    done
done

echo "[run_faith] all done. results in $OUT_DIR"
