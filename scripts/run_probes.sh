#!/usr/bin/env bash
# Driver for the linear-probe extraction sweep (PRISM Axis 1).
#
# Runs `interpretability/probe_extract.py` for the 5 must-have checkpoints
# × 3 benchmarks = 15 jobs. Writes `.pt` caches into
# OUT_DIR for `probe_train.py` to consume.
#
# Single-GPU sequential. Override CUDA_VISIBLE_DEVICES to pick a different
# GPU. Total wall-clock ~2 hours on a free H100.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_probes.sh
#
# Or to run a single variant for debugging:
#   CUDA_VISIBLE_DEVICES=1 ONLY_VARIANT=lvr_baseline bash scripts/run_probes.sh

set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON_BIN:-$(which python)}
OUT_DIR=${OUT_DIR:-${WORKSPACE}/interpretability_results/probes_20260514}
STEPS=${STEPS:-8}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-16}
LIMIT=${LIMIT:-0}

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# (variant_tag, checkpoint_path) pairs. Paths match each finetune_*.sh's
# OUTPUT_DIR exactly; if you change a script's output dir, update the matching
# entry here. Pass ONLY_VARIANT=<tag> to limit scope.
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
    if [ -n "${ONLY_VARIANT:-}" ] && [ "$variant" != "$ONLY_VARIANT" ]; then
        continue
    fi
    if [ ! -d "$ckpt" ]; then
        echo "[run_probes] WARN: missing checkpoint $ckpt for $variant; skipping"
        continue
    fi

    for bench in "${BENCHMARKS[@]}"; do
        if [ -n "${ONLY_BENCH:-}" ] && [ "$bench" != "$ONLY_BENCH" ]; then
            continue
        fi
        out_file="$OUT_DIR/extract_${variant}_${bench}_seed${SEED}.pt"
        if [ -f "$out_file" ] && [ "${FORCE:-0}" != "1" ]; then
            echo "[run_probes] SKIP existing: $out_file (set FORCE=1 to overwrite)"
            continue
        fi
        log="$LOG_DIR/${variant}_${bench}.log"
        echo "[run_probes] $(date +%H:%M:%S) variant=$variant bench=$bench -> $out_file"
        "$PY" interpretability/probe_extract.py \
            --ckpt "$ckpt" \
            --variant "$variant" \
            --benchmark "$bench" \
            --steps "$STEPS" \
            --seed "$SEED" \
            --batch-size "$BATCH_SIZE" \
            --out-dir "$OUT_DIR" \
            $LIMIT_ARG 2>&1 | tee "$log"
    done
done

echo "[run_probes] all done. caches in $OUT_DIR"
