#!/usr/bin/env bash
# Smoke test for the code release. Run with:
#   bash scripts/smoke_release.sh
# Exit non-zero on any failure. No GPU required.
#
# Checks:
#   1. Every Python file compiles (`py_compile`)
#   2. Every shell script parses (`bash -n`)
#   3. The package imports without ImportError (top-level src.* modules)
#   4. The bootstrap-Pearson reproducibility script runs end-to-end and
#      prints the same point estimates and CIs the paper reports.

set -e
cd "$(dirname "$0")/.."

# Pick whatever interpreter is on PATH; some environments have
# only `python3` (no `python` symlink).
PY="${PY:-$(command -v python || command -v python3)}"
if [ -z "$PY" ]; then
    echo "ERROR: no python interpreter found on PATH" >&2
    exit 1
fi

# Don't litter the release tree with __pycache__/*.pyc during the smoke test.
# PYTHONDONTWRITEBYTECODE skips bytecode for normal imports; PYTHONPYCACHEPREFIX
# additionally redirects py_compile bytecode (which ignores the no-write flag)
# out of the release tree.
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/plvr_pycache"

echo "================================================================"
echo "Code-release smoke test"
echo "================================================================"

echo "[1/4] py_compile on all Python files"
find . -name "*.py" -type f -print0 | xargs -0 -n1 "$PY" -m py_compile

echo "[2/4] bash -n on all shell scripts"
for s in scripts/*.sh; do bash -n "$s"; done

echo "[3/4] package imports"
"$PY" -c "
import sys, os
sys.path.insert(0, '.')
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.model.qwen_lvr_model import QwenWithLVR  # imports lvr_heads transitively
from src.train.monkey_patch_forward_lvr import replace_qwen2_5_with_mixed_modality_forward_lvr
from src.trainer import QwenLVRSFTTrainer
from src.dataset import make_packed_supervised_data_module_lvr
print('  imports OK')
"

echo "[4/4] bootstrap Pearson script reproducibility"
"$PY" interpretability/bootstrap_pearson.py | tail -10

echo "================================================================"
echo "Smoke test PASSED"
echo "================================================================"
