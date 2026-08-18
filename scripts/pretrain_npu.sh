#!/usr/bin/env bash
# Single-card Ascend 910B pre-training: frozen Depth Anything V2 Base +
# trainable Flash3D Gaussian heads. Extra Hydra overrides are forwarded.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CANN_ENV="${CANN_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
NPU_ID="${NPU_ID:-0}"

if [[ ! -f "${CANN_ENV}" ]]; then
  echo "CANN environment script was not found: ${CANN_ENV}" >&2
  exit 1
fi
source "${CANN_ENV}"
export ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

cd "${PROJECT_ROOT}"
python scripts/check_npu_env.py
exec python scripts/pretrain_npu.py "$@"
