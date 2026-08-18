#!/usr/bin/env bash
# NPU single-image inference launcher. All infer_npu.py arguments are passed on.
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

cd "${PROJECT_ROOT}"
python scripts/check_npu_env.py
exec python scripts/infer_npu.py "$@"
