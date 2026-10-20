#!/bin/bash
set -euo pipefail
EID_WORK_REVIEW="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
export PPSI_TECHNICAL_WORK="$EID_WORK_REVIEW/ppsi_v4"
export PYTHONPATH="$PPSI_TECHNICAL_WORK/code${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TMPDIR=/tmp
export PS1='(PPSI technical work) \w \$ '
cd "$PPSI_TECHNICAL_WORK/code"
exec /bin/bash --noprofile --norc "$@"
