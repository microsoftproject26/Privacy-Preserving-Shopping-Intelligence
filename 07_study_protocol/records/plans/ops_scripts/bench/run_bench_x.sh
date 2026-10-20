#!/bin/bash
# POST_HOC_X LEAN driver (EXPLORATORY; inner validation only; never a TEST look). CPU only - no GPU at any time.
# Preflight, then runs the event-driven scheduler code_bench_x/tools_x/lean_x.py (job logic, selection, R6).
#
#   bash run_bench_x.sh --slots 24 --threads 4 --allow-during-sitting [--datasets "..."] [--dry-run]
#   (after SITTING COMPLETE the same command works without --allow-during-sitting; relaunching resumes:
#    DONE_MAIN runs are skipped, partial runs restart with --resume, runs still alive are not started twice)
# Refused: before "SITTING COMPLETE" without --allow-during-sitting; slots x threads > 96 (owner cap, approved
# 2026-10-06 also during the sitting); a running "python -m v4bench.run final"; a code_bench_x file whose sha256
# differs from X_SHA256SUMS; a second driver (flock). Every job: CUDA_VISIBLE_DEVICES="", nice -n 19.
set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
X=$V/code_bench_x
RX=$V/runs_bench_x
PY=/home/jovyan/venvs/ppsi-gpu/bin/python
SLOTS=24; THREADS=4; ALLOW=0; DRY=0; DATASETS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --slots|--cpu-slots) SLOTS="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --allow-during-sitting) ALLOW=1; shift;;
    --datasets) DATASETS="$2"; shift 2;;
    --resume) shift;;
    --dry-run) DRY=1; shift;;
    *) echo "unknown argument $1 (GPU options were removed: CPU only)" >&2; exit 2;;
  esac
done
mkdir -p "$RX/lean"
say() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$RX/lean/driver.log"; }
refuse() { say "REFUSED: $*"; exit 3; }
if ! grep -q "SITTING COMPLETE" "$V/runs_p2_sit.out" 2>/dev/null; then
  [ $ALLOW -eq 1 ] || refuse "the sitting is not complete; CPU mode needs --allow-during-sitting"
fi
[ $((SLOTS * THREADS)) -le 112 ] || refuse "at most 112 CPU threads (slots x threads)"
if pgrep -f "python[0-9.]*( -[A-Za-z]+)* -m v4bench\.run final" >/dev/null; then refuse "a 'v4bench.run final' process is running"; fi
(cd "$X" && sha256sum -c --quiet X_SHA256SUMS) || refuse "code_bench_x sha256 differs from X_SHA256SUMS"
grep -q POST_HOC_X "$X/v4bench/run.py" && grep -q POST_HOC_X "$X/v4bench/final.py" || refuse "$X is not the POST_HOC_X tree"
exec 9>"$RX/.driver.lock"
flock -n 9 || refuse "another driver holds $RX/.driver.lock"
ARGS=(--slots "$SLOTS" --threads "$THREADS"); [ -n "$DATASETS" ] && ARGS+=(--datasets "$DATASETS"); [ $DRY -eq 1 ] && ARGS+=(--dry-run)
say "lean driver: slots=$SLOTS threads=$THREADS allow_during_sitting=$ALLOW dry=$DRY"
cd "$X"
export CUDA_VISIBLE_DEVICES=""
nice -n 19 "$PY" -B tools_x/lean_x.py "${ARGS[@]}"
