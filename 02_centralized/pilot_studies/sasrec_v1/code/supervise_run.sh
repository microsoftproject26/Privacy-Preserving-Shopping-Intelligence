#!/usr/bin/env bash
# supervise_run.sh <run_id> <seed> <gpu> <run_dir> [extra trainer args...]
#
# Supervises one train_sasrec.py run to completion. Owns process lifecycle only;
# train_sasrec.py owns epoch/eval/checkpoint/selection logic, so no LLM involvement
# is needed between epochs (SASREC_INTERFACES.md sec 5+7).
#
# Exit-code policy:
#   0                 COMPLETE            -> SUPERVISOR_DONE, exit 0 (no retry)
#   3/4/5/7           STOPPED (no retry)  -> SUPERVISOR_STOPPED with reason, exit rc
#   6                 GPU_BUSY            -> wait 600s, retry, up to 12 times
#   anything else     crash               -> resume (--resume), up to 3 times
#
# supervisor_state.json in run_dir persists resume_count/gpu_busy_retries so a
# restarted supervisor (this script re-launched from scratch) does not reset the
# retry budget.
#
# SASREC_TRAINER overrides the trainer command (word-split) for tests, e.g.
#   SASREC_TRAINER="$PYTHON $S/code/tests/fixtures/fake_trainer.py"
# The default is the real production trainer.
set -u
set -o pipefail

RUN_ID="${1:?usage: supervise_run.sh <run_id> <seed> <gpu> <run_dir> [extra trainer args...]}"
SEED="${2:?usage: supervise_run.sh <run_id> <seed> <gpu> <run_dir> [extra trainer args...]}"
GPU="${3:?usage: supervise_run.sh <run_id> <seed> <gpu> <run_dir> [extra trainer args...]}"
RUN_DIR="${4:?usage: supervise_run.sh <run_id> <seed> <gpu> <run_dir> [extra trainer args...]}"
shift 4
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
S="$(dirname "$SCRIPT_DIR")"
PYTHON="${LOBOT_PYTHON:-$HOME/venvs/ppsi-gpu/bin/python}"
TRAINER_CMD="${SASREC_TRAINER:-$PYTHON $SCRIPT_DIR/train_sasrec.py}"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"
LOG="$RUN_DIR/logs/supervisor.log"
STATE_FILE="$RUN_DIR/supervisor_state.json"
rm -f "$RUN_DIR/SUPERVISOR_DONE" "$RUN_DIR/SUPERVISOR_STOPPED"

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" | tee -a "$LOG"
}

# --pair-dir, if present among the extra args, so we can write pair status on give-up.
PAIR_DIR=""
n=${#EXTRA_ARGS[@]}
for ((i = 0; i < n; i++)); do
    if [ "${EXTRA_ARGS[$i]}" = "--pair-dir" ] && [ $((i + 1)) -lt "$n" ]; then
        PAIR_DIR="${EXTRA_ARGS[$((i + 1))]}"
    fi
done

write_pair_status() {
    # write_pair_status <state>   ({FAILED|STOPPED})
    [ -z "$PAIR_DIR" ] && return 0
    mkdir -p "$PAIR_DIR" 2>/dev/null || true
    "$PYTHON" - "$PAIR_DIR/$RUN_ID.status.json" "$1" <<'PYEOF'
import json, sys, time
path, state = sys.argv[1], sys.argv[2]
rec = {"state": state, "epoch": None, "record_sha256": None,
       "heartbeat_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "mean_epoch_seconds": None}
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump(rec, f)
import os
os.replace(tmp, path)
PYEOF
}

load_counters() {
    if [ -f "$STATE_FILE" ]; then
        resume_count="$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1])).get('resume_count', 0))" "$STATE_FILE" 2>/dev/null || echo 0)"
        gpu_busy_retries="$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1])).get('gpu_busy_retries', 0))" "$STATE_FILE" 2>/dev/null || echo 0)"
    else
        resume_count=0
        gpu_busy_retries=0
    fi
}

save_counters() {
    "$PYTHON" - "$STATE_FILE" "$resume_count" "$gpu_busy_retries" <<'PYEOF'
import json, os, sys
path, resume_count, gpu_busy_retries = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tmp = path + ".tmp"
with open(tmp, "w") as f:
    json.dump({"resume_count": resume_count, "gpu_busy_retries": gpu_busy_retries}, f)
os.replace(tmp, path)
PYEOF
}

MAX_RESUMES=3
MAX_GPU_BUSY_RETRIES=12
# Test-only overrides: production never sets these, so the real waits stay 600s/60s.
GPU_BUSY_SLEEP="${SASREC_SUPERVISE_GPU_BUSY_SLEEP:-600}"
CRASH_SLEEP="${SASREC_SUPERVISE_CRASH_SLEEP:-60}"

log "supervisor start run_id=$RUN_ID seed=$SEED gpu=$GPU run_dir=$RUN_DIR pid=$$ trainer=[$TRAINER_CMD]"
load_counters

while true; do
    resume_flag=()
    if [ -f "$RUN_DIR/checkpoints/latest.pt" ] || [ -f "$RUN_DIR/checkpoints/previous.pt" ]; then
        resume_flag=(--resume)
    fi

    log "launching trainer (resume_count=$resume_count gpu_busy_retries=$gpu_busy_retries) resume=${resume_flag[*]:-<fresh>} extra=${EXTRA_ARGS[*]:-<none>}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        $TRAINER_CMD --run-id "$RUN_ID" --seed "$SEED" --run-dir "$RUN_DIR" \
        "${resume_flag[@]}" "${EXTRA_ARGS[@]}" \
        >> "$RUN_DIR/logs/train_stdout.log" 2>&1 &
    child_pid=$!
    echo "$child_pid" > "$RUN_DIR/logs/trainer.pid"
    wait "$child_pid"
    rc=$?
    rm -f "$RUN_DIR/logs/trainer.pid"
    log "trainer exited rc=$rc"

    case "$rc" in
        0)
            log "COMPLETE (exit 0)"
            echo "SUPERVISOR_DONE rc=0 $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/SUPERVISOR_DONE"
            exit 0
            ;;
        3|4|5)
            reason="NONFINITE"; [ "$rc" = 4 ] && reason="DISK"; [ "$rc" = 5 ] && reason="CONFIG"
            log "SUPERVISOR_STOPPED ($reason, exit $rc) -- no retry"
            echo "SUPERVISOR_STOPPED rc=$rc reason=$reason $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/SUPERVISOR_STOPPED"
            write_pair_status STOPPED
            exit "$rc"
            ;;
        7)
            log "SUPERVISOR_STOPPED (PAIR_PARTNER_STOPPED, exit 7) -- no retry"
            echo "SUPERVISOR_STOPPED rc=7 reason=PAIR_PARTNER_STOPPED $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/SUPERVISOR_STOPPED"
            write_pair_status STOPPED
            exit 7
            ;;
        6)
            gpu_busy_retries=$((gpu_busy_retries + 1))
            save_counters
            log "GPU_BUSY (exit 6) -- retry $gpu_busy_retries/$MAX_GPU_BUSY_RETRIES in 600s (not counted as a resume)"
            if [ "$gpu_busy_retries" -gt "$MAX_GPU_BUSY_RETRIES" ]; then
                log "max GPU_BUSY retries ($MAX_GPU_BUSY_RETRIES) reached -- giving up"
                echo "SUPERVISOR_STOPPED rc=6 reason=GPU_BUSY_MAX_RETRIES $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/SUPERVISOR_STOPPED"
                write_pair_status FAILED
                exit 6
            fi
            sleep "$GPU_BUSY_SLEEP"
            ;;
        *)
            resume_count=$((resume_count + 1))
            save_counters
            log "crash (exit $rc) -- resume $resume_count/$MAX_RESUMES"
            if [ "$resume_count" -gt "$MAX_RESUMES" ]; then
                log "max resumes ($MAX_RESUMES) exceeded -- giving up"
                echo "SUPERVISOR_STOPPED rc=$rc reason=MAX_RESUMES $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/SUPERVISOR_STOPPED"
                write_pair_status FAILED
                exit "$rc"
            fi
            sleep "$CRASH_SLEEP"
            ;;
    esac
done
