#!/usr/bin/env bash
# supervise_full.sh <arm> <physical_gpu> <run_dir>
#
# Supervises one train_full.py run to completion. Owns process lifecycle
# only -- train_full.py itself owns epoch/eval/checkpoint/selection logic,
# so no LLM involvement is needed between epochs.
#
# Exit-code policy (FULL_RUN_PROTOCOL.md sec. 12 / FULL_INTERFACES.md):
#   0            COMPLETE      -> write SUPERVISOR_DONE, exit 0 (no restart)
#   3/4/5        NONFINITE/DISK/CONFIG -> record, exit (no restart)
#   6            GPU_BUSY      -> wait 300s, retry (NOT counted as a restart)
#   anything else (crash)      -> restart after 60s, max 5 restarts, logged
set -u
set -o pipefail

ARM="${1:?usage: supervise_full.sh <arm> <physical_gpu> <run_dir> [extra_train_full_args]}"
GPU="${2:?usage: supervise_full.sh <arm> <physical_gpu> <run_dir> [extra_train_full_args]}"
RUN_DIR="${3:?usage: supervise_full.sh <arm> <physical_gpu> <run_dir> [extra_train_full_args]}"
# Optional 4th arg: extra CLI args passed through to train_full.py, word-split
# (test-only use, e.g. "--test-tiny --max-epochs 2"). Never used for
# production launches (launch_full.sh never passes a 4th arg).
EXTRA_ARGS="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$(dirname "$SCRIPT_DIR")"
PYTHON="${LOBOT_PYTHON:-$HOME/venvs/ppsi-gpu/bin/python}"
TRAIN_PY="$SCRIPT_DIR/train_full.py"

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"
LOG="$RUN_DIR/logs/supervisor.log"
rm -f "$RUN_DIR/SUPERVISOR_DONE" "$RUN_DIR/SUPERVISOR_STOPPED"

log() {
    printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" | tee -a "$LOG"
}

free_bytes() {
    df --output=avail -B1 "$W" 2>/dev/null | tail -n 1 | tr -d '[:space:]'
}

DISK_GUARD_BYTES=$((8 * 1024 * 1024 * 1024))

log "supervisor start arm=$ARM gpu=$GPU run_dir=$RUN_DIR pid=$$"

free_now="$(free_bytes)"
if [ -z "$free_now" ] || [ "$free_now" -lt "$DISK_GUARD_BYTES" ]; then
    log "refusing to start: free disk '${free_now}' bytes < 8 GiB"
    echo "SUPERVISOR_STOPPED rc=4 reason=DISK_AT_START" > "$RUN_DIR/SUPERVISOR_STOPPED"
    exit 4
fi

restarts=0
MAX_RESTARTS=5

while true; do
    resume_flag=()
    # previous.pt alone also means "resume": a crash between the two os.replace
    # calls of a latest-rotation leaves only previous.pt; train_full.py falls back to it.
    if [ -f "$RUN_DIR/checkpoints/latest.pt" ] || [ -f "$RUN_DIR/checkpoints/previous.pt" ]; then
        resume_flag=(--resume)
    fi

    log "launching train_full.py (restarts_so_far=$restarts) args=${resume_flag[*]:-<fresh>} extra=$EXTRA_ARGS"
    # Backgrounded (not a blocking foreground call) so the child's PID can be
    # recorded for the supervisor test (SIGKILL mid-epoch, then verify
    # auto-resume). Production behaviour is unaffected: `wait` blocks this
    # loop exactly as a foreground call would.
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" "$TRAIN_PY" \
        --arm "$ARM" --run-dir "$RUN_DIR" "${resume_flag[@]}" $EXTRA_ARGS \
        >> "$RUN_DIR/logs/train_full_stdout.log" 2>&1 &
    child_pid=$!
    echo "$child_pid" > "$RUN_DIR/logs/train_full.pid"
    wait "$child_pid"
    rc=$?
    rm -f "$RUN_DIR/logs/train_full.pid"
    log "train_full.py exited rc=$rc"

    case "$rc" in
        0)
            log "COMPLETE (exit 0)"
            echo "SUPERVISOR_DONE rc=0 $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/SUPERVISOR_DONE"
            exit 0
            ;;
        3)
            log "STOPPED_NONFINITE (exit 3) -- no restart"
            echo "SUPERVISOR_STOPPED rc=3 reason=NONFINITE $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/SUPERVISOR_STOPPED"
            exit 3
            ;;
        4)
            log "STOPPED_DISK (exit 4) -- no restart"
            echo "SUPERVISOR_STOPPED rc=4 reason=DISK $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/SUPERVISOR_STOPPED"
            exit 4
            ;;
        5)
            log "STOPPED_CONFIG (exit 5) -- no restart"
            echo "SUPERVISOR_STOPPED rc=5 reason=CONFIG $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/SUPERVISOR_STOPPED"
            exit 5
            ;;
        6)
            log "GPU_BUSY (exit 6) -- retrying in 300s (NOT counted as a restart)"
            sleep 300
            ;;
        *)
            restarts=$((restarts + 1))
            log "unexpected exit rc=$rc -- restart #$restarts of $MAX_RESTARTS"
            if [ "$restarts" -ge "$MAX_RESTARTS" ]; then
                log "max restarts ($MAX_RESTARTS) reached -- giving up"
                echo "SUPERVISOR_STOPPED rc=$rc reason=MAX_RESTARTS $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/SUPERVISOR_STOPPED"
                exit "$rc"
            fi
            sleep 60
            ;;
    esac

    free_now="$(free_bytes)"
    if [ -z "$free_now" ] || [ "$free_now" -lt "$DISK_GUARD_BYTES" ]; then
        log "refusing to continue: free disk '${free_now}' bytes < 8 GiB"
        echo "SUPERVISOR_STOPPED rc=4 reason=DISK $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$RUN_DIR/SUPERVISOR_STOPPED"
        exit 4
    fi
done
