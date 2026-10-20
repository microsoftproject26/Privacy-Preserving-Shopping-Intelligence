#!/usr/bin/env bash
# launch_full.sh <arm> <gpu> <run_dir>
#
# Starts supervise_full.sh detached in a tmux session named
# gru-full-<F0|F4>; refuses if that session already exists. Prints the
# session name and its pane PID. Does NOT launch any production run itself
# -- this is infrastructure only; Opus decides when to invoke it.
set -u
set -o pipefail

ARM="${1:?usage: launch_full.sh <arm> <gpu> <run_dir> [train_full args, e.g. --adamw-impl fused]}"
GPU="${2:?usage: launch_full.sh <arm> <gpu> <run_dir> [train_full args]}"
RUN_DIR="${3:?usage: launch_full.sh <arm> <gpu> <run_dir> [train_full args]}"
# Opus integration: forwarded to supervise_full.sh, which re-applies them on EVERY (re)start,
# so a resume always uses the same --adamw-impl the run was launched with.
EXTRA="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPERVISE="$SCRIPT_DIR/supervise_full.sh"

case "$ARM" in
    F0_item_only_h256) TAG=F0 ;;
    F4_rich_side_information_h256) TAG=F4 ;;
    *) echo "unknown arm: $ARM" >&2; exit 2 ;;
esac
SESSION="gru-full-$TAG"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux not found" >&2
    exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "refusing: tmux session '$SESSION' already exists" >&2
    exit 2
fi

mkdir -p "$RUN_DIR/logs"
chmod +x "$SUPERVISE" 2>/dev/null || true

tmux new-session -d -s "$SESSION" \
    "bash '$SUPERVISE' '$ARM' '$GPU' '$RUN_DIR' '$EXTRA'"

for _ in 1 2 3 4 5; do
    pid="$(tmux list-panes -t "$SESSION" -F '#{pane_pid}' 2>/dev/null | head -n1)"
    if [ -n "${pid:-}" ]; then
        break
    fi
    sleep 1
done

if [ -z "${pid:-}" ]; then
    echo "warning: could not read pane pid for session '$SESSION'" >&2
fi

echo "$SESSION ${pid:-unknown}"
