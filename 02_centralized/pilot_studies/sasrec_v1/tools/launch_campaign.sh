#!/usr/bin/env bash
# launch_campaign.sh (owner: Opus): pre-launch gates, then the detached queue runner.
# Runs queue_runner.py inside tmux session `sasrec-queue` with a restart loop; the runner is
# idempotent (state file + PID lock) and relaunches dead supervisors itself. No LLM involved.
set -euo pipefail
S=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1
PY=~/venvs/ppsi-gpu/bin/python
cd "$S/code"

fail() { echo "LAUNCH REFUSED: $1"; exit 2; }
[ -f "$S/review/GATE_C_PASS.json" ] || fail "no Gate C pass"
[ -f "$S/bench/TF32_PARITY.json" ] || fail "no TF32 ruling"
[ -f "$S/bench/SYSTEMS_LOCK.json" ] || fail "no systems lock (real-data benchmark)"
for g in 0 1; do
  n=$(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$n" = "0" ] || fail "GPU $g busy"
done
free=$(df -B1 /home/jovyan | tail -1 | awk '{print $4}')
[ "$free" -gt $((12 * 1024 * 1024 * 1024)) ] || fail "free disk below 12 GiB"
if tmux has-session -t sasrec-queue 2>/dev/null; then fail "sasrec-queue already running"; fi
mkdir -p "$S/logs"
cat > "$S/logs/queue_loop.sh" <<EOS
#!/usr/bin/env bash
cd "$S/code"
while true; do
  $PY queue_runner.py >> "$S/logs/queue_runner.log" 2>&1
  phase=\$($PY -c "import json;print(json.load(open('$S/state/queue_state.json'))['phase'])" 2>/dev/null || echo UNKNOWN)
  echo "[\$(date -u +%FT%TZ)] queue_runner exited, phase=\$phase" >> "$S/logs/queue_runner.log"
  [ "\$phase" = "DONE" ] && break
  sleep 60
done
EOS
chmod +x "$S/logs/queue_loop.sh"
tmux new-session -d -s sasrec-queue "bash $S/logs/queue_loop.sh"
sleep 5
tmux ls | grep sasrec || true
echo "LAUNCHED $(date -u +%FT%TZ)"
