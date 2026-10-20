set -euo pipefail
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4
LOG=$V/runs_ds3/phase1/phase1.log
test -f "$LOG"
test "$(wc -l < "$LOG")" -eq 1
grep -q 'WAIT_STRICT gpu0 code_ds3r1' "$LOG"
test ! -e "$V/contracts/eval_device/EVAL_DEVICE_PIN_A40.json"
kill -TERM 4317
echo "$(date -u +%FT%TZ) SUPERSEDED_BEFORE_PIN canonical_json_rule_fix" >> "$LOG"
echo R1_QUEUE_STOPPED_BEFORE_PIN
