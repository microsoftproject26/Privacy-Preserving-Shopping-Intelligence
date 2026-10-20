#!/usr/bin/env bash
# resume_real_test.sh <gpu> <arm>  (owner: Opus) -- launch gate 6 on the PRODUCTION code path.
# Run A: uninterrupted to >= update 1200.  Run B: killed after update 650 (latest.pt = update 400,
# previous.pt = none/older), resumed with --resume, run to >= update 1200.  The 100-update
# `update_agg` records for updates 500..1200 must be identical between A and B (bit-identical
# floats expected: same permutation, same RNG restore, strict FP32, deterministic kernels).
# Scratch: $W/tests_full/opus/resume_real/{A,B}; deleted by the caller after inspection.
set -u
GPU="${1:?gpu}"; ARM="${2:?arm}"
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1
PYB=~/venvs/ppsi-gpu/bin/python
S=$W/tests_full/opus/resume_real
mkdir -p "$S"
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 CUDA_VISIBLE_DEVICES="$GPU"
FLAGS="--arm $ARM --adamw-impl fused --ckpt-every-updates 400 --diag-every-updates 0"

wait_update() {  # $1 run dir, $2 update number to reach in update_agg
  local d="$1" u="$2"
  for _ in $(seq 1 360); do
    if [ -f "$d/metrics.jsonl" ] && grep -q "\"type\": \"update_agg\", \"update\": $u," "$d/metrics.jsonl"; then return 0; fi
    sleep 5
  done
  return 1
}

run_bg() {  # $1 run dir, $2 extra flag; child PID -> $1.pid (no $(...): a subshell there would hold the pipe open)
  cd $W/code
  setsid $PYB train_full.py $FLAGS --run-dir "$1" $2 >> "$1.stdout.log" 2>&1 < /dev/null &
  echo $! > "$1.pid"
}

echo "[A] start"; run_bg "$S/A" ""; PA=$(cat "$S/A.pid")
wait_update "$S/A" 1200 || { echo "A never reached 1200"; tail -30 "$S/A.stdout.log"; kill $PA; exit 2; }
kill -9 $PA; echo "[A] killed at >=1200"

echo "[B1] start"; run_bg "$S/B" ""; PB=$(cat "$S/B.pid")
wait_update "$S/B" 600 || { echo "B1 never reached 600"; tail -30 "$S/B.stdout.log"; kill $PB; exit 2; }
sleep 3; kill -9 $PB; echo "[B1] killed after >=600 (latest.pt should be update 400)"
ls -la "$S/B/checkpoints"

echo "[B2] resume"; run_bg "$S/B" "--resume"; PB=$(cat "$S/B.pid")
wait_update "$S/B" 1200 || { echo "B2 never reached 1200"; tail -30 "$S/B.stdout.log"; kill $PB; exit 2; }
kill -9 $PB; echo "[B2] killed at >=1200"

$PYB - "$S" <<'PY'
import json, sys
S = sys.argv[1]
def agg(d):
    out = {}
    for line in open(f"{S}/{d}/metrics.jsonl"):
        r = json.loads(line)
        if r.get("type") == "update_agg":
            out[r["update"]] = (r["mean_loss"], r["mean_grad_norm"], r["max_grad_norm"], r["lr"])
    return out
a, b = agg("A"), agg("B")
ups = [u for u in range(500, 1201, 100)]
rows = [(u, a.get(u), b.get(u), a.get(u) == b.get(u)) for u in ups]
for r in rows: print(r)
resumed = [l for l in open(f"{S}/B/logs/train_full.log") if "resumed from" in l]
print("resume log:", resumed[-1].strip() if resumed else None)
ok = all(r[3] and r[1] is not None for r in rows) and bool(resumed) and "update=400" in resumed[-1]
print("RESUME_REAL_PASS" if ok else "RESUME_REAL_FAIL")
PY
echo RESUME_TEST_DONE
