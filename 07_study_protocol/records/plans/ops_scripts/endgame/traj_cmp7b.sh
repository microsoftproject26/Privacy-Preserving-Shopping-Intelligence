#!/bin/bash
# FIX-1 (SERVER, CPU): verify r7b against r7 before H onnx uses it (Fable gate 2026-10-03; model traj_cmp7.sh, DL 342).
#  (1) tiny-run server digests r7 == r7b for all 7 cold arms (test_fl10_fedadam_launch.run_tiny);
#  (2) test_fl16.py + test_fl16b.py IN r7b (the one known Windows-pin failure accepted exactly as in DL 245 / 342);
#  (4) the DEV-05 / P2 tests IN r7b: tests_dev/test_dump_rows_retro_view.py, test_bench_latency_only.py, tests_int/test_int07_test_path.py;
#  (3) dry runs r7 vs r7b: cold DP8 with contracts/dp/S_RECORD.json and DP8_FROZEN with S_RECORD_WARM_FROZEN.json (rc 0,
#      equal arm_spec_sha256 / fl_config_digest). Dry-run run dirs live under $D/tmp (never FL_T).
# Predecessor: $R/deploy_r7b/R7B_DEPLOYED.ok. Output: $R/deploy_r7b/R7B_VERIFIED.ok (the B_reeval_panels.sh predecessor).
V=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4; R=$V/runs_ds3; D=$R/deploy_r7b
Q=$R/queues; LOG=$Q/endgame_C.log; PY=/home/jovyan/venvs/ppsi-gpu/bin/python
st() { echo "$(date -u +%FT%TZ) $*" >> $LOG; }
die() { st "LANE_STOP traj_cmp7b: $*"; echo "STOP: $*" >&2; exit 1; }
[ -e $D/R7B_DEPLOYED.ok ] || die "missing predecessor $D/R7B_DEPLOYED.ok"
[ ! -e $D/R7B_VERIFIED.ok ] || die "already verified"
mkdir -p $D/tmp; st "PHASE_START C traj_cmp7b"
# (1) tiny-run digests, both trees, same process recipe as traj_cmp6
for T in code_ds3r7 code_ds3r7b; do
 ( export PYTHONPATH=$V/$T:$V/$T/tests_int PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
   cd $V/$T; mkdir -p $D/tmp/traj_$T
   timeout 600 $PY -B - <<PY > $D/tmp/traj_$T.txt 2>&1
import sys, pathlib, tempfile, torch, json
torch.backends.cudnn.allow_tf32=False; torch.backends.cuda.matmul.allow_tf32=False
sys.argv=['x']
import test_fl10_fedadam_launch as t
p=pathlib.Path(tempfile.mkdtemp(dir='$D/tmp/traj_$T'))
got={arm: t.run_tiny(p, t._tiny(arm), arm)["summary"]["server_digest"] for arm in t.F.ARMS}
print('TRAJ', json.dumps(sorted(got.items())))
PY
 )
done
A=$(grep '^TRAJ' $D/tmp/traj_code_ds3r7.txt); B=$(grep '^TRAJ' $D/tmp/traj_code_ds3r7b.txt)
[ -n "$A" ] && [ "$A" = "$B" ] || die "tiny-run digests differ or missing (see $D/tmp/traj_*.txt)"
N=$(echo "$A" | $PY -c "import json,sys; print(len(json.loads(sys.stdin.read()[5:])))"); [ "$N" = 7 ] || die "expected 7 arms, got $N"
st "TRAJ_EQUAL r7 == r7b arms=$N"
# (2) the FL-16 / FL-16b tests in r7b (CPU; read-only tree -> basetemp + no cache outside it)
( export PYTHONPATH=$V/code_ds3r7b:$V/code_ds3r7b/tests_int PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
  cd $V/code_ds3r7b/tests_int && timeout 3600 $PY -B -m pytest -q -p no:cacheprovider --basetemp $D/tmp/pytest_fl16 test_fl16.py test_fl16b.py ) > $D/tmp/pytest_fl16.txt 2>&1 \
  || { F=$(grep -E "^(\[[0-9;]*m)*FAILED" $D/tmp/pytest_fl16.txt | sed 's/\[[0-9;]*m//g' | awk '{print $2}' | sort -u | tr '
' ' ')
         # DL 245 / 342: the PRE_FL10_TRAJ pins are Windows-CPU values; on the server only this test fails, identically on every tree,
         # and step (1) above has just proven r7 == r7b for all 7 arms. Any other failure stops.
         [ "$F" = "test_fl16.py::test_existing_arm_digests_and_trajectories_unchanged " ] || die "test_fl16 / test_fl16b failed in r7b ($D/tmp/pytest_fl16.txt): $F"
         st "KNOWN_WINDOWS_PIN_FAILURE accepted (DL 245 / 342): $F"; }
st "TEST_FL16_FL16B_PASS $(tail -1 $D/tmp/pytest_fl16.txt)"
# (3) dry runs r7 vs r7b (rc 0, equal arm_spec_sha256 / fl_config_digest)
T8="--fl-tuning $V/contracts/fl/FL_TUNING_SELECTION.json"
WI="--init-from $V/runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026/PRETRAIN_FOR_FL_INIT.json --freeze-item-tables --warm-selection $V/contracts/fl/FL_WARM_SELECTION_S8.json"
for T in code_ds3r7 code_ds3r7b; do
 ( export PYTHONPATH=$V/$T PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 V4_DATASET_ROOT=/home/jovyan/shared_data/rees46_nextitem_v1 OMP_NUM_THREADS=2
   cd $V/$T
   FL="$PY -B v4int/launch_fl.py --gpu 0 --history-policy USER_HISTORY_L50 --allow-shared-gpu"
   $FL --arm DP --epsilon 8 --seed 2026 $T8 --clip-record $V/contracts/dp/S_RECORD.json --run-dir $D/tmp/dry_${T}_DP8 --dry-run > $D/tmp/dry_${T}_DP8.txt 2>&1; echo "rc=$?" >> $D/tmp/dry_${T}_DP8.txt
   $FL --arm DP_WARM_FROZEN --epsilon 8 --seed 2026 $T8 $WI --clip-record $V/contracts/dp/S_RECORD_WARM_FROZEN.json --run-dir $D/tmp/dry_${T}_DP8F --dry-run > $D/tmp/dry_${T}_DP8F.txt 2>&1; echo "rc=$?" >> $D/tmp/dry_${T}_DP8F.txt )
done
$PY -B - $D/tmp <<'PY' >> $LOG 2>&1 || die "dry-run r7 vs r7b mismatch / refusal (see $D/tmp/dry_*.txt)"
import json, re, sys
from pathlib import Path
d = Path(sys.argv[1])
def keys(p):
    t = p.read_text(); assert t.strip().endswith("rc=0"), f"{p.name}: rc != 0"; assert "REFUSED" not in t, f"{p.name}: REFUSED"
    found = {}
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("arm_spec_sha256", "fl_config_digest", "spec_digest") and isinstance(v, str): found.setdefault(k, v)
                walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    i, j = t.find("{"), t.rfind("}")          # the dry run prints ONE indented (multi-line) JSON object (DL 342)
    walk(json.loads(t[i:j + 1]))
    assert "arm_spec_sha256" in found, f"{p.name}: no arm_spec_sha256 in the dry-run output"
    return found
for n in ("DP8", "DP8F"):
    a, b = keys(d / f"dry_code_ds3r7_{n}.txt"), keys(d / f"dry_code_ds3r7b_{n}.txt")
    assert a == b, f"{n}: r7 {a} != r7b {b}"
    print("DRY_EQUAL", n, {k: v[:12] for k, v in a.items()})
PY
# (4) the DEV-05 / P2 tests in the deployed r7b bytes (CPU; onnx-only tests may skip)
( export PYTHONPATH=$V/code_ds3r7b:$V/code_ds3r7b/tests_int PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
  cd $V/code_ds3r7b && timeout 3600 $PY -B -m pytest -q -p no:cacheprovider --basetemp $D/tmp/pytest_dev05     tests_dev/test_dump_rows_retro_view.py tests_dev/test_bench_latency_only.py tests_int/test_int07_test_path.py ) > $D/tmp/pytest_dev05.txt 2>&1   || die "DEV-05 / P2 tests failed in r7b ($D/tmp/pytest_dev05.txt)"
st "TEST_DEV05_P2_PASS $(tail -1 $D/tmp/pytest_dev05.txt)"
date -u +%FT%TZ > $D/R7B_VERIFIED.ok; st "QUEUE_DONE C traj_cmp7b rc=0 (R7B_VERIFIED)"; echo R7B_VERIFIED
