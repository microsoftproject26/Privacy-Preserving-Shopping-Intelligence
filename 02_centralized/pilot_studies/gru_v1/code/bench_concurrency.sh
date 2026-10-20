#!/usr/bin/env bash
# bench_concurrency.sh [N_UPDATES]  (owner: Opus) -- FULL_RUN_PROTOCOL.md sec. 13.0 system benchmark.
# Phase 1: F4 foreach alone (GPU1). Phase 2: F4 fused alone (GPU1) -> kernel decision (fused iff >= 5% faster).
# Phase 3: F0 <kernel> alone (GPU0). Phase 4: F0 (GPU0) + F4 (GPU1) concurrently, <kernel>.
# Each phase runs bench_sampler.py alongside. Writes $W/bench/*.json and $W/bench/BENCH_SUMMARY.json.
set -u
N="${1:-3000}"
W=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/gru_v1
B=$W/bench
PYB=~/venvs/ppsi-gpu/bin/python
mkdir -p "$B"
cd $W/code
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3

for g in 0 1; do
  busy=$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$busy" != "0" ]; then echo "GPU $g busy ($busy procs) -- aborting benchmark"; exit 6; fi
done

phase() {  # $1 name, then pairs of "gpu:arm:impl"
  local name="$1"; shift
  rm -f "$B/$name.stop" "$B/$name.pids"
  $PYB bench_sampler.py --out "$B/$name.sampler.jsonl" --stop-file "$B/$name.stop" \
       --pid-file "$B/$name.pids" --interval 5 --max-seconds 3600 > /dev/null 2>&1 &
  local sp=$! pids=() i=0
  for spec in "$@"; do
    IFS=: read -r gpu arm impl <<< "$spec"
    CUDA_VISIBLE_DEVICES=$gpu $PYB train_full.py --arm "$arm" --adamw-impl "$impl" \
        --benchmark-updates "$N" > "$B/$name.$i.out" 2> "$B/$name.$i.err" &
    pids+=($!); echo $! >> "$B/$name.pids"; i=$((i+1))
  done
  for p in "${pids[@]}"; do wait "$p"; done
  touch "$B/$name.stop"; wait $sp
  echo "== $name"; for j in $(seq 0 $((i-1))); do tail -1 "$B/$name.$j.out"; done
}

phase p1_f4_foreach "1:F4_rich_side_information_h256:foreach"
phase p2_f4_fused   "1:F4_rich_side_information_h256:fused"
KERNEL=$($PYB - "$B" <<'PY'
import json, sys
B = sys.argv[1]
fe = json.loads(open(f"{B}/p1_f4_foreach.0.out").read().strip().splitlines()[-1])
fu = json.loads(open(f"{B}/p2_f4_fused.0.out").read().strip().splitlines()[-1])
print("fused" if fu["wall_decisions_per_s"] >= 1.05 * fe["wall_decisions_per_s"] else "foreach")
PY
)
echo "KERNEL=$KERNEL"
phase p3_f0_alone "0:F0_item_only_h256:$KERNEL"
if [ "$KERNEL" = "fused" ]; then F4_ALONE=p2_f4_fused; else F4_ALONE=p1_f4_foreach; fi
phase p4_concurrent "0:F0_item_only_h256:$KERNEL" "1:F4_rich_side_information_h256:$KERNEL"

$PYB - "$B" "$KERNEL" "$F4_ALONE" <<'PY'
import json, sys
B, kernel, f4_alone = sys.argv[1], sys.argv[2], sys.argv[3]
def res(name, i=0):
    return json.loads(open(f"{B}/{name}.{i}.out").read().strip().splitlines()[-1])
def samp(name):
    return json.load(open(f"{B}/{name}.sampler.jsonl.summary.json"))
f0a, f4a = res("p3_f0_alone"), res(f4_alone)
f0c, f4c = res("p4_concurrent", 0), res("p4_concurrent", 1)
r0 = f0c["wall_decisions_per_s"] / f0a["wall_decisions_per_s"]
r4 = f4c["wall_decisions_per_s"] / f4a["wall_decisions_per_s"]
rss_total_gib = (f0c["host_rss_bytes"] + f4c["host_rss_bytes"]) / 2**30
concurrent = (r0 >= 0.85) and (r4 >= 0.85) and (64 - rss_total_gib >= 12)
summary = {
    "kernel_rule": "fused iff F4 fused wall dec/s >= 1.05 x foreach",
    "kernel": kernel,
    "f4_foreach_alone": res("p1_f4_foreach"), "f4_fused_alone": res("p2_f4_fused"),
    "f0_alone": f0a, "f0_concurrent": f0c, "f4_concurrent": f4c,
    "ratio_f0_concurrent_over_alone": r0, "ratio_f4_concurrent_over_alone": r4,
    "aggregate_concurrent_dec_s": f0c["wall_decisions_per_s"] + f4c["wall_decisions_per_s"],
    "host_rss_concurrent_total_gib": rss_total_gib,
    "concurrency_rule": "concurrent iff both ratios >= 0.85 and >= 12 GiB pod headroom",
    "decision": "CONCURRENT" if concurrent else "SERIALISE",
    "sampler": {n: samp(n) for n in ("p1_f4_foreach", "p2_f4_fused", "p3_f0_alone", "p4_concurrent")},
}
json.dump(summary, open(f"{B}/BENCH_SUMMARY.json", "w"), indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "sampler"}, indent=1))
PY
echo BENCH_DONE
