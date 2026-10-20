#!/usr/bin/env bash
# v4fs_data/builds.sh (owner: Data engineer, FS-01) - the FS build and the M05/M06 equivalence run in parallel.
# Called by job_manifests.sh under ONE v4data.run_bounded guard (16 GiB over the whole process tree), from ppsi_v4/code.
# Usage: bash builds.sh <python> <data_fs dir> <run dir>
set -u
PY=$1
OUT=$2
RUN=$3
"$PY" -B -m v4fs_data.fs_build --mode equivalence --out "$OUT" > "$RUN/equivalence.log" 2>&1 &
A=$!
"$PY" -B -m v4fs_data.fs_build --mode fs --out "$OUT" > "$RUN/fs.log" 2>&1 &
B=$!
wait "$A"
RA=$?
wait "$B"
RB=$?
echo "equivalence_rc=$RA fs_rc=$RB" > "$RUN/build_rc.txt"
[ "$RA" -eq 0 ] && [ "$RB" -eq 0 ]
