#!/bin/bash
# P_park_smoke_dirs.sh --i-confirm — OPS-ENDGAME-2 (2026-10-02). PREPARED, NOT RUN: an Ops decision (README §4 item 6).
# Why: tools/ana_block_index.py (r7, 6e7043bb) calls check_smoke_name on EVERY dir of every --reeval / --fl / --dev root
# BEFORE any exclusion rule (L320-323; called at L350, L418, L510) and refuses the WHOLE build on a name holding SMOKE / LRSCREEN /
# LR_SCREEN / CALIB. MEASURED 10-02: $R/DEV holds FT_PRE_SMOKE_s2026, FT_FA_WARM_SMOKE_s2026 / s2027 and
# FT_FA_WARM_FROZEN_SMOKE_s2026 / s2027 (SMOKE_NOT_EVIDENCE runs of the FTW / FTP lanes), so E would refuse.
# What: moves each such dir (paths only; nothing opened but its RUN_SUMMARY status) into $R/_PARKED_SMOKE/<root>/<name>
# with a write-once PARK_MANIFEST.json (tree digest before == after), then makes the parked copy read-only. The bytes are
# kept; the dirs are never evidence (MASTER_PLAN §6 smoke rule); the freeze lists them by name.
# Predecessor: no running process names the dir; Output: $Q/endgame/ok/P_PARK.ok (D_freeze refuses while any remain).
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/endgame_P.log
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r7.sh
source $Q/endgame/_endgame_lib.sh
confirm "$1"
once $OK/P_PARK.ok
P=$R/_PARKED_SMOKE; MAN=$P/PARK_MANIFEST.json; once $MAN
mapfile -t S < <(smoke_dirs "$R/REEVAL_A40" "$R/FL_T" "$R/DEV")
[ ${#S[@]} -gt 0 ] || eg_refuse "no smoke-named dir in REEVAL_A40 / FL_T / DEV (nothing to park)"
step "PHASE_START P park ${#S[@]} smoke dir(s)"
for d in "${S[@]}"; do n=$(basename "$d")
  pgrep -f "$n" > /dev/null && eg_refuse "a running process names $n"
  st=$(run_status "$d"); case "$st" in DONE|DONE_MAIN) eg_refuse "$d has RUN_SUMMARY status $st (not a smoke run?)";; esac
done
mkdir -p $P; echo "[" > $MAN.tmp; first=1
for d in "${S[@]}"; do n=$(basename "$d"); root=$(basename "$(dirname "$d")"); to=$P/$root/$n
  [ ! -e "$to" ] || eg_refuse "$to exists"
  a=$(tree_digest "$d"); mkdir -p $P/$root; mv -n "$d" "$to"; [ -d "$to" ] && [ ! -e "$d" ] || eg_refuse "move of $d failed"
  b=$(tree_digest "$to"); [ "$a" = "$b" ] || eg_refuse "$to tree digest changed in the move"
  chmod -R a-w "$to"
  [ $first = 1 ] || echo "," >> $MAN.tmp; first=0
  printf ' {"from": "%s", "to": "%s", "tree_digest": "%s", "status": "%s", "label": "SMOKE_NOT_EVIDENCE"}' "$d" "$to" "$a" "$(run_status "$to")" >> $MAN.tmp
  step "PARKED $d -> $to ${a:0:16}"
done
echo "]" >> $MAN.tmp; mv -n $MAN.tmp $MAN; chmod a-w $MAN
date -u +%FT%TZ > $OK/P_PARK.ok; step "QUEUE_DONE P rc=0 manifest $(sha $MAN | cut -c1-16) — DL row with the manifest sha"
