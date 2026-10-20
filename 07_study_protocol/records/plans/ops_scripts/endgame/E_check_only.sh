#!/bin/bash
# E_check_only.sh <freeze_sha256> — review §1 Phase E: the §9 VALIDATION block index (paths / status / hashes only) and
# retention_report --check-only on it (binds + every check; prints no value, writes nothing). The LAST step before the read.
# FINAL r7 form (OPS-ENDGAME-2), NOT RUN. Predecessors: ok/D_<tag>.ok + the registered freeze uploaded to $V/company/gates/ANA_SCRIPT_FREEZE.json
# (sha = the DL row). Output: $R/ANA/E_CHECK_PASS_<tag>.json (block digest + freeze sha; the F predecessor) + ok/E_<tag>.ok.
# A refusal here is NEVER fixed by editing a flag: new tree + new freeze + new DL row (review D2).
LOG=/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/endgame_E.log
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r7.sh
source $Q/endgame/_endgame_lib.sh
source $Q/endgame/_endgame_ana_args.sh
source $Q/endgame/_endgame_freeze_check.sh
need $OK/D_$ANA_TAG.ok; once $OK/E_$ANA_TAG.ok; once $ANA/E_CHECK_PASS_$ANA_TAG.json
freeze_check "$1"
step "PHASE_START E check-only freeze=${1:0:16} script=${RSHA:0:16}"
once $ANA_BLOCK_VAL
ejob EGE_INDEX_VAL "${ANA_INDEX_VAL[@]}"                       # "INDEX: written {...counts...}" (no metric)
need $ANA_BLOCK_VAL/BLOCK_INDEX.json
chmod -R a-w $ANA_BLOCK_VAL
ejob EGE_CHECK_ONLY $(ana_report_val "$RSHA" "$ANA_OUT_CHECK" --check-only)
BD=$(tree_digest $ANA_BLOCK_VAL); BI=$(sha $ANA_BLOCK_VAL/BLOCK_INDEX.json)
printf '{"phase": "E_CHECK_ONLY_PASS", "utc": "%s", "freeze_sha256": "%s", "retention_report_sha256": "%s", "block_dir": "%s", "block_tree_digest": "%s", "block_index_sha256": "%s", "value_read": false}\n' \
  "$(date -u +%FT%TZ)" "$1" "$RSHA" "$ANA_BLOCK_VAL" "$BD" "$BI" > $ANA/E_CHECK_PASS_$ANA_TAG.json
chmod a-w $ANA/E_CHECK_PASS_$ANA_TAG.json; date -u +%FT%TZ > $OK/E_$ANA_TAG.ok
step "QUEUE_DONE E rc=0 block=${BD:0:16} — next command READS (owner informed first; DL row 'VALIDATION check-only PASS, no value read')"
