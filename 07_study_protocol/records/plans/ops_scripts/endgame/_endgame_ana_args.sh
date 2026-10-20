# _endgame_ana_args.sh — the ONE definition of the §9 VALIDATION and §11.6 TEST command lines (D4 corrections, DL 317;
# review §1 E1/E2/H6, §5 D4). Sourced by D_freeze.sh (which writes the VALIDATION text into ANA_SCRIPT_FREEZE),
# E_check_only.sh / F_validation_read.sh (which refuse unless the text they would run is byte-identical to the frozen text)
# and H_test.sh (analysis). FINAL r7 form (OPS-ENDGAME-2): every flag below parses under the r7 tools' own parse_args
# (README §6: ana_block_index 6e7043bb, retention_report 1d50f7f2).
# D4: --fl $R/FL_T (NEVER $R/FL: pre-S4 ladder, same dir names); --pin; --contrasts v4; --records += S_RECORD_WARM_FROZEN,
# FL_WARM_SELECTION_S7 / S8, SELECTION_FT_FA; NO --allow-single-seed (D2: cold FA_1024 / DP8 run at 2026 AND 2027).
ANA_TAG=${ANA_TAG:-v1}                # a NEW freeze (after a refusal) uses a new tag -> new dirs; the tag is in the frozen text
ANA_BLOCK_VAL=$ANA/block_val_$ANA_TAG
ANA_OUT_CHECK=$ANA/out_val_check_$ANA_TAG
ANA_OUT_VAL=$ANA/out_val_$ANA_TAG
ANA_BLOCK_TEST=$ANA/block_test_$ANA_TAG
ANA_OUT_TEST=$ANA/out_test_$ANA_TAG
# K16 ran iff laneD3_INT8_r7 wrote BENCH_RESULT.json into ONNX_R7 or ONNX_R7_U16384 (DL 343); otherwise K16 is declared
# NOT RUN in the freeze (MUST tier, 06D §8: that declaration needs its own DL decision before E).
ANA_ONNX_DIR=$(onnx_val_dir); [ "$ANA_ONNX_DIR" = BOTH ] && eg_refuse "both ONNX_R7 and ONNX_R7_U16384 hold a BENCH_RESULT (the lane writes one)"
ANA_ONNX=(); [ -n "$ANA_ONNX_DIR" ] && ANA_ONNX=(--onnx "$ANA_ONNX_DIR")
# TRUNC: Study B declared not run (D3); T41 / Study A only if run before the freeze. $R/TRUNC is empty (MEASURED 10-02).
ANA_TRUNC=(); [ -n "$(ls -A "$R/TRUNC" 2>/dev/null)" ] && ANA_TRUNC=(--trunc "$R/TRUNC")
# Records (index records_files() classifies each by file name, MEASURED L724-737). NOT passed, by evidence:
#  LR_SELECTION.json — optional in the report (reported only when present, retention_report L4686); the S4 tuning
#    (FL_TUNING_SELECTION 9bca186e, hashed into the freeze) replaced the LR screen (README Q6);
#  EVENT_SIZE*.json — no registered per-event size exists; the report labels the raw-event bytes level NOT_REGISTERED
#    (retention_report L4054-4065); declared in the freeze (README Q15).
ANA_RECORDS=(
  "$V/contracts/strata/STRATA_CUTPOINTS_MAIN.json"
  "$V/contracts/dp/S_RECORD.json"
  "$V/contracts/dp/S_RECORD_WARM_FROZEN.json"
  "$V/contracts/device/SELECTION_FT_C.json"
  "$V/contracts/device/SELECTION_LO.json"
  "$V/contracts/device/SELECTION_FT_FA.json"
  "$V/contracts/fl/FL_WARM_SELECTION_S7.json"
  "$V/contracts/fl/FL_WARM_SELECTION_S8.json"
  "$CR/tests_fl/receipts/DP_WOR_SENSITIVITY_FL06.json"
)
ANA_INDEX_VAL=($PY -B "$TOOLS/ana_block_index.py" --split VALIDATION --device-class NVIDIA_A40-FP32-DET --pin "$PIN"
  --reeval "$R/REEVAL_A40" --fl "$R/FL_T" --dev "$R/DEV" --base "$R/BASE_R4" "${ANA_ONNX[@]}" "${ANA_TRUNC[@]}"
  --records "${ANA_RECORDS[@]}" --out "$ANA_BLOCK_VAL")
# the frozen script sha comes from the freeze file (E / F), never from the deployed script itself
ana_report_val() { # ana_report_val <expected_script_sha256> <out_dir> [--check-only]
  echo $PY -B "$TOOLS/retention_report.py" --block "$ANA_BLOCK_VAL" --split VALIDATION --contrasts "$CONTRASTS" \
    --code-root "$CR" --out "$2" --expect-script-sha256 "$1" ${3:+"$3"}; }
ana_cmd_text() { # the canonical text that D freezes and E / F compare (placeholder for the sha, filled by E / F)
  printf '%s\n' "INDEX: ${ANA_INDEX_VAL[*]}" "CHECK: $(ana_report_val '<FROZEN_RETENTION_REPORT_SHA256>' "$ANA_OUT_CHECK" --check-only)" \
    "READ: $(ana_report_val '<FROZEN_RETENTION_REPORT_SHA256>' "$ANA_OUT_VAL")"; }
# §11.6 TEST (README Q14, RESOLVED): no --fl (ana_block_index L414-415: "--fl on a TEST block" refuses; the FL checkpoints come
# back through --reeval); --registry-ext = the DATA-09 TEST extension; --records = the VALIDATION records +
# VALIDATION_RESULT=<F's RETENTION_RESULT.json> (KEY=file form, RECORD_KEYS; retention_report check_validation_result
# binds script / library / contrasts content); --onnx only if the TEST bench wrote its result.
ANA_TEST_EXT=$V/contracts/manifest_registry/REGISTRY_V2_TEST_EXT.json
ana_index_test() { # prints one argument per line (H reads it with mapfile; the validation harness parses it)
  local o=(); [ -f "$R/TEST/ONNX/BENCH_RESULT.json" ] && o=(--onnx "$R/TEST/ONNX")
  printf '%s\n' $PY -B "$TOOLS/ana_block_index.py" --split TEST --device-class NVIDIA_A40-FP32-DET --pin "$PIN" \
    --registry-ext "$ANA_TEST_EXT" --reeval "$R/TEST/REEVAL" --dev "$R/TEST/DEV" --base "$R/TEST/BASE" "${o[@]}" \
    --records "${ANA_RECORDS[@]}" "VALIDATION_RESULT=$ANA_OUT_VAL/RETENTION_RESULT.json" --out "$ANA_BLOCK_TEST"; }
ana_report_test() { # ana_report_test <expected_script_sha256>; --contrasts IS the lock (retention_report L5981, A-F8)
  printf '%s\n' $PY -B "$TOOLS/retention_report.py" --block "$ANA_BLOCK_TEST" --split TEST --contrasts "$LOCK" \
    --all-methods-lock "$LOCK" --test-exposure-register "$REGISTER" --registry-ext "$ANA_TEST_EXT" --code-root "$CR" \
    --out "$ANA_OUT_TEST" --expect-script-sha256 "$1"; }
for f in "${ANA_RECORDS[@]}" "$PIN"; do [ -f "$f" ] || eg_refuse "VALIDATION record missing: $f"; done
