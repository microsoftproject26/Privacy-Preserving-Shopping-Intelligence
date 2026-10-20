# warm-line helpers on code_ds3r6 (06N N3.2, FL-16 command forms, FA_WARM* = S8 frozen; DL 292)
source /home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/ppsi_v4/runs_ds3/queues/_common_code_ds3r6.sh
I="--init-from $V/runs_pretrain/PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026/PRETRAIN_FOR_FL_INIT.json"
T="--fl-tuning $V/contracts/fl/FL_TUNING_SELECTION.json"
W8="--warm-selection $V/contracts/fl/FL_WARM_SELECTION_S8.json"
FZ="--freeze-item-tables"
SM=$R/SMOKE_R6; mkdir -p $SM
SREC=$V/contracts/dp/S_RECORD_WARM_FROZEN.json
# warm3 <name> <arm args...>: dry run (MAIN binding) -> 3-round smoke (no warm selection) -> MAIN
warm3() { local n=$1; shift
  job DRY_$n fl "$@" $T $I $FZ $W8 --run-dir $R/FL_T/$n --dry-run
  job SMK_$n fl "$@" $T $I $FZ --smoke --smoke-rounds 3 --run-dir $SM/$n
  job $n fl "$@" $T $I $FZ $W8 --run-dir $R/FL_T/$n
}
# dpmain <name> <arm args...>: waits for the registered warm-frozen S record; dry run -> MAIN
dpmain() { local n=$1; shift
  job DRY_$n fl "$@" $T $I $FZ $W8 --clip-record $SREC --run-dir $R/FL_T/$n --dry-run
  job $n fl "$@" $T $I $FZ $W8 --clip-record $SREC --run-dir $R/FL_T/$n
}
