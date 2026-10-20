# Baseline checkpoint rule — SASREC_FEATURE_OPT_V1 (registered 2026-09-25T10:15Z, before any study score)

Written by the lead (Opus) before any probe, screen or feature-study score exists.

## Rule
The study baseline is ONE immutable, completed checkpoint of the corrected SASRec v3 **rich arm
(A1_RICH256, group `rich_frozen_channels`)**. It is chosen by the campaign's own VS-only record,
using this rule:

- **Candidates:** the completed per-epoch weight files of the A1 runs that exist when the campaign's
  final stage has finished (`state/FINAL_STAGE_DONE` present):
  - `epoch3_weights.pt` / `epoch4_weights.pt` of each A1 base run;
  - `epoch5_weights.pt` of each COMPLETE A1 cool5 run.
- **Score:** the full-VALIDATION_SELECT end-to-end client-macro MRR@20 already recorded by the
  campaign (`epoch_records.json` / cool5 summaries). No new scoring is used for the choice.
- **Selection:** the highest score wins. On an exact tie, the earlier epoch wins, then seed 2026.
- VC/TEST results (retro_eval/) are never read for this choice.

The chosen file's sha256, run id, seed, epoch, config sha and code hashes are recorded in
`config/BASELINE_PIN.json` BEFORE the first probe runs. The baseline is never changed mid-study. A
later, better campaign checkpoint could only enter as an explicitly new baseline family, with every
comparison rerun.

## Why this timing
- **Resource rules.** The study may use a GPU only after the campaign's already-booked training and
  evaluation are finished.
- **Stability.** Reading a half-written `latest.pt` is prohibited. Pinning at that moment therefore
  uses the latest completed, already-VS-selected checkpoint (the package's stated preference), and
  never a file that could still change.

## Development population (probes/screens)
- **Cohort:** the existing verified 10k-user VALIDATION_SELECT cohort,
  `gru_v1/artifacts/validation_select_cohort.parquet`. It covers 112,365 decisions, with all
  eligible decisions for those users and OOV targets included.
- **Ranking:** exhaustive K = 158,486 ranking, strict FP32 evaluation.
- **Primary metric:** client-macro MRR@20. Also reported: micro MRR@20 and micro Recall@1/5/20.
- **Status:** all study selection on VS is adaptive development, not a fresh holdout.
