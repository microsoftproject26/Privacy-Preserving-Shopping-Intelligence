# 02_centralized - central training: pilots, contest, references, pretraining

## Purpose

Everything trained on a server from raw events: the pilot-study GRU and SASRec campaigns, the
backbone contest that chose SASRec under a registered rule, the cloud reference models
(`C_BIG`, `C_FULL`, `C_SMALL`), the server-side pretrained model (`PRE`, written theta_pre) that
warms the federated line, and the closure-phase central checks (second pretraining seed, GRU
rehearsal).

## Experiments, in order

1. **Pilot studies (20-26 September 2026).** On the frozen two-month dataset, a GRU pilot
   (ContextGRU-v1, full-catalogue cross-entropy over 158,486 products) failed its registered
   gate at two passes (macro MRR@20 0.100795 item-only, 0.101160 with side channels, against
   0.13308 for a transition rule; verdict NO_GO) and was followed by an authorized full run
   whose item-only arm locked at 0.173874 on the selection split. Validation. The SASRec pilot
   then beat that lock by +0.00755 at epoch 3 (validation); its locked headline
   (rich arm, seed 2027, cooldown epoch 5) reached 0.186513 on the selection split and, on the
   one retrospective TEST look, +0.001780 [0.001538, 0.002022] over the locked GRU. The TEST
   look is descriptive (retrospective, recorded as a historical exposure).
2. **Main-study backbone contest (27-28 September 2026).** Equal-effort tuning on CALIB, then
   eight sealed runs on MAIN and one mechanical decision. At the registered best-mark rule the
   GRU-minus-SASRec difference was -0.004271 [-0.005180, -0.003261] (DECISIVE_STABLE for
   SASRec); at the fixed 6.0-pass endpoint the order reversed (+0.004213
   [+0.003283, +0.005216], a recorded diagnostic). Validation. The architecture lock fixed the
   cloud model (width 256, 3 blocks, 83,881,764 parameters) and the device-size model (width
   64, 2 blocks, 20,710,048 parameters).
3. **Cloud references and pretraining (28-30 September 2026).** `C_BIG` on the 131,072 MAIN
   users, `C_FULL` on all 1,348,682 users (26,677,600 decisions), and `PRE` on the 1,217,610
   PRE-band users (24,089,824 decisions), sealed with values unread. Descriptive until the
   confirmatory test evaluation. On MAIN TEST (seed 2026) `C_FULL` scores 0.180388 and `PRE`
   0.160129 client-macro MRR@20. Descriptive.
4. **Confirmatory test evaluation (5-7 October 2026).** The matched-size central model keeps
   94.69% [94.03, 95.31] of the large one (K1) and the pretrained model alone keeps 88.77%
   [88.03, 89.47] of `C_FULL` (K20c). Confirmatory.
5. **Closure-phase extensions (12-17 October 2026).** A second pretraining seed moved every
   matched warm contrast down by 0.0996 to 0.1153 percentage points (all eight negative).
   Exploratory. A GRU-versus-`C_SMALL` CPU rehearsal favoured the GRU by +0.2506 pp at the
   best-mark rule; exploratory, no general claim.

## Code map

| Package | What it does | Origin |
|---|---|---|
| `code/ppsi/seqrec` | Sequence models: SASRec and GRU4Rec-style cores | Main study (prototype lineage) |
| `code/ppsi/models` | Model assembly, variants, monitors | Main study |
| `code/ppsi/central` | Central trainer for the references and pretraining | Main study |
| `code/ppsi/training` | Training loop, recipes, checkpointing | Main study (prototype lineage) |
| `code/ppsi/baselines` | Session kNN, last-item, popularity baselines | Main study |
| `code/snap_v1` -> `code/pilot_code_snapshot` (byte-locked, put on `sys.path` by `ppsi_env`) | Snapshot of the pilot-study SASRec trainer (`SNAPSHOT_MANIFEST.json`, `reuse/`), kept byte-identical | Pilot studies |
| `code/v4base` -> `code/baselines` (import name `v4base`) | Grid, guards, scoring, strata and the training-free baselines (knn, last, pop, rule, sknn) of the research tree | Main study |
| `code/v4fs` -> `code/feature_search` (import name `v4fs`) | Feature-search runner and decision tool (`decide.py`, `importance.py`, `theta0.py`) | Main study |
| `code/v4models` -> `code/model_variants` (import name `v4models`) | Model builders of the research tree (`gru_v2.py`, `sasrec_v2.py`, inventory, monitors) | Main study |
| `code/v4train` -> `code/central_trainer` (import name `v4train`, as used in the recorded runs) | Training drivers of the contest and references (`p2_driver.py`, `p3_driver.py`), recipes, registries, calibration, bootstrap intervals | Main study |
| `code/experiment_extensions/pretraining` -> `code/closure_experiments/pretraining` (import name `experiment_extensions.pretraining`) | Closure-phase pretraining extensions (second seed) | Closure phase |
| `pilot_studies/` | Record of the pilot-study campaigns (frozen record; agent working notes, reviewer briefs, runtime markers and transfer tarballs are not published, so a few kept files name documents that are not here) | Pilot studies |
| `scripts/prototype/{model,baselines,t2,t3,t5,t8}` | Prototype model scripts (architecture comparison, tasks) | Prototype stage |
| `tests/prototype/{baselines,central,models,seqrec,training}`, `tests/research/` (renamed: `tests_baselines` was `tests_base`, `tests_feature_search` was `tests_fs`, `tests_model_variants` was `tests_models`, `tests_central_trainer{,_gpu}` was `tests_train{,_gpu}`) | CPU tests and research suites | Prototype / main study |

## How to run

```bash
uv run python 02_centralized/run.py --from-results   # rebuild this stage's tables and figures from saved results
uv run python 02_centralized/run.py --mode auto      # per training unit: skip if complete, resume from checkpoint if supported, else start
uv run python 02_centralized/run.py --force          # retrain from scratch (old outputs are moved aside first)
```

Training the references and the pretrain needs the frozen dataset, the GPU environment
(`environment/requirements-gpu.txt`) and the server run trees under `PPSI_RUNS_ROOT`; the
research code is hash-pinned and is re-run from the flat tree recorded in
[`../07_study_protocol/`](../07_study_protocol/README.md). `--from-results` rebuilds the stage's
tables from the saved run records without GPU or data.

## results/ and figures/

`results/` holds the curated summaries of this stage (contest tables, reference-model metrics,
pretraining records) and the saved run records under `results/runs/main_study/` and
`results/runs/closure_phase/`. `figures/` holds the rebuilt learning-curve and contest figures.

## Known limitations

The contest verdict depends on the checkpoint rule (it reverses at the fixed endpoint), only
width-256 SASRec candidates reached the MAIN runs, and the tuned learning rates sat at grid
edges; the report therefore claims only that SASRec scored higher in this contest at the
registered rule. `C_FULL`'s recipe was not re-tuned at its scale, so retentions against it
carry an upward bias, which is disclosed. Pilot-study numbers belong to their own regime and
are never set beside main-study numbers. The closure GRU comparison is a CPU rehearsal with
unmatched budgets.
