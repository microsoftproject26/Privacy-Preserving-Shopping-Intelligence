"""v4int.launch_fl — the real-data launcher of the federated arms at SMALL (INT-03 item 1).

  python -B code/v4int/launch_fl.py --arm FA|FP|PF|FA_1024|DP|S_CAL|FA_Q8 (+ FA_WARM*, FL-16 *_WARM[_FROZEN]) \
         --seed 2026 --gpu 0 --run-dir <dir> \
         --history-policy USER_HISTORY_L50 [--epsilon 8] [--clip-record <S_RECORD.json>] [--small-recipe <file>] \
         [--lr-mult 0.5|1|2|4|8|16|32 | --lr-selection <LR_SELECTION[_EXT].json> | --fl-tuning <FL_TUNING_SELECTION.json>] \
         [--server-opt fedavg|fedadam --server-lr 0.01|0.1] [--role MAIN|CALIBRATION] \
         [--resume] [--allow-shared-gpu] [--checkpoint-every 64] [--smoke --smoke-rounds 3 [--smoke-interrupt-at 2]]
         [--dry-run]

Authority: PREG1_AMENDMENT_03 §2 (all FL arms at SMALL, one recipe, one theta0 per seed, MAIN, 6.0 EFE, frozen T3),
_04 §3 (knobs), _06 §1-§2 (SASREC_D64_B2; Q1-Q5), FL-0304 (the fl library API), ARCHITECTURE_LOCK (winner only).

Registered knobs (KNOBS; never a CLI value):
  FA / FP / PF  64 clients per round, seeded SWEEP sampling, 10% seeded drop-out, 6,827 rounds (round-count endpoint,
                normalized schedule s(6 E / E_total)); B16 x 2 local passes, fresh AdamW per visit (wd 1e-5, clip 1.0);
                FP mu 0.01; PF personal vector p_u (lambda 1e-4, personal LR = SMALL peak x the round factor, clip 1.0).
  FA_1024 / DP  POISSON participation (PREG1_AMENDMENT_06C, FL-07): each client joins round r independently with
                q = m / N = 1,024 / 131,072, seeded derive_seed(seed, "poisson", r); FIXED denominator m = 1,024; no
                drop-out, T = 384 rounds (expected 6.0 EFE; LR progress and the 12 marks follow (r + 1) / T),
                DP-FedAvg aggregation with flat clip S (from the S_CAL record); FA_1024 z = 0;
                DP z from the registered accountant record (eps 8 -> 0.5642, eps 1 -> 1.3094; delta 5e-6, N 131,072).
  S_CAL         the S calibration pass (Q3): the exact FA_1024 config at seed 2026 with DPConfig(None, 0, 1024),
                rounds 1-20 only; writes S_RECORD.json (median pre-clip client-delta norm, fl.dp.median_clip_norm) and
                keeps no weights.
Clients = MAIN_TRAIN users (one client = one user, key = client_key_hex via v4base.identity, the manifest's own
verified tables); a client's examples = its own canonical MAIN_TRAIN rows served by v4int.providers.StoreTrainProvider
(a pure function of the positions). The model = SASREC_D64_B2 with the registered theta0 of the seed, built by the
locked CentralTrainer that ALSO serves as the evaluation harness: its frozen T3 path (`_evaluate`, MAIN_VALIDATION,
predict_adapter + evaluate + rows_from_result) evaluates the FL server's weights (the server adapter IS the harness
model), so the rows have exactly the central RESULT_ROWS format; the launcher relabels only the method / paradigm /
status fields (fl_rows). Evaluations fire on the normalized half-EFE units (12 per run: marks 0.5 .. 6.0); raw EFE is
recorded in every row. At the end: RESULT_ROWS.json (PRACTICAL_BEST = the strictly-best mark, CONTROLLED_BUDGET = 6.0),
RUN_SUMMARY.json (FLRun summary, bytes per visit / per client / fleet, participation), for PF PF_STORE.pt (end store) + PF_STORE_best.pt
(the store of the PRACTICAL_BEST mark, written at every new best; 06E §7 / REV-FL-01 B1-B2: always run.store), and the
resumable latest.pt removed. Preflight / lock binding / determinism exports reuse v4int.launch_central (the G1 gate,
the pinned DATA_PROTOCOL_LOCK_V2, CUBLAS_WORKSPACE_CONFIG, CUDA_VISIBLE_DEVICES, the GPU checks incl.
--allow-shared-gpu); the SMALL recipe comes from v4int.small_recipe (production path for a non-smoke run).
--smoke (SMOKE_NOT_EVIDENCE): at most MAX_SMOKE_ROUNDS rounds (below the first evaluation), then one relabelled
evaluation (SMOKE_EVAL.json) and every weight file deleted; --smoke-interrupt-at k stops after a `latest` save at k.

FL-05 (PREG1_AMENDMENT_06B §1, §5; every option OFF by default, and then every arm-spec digest and fl RunConfig digest
is byte-identical to the pre-FL-05 code — spec_digest drops the FL-05 keys at their defaults, like RunConfig.digest):
  --lr-mult m      m in {0.5, 1, 2} (default 1) multiplies lr_peak_local (and the local solver LR) of the arm only; the
                   PF personal LR (SMALL peak x round factor), FP mu, DP / clip settings are unchanged. Recorded in the
                   arm spec (lr_mult, lr_peak_base) and in every row (fl_lr_mult).
  --lr-selection f the verified LR_SELECTION.json of v4int.lr_screen (format, self-sha256, the 06B §1 rule recomputed
                   from its three values); its selected m is used as --lr-mult (the two options are exclusive).
  --role CALIBRATION  FA only, seed 2026: the federated local-LR screen. Clients = CALIB_TRAIN users keyed by
                   client_key_hex, evaluation on CALIB_VALIDATION through the same frozen T3 path (the harness model
                   scored by predict_adapter / evaluate / rows_from_result with the CALIBRATION key fields), FA knobs
                   with rounds = ceil(3 N / 64 / 0.9) = 1,707 for N = 32,768; rows role CALIBRATION, run_label
                   FL_LR_SCREEN. v4int.lr_screen selects m from the three runs.
  --arm FA_Q8      FA exactly (method code FA, rows labelled FA_Q8) with int8 stochastic-rounding uploads (fl.quant:
                   per-tensor symmetric, scale = max|delta| / 127, seeded derive_seed(seed, "q8", round, client));
                   upload bytes = sum(numel) + 4 x n_tensors, download dense FP32.
Every run writes ARM_SPEC.json (the arm spec and its sha256); --resume refuses a run dir whose ARM_SPEC.json differs
(or is missing while a latest.pt exists): the fl RunConfig digest alone does not carry the Q8 codec.

FL-09 (PREG1_AMENDMENT_06F §1, §3): --lr-mult also takes 4, 8, 16, 32 (arm spec unchanged in shape: every digest at
m in {0.5, 1, 2} is byte-identical); --lr-selection takes the 06B file (PPSI_V4_FL_LR_SELECTION_1) or the extended
screen's PPSI_V4_FL_LR_SELECTION_EXT_1 (rule re-applied).
FIX-DELTA-01 F2 (REV-DELTA-01; supersedes FL-09's / FL-10's sibling-file rules): a non-smoke MAIN run takes exactly the
latest REGISTERED selection — contracts/fl/FL_TUNING_SELECTION.json (--fl-tuning) once it exists, else
contracts/fl/LR_SELECTION_EXT.json (--lr-selection); the given path must resolve to it (a preview / copy is refused),
its file sha256 is recorded (LAUNCH main_selection), and the EXT file (directly, or as the tuning file's S1b source)
must carry the registered first-screen sha256, screen ⊇ {0.5, 1, 2, 4, 8, 16}, screen x32 iff x16 wins without it,
and lr_peak_base == the SMALL recipe peak. The 06B file is refused for MAIN outright; smoke / CALIBRATION / dry runs
are not bound. Arm specs and digests are unchanged.

FL-10 (PREG1_AMENDMENT_06G §1-§3, 06I §3, 06J §1-§2; every option OFF by default -> every arm spec, RunConfig digest,
ARM_SPEC.json, row and checkpoint byte-identical to the pre-FL-10 code):
  --server-opt fedavg|fedadam  (default fedavg = server LR 1, the unchanged path) and --server-lr 0.01|0.1 (required for
                   fedadam, refused for fedavg): FedAdam of fl.server (Reddi et al. 2021 Alg. 2; beta1 0.9, beta2 0.99,
                   tau 1e-3, m_0 = 0, v_0 = tau^2, no bias correction). Recorded in the arm spec (server_opt, server_lr,
                   cfg.server_opt) and every row (fl_server_opt, fl_server_lr) only when fedadam. A screen / smoke option
                   (exclusive with --lr-selection / --fl-tuning); CALIB run id FL_FA_LRSCREEN_FEDADAM_eta<eta>_x<m>_CALIB_s2026.
  --server-opt fedavgm  FedAvgM (server momentum 0.9, server LR 1.0 fixed; 06J §1); --server-lr refused unless 1.0.
  --server-tau 1e-05    FedAdam only: the 06J §1 edge tau (default 1e-3); FedAdam eta_s also admits the 06J edge {0.003, 0.001}.
  --lr-shape central|fl_const   (default central = the s(.) map): FL_CONST of fl.runtime (06I §3), recorded as cfg.lr_shape.
  --client-opt adamw_fresh|sgd_m (default adamw_fresh) and --client-lr {0.01, 0.1, 1.0} (+ the 06J edge {3, 10, 0.003};
                   required for sgd_m, refused for adamw_fresh): client SGD momentum 0.9, fresh per visit, clip 1.0 kept,
                   no weight decay; the local peak LR is the client LR (m must be 1; the PF personal LR is unchanged).
  Every non-default dimension is recorded in the arm spec, cfg and every row (fl_server_opt / fl_server_lr /
  fl_server_tau / fl_lr_shape / fl_client_opt / fl_client_lr); all are screen / smoke options.
  --fl-tuning f    a verified stage file of `lr_screen select-joint` (PPSI_V4_FL_TUNING_SELECTION_1, stage S2 / S3; the
                   registered stage rule re-applied): its winner (every dimension above + m) configures the run. Once that file exists (at contracts/fl/ or next to
                   the given selection file) a non-smoke MAIN run refuses every LR_SELECTION file and needs --fl-tuning.

FL-12/13 (PREG1_AMENDMENT_06J §3 S4, 06K §2 S7, 06K §3 S8, DL 206; every option OFF by default -> every arm spec,
RunConfig digest, row and checkpoint byte-identical to the pre-FL-12 code):
  --local-passes 1|2|4, --local-batch 8|16 (defaults 2 / 16 = KNOBS; one at a time, 06J §3): the S4 screen dimensions of
                   the tuning point (label '|passes=P' / '|batch=B', rows fl_local_passes / fl_local_batch, spec
                   local_passes / local_batch). The round count keeps the registered 6.0-EFE budget (06J §3):
                     rounds = ceil(6.0 N_dec / (64 x 0.9 x P x N_dec / N_clients)) = ceil(60 N_clients / (576 P))
                   (P = 2: 6,827 MAIN / 1,707 CALIB, the registered values), recorded as spec exposure_budget; the batch
                   does not change exposures. Poisson arms (FA_1024 / S_CAL / DP) refuse P != 2 (their T = 384 is the
                   registered DP accountant's) and, since FIX-DELTA-02 (DL 219), batch != 16. A screen / smoke option like the FL-10 ones: MAIN takes it only from
                   the registered FL_TUNING_SELECTION.json (lr_screen select-joint --stage S4).
  --arm FA_WARM --init-from <PRETRAIN_FOR_FL_INIT.json> (S7): FA (method code FA) from theta_pre = the PRE run's
                   endpoint_6.0 checkpoint, sha256-checked against the init record, run label PRETRAIN_FOR_FL, train
                   manifest PRETRAIN_TRAIN_5_100_NONMAIN only (a checkpoint trained on MAIN users — C_BIG / C_SMALL /
                   C_FULL, MAIN / CALIB / FULL_TRAIN manifests — is refused); the loaded state must hash to the
                   checkpoint's own state_digest (the server's init_sha256). Recorded in the spec as init_from
                   (identity: shas, run id, manifest; no path). --warm-lr-factor f multiplies the tuning point's local LR
                   (S7 screen f in {1, 1/4} = m* / 4; S8 f in {1, 4}); a non-smoke MAIN warm run takes f from the
                   registered contracts/fl/FL_WARM_SELECTION_S7|S8.json (--warm-selection; lr_screen select-warm).
  --arm FA_WARM_FROZEN --init-from f --freeze-item-tables (S8): as FA_WARM with item_embed.weight / output_embed /
                   output_bias frozen at theta_pre (fl.runtime.freeze_item_tables: not trained, not uploaded, not
                   aggregated; bytes per visit = the uploaded parameters; the frozen tables' download counted once per
                   client; a DP variant clips / noises the uploaded dimension).

FIX-DELTA-02 (REV-DELTA-02 M1 / M2 / m6; DECISION_LOG row 219; 06G §4 / 06I §3 / 06J §3-§4):
  M2  a non-smoke MAIN run takes ONLY the FINAL contracts/fl/FL_TUNING_SELECTION.json at stage S4 (--fl-tuning, the
      registered path, bound as in F2); --lr-selection (LR_SELECTION_EXT.json, the 06B file) and S2 / S3 stage files are
      refused for MAIN (they stay smoke / screen inputs). This supersedes F2's "else LR_SELECTION_EXT.json" branch.
  M1  FA_1024 / DP / S_CAL (POISSON_ARMS) run the tuning file's optimizer / LR / shape fields (optimizer, server_lr,
      server_tau, lr_mult, lr_shape, client_opt, client_lr) with local passes 2 AND local batch 16 whatever S4 selects
      (DL 219). When the selection's passes / batch are not 2 / 16 the arm spec records poisson_local_rule (and rows
      fl_poisson_local_rule / fl_selected_local_passes / fl_selected_local_batch); LAUNCH records poisson_local_rule for
      every Poisson arm bound to a tuning file. Explicit --local-passes != 2 / --local-batch != 16 are refused for them.
  load_fl_tuning re-checks 06J §4(ii) (no measurable gain < 0.002) and the required / edge runs through the same
  lr_screen functions as select-joint. m6: the frozen-table bytes record carries a bootstrap / dropout note.
  Every default arm spec / RunConfig digest is unchanged (the new keys exist only when a rule changes a value).

FL-14 (PREG1_AMENDMENT_06L A1 / A2 / A4, read by 06L_D D1-D4):
  load_fl_tuning re-checks the 06L_D D1 (S2 + FedAvgM m {1, 2}, A1 m 0.5 edge) and D2 (S3 + the A2 cells) sets through
  lr_screen.stage_required / check_stage_runs. A4 as narrowed by 06L_E E1: a clipped z = 0 run (FA_1024) writes
  CLIP_LOG.jsonl, one line per round with survivors: round, n_clipped, n_deltas (clients with n_consumed > 0),
  clipped_fraction; never per-client norms. z > 0 runs (DP) record NOTHING about clipping, even if fl.dp ever releases
  a count (an un-noised count is outside the RDP accountant). RUN_SUMMARY clip_log cross-checks the logged rounds
  against the checkpointed participation log (complete / missing_rounds; REV-FL-14 m1). Spec / RunConfig digests and
  trajectories are unchanged.

FL-15 (PREG1_AMENDMENT_06M_B B2; REAL_WORLD_FL_PRACTICE_AUDIT R5; every option OFF by default -> every arm spec,
RunConfig digest, row and trajectory byte-identical):
  --warm-server-lr-factor f  (warm arms, screen / smoke only) the server-LR axis of the 2-D S7 / S8 screens: FedAdam
                   f in {1, 0.1} (eta* / 10), FedAvgM / FedAvg f in {1, 0.25} (eta* / 4); recorded as spec
                   warm_server_lr_factor (absent at 1), cfg.server_opt (FedAvg at 1/4 = FedAvgM beta 0 at LR 0.25) and rows
                   fl_warm_server_lr_factor / fl_warm_server_lr. A non-smoke MAIN warm run takes BOTH the local factor and
                   the server-LR factor from the registered 2-D contracts/fl/FL_WARM_SELECTION_S7|S8.json
                   (PPSI_V4_FL_WARM_SELECTION_2; the 1-D PPSI_V4_FL_WARM_SELECTION_1 file is refused).
  R5               every run except z > 0 (DP) writes TRAIN_LOSS.jsonl: one line per round (round, n_clients, n_steps,
                   loss_sum, mean_step_loss, client_mean_loss) from fl VisitRecord.loss_sum (fl.runtime step record
                   train_loss); trimmed on resume like CLIP_LOG; RUN_SUMMARY train_loss_log (complete / missing rounds).

FL-16 (PREG1_AMENDMENT_06N N3.2, 06K §3 DP8_FROZEN, 06L_E E1; new arm labels only -> every existing arm spec, RunConfig
digest, row, refusal and trajectory byte-identical):
  --arm FP_WARM | PF_WARM | FA_Q8_WARM | S_CAL_WARM | FA_1024_WARM | DP_WARM          (S7-bound: full model)
  --arm FP_WARM_FROZEN | PF_WARM_FROZEN | FA_Q8_WARM_FROZEN | S_CAL_WARM_FROZEN | FA_1024_WARM_FROZEN | DP_WARM_FROZEN
                   (S8-bound: item tables frozen; DP_WARM_FROZEN --epsilon 8 = the registered DP8_FROZEN)
                   = the cold arm's semantics (WARM_LINE: plan, knobs, FedProx mu, the PF personal component with its
                   never-multiplied personal LR, the Q8 codec, S_CAL's 20 rounds / FA_1024's z 0 / DP's registered z and
                   accountant, the DL 219 Poisson rule) started from theta_pre exactly as FA_WARM (--init-from:
                   load_pretrain_init / apply_init_state / check_init_identity, unchanged), with the warm factors of
                   the variant (FA_WARM / FA_WARM_FROZEN): a non-smoke MAIN run takes the local factor AND the server-LR
                   factor only from the registered contracts/fl/FL_WARM_SELECTION_S7|S8.json of its variant
                   (--warm-selection; smoke / dry-run: --warm-lr-factor / --warm-server-lr-factor on the variant's grid);
                   *_WARM_FROZEN needs --freeze-item-tables (clip / noise / bytes on the uploaded keys only; RUN_SUMMARY
                   dp_noise_dimension for warm DP-configured runs; the Q8 bytes carry the frozen fields). Rows / spec:
                   method = the label, method_code = the cold arm's, the FA_WARM init / factor / frozen fields.
  S_CAL_WARM[_FROZEN] writes its OWN record S_RECORD_WARM[_FROZEN].json (format PPSI_V4_DP_S_RECORD_WARM_1: the cold
                   fields + arm, variant / stage, arm_spec_sha256, init_from, frozen keys, warm factors), never the cold
                   contracts/dp/S_RECORD.json. FA_1024_WARM / DP_WARM[_FROZEN] refuse the cold record and take only the
                   warm record of their variant's S_CAL arm whose arm-spec / config digest = the S calibration of this
                   run's tuning point, warm factors and theta_pre (warm_s_cal_spec), whose init = this run's theta_pre,
                   and (in `run`) whose plan digest = the seed-2026 Poisson plan of this run's clients.
  06L_E E1 holds by the spec (unchanged gates): z > 0 (DP_WARM*) records nothing about clipping and writes no
                   TRAIN_LOSS.jsonl; FA_1024_WARM* writes CLIP_LOG / TRAIN_LOSS; S_CAL_WARM* TRAIN_LOSS only.

FL-07 (PREG1_AMENDMENT_06C; Ops follow-ups of FL-05):
  * FA_1024 / S_CAL / DP plans are {"group_size": 1024, "sampling": "poisson"} (their arm-spec and fl RunConfig
    digests changed on purpose; FA / FP / PF / FA_Q8 are byte-identical). fl.runtime refuses any DP-configured run
    whose plan is not Poisson with group_size == the DP denominator.
  * A non-smoke --role MAIN run (every arm, S_CAL included) is refused without --lr-selection (so --lr-mult is a
    smoke / CALIBRATION option only); the dry run (no training) is not refused.
  * FA_Q8 upload bytes = the codec payload + the 8-byte n_consumed header, as FA (fl.quant).

P2-01 / P2-02 (PREG2 §3.3 / §5; PHASE2_IMPLEMENTATION_PLAN; Fable gate FABLE_GATE_ENDGAME_SCRIPTS_20261003 §9). A NEW arm
family; with no P2 arm every Phase-1 arm spec, RunConfig digest, row, refusal and trajectory is byte-identical (the P2
names are outside ALL_ARMS / ARMS / WARM_*; WARM_FACTORS / WARM_SERVER_FACTORS / CLIENT_LRS / CLIENT_EDGE_LRS untouched):
  --arm P2_W --init-from <PRETRAIN_FOR_FL_INIT.json> --p2-recipe <PPSI_V4_P2_RECIPE_1 file> [--role CALIBRATION]
                   (smoke / CALIBRATION screens / dry runs) or, for a non-smoke MAIN run, --p2-selection
                   contracts/fl/P2_WARM_SELECTION_FINAL.json (the registered path only, bound by sha256 and verified by
                   v4int.p2_select.load_p2_selection; a recipe file is refused for MAIN). The recipe (v4int.p2_select
                   layout: server_opt / server_lr / client_lr / items / item_lr_mult / upload / lora_rank /
                   lora_lr_factor / personal_head) sets the server step (p2_server_opt: FedAvg eta_s != 1 = FedAvgM beta 0
                   at LR eta_s, FedAvgM beta 0.9, FedAdam tau 1e-3), the client SGD-m LR (P2 grid {0.001 .. 0.1}, admitted
                   for P2 arms only), frozen or unfrozen item tables (+ the item-LR multiplier, fl.client.ItemLRSolver)
                   and the WC5 personal head (method PF); PREG2 §5.1 fixes the rest (P2_FIXED: 1 local pass, batch 16,
                   FL_CONST, clip 1.0, 64 clients, sweep, 10 % drop-out, 6.0 EFE). Every Phase-1 tuning / warm / DP flag is
                   refused for P2_W and every P2 flag for a Phase-1 arm. Smoke runs may take up to 65 rounds (P2-T3:
                   64 server updates + the 65th TRAIN_LOSS line that depends on theta_64; tools/p2_t3_check.py).
                   Hooks for the later packets: p2_pre_server (P2-04 LoRA, before the Server reads the manifest),
                   p2_post_run (P2-03 delta_rows channel), check_p2_recipe / p2_arm_spec (P2-08 DP fields).
  D1-ext (P2 arms only)  UPDATE_NORM_LOG.jsonl: one line per round from fl.update_log.UpdateNormObserver attached as
                   fl.server.Server.observer (per tensor group of the trained subset: ||Delta_agg||, the server step,
                   ||theta - theta_pre||, cos with the previous step; n_clients; post-noise only under DP); trimmed on
                   resume like TRAIN_LOSS; RUN_SUMMARY update_norm_log (complete flag, the trained keys).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

from v4int.launch_central import (DETERMINISM_MODE, G1_GATE, LaunchRefused, _json_safe, _next_launch_path,  # noqa: E402
                                  bind_data_protocol_lock_v2, build_trainer_registry_view, data_paths,
                                  determinism_mode_in_effect, disk_record, export_environment, gpu_checks,
                                  load_production_schema, utc)

ARMS = ("FA", "FP", "PF", "FA_1024", "DP", "S_CAL", "FA_Q8")      # the registered cold-start arms (unchanged)
# FL-16 (06N N3.2; 06K §3 DP8_FROZEN): the warm method set = each cold arm's semantics started from theta_pre, bound to
# the S7 variant (FA_WARM: full model) or the S8 variant (FA_WARM_FROZEN: frozen item tables). label -> (the cold arm
# whose semantics it runs, the warm variant whose registered selection / freeze it takes).
WARM_LINE = {"FP_WARM": ("FP", "FA_WARM"), "PF_WARM": ("PF", "FA_WARM"), "FA_Q8_WARM": ("FA_Q8", "FA_WARM"),
             "S_CAL_WARM": ("S_CAL", "FA_WARM"), "FA_1024_WARM": ("FA_1024", "FA_WARM"), "DP_WARM": ("DP", "FA_WARM"),
             "FP_WARM_FROZEN": ("FP", "FA_WARM_FROZEN"), "PF_WARM_FROZEN": ("PF", "FA_WARM_FROZEN"),
             "FA_Q8_WARM_FROZEN": ("FA_Q8", "FA_WARM_FROZEN"), "S_CAL_WARM_FROZEN": ("S_CAL", "FA_WARM_FROZEN"),
             "FA_1024_WARM_FROZEN": ("FA_1024", "FA_WARM_FROZEN"), "DP_WARM_FROZEN": ("DP", "FA_WARM_FROZEN")}
WARM_LINE_ARMS = tuple(WARM_LINE)
ALL_ARMS = ARMS + ("FA_WARM", "FA_WARM_FROZEN") + WARM_LINE_ARMS  # FL-12 (06K) warm arms + FL-16 warm method set
METHOD_CODE = {"FA": "FA", "FP": "FP", "PF": "PF", "FA_1024": "FA", "DP": "FA", "S_CAL": "FA", "FA_Q8": "FA",
               "FA_WARM": "FA", "FA_WARM_FROZEN": "FA"}
METHOD_CODE.update({a: METHOD_CODE[b] for a, (b, _v) in WARM_LINE.items()})   # FL-16: the cold arm's method code
SWEEP_ARMS = ("FA", "FP", "PF", "FA_Q8", "FA_WARM", "FA_WARM_FROZEN")
POISSON_ARMS = ("FA_1024", "DP", "S_CAL")                         # 06C Poisson arms (T = 384, registered accountant)
KNOBS = {"group": 64, "sampling": "sweep", "dropout_p": 0.1, "rounds": 6827, "batch": 16, "passes": 2, "mu": 0.01,
         "pf_lam": 1e-4, "pf_clip": 1.0, "clip": 1.0, "weight_decay": 1e-5, "m": 1024, "rounds_1024": 384,
         "s_cal_rounds": 20, "s_cal_seed": 2026, "n_shards": 2, "max_attempts": 2, "endpoint_efe": 6.0}
DP_Z_RECORD = CODE / "tests_fl" / "receipts" / "DP_Z_RECORD_FL04.json"
DP_REGISTERED_Z = {8.0: 0.5642, 1.0: 1.3094}                        # amendment 06 §2 Q5 (receipt FL-04)
DP_ACCOUNTING = {"m": 1024, "N": 131072, "T": 384, "delta": 5e-06}
S_FORMAT = "PPSI_V4_DP_S_RECORD_1"
ARCH_LOCK = V4 / "contracts" / "ARCHITECTURE_LOCK.json"
FAMILY = "SASREC"
MAX_SMOKE_ROUNDS = 50
EXPOSURE_POINT = "end"
# FL-05 (amendment 06B). Never part of KNOBS (KNOBS is inside every arm spec digest).
# FL-09 (amendment 06F §1): the grid is extended to 4, 8, 16 (+ 32 on the edge rule); m = 1 still drops out of the digest.
LR_MULTS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
LR_MULT_TAG = {0.5: "0.5", 1.0: "1", 2.0: "2", 4.0: "4", 8.0: "8", 16.0: "16", 32.0: "32"}
FIRST_SCREEN_MULTS = (0.5, 1.0, 2.0)                      # the 06B §1 screen (the LR_SELECTION_1 file's values)
ROLES = ("MAIN", "CALIBRATION")
CALIB = {"arms": ("FA", "FA_WARM", "FA_WARM_FROZEN"), "seed": 2026, "n_clients": 32768, "rounds": 1707,
         "run_label": "FL_LR_SCREEN"}
MAIN_N_CLIENTS = 131072                                    # registered MAIN_TRAIN clients (= DP_ACCOUNTING N)
Q8_CODEC = {"name": "Q8_SR_PER_TENSOR", "bits": 8, "qmax": 127, "scale": "max|delta| / 127 per tensor (FP32)",
            "rounding": "stochastic, torch.Generator seeded derive_seed(seed, 'q8', round, client_key)",
            "upload_bytes": "sum(numel) + 4 x n_tensors", "download": "dense FP32", "authority": "amendment 06B §5"}
LR_SELECTION_FORMAT = "PPSI_V4_FL_LR_SELECTION_1"
LR_SELECTION_EXT_FORMAT = "PPSI_V4_FL_LR_SELECTION_EXT_1"   # FL-09 (06F §3): the extended-screen selection
LR_SELECTION_EXT_NAME = "LR_SELECTION_EXT.json"
LR_SELECTION_EXT_PATH = V4 / "contracts" / "fl" / LR_SELECTION_EXT_NAME
FL05_DEFAULTS = {"lr_mult": 1.0, "role": "MAIN", "upload_codec": None}
# FL-10 (amendments 06G / 06I §3 / 06J). Spec / cfg / row keys exist ONLY for non-default values (defaults: no key, so
# every pre-FL-10 digest and file is unchanged).
SERVER_OPTS = ("fedavg", "fedadam", "fedavgm")
FEDADAM_LRS = (0.01, 0.1)                                  # 06G §2
FEDADAM_EDGE_LRS = (0.003, 0.001)                          # 06J §1 edge rule
FEDADAM_TAUS = (1e-3, 1e-5)                                # default (06G) + the 06J §1 edge
FEDAVGM_LR = 1.0                                           # 06J §1 (fixed)
FEDADAM_HYPER = {"beta1": 0.9, "beta2": 0.99, "tau": 1e-3, "v_init": "tau_squared"}   # = fl.server defaults
FEDAVGM_HYPER = {"beta1": 0.9}                              # server momentum (= fl.server FEDAVGM_DEFAULTS)
LR_SHAPES = ("central", "fl_const")
CLIENT_OPTS = ("adamw_fresh", "sgd_m")
CLIENT_LRS = (0.01, 0.1, 1.0)                              # 06I §3
CLIENT_EDGE_LRS = (3.0, 10.0, 0.003)                       # 06J §2 edge rule
TUNING_KEYS = ("optimizer", "server_lr", "server_tau", "lr_mult", "lr_shape", "client_opt", "client_lr",
               "local_passes", "local_batch")
TUNING_DEFAULTS = {"optimizer": "fedavg", "server_lr": None, "server_tau": None, "lr_mult": 1.0,
                   "lr_shape": "central", "client_opt": "adamw_fresh", "client_lr": None,
                   "local_passes": 2, "local_batch": 16}
LOCAL_PASSES = (1, 2, 4)                                   # FL-13 (06J §3 S4); default 2 = KNOBS["passes"]
LOCAL_BATCHES = (8, 16)                                    # default 16 = KNOBS["batch"]
# FL-12 (06K §2-§3): warm start from the PRE model (S7) and frozen item tables (S8).
WARM_ARMS = ("FA_WARM", "FA_WARM_FROZEN")
WARM_STAGE = {"FA_WARM": "S7", "FA_WARM_FROZEN": "S8"}
WARM_FACTORS = {"FA_WARM": (1.0, 0.25), "FA_WARM_FROZEN": (1.0, 4.0)}   # 06K: m in {m*, m*/4} / {m*, 4 m*}
FROZEN_ITEM_KEYS = ("item_embed.weight", "output_embed", "output_bias")   # = fl.runtime.FROZEN_ITEM_KEYS
# FL-16: the warm DP chain's own S record (never contracts/dp/S_RECORD.json, which is the cold chain's).
S_FORMAT_WARM = "PPSI_V4_DP_S_RECORD_WARM_1"
S_CAL_WARM_ARM = {"FA_WARM": "S_CAL_WARM", "FA_WARM_FROZEN": "S_CAL_WARM_FROZEN"}      # variant -> its S_CAL arm
S_RECORD_WARM_FILE = {"FA_WARM": "S_RECORD_WARM.json", "FA_WARM_FROZEN": "S_RECORD_WARM_FROZEN.json"}
FL16_AUTHORITY = "PREG1_AMENDMENT_06N N3.2 + 06K §3 (DP8_FROZEN) + 06L_E E1 (FL-16)"
# P2-01 / P2-02 (PREG2 §3.3 / §5; PHASE2_IMPLEMENTATION_PLAN; Fable gate FABLE_GATE_ENDGAME_SCRIPTS_20261003 §9). The
# Phase-2 warm-FL arm family is NOT in ALL_ARMS / ARMS / WARM_* (every Phase-1 list, grid and refusal is untouched);
# its values live only in the P2_* names below and are admitted only for a P2 arm.
P2_ARMS = ("P2_W",)                                         # method FA (PF when the recipe has personal_head, WC5)
METHOD_CODE.update({"P2_W": "FA"})                          # the ARCHITECTURE_LOCK method code (rows: PF for WC5)
P2_RECIPE_FORMAT = "PPSI_V4_P2_RECIPE_1"                    # the --p2-recipe file (smoke / CALIBRATION)
P2_RECIPE_KEYS = ("server_opt", "server_lr", "client_lr", "items", "item_lr_mult", "upload", "lora_rank",
                  "lora_lr_factor", "personal_head")       # = v4int.p2_select.RECIPE_KEYS (P2-06)
P2_SERVER_LRS = {                                           # PREG2 §5.2 WA core / WA-ext / incumbent + the §5.7 steps
    "fedavg": {"core": (0.3, 1.0), "edge": (0.1, 2.0)},
    "fedavgm": {"core": (0.03, 0.1), "edge": (0.01, 0.3)},
    "fedadam": {"core": (1e-4, 3e-4), "ext": (1e-5, 1e-4), "incumbent": (1e-3,), "edge": (3e-5, 1e-3, 3e-6)}}
P2_CLIENT_LRS = {"core": (0.003, 0.01, 0.03), "ext": (0.1,), "edge": (0.001, 0.1)}   # client SGD-m 0.9 (§5.2 / §5.7)
P2_ITEM_LR_MULTS = (0.1, 0.01, 0.001)                       # WB1 / WB3 0.1, WB2 0.01, §5.7 edge 0.001 (§5.3)
P2_UPLOADS = ("dense",)                                     # P2-03 hook: + "delta_rows" (WB3) with install_sparse
P2_FIXED = {"client_opt": "sgd_m", "client_momentum": 0.9, "lr_shape": "fl_const", "local_passes": 1, "local_batch": 16,
            "clip": 1.0, "weight_decay": 0.0, "group": 64, "sampling": "sweep", "dropout_p": 0.1,
            "endpoint_efe": 6.0, "authority": "PREG2 §5.1"}
P2_WA_CELLS = {"core_server": (("fedavg", 0.3), ("fedavg", 1.0), ("fedavgm", 0.03), ("fedavgm", 0.1),
                               ("fedadam", 1e-4), ("fedadam", 3e-4)),
               "core_client": (0.003, 0.01, 0.03), "ext": (("fedadam", 1e-5), ("fedadam", 1e-4)), "ext_client": 0.1,
               "incumbent": ("fedadam", 1e-3, 0.1)}          # 18 + 2 + 1 (§5.2; S8's f1/sf0.1 = run S8Tc)
P2_CALIB_TIE_BAND = 0.001                                   # Fable gate §9.2 E-O4a (applied by v4int.p2_select)
P2_FUTILITY = 0.001                                         # Fable gate §9.2 E-O4b (applied by v4int.p2_select)
P2_MAX_SMOKE_ROUNDS = 65                                    # P2-T3: 64 updates + the 65th loss line (REV-P2-01_02 m2)
P2_SELECTION_FORMAT = "PPSI_V4_P2_WARM_SELECTION_FINAL_1"   # = v4int.p2_select FORMATS["FINAL"]
P2_SELECTION_PATH = V4 / "contracts" / "fl" / "P2_WARM_SELECTION_FINAL.json"
P2_REFUSED_FLAGS = ("lr_mult", "lr_selection", "fl_tuning", "server_opt", "server_lr", "server_tau", "lr_shape",
                    "client_opt", "client_lr", "local_passes", "local_batch", "warm_lr_factor", "warm_selection",
                    "warm_server_lr_factor", "clip_record", "epsilon", "theta0_variants")   # Phase-1 options
P2_FLAGS = ("p2_recipe", "p2_selection")
P2_AUTHORITY = ("PREG2 §5 (contracts/PREG2_FOLLOWUP_DRAFT §5 as edited by FABLE_GATE_ENDGAME_SCRIPTS_20261003 §9); "
                "PHASE2_IMPLEMENTATION_PLAN P2-02")
UPDATE_NORM_FILE = "UPDATE_NORM_LOG.jsonl"                  # P2-01 D1-ext (PREG2 §3.3)
UPDATE_NORM_AUTHORITY = ("PREG2 §3.3 D1-ext: per round, per tensor group of the trained subset: ||Delta_agg||, the server "
                         "step, ||theta - theta_pre||, cos(Delta_r, Delta_r-1), clients aggregated (omitted for z > 0); "
                         "DESCRIPTIVE, never selects; post-noise server-side quantities only under DP (06L_E E1)")


def is_p2_arm(arm) -> bool:
    """P2-02: a Phase-2 warm-FL arm (never a Phase-1 arm)."""
    return arm in P2_ARMS


def base_arm(arm: str) -> str:
    """FL-16: the cold arm whose semantics an arm runs (a warm-line arm -> its cold arm; every other arm -> itself)."""
    return WARM_LINE[arm][0] if arm in WARM_LINE else arm


def warm_variant(arm: str) -> Optional[str]:
    """FL-16: the warm variant of an arm (FA_WARM = S7 full model, FA_WARM_FROZEN = S8 frozen tables), None if cold."""
    if arm in WARM_ARMS:
        return arm
    return WARM_LINE[arm][1] if arm in WARM_LINE else None


def is_warm_arm(arm: str) -> bool:
    return warm_variant(arm) is not None


def is_frozen_arm(arm: str) -> bool:
    return warm_variant(arm) == "FA_WARM_FROZEN"


def is_sweep_arm(arm) -> bool:
    return arm in SWEEP_ARMS or base_arm(arm) in SWEEP_ARMS


def is_poisson_arm(arm) -> bool:
    return base_arm(arm) in POISSON_ARMS


PRETRAIN = {"run_label": "PRETRAIN_FOR_FL", "run_id_prefix": "PRETRAIN_FOR_FL_", "fl_init_kind": "endpoint_6.0",
            "train_manifest": "PRETRAIN_TRAIN_5_100_NONMAIN", "family": "SASREC"}   # = v4int.pretrain_small
MAIN_TRAIN_MANIFESTS = ("MAIN_TRAIN_5_100_131072", "CALIB_TRAIN_5_100_32768", "FULL_TRAIN_5_100_TIEBREAK")
MAIN_RUN_MARKERS = ("C_BIG", "C_SMALL", "C_FULL", "C_MATCHED")   # central runs whose data include MAIN users
WARM_SELECTION_FORMAT = "PPSI_V4_FL_WARM_SELECTION_1"            # the superseded 1-D (local factor) file
WARM_SELECTION_PATH = {st: V4 / "contracts" / "fl" / f"FL_WARM_SELECTION_{st}.json" for st in ("S7", "S8")}
# FL-15 (PREG1_AMENDMENT_06M_B B2): the registered S7 / S8 screens are 2-D = local factor x server-LR factor.
WARM_SELECTION_FORMAT_2D = "PPSI_V4_FL_WARM_SELECTION_2"
WARM_SERVER_FACTORS = {"fedadam": (1.0, 0.1), "fedavgm": (1.0, 0.25), "fedavg": (1.0, 0.25)}   # {eta*, eta*/10 | /4}
WARM_SERVER_DIVISORS = {0.1: 10, 0.25: 4}                   # eta* / 10, eta* / 4 (exact division, no 0.1 x rounding)
FL_TUNING_FORMAT = "PPSI_V4_FL_TUNING_SELECTION_1"
FL_TUNING_NAME = "FL_TUNING_SELECTION.json"
FL_TUNING_PATH = V4 / "contracts" / "fl" / FL_TUNING_NAME


def fa_rounds(n_clients: int, passes: int = 2) -> int:
    """The 6.0-EFE round count of the sweep arms (06J §3): ceil(6.0 N_dec / (64 x 0.9 x P x N_dec / N)) =
    ceil(60 N / (576 P)) in integer arithmetic (group 64, drop-out 0.1): P = 2 -> 6,827 for MAIN, 1,707 for CALIB."""
    return -(-(60 * int(n_clients)) // (KNOBS["group"] * 9 * int(passes)))


def exposure_budget(n_clients: int, passes: int) -> dict:
    """FL-13: the recomputed round count of a non-default local-pass count (06J §3), as recorded in the spec."""
    return {"endpoint_efe": KNOBS["endpoint_efe"], "n_clients": int(n_clients), "passes": int(passes),
            "group": KNOBS["group"], "dropout_p": KNOBS["dropout_p"], "rounds": fa_rounds(n_clients, passes),
            "formula": "rounds = ceil(6.0 x N_dec / (64 x 0.9 x P x N_dec / N_clients)) = ceil(60 N_clients / (576 P))",
            "authority": "PREG1_AMENDMENT_06J §3"}


def lr_mult_value(m) -> float:
    """The registered multipliers only (06B §1 + 06F §1); any other value is refused."""
    try:
        v = float(m)
    except (TypeError, ValueError):
        raise LaunchRefused(f"--lr-mult {m!r} is not a number") from None
    if v not in LR_MULTS:
        raise LaunchRefused(f"--lr-mult must be one of {LR_MULTS} (amendments 06B §1 / 06F §1), not {m}")
    return v


def _num(v, what):
    try:
        return float(v)
    except (TypeError, ValueError):
        raise LaunchRefused(f"{what} {v!r} is not a number") from None


def num_tag(v) -> str:
    """Short numeric tag for labels / run ids (0.5 -> '0.5', 1.0 -> '1', 1e-05 -> '1e-05')."""
    return LR_MULT_TAG[float(v)] if float(v) in LR_MULT_TAG else format(float(v), "g")


def server_opt_value(opt, lr) -> tuple:
    """FL-10: (optimizer, eta_s) — ("fedavg", None), ("fedadam", 0.01 | 0.1 | 0.003 | 0.001) or ("fedavgm", 1.0)."""
    o = "fedavg" if opt is None else str(opt).lower()
    if o not in SERVER_OPTS:
        raise LaunchRefused(f"--server-opt must be one of {SERVER_OPTS}, not {opt}")
    if o == "fedavg":
        if lr is not None:
            raise LaunchRefused("--server-lr is a FedAdam option: refused with --server-opt fedavg (server LR 1)")
        return o, None
    if o == "fedavgm":
        if lr is not None and _num(lr, "--server-lr") != FEDAVGM_LR:
            raise LaunchRefused(f"FedAvgM runs at the registered server LR {FEDAVGM_LR} only (06J §1), not {lr}")
        return o, FEDAVGM_LR
    if lr is None:
        raise LaunchRefused("--server-opt fedadam needs --server-lr (0.01 or 0.1; amendment 06G §2)")
    v = _num(lr, "--server-lr")
    if v not in FEDADAM_LRS + FEDADAM_EDGE_LRS:
        raise LaunchRefused(f"--server-lr must be one of {FEDADAM_LRS + FEDADAM_EDGE_LRS} (06G §2 / 06J §1), not {lr}")
    return o, v


def local_value(v, grid, default, what) -> int:
    """FL-13: a registered local-computation value (None -> the KNOBS default)."""
    if v is None:
        return int(default)
    x = _num(v, what)
    if x != int(x) or int(x) not in grid:
        raise LaunchRefused(f"{what} must be one of {grid} (PREG1_AMENDMENT_06J §3), not {v}")
    return int(x)


def tuning_config(*, optimizer=None, server_lr=None, server_tau=None, lr_mult=1.0, lr_shape=None, client_opt=None,
                  client_lr=None, local_passes=None, local_batch=None) -> dict:
    """FL-10: the normalised, validated tuning point (TUNING_KEYS); anything unregistered is refused.
    FL-13: + local_passes {1, 2, 4} / local_batch {8, 16} (06J §3: one at a time; None = the default 2 / 16)."""
    opt, slr = server_opt_value(optimizer, server_lr)
    tau = None
    if server_tau is not None:
        tau = _num(server_tau, "--server-tau")
        if opt != "fedadam":
            raise LaunchRefused("--server-tau is a FedAdam option")
        if tau not in FEDADAM_TAUS:
            raise LaunchRefused(f"--server-tau must be one of {FEDADAM_TAUS} (06G / 06J §1), not {server_tau}")
    if opt == "fedadam" and tau is None:
        tau = FEDADAM_HYPER["tau"]
    m = lr_mult_value(1.0 if lr_mult is None else lr_mult)
    shape = "central" if lr_shape is None else str(lr_shape).lower()
    if shape not in LR_SHAPES:
        raise LaunchRefused(f"--lr-shape must be one of {LR_SHAPES}, not {lr_shape}")
    copt = "adamw_fresh" if client_opt is None else str(client_opt).lower()
    if copt not in CLIENT_OPTS:
        raise LaunchRefused(f"--client-opt must be one of {CLIENT_OPTS}, not {client_opt}")
    clr = None
    if copt == "adamw_fresh":
        if client_lr is not None:
            raise LaunchRefused("--client-lr is a client-SGD option: refused with --client-opt adamw_fresh")
    else:
        if client_lr is None:
            raise LaunchRefused("--client-opt sgd_m needs --client-lr (0.01, 0.1 or 1.0; 06I §3)")
        clr = _num(client_lr, "--client-lr")
        if clr not in CLIENT_LRS + CLIENT_EDGE_LRS:
            raise LaunchRefused(f"--client-lr must be one of {CLIENT_LRS + CLIENT_EDGE_LRS} (06I §3 / 06J §2), not "
                                f"{client_lr}")
        if m != 1.0:
            raise LaunchRefused("client SGD takes its own LR grid (06I §3): --lr-mult must be 1 with --client-opt sgd_m")
    lp = local_value(local_passes, LOCAL_PASSES, KNOBS["passes"], "--local-passes")
    lb = local_value(local_batch, LOCAL_BATCHES, KNOBS["batch"], "--local-batch")
    if lp != KNOBS["passes"] and lb != KNOBS["batch"]:
        raise LaunchRefused("S4 screens local passes and local batch one at a time (06J §3): not both non-default")
    return {"optimizer": opt, "server_lr": slr, "server_tau": tau, "lr_mult": m, "lr_shape": shape,
            "client_opt": copt, "client_lr": clr, "local_passes": lp, "local_batch": lb}


def tuning_label(t: Mapping) -> str:
    """'fedavg|m=2', 'fedadam|eta=0.01|m=4', 'fedadam|eta=0.01|tau=1e-05|m=4', 'fedavgm|m=1|shape=fl_const|client=sgd_m|clr=0.1'."""
    parts = [t["optimizer"]]
    if t["optimizer"] == "fedadam":
        parts.append(f"eta={num_tag(t['server_lr'])}")
        if t["server_tau"] != FEDADAM_HYPER["tau"]:
            parts.append(f"tau={num_tag(t['server_tau'])}")
    parts.append(f"m={num_tag(t['lr_mult'])}")
    if t["lr_shape"] != "central":
        parts.append(f"shape={t['lr_shape']}")
    if t["client_opt"] != "adamw_fresh":
        parts.append(f"client={t['client_opt']}|clr={num_tag(t['client_lr'])}")
    if t.get("local_passes", KNOBS["passes"]) != KNOBS["passes"]:      # FL-13 (labels of S2 / S3 points unchanged)
        parts.append(f"passes={int(t['local_passes'])}")
    if t.get("local_batch", KNOBS["batch"]) != KNOBS["batch"]:
        parts.append(f"batch={int(t['local_batch'])}")
    return "|".join(parts)


def row_labels(t: Mapping) -> dict:
    """The row fields of a tuning point (non-default dimensions only)."""
    out = {}
    if t["optimizer"] != "fedavg":
        out.update(fl_server_opt=t["optimizer"], fl_server_lr=float(t["server_lr"]))
        if t["optimizer"] == "fedadam" and t["server_tau"] != FEDADAM_HYPER["tau"]:
            out["fl_server_tau"] = float(t["server_tau"])
    if t["lr_shape"] != "central":
        out["fl_lr_shape"] = t["lr_shape"]
    if t["client_opt"] != "adamw_fresh":
        out.update(fl_client_opt=t["client_opt"], fl_client_lr=float(t["client_lr"]))
    if t.get("local_passes", KNOBS["passes"]) != KNOBS["passes"]:      # FL-13
        out["fl_local_passes"] = int(t["local_passes"])
    if t.get("local_batch", KNOBS["batch"]) != KNOBS["batch"]:
        out["fl_local_batch"] = int(t["local_batch"])
    return out


def _apply_tuning(spec: dict, t: Mapping) -> None:
    """Write the non-default dimensions of t into the spec / cfg (arm_spec; mirrored by spec_tuning)."""
    cfg = spec["cfg"]
    if t["optimizer"] == "fedadam":
        cfg["server_opt"] = {"name": "fedadam", "lr": t["server_lr"], **FEDADAM_HYPER, "tau": t["server_tau"]}
        spec.update(server_opt="fedadam", server_lr=t["server_lr"])
        if t["server_tau"] != FEDADAM_HYPER["tau"]:
            spec["server_tau"] = t["server_tau"]
    elif t["optimizer"] == "fedavgm":
        cfg["server_opt"] = {"name": "fedavgm", "lr": FEDAVGM_LR, **FEDAVGM_HYPER}
        spec.update(server_opt="fedavgm", server_lr=FEDAVGM_LR)
    if t["lr_shape"] != "central":
        cfg["lr_shape"] = t["lr_shape"]
        spec["lr_shape"] = t["lr_shape"]
    if t["client_opt"] == "sgd_m":
        cfg["lr_peak_local"] = t["client_lr"]
        cfg["solver"].update(lr=t["client_lr"], optimizer="sgd_m", weight_decay=0.0, fused=None)
        spec.update(client_opt="sgd_m", client_lr=t["client_lr"])
    lp, lb = int(t.get("local_passes", KNOBS["passes"])), int(t.get("local_batch", KNOBS["batch"]))
    if lb != KNOBS["batch"]:                              # FL-13 (06J §3): exposures unchanged
        if is_poisson_arm(spec.get("arm")):               # FIX-DELTA-02 M1 (DL 219): batch 16 for the Poisson arms
            raise LaunchRefused(f"--local-batch {lb}: the Poisson arms (FA_1024 / S_CAL / DP) keep local batch "
                                f"{KNOBS['batch']} (DECISION_LOG row 219)")
        cfg["solver"]["batch_size"] = lb
        spec["local_batch"] = lb
    if lp != KNOBS["passes"]:                             # FL-13 (06J §3): rounds keep the 6.0-EFE budget
        if not is_sweep_arm(spec.get("arm")):            # FL-16: + the warm Poisson arms
            raise LaunchRefused(f"--local-passes {lp}: the Poisson arms (FA_1024 / S_CAL / DP) keep P = 2 (their T = "
                                f"{KNOBS['rounds_1024']} is the registered DP accountant's; an amendment must register "
                                "another T first)")
        n = CALIB["n_clients"] if spec.get("role") == "CALIBRATION" else MAIN_N_CLIENTS
        eb = exposure_budget(n, lp)
        cfg["solver"]["passes"] = lp
        cfg["endpoint_rounds"] = eb["rounds"]
        spec.update(local_passes=lp, exposure_budget=eb, max_rounds=eb["rounds"])


def spec_tuning(spec: Mapping) -> dict:
    """FL-10: the tuning point recorded in an arm spec, verified against its cfg exactly as arm_spec writes it
    (default dimensions must be ABSENT; LaunchRefused otherwise)."""
    t = tuning_config(optimizer=spec.get("server_opt"), server_lr=spec.get("server_lr"),
                      server_tau=spec.get("server_tau"), lr_mult=spec.get("lr_mult", 1.0),
                      lr_shape=spec.get("lr_shape"), client_opt=spec.get("client_opt"), client_lr=spec.get("client_lr"),
                      local_passes=spec.get("local_passes"), local_batch=spec.get("local_batch"))
    if (spec.get("server_opt") == "fedavg" or spec.get("lr_shape") == "central" or spec.get("client_opt") == "adamw_fresh"
            or ("server_tau" in spec and t["server_tau"] == FEDADAM_HYPER["tau"])
            or spec.get("local_passes") == KNOBS["passes"] or spec.get("local_batch") == KNOBS["batch"]
            or ("warm_lr_factor" in spec and float(spec["warm_lr_factor"]) == 1.0)
            or ("warm_server_lr_factor" in spec and float(spec["warm_server_lr_factor"]) == 1.0)):   # FL-15
        raise LaunchRefused("a default tuning dimension is recorded in the spec (arm_spec never writes one)")
    sol0 = spec["cfg"]["solver"]                          # FL-13: the solver's passes / batch = the tuning point's
    if (sol0.get("passes"), sol0.get("batch_size")) != (t["local_passes"], t["local_batch"]):
        raise LaunchRefused(f"solver passes / batch {sol0.get('passes')} / {sol0.get('batch_size')} != the spec's "
                            f"{t['local_passes']} / {t['local_batch']}")
    eb = spec.get("exposure_budget")
    if (eb is not None) != (t["local_passes"] != KNOBS["passes"]):
        raise LaunchRefused("exposure_budget is recorded iff local_passes is not the default")
    if eb is not None and (eb != exposure_budget(eb.get("n_clients", 0), t["local_passes"])
                           or spec["cfg"].get("endpoint_rounds") != eb["rounds"]):
        raise LaunchRefused(f"exposure_budget {eb} is not the 06J §3 round count of the spec (endpoint_rounds "
                            f"{spec['cfg'].get('endpoint_rounds')})")
    probe = {"arm": spec.get("arm"), "role": spec.get("role", "MAIN"), "cfg": {"solver": {}}}
    _apply_tuning(probe, t)
    if "warm_server_lr_factor" in spec:                   # FL-15 (06M_B B2): the warm server-LR factor's server step
        probe["cfg"]["server_opt"] = warm_server_opt(t, warm_server_factor_value(
            spec.get("arm"), t["optimizer"], spec["warm_server_lr_factor"]))
    if spec["cfg"].get("server_opt") != probe["cfg"].get("server_opt") or             spec["cfg"].get("lr_shape") != probe["cfg"].get("lr_shape"):
        raise LaunchRefused(f"cfg server_opt / lr_shape {spec['cfg'].get('server_opt')} / {spec['cfg'].get('lr_shape')} "
                            f"do not match the spec's tuning point {tuning_label(t)}")
    sol = spec["cfg"]["solver"]
    wf = float(spec.get("warm_lr_factor", 1.0))           # FL-12: the warm-start LR factor (1 = absent)
    if t["client_opt"] == "sgd_m":
        clr = t["client_lr"] if wf == 1.0 else wf * t["client_lr"]
        if (sol.get("optimizer"), sol.get("lr"), sol.get("weight_decay"), sol.get("fused"),
                spec["cfg"]["lr_peak_local"]) != ("sgd_m", clr, 0.0, None, clr):
            raise LaunchRefused("the solver does not match the spec's client SGD point")
    elif sol.get("optimizer", "adamw") != "adamw":
        raise LaunchRefused("an AdamW-fresh spec has a non-AdamW solver")
    if "poisson_local_rule" in spec:                      # FIX-DELTA-02 M1: exactly as arm_spec writes it
        check_poisson_rule(spec.get("arm"), spec["poisson_local_rule"], t)
    return t


POISSON_RULE_AUTHORITY = ("DECISION_LOG row 219 (registered 2026-09-29T15:00Z before any S4 read; REV-DELTA-02 M1 / "
                          "FIX-DELTA-02): FA_1024 / DP / S_CAL keep local passes 2 and local batch 16 whatever S4 "
                          "selects; only the optimizer / LR / shape fields of FL_TUNING_SELECTION.json apply")
POISSON_TAKEN_FIELDS = ("optimizer", "server_lr", "server_tau", "lr_mult", "lr_shape", "client_opt", "client_lr")


def poisson_rule_record(selected_passes: int, selected_batch: int) -> dict:
    """The arm-spec record of the DL 219 rule when it overrides the selection (its passes / batch are not 2 / 16)."""
    return {"local_passes": KNOBS["passes"], "local_batch": KNOBS["batch"], "selected_local_passes": int(selected_passes),
            "selected_local_batch": int(selected_batch), "authority": "DECISION_LOG row 219"}


def poisson_tuning(arm: str, t: Mapping) -> tuple:
    """FIX-DELTA-02 M1 (DL 219): (the tuning point a Poisson arm runs, the spec record or None, the LAUNCH record) for
    a selection-file winner t: the optimizer / LR / shape fields of t with local passes 2 and batch 16. The spec record
    exists only when the rule changes t (so a default-local selection gives the plain spec, digest unchanged)."""
    if not is_poisson_arm(arm):                           # FL-16: + FA_1024 / DP / S_CAL _WARM[_FROZEN]
        return dict(t), None, None
    lp, lb = int(t.get("local_passes", KNOBS["passes"])), int(t.get("local_batch", KNOBS["batch"]))
    eff = tuning_config(**{k: t.get(k) for k in POISSON_TAKEN_FIELDS})
    rec = None if (lp, lb) == (KNOBS["passes"], KNOBS["batch"]) else poisson_rule_record(lp, lb)
    launch = {"applied": rec is not None, "arm": arm, "selected_local_passes": lp, "selected_local_batch": lb,
              "local_passes": KNOBS["passes"], "local_batch": KNOBS["batch"], "fields_taken": list(POISSON_TAKEN_FIELDS),
              "authority": POISSON_RULE_AUTHORITY}
    return eff, rec, launch


def check_poisson_rule(arm, rec, t: Mapping) -> None:
    """A spec's poisson_local_rule: a Poisson arm at passes 2 / batch 16 whose record is exactly the DL 219 record of a
    non-default selected (passes, batch)."""
    if not is_poisson_arm(arm):
        raise LaunchRefused(f"poisson_local_rule is a FA_1024 / DP / S_CAL record, not {arm}'s")
    if (t["local_passes"], t["local_batch"]) != (KNOBS["passes"], KNOBS["batch"]):
        raise LaunchRefused("poisson_local_rule with a non-default local passes / batch")
    try:
        want = poisson_rule_record(rec["selected_local_passes"], rec["selected_local_batch"])
        local_value(rec["selected_local_passes"], LOCAL_PASSES, KNOBS["passes"], "selected_local_passes")
        local_value(rec["selected_local_batch"], LOCAL_BATCHES, KNOBS["batch"], "selected_local_batch")
    except (KeyError, TypeError) as e:
        raise LaunchRefused(f"poisson_local_rule {rec!r} is malformed ({e})") from e
    if dict(rec) != want or (want["selected_local_passes"], want["selected_local_batch"]) == (KNOBS["passes"],
                                                                                              KNOBS["batch"]):
        raise LaunchRefused(f"poisson_local_rule {rec!r} is not the DL 219 record of a non-default selection")


def poisson_row_labels(spec: Mapping) -> dict:
    """FIX-DELTA-02 M1: row fields of a Poisson arm whose selection's passes / batch were overridden (else none)."""
    rec = spec.get("poisson_local_rule")
    if not rec:
        return {}
    return {"fl_poisson_local_rule": "DL219_P2_B16", "fl_selected_local_passes": int(rec["selected_local_passes"]),
            "fl_selected_local_batch": int(rec["selected_local_batch"])}


def parse_args(argv=None) -> argparse.Namespace:
    from v4hist.policy import HISTORY_POLICIES
    ap = argparse.ArgumentParser(description="PPSI V4 federated-arm launcher (INT-03)")
    ap.add_argument("--arm", required=True, type=str.upper, choices=ALL_ARMS + P2_ARMS)   # P2-02: + the P2 family
    ap.add_argument("--family", default=FAMILY, type=str.upper)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--history-policy", required=True, choices=HISTORY_POLICIES)
    ap.add_argument("--small-recipe", type=Path, default=None, help="default v4int.small_recipe.RECIPE_PATH")
    ap.add_argument("--epsilon", type=float, default=None, help="DP only: 8 or 1 (registered z)")
    ap.add_argument("--lr-mult", type=float, default=None, help="FL-05/FL-09: 0.5, 1, 2, 4, 8, 16 or 32 x the local peak LR (default 1)")
    ap.add_argument("--lr-selection", type=Path, default=None, help="FL-05: LR_SELECTION.json of v4int.lr_screen")
    ap.add_argument("--role", type=str.upper, default="MAIN", help="FL-05: MAIN (default) or CALIBRATION (FA only)")
    ap.add_argument("--server-opt", type=str.lower, default=None, choices=SERVER_OPTS,
                    help="FL-10: fedavg (default, server LR 1), fedadam or fedavgm")
    ap.add_argument("--server-lr", type=float, default=None, help="FL-10: FedAdam eta_s 0.01 / 0.1 (+ 06J edge)")
    ap.add_argument("--server-tau", type=float, default=None, help="FL-10: FedAdam tau (default 1e-3; 06J edge 1e-5)")
    ap.add_argument("--lr-shape", type=str.lower, default=None, choices=LR_SHAPES,
                    help="FL-10: central (default) or fl_const")
    ap.add_argument("--client-opt", type=str.lower, default=None, choices=CLIENT_OPTS,
                    help="FL-10: adamw_fresh (default) or sgd_m")
    ap.add_argument("--client-lr", type=float, default=None, help="FL-10: client SGD LR 0.01 / 0.1 / 1.0 (+ 06J edge)")
    ap.add_argument("--fl-tuning", type=Path, default=None, help="FL-10: FL_TUNING_SELECTION.json of lr_screen")
    ap.add_argument("--local-passes", type=int, default=None, help="FL-13 (06J §3 S4): 1, 2 (default) or 4")
    ap.add_argument("--local-batch", type=int, default=None, help="FL-13 (06J §3 S4): 8 or 16 (default)")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="FL-12 (06K §2): PRETRAIN_FOR_FL_INIT.json of the PRE run (FA_WARM* and the FL-16 *_WARM* arms)")
    ap.add_argument("--freeze-item-tables", action="store_true",
                    help="FL-12 (06K §3): FA_WARM_FROZEN / FL-16 *_WARM_FROZEN — item tables frozen at theta_pre, never uploaded")
    ap.add_argument("--warm-lr-factor", type=float, default=None,
                    help="FL-12: local-LR factor of a warm arm (S7 1 / 0.25, S8 1 / 4); screen / smoke only")
    ap.add_argument("--warm-selection", type=Path, default=None,
                    help="FL-12: the registered contracts/fl/FL_WARM_SELECTION_S7|S8.json (non-smoke MAIN warm runs)")
    ap.add_argument("--warm-server-lr-factor", type=float, default=None,
                    help="FL-15 (06M_B B2): server-LR factor of a warm arm (FedAdam 1 / 0.1; FedAvgM / FedAvg 1 / 0.25); "
                         "screen / smoke only")
    ap.add_argument("--clip-record", type=Path, default=None, help="FA_1024 / DP: the S_CAL S_RECORD.json (FL-16 warm: S_RECORD_WARM[_FROZEN].json)")
    ap.add_argument("--dp-z-record", type=Path, default=DP_Z_RECORD)
    ap.add_argument("--theta0-variants", type=Path, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=64, help="rounds between latest.pt saves")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-shared-gpu", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-rounds", type=int, default=3)
    ap.add_argument("--smoke-interrupt-at", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--eval-prefetch", type=int, default=2)
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--lock-path", type=Path, default=None)
    ap.add_argument("--manifest-hashes", type=Path, default=None)
    ap.add_argument("--header-list", type=Path, default=None)
    ap.add_argument("--transform-state", type=Path, default=None)
    ap.add_argument("--p2-recipe", type=Path, default=None,
                    help="P2-02: a PPSI_V4_P2_RECIPE_1 file (P2_W smoke / CALIBRATION runs only)")
    ap.add_argument("--p2-selection", type=Path, default=None,
                    help="P2-02: the registered contracts/fl/P2_WARM_SELECTION_FINAL.json (P2_W non-smoke MAIN runs)")
    return ap.parse_args(argv)


# ------------------------------------------------------------------------------------------------ arm config (pure)
def load_z(epsilon: float, record_path: Path = DP_Z_RECORD) -> dict:
    """The registered z of eps (the receipt must agree with the amendment-06 value and the accounting setup)."""
    eps = float(epsilon)
    if eps not in DP_REGISTERED_Z:
        raise LaunchRefused(f"--epsilon must be one of {sorted(DP_REGISTERED_Z)} (registered), not {epsilon}")
    rec = json.loads(Path(record_path).read_text(encoding="utf-8"))
    hits = [r for r in rec.get("records", []) if float(r.get("epsilon_target", -1)) == eps]
    if len(hits) != 1:
        raise LaunchRefused(f"{record_path}: no unique record for eps {eps}")
    r = hits[0]
    if float(r["z"]) != DP_REGISTERED_Z[eps] or any(r.get(k) != v for k, v in DP_ACCOUNTING.items()):
        raise LaunchRefused(f"{record_path}: eps {eps} record differs from the registered z / accounting setup")
    return {"epsilon": eps, "z": float(r["z"]), "record": str(record_path),
            "record_sha256": hashlib.sha256(Path(record_path).read_bytes()).hexdigest(), **DP_ACCOUNTING}


def s_record_obj(norm: Mapping, *, run_id: str, seed: int, config_digest: str, plan_digest: str,
                 init_sha256: str, smoke: bool) -> dict:
    obj = {"format": S_FORMAT, "S": float(norm["S"]), "n_norms": int(norm["n_norms"]),
           "rounds_0based": list(norm["rounds_0based"]), "rule": norm["rule"], "run_id": run_id, "seed": int(seed),
           "config_digest": config_digest, "plan_digest": plan_digest, "init_sha256": init_sha256,
           "label": "SMOKE_NOT_EVIDENCE" if smoke else "S_CALIBRATION",
           "disclosure": "non-private; follows the unclipped trajectory (amendment 06 §2 Q3)"}
    obj["self_sha256"] = _canon(obj)
    return obj


def _canon(obj: Mapping) -> str:
    body = {k: v for k, v in obj.items() if k != "self_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _s_verified(obj, fmt: str) -> bool:
    """FL-16b (REV-FL-16 m5): an S record is a JSON object of format `fmt` whose self_sha256 verifies; a record that is
    not an object or cannot be canonicalised (NaN / inf: allow_nan=False) is unverified (refused, not a crash)."""
    if not isinstance(obj, dict) or obj.get("format") != fmt:
        return False
    try:
        return obj.get("self_sha256") == _canon(obj)
    except (TypeError, ValueError):
        return False


def check_s_content(obj: Mapping, path) -> None:
    """FL-16b (REV-FL-16 m5): the S fields of a verified S record (cold or warm) follow the registered S rule
    (fl.dp.median_clip_norm): S a finite real > 0, n_norms an int > 0, the rule a median, and rounds_0based = the
    registered rounds 0..s_cal_rounds-1 for the production record (label S_CALIBRATION) or 0..k-1 with
    1 <= k <= MAX_SMOKE_ROUNDS for a smoke record (label SMOKE_NOT_EVIDENCE); any other label is refused. The self-sha256
    guards integrity only; this refuses a hand-edited, re-signed record whose S content no S calibration can write."""
    s, n, rounds, rule, label = (obj.get(k) for k in ("S", "n_norms", "rounds_0based", "rule", "label"))
    if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0:
        raise LaunchRefused(f"--clip-record {path}: S must be a finite real > 0 (the median client-delta norm)")
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise LaunchRefused(f"--clip-record {path}: n_norms must be an int > 0")
    if not isinstance(rule, str) or not rule.startswith("median"):
        raise LaunchRefused(f"--clip-record {path}: rule {rule!r} is not the registered median S rule")
    if label == "S_CALIBRATION":
        want = list(range(KNOBS["s_cal_rounds"]))
    elif label == "SMOKE_NOT_EVIDENCE" and isinstance(rounds, list) and 1 <= len(rounds) <= MAX_SMOKE_ROUNDS:
        want = list(range(len(rounds)))
    elif label == "SMOKE_NOT_EVIDENCE":
        raise LaunchRefused(f"--clip-record {path}: a smoke record covers 1..{MAX_SMOKE_ROUNDS} rounds")
    else:
        raise LaunchRefused(f"--clip-record {path}: label {label!r} is neither S_CALIBRATION nor SMOKE_NOT_EVIDENCE")
    if not isinstance(rounds, list) or rounds != want or \
            any(isinstance(r, bool) or not isinstance(r, int) for r in rounds):
        raise LaunchRefused(f"--clip-record {path}: rounds_0based != the registered S rounds 0..{len(want) - 1}")


def load_s_record(path: Path, *, smoke: bool) -> dict:
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LaunchRefused(f"--clip-record {path} unreadable ({e})") from e
    if not _s_verified(obj, S_FORMAT):                    # FL-16b: a non-object / NaN record is refused, not a crash
        raise LaunchRefused(f"--clip-record {path} is not a verified {S_FORMAT} object")
    if not smoke and (obj.get("label") != "S_CALIBRATION" or obj.get("seed") != KNOBS["s_cal_seed"]):
        raise LaunchRefused("a non-smoke run needs the production S_CAL record (seed 2026, not a smoke record)")
    check_s_content(obj, path)                             # FL-16b (REV-FL-16 m5)
    return obj


# ------------------------------------------------------------------------------------------------ FL-16 warm S record
def s_record_obj_warm(norm: Mapping, *, spec: Mapping, run_id: str, seed: int, config_digest: str, plan_digest: str,
                      init_sha256: str, smoke: bool) -> dict:
    """FL-16 (06N N3.2): the S record of a warm S calibration (S_CAL_WARM / S_CAL_WARM_FROZEN) — the cold record's
    fields under its own format PPSI_V4_DP_S_RECORD_WARM_1, plus the arm, the warm variant / stage, the S_CAL arm-spec
    sha256, theta_pre's identity and the frozen keys (the norms are over the uploaded parameters only when frozen)."""
    arm = spec["arm"]
    if base_arm(arm) != "S_CAL" or not is_warm_arm(arm):
        raise LaunchRefused(f"a warm S record is written by S_CAL_WARM / S_CAL_WARM_FROZEN only, not {arm}")
    obj = s_record_obj(norm, run_id=run_id, seed=seed, config_digest=config_digest, plan_digest=plan_digest,
                       init_sha256=init_sha256, smoke=smoke)
    obj.pop("self_sha256")
    v = warm_variant(arm)
    obj.update(format=S_FORMAT_WARM, arm=arm, warm_variant=v, stage=WARM_STAGE[v], arm_spec_sha256=spec_digest(spec),
               init_from=dict(spec["init_from"]), frozen_item_tables=spec.get("frozen_item_tables"),
               warm_lr_factor=float(spec.get("warm_lr_factor", 1.0)),
               warm_server_lr_factor=float(spec.get("warm_server_lr_factor", 1.0)),
               scope=("the warm trajectory from theta_pre (" + ("S8: item tables frozen, norms over the uploaded "
                                                               "parameters only" if spec.get("frozen_item_tables")
                                                               else "S7: full model") + ")"),
               authority=FL16_AUTHORITY)
    obj["self_sha256"] = _canon(obj)
    return obj


def warm_s_cal_spec(arm: str, *, peak_lr: float, fused: Optional[bool], init_from: Mapping, tun: Mapping,
                    warm_lr_factor=None, warm_server_lr_factor=None, poisson_local_rule=None) -> dict:
    """FL-16: the S_CAL_WARM[_FROZEN] spec a warm FA_1024 / DP arm's S must come from: its variant's S_CAL arm at seed
    2026 with the SAME tuning point, poisson rule, warm factors and theta_pre as the run."""
    v = warm_variant(arm)
    if v is None or base_arm(arm) not in ("FA_1024", "DP"):
        raise LaunchRefused(f"{arm}: the warm S calibration spec belongs to FA_1024_WARM / DP_WARM[_FROZEN]")
    return arm_spec(S_CAL_WARM_ARM[v], seed=KNOBS["s_cal_seed"], peak_lr=peak_lr, fused=fused, role="MAIN",
                    init_from=init_from, warm_lr_factor=warm_lr_factor, warm_server_lr_factor=warm_server_lr_factor,
                    poisson_local_rule=poisson_local_rule, **tuning_kwargs(tun))


def load_warm_s_record(path: Path, *, arm: str, scal_spec: Mapping, init_identity: Optional[Mapping],
                       smoke: bool) -> dict:
    """FL-16 (06N N3.2): the verified S record of a warm FA_1024 / DP arm. The cold PPSI_V4_DP_S_RECORD_1 (e.g.
    contracts/dp/S_RECORD.json) is refused; the warm record must be self-sha256-verified, the production S calibration
    (label S_CALIBRATION, seed 2026) unless smoke, written by this arm's variant's S_CAL arm (S7 -> S_CAL_WARM, S8 ->
    S_CAL_WARM_FROZEN), from this run's theta_pre (identity and init_sha256 = its state digest), and bound to the
    expected S calibration config: arm-spec sha256 = spec_digest(scal_spec) and config_digest = the fl RunConfig
    digest of scal_spec under the record's run id (warm_s_cal_spec). The plan digest is checked in `run` against the
    seed-2026 Poisson plan of this run's clients (check_warm_s_plan)."""
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LaunchRefused(f"--clip-record {path} unreadable ({e})") from e
    if isinstance(obj, dict) and obj.get("format") == S_FORMAT:
        raise LaunchRefused(f"--clip-record {path} is the cold-start {S_FORMAT} record (S calibrated on the theta0 "
                            f"trajectory): {arm} needs the warm S record {S_FORMAT_WARM} of "
                            f"{S_CAL_WARM_ARM.get(warm_variant(arm) or '', 'S_CAL_WARM')} (06N N3.2)")
    if not _s_verified(obj, S_FORMAT_WARM):
        raise LaunchRefused(f"--clip-record {path} is not a verified {S_FORMAT_WARM} object")
    if not smoke and (obj.get("label") != "S_CALIBRATION" or obj.get("seed") != KNOBS["s_cal_seed"]):
        raise LaunchRefused("a non-smoke run needs the production S_CAL_WARM record (seed 2026, not a smoke record)")
    check_s_content(obj, path)                             # FL-16b (REV-FL-16 m5)
    want_arm = S_CAL_WARM_ARM.get(warm_variant(arm) or "")
    if want_arm is None or obj.get("arm") != want_arm or scal_spec.get("arm") != want_arm:
        raise LaunchRefused(f"--clip-record {path}: written by {obj.get('arm')!r}; {arm} takes the S of {want_arm} "
                            "(S7 -> S_CAL_WARM, S8 -> S_CAL_WARM_FROZEN)")
    check_s_record_init(obj, init_identity, path)
    check_s_record_config(obj, scal_spec, path)
    return obj


def check_s_record_init(obj: Mapping, init_identity: Optional[Mapping], path) -> None:
    """FL-16: a warm S record's theta_pre = this run's (the identity and init_sha256 = theta_pre's state digest)."""
    if init_identity is None or obj.get("init_from") != dict(init_identity) or \
            obj.get("init_sha256") != init_identity.get("state_digest"):
        raise LaunchRefused(f"--clip-record {path}: its theta_pre identity / init_sha256 is not this run's --init-from")


def check_s_record_config(obj: Mapping, scal_spec: Mapping, path) -> None:
    """FL-16: a warm S record is bound to the S calibration of THIS run's configuration: frozen keys, the S_CAL arm-spec
    sha256 and the fl RunConfig digest (under the record's run id) of scal_spec (warm_s_cal_spec)."""
    if obj.get("frozen_item_tables") != scal_spec.get("frozen_item_tables"):
        raise LaunchRefused(f"--clip-record {path}: frozen tables {obj.get('frozen_item_tables')} != the run's "
                            f"{scal_spec.get('frozen_item_tables')}")
    want_spec = spec_digest(scal_spec)
    want_cfg = fl_run_config(scal_spec, str(obj.get("run_id")), 64).digest()   # ckpt_every: not digested
    if obj.get("arm_spec_sha256") != want_spec or obj.get("config_digest") != want_cfg:
        raise LaunchRefused(f"--clip-record {path}: S calibration spec / config digest "
                            f"{str(obj.get('arm_spec_sha256'))[:12]}… / {str(obj.get('config_digest'))[:12]}… != this "
                            f"run's warm S calibration config {want_spec[:12]}… / {want_cfg[:12]}… (tuning, warm "
                            "factors, theta_pre, poisson rule)")


def check_warm_s_plan(srec: Mapping, keys, manifest_hash: str, spec: Mapping) -> str:
    """FL-16: the warm S record's plan digest = the seed-2026 Poisson plan (the S calibration's) of THIS run's clients
    (same keys, manifest hash, group size and sampling as the run's plan). Returns the digest."""
    from fl.participation import ParticipationPlan
    d = ParticipationPlan.build(list(keys), seed=KNOBS["s_cal_seed"], manifest_hash=manifest_hash,
                                group_size=spec["plan"]["group_size"], sampling=spec["plan"]["sampling"]).digest()
    if srec.get("plan_digest") != d:
        raise LaunchRefused(f"--clip-record: plan digest {str(srec.get('plan_digest'))[:12]}… != the seed-2026 "
                            f"S-calibration plan of this run's clients {d[:12]}…")
    return d


def arm_spec(arm: str, *, seed: int, peak_lr: float, clip_norm: Optional[float] = None,
             z: Optional[float] = None, fused: Optional[bool] = None, lr_mult: float = 1.0,
             role: str = "MAIN", server_opt: Optional[str] = None, server_lr: Optional[float] = None,
             server_tau: Optional[float] = None, lr_shape: Optional[str] = None, client_opt: Optional[str] = None,
             client_lr: Optional[float] = None, local_passes: Optional[int] = None, local_batch: Optional[int] = None,
             init_from: Optional[Mapping] = None, warm_lr_factor: Optional[float] = None,
             poisson_local_rule: Optional[Mapping] = None, warm_server_lr_factor: Optional[float] = None,
             p2_recipe: Optional[Mapping] = None) -> dict:
    """The registered configuration of one arm (plan + fl.runtime.RunConfig kwargs), as plain JSON-able values.
    FL-05: `lr_mult` (0.5 / 1 / 2) multiplies the local shared-model LR only (lr_peak_local and the solver LR; the PF
    personal LR stays the SMALL peak); `role` CALIBRATION = the FA local-LR screen on CALIB (1,707 rounds).
    FL-13: local_passes / local_batch (06J §3). FL-12: FA_WARM / FA_WARM_FROZEN need `init_from` (the verified theta_pre
    identity, check_init_identity) and take `warm_lr_factor` (06K: S7 {1, 1/4}, S8 {1, 4}); FA_WARM_FROZEN records the
    frozen item tables (spec frozen_item_tables). FIX-DELTA-02 M1: `poisson_local_rule` (poisson_tuning) records the DL 219
    override of a selection's passes / batch for FA_1024 / DP / S_CAL (key absent otherwise).
    FL-15 (06M_B B2): `warm_server_lr_factor` (warm arms; FedAdam {1, 0.1}, FedAvgM / FedAvg {1, 0.25}) sets the server
    step to eta* / 10 or eta* / 4 (warm_server_opt; key warm_server_lr_factor only when != 1, so every factor-1 spec
    and digest is unchanged).
    P2-02: a P2 arm (P2_ARMS) is built by p2_arm_spec from `p2_recipe` (+ seed, the SMALL peak for the WC5 personal
    LR, role, init_from); every Phase-1 tuning / warm / DP keyword is refused for it, and `p2_recipe` for every other
    arm (the Phase-1 branch below is unchanged)."""
    if is_p2_arm(arm) or p2_recipe is not None:            # P2-02 (the Phase-1 path never reaches this branch)
        if not is_p2_arm(arm):
            raise LaunchRefused(f"p2_recipe is a {P2_ARMS} option, not {arm}'s")
        p1 = {"clip_norm": clip_norm, "z": z, "server_opt": server_opt, "server_lr": server_lr, "server_tau": server_tau,
              "lr_shape": lr_shape, "client_opt": client_opt, "client_lr": client_lr, "local_passes": local_passes,
              "local_batch": local_batch, "warm_lr_factor": warm_lr_factor, "poisson_local_rule": poisson_local_rule,
              "warm_server_lr_factor": warm_server_lr_factor}
        given = sorted(k for k, v in p1.items() if v is not None) + ([] if float(lr_mult) == 1.0 else ["lr_mult"])
        if given:
            raise LaunchRefused(f"{arm}: Phase-1 options {given} are refused for a P2 arm (the P2 recipe fixes them)")
        return p2_arm_spec(arm, recipe=p2_recipe, seed=seed, peak_lr=peak_lr, role=role, init_from=init_from)
    if arm not in ALL_ARMS:
        raise LaunchRefused(f"unknown arm {arm}")
    m = lr_mult_value(lr_mult)
    tun = tuning_config(optimizer=server_opt, server_lr=server_lr, server_tau=server_tau, lr_mult=m,
                        lr_shape=lr_shape, client_opt=client_opt, client_lr=client_lr, local_passes=local_passes,
                        local_batch=local_batch)
    wf = warm_factor_value(arm, warm_lr_factor)
    wsf = warm_server_factor_value(arm, tun["optimizer"], warm_server_lr_factor)          # FL-15
    if is_warm_arm(arm) != (init_from is not None):         # FL-16: + the warm method set (WARM_LINE)
        raise LaunchRefused(f"{arm}: " + ("a warm-start arm needs the verified init_from (--init-from)"
                                          if is_warm_arm(arm) else "init_from / --init-from is a FA_WARM / "
                                                                   "FA_WARM_FROZEN option (and of the FL-16 warm "
                                                                   "method set *_WARM / *_WARM_FROZEN)"))
    if init_from is not None:
        check_init_identity(init_from)
    if role not in ROLES:
        raise LaunchRefused(f"--role must be one of {ROLES}, not {role}")
    if role == "CALIBRATION":
        if arm not in CALIB["arms"]:
            raise LaunchRefused(f"--role CALIBRATION is the FA local-LR screen only (amendment 06B §1), not {arm}")
        if int(seed) != CALIB["seed"]:
            raise LaunchRefused("the CALIBRATION LR screen runs at seed 2026 only (amendment 06B §1)")
    K = KNOBS
    base = float(peak_lr)
    local = base if m == 1.0 else m * base                     # m = 1: the pre-FL-05 value, bit for bit
    solver = {"lr": local, "batch_size": K["batch"], "passes": K["passes"], "clip": K["clip"],
              "weight_decay": K["weight_decay"], "fused": fused}
    cfg = {"method": METHOD_CODE[arm], "seed": int(seed), "lr_peak_local": local, "solver": solver,
           "n_shards": K["n_shards"], "max_attempts": K["max_attempts"], "endpoint_efe": K["endpoint_efe"],
           "exposure_point": EXPOSURE_POINT, "mu": 0.0, "pf": None, "dp": None}
    ca = base_arm(arm)                                         # FL-16: a warm-line arm runs its cold arm's semantics
    if is_sweep_arm(arm):
        plan = {"group_size": K["group"], "sampling": "sweep"}
        rounds = CALIB["rounds"] if role == "CALIBRATION" else K["rounds"]
        cfg.update(dropout_p=K["dropout_p"], endpoint_rounds=rounds)
        if ca == "FP":
            cfg["mu"] = K["mu"]
        if ca == "PF":
            cfg["pf"] = {"lr": base, "lam": K["pf_lam"], "clip": K["pf_clip"]}     # personal LR: never multiplied
        max_rounds = rounds
    else:                                                      # FL-07 / amendment 06C: Poisson, fixed denominator m
        plan = {"group_size": K["m"], "sampling": "poisson"}
        cfg.update(dropout_p=0.0, endpoint_rounds=K["rounds_1024"])
        if ca == "S_CAL":
            if int(seed) != K["s_cal_seed"]:
                raise LaunchRefused("S_CAL runs at seed 2026 only (amendment 06 §2 Q3)")
            cfg["dp"] = {"clip_norm": None, "noise_multiplier": 0.0, "denominator": K["m"]}
            max_rounds = K["s_cal_rounds"]
        else:
            if clip_norm is None:
                raise LaunchRefused(f"{arm} needs --clip-record (the S_CAL S_RECORD.json)")
            if ca == "DP" and not z:
                raise LaunchRefused("DP needs --epsilon (a registered z)")
            cfg["dp"] = {"clip_norm": float(clip_norm), "noise_multiplier": float(z) if ca == "DP" else 0.0,
                         "denominator": K["m"]}
            max_rounds = K["rounds_1024"]
    spec = {"arm": arm, "plan": plan, "cfg": cfg, "max_rounds": max_rounds, "knobs": dict(KNOBS),
            "lr_mult": m, "lr_peak_base": base, "role": role,
            "upload_codec": dict(Q8_CODEC) if ca == "FA_Q8" else None}
    _apply_tuning(spec, tun)                                   # FL-10: keys only when non-default (pre-FL-10 bytes)
    if poisson_local_rule is not None:                         # FIX-DELTA-02 M1 (DL 219): only when it overrides
        check_poisson_rule(arm, poisson_local_rule, tun)
        spec["poisson_local_rule"] = dict(poisson_local_rule)
    if is_warm_arm(arm):                                       # FL-12 (06K §2-§3): keys exist for warm arms only
        spec["init_from"] = dict(init_from)
        if wf != 1.0:
            cfg["lr_peak_local"] = wf * cfg["lr_peak_local"]
            cfg["solver"]["lr"] = cfg["lr_peak_local"]
            spec["warm_lr_factor"] = wf
        if wsf != 1.0:                                         # FL-15 (06M_B B2): eta* / 10 or eta* / 4
            cfg["server_opt"] = warm_server_opt(tun, wsf)
            spec["warm_server_lr_factor"] = wsf
        if is_frozen_arm(arm):                                 # FL-16: every S8-bound arm (06K §3)
            spec["frozen_item_tables"] = list(FROZEN_ITEM_KEYS)
    return spec


def warm_factor_value(arm: str, f) -> float:
    """FL-12: the registered warm-start LR factor of the arm (06K: S7 {1, 1/4}, S8 {1, 4}); None -> 1."""
    if f is None:
        return 1.0
    if not is_warm_arm(arm):
        raise LaunchRefused("--warm-lr-factor is a FA_WARM / FA_WARM_FROZEN option")
    grid = WARM_FACTORS[warm_variant(arm)]                 # FL-16: a warm-line arm takes its variant's grid
    v = _num(f, "--warm-lr-factor")
    if v not in grid:
        raise LaunchRefused(f"--warm-lr-factor for {arm} must be one of {grid} (06K), not {f}")
    return v


def warm_server_factor_value(arm: str, optimizer: str, f) -> float:
    """FL-15 (06M_B B2): the registered warm-start server-LR factor of the tuning point's server optimiser (FedAdam
    {1, 0.1} = {eta*, eta*/10}; FedAvgM / FedAvg {1, 0.25} = {eta*, eta*/4}); None -> 1."""
    if f is None:
        return 1.0
    if not is_warm_arm(arm):
        raise LaunchRefused("--warm-server-lr-factor is a FA_WARM / FA_WARM_FROZEN option")
    v = _num(f, "--warm-server-lr-factor")
    grid = WARM_SERVER_FACTORS.get(optimizer)
    if grid is None or v not in grid:
        raise LaunchRefused(f"--warm-server-lr-factor for server optimiser {optimizer} must be one of {grid} "
                            f"(06M_B B2), not {f}")
    return v


def warm_server_opt(t: Mapping, wsf: float) -> dict:
    """FL-15 (06M_B B2): the cfg server_opt of a warm run at server-LR factor wsf != 1 (the tuning point t keeps eta*):
    FedAdam eta* / 10 (beta1 / beta2 / tau / v_0 of t unchanged); FedAvgM eta* / 4 (beta 0.9); FedAvg (server LR 1)
    at LR 1 / 4 = FedAvgM with beta 0 at LR 0.25, i.e. theta + 0.25 (aggregate - theta) (fl.server FedAvgM: m_r =
    Delta_r at beta 0; disclosed: the momentum buffer is carried in the checkpoint, always equal to the last Delta)."""
    div = WARM_SERVER_DIVISORS[float(wsf)]
    if t["optimizer"] == "fedadam":
        return {"name": "fedadam", "lr": float(t["server_lr"]) / div, **FEDADAM_HYPER, "tau": t["server_tau"]}
    if t["optimizer"] == "fedavgm":
        return {"name": "fedavgm", "lr": FEDAVGM_LR / div, **FEDAVGM_HYPER}
    return {"name": "fedavgm", "lr": 1.0 / div, "beta1": 0.0}


def warm_grid(arm: str, optimizer: str) -> list:
    """FL-15 (06M_B B2): the registered 2-D warm screen cells (local factor, server-LR factor): S7 {m*, m*/4} x
    {eta*, eta*/10 (FedAdam) | eta*/4}; S8 {m*, 4 m*} x the same server axis. 4 cells, (1, 1) = the incumbent first."""
    return [(float(wf), float(sf)) for wf in WARM_FACTORS[arm] for sf in WARM_SERVER_FACTORS[optimizer]]


def warm_cell_label(wf: float, sf: float) -> str:
    """'f=1|sf=1', 'f=0.25|sf=0.1', 'f=4|sf=0.25' (local factor | server-LR factor)."""
    return f"f={num_tag(wf)}|sf={num_tag(sf)}"


INIT_IDENTITY_KEYS = ("kind", "run_id", "run_label", "train_manifest", "fl_init_kind", "checkpoint_sha256",
                      "checkpoint_bytes", "state_digest", "model_variant", "family", "seed")


def _main_trained(manifests=(), ids=()) -> Optional[str]:
    """The MAIN-users marker: a train manifest id that contains MAIN users (MAIN / CALIB / FULL_TRAIN), or a run id /
    label of a central run on MAIN users (C_BIG / C_SMALL / C_FULL / C_MATCHED as its prefix), else None."""
    for t in manifests:
        if str(t or "") in MAIN_TRAIN_MANIFESTS:
            return str(t)
    for t in ids:
        t = str(t or "").upper()
        for mk in MAIN_RUN_MARKERS:
            if t == mk or t.startswith(mk + "_"):
                return mk
    return None


def check_init_identity(ident: Mapping) -> None:
    """FL-12 (06K §1-§2): a theta_pre identity (spec init_from) is the PRE run's endpoint_6.0 on the non-MAIN cohort."""
    if not isinstance(ident, Mapping) or sorted(ident) != sorted(INIT_IDENTITY_KEYS):
        raise LaunchRefused(f"init_from must carry exactly {INIT_IDENTITY_KEYS}")
    mk = _main_trained([ident["train_manifest"]], [ident["run_id"], ident["run_label"]])
    if mk is not None:
        raise LaunchRefused(f"init_from was trained on MAIN users' data ({mk}): never warm-start from C_BIG / C_SMALL / "
                            "C_FULL (06K §2)")
    want = {"kind": PRETRAIN["run_label"], "run_label": PRETRAIN["run_label"], "fl_init_kind": PRETRAIN["fl_init_kind"],
            "train_manifest": PRETRAIN["train_manifest"], "family": PRETRAIN["family"]}
    bad = {k: ident[k] for k, v in want.items() if ident[k] != v}
    if bad or not str(ident["run_id"]).startswith(PRETRAIN["run_id_prefix"]):
        raise LaunchRefused(f"init_from is not the PRE run's {PRETRAIN['fl_init_kind']} (PRETRAIN_FOR_FL on "
                            f"{PRETRAIN['train_manifest']}): {bad or ident['run_id']}")
    if not (_is_sha(ident["checkpoint_sha256"]) and _is_sha(ident["state_digest"])):
        raise LaunchRefused("init_from checkpoint_sha256 / state_digest must be sha256 hex digests")


def pretrain_identity(rec: Mapping, meta: Mapping, *, ckpt_sha256: str, ckpt_bytes: int) -> dict:
    """FL-12: the theta_pre identity from the PRETRAIN_FOR_FL_INIT.json record and the checkpoint's own meta, with
    every refusal of 06K (checkpoint = the PRE run's endpoint_6.0; run label PRETRAIN_FOR_FL; train manifest
    PRETRAIN_TRAIN_5_100_NONMAIN; nothing trained on MAIN users; the file sha256 = the record's)."""
    fl = rec.get("fl_init") or {}
    reg = meta.get("manifest_registry") or {}
    mk = _main_trained([rec.get("train_manifest"), reg.get("train_manifest")],
                       [rec.get("run_id"), rec.get("run_label"), meta.get("run_id"), meta.get("run_label")])
    if mk is not None:
        raise LaunchRefused(f"--init-from: the checkpoint's training data include MAIN users ({mk}); never warm-start "
                            "from C_BIG / C_SMALL / C_FULL (06K §2)")
    if fl.get("sha256") != ckpt_sha256 or int(fl.get("bytes", -1)) != int(ckpt_bytes):
        raise LaunchRefused(f"--init-from: checkpoint sha256 / bytes {ckpt_sha256[:12]}… / {ckpt_bytes} != the init "
                            f"record's {str(fl.get('sha256'))[:12]}… / {fl.get('bytes')}")
    if rec.get("run_label") != PRETRAIN["run_label"] or meta.get("run_label") != PRETRAIN["run_label"]:
        raise LaunchRefused(f"--init-from: run label {rec.get('run_label')!r} / {meta.get('run_label')!r} is not "
                            f"{PRETRAIN['run_label']}")
    if fl.get("kind") != PRETRAIN["fl_init_kind"] or meta.get("_kind") != PRETRAIN["fl_init_kind"]:
        raise LaunchRefused(f"--init-from: checkpoint kind {fl.get('kind')!r} / {meta.get('_kind')!r} is not the "
                            f"registered {PRETRAIN['fl_init_kind']} (no checkpoint selection, 06K §1)")
    if meta.get("run_id") != rec.get("run_id") or not str(rec.get("run_id", "")).startswith(PRETRAIN["run_id_prefix"]):
        raise LaunchRefused(f"--init-from: run id {meta.get('run_id')!r} / {rec.get('run_id')!r} is not the PRE run's")
    if rec.get("train_manifest") != PRETRAIN["train_manifest"] or reg.get("train_manifest") != PRETRAIN["train_manifest"]:
        raise LaunchRefused(f"--init-from: train manifest {rec.get('train_manifest')!r} / {reg.get('train_manifest')!r} "
                            f"is not {PRETRAIN['train_manifest']}")
    if str(meta.get("family", "")).upper() != PRETRAIN["family"]:
        raise LaunchRefused(f"--init-from: family {meta.get('family')!r} is not {PRETRAIN['family']}")
    ident = {"kind": PRETRAIN["run_label"], "run_id": rec["run_id"], "run_label": PRETRAIN["run_label"],
             "train_manifest": PRETRAIN["train_manifest"], "fl_init_kind": PRETRAIN["fl_init_kind"],
             "checkpoint_sha256": ckpt_sha256, "checkpoint_bytes": int(ckpt_bytes),
             "state_digest": meta.get("state_digest"), "model_variant": meta.get("model_variant"),
             "family": PRETRAIN["family"], "seed": meta.get("seed")}
    check_init_identity(ident)
    return ident


def load_pretrain_init(init_json: Path, *, model_variant: str) -> tuple:
    """FL-12 (06K §2): (theta_pre state dict, identity, launch record) from PRETRAIN_FOR_FL_INIT.json. The checkpoint
    path is taken relative to the record's directory; its bytes are hashed BEFORE torch.load and must equal the record;
    the checkpoint's own kind / meta must be the PRE run's endpoint_6.0 of the model variant."""
    import torch
    try:
        rec = json.loads(Path(init_json).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LaunchRefused(f"--init-from {init_json} unreadable ({e})") from e
    rel = (rec.get("fl_init") or {}).get("path")
    if not rel:
        raise LaunchRefused(f"--init-from {init_json}: no fl_init.path")
    ck = (Path(init_json).parent / rel).resolve()
    if not ck.is_file():
        raise LaunchRefused(f"--init-from: checkpoint {ck} is absent")
    blob = ck.read_bytes()
    sha = hashlib.sha256(blob).hexdigest()
    import io
    try:
        obj = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        raise LaunchRefused(f"--init-from: checkpoint {ck} unreadable ({type(e).__name__}: {e})") from e
    if not isinstance(obj, dict) or not isinstance(obj.get("meta"), dict) or not isinstance(obj.get("model"), dict):
        raise LaunchRefused(f"--init-from: {ck} is not a v4train weights checkpoint (kind / meta / model)")
    ident = pretrain_identity(rec, dict(obj["meta"], _kind=obj.get("kind")), ckpt_sha256=sha, ckpt_bytes=len(blob))
    if ident["model_variant"] != model_variant:
        raise LaunchRefused(f"--init-from: model variant {ident['model_variant']!r} != the SMALL recipe's "
                            f"{model_variant!r}")
    launch = {"init_record": str(init_json), "init_record_sha256": _file_sha(init_json), "checkpoint": str(ck),
              "identity": ident}
    return obj["model"], ident, launch


def apply_init_state(adapter, state: Mapping, want_digest: str) -> str:
    """FL-12: load theta_pre into the (server / harness) model, strictly, and prove it: the broadcast state must hash
    to the checkpoint's own state_digest, which becomes the server's registered init_sha256."""
    import torch
    from fl.numerics import state_digest
    with torch.no_grad():
        missing, unexpected = adapter.module.load_state_dict(dict(state), strict=True)
    got = state_digest(adapter.broadcast_state())
    if got != want_digest:
        raise LaunchRefused(f"--init-from: loaded state digest {got[:12]}… != theta_pre's {str(want_digest)[:12]}…")
    return got


def fl_run_config(spec: Mapping, run_id: str, ckpt_every: int):
    from fl.runtime import RunConfig
    return RunConfig.from_dict(dict(spec["cfg"], run_id=run_id, ckpt_every=int(ckpt_every)))


def digest_body(spec: Mapping) -> dict:
    """The spec as hashed: the FL-05 keys are dropped at their defaults (lr_mult 1 also drops the then-redundant
    lr_peak_base), so an arm with every FL-05 option off has exactly the pre-FL-05 digest."""
    d = dict(spec)
    for k, default in FL05_DEFAULTS.items():
        if k in d and d[k] == default:
            d.pop(k)
    if "lr_mult" not in d:
        d.pop("lr_peak_base", None)
    return d


def spec_digest(spec: Mapping) -> str:
    return hashlib.sha256(json.dumps(digest_body(spec), sort_keys=True, default=str).encode()).hexdigest()


def assert_arm_allowed(arm: str, family: str, lock_path: Path = ARCH_LOCK) -> dict:
    """ARCHITECTURE_LOCK: the winner family only (v4train.selection.assert_downstream_allowed) and a registered method."""
    from v4train.selection import LosingFamilyError, assert_downstream_allowed
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    decision = {"status": lock.get("decision_status"), "winner_family": lock.get("winner_family"),
                "forbidden_downstream_families": lock.get("forbidden_downstream_families", [])}
    code = METHOD_CODE.get(arm, arm)
    try:
        assert_downstream_allowed(decision, family, code)
    except LosingFamilyError as e:
        raise LaunchRefused(str(e)) from e
    if code not in (lock.get("allowed_subsequent_method_codes") or []):
        raise LaunchRefused(f"method {code} is not an allowed subsequent method of the ARCHITECTURE_LOCK")
    return {"path": str(lock_path), "sha256": hashlib.sha256(Path(lock_path).read_bytes()).hexdigest(),
            "winner_family": lock.get("winner_family"), "method_code": code}


# ------------------------------------------------------------------------------------------------ rows
def fl_rows(rows, *, arm: str, run_id: str, efe_raw: float, mark: float, digest: str, paradigm: str = "FEDERATED",
            lifecycle: str = "FRESH_ADAMW_PER_VISIT", role: str = "MAIN", lr_mult: float = 1.0,
            tuning_labels: Optional[Mapping] = None) -> list:
    """The harness rows (central RESULT_ROWS format) relabelled for a downstream arm: method, paradigm, status.
    FL-05: every row carries fl_lr_mult; a CALIBRATION (LR-screen) row is labelled role / manifest_role CALIBRATION,
    run_label / row_status / comparison_status FL_LR_SCREEN (never a downstream retention row)."""
    out = []
    for r in rows:
        r = dict(r)
        r.update(method=arm, method_code=METHOD_CODE.get(arm, arm), training_paradigm=paradigm,
                 optimizer_lifecycle=lifecycle, row_status="DOWNSTREAM_ARM", comparison_status="DOWNSTREAM_RETENTION",
                 c_matched_eligible=False, downstream_seed_allowed=False, efe=float(efe_raw), schedule_mark=mark,
                 arm_spec_sha256=digest, fl_run_id=run_id, fl_lr_mult=float(lr_mult))
        if tuning_labels:                                  # FL-10: non-default dimensions only (FedAvg rows unchanged)
            r.update(tuning_labels)
        if role == "CALIBRATION":                          # a smoke harness keeps its SMOKE_NOT_EVIDENCE run label
            label = r.get("run_label") if r.get("run_label") == "SMOKE_NOT_EVIDENCE" else CALIB["run_label"]
            r.update(role="CALIBRATION", manifest_role="CALIBRATION", run_role="CALIBRATION",
                     population_scope="CALIBRATION", run_label=label, row_status=CALIB["run_label"],
                     comparison_status=CALIB["run_label"])
        out.append(r)
    return out


def bytes_record(manifest, counters, ledger, codec: Optional[Mapping] = None, frozen_keys=()) -> dict:
    from fl.comm import visit_bytes
    if codec is not None:                                  # FA_Q8: int8 uploads (fl.quant), dense FP32 download
        from fl.quant import q8_visit_bytes
        vb = q8_visit_bytes(manifest)
        if frozen_keys:                                    # FL-16 (FA_Q8_WARM_FROZEN): Q8 on the uploaded keys only;
            fz = visit_bytes(manifest, frozen_keys=frozen_keys)   # the frozen fields of fl.comm, download unchanged
            vb.update({k: fz[k] for k in ("frozen_keys", "frozen_bytes_once", "frozen_numel", "uploaded_numel",
                                          "frozen_disclosure")})
    else:
        vb = visit_bytes(manifest, frozen_keys=frozen_keys) if frozen_keys else visit_bytes(manifest)
    import numpy as np
    visits = np.asarray(counters.visit_counts, dtype=np.int64)
    per = visits * int(vb["send_plus_receive_bytes"])
    snap = ledger.snapshot()
    out = {"per_visit": vb, "per_client": {"mean": float(per.mean()), "median": float(np.median(per)),
                                           "max": int(per.max()), "min": int(per.min()),
                                           "rule": "visits x (down + up) of that client"},
           "fleet": {"messages": snap[0], "bytes_down": snap[1], "bytes_up": snap[2], "retry_bytes_down": snap[3],
                     "total": snap[1] + snap[2] + snap[3]},
           "disclosure": "dropped clients (10% drop-out) are not charged a download (the ledger counts survivors)"}
    if frozen_keys:                                        # FL-12b (06K §3): the frozen tables, ONCE per client
        once = int(vb["frozen_bytes_once"])
        per_f = per + (visits > 0).astype(np.int64) * once
        n_seen = int((visits > 0).sum())
        out["per_client_incl_frozen_once"] = {"mean": float(per_f.mean()), "median": float(np.median(per_f)),
                                              "max": int(per_f.max()), "min": int(per_f.min()),
                                              "rule": "visits x (down + up) + the frozen tables once if visited"}
        out["fleet"].update(frozen_download_once_clients=n_seen, frozen_download_once_bytes=n_seen * once,
                            total_incl_frozen_once=out["fleet"]["total"] + n_seen * once)
        out["frozen_disclosure"] = ("frozen item tables: downloaded once per client at its first (surviving) visit, "
                                    "never uploaded; the ledger / per-visit bytes count the uploaded parameters only")
        out["frozen_bootstrap_note"] = (                  # FIX-DELTA-02 m6 (REV-DELTA-02): reporting, no numerics
            "per_visit.bootstrap_bytes_once also contains the frozen tables (they are FIXED buffers after the freeze), "
            "so it repeats per_visit.frozen_bytes_once for those keys; no total adds both (per_client_incl_frozen_once "
            "and fleet.total_incl_frozen_once use frozen_bytes_once only). A client dropped at its first visit is not "
            "charged the table download (the ledger counts survivors; 06A §3 rule)")
    return out


def warm_row_labels(spec: Mapping) -> dict:
    """FL-12: row fields of a warm-start arm (absent for every other arm)."""
    ident = spec.get("init_from")
    if not ident:
        return {}
    out = {"fl_init": ident["kind"], "fl_init_run_id": ident["run_id"],
           "fl_init_checkpoint_sha256": ident["checkpoint_sha256"], "fl_init_state_digest": ident["state_digest"]}
    if "warm_lr_factor" in spec:
        out["fl_warm_lr_factor"] = float(spec["warm_lr_factor"])
    if "warm_server_lr_factor" in spec:                    # FL-15 (06M_B B2): factor + the effective server LR
        out["fl_warm_server_lr_factor"] = float(spec["warm_server_lr_factor"])
        out["fl_warm_server_lr"] = float(spec["cfg"]["server_opt"]["lr"])
    if spec.get("frozen_item_tables"):
        out["fl_frozen_item_tables"] = list(spec["frozen_item_tables"])
    return out


def check_warm_spec(spec: Mapping) -> None:
    """FL-12: the warm keys of a spec are exactly what arm_spec writes (refused otherwise)."""
    arm = spec["arm"]
    if not is_warm_arm(arm):                               # FL-16: + the warm method set
        if any(k in spec for k in ("init_from", "warm_lr_factor", "frozen_item_tables", "warm_server_lr_factor")):
            raise LaunchRefused(f"{arm}: warm-start keys in a cold-start arm spec")
        return
    check_init_identity(spec.get("init_from"))
    warm_factor_value(arm, spec.get("warm_lr_factor"))
    warm_server_factor_value(arm, spec.get("server_opt") or "fedavg", spec.get("warm_server_lr_factor"))   # FL-15
    want = list(FROZEN_ITEM_KEYS) if is_frozen_arm(arm) else None
    if spec.get("frozen_item_tables") != want:
        raise LaunchRefused(f"{arm}: frozen_item_tables {spec.get('frozen_item_tables')} != {want}")


# ------------------------------------------------------------------------------------------------ the engine
def write_json(path: Path, obj) -> None:
    from v4train.checkpoint import write_json_atomic
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(Path(path), _json_safe(obj))


PF_STORE_FILE, PF_STORE_BEST_FILE = "PF_STORE.pt", "PF_STORE_best.pt"
CLIP_LOG_FILE = "CLIP_LOG.jsonl"                           # FL-14 (06L A4 / 06L_E E1)
CLIP_LOG_AUTHORITY = ("PREG1_AMENDMENT_06L A4 / 06L_E E1: z = 0 runs only (clipped fraction per round, no per-client "
                      "norms); z > 0 runs record nothing about clipping")


def clip_log_applies(spec: Mapping) -> bool:
    """06L_E E1: CLIP_LOG is written only by a clipped z = 0 run (FA_1024)."""
    dp = spec.get("cfg", {}).get("dp")
    return bool(dp) and dp.get("clip_norm") is not None and float(dp.get("noise_multiplier") or 0.0) == 0.0


def clip_round_record(out_r: Mapping) -> Optional[dict]:
    """06L A4 / 06L_E E1: the clipping record of one round from fl.dp's round info, or None. Only a clipped z = 0 round
    with a released count: round, n_clipped, n_deltas (clients with n_consumed > 0), clipped_fraction (None for 0
    deltas). A z > 0 round gives None whatever fl.dp returns. Never per-client norms."""
    info = out_r.get("dp")
    if not info or info.get("clip_norm") is None or "n_clipped" not in info:
        return None
    if float(info.get("noise_multiplier", 1.0)) != 0.0:   # 06L_E E1: z > 0 records nothing about clipping
        return None
    n = int(info["n_survived"]) - int(info["n_noop"])
    k = int(info["n_clipped"])
    return {"round": int(out_r["round"]), "n_clipped": k, "n_deltas": n, "clipped_fraction": k / n if n else None}


def append_clip_log(run_dir: Path, rec: Mapping) -> None:
    with open(Path(run_dir) / CLIP_LOG_FILE, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(dict(rec), sort_keys=True) + "\n")


def read_clip_log(run_dir: Path) -> list:
    p = Path(run_dir) / CLIP_LOG_FILE
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def trim_clip_log(run_dir: Path, cursor: int) -> None:
    """On resume: drop the lines of rounds >= the checkpoint cursor (those rounds are re-run and logged again)."""
    p = Path(run_dir) / CLIP_LOG_FILE
    if p.is_file():
        keep = [r for r in read_clip_log(run_dir) if int(r["round"]) < int(cursor)]
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in keep), encoding="utf-8", newline="\n")


def clip_log_summary(run_dir: Path, expected_rounds: Optional[Sequence[int]] = None) -> Optional[dict]:
    """The RUN_SUMMARY record of CLIP_LOG.jsonl (None when nothing is logged or expected). REV-FL-14 m1:
    `expected_rounds` = the rounds with survivors in the checkpointed participation log; a line lost between the
    in-step checkpoint and the append (a crash) shows as complete False + missing_rounds (flagged, not refused: the
    log is descriptive)."""
    recs = read_clip_log(run_dir)
    if not recs and not expected_rounds:
        return None
    rounds = [int(r["round"]) for r in recs]
    if rounds != sorted(set(rounds)):
        raise RuntimeError(f"{CLIP_LOG_FILE}: rounds are not strictly increasing")
    out = {"file": CLIP_LOG_FILE, "rounds_logged": len(recs), "fields": sorted(recs[0]) if recs else [],
           "sha256": _file_sha(Path(run_dir) / CLIP_LOG_FILE) if recs else None, "authority": CLIP_LOG_AUTHORITY,
           "complete": None}
    if expected_rounds is not None:
        want = [int(r) for r in expected_rounds]
        missing, extra = sorted(set(want) - set(rounds)), sorted(set(rounds) - set(want))
        out.update(complete=rounds == want, rounds_expected=len(want), n_missing=len(missing),
                   missing_rounds=missing[:50], unexpected_rounds=extra[:50])
    return out


def clip_expected_rounds(spec: Mapping, run) -> Optional[list]:
    """The rounds that must have a CLIP_LOG line: survivors > 0 in the (checkpointed) participation log."""
    if not clip_log_applies(spec):
        return None
    return [int(p["round"]) for p in run.participation_log if int(p["n_survived"]) > 0]


TRAIN_LOSS_FILE = "TRAIN_LOSS.jsonl"                       # FL-15 R5
TRAIN_LOSS_AUTHORITY = ("FL-15 R5 (REAL_WORLD_FL_PRACTICE_AUDIT_20260930 R5): per-round client training loss from "
                        "fl VisitRecord.loss_sum (sum of the batch-mean CE over a visit's steps); descriptive monitoring "
                        "only, numerics unchanged; z > 0 (DP) runs record nothing (an un-noised data-dependent release, "
                        "the 06L_E E1 reading)")


def train_loss_applies(spec: Mapping) -> bool:
    """FL-15 R5: every run except a z > 0 DP run writes TRAIN_LOSS.jsonl."""
    dp = spec.get("cfg", {}).get("dp")
    return not (dp and float(dp.get("noise_multiplier") or 0.0) > 0.0)


def train_loss_round_record(out_r: Mapping) -> Optional[dict]:
    """FL-15 R5: one TRAIN_LOSS.jsonl line from fl.runtime's step record (round + fl.runtime.round_loss_record:
    n_clients / n_steps with steps > 0, loss_sum, mean_step_loss = loss_sum / n_steps, client_mean_loss = the mean over
    those clients of loss_sum_u / steps_u; None losses for a round without training clients)."""
    tl = out_r.get("train_loss")
    if tl is None:
        return None
    return dict(tl, round=int(out_r["round"]))


def append_train_loss(run_dir: Path, rec: Mapping) -> None:
    with open(Path(run_dir) / TRAIN_LOSS_FILE, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(dict(rec), sort_keys=True, allow_nan=False) + "\n")


def read_train_loss(run_dir: Path) -> list:
    p = Path(run_dir) / TRAIN_LOSS_FILE
    if not p.is_file():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def trim_train_loss(run_dir: Path, cursor: int) -> None:
    """On resume: drop the lines of rounds >= the checkpoint cursor (re-run and logged again); cursor 0 clears."""
    p = Path(run_dir) / TRAIN_LOSS_FILE
    if p.is_file():
        keep = [r for r in read_train_loss(run_dir) if int(r["round"]) < int(cursor)]
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in keep), encoding="utf-8", newline="\n")


def train_loss_summary(run_dir: Path, rounds_done: int) -> dict:
    """The RUN_SUMMARY record of TRAIN_LOSS.jsonl: one line per round 0 .. rounds_done - 1 (complete flag + missing
    rounds; flagged, not refused: the log is descriptive)."""
    recs = read_train_loss(run_dir)
    rounds = [int(r["round"]) for r in recs]
    if rounds != sorted(set(rounds)):
        raise RuntimeError(f"{TRAIN_LOSS_FILE}: rounds are not strictly increasing")
    want = list(range(int(rounds_done)))
    missing = sorted(set(want) - set(rounds))
    return {"file": TRAIN_LOSS_FILE, "rounds_logged": len(recs), "rounds_expected": len(want),
            "complete": rounds == want, "missing_rounds": missing[:50],
            "sha256": _file_sha(Path(run_dir) / TRAIN_LOSS_FILE) if recs else None,
            "fields": sorted(recs[0]) if recs else [], "authority": TRAIN_LOSS_AUTHORITY}


def update_norm_applies(spec: Mapping) -> bool:
    """P2-01 (PREG2 §3.3): D1-ext UPDATE_NORM_LOG.jsonl is written by Phase-2 arms only (never by a Phase-1 arm)."""
    return is_p2_arm(spec.get("arm"))


def update_norm_round_record(out_r: Mapping) -> Optional[dict]:
    """P2-01: one UPDATE_NORM_LOG.jsonl line from fl.runtime's step record (`update_norms`, fl.update_log) + round."""
    un = out_r.get("update_norms")
    if un is None:
        return None
    return dict(un, round=int(out_r["round"]))


def append_update_norm_log(run_dir: Path, rec: Mapping) -> None:
    with open(Path(run_dir) / UPDATE_NORM_FILE, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(dict(rec), sort_keys=True, allow_nan=False) + "\n")


def _update_norm_lines(run_dir: Path) -> tuple:
    """(parsed records, number of unparsable lines): a torn line (crash during an append) is counted, never raised —
    the log is descriptive and must not block a run's completion or resume (REV-P2-01_02 m5)."""
    p = Path(run_dir) / UPDATE_NORM_FILE
    if not p.is_file():
        return [], 0
    recs, torn = [], 0
    for x in p.read_text(encoding="utf-8").splitlines():
        if not x.strip():
            continue
        try:
            r = json.loads(x)
            int(r["round"])
        except (ValueError, TypeError, KeyError):
            torn += 1
            continue
        recs.append(r)
    return recs, torn


def read_update_norm_log(run_dir: Path) -> list:
    return _update_norm_lines(run_dir)[0]


def trim_update_norm_log(run_dir: Path, cursor: int) -> None:
    """On resume: drop the lines of rounds >= the checkpoint cursor (re-run and logged again); cursor 0 clears."""
    p = Path(run_dir) / UPDATE_NORM_FILE
    if p.is_file():
        keep = [r for r in read_update_norm_log(run_dir) if int(r["round"]) < int(cursor)]
        p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in keep), encoding="utf-8", newline="\n")


def update_norm_summary(run_dir: Path, rounds_done: int, trained_keys=(), trained_numel: int = 0) -> dict:
    """P2-01: the RUN_SUMMARY record of UPDATE_NORM_LOG.jsonl (complete flag + missing rounds, flagged not refused: the
    log is descriptive), the trained subset T (read back by tools/p2_noop_check.py) and the last record's totals."""
    recs, torn = _update_norm_lines(run_dir)
    rounds = [int(r["round"]) for r in recs]
    monotonic = rounds == sorted(set(rounds))              # flagged, never raised (REV-P2-01_02 m5)
    want = list(range(int(rounds_done)))
    missing = sorted(set(want) - set(rounds))
    last = next((r for r in reversed(recs) if r.get("server_step")), None)
    return {"file": UPDATE_NORM_FILE, "rounds_logged": len(recs), "rounds_expected": len(want),
            "complete": rounds == want and torn == 0, "missing_rounds": missing[:50],
            "strictly_increasing": monotonic, "unparsable_lines": torn,
            "server_steps": sum(1 for r in recs if r.get("server_step")),
            "cos_prev_unavailable_rounds": [int(r["round"]) for r in recs
                                            if r.get("server_step") and not r.get("cos_prev_available")][:50],
            "sha256": _file_sha(Path(run_dir) / UPDATE_NORM_FILE) if recs else None,
            "trained_keys": list(trained_keys), "trained_numel": int(trained_numel),
            "last_total_T": None if last is None else last.get("total_T"),
            "authority": UPDATE_NORM_AUTHORITY}


def device_record(adapter) -> dict:
    """INT-06 / REV-FL-01 M2: the evaluation device class (and registered pin) a leg runs on; ARM_SPEC.json keeps the
    first leg's, and a resume leg on another class / pin is refused (training numerics never mix device classes)."""
    from v4eval.device_pin import pin_for_device
    from v4eval.frozen_config import eval_device_class
    pin = pin_for_device(adapter.device)
    return {"eval_device_class": eval_device_class(adapter.device),
            "device_pin_sha256": pin["self_sha256"] if pin else None}


def fl_code_sha256() -> str:
    """06E §8: the code identity of an FL leg (fl/*.py + this launcher; v4eval.hashing rule)."""
    from v4eval.hashing import code_sha256
    return code_sha256(sorted((CODE / "fl").glob("*.py")) + [Path(__file__).resolve()])


def save_pf_store(path: Path, run, *, run_id: str, config_digest: str, theta_state) -> dict:
    """REV-FL-01 B1/B2: the run's CURRENT personal store (run.store, never a stale pre-resume object) with the mark it
    belongs to and the digest of the shared weights it pairs with, so shared + p_u of that mark can be rebuilt."""
    from fl.checkpoint import atomic_save, sha256_file
    from fl.numerics import state_digest
    half = int(run.state.evals[-1][0]) if run.state.evals else None
    payload = {"kind": "pf_store", "run_id": run_id, "config_digest": config_digest, "store": run.store.state_dict(),
               "half_efe": half, "mark": None if half is None else half / 2.0, "round": int(run.state.cursor),
               "theta_digest": state_digest(theta_state), "store_digest": run.store.digest(),
               "clients": len(run.store)}
    atomic_save(payload, Path(path))
    return {"file": Path(path).name, "sha256": sha256_file(Path(path)), "store_digest": payload["store_digest"],
            "theta_digest": payload["theta_digest"], "half_efe": half, "mark": payload["mark"],
            "clients": payload["clients"]}


def pf_best_record(run_dir: Path, run, best_u: int, *, run_id: str, config_digest: str, theta_state) -> dict:
    """The PF_STORE_best.pt of the PRACTICAL_BEST mark (B2), verified: same run / config, the best mark. A crash between
    best.pt and PF_STORE_best.pt is repaired only when the current state IS the best mark (else refused: re-run)."""
    from fl.checkpoint import load_checkpoint, sha256_file
    p = Path(run_dir) / PF_STORE_BEST_FILE
    obj = load_checkpoint(p) if p.is_file() else None
    ok = (obj is not None and obj.get("run_id") == run_id and obj.get("config_digest") == config_digest
          and obj.get("half_efe") == int(best_u))
    if not ok:
        if run.state.evals and int(run.state.evals[-1][0]) == int(best_u):
            return save_pf_store(p, run, run_id=run_id, config_digest=config_digest, theta_state=theta_state)
        raise RuntimeError(f"{p} does not hold the PF store of the best mark {best_u / 2:.1f}: the PRACTICAL_BEST "
                           "PF model cannot be rebuilt (re-run this PF arm)")
    return {"file": p.name, "sha256": sha256_file(p), "store_digest": obj["store_digest"],
            "theta_digest": obj["theta_digest"], "half_efe": obj["half_efe"], "mark": obj["mark"],
            "clients": obj["clients"]}


class Interrupted(RuntimeError):
    pass


def execute(*, spec: Mapping, run_id: str, run_dir: Path, server_adapter, init_sha256: str, workers,
            keys, n_valid_by_key: Mapping[str, int], get_client: Callable, n_decisions: int, manifest_hash: str,
            evaluate: Optional[Callable], resume: bool = False, ckpt_every: int = 64,
            smoke_rounds: Optional[int] = None, interrupt_at: int = 0, say: Callable = print,
            evaluate_pf: Optional[Callable] = None) -> dict:
    """Run / resume one arm. `evaluate(mark) -> (value, rows, ranks, result)` evaluates the CURRENT server weights
    (the harness model is the server adapter). INT-04 item 3: for PF, `evaluate_pf(mark, store)` (same return) scores
    shared + p_u per evaluation user (v4int.launch_device.pf_evaluator); without it PF would be scored at p = 0.
    Returns the status record."""
    import numpy as np
    from fl.checkpoint import CheckpointManager, atomic_save
    from fl.dp import median_clip_norm
    from fl.participation import ParticipationPlan
    from fl.personal import PersonalStore
    from fl.runtime import FLRun
    from fl.server import Server
    run_dir = Path(run_dir)
    arm = spec["arm"]
    digest = spec_digest(spec)
    role = spec.get("role", "MAIN")
    lr_mult = float(spec.get("lr_mult", 1.0))
    codec = spec.get("upload_codec")
    p2 = is_p2_arm(arm)                                    # P2-02: the Phase-2 family (every branch below is guarded)
    q8_codec = codec if is_q8_codec(codec) else None       # REV-P2-03_04 (a): Q8 is dispatched on the codec NAME
    if codec is not None and q8_codec is None and not p2:
        raise LaunchRefused(f"{arm}: upload codec {codec.get('name')!r} is not a Phase-1 codec")
    if p2:
        check_p2_spec(spec)
        sopt_kw = {"tuning_labels": p2_run_labels(spec)}
    else:
        check_warm_spec(spec)                                                              # FL-12
        sopt_kw = {"tuning_labels": dict(row_labels(spec_tuning(spec)), **warm_row_labels(spec),   # FL-10 / FL-12 rows
                                         **poisson_row_labels(spec))}                            # FIX-DELTA-02 M1
    cfg = fl_run_config(spec, run_id, ckpt_every)
    plan = ParticipationPlan.build(list(keys), seed=int(spec["cfg"]["seed"]), manifest_hash=manifest_hash,
                                   group_size=spec["plan"]["group_size"], sampling=spec["plan"]["sampling"])
    frozen = tuple(spec.get("frozen_item_tables") or ())
    freeze_rec = None
    if frozen:                                             # FL-12b (06K §3): before the Server reads the manifest
        from fl.runtime import freeze_item_tables
        freeze_rec = freeze_item_tables(server_adapter, frozen)
        theta_pre = server_adapter.broadcast_state()
        for w in workers:                                  # the one-time bootstrap download into each worker
            freeze_item_tables(w, frozen, source=theta_pre)
    p2_pre = None
    if p2:                                                 # P2 HOOK 1 (P2-04 LoRA)
        p2_pre, init_sha256 = p2_pre_server(spec, server_adapter, workers, init_sha256)
    server = Server(server_adapter, init_sha256=init_sha256)
    un_obs = None
    if update_norm_applies(spec):                          # P2-01 D1-ext: theta_pre = the server's verified init state
        from fl.update_log import UpdateNormObserver       # (captured before any resume loads latest.pt)
        un_obs = server.observer = UpdateNormObserver(server.adapter.broadcast_state(clone=True),
                                                      server.manifest.shared_keys)
    store = PersonalStore(server_adapter.query_dim) if cfg.method == "PF" else None
    ca = base_arm(arm)                                     # FL-16: a warm-line arm runs its cold arm's semantics
    ckpt = CheckpointManager(run_dir / "ckpt", run_id, retain_weights=ca != "S_CAL" and smoke_rounds is None)
    holder: dict = {}

    def eval_fn(_server) -> float:
        run = holder["run"]
        mark = run._half_units() / 2.0
        if evaluate_pf is not None and run.store is not None:  # B1: the run's live store (replaced on resume)
            value, rows, ranks, result = evaluate_pf(mark, run.store)
        else:
            value, rows, ranks, result = evaluate(mark)
        efe = run.counters.efe
        rows = fl_rows(rows, arm=arm, run_id=run_id, efe_raw=efe, mark=mark, digest=digest, role=role,
                       lr_mult=lr_mult, **sopt_kw)
        tag = f"{mark:.1f}"
        write_json(run_dir / "evals" / f"eval_{tag}.rows.json", {"mark": mark, "efe": efe, "rows": rows})
        if result is not None:
            write_json(run_dir / "evals" / f"eval_{tag}.result.json", result)
        if ranks is not None:
            np.save(run_dir / "evals" / f"eval_{tag}.ranks_int32.npy", np.asarray(ranks, dtype=np.int32))
        with open(run_dir / "metrics.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps({"kind": "eval", "mark": mark, "efe": efe, "round": run.state.cursor,
                                "rows_file": f"evals/eval_{tag}.rows.json"}) + "\n")
        return float(value)

    use_eval = evaluate is not None and ca != "S_CAL" and smoke_rounds is None
    tl_on = train_loss_applies(spec)                       # FL-15 R5: TRAIN_LOSS.jsonl (never for z > 0)
    kw = dict(workers=list(workers), store=store, ckpt=ckpt, eval_fn=eval_fn if use_eval else None,
              n_valid_by_key=dict(n_valid_by_key))
    spec_file = run_dir / "ARM_SPEC.json"
    dev_rec, code_sha = device_record(server_adapter), fl_code_sha256()
    spec_rec = {"arm": arm, "run_id": run_id, "arm_spec_sha256": digest, "arm_spec": dict(spec), "device": dev_rec,
                "code_sha256": code_sha}
    if resume:
        if spec_file.exists() or (ckpt.dir / ckpt.LATEST).exists():
            try:
                prev = json.loads(spec_file.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                raise LaunchRefused(f"--resume: {spec_file} unreadable ({e})") from e
            if prev.get("arm_spec_sha256") != digest or prev.get("arm") != arm:
                raise LaunchRefused(f"--resume: {spec_file} is arm {prev.get('arm')} / spec "
                                    f"{prev.get('arm_spec_sha256')}, not {arm} / {digest}")
            if prev.get("device") != dev_rec:                      # M2: fail closed (a pre-INT-06 leg has none)
                raise LaunchRefused(f"--resume: {spec_file} was started on device {prev.get('device')}, this leg "
                                    f"runs on {dev_rec}: a run never mixes evaluation device classes / pins")
            if prev.get("code_sha256") != code_sha:                # 06E §8: one code identity per run
                raise LaunchRefused(f"--resume: {spec_file} was started by FL code {prev.get('code_sha256')}, this "
                                    f"leg is {code_sha}: resume with the tree that started the run")
        else:                                              # --resume on an empty run dir starts fresh (as before)
            write_json(spec_file, spec_rec)
        run = FLRun.resume(cfg, plan, server, get_client, int(n_decisions), **kw)
        trim_clip_log(run_dir, run.state.cursor)          # FL-14 (06L A4): re-run rounds are logged again
        trim_train_loss(run_dir, run.state.cursor if tl_on else 0)   # FL-15 R5 (the same rule)
        if un_obs is not None:                             # P2-01 (the same rule)
            trim_update_norm_log(run_dir, run.state.cursor)
    else:
        if (ckpt.dir / ckpt.LATEST).exists():
            raise LaunchRefused(f"{ckpt.dir / ckpt.LATEST} exists: use --resume")
        run = FLRun(cfg, plan, server, get_client, int(n_decisions), **kw)
        write_json(spec_file, spec_rec)
        trim_clip_log(run_dir, 0)                          # FL-14: a fresh start never keeps stale clip lines
        trim_train_loss(run_dir, 0)                        # FL-15 R5
        if un_obs is not None:                             # P2-01
            trim_update_norm_log(run_dir, 0)
    p2_post = p2_post_run(run, spec) if p2 else None       # P2 HOOK 2 (P2-03 install_sparse)
    q8 = None
    if q8_codec is not None:                               # FA_Q8: after construction AND resume (both rebuild it)
        from fl.quant import install_q8
        q8 = install_q8(run, seed=cfg.seed)
    holder["run"] = run
    best_seen = run.state.best_half_efe                    # B2: PF_STORE_best.pt follows every new best mark
    if smoke_rounds is not None:                           # (FL-07: the run's own unit rule, Poisson included)
        first_eval = next((i for i in range(len(run.lr_table["rows"])) if run.units_after_row(i) >= 1), None)
        if first_eval is not None and smoke_rounds > first_eval:
            raise LaunchRefused(f"--smoke-rounds {smoke_rounds} reaches the first evaluation (round {first_eval})")
    status_path = run_dir / "STATUS.json"
    max_rounds = int(spec["max_rounds"]) if smoke_rounds is None else int(smoke_rounds)
    write_json(status_path, {"run_id": run_id, "arm": arm, "status": "RUNNING", "round": run.state.cursor,
                             "utc": utc()})
    t0 = time.perf_counter()
    while not run.state.done and run.state.cursor < max_rounds:
        out_r = run.step()
        clip_rec = clip_round_record(out_r)                # FL-14 (06L A4): FA_1024 clipped fraction per round
        if clip_rec is not None:
            append_clip_log(run_dir, clip_rec)
        tl_rec = train_loss_round_record(out_r) if tl_on else None   # FL-15 R5: per-round client training loss
        if tl_rec is not None:
            append_train_loss(run_dir, tl_rec)
        un_rec = update_norm_round_record(out_r) if un_obs is not None else None   # P2-01 D1-ext
        if un_rec is not None:
            append_update_norm_log(run_dir, un_rec)
        if q8 is not None:                               # every survivor's upload went through the Q8 decode
            if run.channel is not q8:
                raise RuntimeError("FA_Q8: the run's channel is no longer the Q8 channel")
            q8.check_round(out_r["round"], out_r["survived"])
        if run.store is not None and run.state.best_half_efe != best_seen:
            best_seen = run.state.best_half_efe            # this step's evaluation improved: best.pt was just saved
            save_pf_store(run_dir / PF_STORE_BEST_FILE, run, run_id=run_id, config_digest=cfg.digest(),
                          theta_state=server_adapter.broadcast_state())
        if interrupt_at and run.state.cursor == int(interrupt_at):
            ckpt.save_latest(run.state_dict())
            write_json(status_path, {"run_id": run_id, "arm": arm, "status": "INTERRUPTED", "round": interrupt_at,
                                     "utc": utc()})
            say(f"[fl] {run_id}: interrupted after round {interrupt_at} (latest saved)")
            return {"status": "INTERRUPTED", "round": int(interrupt_at), "summary": run.summary()}
    wall = time.perf_counter() - t0
    extra_summary = {"frozen_item_tables": freeze_rec} if freeze_rec is not None else {}
    if spec.get("init_from") and spec["cfg"].get("dp"):    # FL-16 (06L_C C8): the DP clip / noise dimension, measured
        extra_summary["dp_noise_dimension"] = {"uploaded_numel": int(server.manifest.shared_numel),
                                               "uploaded_keys": list(server.manifest.shared_keys),
                                               "frozen_keys_excluded": list(frozen),
                                               "rule": "fl.dp clips / noises the manifest's shared (uploaded) keys "
                                                       "only; accounting (z, q, T, delta) unchanged"}
    clip_sum = clip_log_summary(run_dir, clip_expected_rounds(spec, run))   # FL-14 (06L A4 / 06L_E E1)
    if clip_sum is not None:
        extra_summary["clip_log"] = clip_sum
        if clip_sum["complete"] is False:                  # REV-FL-14 m1: flagged in RUN_SUMMARY and the console
            say(f"[fl] {run_id}: WARNING {CLIP_LOG_FILE} incomplete: {clip_sum['n_missing']} round(s) missing "
                f"{clip_sum['missing_rounds'][:5]}, unexpected {clip_sum['unexpected_rounds'][:5]}")
    if tl_on:                                              # FL-15 R5
        extra_summary["train_loss_log"] = tl_sum = train_loss_summary(run_dir, run.state.cursor)
        if not tl_sum["complete"]:
            say(f"[fl] {run_id}: WARNING {TRAIN_LOSS_FILE} incomplete: missing rounds {tl_sum['missing_rounds'][:5]}")
    if un_obs is not None:                                 # P2-01 D1-ext (+ the trained subset T for p2_noop_check)
        extra_summary["update_norm_log"] = un_sum = update_norm_summary(
            run_dir, run.state.cursor, trained_keys=server.manifest.shared_keys,
            trained_numel=server.manifest.shared_numel)
        if not un_sum["complete"]:
            say(f"[fl] {run_id}: WARNING {UPDATE_NORM_FILE} incomplete: missing rounds {un_sum['missing_rounds'][:5]}")
    if p2:                                                 # P2-02: the recipe and the hook records
        extra_summary["p2"] = {"label": spec["p2_label"], "recipe": dict(spec["p2_recipe"]), "pre_server": p2_pre,
                               "post_run": p2_post, "authority": P2_AUTHORITY}
    summary = dict(run.summary(), **extra_summary, arm=arm, arm_spec=dict(spec), arm_spec_sha256=digest, wall_s_this_leg=round(wall, 2),
                   bytes=bytes_record(server.manifest, run.counters, run.ledger, codec=q8_codec, frozen_keys=frozen),
                   lr_table={k: run.lr_table[k] for k in ("digest", "n_rounds", "E_total", "planned_efe",
                                                         "min_lr", "final_lr")})
    if smoke_rounds is not None:
        return _finish_smoke(run, ckpt, evaluate, run_dir, run_id, arm, digest, summary, say, role=role,
                             lr_mult=lr_mult, **sopt_kw)
    if ca == "S_CAL":
        norm = median_clip_norm(run.dp_norm_log)
        s_file = "S_RECORD.json"
        if is_warm_arm(arm):                               # FL-16: the warm chain's own record (own format / file)
            s_file = S_RECORD_WARM_FILE[warm_variant(arm)]
            obj = s_record_obj_warm(norm, spec=spec, run_id=run_id, seed=cfg.seed, config_digest=cfg.digest(),
                                    plan_digest=plan.digest(), init_sha256=server.init_sha256, smoke=False)
        else:
            obj = s_record_obj(norm, run_id=run_id, seed=cfg.seed, config_digest=cfg.digest(),
                               plan_digest=plan.digest(), init_sha256=server.init_sha256, smoke=False)
        from v4models.variants import write_exclusive
        write_exclusive(run_dir / s_file, json.dumps(obj, indent=1, sort_keys=True) + "\n")
        ckpt.complete()
        for name in (ckpt.BEST, ckpt.ENDPOINT):
            ckpt._remove(ckpt.dir / name, "S_CAL: no weights retained")
        summary["s_record"] = {"S": obj["S"], "self_sha256": obj["self_sha256"], "n_norms": obj["n_norms"]}
        if s_file != "S_RECORD.json":                      # FL-16 (the cold summary is unchanged)
            summary["s_record"].update(file=s_file, format=obj["format"])
        write_json(run_dir / "RUN_SUMMARY.json", dict(summary, status="DONE_S_CAL"))
        write_json(status_path, {"run_id": run_id, "arm": arm, "status": "DONE_S_CAL", "utc": utc()})
        say(f"[fl] {run_id}: DONE_S_CAL ({s_file} written)")
        return {"status": "DONE_S_CAL", "summary": summary}
    evals = run.state.evals
    if not evals or evals[-1][0] != 12:
        raise RuntimeError("the run ended without its 6.0 evaluation")
    best_u = run.state.best_half_efe
    pf_ref = {}
    if run.store is not None:                              # B1: run.store (the resumed run's own store)
        pf_ref["CONTROLLED_BUDGET"] = save_pf_store(run_dir / PF_STORE_FILE, run, run_id=run_id,
                                                    config_digest=cfg.digest(),
                                                    theta_state=server_adapter.broadcast_state())
        pf_ref["PRACTICAL_BEST"] = pf_best_record(run_dir, run, best_u, run_id=run_id, config_digest=cfg.digest(),
                                                  theta_state=server_adapter.broadcast_state())
        summary["pf_store"] = {"file": PF_STORE_FILE, "digest": run.store.digest(), "clients": len(run.store),
                               "record": pf_ref["CONTROLLED_BUDGET"]}
        summary["pf_store_best"] = pf_ref["PRACTICAL_BEST"]
    rows = []
    for rule, u, cid in (("PRACTICAL_BEST", best_u, f"{run_id}:best@{best_u / 2:.1f}"),
                         ("CONTROLLED_BUDGET", 12, f"{run_id}:endpoint_6.0")):
        rec = json.loads((run_dir / "evals" / f"eval_{u / 2:.1f}.rows.json").read_text(encoding="utf-8"))
        extra = {} if rule not in pf_ref else {"pf_store_file": pf_ref[rule]["file"],
                                                "pf_store_sha256": pf_ref[rule]["sha256"],
                                                "pf_store_digest": pf_ref[rule]["store_digest"]}
        if rule in pf_ref:                                 # 06E §7: the store digest is part of the checkpoint id
            cid = f"{cid}|pf_store={pf_ref[rule]['store_digest']}"
        rows += [dict(r, selection_rule=rule, checkpoint_id=cid, **extra) for r in rec["rows"]]
    write_json(run_dir / "RESULT_ROWS.json", {"run_id": run_id, "arm": arm, "rows": rows})
    ckpt.complete()                                        # latest.pt removed (disk rule 2); best / endpoint kept
    summary["checkpoints"] = ckpt.files()
    write_json(run_dir / "RUN_SUMMARY.json", dict(summary, status="DONE_MAIN"))
    write_json(status_path, {"run_id": run_id, "arm": arm, "status": "DONE_MAIN", "utc": utc()})
    say(f"[fl] {run_id}: DONE_MAIN")                       # the status word only (no value printed)
    return {"status": "DONE_MAIN", "summary": summary}


def _finish_smoke(run, ckpt, evaluate, run_dir, run_id, arm, digest, summary, say, role: str = "MAIN",
                  lr_mult: float = 1.0, tuning_labels: Optional[Mapping] = None) -> dict:
    from v4int.smoke import smoke_rows
    deleted = {}
    for p in sorted(ckpt.dir.glob("*.pt")):
        from fl.checkpoint import sha256_file
        deleted[p.name] = sha256_file(p)
        ckpt._remove(p, "INT-03 smoke: weights deleted")
    ev = None
    if evaluate is not None:
        mark = run._half_units() / 2.0
        value, rows, _ranks, _res = evaluate(mark)
        rows = smoke_rows(fl_rows(rows, arm=arm, run_id=run_id, efe_raw=run.counters.efe, mark=mark, digest=digest,
                                  role=role, lr_mult=lr_mult, tuning_labels=tuning_labels))
        write_json(run_dir / "SMOKE_EVAL.json", {"label": "SMOKE_NOT_EVIDENCE", "run_id": run_id, "rows": rows})
        ev = {"rows": len(rows)}
    if base_arm(arm) == "S_CAL" and run.dp_norm_log:
        from fl.dp import median_clip_norm
        norm = median_clip_norm(run.dp_norm_log, rounds=range(len(run.dp_norm_log)))
        if is_warm_arm(arm):                               # FL-16: the warm smoke record (label SMOKE_NOT_EVIDENCE)
            obj = s_record_obj_warm(norm, spec=summary["arm_spec"], run_id=run_id, seed=run.cfg.seed,
                                    config_digest=run.cfg.digest(), plan_digest=run.plan.digest(),
                                    init_sha256=run.server.init_sha256, smoke=True)
            write_json(run_dir / S_RECORD_WARM_FILE[warm_variant(arm)], obj)
        else:
            obj = s_record_obj(norm, run_id=run_id, seed=run.cfg.seed, config_digest=run.cfg.digest(),
                               plan_digest=run.plan.digest(), init_sha256=run.server.init_sha256, smoke=True)
            write_json(run_dir / "S_RECORD.json", obj)
    summ = dict(summary, label="SMOKE_NOT_EVIDENCE", weights_deleted=deleted, eval=ev, status="SMOKE_DONE_NOT_EVIDENCE")
    write_json(run_dir / "SMOKE_SUMMARY.json", summ)
    write_json(run_dir / "STATUS.json", {"run_id": run_id, "arm": arm, "status": "SMOKE_DONE_NOT_EVIDENCE",
                                         "utc": utc()})
    say(f"[fl] {run_id}: SMOKE_SUMMARY.json written")
    return {"status": "SMOKE_DONE_NOT_EVIDENCE", "summary": summ}


# ------------------------------------------------------------------------------------------------ preflight / real run
def preflight(args, *, runner=None, environ=None, gate_path: Path = G1_GATE) -> dict:
    """Torch-free refusals + the shared GPU checks (launch_central.gpu_checks)."""
    from v4hist.policy import HistoryPolicyError, require_policy
    from v4sup.gpu import default_runner
    env = os.environ if environ is None else environ
    rec = {"utc": utc(), "argv": list(sys.argv), "host": platform.node(), "smoke": bool(args.smoke),
           "arm": args.arm, "allow_shared_gpu": bool(args.allow_shared_gpu)}
    try:
        rec["history_policy"] = require_policy(args.history_policy)
    except HistoryPolicyError as e:
        raise LaunchRefused(str(e)) from e
    if args.gpu is None or int(args.gpu) < 0:
        raise LaunchRefused("an explicit physical GPU id (--gpu >= 0) is required")
    if args.family != FAMILY:
        raise LaunchRefused(f"--family {args.family}: the FL arms run the ARCHITECTURE_LOCK winner ({FAMILY}) only")
    ca = base_arm(args.arm)                                # FL-16: DP_WARM[_FROZEN] etc. as their cold arm
    if ca == "DP" and args.epsilon is None:
        raise LaunchRefused("DP needs --epsilon")
    if ca != "DP" and args.epsilon is not None:
        raise LaunchRefused("--epsilon is valid for DP only")
    if ca in ("FA_1024", "DP") and args.clip_record is None:
        raise LaunchRefused(f"{args.arm} needs --clip-record (the S_CAL S_RECORD.json"
                            + (f"; warm: {S_RECORD_WARM_FILE[warm_variant(args.arm)]} of "
                               f"{S_CAL_WARM_ARM[warm_variant(args.arm)]})" if is_warm_arm(args.arm) else ")"))
    if ca == "S_CAL" and int(args.seed) != KNOBS["s_cal_seed"]:
        raise LaunchRefused("S_CAL runs at seed 2026 only")
    if int(args.checkpoint_every) < 1:
        raise LaunchRefused("--checkpoint-every must be >= 1")
    fl05_preflight(args)
    sel = getattr(args, "lr_selection", None)
    rec["fl05"] = {"role": getattr(args, "role", "MAIN"), "lr_mult": getattr(args, "lr_mult", None),
                   "lr_selection": None if sel is None else str(sel)}
    fl10 = {k: getattr(args, k, None) for k in FL10_OPTS + ("fl_tuning",)}
    if any(v is not None for v in fl10.values()):          # FL-10 (absent when every option is off)
        rec["fl10"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in fl10.items()}
    if args.smoke:
        lim = P2_MAX_SMOKE_ROUNDS if is_p2_arm(args.arm) else MAX_SMOKE_ROUNDS   # P2-02: P2-T3 smokes 65 rounds
        if not 1 <= int(args.smoke_rounds) <= lim:
            raise LaunchRefused(f"--smoke-rounds must be in [1, {lim}]")
        k = int(args.smoke_interrupt_at)
        if k < 0 or k >= int(args.smoke_rounds):
            raise LaunchRefused("--smoke-interrupt-at must be 0 or below --smoke-rounds")
        rec["g1_gate"] = {"required": False, "reason": "smoke (SMOKE_NOT_EVIDENCE)"}
    else:
        if int(args.smoke_interrupt_at):
            raise LaunchRefused("--smoke-interrupt-at is a smoke-only option")
        if not Path(gate_path).is_file():
            raise LaunchRefused(f"non-smoke run refused: {gate_path} does not exist (G1 not approved)")
        rec["g1_gate"] = {"required": True, "path": str(gate_path),
                          "sha256": hashlib.sha256(Path(gate_path).read_bytes()).hexdigest()}
    if args.dry_run:
        rec["gpu_check"] = {"skipped": "dry-run"}
        return rec
    rec["gpu_check"] = gpu_checks(args, runner or default_runner, env)
    return rec


def fl05_preflight(args) -> None:
    """FL-05 refusals (torch-free): the registered multipliers, the exclusive LR options, CALIBRATION = FA at 2026.
    P2-02: a P2 arm takes p2_preflight instead; a Phase-1 arm refuses every P2 flag (--p2-recipe / --p2-selection)."""
    if is_p2_arm(args.arm):                                # P2-02 (the Phase-1 checks below are unchanged)
        p2_preflight(args)
        return
    p2_given = [o for o in P2_FLAGS if getattr(args, o, None) is not None]
    if p2_given:
        raise LaunchRefused(f"--{p2_given[0].replace('_', '-')} is a P2 arm option ({P2_ARMS}), not {args.arm}'s")
    role = getattr(args, "role", "MAIN")
    if role not in ROLES:
        raise LaunchRefused(f"--role must be one of {ROLES}, not {role}")
    if getattr(args, "lr_mult", None) is not None:
        lr_mult_value(args.lr_mult)
    if getattr(args, "lr_selection", None) is not None:
        if getattr(args, "lr_mult", None) is not None:
            raise LaunchRefused("--lr-mult and --lr-selection are exclusive")
        if role == "CALIBRATION":
            raise LaunchRefused("--lr-selection is the screen's output: a CALIBRATION (screen) run takes --lr-mult")
    if role == "CALIBRATION":
        if args.arm not in CALIB["arms"]:
            raise LaunchRefused(f"--role CALIBRATION is the FA local-LR screen only (amendment 06B §1), not {args.arm}")
        if int(args.seed) != CALIB["seed"]:
            raise LaunchRefused("the CALIBRATION LR screen runs at seed 2026 only (amendment 06B §1)")
    fl10_preflight(args)
    warm_preflight(args)


def warm_preflight(args) -> None:
    """FL-12 refusals (torch-free): FA_WARM / FA_WARM_FROZEN <-> --init-from; FA_WARM_FROZEN <-> --freeze-item-tables;
    --warm-lr-factor registered for the arm and a screen / smoke option; --warm-selection = non-smoke MAIN warm runs
    only (and required there)."""
    arm = args.arm
    init = getattr(args, "init_from", None)
    frz = bool(getattr(args, "freeze_item_tables", False))
    wf = getattr(args, "warm_lr_factor", None)
    wsel = getattr(args, "warm_selection", None)
    if is_warm_arm(arm) != (init is not None):             # FL-16: + the warm method set (WARM_LINE)
        raise LaunchRefused(f"--arm {arm}: " + ("a warm-start arm needs --init-from <PRETRAIN_FOR_FL_INIT.json>"
                                                if is_warm_arm(arm) else "--init-from is a FA_WARM / FA_WARM_FROZEN "
                                                                         "option (and of the FL-16 *_WARM / "
                                                                         "*_WARM_FROZEN arms)"))
    if frz != is_frozen_arm(arm):
        raise LaunchRefused("--freeze-item-tables is the FA_WARM_FROZEN arm (06K §3; it needs --init-from) and "
                            "FA_WARM_FROZEN needs it (FL-16: likewise every *_WARM_FROZEN arm, S8-bound)")
    if wf is not None:
        warm_factor_value(arm, wf)
    main_run = not args.smoke and getattr(args, "role", "MAIN") == "MAIN" and not getattr(args, "dry_run", False)
    if wsel is not None and (not is_warm_arm(arm) or getattr(args, "role", "MAIN") != "MAIN" or args.smoke):
        raise LaunchRefused("--warm-selection is the non-smoke MAIN input of a warm arm")
    if is_warm_arm(arm) and main_run:
        if wf is not None:
            raise LaunchRefused("--warm-lr-factor is a screen / smoke option: a non-smoke MAIN warm run takes the "
                                "registered --warm-selection")
        if wsel is None:
            raise LaunchRefused(f"a non-smoke MAIN {arm} run needs --warm-selection "
                                f"{WARM_SELECTION_PATH[WARM_STAGE[warm_variant(arm)]]} (06K: the S7 / S8 CALIB "
                                "screen)")
    if wsel is not None and wf is not None:
        raise LaunchRefused("--warm-selection and --warm-lr-factor are exclusive")
    warm_server_preflight(args, main_run=main_run)         # FL-15 (06M_B B2)


def warm_server_preflight(args, *, main_run: bool) -> None:
    """FL-15 (06M_B B2; torch-free): --warm-server-lr-factor is a warm-arm screen / smoke option registered for the
    tuning point's server optimiser (FedAdam {1, 0.1}; FedAvgM / FedAvg {1, 0.25}); a non-smoke MAIN warm run takes the
    server LR from the registered --warm-selection only (exclusive)."""
    wsf = getattr(args, "warm_server_lr_factor", None)
    if wsf is None:
        return
    arm = args.arm
    if not is_warm_arm(arm):
        raise LaunchRefused("--warm-server-lr-factor is a FA_WARM / FA_WARM_FROZEN option")
    if getattr(args, "fl_tuning", None) is None and getattr(args, "lr_selection", None) is None:
        warm_server_factor_value(arm, args_tuning(args)["optimizer"], wsf)
    elif _num(wsf, "--warm-server-lr-factor") not in {v for g in WARM_SERVER_FACTORS.values() for v in g}:
        raise LaunchRefused(f"--warm-server-lr-factor {wsf} is not a registered factor (06M_B B2)")
    if main_run:
        raise LaunchRefused("--warm-server-lr-factor is a screen / smoke option: a non-smoke MAIN warm run takes "
                            "the local factor AND the server LR from the registered --warm-selection (06M_B B2)")
    if getattr(args, "warm_selection", None) is not None:
        raise LaunchRefused("--warm-selection and --warm-server-lr-factor are exclusive")


FL10_OPTS = ("server_opt", "server_lr", "server_tau", "lr_shape", "client_opt", "client_lr", "local_passes",
             "local_batch")


def args_tuning(args) -> dict:
    lm = getattr(args, "lr_mult", None)
    return tuning_config(optimizer=getattr(args, "server_opt", None), server_lr=getattr(args, "server_lr", None),
                         server_tau=getattr(args, "server_tau", None), lr_mult=1.0 if lm is None else lm,
                         lr_shape=getattr(args, "lr_shape", None), client_opt=getattr(args, "client_opt", None),
                         client_lr=getattr(args, "client_lr", None), local_passes=getattr(args, "local_passes", None),
                         local_batch=getattr(args, "local_batch", None))


def fl10_preflight(args) -> None:
    """FL-10 refusals (torch-free): every tuning value registered (tuning_config); the tuning options and --lr-mult are
    screen / smoke options, exclusive with the selection files; --fl-tuning is a screen output (never a CALIBRATION
    input)."""
    given = [o for o in FL10_OPTS if getattr(args, o, None) is not None]
    tuning = getattr(args, "fl_tuning", None)
    if given and getattr(args, "lr_selection", None) is not None:
        raise LaunchRefused(f"--{given[0].replace('_', '-')} and --lr-selection are exclusive (a MAIN run takes the "
                            "--fl-tuning selection)")
    if tuning is not None:
        for opt in ("lr_mult", "lr_selection") + FL10_OPTS:
            if getattr(args, opt, None) is not None:
                raise LaunchRefused(f"--fl-tuning and --{opt.replace('_', '-')} are exclusive (the file fixes every "
                                    "tuning dimension)")
        if getattr(args, "role", "MAIN") == "CALIBRATION":
            raise LaunchRefused("--fl-tuning is the screen's output: a CALIBRATION (screen) run takes --lr-mult and "
                                "the tuning options")
    elif getattr(args, "lr_selection", None) is None:
        args_tuning(args)


def fedadam_grid(m_star: float) -> list:
    """06G §2: FedAdam x eta_s in {0.01, 0.1} x m in {1, m*} (m* = 1 -> {1, 4}), as (eta_s, m) pairs."""
    ms = (1.0, 4.0) if float(m_star) == 1.0 else (1.0, float(m_star))
    return [(lr, m) for lr in FEDADAM_LRS for m in ms]


def load_fl_tuning(path: Path) -> dict:
    """A verified stage file of `lr_screen select-joint` (PPSI_V4_FL_TUNING_SELECTION_1): format, self-sha256, a
    registered stage, every configuration registered, and the stage rule (06G §3 / 06J §1-§2) recomputed from its
    values and incumbent flags gives its winner and tie set.
    FIX-DELTA-02: the same lr_screen functions re-check 06J §4(ii) (stage_decision: winner, tie set, gain,
    best_candidate, no_measurable_gain, tie_band_set, min_gain as recorded), the incumbent / required runs
    (stage_required from the file's incumbent; S2: FedAvg at m*) and the registered edge runs (check_stage_runs: a
    condition-true edge group must be complete; the recorded `edges` must match).
    FL-14 (06L_D D1-D3): the same functions give the 06L sets — S2 required + FedAvgM m {1, 2} with the A1 m 0.5
    edge, S3 required + the two A2 cells (06J SGD edge on the 06J cells only); S4 accepts any S3 winner (A2 included);
    a file without the 06L cells is refused (required / edges mismatch)."""
    from v4int.lr_screen import MIN_GAIN, STAGES, SelectionRefused, check_stage_runs, stage_decision, stage_required
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LaunchRefused(f"--fl-tuning {path} unreadable ({e})") from e
    if obj.get("format") != FL_TUNING_FORMAT or obj.get("self_sha256") != _canon(obj):
        raise LaunchRefused(f"--fl-tuning {path} is not a verified {FL_TUNING_FORMAT} object")
    if obj.get("stage") not in STAGES:
        raise LaunchRefused(f"--fl-tuning {path}: stage {obj.get('stage')!r} is not one of {STAGES}")
    chain = ["S1b"] + list(STAGES)                         # FL-13: the history must end at the previous stage
    hist = [h.get("stage") for h in obj.get("stage_history") or [] if isinstance(h, Mapping)]
    if hist != chain[:chain.index(obj["stage"])]:
        raise LaunchRefused(f"--fl-tuning {path}: stage {obj['stage']} with stage history {hist} (expected "
                            f"{chain[:chain.index(obj['stage'])]})")
    try:
        configs = list(obj.get("configs") or [])
        for c in configs:
            if tuning_label(tuning_config(**{k: c.get(k) for k in TUNING_KEYS})) != c["label"]:
                raise SelectionRefused(f"label {c['label']} does not match its configuration")
        dec = stage_decision(obj["stage"], [c for c in configs if not c.get("excluded")])
        win, tie = dec["winner"], dec["tie_set"]
        for k in ("gain", "best_candidate", "no_measurable_gain", "tie_band_set", "min_gain"):   # 06J §4(ii)
            if k not in obj or obj[k] != dec[k]:
                raise SelectionRefused(f"{k} {obj.get(k, '<absent>')!r} != the 06J §4(ii) decision {dec[k]!r}")
        if dec["min_gain"] != MIN_GAIN:
            raise SelectionRefused(f"min_gain {dec['min_gain']} is not the registered {MIN_GAIN}")
        incs = [c for c in configs if c.get("incumbent")]
        if len(incs) != 1 or incs[0]["label"] != obj.get("incumbent"):
            raise SelectionRefused(f"incumbent {obj.get('incumbent')!r} is not the one incumbent configuration")
        inc_t = tuning_config(**{k: incs[0].get(k) for k in TUNING_KEYS})
        if obj["stage"] == "S2" and inc_t != tuning_config(lr_mult=lr_mult_value(obj.get("m_star"))):
            raise SelectionRefused(f"the S2 incumbent {incs[0]['label']} is not FedAvg at m* = {obj.get('m_star')}")
        req = [tuning_label(t) for t in stage_required(obj["stage"], inc_t, obj.get("m_star"))]
        if list(obj.get("required") or []) != req:
            raise SelectionRefused(f"required {obj.get('required')} != the registered {obj['stage']} set {req}")
        edges = check_stage_runs(obj["stage"], configs, incs[0]["label"], req)
        if obj.get("edges") != edges:
            raise SelectionRefused(f"edges {obj.get('edges')} != the registered edge groups {edges}")
        if obj["stage"] == "S4":                           # FL-13: one at a time at one (the S3 winner's) point
            cores = {tuning_label(dict(tuning_config(**{k: c.get(k) for k in TUNING_KEYS}),
                                       local_passes=KNOBS["passes"], local_batch=KNOBS["batch"])) for c in configs}
            if len(cores) != 1:
                raise SelectionRefused(f"S4 configurations differ beyond the local computation: {sorted(cores)}")
        w = tuning_config(**{k: (obj.get("winner") or {}).get(k) for k in TUNING_KEYS})
        if obj["stage"] == "S2":
            grid = sorted((float(c["server_lr"]), float(c["lr_mult"])) for c in configs
                          if c["optimizer"] == "fedadam" and c["server_lr"] in FEDADAM_LRS
                          and c["server_tau"] == FEDADAM_HYPER["tau"])
            want_grid = sorted(fedadam_grid(lr_mult_value(obj.get("m_star"))))
    except (SelectionRefused, LaunchRefused, KeyError, TypeError, ValueError) as e:
        raise LaunchRefused(f"--fl-tuning {path}: invalid tuning selection ({e})") from e
    if obj["stage"] == "S2" and grid != want_grid:
        raise LaunchRefused(f"--fl-tuning {path}: FedAdam configs {grid} != the 06G §2 grid of m* = {obj.get('m_star')}")
    if w != win or obj.get("tie_set") != tie:
        raise LaunchRefused(f"--fl-tuning {path}: the {obj['stage']} rule gives {tuning_label(win)} (tie set {tie}), "
                            f"the file says {tuning_label(w)} ({obj.get('tie_set')})")
    return obj


def fl_tuning_exists(path: Optional[Path] = None) -> list:
    """06G §3: the joint tuning files that exist (the registered path, and next to the given selection file)."""
    cands = [FL_TUNING_PATH] + ([Path(path).parent / FL_TUNING_NAME] if path is not None else [])
    return sorted({str(p) for p in cands if Path(p).is_file()})


def load_lr_selection(path: Path) -> dict:
    """A verified LR selection file (v4int.lr_screen): format PPSI_V4_FL_LR_SELECTION_1 (06B §1: values exactly for
    0.5 / 1 / 2) or PPSI_V4_FL_LR_SELECTION_EXT_1 (06F §3: any registered screened set), self-sha256, and the file's
    rule recomputed from its values must give its selected multiplier (and, for EXT, its tie set)."""
    from v4int.lr_screen import SelectionRefused, apply_rule, apply_rule_ext
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LaunchRefused(f"--lr-selection {path} unreadable ({e})") from e
    fmt = obj.get("format")
    if fmt not in (LR_SELECTION_FORMAT, LR_SELECTION_EXT_FORMAT) or obj.get("self_sha256") != _canon(obj):
        raise LaunchRefused(f"--lr-selection {path} is not a verified {LR_SELECTION_FORMAT} / "
                            f"{LR_SELECTION_EXT_FORMAT} object")
    vals = obj.get("values") or {}
    sel = lr_mult_value(obj.get("selected_lr_mult"))
    tag_to_m = {t: m for m, t in LR_MULT_TAG.items()}
    if fmt == LR_SELECTION_FORMAT:
        first = [LR_MULT_TAG[m] for m in FIRST_SCREEN_MULTS]
        if sorted(vals) != sorted(first):
            raise LaunchRefused(f"--lr-selection {path}: values must be exactly the multipliers {sorted(first)}")
        want, _tie = apply_rule({m: float(vals[LR_MULT_TAG[m]]) for m in FIRST_SCREEN_MULTS})
        if want != sel:
            raise LaunchRefused(f"--lr-selection {path}: the 06B §1 rule gives {want}, the file says {sel}")
        return obj
    if not vals or any(t not in tag_to_m for t in vals):                    # EXT (06F §3)
        raise LaunchRefused(f"--lr-selection {path}: values must be keyed by registered multipliers {list(tag_to_m)}")
    screened = obj.get("screened") or []
    excluded = [e.get("lr_mult") for e in (obj.get("excluded") or [])]
    try:
        scr = sorted(lr_mult_value(m) for m in screened)
        exc = sorted(lr_mult_value(m) for m in excluded)
        values = {tag_to_m[t]: float(v) for t, v in vals.items()}
        want, tie = apply_rule_ext(values)
    except (SelectionRefused, TypeError, ValueError) as e:
        raise LaunchRefused(f"--lr-selection {path}: invalid EXT selection ({e})") from e
    if scr != sorted(list(values) + exc) or len(set(scr)) != len(scr):
        raise LaunchRefused(f"--lr-selection {path}: screened {scr} != values {sorted(values)} + excluded {exc}")
    if want != sel or [float(m) for m in obj.get("tie_set") or []] != tie:
        raise LaunchRefused(f"--lr-selection {path}: the 06F §3 rule gives {want} (tie set {tie}), the file says "
                            f"{sel} ({obj.get('tie_set')})")
    return obj


def ext_selection_exists(path: Optional[Path] = None) -> list:
    """06F §3: the extended-screen selection files that exist (next to the given file, and the registered path)."""
    cands = [LR_SELECTION_EXT_PATH] + ([Path(path).parent / LR_SELECTION_EXT_NAME] if path is not None else [])
    return sorted({str(p) for p in cands if Path(p).is_file()})


# FIX-DELTA-01 F2 (REV-DELTA-01): a non-smoke MAIN run is bound to the REGISTERED selection file (never a copy / preview).
FL06F_SCREEN_SET = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)          # 06F §1: every one screened (an excluded run is screened)
FL06F_EDGE, FL06F_EDGE_NEXT = 16.0, 32.0                    # 06F §1: x32 screened iff x16 is the best of the rest
FIRST_SCREEN_NAME = "LR_SELECTION.json"
FIRST_SCREEN_PATH = V4 / "contracts" / "fl" / FIRST_SCREEN_NAME
FIRST_SCREEN_SHA256_PREFIX = "cdf6663d"                     # DL 181: contracts/fl/LR_SELECTION.json sha cdf6663d…


def _file_sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _same_path(a, b) -> bool:
    return os.path.normcase(os.path.realpath(str(a))) == os.path.normcase(os.path.realpath(str(b)))


MAIN_TUNING_STAGE = "S4"                                    # FIX-DELTA-02 M2: 06G §4 / 06I §3 / 06J §3


def registered_selection() -> Optional[tuple]:
    """(kind, path) of the registered MAIN selection: FL_TUNING_SELECTION.json (the FINAL, stage S4 file; 06G §4 / 06I §3
    / 06J §3) once it exists; None before (FIX-DELTA-02 M2: LR_SELECTION_EXT.json and the 06B file never qualify for
    MAIN; the stage is checked by bind_main_selection / check_tuning_for_main)."""
    if Path(FL_TUNING_PATH).is_file():
        return "fl_tuning", Path(FL_TUNING_PATH)
    return None


def _is_sha(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def check_ext_for_main(obj: Mapping, path, *, peak_lr: Optional[float] = None) -> None:
    """06F §1 on a verified (load_lr_selection) file, for a MAIN run: the EXT format (06B refused), a first-screen
    sha256 = the registered 06B file's, screened ⊇ {0.5, 1, 2, 4, 8, 16}, x32 screened iff x16 is the 06F rule's winner
    over the other screened values (Ops reading, DL 193), lr_peak_base == the SMALL recipe peak."""
    from v4int.lr_screen import SelectionRefused, apply_rule_ext
    if obj.get("format") != LR_SELECTION_EXT_FORMAT:
        raise LaunchRefused(f"{path}: a non-smoke MAIN run takes the {LR_SELECTION_EXT_FORMAT} file only (06F §1: the "
                            f"06B {LR_SELECTION_FORMAT} file is superseded), not {obj.get('format')!r}")
    fs = obj.get("first_screen_file_sha256")
    if not _is_sha(fs) or (obj.get("first_screen") or {}).get("sha256") != fs:
        raise LaunchRefused(f"{path}: first_screen_file_sha256 {fs!r} is missing / not the first_screen record's sha")
    if not Path(FIRST_SCREEN_PATH).is_file():
        raise LaunchRefused(f"{path}: the registered first-screen file {FIRST_SCREEN_PATH} is absent")
    reg = _file_sha(FIRST_SCREEN_PATH)
    if fs != reg or not reg.startswith(FIRST_SCREEN_SHA256_PREFIX):
        raise LaunchRefused(f"{path}: first_screen_file_sha256 {fs[:12]}… != the registered {FIRST_SCREEN_PATH.name} "
                            f"{reg[:12]}… (registered prefix {FIRST_SCREEN_SHA256_PREFIX})")
    scr = {float(m) for m in obj.get("screened") or []}
    missing = sorted(set(FL06F_SCREEN_SET) - scr)
    if missing:
        raise LaunchRefused(f"{path}: screened {sorted(scr)} lacks the 06F §1 multipliers {missing}")
    tag_to_m = {t: m for m, t in LR_MULT_TAG.items()}
    rest = {tag_to_m[t]: float(v) for t, v in (obj.get("values") or {}).items() if tag_to_m[t] != FL06F_EDGE_NEXT}
    try:
        win, _tie = apply_rule_ext(rest)
    except SelectionRefused as e:
        raise LaunchRefused(f"{path}: the 06F rule over the screened set without x32 fails ({e})") from e
    if (FL06F_EDGE_NEXT in scr) != (win == FL06F_EDGE):
        raise LaunchRefused(f"{path}: 06F §1 edge rule: without x32 the winner is x{LR_MULT_TAG[win]}, so x32 must "
                            f"{'' if win == FL06F_EDGE else 'not '}be screened (screened {sorted(scr)})")
    if peak_lr is not None and float(obj.get("lr_peak_base") or 0.0) != float(peak_lr):
        raise LaunchRefused(f"{path}: lr_peak_base {obj.get('lr_peak_base')} != the SMALL recipe peak {peak_lr}")


def check_tuning_for_main(obj: Mapping, path, *, peak_lr: Optional[float] = None) -> dict:
    """A verified (load_fl_tuning) stage file for a MAIN run: the FINAL stage S4 (FIX-DELTA-02 M2; S2 / S3 files are
    refused); its S1b history entry is the registered LR_SELECTION_EXT.json (sha256), which itself passes
    check_ext_for_main; m* = the EXT selection; lr_peak_base == the EXT's == the SMALL recipe peak. Returns the EXT
    binding."""
    if obj.get("stage") != MAIN_TUNING_STAGE:
        raise LaunchRefused(f"{path}: stage {obj.get('stage')!r}; a non-smoke MAIN FL run takes the FINAL (stage "
                            f"{MAIN_TUNING_STAGE}) FL_TUNING_SELECTION.json only (06G §4 / 06I §3 / 06J §3)")
    hist = list(obj.get("stage_history") or [])
    if not hist or hist[0].get("stage") != "S1b":
        raise LaunchRefused(f"{path}: the stage history does not start at the S1b (06F) selection")
    if not Path(LR_SELECTION_EXT_PATH).is_file():
        raise LaunchRefused(f"{path}: the registered {LR_SELECTION_EXT_PATH} (its S1b source) is absent")
    ext_sha = _file_sha(LR_SELECTION_EXT_PATH)
    if hist[0].get("sha256") != ext_sha:
        raise LaunchRefused(f"{path}: its S1b source sha {str(hist[0].get('sha256'))[:12]}… is not the registered "
                            f"{LR_SELECTION_EXT_NAME} {ext_sha[:12]}…")
    ext = load_lr_selection(LR_SELECTION_EXT_PATH)
    check_ext_for_main(ext, LR_SELECTION_EXT_PATH, peak_lr=peak_lr)
    if lr_mult_value(obj.get("m_star")) != lr_mult_value(ext["selected_lr_mult"]):
        raise LaunchRefused(f"{path}: m* {obj.get('m_star')} != the registered EXT selection "
                            f"{ext['selected_lr_mult']}")
    if float(obj.get("lr_peak_base") or 0.0) != float(ext.get("lr_peak_base") or -1.0) or (
            peak_lr is not None and float(obj["lr_peak_base"]) != float(peak_lr)):
        raise LaunchRefused(f"{path}: lr_peak_base {obj.get('lr_peak_base')} != the EXT's {ext.get('lr_peak_base')}"
                            f"{'' if peak_lr is None else f' / the SMALL recipe peak {peak_lr}'}")
    return {"path": str(LR_SELECTION_EXT_PATH), "sha256": ext_sha, "self_sha256": ext["self_sha256"],
            "first_screen_file_sha256": ext["first_screen_file_sha256"]}


def bind_main_selection(args, peak_lr: Optional[float] = None) -> Optional[dict]:
    """FIX-DELTA-01 F2 + FIX-DELTA-02 M2: a non-smoke MAIN run takes exactly the registered FINAL selection
    contracts/fl/FL_TUNING_SELECTION.json at stage S4 (--fl-tuning; the given path must resolve to it; LR_SELECTION_EXT
    / the 06B file / S2 / S3 stage files are refused), verified and bound by its file sha256 (hashed before and after
    verification), plus the 06F / 06G content checks (check_tuning_for_main -> check_ext_for_main on its S1b source).
    Returns the binding (None when not bound)."""
    if args.smoke or getattr(args, "role", "MAIN") != "MAIN":
        return None
    sel = getattr(args, "lr_selection", None)
    tuning = getattr(args, "fl_tuning", None)
    if tuning is not None and sel is not None:
        raise LaunchRefused("--fl-tuning and --lr-selection are exclusive")
    if sel is not None:
        raise LaunchRefused(f"--lr-selection {sel}: a non-smoke MAIN {args.arm} run takes the FINAL (stage "
                            f"{MAIN_TUNING_STAGE}) {FL_TUNING_PATH} via --fl-tuning (06G §4 / 06I §3 / 06J §3; "
                            "FIX-DELTA-02 M2: LR_SELECTION_EXT.json / the 06B file are screen inputs, never MAIN)")
    reg = registered_selection()
    if reg is None:
        raise LaunchRefused(f"a non-smoke MAIN {args.arm} run needs the registered FINAL selection {FL_TUNING_PATH} "
                            f"(stage {MAIN_TUNING_STAGE}; 06G §4 / 06I §3 / 06J §3), which does not exist "
                            "(LR_SELECTION_EXT.json / the 06B file never qualify for MAIN; --lr-mult is a smoke / "
                            "screen option)")
    kind, rpath = reg
    if tuning is None:
        raise LaunchRefused(f"a non-smoke MAIN {args.arm} run must use --fl-tuning {rpath} (the registered FINAL "
                            f"stage-{MAIN_TUNING_STAGE} selection)")
    if not _same_path(tuning, rpath):
        raise LaunchRefused(f"--fl-tuning {tuning} is not the registered {rpath}: a non-smoke MAIN run takes the "
                            "registered file only (a preview / copy never qualifies)")
    sha = _file_sha(rpath)
    obj = load_fl_tuning(rpath)
    ext = check_tuning_for_main(obj, rpath, peak_lr=peak_lr)
    if _file_sha(rpath) != sha:
        raise LaunchRefused(f"{rpath} changed while it was verified")
    return {"kind": kind, "path": str(rpath), "sha256": sha, "self_sha256": obj["self_sha256"],
            "format": obj["format"], "stage": obj["stage"], "ext": ext,
            "first_screen_file_sha256": ext["first_screen_file_sha256"],
            "lr_peak_base_checked": None if peak_lr is None else float(peak_lr),
            "authority": "FIX-DELTA-01 F2 + FIX-DELTA-02 M2"}


def require_lr_selection(args) -> None:
    """FL-07 item 5a (Ops decision on FL-05's open point): a non-smoke MAIN-role FL run of ANY arm (S_CAL included:
    S depends on the local LR) needs the verified selection; there is no silent m = 1 default. FIX-DELTA-02 M2: the
    FINAL (stage S4) FL_TUNING_SELECTION.json only.
    FIX-DELTA-01 F2 (supersedes the FL-09 / FL-10 sibling-file rules): the selection must be the latest REGISTERED file
    (bind_main_selection); the SMALL-peak check follows in resolve_spec, once the recipe is known."""
    bind_main_selection(args)


def resolve_tuning(args) -> tuple:
    """FL-10: (tuning point, provenance record) from --fl-tuning, else from --lr-mult / --lr-selection + the tuning
    options."""
    if getattr(args, "fl_tuning", None) is not None:
        obj = load_fl_tuning(args.fl_tuning)
        t = tuning_config(**{k: obj["winner"].get(k) for k in TUNING_KEYS})
        return t, {"source": "fl_tuning", "path": str(args.fl_tuning), "self_sha256": obj["self_sha256"],
                   "sha256": hashlib.sha256(Path(args.fl_tuning).read_bytes()).hexdigest(), "stage": obj["stage"],
                   "selected_lr_mult": t["lr_mult"], "winner": dict(t), "label": tuning_label(t),
                   "format": obj["format"]}
    m, prov = resolve_lr_mult(args)
    t = tuning_config(optimizer=getattr(args, "server_opt", None), server_lr=getattr(args, "server_lr", None),
                      server_tau=getattr(args, "server_tau", None), lr_mult=m, lr_shape=getattr(args, "lr_shape", None),
                      client_opt=getattr(args, "client_opt", None), client_lr=getattr(args, "client_lr", None),
                      local_passes=getattr(args, "local_passes", None), local_batch=getattr(args, "local_batch", None))
    return t, prov


def resolve_lr_mult(args) -> tuple:
    """(multiplier, provenance record) from --lr-mult / --lr-selection (default 1.0)."""
    if getattr(args, "lr_selection", None) is not None:
        obj = load_lr_selection(args.lr_selection)
        m = lr_mult_value(obj["selected_lr_mult"])
        return m, {"source": "lr_selection", "path": str(args.lr_selection), "self_sha256": obj["self_sha256"],
                   "sha256": _file_sha(args.lr_selection), "selected_lr_mult": m, "format": obj["format"]}
    if getattr(args, "lr_mult", None) is not None:
        m = lr_mult_value(args.lr_mult)
        return m, {"source": "--lr-mult", "selected_lr_mult": m}
    return 1.0, {"source": "default", "selected_lr_mult": 1.0}


def default_run_id(args) -> str:
    """FL_<tag>_SMALL_s<seed> (unchanged for MAIN); the CALIBRATION screen: FL_FA_LRSCREEN_x<m>_CALIB_s2026.
    P2-02: a P2 arm -> p2_run_id."""
    if is_p2_arm(args.arm):
        return p2_run_id(args)
    from v4int.smoke import SMOKE_RUN_PREFIX
    prefix = SMOKE_RUN_PREFIX if args.smoke else ""
    if getattr(args, "role", "MAIN") == "CALIBRATION":
        t = args_tuning(args)                              # FL-10: default-tuning ids unchanged
        fa = ""
        if t["optimizer"] == "fedadam":
            fa += f"FEDADAM_eta{num_tag(t['server_lr'])}_"
            if t["server_tau"] != FEDADAM_HYPER["tau"]:
                fa += f"tau{num_tag(t['server_tau'])}_"
        elif t["optimizer"] == "fedavgm":
            fa += "FEDAVGM_"
        if t["lr_shape"] == "fl_const":
            fa += "FLCONST_"
        if t["client_opt"] == "sgd_m":
            fa += f"SGDM_lr{num_tag(t['client_lr'])}_"
        if t["local_passes"] != KNOBS["passes"]:              # FL-13
            fa += f"P{t['local_passes']}_"
        if t["local_batch"] != KNOBS["batch"]:
            fa += f"B{t['local_batch']}_"
        tail = ""
        wf = getattr(args, "warm_lr_factor", None)            # FL-12
        if wf is not None and float(wf) != 1.0:
            tail = f"_wf{num_tag(wf)}"
        sf = getattr(args, "warm_server_lr_factor", None)     # FL-15 (06M_B B2)
        if sf is not None and float(sf) != 1.0:
            tail += f"_sf{num_tag(sf)}"
        return f"{prefix}FL_{args.arm}_LRSCREEN_{fa}x{LR_MULT_TAG[t['lr_mult']]}{tail}_CALIB_s{args.seed}"
    tag = args.arm if base_arm(args.arm) != "DP" else f"{args.arm}_eps{int(args.epsilon)}"   # DP_eps8 unchanged
    return f"{prefix}FL_{tag}_SMALL_s{args.seed}"


def role_evaluator(tr, rep, key_base: Mapping, overrides: Mapping) -> Callable:
    """evaluate(mark) on another role's evaluation manifest (CALIBRATION: CALIB_VALIDATION) through the harness
    trainer's frozen T3 path — exactly CentralTrainer._evaluate (predict_adapter -> evaluate -> rows_from_result +
    the harness row labels) with the provider, key base and the role's identity fields replaced."""
    from fl.numerics import state_digest
    from v4eval.evaluate import evaluate
    from v4eval.frozen_config import apply_runtime_flags, check_runtime_flags
    from v4eval.predict import predict_adapter
    from v4eval.rows import rows_from_result
    from v4eval.values import Undefined

    def evaluate_role(mark: float):
        m, fz = rep.manifest, tr.frozen
        model_sha = state_digest(tr.adapter.broadcast_state())
        apply_runtime_flags(fz)
        try:
            check_runtime_flags(fz, tr.device)
            art = predict_adapter(tr.adapter, rep.batches(fz.eval_batch), m, config=fz, model_sha256=model_sha)
            res = evaluate(m, art, config=fz, evaluator_code_sha256=tr.evaluator_sha)
        finally:
            tr._apply_determinism()
        v = res["E2E"]["macro"]["mrr@20"]
        if isinstance(v, Undefined):
            raise RuntimeError(f"primary metric UNDEFINED on {m.manifest_id}: {v}")
        rows = rows_from_result(res, key_base, method="C", seed=tr.cfg.seed, selection_rule=None,
                                efe=tr.state.exposures / tr.N)
        out = []
        for r in rows:
            r = dict(r)
            r.update(tr._row_labels())
            r.update(overrides)
            out.append(r)
        return float(v), out, art.ranks, res
    return evaluate_role


def _small_recipe(args) -> dict:
    """The SMALL recipe (hash-checked; production path for a non-smoke run) through v4int.small_recipe."""
    from v4int import small_recipe as SR
    import v4train.tuned_recipes as TRM
    SR.install()
    path = Path(args.small_recipe) if args.small_recipe else SR.RECIPE_PATH
    try:
        reg = TRM.load_tuned_recipes(path)
        pin = None if args.smoke else TRM.check_production_registry(path, reg)
        r = TRM.recipe(reg, SR.RECIPE_ID)
    except TRM.TunedRecipeError as e:
        raise LaunchRefused(f"SMALL recipe refused: {e}") from e
    if r["family"] != FAMILY or r["model_variant"] != SR.SMALL_VARIANT:
        raise LaunchRefused(f"the SMALL recipe is {r['family']} / {r['model_variant']}, not {FAMILY} / "
                            f"{SR.SMALL_VARIANT}")
    return {"recipe_id": r["recipe_id"], "recipe_sha256": r["sha256"], "recipe_registry": str(path),
            "recipe_registry_self_sha256": reg["self_sha256"], "production_pin": pin, "peak_lr": float(r["peak_lr"]),
            "dropout": list(r["dropout"]), "schedule": r["schedule"], "model_variant": r["model_variant"]}


def tuning_kwargs(t: Mapping) -> dict:
    """arm_spec keyword arguments of a tuning point."""
    adam = t["optimizer"] == "fedadam"
    return {"lr_mult": t["lr_mult"], "server_opt": t["optimizer"], "server_lr": t["server_lr"] if adam else None,
            "server_tau": t["server_tau"] if adam else None, "lr_shape": t["lr_shape"], "client_opt": t["client_opt"],
            "client_lr": t["client_lr"], "local_passes": t.get("local_passes"), "local_batch": t.get("local_batch")}


def resolve_spec(args, peak_lr: float, fused: Optional[bool], init_identity: Optional[Mapping] = None) -> tuple:
    if is_p2_arm(args.arm):                                # P2-02: recipe-driven (the Phase-1 path below unchanged)
        return p2_resolve_spec(args, peak_lr, init_identity)
    extra: dict = {}
    clip = z = None
    fl05_preflight(args)
    binding = None
    if not getattr(args, "dry_run", False):                # FIX-DELTA-01 F2 (FL-07: a dry run is not refused)
        binding = bind_main_selection(args, peak_lr=peak_lr)
        if binding is not None:
            extra["main_selection"] = binding
    wf = getattr(args, "warm_lr_factor", None)
    wsf = getattr(args, "warm_server_lr_factor", None)     # FL-15 (06M_B B2)
    if is_warm_arm(args.arm) and init_identity is None:
        raise LaunchRefused(f"{args.arm}: the theta_pre identity (load_pretrain_init) is required")
    if getattr(args, "warm_selection", None) is not None:  # FL-12: non-smoke MAIN warm run
        if getattr(args, "dry_run", False):                # REV-FL-15 m4: the dry run runs the SAME binding checks
            wbind = bind_warm_selection(args, init_identity, dry_run_tuning_binding(args))
            wf, wsf = wbind["winner_factor"], wbind["winner_server_lr_factor"]
            extra["warm_selection"] = dict(wbind, dry_run=True)
        else:                                              # FL-15: BOTH the local factor and the server LR
            wbind = bind_warm_selection(args, init_identity, binding)
            wf, wsf = wbind["winner_factor"], wbind["winner_server_lr_factor"]
            extra["warm_selection"] = wbind
    tun, extra["lr_mult_source"] = resolve_tuning(args)
    prule = None
    if extra["lr_mult_source"]["source"] == "fl_tuning":
        extra["fl_tuning"] = extra["lr_mult_source"]
        tun, prule, plaunch = poisson_tuning(args.arm, tun)  # FIX-DELTA-02 M1 (DL 219): P = 2 / B = 16 for Poisson
        if plaunch is not None:
            extra["poisson_local_rule"] = plaunch
    role = getattr(args, "role", "MAIN")
    ca = base_arm(args.arm)
    if ca in ("FA_1024", "DP") and is_warm_arm(args.arm):  # FL-16 (06N N3.2): the warm chain's own S record only
        scal = warm_s_cal_spec(args.arm, peak_lr=peak_lr, fused=fused, init_from=init_identity, tun=tun,
                               warm_lr_factor=wf, warm_server_lr_factor=wsf, poisson_local_rule=prule)
        s = load_warm_s_record(args.clip_record, arm=args.arm, scal_spec=scal, init_identity=init_identity,
                               smoke=bool(args.smoke))
        clip = s["S"]
        extra["s_record"] = {"path": str(args.clip_record), "S": clip, "self_sha256": s["self_sha256"],
                             "file_sha256": _file_sha(args.clip_record), "authority": FL16_AUTHORITY,
                             **{k: s.get(k) for k in ("format", "arm", "run_id", "config_digest", "arm_spec_sha256",
                                                      "plan_digest", "init_sha256", "label")}}
    elif ca in ("FA_1024", "DP"):
        s = load_s_record(args.clip_record, smoke=bool(args.smoke))
        clip = s["S"]
        extra["s_record"] = {"path": str(args.clip_record), "S": clip, "self_sha256": s["self_sha256"]}
    if ca == "DP":
        zr = load_z(args.epsilon, args.dp_z_record)
        z = zr["z"]
        extra["dp_z"] = zr
    return arm_spec(args.arm, seed=int(args.seed), peak_lr=peak_lr, clip_norm=clip, z=z, fused=fused,
                    role=role, init_from=init_identity if is_warm_arm(args.arm) else None, warm_lr_factor=wf,
                    poisson_local_rule=prule, warm_server_lr_factor=wsf, **tuning_kwargs(tun)), extra


def load_warm_selection(path: Path) -> dict:
    """FL-12: a verified PPSI_V4_FL_WARM_SELECTION_1 file of `lr_screen select-warm` (format, self-sha256, stage / arm,
    the registered factors, the 06K rule recomputed from its values: best; ties |delta| <= 0.0005 -> factor 1).
    FL-15 (06M_B B2): also the 2-D PPSI_V4_FL_WARM_SELECTION_2 file (load_warm_selection_2d). The 1-D file still
    verifies here (it is a record of what was screened) but never configures a run (warm_selection_winner refuses it)."""
    from v4int.lr_screen import SelectionRefused, apply_rule_warm
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LaunchRefused(f"--warm-selection {path} unreadable ({e})") from e
    if obj.get("format") == WARM_SELECTION_FORMAT_2D:          # FL-15
        return load_warm_selection_2d(obj, path)
    if obj.get("format") != WARM_SELECTION_FORMAT or obj.get("self_sha256") != _canon(obj):
        raise LaunchRefused(f"--warm-selection {path} is not a verified {WARM_SELECTION_FORMAT} object")
    arm, stage = obj.get("arm"), obj.get("stage")
    if arm not in WARM_ARMS or WARM_STAGE[arm] != stage:
        raise LaunchRefused(f"--warm-selection {path}: stage {stage} / arm {arm} is not a registered warm stage")
    try:
        vals = {float(k): float(v) for k, v in (obj.get("values") or {}).items()}
        win, tie = apply_rule_warm(arm, vals)
    except (SelectionRefused, TypeError, ValueError) as e:
        raise LaunchRefused(f"--warm-selection {path}: invalid ({e})") from e
    if sorted(float(f) for f in obj.get("screened") or []) != sorted(WARM_FACTORS[arm]):
        raise LaunchRefused(f"--warm-selection {path}: screened factors != {WARM_FACTORS[arm]}")
    if float(obj.get("winner_factor", -1)) != win or [float(f) for f in obj.get("tie_set") or []] != tie:
        raise LaunchRefused(f"--warm-selection {path}: the 06K rule gives {win} ({tie}), the file says "
                            f"{obj.get('winner_factor')} ({obj.get('tie_set')})")
    check_init_identity(obj.get("init_from"))
    return obj


WARM_2D_DECISION_KEYS = ("tie_set", "tie_band_set", "best_candidate", "gain", "no_measurable_gain", "min_gain",
                         "incumbent")


def load_warm_selection_2d(obj: Mapping, path) -> dict:
    """FL-15 (06M_B B2): a verified PPSI_V4_FL_WARM_SELECTION_2 object: self-sha256, a registered stage / arm, the
    server axis of its server optimiser (FedAdam {1, 0.1}; FedAvgM / FedAvg {1, 0.25}) and eta*, the screened cells =
    the registered 4-cell grid (values + excluded), and the 06J rule restricted to these runs (lr_screen.warm_decision:
    0.0005 band, >= 0.002 over the incumbent (m*, eta*), ties -> (m*, eta*)) recomputed: winner, tie set, band,
    gain, best candidate, no_measurable_gain, min_gain, incumbent as recorded."""
    from v4int.lr_screen import MIN_GAIN, SelectionRefused, warm_decision
    if obj.get("self_sha256") != _canon(obj):
        raise LaunchRefused(f"--warm-selection {path} is not a verified {WARM_SELECTION_FORMAT_2D} object")
    arm, stage = obj.get("arm"), obj.get("stage")
    if arm not in WARM_ARMS or WARM_STAGE[arm] != stage:
        raise LaunchRefused(f"--warm-selection {path}: stage {stage} / arm {arm} is not a registered warm stage")
    try:
        ax = obj.get("server_axis") or {}
        opt = ax.get("optimizer")
        if opt not in WARM_SERVER_FACTORS or [float(f) for f in ax.get("factors") or []] != list(WARM_SERVER_FACTORS[opt]):
            raise SelectionRefused(f"server axis {ax} is not the registered 06M_B B2 axis of {opt}")
        eta = float(ax["eta_star"])
        eta_ok = (eta == 1.0 if opt == "fedavg" else eta == FEDAVGM_LR if opt == "fedavgm"
                  else eta in FEDADAM_LRS + FEDADAM_EDGE_LRS)
        if not eta_ok:
            raise SelectionRefused(f"eta* {eta} is not a registered server LR of {opt}")
        want_lrs = {num_tag(f): (eta if f == 1.0 else eta / WARM_SERVER_DIVISORS[f]) for f in WARM_SERVER_FACTORS[opt]}
        if {k: float(v) for k, v in (ax.get("server_lrs") or {}).items()} != want_lrs:
            raise SelectionRefused(f"server_lrs {ax.get('server_lrs')} != {want_lrs}")
        if str((obj.get("tuning_selection") or {}).get("winner", "")).split("|")[0] != opt:
            raise SelectionRefused(f"the tuning selection winner {obj.get('tuning_selection')} is not a {opt} point")
        grid = warm_grid(arm, opt)
        lab = {warm_cell_label(*c): c for c in grid}
        vals = {lab[k]: float(v) for k, v in (obj.get("values") or {}).items()}
        exc = [lab[e["label"]] for e in obj.get("excluded") or []]
        scr = [lab[k] for k in obj.get("screened") or []]
        if sorted(scr) != sorted(grid) or sorted(list(vals) + exc) != sorted(grid):
            raise SelectionRefused(f"screened {obj.get('screened')} / values + excluded != the 4 registered cells "
                                   f"{sorted(lab)}")
        dec = warm_decision(arm, opt, vals)
        for k in WARM_2D_DECISION_KEYS:
            if k not in obj or obj[k] != dec[k]:
                raise SelectionRefused(f"{k} {obj.get(k, '<absent>')!r} != the 06J decision {dec[k]!r}")
        if dec["min_gain"] != MIN_GAIN:
            raise SelectionRefused(f"min_gain {dec['min_gain']} is not the registered {MIN_GAIN}")
        w = obj.get("winner") or {}
        want_w = {"warm_lr_factor": dec["winner"][0], "warm_server_lr_factor": dec["winner"][1],
                  "label": warm_cell_label(*dec["winner"]), "server_lr": want_lrs[num_tag(dec["winner"][1])]}
        if w != want_w or float(obj.get("winner_factor", -1)) != dec["winner"][0] or \
                float(obj.get("winner_server_lr_factor", -1)) != dec["winner"][1]:
            raise SelectionRefused(f"winner {w} / {obj.get('winner_factor')} / {obj.get('winner_server_lr_factor')} "
                                   f"!= the 06J decision {want_w}")
    except (SelectionRefused, KeyError, TypeError, ValueError) as e:
        raise LaunchRefused(f"--warm-selection {path}: invalid 2-D warm selection ({e})") from e
    check_init_identity(obj.get("init_from"))
    return dict(obj)


def warm_selection_winner(obj: Mapping, path) -> tuple:
    """FL-15 (06M_B B2): (local factor, server-LR factor) of a verified warm selection that may configure a run: the
    2-D file only (the 1-D PPSI_V4_FL_WARM_SELECTION_1 file of the superseded {m*, m*/4} screen is refused)."""
    if obj.get("format") != WARM_SELECTION_FORMAT_2D:
        raise LaunchRefused(f"{path}: a {obj.get('format')} (1-D local-factor) warm selection; the registered S7 / S8 "
                            f"grid is 2-D (local factor x server LR, PREG1_AMENDMENT_06M_B B2): a warm run takes the "
                            f"{WARM_SELECTION_FORMAT_2D} file only")
    return float(obj["winner_factor"]), float(obj["winner_server_lr_factor"])


def dry_run_tuning_binding(args) -> dict:
    """REV-FL-15 m4: the tuning binding a MAIN warm --dry-run checks the warm selection against (bind_warm_selection:
    tuning sha256 + server axis vs the winner), i.e. the --fl-tuning file the real launch binds (the dry run skips only
    the registered-path check of bind_main_selection, as FL-07 allows); refused without --fl-tuning."""
    tpath = getattr(args, "fl_tuning", None)
    if tpath is None or not Path(tpath).is_file():
        raise LaunchRefused(f"--warm-selection needs the --fl-tuning file it is bound to (got {tpath}); a MAIN warm dry "
                            "run checks the same binding as the real launch (REV-FL-15 m4)")
    return {"kind": "fl_tuning", "path": str(tpath), "sha256": _file_sha(tpath), "dry_run": True}


def check_warm_server_axis(obj: Mapping, tw: Mapping, path) -> None:
    """FL-15 (06M_B B2): the 2-D warm file's server axis (optimiser, eta*) and tuning winner label are the bound tuning
    selection's winner tw (so the server-LR factor scales the eta* this MAIN run actually uses)."""
    ax = obj.get("server_axis") or {}
    if not ax:
        raise LaunchRefused(f"{path}: no server axis (a 1-D warm selection); the registered grid is 2-D (06M_B B2)")
    eta = 1.0 if tw["optimizer"] == "fedavg" else float(tw["server_lr"])
    if ax["optimizer"] != tw["optimizer"] or float(ax["eta_star"]) != eta or \
            (obj.get("tuning_selection") or {}).get("winner") != tuning_label(tw):
        raise LaunchRefused(f"{path}: server axis {ax['optimizer']} / eta* {ax['eta_star']} / winner "
                            f"{(obj.get('tuning_selection') or {}).get('winner')} is not the bound tuning winner "
                            f"{tuning_label(tw)}")


def bind_warm_selection(args, init_identity: Optional[Mapping], main_binding: Optional[Mapping]) -> dict:
    """FL-12: a non-smoke MAIN warm run takes its LR factor from the REGISTERED warm selection of its stage only: the
    given path must resolve to contracts/fl/FL_WARM_SELECTION_<S7|S8>.json, whose tuning selection sha256 = the
    registered selection this run is bound to (bind_main_selection) and whose theta_pre = this run's --init-from.
    FL-15 (06M_B B2): the file must be the 2-D PPSI_V4_FL_WARM_SELECTION_2 (the 1-D file is refused), its server axis
    must be the bound tuning file's winner (optimiser, eta*), and the run takes BOTH its local factor and its server-LR
    factor (winner_factor, winner_server_lr_factor)."""
    variant = warm_variant(args.arm)                       # FL-16: a warm-line arm binds its variant's file
    stage = WARM_STAGE.get(variant)
    if stage is None:
        raise LaunchRefused("--warm-selection is a FA_WARM / FA_WARM_FROZEN option")
    rpath = WARM_SELECTION_PATH[stage]
    if not Path(rpath).is_file() or not _same_path(args.warm_selection, rpath):
        raise LaunchRefused(f"--warm-selection {args.warm_selection} is not the registered {rpath} (a preview / copy "
                            "never qualifies)")
    sha = _file_sha(rpath)
    obj = load_warm_selection(rpath)
    wf, wsf = warm_selection_winner(obj, rpath)                # FL-15: the 1-D file never configures a MAIN run
    if obj["arm"] != variant:
        raise LaunchRefused(f"{rpath} selects for {obj['arm']}, not {args.arm} (variant {variant})")
    ts = obj.get("tuning_selection") or {}
    if main_binding is None or ts.get("sha256") != main_binding.get("sha256"):
        raise LaunchRefused(f"{rpath}: its tuning selection sha {str(ts.get('sha256'))[:12]}… is not the registered "
                            f"selection this run is bound to ({str((main_binding or {}).get('sha256'))[:12]}…)")
    tpath = main_binding.get("path")                           # FL-15: the server axis = the bound winner's
    if not tpath or not Path(tpath).is_file() or _file_sha(tpath) != main_binding["sha256"]:
        raise LaunchRefused(f"{rpath}: the bound tuning selection {tpath} is absent / not the bound sha256")
    tw = tuning_config(**{k: load_fl_tuning(tpath)["winner"].get(k) for k in TUNING_KEYS})
    check_warm_server_axis(obj, tw, rpath)
    if init_identity is None or dict(obj["init_from"]) != dict(init_identity):
        raise LaunchRefused(f"{rpath}: its theta_pre identity differs from --init-from")
    if _file_sha(rpath) != sha:
        raise LaunchRefused(f"{rpath} changed while it was verified")
    return {"path": str(rpath), "sha256": sha, "self_sha256": obj["self_sha256"], "stage": stage,
            "format": obj["format"], "winner_factor": wf, "winner_server_lr_factor": wsf,
            "winner_server_lr": (obj.get("winner") or {}).get("server_lr"), "server_optimizer": tw["optimizer"],
            "authority": "PREG1_AMENDMENT_06K §2-§3 + 06M_B B2"}


# ------------------------------------------------------------------------------------------------ P2-02 (PREG2 §5)
def p2_server_lrs(opt: str) -> tuple:
    """P2-02: every admitted server LR of a server point (WA core / ext / incumbent / §5.7 steps), sorted."""
    return tuple(sorted({float(v) for vs in P2_SERVER_LRS[opt].values() for v in vs}))


def p2_client_lrs() -> tuple:
    return tuple(sorted({float(v) for vs in P2_CLIENT_LRS.values() for v in vs}))


def p2_recipe_value(key: str, v):
    """P2-02: one normalised P2 recipe value (server_lr is checked per server point in check_p2_recipe)."""
    if key == "server_opt":
        o = str(v).lower()
        if o not in P2_SERVER_LRS:
            raise LaunchRefused(f"P2 server_opt must be one of {tuple(P2_SERVER_LRS)}, not {v!r}")
        return o
    if key in ("server_lr", "client_lr", "item_lr_mult", "lora_lr_factor"):
        if isinstance(v, bool):
            raise LaunchRefused(f"P2 {key} {v!r} is not a number")
        x = _num(v, f"P2 {key}")
        if not math.isfinite(x) or x <= 0:
            raise LaunchRefused(f"P2 {key} must be a finite number > 0, not {v!r}")
        if key == "client_lr" and x not in p2_client_lrs():
            raise LaunchRefused(f"P2 client_lr must be one of {p2_client_lrs()} (PREG2 §5.2 / §5.7), not {v}")
        if key == "item_lr_mult" and x not in P2_ITEM_LR_MULTS:
            raise LaunchRefused(f"P2 item_lr_mult must be one of {P2_ITEM_LR_MULTS} (PREG2 §5.3 / §5.7), not {v}")
        return x
    if key == "items":
        if v not in ("frozen", "unfrozen"):
            raise LaunchRefused(f"P2 items must be 'frozen' or 'unfrozen', not {v!r}")
        return v
    if key == "upload":
        if v not in P2_UPLOADS:
            raise LaunchRefused(f"P2 upload {v!r} is not installed in this launcher (admitted {P2_UPLOADS}; the WB3 "
                                "delta_rows channel is work packet P2-03)")
        return v
    if key == "personal_head":
        if v not in (True, False):
            raise LaunchRefused(f"P2 personal_head must be a boolean, not {v!r}")
        return bool(v)
    raise LaunchRefused(f"unknown P2 recipe key {key!r}")


def p2_label(r: Mapping) -> str:
    """'fedadam|eta=0.0003|c=0.01', '...|items=unfrozen|ilr=0.1', '...|up=delta_rows', '...|lora=r4|lrx=10', '...|pf'
    (= v4int.p2_select.p2_label, the selection tool's label of the same recipe)."""
    tag = (lambda x: format(float(x), "g"))  # noqa: E731
    s = f"{r['server_opt']}|eta={tag(r['server_lr'])}|c={tag(r['client_lr'])}"
    if r["items"] == "unfrozen":
        s += f"|items=unfrozen|ilr={tag(r['item_lr_mult'])}"
    if r["upload"] != "dense":
        s += f"|up={r['upload']}"
    if r.get("lora_rank") is not None:
        s += f"|lora=r{int(r['lora_rank'])}|lrx={tag(r['lora_lr_factor'])}"
    if r["personal_head"]:
        s += "|pf"
    return s


def check_p2_recipe(raw, where: str = "P2 recipe") -> dict:
    """P2-02: the normalised recipe (exactly P2_RECIPE_KEYS, the v4int.p2_select layout) of a raw recipe mapping; every
    value registered (PREG2 §5.2-§5.4 / §5.7 axes), structural combinations checked; LoRA (WC1-WC4) is refused until
    work packet P2-04 installs it (hook: p2_pre_server)."""
    if not isinstance(raw, Mapping):
        raise LaunchRefused(f"{where}: not a recipe object")
    unknown = sorted(set(raw) - set(P2_RECIPE_KEYS))
    missing = sorted({"server_opt", "server_lr", "client_lr"} - set(raw))
    if unknown or missing:
        raise LaunchRefused(f"{where}: unknown keys {unknown} / missing keys {missing} (keys {P2_RECIPE_KEYS})")
    opt = p2_recipe_value("server_opt", raw["server_opt"])
    slr = p2_recipe_value("server_lr", raw["server_lr"])
    if slr not in p2_server_lrs(opt):
        raise LaunchRefused(f"{where}: {opt} server_lr must be one of {p2_server_lrs(opt)} (PREG2 §5.2 / §5.7), not "
                            f"{raw['server_lr']}")
    r = {"server_opt": opt, "server_lr": slr, "client_lr": p2_recipe_value("client_lr", raw["client_lr"]),
         "items": p2_recipe_value("items", raw.get("items", "frozen")), "item_lr_mult": None,
         "upload": p2_recipe_value("upload", raw.get("upload", "dense")), "lora_rank": None, "lora_lr_factor": None,
         "personal_head": p2_recipe_value("personal_head", raw.get("personal_head", False))}
    if r["items"] == "unfrozen":
        if raw.get("item_lr_mult") is None:
            raise LaunchRefused(f"{where}: unfrozen item tables need item_lr_mult {P2_ITEM_LR_MULTS} (PREG2 §5.3)")
        r["item_lr_mult"] = p2_recipe_value("item_lr_mult", raw["item_lr_mult"])
    elif raw.get("item_lr_mult") is not None:
        raise LaunchRefused(f"{where}: item_lr_mult needs unfrozen item tables (PREG2 §5.3)")
    if raw.get("lora_rank") is not None or raw.get("lora_lr_factor") is not None:
        raise LaunchRefused(f"{where}: LoRA cells (WC1-WC4) are not installed in this launcher (work packet P2-04)")
    if r["personal_head"] and r["items"] != "frozen":
        raise LaunchRefused(f"{where}: the personalised head (WC5) runs at A*'s frozen-table recipe (PREG2 §5.4)")
    return r


def load_p2_recipe(path: Path) -> tuple:
    """P2-02: (normalised recipe, source record) of a --p2-recipe file: {"format": PPSI_V4_P2_RECIPE_1, "label":
    p2_label(recipe), "recipe": {...}} (no other key); the label must be the recipe's own."""
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LaunchRefused(f"--p2-recipe {path} unreadable ({e})") from e
    if not isinstance(obj, dict) or obj.get("format") != P2_RECIPE_FORMAT or sorted(obj) != ["format", "label", "recipe"]:
        raise LaunchRefused(f"--p2-recipe {path} is not a {P2_RECIPE_FORMAT} object {{format, label, recipe}}")
    rec = check_p2_recipe(obj["recipe"], f"--p2-recipe {path}")
    if obj["label"] != p2_label(rec):
        raise LaunchRefused(f"--p2-recipe {path}: label {obj['label']!r} != the recipe's {p2_label(rec)!r}")
    return rec, {"source": "p2_recipe", "path": str(path), "sha256": _file_sha(path), "label": p2_label(rec),
                 "format": P2_RECIPE_FORMAT}


def p2_recipe_file_obj(rec: Mapping) -> dict:
    """The --p2-recipe file content of a recipe (tests / Ops helpers)."""
    r = check_p2_recipe(rec)
    return {"format": P2_RECIPE_FORMAT, "label": p2_label(r), "recipe": r}


def p2_wa_recipes() -> list:
    """P2-02: the 21 registered WA recipes (18 core + 2 WA-ext + the incumbent; PREG2 §5.2), frozen tables, dense."""
    out = [check_p2_recipe({"server_opt": o, "server_lr": e, "client_lr": c})
           for o, e in P2_WA_CELLS["core_server"] for c in P2_WA_CELLS["core_client"]]
    out += [check_p2_recipe({"server_opt": o, "server_lr": e, "client_lr": P2_WA_CELLS["ext_client"]})
            for o, e in P2_WA_CELLS["ext"]]
    o, e, c = P2_WA_CELLS["incumbent"]
    return out + [check_p2_recipe({"server_opt": o, "server_lr": e, "client_lr": c})]


def p2_server_opt(rec: Mapping) -> Optional[dict]:
    """P2-02: the cfg server_opt of a recipe (None = plain FedAvg, server LR 1): FedAvg eta_s != 1 -> FedAvgM beta 0 at LR
    eta_s, i.e. theta + eta_s (aggregate - theta) (the warm_server_opt path); FedAvgM beta 0.9 at eta_s; FedAdam beta1 0.9,
    beta2 0.99, tau 1e-3, v_0 = tau^2 at eta_s (fl.server)."""
    opt, lr = rec["server_opt"], float(rec["server_lr"])
    if opt == "fedavg":
        return None if lr == 1.0 else {"name": "fedavgm", "lr": lr, "beta1": 0.0}
    if opt == "fedavgm":
        return {"name": "fedavgm", "lr": lr, **FEDAVGM_HYPER}
    return {"name": "fedadam", "lr": lr, **FEDADAM_HYPER}


def p2_arm_spec(arm: str, *, recipe: Mapping, seed: int, peak_lr: float, role: str = "MAIN",
                init_from: Optional[Mapping] = None) -> dict:
    """P2-02: the arm spec of a P2 warm-FL run (PREG2 §5.1 fixed parts + the recipe): theta_pre start (init_from, the
    unchanged identity rule), SASREC_D64_B2, method FA (PF for personal_head, personal LR = the SMALL peak x the round
    factor, the registered PF rule), 64 clients / seeded sweep / 10 % drop-out, client SGD-m 0.9 at the recipe's client
    LR (clip 1.0, no weight decay, fresh per visit), 1 local pass / batch 16, FL_CONST, 6.0 EFE (CALIBRATION 3,414 /
    MAIN 13,654 rounds), the recipe's server step, item tables frozen unless the recipe unfreezes them (then the item-LR
    multiplier, fl.client.ItemLRSolver). The cfg has exactly the Phase-1 cfg layout (so the incumbent recipe's cfg is
    S8Tc's cfg)."""
    if arm not in P2_ARMS:
        raise LaunchRefused(f"p2_arm_spec builds {P2_ARMS}, not {arm}")
    rec = check_p2_recipe(recipe)
    if init_from is None:
        raise LaunchRefused(f"{arm}: a P2 arm starts from theta_pre and needs the verified init_from (--init-from)")
    check_init_identity(init_from)                           # the identity rule is unchanged: theta_pre only
    if role not in ROLES:
        raise LaunchRefused(f"--role must be one of {ROLES}, not {role}")
    if role == "CALIBRATION" and int(seed) != CALIB["seed"]:
        raise LaunchRefused("P2 CALIBRATION screens run at seed 2026 only (PREG2 §5.1)")
    n = CALIB["n_clients"] if role == "CALIBRATION" else MAIN_N_CLIENTS
    eb = exposure_budget(n, P2_FIXED["local_passes"])
    c = float(rec["client_lr"])
    solver = {"lr": c, "batch_size": P2_FIXED["local_batch"], "passes": P2_FIXED["local_passes"],
              "clip": P2_FIXED["clip"], "weight_decay": P2_FIXED["weight_decay"], "fused": None, "optimizer": "sgd_m"}
    if rec["item_lr_mult"] is not None:                      # WB: fl.client.ItemLRSolver (digest carries it)
        solver["item_lr_mult"] = float(rec["item_lr_mult"])
    cfg = {"method": "PF" if rec["personal_head"] else "FA", "seed": int(seed), "lr_peak_local": c, "solver": solver,
           "n_shards": KNOBS["n_shards"], "max_attempts": KNOBS["max_attempts"], "endpoint_efe": KNOBS["endpoint_efe"],
           "exposure_point": EXPOSURE_POINT, "mu": 0.0, "pf": None, "dp": None, "dropout_p": KNOBS["dropout_p"],
           "endpoint_rounds": eb["rounds"], "lr_shape": P2_FIXED["lr_shape"]}
    if rec["personal_head"]:                                 # WC5: the registered PF personal component
        cfg["pf"] = {"lr": float(peak_lr), "lam": KNOBS["pf_lam"], "clip": KNOBS["pf_clip"]}
    so = p2_server_opt(rec)
    if so is not None:
        cfg["server_opt"] = so
    spec = {"arm": arm, "study": "PREG2", "plan": {"group_size": KNOBS["group"], "sampling": "sweep"}, "cfg": cfg,
            "max_rounds": eb["rounds"], "lr_mult": 1.0, "lr_peak_base": float(peak_lr), "role": role,
            "upload_codec": None, "init_from": dict(init_from), "p2_recipe": rec, "p2_label": p2_label(rec),
            "p2_fixed": dict(P2_FIXED), "exposure_budget": eb}
    if rec["items"] == "frozen":                             # the deployed S8 structure (fl.runtime.freeze_item_tables)
        spec["frozen_item_tables"] = list(FROZEN_ITEM_KEYS)
    return spec


def check_p2_spec(spec: Mapping) -> None:
    """P2-02: a P2 spec is exactly what p2_arm_spec writes for its own recipe / seed / role / theta_pre (refused
    otherwise, so a hand-edited ARM_SPEC never runs)."""
    try:
        want = p2_arm_spec(spec["arm"], recipe=spec["p2_recipe"], seed=spec["cfg"]["seed"], peak_lr=spec["lr_peak_base"],
                           role=spec.get("role", "MAIN"), init_from=spec.get("init_from"))
    except (KeyError, TypeError) as e:
        raise LaunchRefused(f"P2 spec malformed ({e})") from e
    if json.dumps(dict(spec), sort_keys=True, default=str) != json.dumps(want, sort_keys=True, default=str):
        raise LaunchRefused(f"{spec.get('arm')}: the spec is not p2_arm_spec's spec of its recipe {spec.get('p2_label')}")


def p2_row_labels(spec: Mapping) -> dict:
    """P2-02: row fields of a P2 run (study PREG2; never a Phase-1 DOWNSTREAM_RETENTION row)."""
    r = spec["p2_recipe"]
    out = {"study": "PREG2", "row_status": "P2_ARM", "comparison_status": "P2_FOLLOWUP", "p2_recipe_label": p2_label(r),
           "method_code": spec["cfg"]["method"], "fl_server_opt": r["server_opt"], "fl_server_lr": float(r["server_lr"]),
           "fl_client_opt": "sgd_m", "fl_client_lr": float(r["client_lr"]), "fl_lr_shape": P2_FIXED["lr_shape"],
           "fl_local_passes": P2_FIXED["local_passes"], "p2_items": r["items"]}
    if r["item_lr_mult"] is not None:
        out["p2_item_lr_mult"] = float(r["item_lr_mult"])
    return out


def p2_run_labels(spec: Mapping) -> dict:
    """P2-02: the row fields execute writes for a P2 run (fl_rows `tuning_labels`, applied after fl_rows' own labels and
    before the CALIBRATION relabel): warm_row_labels(spec) + p2_row_labels(spec). The ONE function a P2 run reader
    (v4int.p2_select) should compare rows against (REV-P2-06 M2)."""
    return dict(warm_row_labels(spec), **p2_row_labels(spec))


def is_q8_codec(codec) -> bool:
    """REV-P2-03_04 (a): the FA_Q8 codec is recognised by its NAME only (a P2-03 sparse codec never installs Q8)."""
    return isinstance(codec, Mapping) and codec.get("name") == Q8_CODEC["name"]


def p2_preflight(args) -> None:
    """P2-02 refusals (torch-free) of a P2 arm: every Phase-1 tuning / warm / DP option refused (P2_REFUSED_FLAGS,
    --freeze-item-tables: the recipe decides), --init-from required (theta_pre), CALIBRATION at seed 2026, exactly one
    of --p2-recipe (smoke / CALIBRATION / dry run) and --p2-selection (MAIN only), and a non-smoke MAIN run refused
    without the registered selection; the recipe file is verified here."""
    arm = args.arm
    if not is_p2_arm(arm):
        raise LaunchRefused(f"p2_preflight: {arm} is not a P2 arm")
    given = [o for o in P2_REFUSED_FLAGS if getattr(args, o, None) is not None]
    if given:
        raise LaunchRefused(f"--arm {arm}: --{given[0].replace('_', '-')} is a Phase-1 option (the P2 recipe fixes the "
                            "Phase-2 configuration; P2_REFUSED_FLAGS)")
    if getattr(args, "freeze_item_tables", False):
        raise LaunchRefused(f"--arm {arm}: --freeze-item-tables is refused (the P2 recipe's `items` decides)")
    if getattr(args, "init_from", None) is None:
        raise LaunchRefused(f"--arm {arm} needs --init-from <PRETRAIN_FOR_FL_INIT.json> (theta_pre, PREG2 §5.1)")
    role = getattr(args, "role", "MAIN")
    if role not in ROLES:
        raise LaunchRefused(f"--role must be one of {ROLES}, not {role}")
    if role == "CALIBRATION" and int(args.seed) != CALIB["seed"]:
        raise LaunchRefused("P2 CALIBRATION screens run at seed 2026 only (PREG2 §5.1)")
    rp, sp = getattr(args, "p2_recipe", None), getattr(args, "p2_selection", None)
    main_run = role == "MAIN" and not args.smoke and not getattr(args, "dry_run", False)
    if sp is not None and role != "MAIN":
        raise LaunchRefused("--p2-selection is the MAIN input; a CALIBRATION screen takes --p2-recipe")
    if main_run and sp is None:
        raise LaunchRefused(f"a non-smoke MAIN {arm} run needs --p2-selection {P2_SELECTION_PATH} (the registered "
                            "CALIB selection; a recipe file never configures a MAIN run)")
    if (rp is None) == (sp is None):
        raise LaunchRefused(f"--arm {arm} takes exactly one of --p2-recipe and --p2-selection")
    if rp is not None:
        load_p2_recipe(rp)


def bind_p2_selection(args, init_identity: Optional[Mapping]) -> dict:
    """P2-02: a MAIN P2 run takes its recipe from the REGISTERED contracts/fl/P2_WARM_SELECTION_FINAL.json only (the
    given path must resolve to it; a copy never qualifies), hashed before and after verification: format, self_sha256
    (_canon), stage FINAL, the winner label = p2_label(winner_recipe) of a registered recipe, the selection rule
    re-derived by v4int.p2_select.load_p2_selection (P2-06; fails closed when absent), and theta_pre: the selection's
    common init_from (when recorded) = this run's --init-from. Returns the binding (with the recipe)."""
    rpath = P2_SELECTION_PATH
    sel = getattr(args, "p2_selection", None)
    if sel is None or not Path(rpath).is_file() or not _same_path(sel, rpath):
        raise LaunchRefused(f"--p2-selection {sel} is not the registered {rpath} (a preview / copy never qualifies)")
    sha = _file_sha(rpath)
    try:
        obj = json.loads(Path(rpath).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LaunchRefused(f"{rpath} unreadable ({e})") from e
    try:
        ok = isinstance(obj, dict) and obj.get("format") == P2_SELECTION_FORMAT and obj.get("self_sha256") == _canon(obj)
    except (TypeError, ValueError):
        ok = False
    if not ok or obj.get("stage") != "FINAL":
        raise LaunchRefused(f"{rpath} is not a verified {P2_SELECTION_FORMAT} (stage FINAL) object")
    rec = check_p2_recipe(obj.get("winner_recipe"), f"{rpath} winner_recipe")
    if obj.get("winner") != p2_label(rec):
        raise LaunchRefused(f"{rpath}: winner {obj.get('winner')!r} != its recipe's label {p2_label(rec)!r}")
    try:
        from v4int.p2_select import load_p2_selection
    except ImportError as e:
        raise LaunchRefused(f"the P2 selection verifier v4int.p2_select is not installed ({e})") from e
    try:
        load_p2_selection(rpath, "FINAL")
    except Exception as e:  # noqa: BLE001 — any verifier refusal refuses the launch
        raise LaunchRefused(f"{rpath}: the selection rule does not verify ({type(e).__name__}: {e})") from e
    ident = (obj.get("common") or {}).get("init_from")
    if ident is not None and (init_identity is None or dict(ident) != dict(init_identity)):
        raise LaunchRefused(f"{rpath}: its theta_pre identity differs from --init-from")
    if _file_sha(rpath) != sha:
        raise LaunchRefused(f"{rpath} changed while it was verified")
    return {"source": "p2_selection", "path": str(rpath), "sha256": sha, "self_sha256": obj["self_sha256"],
            "format": obj["format"], "stage": obj["stage"], "label": p2_label(rec), "recipe": rec,
            "init_from_checked": ident is not None, "flags": list(obj.get("flags") or []), "authority": P2_AUTHORITY}


def p2_resolve_spec(args, peak_lr: float, init_identity: Optional[Mapping]) -> tuple:
    """P2-02: (spec, extra) of a P2 arm: the recipe from --p2-recipe or the bound --p2-selection."""
    p2_preflight(args)
    if init_identity is None:
        raise LaunchRefused(f"{args.arm}: the theta_pre identity (load_pretrain_init) is required")
    if getattr(args, "p2_selection", None) is not None:
        b = bind_p2_selection(args, init_identity)
        rec, extra = b["recipe"], {"p2_selection": b}
    else:
        rec, src = load_p2_recipe(args.p2_recipe)
        extra = {"p2_recipe": src}
    spec = p2_arm_spec(args.arm, recipe=rec, seed=int(args.seed), peak_lr=peak_lr, role=getattr(args, "role", "MAIN"),
                       init_from=init_identity)
    return spec, extra


def p2_run_tag(rec: Mapping) -> str:
    """'FEDADAM_eta0.0003_c0.01', '..._UNFROZEN_ilr0.1', '..._PF' (run-id fragment of a recipe)."""
    tag = (lambda x: format(float(x), "g"))  # noqa: E731
    s = f"{rec['server_opt'].upper()}_eta{tag(rec['server_lr'])}_c{tag(rec['client_lr'])}"
    if rec["items"] == "unfrozen":
        s += f"_UNFROZEN_ilr{tag(rec['item_lr_mult'])}"
    if rec["upload"] != "dense":
        s += f"_{rec['upload'].upper()}"
    if rec.get("lora_rank") is not None:
        s += f"_LORA{int(rec['lora_rank'])}x{tag(rec['lora_lr_factor'])}"
    if rec["personal_head"]:
        s += "_PF"
    return s


def p2_run_id(args) -> str:
    """P2-02: P2_W_<recipe tag>_CALIB_s2026 (screen), P2_WSTAR_SMALL_s<seed> (MAIN, bound selection) or
    P2_W_<recipe tag>_SMALL_s<seed> (a MAIN smoke with a recipe file); the smoke prefix as for every arm."""
    from v4int.smoke import SMOKE_RUN_PREFIX
    prefix = SMOKE_RUN_PREFIX if args.smoke else ""
    if getattr(args, "p2_selection", None) is not None:
        return f"{prefix}P2_WSTAR_SMALL_s{args.seed}"
    rec, _src = load_p2_recipe(args.p2_recipe)
    tail = "CALIB" if getattr(args, "role", "MAIN") == "CALIBRATION" else "SMALL"
    return f"{prefix}{args.arm}_{p2_run_tag(rec)}_{tail}_s{args.seed}"


def p2_pre_server(spec: Mapping, server_adapter, workers, init_sha256: str) -> tuple:
    """P2 HOOK 1 (execute, after the item-table freeze and BEFORE fl.server.Server reads the manifest): structural
    changes of the trained parameter set. Returns (record or None, the init_sha256 the Server must verify). Wave A
    installs nothing (frozen tables use the existing freeze path) and refuses every recipe feature that is not installed.
    P2-04 (LoRA, company/handoffs/P2-03_04.md §5): when spec["p2_recipe"]["lora_rank"] is set, apply_lora +
    freeze_all_but on server_adapter, init_sha256 = fl.peft.lora_init_sha256(server_adapter, init_sha256), then
    apply_lora + freeze_all_but(source=server broadcast) on every worker; return (record, new init_sha256)."""
    r = spec["p2_recipe"]
    if r.get("lora_rank") is not None:
        raise LaunchRefused("LoRA (P2-04) is not installed in this launcher")
    return None, init_sha256


def p2_post_run(run, spec: Mapping) -> Optional[dict]:
    """P2 HOOK 2 (execute, after FLRun construction AND after FLRun.resume — both rebuild run.channel — where FA_Q8
    installs its channel): run-level channels / payload hooks. Wave A installs nothing. P2-03 (WB3, company/handoffs/
    P2-03_04.md §5): when spec["p2_recipe"]["upload"] == "delta_rows", sp = fl.sparse_rows.install_sparse(run,
    k_top=int(spec["upload_codec"]["k_top"])) and return {"channel": sp, ...}; the round loop then checks
    `run.channel is sp` and sp.check_round(out_r["round"], out_r["survived"]) like q8, and bytes_record adds
    sp.summary() (the per-visit upload is client-dependent: the ledger is exact). P2-04 (LoRA): fl.peft.install_peft(run)
    here (best.pt / endpoint_6.0.pt then carry the merged theta + p2_lora)."""
    if spec["p2_recipe"]["upload"] != "dense":
        raise LaunchRefused(f"upload {spec['p2_recipe']['upload']!r} (P2-03) is not installed in this launcher")
    return None


def client_index(tp, manifest_dir) -> tuple:
    """(sorted keys, {key: canonical positions}) — one client = one MAIN_TRAIN user, keyed by client_key_hex."""
    import numpy as np
    from v4base.identity import train_client_key_hex_by_canonical_position
    hexes = train_client_key_hex_by_canonical_position(tp, manifest_dir=manifest_dir)
    order = np.argsort(hexes, kind="stable")
    sh = hexes[order]
    cut = np.flatnonzero(sh[1:] != sh[:-1]) + 1
    groups = np.split(order.astype(np.int64), cut)
    keys = [str(sh[0])] + [str(sh[i]) for i in cut] if sh.size else []
    return keys, {k: np.sort(g) for k, g in zip(keys, groups)}


def run(args, pre: dict, env_rec: dict, say: Callable = print) -> int:
    if is_p2_arm(args.arm):                                # P2-02: the P2 selection is bound in p2_resolve_spec
        p2_preflight(args)
    else:
        require_lr_selection(args)                         # FL-07 5a: before any import / data / GPU work
    import v4train
    import ppsi_env
    snap = ppsi_env.verify_snapshot()
    iso = ppsi_env.assert_isolated()
    if v4train.LAUNCH_CUDA_VISIBLE_DEVICES != str(int(args.gpu)):
        raise LaunchRefused("the CUDA_VISIBLE_DEVICES export did not precede the v4train import")
    import numpy as np
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise LaunchRefused("expected exactly one visible CUDA device")
    torch.set_num_threads(int(args.n_threads))
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    from fl.client import ClientData
    from v4data.common import dataset_root
    from v4int.launch_central import _theta0_for
    from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
    from v4int.smoke import SMOKE_RUN_PREFIX
    from v4models import build
    from v4train.provider import expected_manifests
    from v4train.recipe import RunConfig
    from v4train.trainer import CentralTrainer, trainer_code_sha256
    allowed = assert_arm_allowed(args.arm, args.family)
    recipe = _small_recipe(args)
    init_state = init_ident = init_launch = None
    if is_warm_arm(args.arm) or is_p2_arm(args.arm):       # FL-12 (06K §2) / P2-02: theta_pre, hash-checked, first
        init_state, init_ident, init_launch = load_pretrain_init(args.init_from, model_variant=recipe["model_variant"])
    spec, spec_extra = resolve_spec(args, recipe["peak_lr"], fused=True, init_identity=init_ident)
    if init_launch is not None:
        spec_extra["init_from"] = init_launch
    run_dir = Path(args.run_dir)
    run_id = args.run_id or default_run_id(args)
    if args.smoke and not run_id.startswith(SMOKE_RUN_PREFIX):
        raise LaunchRefused(f"a smoke run id must start with {SMOKE_RUN_PREFIX}")
    paths = data_paths(args)
    lockctx = bind_data_protocol_lock_v2(paths, smoke=bool(args.smoke))
    view = build_trainer_registry_view(lockctx["manifest_registry_path"], run_dir,
                                       registry_rel_path=lockctx["manifest_registry_rel_path"],
                                       registry_file_sha256=lockctx["manifest_registry_file_sha256"])
    train_id, eval_id = expected_manifests("COHORT", "MAIN", registry=str(view["path"]))
    t0 = time.time()
    tp = StoreTrainProvider(train_id, history_policy=args.history_policy, paths=paths, log=say,
                           manifest_registry=lockctx["manifest_registry_path"], registry_root=V4)
    ep = StoreEvalProvider(eval_id, history_policy=args.history_policy, paths=paths, prefetch=int(args.eval_prefetch),
                           log=say, manifest_registry=lockctx["manifest_registry_path"], registry_root=V4)
    key = comparison_key_base(paths, tp, ep)
    theta0 = _theta0_for(argparse.Namespace(theta0_variants=args.theta0_variants, smoke=args.smoke),
                         lockctx["schema_id"], {"model_variant": recipe["model_variant"]}, int(args.seed))
    cfg = RunConfig(run_id=run_id, family=FAMILY, seed=int(args.seed), exposure_point="end",
                    cuda_visible_devices=str(int(args.gpu)), phase="MAIN", population="COHORT",
                    peak_lr=recipe["peak_lr"], log_every=50, checkpoint_every=0, key=key,
                    test_overrides=bool(args.smoke), determinism_mode=DETERMINISM_MODE,
                    manifest_registry=str(view["path"]), data_version=lockctx["data_version"],
                    schema_registry=str(lockctx["schema_registry_path"]), recipe_id=recipe["recipe_id"],
                    recipe_registry=recipe["recipe_registry"], model_variant=recipe["model_variant"],
                    dropout=tuple(recipe["dropout"]), schedule=recipe["schedule"], theta0_init_sha256=theta0)
    schema = load_production_schema(lockctx["schema_id"], lockctx["schema_self_sha256"], lockctx["schema_registry_path"])
    tr = CentralTrainer(cfg, schema, tp, ep, run_dir / "eval_harness", device="cuda:0", v1_root=dataset_root())
    worker = build(FAMILY, schema, int(args.seed), "cuda:0", v1_root=dataset_root(), variant=recipe["model_variant"],
                   dropout=tuple(recipe["dropout"])).adapter
    role = spec["role"]
    ctp, role_rec = tp, None                               # the client train provider (MAIN: the harness pair's)
    if role == "CALIBRATION":                              # FL-05: CALIB_TRAIN clients, CALIB_VALIDATION evaluation
        ctrain_id, ceval_id = expected_manifests("COHORT", "CALIBRATION", registry=str(view["path"]))
        ctp = StoreTrainProvider(ctrain_id, history_policy=args.history_policy, paths=paths, log=say,
                                 manifest_registry=lockctx["manifest_registry_path"], registry_root=V4)
        cep = StoreEvalProvider(ceval_id, history_policy=args.history_policy, paths=paths,
                                prefetch=int(args.eval_prefetch), log=say,
                                manifest_registry=lockctx["manifest_registry_path"], registry_root=V4)
        rk = comparison_key_base(paths, ctp, cep)
        kb = dict(tr.key_base)
        for f in ("population_hash", "candidate_order_hash", "train_manifest_id", "eval_manifest_id"):
            kb[f] = rk[f]
        role_over = {"manifest_role": "CALIBRATION", "run_role": "CALIBRATION",
                     "train_manifest_id": rk["train_manifest_id"], "eval_manifest_id": rk["eval_manifest_id"]}
        role_rec = {"role": role, "manifests": {"train": ctrain_id, "eval": ceval_id}, "comparison_key": kb,
                    "train_provider": ctp.describe(), "eval_provider": cep.describe()}
    keys, pos = client_index(ctp, paths.manifest_dir)
    n_valid = {k: int(v.size) for k, v in pos.items()}
    if base_arm(args.arm) == "DP" and not args.smoke and len(keys) != DP_ACCOUNTING["N"]:
        raise LaunchRefused(f"DP accounting assumes N = {DP_ACCOUNTING['N']} clients, the manifest has {len(keys)}")
    if spec_extra.get("s_record", {}).get("format") == S_FORMAT_WARM:   # FL-16: the warm S plan = this run's clients
        spec_extra["s_record"]["plan_digest_checked"] = check_warm_s_plan(spec_extra["s_record"], keys,
                                                                          ctp.fingerprint, spec)
    eb = spec.get("exposure_budget")                      # FL-13: the recomputed rounds assume the registered N
    if eb is not None and not args.smoke and len(keys) != eb["n_clients"]:
        raise LaunchRefused(f"the S4 exposure budget assumes N = {eb['n_clients']} clients; the manifest has {len(keys)}")
    if role == "CALIBRATION" and not args.smoke and (
            len(keys) != CALIB["n_clients"]
            or fa_rounds(len(keys), spec["cfg"]["solver"]["passes"]) != spec["cfg"]["endpoint_rounds"]):
        raise LaunchRefused(f"the CALIBRATION screen is registered for N = {CALIB['n_clients']} clients / "
                            f"{CALIB['rounds']} rounds; the manifest has {len(keys)} clients")
    t_prov = time.time() - t0

    def get_client(k: str) -> ClientData:
        return ClientData(k, ctp.batch(pos[k]))

    if role == "CALIBRATION":
        evaluate = role_evaluator(tr, cep, kb, role_over)
    else:
        def evaluate(mark: float):
            out = tr._evaluate(mark)
            return out.value, out.rows, out.ranks, out.result

    evaluate_pf = None
    if base_arm(args.arm) == "PF" or (is_p2_arm(args.arm) and spec["cfg"]["method"] == "PF"):   # INT-04 / P2 WC5
        from v4base.identity import eval_client_key_hex_by_eval_user_key
        from v4int.launch_device import pf_evaluator
        evaluate_pf = pf_evaluator(tr, eval_client_key_hex_by_eval_user_key(ep, manifest_dir=paths.manifest_dir))
    disk = disk_record(run_dir, 4 * 4 * tr.adapter.manifest.shared_numel)
    launch = {"task": "INT-03 launch_fl", "label": "SMOKE_NOT_EVIDENCE" if args.smoke else "PRODUCTION",
              "arm": args.arm, "run_id": run_id, "arm_spec": spec, "arm_spec_sha256": spec_digest(spec),
              "fl_config_digest": fl_run_config(spec, run_id, args.checkpoint_every).digest(), **spec_extra,
              "architecture_lock": allowed, "small_recipe": recipe, "history_policy": args.history_policy,
              "comparison_key": key, "manifests": {"train": train_id, "eval": eval_id},
              "data_paths": paths.as_json(),
              "data_protocol_lock_v2": {"lock_self_sha256": lockctx["lock_self_sha256"],
                                        "data_version": lockctx["data_version"],
                                        "trainer_registry_view": {"self_sha256": view["self_sha256"]},
                                        "schema_id": lockctx["schema_id"]},
              "preflight": pre, "environment": env_rec,
              "snapshot": {"verified_files": len(snap["files"]), "isolated_modules": len(iso)},
              "device": {"physical_gpu": int(args.gpu), "name": torch.cuda.get_device_name(dev)},
              **({"device_pin_sha256": tr.device_pin_sha256} if tr.device_pin_sha256 else {}),
              "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
              "harness": {"config_sha256": tr.config_sha256, "init_sha256": tr.built.init_sha256,
                          "trainer_code_sha256": trainer_code_sha256(), "theta0_init_sha256": theta0},
              "clients": {"n": len(keys), "n_decisions": int(ctp.n_decisions), "manifest_hash": ctp.fingerprint},
              "fl05": {"role": role, "lr_mult": spec["lr_mult"], "lr_peak_base": spec["lr_peak_base"],
                       "upload_codec": spec["upload_codec"], "role_record": role_rec},
              "providers_build_s": round(t_prov, 2), "train_provider": tp.describe(), "eval_provider": ep.describe(),
              "determinism": {"constant": DETERMINISM_MODE, "at_start": determinism_mode_in_effect()},
              "disk_at_start": disk, "mode": "resume" if args.resume else "fresh"}
    lp = _next_launch_path(run_dir)
    write_json(lp, launch)
    say(f"[fl] {run_id}: {lp.name} written; {len(keys)} clients; mode {launch['mode']}")
    init_sha = tr.built.init_sha256
    if init_state is not None:                             # FL-12: the server (= harness) model starts at theta_pre
        init_sha = apply_init_state(tr.adapter, init_state, init_ident["state_digest"])
        say(f"[fl] {run_id}: theta_pre loaded ({init_ident['run_id']}, checkpoint {init_ident['checkpoint_sha256'][:12]}…)")
    out = execute(spec=spec, run_id=run_id, run_dir=run_dir, server_adapter=tr.adapter,
                  init_sha256=init_sha, workers=[worker], keys=keys, n_valid_by_key=n_valid,
                  get_client=get_client, n_decisions=int(ctp.n_decisions), manifest_hash=ctp.fingerprint,
                  evaluate=evaluate, resume=bool(args.resume), ckpt_every=int(args.checkpoint_every),
                  smoke_rounds=int(args.smoke_rounds) if args.smoke else None,
                  interrupt_at=int(args.smoke_interrupt_at), say=say, evaluate_pf=evaluate_pf)
    return 0 if out["status"] in ("DONE_MAIN", "DONE_S_CAL", "SMOKE_DONE_NOT_EVIDENCE", "INTERRUPTED") else 1


def dry_run(args, say: Callable = print) -> int:
    """Config hash without data or GPU: the arm spec, its sha256 and the fl RunConfig digest."""
    import v4train  # noqa: F401  (ppsi_env first)
    recipe = _small_recipe(args)
    ident = None
    if is_warm_arm(args.arm) or is_p2_arm(args.arm):       # FL-12 / P2-02: the dry run verifies theta_pre too
        _state, ident, _launch = load_pretrain_init(args.init_from, model_variant=recipe["model_variant"])
    spec, extra = resolve_spec(args, recipe["peak_lr"], fused=True, init_identity=ident)
    run_id = args.run_id or default_run_id(args)
    say(json.dumps({"arm_spec": spec, "arm_spec_sha256": spec_digest(spec),
                    "fl_config_digest": fl_run_config(spec, run_id, args.checkpoint_every).digest(),
                    "small_recipe": recipe, **extra}, indent=1, sort_keys=True, default=str))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        pre = preflight(args)
        if args.dry_run:
            return dry_run(args)
        env_rec = export_environment(int(args.gpu))
        return run(args, pre, env_rec)
    except LaunchRefused as e:
        print(f"LAUNCH REFUSED: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
