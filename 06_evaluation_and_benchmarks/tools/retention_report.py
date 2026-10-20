"""tools/retention_report.py — the PPSI V4 retention report (task ANA-01 + its amendment-06B addendum + ANA-02).

  python -B tools/retention_report.py --block <dir> --split VALIDATION|TEST --contrasts <ALL_METHODS_LOCK.json|contrasts.json>
         --out <new dir> [--code-root <V4>/code] [--expect-script-sha256 <hex>] [--check-only]
         [TEST only: --all-methods-lock <ALL_METHODS_LOCK.json> --test-exposure-register <REGISTER.json>
                     --registry-ext <REGISTRY_V2_TEST_EXT.json>]
         [VALIDATION dry check only: --allow-v1-contrasts  (company/gates/ANA_CONTRASTS.json, K1-K16)]

ANA-03 (REV-DELTA-01 F9 / F11; 06I §1, §7b; 06K, DL 206): K17-K19 (R vs C_FULL, seed 2026); K20 R(FA_WARM / C_BIG) and
K21 FA_WARM - PRE (secondary, seeds 2026 + 2027, PRACTICAL_BEST; PRE = ONE theta_pre seed shared by both FA_WARM seeds) with
the DESCRIPTIVE rows R(FA_WARM / C_FULL) and R(PRE / C_FULL) beside K20 (result key `warm_start`; K20 / K21 are CONFIRMATORY
only when the loaded file is format _3, else computed as DESCRIPTIVE); a second bytes-per-client column with the dropped
visits charged (`bytes_per_client_charged_dropped`; FIX-DELTA-02 M4 / DL 218: the registered 06I §7b variant charges a
dropped visit its DOWNLOAD only, and the up + down variant `bytes_per_client_charged_dropped_up_down` is DESCRIPTIVE);
the deployed C_BIG PRACTICAL_BEST id `...:best` accepted when the
row's checkpoint_sha256 is the ARCHITECTURE_LOCK c_big best.pt sha256 of that seed.

ANA-04 (PREG1_AMENDMENT_06L_C, registered 2026-09-29; review MULTI_EXPERT_REVIEW_20260929 §4 Gate C):
  C3  baselines with per-row rank files enter a BASELINE_CPU sub-block (comparison_block "BASELINE_CPU", RULE_FREE,
      v4base.run ranks_int32.npy [+ n_tied_int32.npy]); every arm gets the paired Δ(arm − LAST) with CI, the headroom
      R* = (arm − LAST) / (C_ref − LAST) (C_ref = C_BIG, same rule and seed; per resample on the same W) and the flag
      BELOW_DEVICE_HEURISTIC (CI of Δ below 0); K11b = FA − LAST (secondary confirmatory, LAST = a seed-free anchor);
      the device-only envelope {LO, LAST, RULE} as one row group.
  C4  company/gates/ANA_CONTRASTS_v3.json (format PPSI_V4_ANA_CONTRASTS_3): K1-K21 + K9b (FT_C − C_SMALL on
      UNSEEN_HOLDOUT), K9c (FT_PRE − PRE), K11b, K20b = R(FA_WARM / C_FULL) (2 FA_WARM seeds vs C_FULL s2026), K20c =
      R(PRE / C_FULL) — 26 confirmatory — and the registered DESCRIPTIVE contrast S8-S7 (FA_WARM_FROZEN − FA_WARM, with the
      bytes ratio); texts, seeds, rule map, hierarchy, populations, pairing notes, reading rules and the algebraic
      dependency list are all checked against this script. A v2 file still loads (the 06K / 06L-C rows are then
      computed and labelled DESCRIPTIVE, "registered but not carried by the loaded file").
  C8  one bytes axis for every privacy level (raw events -> model download -> FL traffic -> S8) beside the 06L-C C1
      privacy matrix; S8 − S7 with the bytes ratio.
  C11 GRU_D128 vs C_SMALL (deployed-size family comparison), DESCRIPTIVE, both rules.
  C12 CLAIM_REGISTER.json (allowed / forbidden sentence templates per contrast; the applicable template key is stamped
      on every K) and HEADLINE_TEMPLATE.md (number-free); `--claims-only --contrasts <file> --out <dir>` writes both
      without a block (for the freeze).
  C13-C15 per-seed ratios, seed spread, a user + seed interval and SEED_SENSITIVE beside every K; a Holm band (normal
      intervals from the bootstrap SE, family = the loaded registry) beside the unadjusted CIs; the CALIB-disjoint
      diagnostic (MAIN \\ CALIB by client_key_hex) for every CALIB-tuned arm and every K that uses one; the harmed-client
      share (with per-user Δ quantiles and a same-arm seed-vs-seed null) for every K; K14's registered equivalence margin
      ±1 % relative (TOST: 90% CI inside [99, 101]); the repeat / novel target stratum (REPEAT = RANKABLE and LAST's
      per-row tie count 0, i.e. the target is in the L50 window; NOVEL = every other E2E row).
  Arms: FT_PRE (device arm; launch_device FT_C rows from the θ_pre base, method FT_C accepted), FA_WARM_FROZEN (S8, two
      seeds), DP8_FROZEN (S8 + DP ε = 8, seed 2026, z as DP8), GRU_D128 (C11, descriptive), and the no-FT UNSEEN_VAL
      references (C_SMALL / FA on role UNSEEN_VAL, beside the UNSEEN_VAL SELECTION value of FT_C / FT_FA), as labelled
      DESCRIPTIVE rows.

FIX-DELTA-03 (company/reviews/REV-DELTA-03_OPUS_REVIEW.md): M1 PRE / C_FULL rows must be v4int.reeval rows of that
registered central source on every role (central_source = the arm; source_train_manifest_id PRETRAIN_TRAIN_5_100_NONMAIN /
FULL_TRAIN_5_100_TIEBREAK; PRE never method "C"; a central-source row under any other arm refuses); M2 FA_WARM /
FA_WARM_FROZEN / DP8_FROZEN need RUN_SUMMARY arm_spec.init_from = the theta_pre identity and no other FL arm may carry one
(rows labelled fl_init likewise); M3 the Holm band after the first non-rejection is at alpha / (m - r), so band and
decision agree; M5 a TEST run refuses every contrasts format but PPSI_V4_ANA_CONTRASTS_3; m3 the BASELINE_CPU rows share
one key among themselves; m4 claim vocabulary "no detectable difference" (contract §7.6); m5 K20c's forbidden sentence is
unconditional; m6 K11b annotated beside the algebraic dependencies; m7 three disclosures; m8 / 06L-C §C10 the matched-h
parity M(h) / C_BIG(h) and the persistence rule (PERSISTENCE_RULE) beside the first-hit rule, with CIs from optional
per-user evidence (records.TRUNC_ROWS[i].per_user, PPSI_V4_TRUNC_PER_USER_1).

DEV-FTPRE (06K §2 / 06L-C §C7): FT_PRE's base check uses the PRE init record, not a run-id prefix. An FT_PRE entry needs
`start_init` (launch_device --arm FT_PRE's LAUNCH start.init_from = the theta_pre identity, init_from_problem None; its
run id = start_run_id) and `start_init_record_sha256` (the PRETRAIN_FOR_FL_INIT.json file sha256), and the block's PRE
entries must carry `pre_init` (the PRE REEVAL_RECORD's binding to the same record: checkpoint_sha256 = fl_init sha256,
init_record_sha256 = the record's file sha256, run_id) with ONE theta_pre that equals FT_PRE's checkpoint sha256, init
record sha256 and run id (FT_PRE without a PRE entry refuses). No other arm may carry start_init or a theta_pre start
run id; only PRE carries pre_init. FT_PRE rows may carry method FT_PRE (launch_device) or FT_C.

ANA-05 (C_FULL s2027; 06M; 06M_B B2 / B4 / B5; 06L-C C8), every item DESCRIPTIVE; ANA_CONTRASTS_v3's 26 confirmatory
contrasts are unchanged:
  C_FULL s2027  a DESCRIPTIVE source seed (06D §8; 06E §2 / DL 176): its rows must carry source_seed_status /
      comparison_status DESCRIPTIVE (v4int.reeval ANA-05), seed-2026 rows must not; K15 / K17-K19 / K20b / K20c keep
      seed 2026 (s2027 only in the retention table and the DESCRIPTIVE extra_seeds rows); result `ana05.descriptive_seeds`.
  FA_WARM / FA_WARM_FROZEN MAIN (run dir)  bound to the 2-D FL_WARM_SELECTION_S7|S8.json: entry `warm_selection` (the
      LAUNCH binding, format PPSI_V4_FL_WARM_SELECTION_2) with winner factors = RUN_SUMMARY arm_spec factors; one file per
      stage; records.WARM_SELECTION_S7 / _S8 (the file) verified and equal to the binding when given (check_ana05).
  FT_FA_WARM / FT_FA_WARM_FROZEN  device arms (PRACTICAL_BEST, device_cell); entry `start_base` = launch_device's
      --base-kind fa_warm start (ft_warm_base_problem), rows carry that base identity, and the base is the block's
      FA_WARM / S8 run of the same seed (run id + arm-spec sha256, theta_pre); `ana05.production_stack` = the paired
      delta FT_FA_WARM − FA_WARM (+ _FROZEN) with the harmed-client share (06M_B B4; 06L-C C13-C15).
  WARM_PREVIEW / S7-S8 CALIB screens  a run whose RUN_SUMMARY arm_spec role is not MAIN refuses (PREVIEW_LABEL).
  06M_B B5  `ana05.rollout_quality` (+ curves/rollout.csv): AUC of the per-mark curve and the first mark reaching
      PRE-alone / LAST for FA, FA_WARM and S8 (ROLLOUT_RULE).
  06L-C C8  `ana05.s8_comm`: S8's bytes per visit / up / down / uploaded + frozen parameter counts from the run's own
      fl.comm record, beside the registered 265,162 and FA_WARM's bytes per visit.

FIX-REEVAL-06O (PREG1_AMENDMENT_06O): a C_FULL / PRE entry on UNSEEN_VAL, and any row labelled preview_reference
(v4int.reeval's NON_MAIN_UNSEEN_VAL preview rows), refuses with PREVIEW_REFERENCE_LABEL: never a block entry, a
contrast, a selection or a claim (tools/ana_block_index.py already lists those REEVAL dirs as excluded).

ANA-06 (PREG1_AMENDMENT_06N; FL-16; DL 287 m3; REV-06O-ANA05 MINOR 1; FABLE_ENDGAME_REVIEW_20261001 D1 / D2 / D5):
  warm line  the FL-16 labels are block arms (WARM_LINE_ARMS = the run-dir names: FP / PF / FA_Q8 / FA_1024 _WARM[_FROZEN],
      DP8_WARM / DP1_WARM, DP8_FROZEN = DP_WARM_FROZEN eps 8, DP1_WARM_FROZEN); each is the launch label's cold-arm
      semantics from theta_pre (warm_line_base_problem) bound like FA_WARM(_FROZEN) to its variant's S7 / S8 file and its
      FL run (check_ana06; TEST: ana06.warm_line bound_runs); records S_RECORD_WARM[_FROZEN] verified (S content, REV-FL-16
      m5 on the ANA side) and bound to every Poisson warm-line run's clip norm; a device arm other than FT_FA_WARM(_FROZEN)
      on a *_WARM* base refuses (DL 287 m3); an entry bound through a run folder must have rows naming that run
      (run_rows_problem, MINOR 1).
  FA_WARM*   06N N2 from the two selection files (fa_warm_star; CALIB winner values; tie 0.0005 -> S8).
  06N rows   with v1-v3 loaded: result `ana06` = K20b* / K21* / K20* / K_INIT / the warm method analogs on FA_WARM*, all
      DESCRIPTIVE (N06_LABEL); v3 outputs unchanged. With company/gates/ANA_CONTRASTS_v4.json (format
      PPSI_V4_ANA_CONTRASTS_4 = contracts/PREG1_AMENDMENT_06P.md P1-P7; tables V4_PROFILE) the 06P set is the
      CONFIRMATORY registry (primary pair K20b + K21 on FA_WARM_FROZEN outside Holm, K2 secondary, K_INIT, P5, K4w / K5w /
      K12w / K13w / K14w secondary, Holm over the enumerated secondary family, a DESCRIPTIVE band within the tertiary set)
      and a block whose selection files choose S7 refuses; a file with draft not false needs --allow-draft-contrasts.

Authority: PREG1_AMENDMENT_04 §1/§4, _06 §6-§7, _06A §2/§3/§7/§8/§9, _06B §2/§3/§4/§7, _06D §1-§5; EVALUATION_CONTRACT_V4
§5, §7, §8, §10, §11.7; review REV-ANA-01 (A-F1 .. A-F17, fixed in ANA-02). Every rule is REUSED from the --code-root
library (put first on sys.path; module origins checked):
v4train.bootstrap_ci (_plan: one shared W per user set, seed 20260926, 1,000 resamples; _stats: macro-E2E MRR@20 user
statistics), v4eval.bootstrap (user_stats / point / resampled / ci95), v4eval.metrics (credits, population_block),
v4eval.quality (retained_percent: E8.2 no clamping, E8.3 zero denominator), v4base.strata (load_registry,
assign_stratum_for_user, NO_TRAIN), v4base.identity (client_key_hex join) and tools/g2_run.py (production manifest
loader; how rows / ranks are bound to the manifest). This script only binds, checks, computes and records.

INPUT: `<block>/BLOCK_INDEX.json` (format PPSI_V4_ANA_BLOCK_INDEX_1; every path in it is relative to <block> and must stay
inside it):
  {"format": "PPSI_V4_ANA_BLOCK_INDEX_1", "split": "VALIDATION", "eval_device_class": "<the A40 class string>",
   "roles": {"MAIN": {"eval_manifest_id": "MAIN_VALIDATION_5_100_131072", "train_manifest_id": "MAIN_TRAIN_5_100_131072"},
             "UNSEEN_HOLDOUT": {...}, "TRUE_COLD": {"eval_manifest_id": "TRUE_COLD_VALIDATION"}},
   "entries": [{"arm": "C_BIG", "seed": 2026, "rule": "PRACTICAL_BEST", "role": "MAIN",
                "rows": "<RESULT_ROWS.json or rows.json>", "ranks": "<int32 ranks .npy, one per manifest row>",
                "row_filter": {"<row field>": <value>, ...}, "run_dir": "<FL run dir: 12 marks + RUN_SUMMARY>",
                "users_subset": "<.npy of eval_user_key (int64)>", "historical_reference": false,
                "meta": {"param_count", "model_bytes", "bytes_per_visit", "bytes_per_client", "bytes_fleet",
                         "epsilon", "rounds", "consumed_efe", "dp_z"}}, ...],
   "records": {"STRATA": "STRATA_CUTPOINTS_MAIN.json", "S_RECORD": "...", "SELECTION": ["..."], "LR_SELECTION": "...",
               "DP_WOR": "DP_WOR_SENSITIVITY_FL06.json", "DP_Z_EPS4": "...", "TRUNC_ROWS": ["..."],
               "VALIDATION_RESULT": "<the VALIDATION RETENTION_RESULT.json (TEST run only: shift indicator)>"}}
  arm   one of ARMS (DP arms are DP8 / DP4 / DP1; C_SMALL_FP32 / C_SMALL_INT8 = the ONNX FP32 / INT8 rows of K16).
  rule  PRACTICAL_BEST | CONTROLLED_BUDGET (a checkpoint rule; rows that carry selection_rule must agree) | RULE_FREE
        (baselines; seed may be null; paired with the denominator of every rule / seed and labelled so).
  ranks optional (baselines): without ranks the entry is point-only (value from its rows, CI "NO_RANKS").
  historical_reference  true exempts the entry from the block comparison key (E9.1); point-only, never confirmatory
        (its rows must still carry the --split split).
  run_dir  FL runs: metrics.jsonl (eval lines: mark / efe / round), evals/eval_<m>.rows.json + .ranks_int32.npy for the
        12 marks and RUN_SUMMARY.json (status DONE_MAIN; bytes / rounds / consumed EFE / z / best_half_efe are read from
        it; meta overrides are recorded). still_improving_at_6.0 = (PRACTICAL_BEST mark >= 5.5) (06D §5, descriptive).
  users_subset  the entry is evaluated on these users only (06A §6 INT8 fallback); its contrasts use the same users.
        Either a .npy of eval_user_key (int64) or {"rule": "SELECTION_KEY_V2_FIRST", "n": N, "set_sha256": <hex>} (= the
        role's first N E2E users in selection_key_v2 order, the v4dev.dump_rows --first-users rule of 06E §3; the set
        sha256 = sha256 of the sorted selected client_key_hex joined by "\\n" must equal rows.meta.json
        first_users_set_sha256).
  comparison_block  "ONNX_CPU" (REQUIRED for C_SMALL_INT8 / C_SMALL_FP32, allowed for nothing else): the onnxruntime CPU
        sub-block (A-F2). Exempt from the A40 device / status checks only; its entries share one runtime identity
        (onnxruntime version, intra_op_num_threads, graph optimisation level, rows npz / meta sha256, manifest_n from the
        BENCH_RESULT `retention[tag]` block whose ranks_sha256 is the rank file's). `rows` may be null: the values are
        then computed from `ranks` by this script's library. K16 is CONFIRMATORY only
        as C_SMALL_INT8 / C_SMALL_FP32 inside this sub-block; the fallback to the A40 C_SMALL is DESCRIPTIVE, DEVICE_MIXED.
  rule restrictions  RULE_FREE only for the baselines (BASELINE_ARMS); device arms and the ONNX arms only
        PRACTICAL_BEST (A-F3). Device-arm entries need row_filter.device_cell (A-F10).
  records.TRUNC_ROWS  a list of {"rows": "<DATA-07 RESULT_ROWS.json>", "rule": "PRACTICAL_BEST"} (A-F1), optionally
        with "per_user": "<.npz PPSI_V4_TRUNC_PER_USER_1>" (arrays format, u__<cell> / m__<cell>, see trunc_cell_key;
        each cell's per-user mean must reproduce its row value) for the 06L-C §C10 persistence CIs.

CHECKS (refusal = exit 2, nothing written): the comparison key inside the block (KEY_FIELDS = eval_device_class, split,
evaluator sha and the E9.1 hashes candidate_order / label_policy / data_protocol / feature_contract, precision_id,
objective_id, eval_score_block, eval_stream; manifest: eval_manifest_id / query_order_hash = the loaded manifest's
eval-ordered hash; INTENDED_DIFFERENCES recorded, not compared; on the pinned A40 class every row must be
MEASURED_PINNED_DEVICE with one device_pin_sha256) over every evidence row and FL mark row of every non-historical entry;
the declared split = --split = the rows' split; each entry's declared rule / seed vs its rows, the rule vs the rows'
efe / checkpoint_id, the arm vs train_manifest_id / method (A-F7), no SMOKE / LR-screen / CALIBRATION rows (A-F13);
each rank file reproduces its rows' macro-E2E MRR@20 / NDCG@10 / HR@10 within 1e-9 and ranks every RANKABLE row of its
view (A-F6); numerator and denominator have the same checkpoint rule and the same seed (06A §7; RULE_FREE / seed-free
baseline numerators excepted); every full-population denominator != 0 (E8.3); the --contrasts registry = the 06B §2
text (K1-K19 in format PPSI_V4_ANA_CONTRASTS_2, K1-K21 in _3; the v1 K1-K16 file only for a VALIDATION
--allow-v1-contrasts dry check) and carries confirmatory_rule_map = the 06D §3 map (and hierarchy = 06D §4 when present). --check-only
runs every check including every pairing and every full-population denominator, and prints no value.
TEST (E10.1, A-F8): --all-methods-lock + --test-exposure-register (the lock sha256 in a MANIFEST_BUILD RETRO_TEST entry of
a verified hash chain; --contrasts IS the lock file), --registry-ext (REGISTRY_V2 entries byte-identical, REGISTRY_V2 =
the DATA_PROTOCOL_LOCK_V2 pin), the registered TEST role map, every TEST manifest header bound to the gate, and a
VALIDATION_RESULT of the same script / library / contrasts. The TEST manifests are loaded only through the DATA-09 gated
reader (hook `test_context_loader`, see there).
OUTPUT (--out must be new or empty; O_EXCL; strict JSON; deterministic: no clock, relative paths only):
RETENTION_RESULT.json (self_sha256), REPORT.md, curves/*.csv + curves/*.json, OUTPUT_MANIFEST.json (sha256 of each).
Only the loaded registry's contrasts (K1-K19 / K1-K21) are CONFIRMATORY (each under its CONFIRMATORY_RULE, registered seeds only); every other number is labelled
DESCRIPTIVE. Curves / strata / bins use PRACTICAL_BEST.
Exit: 0 written (or --check-only pass), 2 refused, 5 error.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
DEFAULT_CODE_ROOT = V4 / "code"
INDEX_NAME = "BLOCK_INDEX.json"
INDEX_FORMAT = "PPSI_V4_ANA_BLOCK_INDEX_1"
SCHEMA = "PPSI_V4_ANA_RETENTION_1"
RESULT_NAME = "RETENTION_RESULT.json"
REPORT_NAME = "REPORT.md"
MANIFEST_NAME = "OUTPUT_MANIFEST.json"
SELF_KEY = "self_sha256"
RULES = ("PRACTICAL_BEST", "CONTROLLED_BUDGET")
RULE_FREE = "RULE_FREE"
CURVE_RULE = "PRACTICAL_BEST"
SEEDS = (2026, 2027)
PRIMARY = ("mrr", 20)
METRICS = (("mrr", 20), ("ndcg", 10), ("hr", 10))
ROW_VALUE_TOL = 1e-9
REF, REF_SMALL = "C_BIG", "C_SMALL"
ARMS = ("C_BIG", "C_FULL", "C_SMALL", "KD_SMALL", "GRU_D128", "FA", "FP", "PF", "CPF", "FA_1024", "DP8", "DP4", "DP1",
        "FA_Q8", "FA_WARM", "FA_WARM_FROZEN", "DP8_FROZEN",
        "FP_WARM", "PF_WARM", "FA_Q8_WARM", "FA_1024_WARM", "DP8_WARM", "DP1_WARM",          # ANA-06 (FL-16 warm
        "FP_WARM_FROZEN", "PF_WARM_FROZEN", "FA_Q8_WARM_FROZEN", "FA_1024_WARM_FROZEN",       # method set; 06N N3.2)
        "DP1_WARM_FROZEN", "PRE", "FT_C", "FT_PRE", "FT_FA", "FT_FA_WARM",
        "FT_FA_WARM_FROZEN", "FT_KD", "LO",
        "C_SMALL_FP32", "C_SMALL_INT8", "KNN_EMB_SMALL", "KNN_EMB_BIG", "RULE", "SKNN", "LAST", "POP")
KEY_FIELDS = ("eval_device_class", "split", "evaluator_code_sha256", "candidate_order_hash", "label_policy_hash",
              "data_protocol_hash", "feature_contract_hash", "precision_id", "objective_id", "eval_score_block",
              "eval_stream")                                           # A-F5: the E9.1 key (v4eval.rows.COMPARISON_KEY)
INTENDED_DIFFERENCES = ("population_hash", "topology_hash", "initialization_block", "eval_batch")   # recorded only
ONNX_KEY_FIELDS = tuple(f for f in KEY_FIELDS if f != "eval_device_class")
PINNED_CLASS = "NVIDIA_A40-FP32-DET"                                   # 06D §1
PINNED_STATUS = "MEASURED_PINNED_DEVICE"
ONNX_BLOCK = "ONNX_CPU"
BASELINE_BLOCK = "BASELINE_CPU"          # 06L-C §C3: baselines WITH per-row rank files (v4base.run ranks_int32.npy)
MAIN_BLOCK = "MAIN_BLOCK"
BLOCKS = (MAIN_BLOCK, ONNX_BLOCK, BASELINE_BLOCK)
BASELINE_EXEMPT_KEY_FIELDS = ("eval_device_class", "precision_id", "objective_id")   # CPU scoring; v4base placeholders
BASELINE_KEY_FIELDS = tuple(f for f in KEY_FIELDS if f not in BASELINE_EXEMPT_KEY_FIELDS)
ANCHOR_ARMS = ("LAST",)                  # 06L-C §C3: the no-training device heuristic every arm is anchored on
HIST_WINDOW = 50                         # USER_HISTORY_L50: a history item's LAST rank is <= 50
SUBSET_RULE_V2 = "SELECTION_KEY_V2_FIRST"                              # 06E §3 / DEV-04 dump_rows --first-users
SPLIT_VALUES = {"VALIDATION": ("VALIDATION",), "TEST": ("TEST", "RETRO_TEST")}
ROLE_MAP = {   # the registered roles of each split (REGISTRY_V2; 06D §2 / A-F8 (iii) for TEST): role -> (eval, train)
    "VALIDATION": {"MAIN": ("MAIN_VALIDATION_5_100_131072", "MAIN_TRAIN_5_100_131072"),
                   "UNSEEN_HOLDOUT": ("UNEW_HOLDOUT_VALIDATION_32768", "UNEW_HOLDOUT_SUPPORT_32768"),
                   "SPARSE_PANEL": ("SPARSE_PANEL_16384_VALIDATION", "SPARSE_PANEL_16384_SUPPORT"),
                   "HEAVY_PANEL": ("HEAVY_PANEL_16384_VALIDATION", "HEAVY_PANEL_16384_SUPPORT"),
                   "TRUE_COLD": ("TRUE_COLD_VALIDATION", None),
                   "UNSEEN_VAL": ("UNEW_VAL_VALIDATION_32768", "UNEW_VAL_SUPPORT_32768")},   # 06L-C §C16 no-FT refs
    "TEST": {"MAIN": ("MAIN_RETRO_TEST_5_100_131072", "MAIN_TRAIN_5_100_131072"),
             "UNSEEN_HOLDOUT": ("UNEW_HOLDOUT_RETRO_TEST_32768", "UNEW_HOLDOUT_SUPPORT_32768"),
             "SPARSE_PANEL": ("SPARSE_PANEL_16384_RETRO_TEST", "SPARSE_PANEL_16384_SUPPORT"),
             "HEAVY_PANEL": ("HEAVY_PANEL_16384_RETRO_TEST", "HEAVY_PANEL_16384_SUPPORT"),
             "TRUE_COLD": ("TRUE_COLD_RETRO_TEST", None)},
}
BASELINE_ARMS = ("RULE", "SKNN", "LAST", "POP", "KNN_EMB_SMALL", "KNN_EMB_BIG")        # the only RULE_FREE arms (A-F3)
DEVICE_ARMS = ("LO", "FT_C", "FT_PRE", "FT_FA", "FT_KD",
               "FT_FA_WARM", "FT_FA_WARM_FROZEN")        # ANA-05 (06M_B B4): theta_pre -> FA_WARM / S8 -> the FT_C procedure
ONNX_ARMS = ("C_SMALL_FP32", "C_SMALL_INT8")
PB_ONLY_ARMS = DEVICE_ARMS + ONNX_ARMS
FL_ARMS = ("FA", "FP", "PF", "CPF", "FA_1024", "DP8", "DP4", "DP1", "FA_Q8", "FA_WARM", "FA_WARM_FROZEN", "DP8_FROZEN",
           "FP_WARM", "PF_WARM", "FA_Q8_WARM", "FA_1024_WARM", "DP8_WARM", "DP1_WARM", "FP_WARM_FROZEN",   # ANA-06
           "PF_WARM_FROZEN", "FA_Q8_WARM_FROZEN", "FA_1024_WARM_FROZEN", "DP1_WARM_FROZEN")
FL_METHOD = {"DP8": "DP", "DP4": "DP", "DP1": "DP", "DP8_FROZEN": "DP"}   # launch_fl labels every DP run method "DP"
FL_METHOD_ALT = {"FA_WARM": ("FA",), "FA_WARM_FROZEN": ("FA",),      # 06K: FA from theta_pre; the launcher may keep "FA"
                 "FT_PRE": ("FT_C",)}      # 06L-C §C7: FT_PRE = the registered FT_C procedure (launch_device FT_C rows)
FT_PRE_BASE_RUN_PREFIX = "PRETRAIN_FOR_FL"   # a theta_pre start run id (never FT_C / FT_FA / LO; DEV-FTPRE: FT_PRE's
#   base itself is checked against the PRE init record: start_init + start_init_record_sha256 vs PRE's pre_init)
CENTRAL_ARMS = ("C_BIG", "C_FULL", "C_SMALL", "KD_SMALL", "PRE", "GRU_D128")   # rows may carry method "C"
FROZEN_ONCE_ARMS = ("FA_WARM_FROZEN", "DP8_FROZEN",  # S8 (06K §3; GATE-B B3, review 09 [8-C1]) and its DP variant
                    "FP_WARM_FROZEN", "PF_WARM_FROZEN", "FA_Q8_WARM_FROZEN", "FA_1024_WARM_FROZEN",   # + ANA-06: every
                    "DP1_WARM_FROZEN")                                                  # S8-bound FL-16 arm
                                                     # (06L-C §C17): item tables frozen at theta_pre, downloaded once
FROZEN_ONCE_RULE = ("PER_CLIENT_INCL_FROZEN_ONCE: visits x (down + up) of the uploaded parameters + the frozen item "
                    "tables downloaded ONCE per visited client (launch_fl per_client_incl_frozen_once; 06K §3)")
FROZEN_CURVE_RULE = ("FROZEN_ONCE_STEP_AT_FIRST_VISIT + PRORATED_BY_ROUND (INFERENCE): the fleet-mean one-time frozen "
                     "table download is charged in full from the first visit (a step, never prorated); the training "
                     "traffic is prorated by round / rounds as for every FL arm")
SHARED_SEED_ARMS = ("PRE",)      # 06K / DL 206 (2): ONE theta_pre (seed 2026) is the denominator of every numerator seed
NO_SELECTION_ARMS = ("PRE",)     # 06K §1: theta_pre = the endpoint_6.0 checkpoint, no selection (either rule label = 6.0)
PRE_TRAIN_MANIFESTS = ("PRETRAIN_TRAIN_5_100_NONMAIN", "MAIN_TRAIN_5_100_131072")   # DATA-10 / the MAIN key pair
SOURCE_TRAIN_MANIFEST = {"PRE": "PRETRAIN_TRAIN_5_100_NONMAIN",      # REV-DELTA-03 M1: the registered v4int.reeval
                         "C_FULL": "FULL_TRAIN_5_100_TIEBREAK"}      # central sources (GATE-B B1): rows carry
#   central_source = the arm and source_train_manifest_id = its training manifest on EVERY role (the key train side of
#   PRE on MAIN / of every panel is the role's, so train_manifest_id alone cannot tell a MAIN-trained checkpoint apart)
WARM_INIT_ARMS = ("FA_WARM", "FA_WARM_FROZEN", "DP8_FROZEN")   # REV-DELTA-03 M2: FL arms started from theta_pre
PRETRAIN_INIT = {"kind": "PRETRAIN_FOR_FL", "run_label": "PRETRAIN_FOR_FL", "fl_init_kind": "endpoint_6.0",
                 "train_manifest": "PRETRAIN_TRAIN_5_100_NONMAIN", "family": "SASREC"}   # = launch_fl.PRETRAIN (06K §2)
PRETRAIN_RUN_PREFIX = "PRETRAIN_FOR_FL_"
DP_Z = {"DP8": 0.5642, "DP1": 1.3094,                                   # amendment 06 registered z (DP4: DP_Z_EPS4 record)
        "DP8_FROZEN": 0.5642,                                           # 06K §3: DP8_FROZEN "with the same z/q/T/δ"
        "DP8_WARM": 0.5642, "DP1_WARM": 1.3094, "DP1_WARM_FROZEN": 1.3094}   # ANA-06: FL-16 load_z (unchanged)
TRAIN_MANIFEST_OF = {"C_FULL": "FULL_TRAIN_5_100_TIEBREAK", "C_BIG": "MAIN_TRAIN_5_100_131072",
                     "C_SMALL": "MAIN_TRAIN_5_100_131072",
                     "GRU_D128": "MAIN_TRAIN_5_100_131072"}             # A-F7 (MAIN role); PRE: PRE_TRAIN_MANIFESTS
REFUSED_ROW_STATUS = ("FL_LR_SCREEN", "DEVICE_SELECTION", "HP_SEARCH", "FEATURE_SEARCH_SCREEN")   # A-F13
REFUSED_RUN_LABELS = ("SMOKE_NOT_EVIDENCE", "FEATURE_SEARCH_SCREEN")
C_BIG_PB_MARK = 3.0                                                     # 06 §7 "best@3.0 per the lock"
ARCH_LOCK_PATH = V4 / "contracts" / "ARCHITECTURE_LOCK.json"           # REV-DELTA-01 F9: C_BIG best.pt sha256 per seed
C_BIG_DEPLOYED_PB_ID = ":best"                                          # the deployed 02B_SASREC id form (no @mark)
G2_RUN_SHA256 = "3893a49a71be49f5baf338b93195e2e885eecb9326705902913bd4d5e1abb44d"   # company/gates/G2_SCRIPT_FREEZE.json
CONTRASTS_FORMAT_KEYS = ("confirmatory_rule_map",)
REGISTER_TOOL = V4 / "tools" / "test_exposure_register.py"
REGISTRY_V2_PATH = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2.json"
TEST_GATE_PURPOSE, TEST_GATE_SPLIT = "MANIFEST_BUILD", "RETRO_TEST"
FL_MARKS = tuple(0.5 * i for i in range(1, 13))
SUPPORT_BINS = ((0, 0, "0"), (5, 10, "5-10"), (11, 20, "11-20"), (21, 50, "21-50"), (51, 100, "51-100"))
X_LEVELS = (0.90, 0.95)
STUDY_A = ("C_SMALL", "FA", "FA_1024", "DP8", "DP4", "DP1")          # 06A §2 context-only (C_SMALL / FA / DP)
STUDY_B = ("LO", "FT_C", "FT_FA", "PF")                            # 06A §2 adaptation
BIN_PANELS = ("MAIN", "UNSEEN_HOLDOUT")
COLD_ROLE = "TRUE_COLD"
EPS_DEFAULT = {"FA_1024": "inf", "DP8": 8.0, "DP4": 4.0, "DP1": 1.0, "DP8_FROZEN": 8.0,
               "FA_1024_WARM": "inf", "FA_1024_WARM_FROZEN": "inf", "DP8_WARM": 8.0, "DP1_WARM": 1.0,   # ANA-06
               "DP1_WARM_FROZEN": 1.0}
NO_FT_ROLE = "UNSEEN_VAL"                                          # 06L-C §C16: the selection panel (never a headline)
NO_FT_ARMS = ("C_SMALL", "FA")                                     # the only arms admitted on UNSEEN_VAL (no-FT refs)
NO_FT_PAIRS = (("FT_C", "C_SMALL"), ("FT_FA", "FA"))               # the SELECTION value of the FT arm vs its base
CALIB_EVAL_ID = "CALIB_VALIDATION_5_100_32768"                     # G2 condition 5 / amendment 05 (CALIB-disjoint)
CALIB_REGISTERED_COUNTS = {"main_users": 51211, "calib_users": 12715, "disjoint_users": 38496}   # amendment 02 §2
CALIB_E2E_REGISTERED = {"main_users": 40764, "calib_users": 10070}   # DATA-06 L46 / L48: users with >= 1 eval row
CALIB_TUNED_ARMS = ("C_SMALL", "FA", "FP", "PF", "CPF", "FA_1024", "DP8", "DP4", "DP1", "FA_Q8", "FA_WARM",
                    "FA_WARM_FROZEN", "DP8_FROZEN", "FT_FA",       # 06L-C §C14: every arm tuned on CALIB users
                    "FT_FA_WARM", "FT_FA_WARM_FROZEN",             # ANA-05: their FL bases were screened on CALIB
                    "FP_WARM", "PF_WARM", "FA_Q8_WARM", "FA_1024_WARM", "DP8_WARM", "DP1_WARM",   # ANA-06: the warm
                    "FP_WARM_FROZEN", "PF_WARM_FROZEN", "FA_Q8_WARM_FROZEN", "FA_1024_WARM_FROZEN",   # factors come
                    "DP1_WARM_FROZEN")                                                       # from the S7 / S8 screens
DEPLOYMENT_ARMS = ("FA_WARM", "FA_WARM_FROZEN", "DP8_FROZEN", "PRE", "FT_PRE",   # 06K / 06L-C §C7 (C_FULL reference)
                   "FT_FA_WARM", "FT_FA_WARM_FROZEN",                              # + ANA-05 (06M_B B4)
                   "FP_WARM", "PF_WARM", "FA_Q8_WARM", "FA_1024_WARM", "DP8_WARM", "DP1_WARM",   # + ANA-06 (06N:
                   "FP_WARM_FROZEN", "PF_WARM_FROZEN", "FA_Q8_WARM_FROZEN",                    # the pretrained
                   "FA_1024_WARM_FROZEN", "DP1_WARM_FROZEN")                                  # line)
FAMILY_PAIR = ("GRU_D128", "C_SMALL")                               # 06L-C §C11 (descriptive, evaluation only)
PERSONALISATION = (("FT_C", "C_SMALL"), ("FT_FA", "FA"), ("PF", "FA"))
SIZE_ARMS = ("C_FULL", "C_BIG", "C_SMALL", "C_SMALL_FP32", "C_SMALL_INT8")
EPS_ARMS = ("FA_1024", "DP8", "DP4", "DP1")
TRUNC_H = (0, 1, 5, 10, 20, 50)
REFUSED, ERROR = 2, 5

# ------------------------------------------------------------------------------------ ANA-05 (06M, 06M_B B2 / B4 / B5)
FT_WARM_BASE = {"FT_FA_WARM": "FA_WARM", "FT_FA_WARM_FROZEN": "FA_WARM_FROZEN"}   # = launch_device.FT_WARM_BASE (FL-15)
WARM_BASE_KIND = "fa_warm"                                                        # their LAUNCH start base_kind
WARM_BASE_FILES = {"best.pt": "PRACTICAL_BEST", "endpoint_6.0.pt": "CONTROLLED_BUDGET"}   # the base checkpoint's rule
PRODUCTION_STACK = tuple(FT_WARM_BASE.items())      # 06M_B B4: paired delta FT_FA_WARM - FA_WARM (+ _FROZEN), DESCRIPTIVE
WARM_SELECTION_STAGE = {"FA_WARM": "S7", "FA_WARM_FROZEN": "S8"}    # = launch_fl.WARM_STAGE (06K; 06M_B B2)
WARM_SELECTION_FORMAT_2D = "PPSI_V4_FL_WARM_SELECTION_2"             # = launch_fl.WARM_SELECTION_FORMAT_2D (FL-15)
WARM_SELECTION_RECORD = {st: f"WARM_SELECTION_{st}" for st in ("S7", "S8")}   # records key of the registered file
WARM_SELECTION_RULE = ("a MAIN FA_WARM / FA_WARM_FROZEN run takes BOTH its local factor and its server-LR factor from the "
                       "registered 2-D contracts/fl/FL_WARM_SELECTION_S7|S8.json (PPSI_V4_FL_WARM_SELECTION_2; launch_fl "
                       "LAUNCH warm_selection; 06M_B B2): the entry's warm_selection = the LAUNCH binding, its winner "
                       "factors = the run's arm_spec factors, one file per stage in the block, and the file itself in "
                       "records (WARM_SELECTION_S7 / _S8) with sha256 / self_sha256 / winner / theta_pre = the binding; on "
                       "VALIDATION the file is required and every warm entry is bound to its run (a run folder, or a "
                       "REEVAL entry's warm_run + rows); on TEST every warm entry is a run bound on VALIDATION "
                       "(REV-ANA-05 F1 / F3)")
PREVIEW_LABEL = ("WARM_PREVIEW / S7-S8 CALIB screen run (06M M2; 06M_B B1-B2): DESCRIPTIVE CALIB values, never a block "
                 "entry, a selection or a contrast (the registered S7 / S8 screens decide only through "
                 "FL_WARM_SELECTION_S7|S8.json)")
PREVIEW_REFERENCE = "NON_MAIN_UNSEEN_VAL"       # 06O O1: v4int.reeval's preview_reference of a C_FULL / PRE UNSEEN_VAL row
PREVIEW_REFERENCE_ARMS = ("C_FULL", "PRE")      # = v4int.reeval.PREVIEW_SOURCES
PREVIEW_REFERENCE_LABEL = ("PREVIEW_REFERENCE NON_MAIN_UNSEEN_VAL (PREG1_AMENDMENT_06O O1-O2): C_FULL / PRE re-evaluated "
                           "on the non-MAIN new-user panel UNSEEN_VAL (12,792 eval users, rankable 10,237) are DESCRIPTIVE "
                           "preview rows, reported only in their "
                           "own table ('central models on never-trained new users'); never a block entry, a contrast, a "
                           "selection, a stopping rule or a claim")
DESCRIPTIVE_SEEDS = {"C_FULL": (2027,)}   # 06D §8 NICE seed; 06E §2 / DL 176 (v4int.reeval source_seed_labels, ANA-05)
DESCRIPTIVE_SEED_STATUS = "DESCRIPTIVE"
DESCRIPTIVE_SEED_RULE = ("06D §8 NICE seed; 06E §2 / DL 176: a run of the second seed of a single-seed contrast is "
                         "DESCRIPTIVE (K15, K17-K19, K20b, K20c stay seed 2026 confirmatory); it appears only in the "
                         "DESCRIPTIVE retention table and in a contrast's DESCRIPTIVE `extra_seeds` row")
ROLLOUT_ARMS = ("FA", "FA_WARM", "FA_WARM_FROZEN")     # 06M_B B5: quality during rollout (FA, FA_WARM, S8)
ROLLOUT_REFS = ("PRE", "LAST")
ROLLOUT_RULE = ("06M_B B5 (DESCRIPTIVE): per FL run (PRACTICAL_BEST entry's 12 marks) the macro-E2E MRR@20 per mark; "
                "AUC = trapezoid over the marks 0.5 .. 6.0 EFE (MRR x EFE), auc_mean = AUC / 5.5 (with its paired "
                "user-bootstrap CI, per resample on the same W); first mark reaching a reference (PRE-alone = theta_pre, "
                "seed 2026, PRACTICAL_BEST label; LAST = the seed-free rank file): POINT rule = the first mark whose "
                "value >= the reference's value, CI rule = the first mark whose paired delta 95% CI lies above 0; "
                "NOT_REACHED / NO_REFERENCE otherwise; seed mean of the per-seed AUC")
S8_REGISTERED_UPLOADED = 265162          # 06L-C §C8 "265,162 uploaded parameters (to be re-measured)"

# ------------------------------------------------------------------------------------ ANA-06 (06N; FL-16 warm method set)
LAUNCH_WARM_LINE = {   # = launch_fl.WARM_LINE (FL-16): launch_fl --arm label -> (the cold arm it runs, its warm variant)
    "FP_WARM": ("FP", "FA_WARM"), "PF_WARM": ("PF", "FA_WARM"), "FA_Q8_WARM": ("FA_Q8", "FA_WARM"),
    "S_CAL_WARM": ("S_CAL", "FA_WARM"), "FA_1024_WARM": ("FA_1024", "FA_WARM"), "DP_WARM": ("DP", "FA_WARM"),
    "FP_WARM_FROZEN": ("FP", "FA_WARM_FROZEN"), "PF_WARM_FROZEN": ("PF", "FA_WARM_FROZEN"),
    "FA_Q8_WARM_FROZEN": ("FA_Q8", "FA_WARM_FROZEN"), "S_CAL_WARM_FROZEN": ("S_CAL", "FA_WARM_FROZEN"),
    "FA_1024_WARM_FROZEN": ("FA_1024", "FA_WARM_FROZEN"), "DP_WARM_FROZEN": ("DP", "FA_WARM_FROZEN")}
WARM_LINE_ARMS = {   # the block arm (= the FL-16 run-dir name) -> (launch label, cold block arm, variant, dp epsilon)
    "FP_WARM": ("FP_WARM", "FP", "FA_WARM", None), "PF_WARM": ("PF_WARM", "PF", "FA_WARM", None),
    "FA_Q8_WARM": ("FA_Q8_WARM", "FA_Q8", "FA_WARM", None),
    "FA_1024_WARM": ("FA_1024_WARM", "FA_1024", "FA_WARM", None),
    "DP8_WARM": ("DP_WARM", "DP8", "FA_WARM", 8.0), "DP1_WARM": ("DP_WARM", "DP1", "FA_WARM", 1.0),
    "FP_WARM_FROZEN": ("FP_WARM_FROZEN", "FP", "FA_WARM_FROZEN", None),
    "PF_WARM_FROZEN": ("PF_WARM_FROZEN", "PF", "FA_WARM_FROZEN", None),
    "FA_Q8_WARM_FROZEN": ("FA_Q8_WARM_FROZEN", "FA_Q8", "FA_WARM_FROZEN", None),
    "FA_1024_WARM_FROZEN": ("FA_1024_WARM_FROZEN", "FA_1024", "FA_WARM_FROZEN", None),
    "DP8_FROZEN": ("DP_WARM_FROZEN", "DP8", "FA_WARM_FROZEN", 8.0),     # 06K §3 DP8_FROZEN = DP_WARM_FROZEN --epsilon 8
    "DP1_WARM_FROZEN": ("DP_WARM_FROZEN", "DP1", "FA_WARM_FROZEN", 1.0)}
WARM_LINE_EXCLUDED = ("S_CAL_WARM", "S_CAL_WARM_FROZEN")   # DONE_S_CAL (20 rounds, no weights): their S record enters
                                                           # through records S_RECORD_WARM / S_RECORD_WARM_FROZEN
WARM_LINE_ONLY = tuple(a for a in WARM_LINE_ARMS if a not in WARM_INIT_ARMS)   # the 11 new FL-16 block arms
WARM_START_ARMS = WARM_INIT_ARMS + WARM_LINE_ONLY                               # every theta_pre start (documentation)


def is_warm_start(arm: str) -> bool:
    """ANA-06: a theta_pre start = WARM_INIT_ARMS (REV-DELTA-03 M2, read at call time) or an FL-16 warm-line arm."""
    return arm in WARM_INIT_ARMS or arm in WARM_LINE_ONLY
WARM_LAUNCH_LABELS = ("FA_WARM", "FA_WARM_FROZEN") + tuple(LAUNCH_WARM_LINE)   # every launch_fl warm --arm label
METHOD_CODE_OF = {"FP": "FP", "PF": "PF", "FA_Q8": "FA", "FA_1024": "FA", "DP8": "FA", "DP1": "FA"}  # = launch_fl
POISSON_BASE = ("FA_1024", "DP8", "DP1")                  # the 06C Poisson arms (cfg.dp set; DP: z > 0)
S_FORMAT_WARM = "PPSI_V4_DP_S_RECORD_WARM_1"              # = launch_fl.S_FORMAT_WARM (FL-16)
S_RECORD_WARM_KEY = {"FA_WARM": "S_RECORD_WARM", "FA_WARM_FROZEN": "S_RECORD_WARM_FROZEN"}   # records key per variant
S_CAL_ROUNDS = 20                                                                # = launch_fl.KNOBS["s_cal_rounds"]
S_CAL_WARM_ARM = {"FA_WARM": "S_CAL_WARM", "FA_WARM_FROZEN": "S_CAL_WARM_FROZEN"}            # = launch_fl.S_CAL_WARM_ARM
STAR_TIE = 0.0005                                         # 06N N2: "ties within 0.0005 -> S8"
STAR_RULE = ("06N N2: FA_WARM* = the variant (S7 = FA_WARM full model, S8 = FA_WARM_FROZEN frozen tables) whose "
             "registered FL_WARM_SELECTION_S7|S8.json has the higher CALIB value of its selected cell "
             "(values[winner.label]); |S8 - S7| <= 0.0005 -> S8. Read from the two files only (CALIB values), never "
             "from a MAIN value; on TEST the VALIDATION_RESULT's choice (and the files, when given, must agree)")
N06_LABEL = ("REGISTERED_06P_DESCRIPTIVE_HERE: registered by PREG1_AMENDMENT_06N N1 / 06P, computed here as DESCRIPTIVE "
             "because the loaded contrasts file is not ANA_CONTRASTS_v4 (load v4 for the confirmatory labels)")
N06_NEEDS_REGISTRATION = (
    "06P P6 'Tertiary (unadjusted, Holm band beside)': implemented as a DESCRIPTIVE step-down band within the 13 "
    "tertiary contrasts (no decision); confirm",
    "06P names the FT_FA_WARM_FROZEN - FA_WARM_FROZEN contrast 'P5' (used as its id)")

# ------------------------------------------------------------------------------------ registered text (06B §2, verbatim)
REGISTERED_CONTRASTS = (
    ("K1", "R(C_SMALL / C_BIG)", "size cost"),
    ("K2", "R(FA / C_BIG)", "federated retention (headline)"),
    ("K3", "R(FA / C_SMALL)", "federation cost at matched size"),
    ("K4", "R(FP / C_BIG)", "FedProx retention"),
    ("K5", "R(PF / C_BIG)", "personalised-FL retention"),
    ("K6", "R(FT_FA / C_BIG)", "deployed on-device system (FL + local fine-tune)"),
    ("K7", "R(FT_C / C_BIG)", "cloud-trained + local fine-tune"),
    ("K8", "R(LO / C_BIG)", "device-only retention"),
    ("K9", "FT_C − C_SMALL", "local fine-tuning gain (paired Δ)"),
    ("K10", "FT_FA − FA", "local fine-tuning gain after FL (paired Δ)"),
    ("K11", "FA − LO", "collaboration value (paired Δ)"),
    ("K12", "R(DP8 / FA_1024)", "privacy cost at ε = 8"),
    ("K13", "R(DP1 / FA_1024)", "privacy cost at ε = 1 (seed 2026)"),
    ("K14", "R(FA_Q8 / FA)", "compression cost (8-bit uploads)"),
    ("K15", "R(C_BIG / C_FULL)", "cloud data advantage (seed 2026; if C_FULL runs)"),
    ("K16", "R(INT8 / FP32) for C_SMALL", "on-device quantisation cost (seed 2026)"),
    # 06I §1 (verbatim; registered in company/gates/ANA_CONTRASTS_v2.json)
    ("K17", "R(FA / C_FULL)", "federated retention vs the full-data cloud model (seed 2026)"),
    ("K18", "R(FT_C / C_FULL)", "cloud-trained + local fine-tune vs the full-data cloud model (seed 2026)"),
    ("K19", "R(LO / C_FULL)", "device-only vs the full-data cloud model (seed 2026)"),
    # 06K §2 / DL 206 (the formulas are registered; the question texts are ANA-03's wording of 06K's own descriptions)
    ("K20", "R(FA_WARM / C_BIG)", "deployment-branch federated retention vs the C_BIG cloud model (warm start from "
                                  "theta_pre; seeds 2026, 2027)"),
    ("K21", "FA_WARM − PRE", "value of federated training on the private users over the pretrained model alone "
                             "(paired Δ)"),
)
REGISTERED_CONTRASTS_06LC = (
    # 06L-C §C3 / §C7 (registered formulas; the question texts are the adopted review recommendations' own wording:
    # review 10 [9-R1 (c)] for K11b, [9-R2 (b), (c)] for K9b / K9c, DL 206 (1) + 06L-C §C7 for K20b / K20c)
    ("K9b", "FT_C − C_SMALL", "local fine-tuning gain (paired Δ) for base-unseen users (UNSEEN_HOLDOUT)"),
    ("K9c", "FT_PRE − PRE", "local fine-tuning gain (paired Δ) over θ_pre on MAIN: the realistic L5 answer (FT_PRE = "
                            "θ_pre + the registered FT_C procedure with SELECTION_FT_C)"),
    ("K11b", "FA − LAST", "collaboration value over the best own-history heuristic (paired Δ; LAST = the no-training "
                          "device heuristic)"),
    ("K20b", "R(FA_WARM / C_FULL)", "deployment-branch federated retention vs its matched cloud reference C_FULL "
                                    "(2 FA_WARM seeds vs C_FULL s2026)"),
    ("K20c", "R(PRE / C_FULL)", "the pretrained model alone vs its matched cloud reference C_FULL (PRE single seed)"),
)
REGISTERED_DESCRIPTIVE_06LC = (   # 06L-C §C8: "Paired contrast S8 − S7 reported with the bytes ratio" (review 09 [8-C2])
    ("S8-S7", "FA_WARM_FROZEN − FA_WARM", "cost of not sending the item tables (paired Δ, reported with the bytes "
                                          "ratio; DESCRIPTIVE)"),
)
CONTRAST_ORDER = ("K1", "K2", "K3", "K4", "K5", "K6", "K7", "K8", "K9", "K9b", "K9c", "K10", "K11", "K11b", "K12", "K13",
                  "K14", "K15", "K16", "K17", "K18", "K19", "K20", "K20b", "K20c", "K21", "S8-S7")
CONTRAST_FORMAT_V1 = "PPSI_V4_ANA_CONTRASTS_1"      # K1-K16: a VALIDATION dry check only (--allow-v1-contrasts)
CONTRAST_FORMAT_V2 = "PPSI_V4_ANA_CONTRASTS_2"      # K1-K19 (06I §1): company/gates/ANA_CONTRASTS_v2.json
CONTRAST_FORMAT_V3 = "PPSI_V4_ANA_CONTRASTS_3"      # 06L-C §C4: company/gates/ANA_CONTRASTS_v3.json (K1-K21 + 06L-C)
V3_IDS = tuple(f"K{i}" for i in range(1, 22)) + tuple(k for k, _c, _q in REGISTERED_CONTRASTS_06LC)
CONTRAST_IDS = {CONTRAST_FORMAT_V1: tuple(f"K{i}" for i in range(1, 17)),
                CONTRAST_FORMAT_V2: tuple(f"K{i}" for i in range(1, 20)), CONTRAST_FORMAT_V3: V3_IDS}
CONTRAST_COUNT = {f: len(v) for f, v in CONTRAST_IDS.items()}
V3_DESCRIPTIVE_IDS = tuple(k for k, _c, _q in REGISTERED_DESCRIPTIVE_06LC)
CONTRAST_POPULATION = {"K9b": "UNSEEN_HOLDOUT"}                      # every other registered contrast: MAIN
CONTRAST_SHARED_DEN = {"K20b": "C_FULL"}   # 06L-C §C7 "2 FA_WARM seeds vs C_FULL s2026": one C_FULL seed, every FA_WARM seed
PAIRING_NOTES = {
    "K9c": "PRE is ONE θ_pre (seed 2026) shared by both FT_PRE seeds (06K / DL 206 (2); 06L-C §C7)",
    "K11b": "LAST is seed-free (RULE_FREE, a BASELINE_CPU rank file, 06L-C §C3) and pairs with each FA seed of the rule",
    "K20b": "2 FA_WARM seeds vs C_FULL s2026 (06L-C §C7): the one registered C_FULL seed is the denominator of both",
    "K20c": "PRE single seed (2026) vs C_FULL s2026 (06L-C §C7)",
    "K21": "PRE is ONE θ_pre (seed 2026) shared by both FA_WARM seeds (06K / DL 206 (2))",
    "S8-S7": "FA_WARM_FROZEN seed s vs FA_WARM seed s (same θ_pre; 06L-C §C8: S8 runs with 2 seeds unconditionally)",
}
K14_MARGIN_REL_PCT = 1.0                   # 06L-C §C15: K14 equivalence margin ±1 % relative (registered before FA_Q8)
READING_RULES = {
    "K14": {"equivalence_margin_relative_percent": K14_MARGIN_REL_PCT, "interval": [100.0 - K14_MARGIN_REL_PCT,
                                                                                     100.0 + K14_MARGIN_REL_PCT],
            "rule": "TOST at alpha 0.05: R(FA_Q8 / FA) is EQUIVALENT to 100 iff its 90% paired user-bootstrap "
                    "percentile CI lies inside [99, 101] (06L-C §C15); the 95% CI and the ordinary claim rule are "
                    "reported beside it"},
}
ALGEBRAIC_DEPENDENCIES = (   # 06L-C §C4 ("K3 = K2/K1; K17-K19 = the matching K x K15") + the other exact identities
    {"id": "K3", "of": ["K2", "K1"], "relation": "K3_s = 100 x K2_s / K1_s (per seed and per resample, same W)"},
    {"id": "K9", "of": ["K7", "K1"], "relation": "K9_s = (K7_s - K1_s) x C_BIG_s / 100 (per seed)"},
    {"id": "K10", "of": ["K6", "K2"], "relation": "K10_s = (K6_s - K2_s) x C_BIG_s / 100 (per seed)"},
    {"id": "K11", "of": ["K2", "K8"], "relation": "K11_s = (K2_s - K8_s) x C_BIG_s / 100 (per seed)"},
    {"id": "K11b", "of": ["K11"], "relation": "K11b_s = K11_s + (LO_s - LAST) (per seed)"},
    {"id": "K17", "of": ["K2", "K15"], "relation": "K17 = K2(s2026) x K15 / 100: a composition, not independent "
                                                   "evidence"},
    {"id": "K18", "of": ["K7", "K15"], "relation": "K18 = K7(s2026) x K15 / 100"},
    {"id": "K19", "of": ["K8", "K15"], "relation": "K19 = K8(s2026) x K15 / 100"},
    {"id": "K20b", "of": ["K20", "K15"], "relation": "K20b_s = K20_s x C_BIG_s / C_FULL(s2026); at s2026 K20b = "
                                                     "K20(s2026) x K15 / 100"},
    {"id": "K20c", "of": ["K20b", "K21"], "relation": "K20b_s - K20c = 100 x K21_s / C_FULL(s2026)"},
)
DEPENDENCY_NOTES = {   # REV-DELTA-03 m6: printed beside the registered list (the v3 registry text itself is unchanged)
    "K11b": "shares LO with K11; LO - LAST is not a registered K, so K11b carries new evidence (it is NOT a composition "
            "of registered contrasts, unlike the other entries of this list)",
}
CONTRAST_ALIASES = {"INT8": "C_SMALL_INT8", "FP32": "C_SMALL_FP32"}
CONTRAST_FALLBACK = {"C_SMALL_FP32": "C_SMALL"}      # K16: DESCRIPTIVE, DEVICE_MIXED fallback when no ONNX FP32 rows exist

# ------------------------------------------------------------------------------------ registered text (06D §3 / §4)
REGISTERED_06D_3 = (
    "K12 = R(DP8 / FA_1024) and K13 = R(DP1 / FA_1024) are CONFIRMATORY under the pairing CONTROLLED_BUDGET / "
    "CONTROLLED_BUDGET (the round-384 endpoint of both arms: a data-independent release). Their PRACTICAL_BEST ratios "
    "are reported as DESCRIPTIVE with the disclosure \"best-of-12 selection on the users' own validation data is not "
    "covered by the accountant\". Every other contrast keeps PRACTICAL_BEST primary with CONTROLLED_BUDGET reported "
    "(06B §2).")
REGISTERED_06D_4 = "Primary headline: **K2**. Secondary: K1, K3, K8, K12. Tertiary: the rest."
DEFAULT_CONFIRMATORY_RULE = "PRACTICAL_BEST"
CONFIRMATORY_RULE = {"K12": "CONTROLLED_BUDGET", "K13": "CONTROLLED_BUDGET"}          # 06D §3 (default PRACTICAL_BEST)
HIERARCHY_06D = {"K2": "primary", "K1": "secondary", "K3": "secondary", "K8": "secondary", "K12": "secondary"}  # 06D §4
HIERARCHY = dict(HIERARCHY_06D, K17="secondary", K20="secondary", K21="secondary",   # + 06I §1 (K18 / K19 tertiary), 06K
                 K11b="secondary", K9b="secondary", K9c="secondary", K20b="secondary", K20c="secondary")  # + 06L-C C3, C7
DEFAULT_HIERARCHY = "tertiary"
DISCLOSURE_06D_3 = ("best-of-12 checkpoint selection on the users' own validation data is not covered by the DP "
                    "accountant (06D §3)")


def order_key(k: str) -> int:
    """The canonical contrast order (K9b / K9c after K9, K11b after K11, K20b / K20c after K20, S8-S7 last)."""
    return CONTRAST_ORDER.index(k) if k in CONTRAST_ORDER else len(CONTRAST_ORDER)


def ids_label(items: Sequence[str]) -> str:
    ids = sorted(items, key=order_key)
    plain = [k for k in ids if re.fullmatch(r"K\d+", k)]
    extra = [k for k in ids if k not in plain]
    head = f"K1-K{len(plain)}" if plain == [f"K{i}" for i in range(1, len(plain) + 1)] else ", ".join(plain)
    return head + (", " + ", ".join(extra) if extra else "")


def multiplicity_line(items: Sequence[str]) -> str:
    """The multiplicity statement for the loaded registry (06D §4; 06I §1; 06K; 06L-C C4 / C13): counts the contrasts it
    carries, names the Holm band and the algebraic dependencies."""
    sec = sorted((k for k in items if HIERARCHY.get(k) == "secondary"), key=order_key)
    return (f"{len(items)} pre-registered contrasts ({ids_label(items)}); unadjusted 95% paired user-bootstrap percentile "
            f"CIs (1,000 resamples, seed 20260926) with a DESCRIPTIVE Holm band beside them (normal intervals from the "
            f"bootstrap SE, family = these {len(items)}; 06L-C C13); hierarchy: K2 primary; {', '.join(sec)} "
            "secondary; the rest tertiary (06D §4; 06I §1; 06K; 06L-C); algebraic dependencies (K3 = K2/K1; K17-K19 = "
            "the matching K x K15; the rest in `algebraic_dependencies`) are not independent evidence"
            + (", except K11b, which shares LO with K11 but carries new evidence (LO - LAST is not a registered K)"
               if "K11b" in items else "") + "; every other number is DESCRIPTIVE")


MULTIPLICITY_06D_4 = multiplicity_line([f"K{i}" for i in range(1, 20)])          # the registered v2 file: 19 contrasts
REGISTERED_SEEDS = {"K13": (2026,), "K15": (2026,), "K16": (2026,), "K17": (2026,), "K18": (2026,), "K19": (2026,),
                    "K9b": (2026,), "K20c": (2026,)}     # 06B §2 / 06I §1 "(seed 2026)"; 06L-C C7; default SEEDS (A-F4)


def rule_map_from_06d(text: str = REGISTERED_06D_3) -> dict:
    """The per-contrast confirmatory rule map as written in 06D §3: every `Kn = R(...)` of the decision sentence is
    CONFIRMATORY under the `<RULE> / <RULE>` pairing it names."""
    ids = re.findall(r"\b(K\d+) = R\(", text)
    pair = re.search(r"under the pairing (\w+) / (\w+)", text)
    if not ids or not pair or pair.group(1) != pair.group(2) or pair.group(1) not in RULES:
        raise AnaRefused("the embedded 06D §3 text does not name one rule pairing")
    return {k: pair.group(1) for k in ids}


def hierarchy_from_06d(text: str = REGISTERED_06D_4) -> dict:
    prim = re.findall(r"K\d+", re.search(r"Primary headline:(.*?)\.", text).group(1))
    sec = re.findall(r"K\d+", re.search(r"Secondary:(.*?)\.", text).group(1))
    return {**{k: "primary" for k in prim}, **{k: "secondary" for k in sec}}

PRIVACY_LADDER = (   # 06B §3 (thesis framing), verbatim cells
    ("L0 cloud-train / cloud-infer", ("C_BIG", "C_FULL", "C_SMALL"), "raw events", "query context", "none"),
    ("L1 cloud-train / device-infer", ("C_SMALL_FP32", "C_SMALL_INT8", "KNN_EMB_SMALL", "KNN_EMB_BIG"),
     "raw events (model trained centrally once)", "nothing", "none"),
    ("L2 federated", ("FA", "FP", "FA_1024", "FA_Q8", "PF"), "model deltas (dense FP32; 8-bit for FA_Q8)", "nothing",
     "none (server sees clipped/unclipped deltas)"),
    ("L3 federated + DP", ("DP8", "DP4", "DP1"), "clipped deltas + Gaussian noise on the sum", "nothing",
     "user-level (ε, δ = 5e-6)-DP of the released model"),
    ("L4 federated + local personalisation", ("PF", "FT_FA"), "as L2; personal parameters never leave", "nothing",
     "none beyond L2"),
    ("L5 cloud model + local personalisation", ("FT_C",), "nothing after the download", "nothing",
     "none (download only)"),
    ("L6 device-only", ("LO", "RULE"), "nothing", "nothing", "data never leaves (shared catalogue/θ0 only)"),
)
LADDER_NOTE = ("Disclosed (CLIENT_CONTRACT C5.6): every arm shares the pooled bootstrap block (catalogue, transform "
               "statistics, θ0).")

# ------------------------------------------------------------------------------------ registered text (06L-C, verbatim)
REGISTERED_06LC = {   # contracts/PREG1_AMENDMENT_06L_C.md (whitespace-folded; checked against the file by the tests)
    "C1": ("Every arm is described by four columns: data seen at TRAINING by the server; data seen at INFERENCE by the "
           "server; formal guarantee; against whom. Placements corrected: FT_C on MAIN = \"L1 base (server-trained on raw "
           "events) + local fine-tune\"; FT_PRE = L5 (base trained only on opted-in users); PRE-alone = L1 on opted-in "
           "data; RULE / POP = \"device + published aggregate\"; LAST added to L6; a column \"secure aggregation: not "
           "simulated\" for every FL arm. Non-DP FL updates reveal the user's viewed items to the server (input-table rows "
           "and output rows move); S8 removes the input-table leak only partly (brand/category deltas may still leak)."),
    "C2": ("Adversary = an honest-but-curious browser vendor (the server); the retailer's own logs are out of scope. DP "
           "guarantee = central DP with a trusted aggregator, client-level add/remove adjacency, (ε, δ = 5e-6), "
           "conditional on the public bootstrap block (catalogue, transform statistics, θ0 / θ_pre from opted-in users); "
           "the FL hyper-parameter tuning on CALIB is not accounted."),
    "C3": ("Baseline runs also write per-row rank files. Reported for every arm: paired Δ(arm − LAST) with CI; headroom "
           "R* = (arm − LAST)/(C_ref − LAST); flag BELOW_DEVICE_HEURISTIC when the CI of Δ lies below 0. New "
           "confirmatory contrast **K11b = FA − LAST** (secondary). The device-only envelope {LO, LAST, RULE} is reported "
           "as one row group."),
    "C4": ("`company/gates/ANA_CONTRASTS_v3.json` = K1–K21 + K11b + (C7) + (C8), with the algebraic dependencies listed "
           "(K3 = K2/K1; K17–K19 = the matching K × K15) and without the word \"co-headline\" for K17."),
    "C5": ("The arm set, seeds, recipes and checkpoint rules for the VALIDATION preflight are exactly those registered up "
           "to this amendment plus the ladder's selected FL configuration; nothing else is confirmatory."),
    "C6": ("\"Matched-exposure retention of the registered FL recipe\" (not \"lower bound\"); the central-LR-shape waste "
           "is \"≈ 50 % of the exposure budget at ≤ 0.1× peak\", with the peak marks cited."),
    "C7": ("R(FA_WARM / C_FULL) and R(PRE / C_FULL) (2 FA_WARM seeds vs C_FULL s2026; PRE single seed), K20, K21, **K9c = "
           "FT_PRE − PRE on MAIN** (new arm FT_PRE: θ_pre + the registered FT_C procedure with SELECTION_FT_C, seeds "
           "2026/2027), **K9b = FT_C − C_SMALL on UNSEEN_HOLDOUT**. The Edge answer and the L5 row lead with these."),
    "C8": ("Paired contrast S8 − S7 reported with the bytes ratio; one bytes axis for every privacy level (raw events → "
           "model download → FL traffic → S8). S8 disclosure corrected: 265,162 uploaded parameters (to be re-measured), "
           "brand/category deltas may still leak, off-support rows move ≈ 1e-9, not exactly 0. S8 runs with 2 seeds "
           "unconditionally (supersedes 06K §3's conditional second seed)."),
    "C11": ("GRU_D128 (the contest's 02B_GRU best@6.0, 20.96M parameters, already trained) vs C_SMALL (20.71M), "
            "PRACTICAL_BEST and CONTROLLED_BUDGET, on the A40 block; never a downstream family (ARCHITECTURE_LOCK "
            "unchanged)."),
    "C12": ("A number-free abstract/headline template and a CLAIM register (allowed / forbidden sentence per contrast) are "
            "frozen together with the ANA script."),
    "C13-C15": ("Per-seed ratios, seed spread and a SEED_SENSITIVE flag beside every K; a Holm band beside the unadjusted "
                "CIs; a CALIB-disjoint diagnostic (MAIN \\ CALIB) for every CALIB-tuned arm; harmed-client share for every "
                "K; a K14 equivalence margin of ±1 % relative registered now (before any FA_Q8 read); repeat vs novel "
                "target stratum."),
    "C16-C17": ("Adopted: an evaluation-only \"no-FT\" reference of C_SMALL and FA on UNSEEN_VAL (descriptive); "
                "DP8_FROZEN (S8 + DP ε = 8, seed 2026, its z = 0 twin = S8) at SHOULD-high."),
}
POST_VALIDATION_NOTE = ("Anything added after the first VALIDATION read is POST_VALIDATION_EXPLORATORY (06L-C status "
                        "line; §C5): never confirmatory, never in the loaded registry.")
SECAGG_NOT_SIMULATED = "secure aggregation: not simulated"
_C2_DP = ("central DP with a trusted aggregator, client-level add/remove adjacency, (ε, δ = 5e-6), conditional on the "
          "public bootstrap block (catalogue, transform statistics, θ0 / θ_pre from opted-in users); the FL "
          "hyper-parameter tuning on CALIB is not accounted (06L-C §C2)")
_LEAK = ("model deltas; non-DP deltas reveal the user's viewed items (input-table rows and output rows move; 06I §7(a), "
         "06L-C §C1)")
PRIVACY_MATRIX = (   # 06L-C §C1 (replaces the one-column 06B §3 ladder for the thesis; the 06B §3 table is still printed)
    # (level, arms, data seen at TRAINING by the server, data seen at INFERENCE by the server, formal guarantee,
    #  against whom, secure aggregation)
    ("L0 cloud-train / cloud-infer", ("C_BIG", "C_FULL", "C_SMALL", "GRU_D128", "SKNN"), "raw events",
     "query context", "none", "-", "n/a (no federated training)"),
    ("L1 cloud-train / device-infer", ("C_SMALL_FP32", "C_SMALL_INT8", "KNN_EMB_SMALL", "KNN_EMB_BIG"),
     "raw events (model trained centrally once)", "nothing", "none", "-", "n/a (no federated training)"),
    ("L1 on opted-in data (PRE-alone)", ("PRE",), "raw events of the opted-in users only (PRETRAIN_TRAIN_5_100_NONMAIN); "
     "none of the MAIN users", "nothing", "none (the MAIN users' data is never used)", "-", "n/a (no federated training)"),
    ("L1 base (server-trained on raw events) + local fine-tune", ("FT_C",), "raw events of the same users (the base) "
     "and nothing after the download", "nothing", "none", "-", "n/a (no federated training)"),
    ("L2 federated", ("FA", "FP", "FA_1024", "FA_Q8"), _LEAK, "nothing", "none",
     "the honest-but-curious browser vendor sees every delta (06L-C §C2)", SECAGG_NOT_SIMULATED),
    ("L2 + consenting-cohort pretraining", ("FA_WARM",), _LEAK + "; θ_pre from opted-in users", "nothing", "none",
     "the honest-but-curious browser vendor sees every delta (06L-C §C2)", SECAGG_NOT_SIMULATED),
    ("L2 + frozen item tables (S8)", ("FA_WARM_FROZEN",), "encoder deltas only (265,162 uploaded parameters, to be "
     "re-measured); the input-table item-set leak is removed only partly: brand/category deltas may still leak, "
     "off-support rows move ≈ 1e-9, not exactly 0 (06L-C §C8)", "nothing", "none",
     "the honest-but-curious browser vendor sees every delta (06L-C §C2)", SECAGG_NOT_SIMULATED),
    ("L3 federated + DP", ("DP8", "DP4", "DP1", "DP8_FROZEN"), "clipped deltas + Gaussian noise on the sum", "nothing",
     _C2_DP, "anyone who sees the released model; not the trusted aggregator itself", SECAGG_NOT_SIMULATED),
    ("L4 federated + local personalisation", ("PF", "FT_FA", "FT_FA_WARM", "FT_FA_WARM_FROZEN"),
     "as L2; personal parameters never leave", "nothing", "none beyond L2",
     "the honest-but-curious browser vendor sees every shared delta", SECAGG_NOT_SIMULATED),
    ("L5 (base trained only on opted-in users) + local fine-tune", ("FT_PRE",), "nothing of the MAIN users (θ_pre from "
     "opted-in users; download only)", "nothing", "none needed for the MAIN users (their data never leaves the device)",
     "-", "n/a (no federated training)"),
    ("L6 device-only", ("LO", "LAST"), "nothing", "nothing", "data never leaves (shared catalogue / θ0 only)", "-",
     "n/a (no federated training)"),
    ("device + published aggregate", ("RULE", "POP"), "nothing of the user (a published MAIN-TRAIN popularity / "
     "category aggregate is downloaded)", "nothing", "none for the users in the aggregate", "-",
     "n/a (no federated training)"),
)

DISCLOSURES_06A_9 = (
    "9. **Disclosures registered for the thesis:** (a) the 6.0-endpoint reversal of the family order (G2 condition 5); "
    "(b) the FL local LR is the single SMALL peak (the {1×, 0.5×} grid of the realistic plan was not carried into "
    "amendment 06 or the code — untuned, like μ); (c) FedBN is inapplicable (LayerNorm backbone; FedLN not added by the "
    "owner); (d) purchase-AUC / comparable-offer tasks of the proposal are out of scope by agreement (next-item task); "
    "(e) battery is not measured; (f) per-client batching in launch_device reorders 6 / 1,251 rows by one rank position "
    "at ranks > 20 (no effect on @≤20 metrics); (g) VALIDATION values are max-of-marks; TEST is the only unbiased number.")
DISCLOSURES_06B_7 = (
    "(a) The review's proposal-bibliography corrections ([5] title; [6] IEEE Intelligent Systems 36(5) 2021; [8] Luo et al. "
    "CIKM 2022 vs Zhang et al. IJCAI 2023 PFedRec; [12] unverified) go into the thesis bibliography. (b) RETRO_TEST contains "
    "Black Friday 29 Nov 2019 — the VALIDATION→TEST change of C_BIG is reported as a shift indicator. (c) The contest selects "
    "the backbone under rule T1; never phrased as \"SASRec beats GRU\". (d) Downstream arms trained on A40; the contest on A16. "
    "(e) UNSEEN_VAL is a selection panel, never a headline panel; applying the FT recipe to TRUE_COLD/SPARSE is an extrapolation.")
DISCLOSURE_06B_1 = "FL local LR screened on CALIB (3 points); μ and server LR (1.0) untuned."
DISCLOSURES_06K = (   # DL 206 (Fable review of 06K), registered before any PRE / FA_WARM value exists
    "(1) PRE is trained on 1,217,610 users (9x MAIN) while C_BIG saw 131,072: K20 = R(FA_WARM / C_BIG) can exceed 100 and "
    "mixes a data advantage into the FL retention. The matched cloud reference of the deployment branch is C_FULL "
    "(TIEBREAK = PRE users + MAIN users): R(FA_WARM / C_FULL) and R(PRE / C_FULL) are printed beside K20 as DESCRIPTIVE, "
    "and the thesis text for the deployment branch leads with them. K21 = FA_WARM - PRE stays the value of federated "
    "training on the private users.",
    "(2) theta_pre is ONE seed (2026) shared by FA_WARM s2026 and s2027: their seed spread measures FL variance only.",
    "(3) PRE applies the SMALL recipe (tuned at MAIN scale, 2.6M decisions) to 24.1M decisions at 6.0 EFE (about 9x more "
    "updates) without re-tuning: this can only understate PRE. PRE marks on UNSEEN_VAL are DESCRIPTIVE and never select.",
)
DISCLOSURE_BYTES_06I_7B = (
    "06I §7(b) (variant registered in DECISION_LOG row 218): bytes per client are reported with the registered "
    "definition (06A §3: visits x (down + up), dropped clients not charged) and, beside it, with the dropped visits "
    "charged their download (a dropped client receives the model and never uploads): bytes_per_client_charged_dropped "
    "= visits x (down + up) + dropped visits x down per client (client mean). The variant that charges a dropped visit "
    "down + up (bytes_per_client_charged_dropped_up_down) is DESCRIPTIVE only. The curves carry the registered charged "
    "variant as a second series (the up + down variant as a DESCRIPTIVE third series).")
DISCLOSURE_NOTES = (
    "06A §9(b) is superseded for the FL arms by amendment 06B §1 (the local-LR screen); both are printed as registered.",
    "Support bins (06A §2) are observational: history length is not manipulated.",
    "Every number outside the confirmatory section (the loaded registry: K1-K19 / K1-K21) is DESCRIPTIVE.",
    "06E §9 (e) The seed-mean retention is the mean of per-seed ratios (06A §7 reading). (f) The contrast-text check "
    "folds whitespace and dash glyphs. (g) FL cumulative bytes at a mark are the fleet mean prorated by rounds.",
    "Device arms (LO, FT_C, FT_FA) and the ONNX rows exist under PRACTICAL_BEST only: K6-K11, K16, K18 and K19 have no "
    "CONTROLLED_BUDGET row (06E §5; REV-ANA-01 A.1 DISCLOSE).",
    "Only retained% is reported; loss_b = 100 - retained_b per resample, so its CI is the mirror interval (E7.5).",
)

DISCLOSURES_ANA04 = (   # how the 06L-C analysis items are computed (DESCRIPTIVE unless stated)
    "06L-C C3: LAST enters as a seed-free BASELINE_CPU rank file (v4base.run ranks_int32.npy, CPU-scored exact ranks on "
    "the same manifest rows; the key differs only in eval_device_class / precision_id / objective_id placeholders); "
    "Δ(arm − LAST) pairs every seed of the arm with the one LAST file on the same W; R* per resample = (arm_b − LAST_b) "
    "/ (C_BIG_b − LAST_b), seed mean of the per-seed R*; a resample with C_BIG_b <= LAST_b makes the R* CI UNDEFINED.",
    "06L-C C13: the seed spread is max - min of the per-seed points; SEED_SENSITIVE = spread > half the width of the "
    "user-bootstrap CI of the seed mean; the user + seed interval = point ± 1.96 x sqrt(var_boot + s²_seed / n_seeds) "
    "(DESCRIPTIVE; the percentile CIs stay conditional on the trained models).",
    "06L-C C13: the Holm band uses normal intervals from the bootstrap SE (sd of the seed-mean resamples): z = (point - "
    "ref) / SE, two-sided p, Holm step-down over the loaded registry's family (the registered count, so a missing "
    "contrast never shrinks the family), band = point ± z_(1 - α_j / 2) x SE at the contrast's Holm level α_j for a "
    "rejected contrast and at α / (m - r) (r = the number of rejections) for every contrast from the first "
    "non-rejection on, so a band excludes the reference exactly when Holm rejects (REV-DELTA-03 M3); the "
    "Bonferroni band (α / m) is printed beside it. It corrects 06D §4's \"not resolvable with 1,000 resamples\" (true "
    "for 99.7% percentile endpoints only).",
    "06L-C C14: MAIN \\ CALIB = the MAIN E2E users whose client_key_hex is not a CALIB_VALIDATION user (the CALIB "
    "manifest's users table); on VALIDATION the counts must be the registered 51,211 / 12,715 / 38,496 when the "
    "production loader supplies them. Report only, never selection.",
    "06L-C C15: the harmed-client share of a ratio contrast is the share of users whose per-user macro difference "
    "(numerator - denominator, seed mean) is < 0; the seed-vs-seed null is the same share for s2027 - s2026 of the "
    "same arm on the same users (a noise floor).",
    "06L-C C15: REPEAT rows = RANKABLE rows whose target is in the user's L50 window, read from LAST's per-row tie count "
    "(LAST scores the window's distinct items by a unique recency, every other class ties at the sentinel, so the "
    "target is in the window iff its tie count is 0); NOVEL = every other E2E row (OOV targets included).",
    "06L-C C8: the raw-event level of the bytes axis needs a registered per-event size (records.EVENT_SIZE, a DL row); "
    "without it that level reads NOT_REGISTERED. TRAIN decisions per client come from the MAIN TRAIN-layout manifest "
    "(a design quantity, not a result). FL levels add the final model download to the training traffic.",
    "06L-C C16: the no-FT UNSEEN_VAL references are C_SMALL / FA re-evaluated on the selection panel; the FT value beside "
    "them is the SELECTION record's value of the selected cell (a max over cells: optimistic); UNSEEN_VAL users are "
    "base-unseen, MAIN users base-seen (review 10 [9-R3]).",
    # REV-DELTA-03 m7 (disclosed, not changed)
    "06L-C C17 (followed literally): DP8_FROZEN's \"z = 0 twin = S8\" differs from DP8_FROZEN in cohort and round count "
    "(the DP plan's 1,024 clients per round / 384 rounds vs S8's FA plan), so R(DP8_FROZEN / S8) mixes the DP noise "
    "with that change of cohort and rounds; it is not a pure privacy cost.",
    "GATE-B B3 (INFERENCE): the S8 bytes curve charges the fleet-mean one-time frozen-table download in full from the "
    "first mark and prorates only the training traffic, so the early marks overstate S8's cumulative bytes "
    "(conservative against S8).",
    "06L-C C8: the raw-event level of the bytes axis uses the client's MAIN TRAIN decisions as its event count (a "
    "design quantity from the TRAIN-layout manifest, not the raw event log), times the registered per-event size.",
)

DISCLOSURES_ANA05 = (   # how the ANA-05 items are computed (all DESCRIPTIVE; the v3 confirmatory set is unchanged)
    "ANA-05 C_FULL s2027 (06D §8 NICE seed; 06E §2 / DL 176): admitted as a DESCRIPTIVE source seed; its v4int.reeval "
    "rows carry comparison_status / source_seed_status DESCRIPTIVE; K15, K17-K19, K20b and K20c stay seed 2026 (the "
    "s2027 pairs appear only in the retention table and in a contrast's DESCRIPTIVE extra_seeds row).",
    "ANA-05 06M_B B2: an FA_WARM / FA_WARM_FROZEN MAIN run must carry launch_fl's LAUNCH warm_selection (the 2-D "
    "FL_WARM_SELECTION_S7|S8.json) whose winner factors equal its arm_spec factors; the registered file itself, when "
    "given in records, must equal that binding. Rows-only / REEVAL entries (no run dir) carry no LAUNCH binding and are "
    "listed as such.",
    "ANA-05 06M_B B4: FT_FA_WARM(_FROZEN) = theta_pre -> FA_WARM (S8) MAIN -> the registered FT_C procedure; its base "
    "(start_base) must be the block's base-arm run of the same seed; the paired delta FT_FA_WARM - FA_WARM (+ _FROZEN) is "
    "reported with the harmed-client share (06L-C C13-C15), PRACTICAL_BEST (a best.pt base) only; DESCRIPTIVE.",
    "ANA-05 06M_B B5: 'quality during rollout' for FA, FA_WARM and S8 from each run's 12 marks: trapezoid AUC of the "
    "macro MRR@20 over 0.5 .. 6.0 EFE and the first mark reaching PRE-alone / LAST (a point rule and a paired-CI rule); "
    "DESCRIPTIVE, VALIDATION marks only (the marks are the same user set as the PRACTICAL_BEST selection).",
    "ANA-05 06L-C C8: S8's bytes per visit, upload / download bytes, uploaded and frozen parameter counts come from the "
    "run's own fl.comm record (RUN_SUMMARY bytes.per_visit); the registered '265,162 uploaded parameters (to be "
    "re-measured)' is printed beside the measured count.",
    "ANA-05 06M M2: WARM_PREVIEW runs (and the S7 / S8 CALIB screens) are CALIB runs: they are refused as block entries "
    "(index and report) and never enter any contrast.",
)

# ------------------------------------------------------------------------------------ C12: claim register (frozen text)
CLAIM_REGISTER_FORMAT = "PPSI_V4_ANA_CLAIM_REGISTER_1"
CLAIM_GENERAL_RULES = (
    "A exceeds B only when the paired 95% CI of the contrast lies above its reference (0 for a paired Δ, 100 for a "
    "retention R); an interval that includes the reference reads 'no detectable difference' (the claim vocabulary of "
    "EVALUATION_CONTRACT_V4 §7), never 'no difference' or 'equivalent': an absence of evidence is not evidence of "
    "absence, and only a registered equivalence test (K14's margin, 06L-C C15) can support 'equivalent'.",
    "Only the loaded registry's contrasts (report section 1, CONFIRMATORY rows under their registered rule and seeds) "
    "carry confirmatory wording; every other number is DESCRIPTIVE and is never phrased as a finding.",
    "Every retention sentence names the recipe and the reference: 'matched-exposure retention of the registered FL "
    "recipe' (06L-C C6), relative to a cloud model trained on 5-100-event users; never 'FL retains X %' without the "
    "recipe (06I §6(c)); never 'a lower bound' (06L-C C6).",
    "VALIDATION values are max-of-marks; the single TEST look is the only unbiased number (06A §9(g)).",
    "A contrast flagged SEED_SENSITIVE carries the flag in the same sentence; every CI covers user sampling only, "
    "conditional on the trained models.",
    "An arm flagged BELOW_DEVICE_HEURISTIC (CI of Δ(arm − LAST) below 0) carries the flag in the same sentence as its "
    "retention (06L-C C3).",
    "Non-DP federated arms are never called private: the uploaded deltas reveal the viewed items and secure aggregation "
    "is not simulated (06I §7(a); 06L-C C1); a DP arm states the 06L-C C2 guarantee in full.",
    "K17-K19 and K20b are compositions with K15 (see algebraic_dependencies); they are never presented as independent "
    "evidence or as a second headline (06L-C C4).",
    "The backbone was selected under rule T1; never 'SASRec beats GRU' (06B §7(c)); GRU_D128 vs C_SMALL is DESCRIPTIVE "
    "and never a family choice (06L-C C11).",
    "Anything added after the first VALIDATION read is POST_VALIDATION_EXPLORATORY and never confirmatory (06L-C C5).",
    "An 'events to parity' (or events-to-X) sentence is written only from a persistent hit of the 06L-C C10 rule (the "
    "parity holds at the next registered h too, with the CI lower bound >= X), naming the rule and the reference "
    "(C_BIG FULL history or matched-h C_BIG(h)); a first-hit h alone, NO_CI, NO_NEXT_H or NO_C_REF supports no such "
    "sentence; the curves are DESCRIPTIVE.",
)
CLAIM_TEMPLATES = {
    "retained": {
        "ABOVE": "{num} exceeds {den} on {population}: R = {point} % [{ci_lo}, {ci_hi}] ({rule}; seeds {seeds}).",
        "BELOW": "{num} retains {point} % [{ci_lo}, {ci_hi}] of {den} on {population} ({rule}; seeds {seeds}).",
        "INCLUDES": "{num} vs {den} on {population}: R = {point} % [{ci_lo}, {ci_hi}] includes 100: no detectable "
                    "difference ({rule}; seeds {seeds}).",
    },
    "delta": {
        "ABOVE": "{num} exceeds {den} on {population} by {point} [{ci_lo}, {ci_hi}] macro MRR@20 ({rule}; seeds {seeds}).",
        "BELOW": "{num} is below {den} on {population} by {abs_point} [{ci_lo}, {ci_hi}] macro MRR@20 ({rule}; seeds "
                 "{seeds}).",
        "INCLUDES": "{num} - {den} on {population} = {point} [{ci_lo}, {ci_hi}] macro MRR@20 includes 0: no detectable "
                    "difference ({rule}; seeds {seeds}).",
    },
    "NOT_AVAILABLE": "{id} was not computed ({status}); no sentence about '{question}' is written.",
    "UNDEFINED": "{id} is UNDEFINED ({status}); no sentence about '{question}' is written.",
}
CLAIM_FORBIDDEN = {   # per contrast (on top of CLAIM_GENERAL_RULES); every entry is a sentence the thesis never writes
    "K1": ["'the small model loses nothing' (unless ABOVE or a registered equivalence test)"],
    "K2": ["'federated learning retains X %' without the registered recipe (06I §6(c))",
           "'a lower bound of what federated learning can reach' (06L-C C6)", "'federated learning is private'"],
    "K3": ["'the federation cost is independent of K2' (K3 = K2 / K1)"],
    "K4": ["'FedProx helps / hurts' in general (06L-A A7: FedProx μ 0.01 under the selected client optimiser only)"],
    "K5": ["'personalisation is useless' from PF − FA ≈ 0 (06L-A A5: PF ≈ FA by construction is possible)"],
    "K6": ["'the deployed on-device system is private' (FL deltas leak viewed items; no secure aggregation)"],
    "K7": ["'privacy-preserving fine-tuning' (FT_C on MAIN = L1 base + local fine-tune, 06L-C C1)"],
    "K8": ["'the device-only ceiling' (LO is an untuned from-θ0 control; the device-only answer is the envelope "
           "{LO, LAST, RULE})"],
    "K9": ["'the L5 fine-tuning gain' on MAIN (the base saw these users; the L5 answers are K9c and K9b)"],
    "K9b": ["'fine-tuning helps every base-unseen user' (read the harmed-client share)"],
    "K9c": ["'a formal privacy guarantee' (no DP; L5 = the MAIN users' data never leaves the device)"],
    "K10": ["'fine-tuning after FL is free' (report the harmed-client share)"],
    "K11": ["'collaboration beats every device-only method' (LO only; see K11b and the envelope)"],
    "K11b": ["'federated learning beats doing nothing clever on the device' unless ABOVE",
             "'federated learning is worse than the device heuristic' unless BELOW (then with BELOW_DEVICE_HEURISTIC)"],
    "K12": ["'the privacy cost at ε = 8 is confirmed' from the PRACTICAL_BEST ratio (DESCRIPTIVE, 06D §3)"],
    "K13": ["'the privacy cost at ε = 1 is confirmed' from the PRACTICAL_BEST ratio (DESCRIPTIVE, 06D §3; one seed)"],
    "K14": ["'8-bit uploads cost nothing' unless EQUIVALENT under the registered margin (90% CI inside [99, 101], "
            "06L-C C15)"],
    "K15": ["'the cloud data advantage explains the federated gap' (K17-K19 are compositions, not a decomposition "
            "test)"],
    "K16": ["'on-device quantisation cost' from the DEVICE_MIXED fallback (confirmatory only inside ONNX_CPU)"],
    "K17": ["'K17 replicates K2' or any second-headline wording (06L-C C4: K17 = K2(s2026) x K15 / 100)"],
    "K18": ["'K18 is independent of K7' (K18 = K7(s2026) x K15 / 100)"],
    "K19": ["'K19 is independent of K8' (K19 = K8(s2026) x K15 / 100)"],
    "K20": ["'warm-started FL beats the cloud' from R > 100 (PRE saw 9x the MAIN users; DL 206 (1))"],
    "K20b": ["'trained on the same data as the cloud' (C_FULL = the TIEBREAK band = PRE users + MAIN users)"],
    "K20c": ["'the pretrained model needs no federated training' (unconditional: K21 including 0 is an absence of "
             "evidence, and no equivalence test is registered for K21; REV-DELTA-03 m5)"],
    "K21": ["'federated training adds value over θ_pre' unless ABOVE"],
    "S8-S7": ["'S8 leaks nothing' (brand/category deltas may still leak; off-support rows move ≈ 1e-9; 06L-C C8)",
              "any confirmatory wording (S8 − S7 is a registered DESCRIPTIVE contrast)"],
}
HEADLINE_TEMPLATE = (   # number-free (06L-C C12); placeholders are filled only from the RETENTION_RESULT of the TEST look
    "# Headline / abstract template (number-free; 06L-C C12)\n\n"
    "Placeholders `{Kx}`, `{Kx_CI}`, `{Kx_CLAIM}` are filled from the frozen RETENTION_RESULT.json of the single TEST "
    "look (`{Kx_CLAIM}` = the claim-register template key stamped on that K); the wording follows CLAIM_REGISTER.json.\n\n"
    "1. Task. Next different viewed item within the session (views only), given the user's prior events across "
    "sessions (USER_HISTORY_L50); every retention is relative to a cloud model trained on 5-100-event users.\n"
    "2. Cloud. The C_BIG cloud model reaches {K15} % [{K15_CI}] of the full-data cloud model C_FULL (K15, one seed).\n"
    "3. Federated (primary, K2). The matched-exposure retention of the registered FL recipe <{FL_RECIPE}> is {K2} % "
    "[{K2_CI}] of C_BIG ({K2_CLAIM}); against the no-training device heuristic LAST, FA − LAST = {K11b} [{K11b_CI}] "
    "({K11b_CLAIM}); headroom R* = {FA_R_STAR} %.\n"
    "4. Federated + DP. At ε = 8 (central DP, trusted aggregator, client-level, δ = 5e-6, conditional on the public "
    "bootstrap block; tuning not accounted) the model keeps {K12} % [{K12_CI}] of its non-private twin (K12, "
    "CONTROLLED_BUDGET).\n"
    "5. Cloud model shipped + fine-tuned on device. FT_C on MAIN is an L1 base + local fine-tune: K7 = {K7} % [{K7_CI}]; "
    "the L5 answers are K9c = FT_PRE − PRE = {K9c} [{K9c_CI}] and, for base-unseen users, K9b = {K9b} [{K9b_CI}].\n"
    "6. Device-only. The device-only envelope (LO, LAST, RULE) retains {ENVELOPE_R} % of C_BIG; LO alone (an untuned "
    "from-θ0 control) {K8} % [{K8_CI}].\n"
    "7. Deployment branch (the Edge answer). Warm-started federation retains {K20b} % [{K20b_CI}] of its matched cloud "
    "reference C_FULL (K20b), the pretrained model alone {K20c} % [{K20c_CI}] (K20c); federated training on the private "
    "users adds {K21} [{K21_CI}] over θ_pre (K21); freezing the item tables costs {S8_S7} [{S8_S7_CI}] at {S8_BYTES_RATIO} "
    "of the bytes (S8 − S7, DESCRIPTIVE).\n"
    "8. Privacy. The adversary is an honest-but-curious browser vendor; non-DP federated updates reveal the viewed "
    "items and secure aggregation is not simulated; only the DP arms carry a formal guarantee (06L-C C1, C2).\n"
)


# ------------------------------------------------------------------------------------ ANA-06: the v4 registry (06P)
# contracts/PREG1_AMENDMENT_06P.md P1-P7 (REGISTERED 2026-10-01, DL 316-317). company/gates/ANA_CONTRASTS_v4.json (format
# PPSI_V4_ANA_CONTRASTS_4) is checked against these constants exactly as v3 is against the v3 ones; a file whose `draft`
# flag is not false loads only with --allow-draft-contrasts on VALIDATION. With v1-v3 loaded no table below is in force.
P06_FILE_SHA256 = "75f8dbade9d48221a4677dd9156308a57f313b030beb6b18cbb38a9d83c8b7a5"   # contracts/PREG1_AMENDMENT_06P.md
CONTRAST_FORMAT_V4 = "PPSI_V4_ANA_CONTRASTS_4"
P06_TEXT = (   # contracts/PREG1_AMENDMENT_06P.md P1-P7, verbatim (REGISTERED 2026-10-01; + ERRATUM 1, DL 324)
    '**P1. Resolution of FA_WARM\\*.** FA_WARM\\* = FA_WARM_FROZEN (S8), by 06N N2 on the registered CALIB selections (DL 292). In K20, K20b\nand K21 the arm `FA_WARM` is read as `FA_WARM_FROZEN`: K20 = R(FA_WARM_FROZEN / C_BIG), K20b = R(FA_WARM_FROZEN / C_FULL) (2 seeds\nvs C_FULL s2026), K21 = FA_WARM_FROZEN − PRE (paired Δ, 2 seeds, PRACTICAL_BEST). **Primary = the pair {K20b, K21}**, read unadjusted\nand reported together; a claim about the deployment branch requires both. K2 = R(FA / C_BIG) becomes SECONDARY ("random-start\nablation"). K20c = R(PRE / C_FULL) unchanged (secondary, seed 2026).',
    '**P2. The S7 variant.** FA_WARM (full-model warm start) keeps 2 MAIN seeds. Its rows are reported as the registered descriptive\ncomparison S8 − S7 (06L_C C8, with the bytes ratio) and as DESCRIPTIVE R rows (R(FA_WARM / C_FULL), FA_WARM − PRE) beside the\nprimary; no confirmatory contrast is defined on FA_WARM.',
    '**P3. K_INIT = FA_WARM_FROZEN − FA** (paired Δ, seeds 2026/2027, PRACTICAL_BEST, CONTROLLED_BUDGET beside; "value of the pretrained\nstart"), SECONDARY, in the Holm family.',
    '**P4. Warm-line method contrasts** (all on MAIN, the warm reference = C_FULL per DL 273): K4w = R(FP_WARM_FROZEN / C_FULL), K5w =\nR(PF_WARM_FROZEN / C_FULL) (2 seeds, PRACTICAL_BEST, SECONDARY; PF_WARM_FROZEN − FA_WARM_FROZEN DESCRIPTIVE with the 06L A5 caveat);\nK12w = R(DP8_FROZEN / FA_1024_WARM_FROZEN) (2 seeds, CONTROLLED_BUDGET / CONTROLLED_BUDGET, SECONDARY; PRACTICAL_BEST ratio\nDESCRIPTIVE with the 06D §3 disclosure); K13w = R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN) (seed 2026, CONTROLLED_BUDGET, SECONDARY);\nK14w = R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN) (2 seeds, TOST ±1 % relative as 06L_C C15, SECONDARY); R(FA_1024_WARM_FROZEN /\nFA_WARM_FROZEN) DESCRIPTIVE (clipping + Poisson-cohort cost). The cold K4, K5, K12, K13, K14 keep their text and tiers on the N33\nre-runs.',
    "**P5. FT_FA_WARM_FROZEN − FA_WARM_FROZEN** (paired Δ, 2 seeds, PRACTICAL_BEST, base = the same-seed best.pt, harmed-client share\nbeside) is SECONDARY (06N N1 supersedes 06M_B B4's DESCRIPTIVE); FT_FA_WARM − FA_WARM stays DESCRIPTIVE.",
    '**P6. Multiplicity (corrected, Fable §5 concurrence).** The primary pair {K20b, K21} is read unadjusted, outside Holm. Holm family\n(secondary, 19): K1, K2, K3, K8, K9b, K9c, K11b, K12, K17, K20, K20c, K_INIT, P5 (FT_FA_WARM_FROZEN − FA_WARM_FROZEN), K4w, K5w,\nK12w, K13w, K14w. Tertiary (unadjusted, Holm band beside): K4, K5, K6, K7, K9, K10, K11, K13, K14, K15, K16, K18, K19. The 06L_C\nC13 convention is unchanged (unadjusted 95 % paired bootstrap CIs, Holm band beside; K14 / K14w TOST ±1 % on the unadjusted 90 %\nCI). DESCRIPTIVE, outside every family: S8 − S7; the S7 R rows; PF_WARM_FROZEN − FA_WARM_FROZEN; R(FA_1024_WARM_FROZEN /\nFA_WARM_FROZEN); the PRACTICAL_BEST ratios of K12 / K13 / K12w / K13w. registered_seeds = the v3 map + K13w [2026]; every other new\nid [2026, 2027]; no seed cuts.',
    '**P7.** `ANA_CONTRASTS_v4.json` carries P1–P6 verbatim, the algebraic dependencies (K20b_s = K20_s × C_BIG_s / C_FULL; K20b − K20c\n= 100·K21 / C_FULL), `post_validation_rule` unchanged; its sha256 is registered by the DL row and copied into ANA_SCRIPT_FREEZE and\nALL_METHODS_LOCK. The `--claims-only` output is regenerated under v4 and frozen with it.',
)
P06_ERRATUM_1 = '**ERRATUM 1 (2026-10-01, DL 324; before any freeze; no value read).** In P6, "Holm family (secondary, 19)" is a counting error.\nThe named list is binding and has **18** members: K1, K2, K3, K8, K9b, K9c, K11b, K12, K17, K20, K20c, K_INIT, P5, K4w, K5w,\nK12w, K13w, K14w. Fable\'s original list (FABLE_ENDGAME_REVIEW §5) also enumerates these 18. Nothing else changes.'   # 06P ERRATUM 1 (DL 324): the Holm family has the 18 listed members
S8A = "FA_WARM_FROZEN"
V4_RESOLVED = {   # 06P P1: K20 / K20b / K21 on FA_WARM* = FA_WARM_FROZEN (question texts = v3's, arm resolved)
    "K20": ("R(FA_WARM_FROZEN / C_BIG)", "deployment-branch federated retention vs the C_BIG cloud model (warm start from "
                                         "theta_pre, S8 frozen item tables = FA_WARM*; seeds 2026, 2027)"),
    "K20b": ("R(FA_WARM_FROZEN / C_FULL)", "deployment-branch federated retention vs its matched cloud reference C_FULL "
                                           "(2 FA_WARM_FROZEN seeds vs C_FULL s2026) — PRIMARY with K21 (06N N1; 06P P1)"),
    "K21": ("FA_WARM_FROZEN − PRE", "value of federated training on the private users over the pretrained model alone "
                                    "(paired Δ) — PRIMARY with K20b (06N N1; 06P P1)"),
}
V4_NEW = (   # 06P P3-P5 (ids as written there: K_INIT, K4w, K5w, K12w, K13w, K14w and "P5" (P6 names it P5))
    ("K_INIT", "FA_WARM_FROZEN − FA", "value of the pretrained start (paired Δ, seeds 2026 / 2027; 06N N1; 06P P3)"),
    ("K4w", "R(FP_WARM_FROZEN / C_FULL)", "FedProx retention on the pretrained line vs C_FULL (06P P4)"),
    ("K5w", "R(PF_WARM_FROZEN / C_FULL)", "personalised-FL retention on the pretrained line vs C_FULL (06P P4)"),
    ("P5", "FT_FA_WARM_FROZEN − FA_WARM_FROZEN", "local fine-tuning gain after the pretrained-line FL (paired Δ, base "
                                                  "= the same-seed best.pt, harmed-client share beside; 06P P5)"),
    ("K12w", "R(DP8_FROZEN / FA_1024_WARM_FROZEN)", "privacy cost at ε = 8 on the pretrained line (06P P4)"),
    ("K13w", "R(DP1_WARM_FROZEN / FA_1024_WARM_FROZEN)", "privacy cost at ε = 1 on the pretrained line (seed 2026; "
                                                          "06P P4)"),
    ("K14w", "R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN)", "compression cost (8-bit uploads) on the pretrained line (06P P4)"),
)
V4_DESCRIPTIVE = (   # 06P P2 / P4 / P5 DESCRIPTIVE rows (with S8-S7 of 06L-C C8)
    ("S8-S7", "FA_WARM_FROZEN − FA_WARM", "cost of not sending the item tables (paired Δ, reported with the bytes ratio; "
                                          "DESCRIPTIVE)"),
    ("D_S7_C_FULL", "R(FA_WARM / C_FULL)", "the S7 full-model warm start vs C_FULL (DESCRIPTIVE beside the primary; 06P P2)"),
    ("D_S7_PRE", "FA_WARM − PRE", "the S7 full-model warm start over theta_pre (DESCRIPTIVE beside the primary; 06P P2)"),
    ("D_PFw_FAw", "PF_WARM_FROZEN − FA_WARM_FROZEN", "personalisation on the pretrained line (DESCRIPTIVE, 06L A5 caveat; "
                                                     "06P P4)"),
    ("D_FA1024w_FAw", "R(FA_1024_WARM_FROZEN / FA_WARM_FROZEN)", "clipping + Poisson-cohort cost on the pretrained line "
                                                                "(DESCRIPTIVE; 06P P4)"),
    ("D_FTw_S7", "FT_FA_WARM − FA_WARM", "local fine-tuning gain after the S7 warm FL (DESCRIPTIVE; 06P P5)"),
)
V4_IDS = V3_IDS + tuple(k for k, _c, _q in V4_NEW)
V4_DESCRIPTIVE_IDS = tuple(k for k, _c, _q in V4_DESCRIPTIVE)
HOLM_FAMILY_V4 = ("K1", "K2", "K3", "K8", "K9b", "K9c", "K11b", "K12", "K17", "K20", "K20c", "K_INIT", "P5", "K4w",
                  "K5w", "K12w", "K13w", "K14w")      # 06P P6 + ERRATUM 1 (DL 324): exactly these 18
TERTIARY_V4 = ("K4", "K5", "K6", "K7", "K9", "K10", "K11", "K13", "K14", "K15", "K16", "K18", "K19")   # 06P P6
HIERARCHY_V4 = dict({k: "secondary" for k in HOLM_FAMILY_V4}, K20b="primary", K21="primary")   # 06P P1 / P4 / P6
CONFIRMATORY_RULE_V4 = dict(CONFIRMATORY_RULE, K12w="CONTROLLED_BUDGET", K13w="CONTROLLED_BUDGET")       # 06P P4
REGISTERED_SEEDS_V4 = dict(REGISTERED_SEEDS, K13w=(2026,))                                              # 06P P4 / P6
CONTRAST_SHARED_DEN_V4 = {"K20b": "C_FULL", "K4w": "C_FULL", "K5w": "C_FULL"}   # the one C_FULL seed (06L-C §C7; 06P P4)
PAIRING_NOTES_V4 = dict(
    PAIRING_NOTES,
    K20b="2 FA_WARM_FROZEN seeds vs C_FULL s2026 (06L-C §C7; 06P P1): the one registered C_FULL seed is the denominator "
         "of both",
    K21="PRE is ONE θ_pre (seed 2026) shared by both FA_WARM_FROZEN seeds (06K / DL 206 (2); 06P P1)",
    K4w="2 FP_WARM_FROZEN seeds vs C_FULL s2026 (the one registered C_FULL seed; 06P P4)",
    K5w="2 PF_WARM_FROZEN seeds vs C_FULL s2026 (the one registered C_FULL seed; 06P P4)",
    P5="FT_FA_WARM_FROZEN seed s vs FA_WARM_FROZEN seed s; only a device run whose base is that seed's best.pt pairs "
       "(06P P5)",
    K_INIT="FA_WARM_FROZEN seed s vs the cold-start FA seed s (06P P3)")
READING_RULES_V4 = dict(READING_RULES, K14w=dict(READING_RULES["K14"], rule=READING_RULES["K14"]["rule"].replace(
    "R(FA_Q8 / FA)", "R(FA_Q8_WARM_FROZEN / FA_WARM_FROZEN)") + " (06P P4)"))
BASE_BEST_ONLY_V4 = ("P5",)
EQUIVALENT_FLAG = "K14 only (06L-C C15)"               # the claim-register flag text (v3)
EQUIVALENT_FLAG_V4 = "K14 and K14w only (06L-C C15; 06P P4 / P6: TOST ±1 % on the unadjusted 90% CI)"   # REV-ANA-06 m3                          # 06P P5: base = the same-seed best.pt (ft_warm start_base)
CONTRAST_ORDER_V4 = (CONTRAST_ORDER[:CONTRAST_ORDER.index("S8-S7")] + ("K_INIT", "P5", "K4w", "K5w", "K12w", "K13w",
                                                                       "K14w") + V4_DESCRIPTIVE_IDS)
CLAIM_FORBIDDEN_V4 = dict(
    CLAIM_FORBIDDEN,
    K2=CLAIM_FORBIDDEN["K2"] + ["'the headline' (06P P1: K2 is the SECONDARY random-start ablation)"],
    K20b=CLAIM_FORBIDDEN["K20b"] + ["a deployment-branch claim from K20b without K21 (06P P1: the primary is the pair)"],
    K21=CLAIM_FORBIDDEN["K21"] + ["a deployment-branch claim from K21 without K20b (06P P1: the primary is the pair)"],
    K_INIT=["'the pretrained start is needed' unless ABOVE", "'a random start is as good' (an interval including 0 is an "
                                                           "absence of evidence; no equivalence test is registered)"],
    K4w=list(CLAIM_FORBIDDEN["K4"]), K5w=list(CLAIM_FORBIDDEN["K5"]), P5=list(CLAIM_FORBIDDEN["K10"]),
    K12w=[CLAIM_FORBIDDEN["K12"][0].replace("ε = 8", "ε = 8 on the pretrained line")],
    K13w=[CLAIM_FORBIDDEN["K13"][0].replace("ε = 1", "ε = 1 on the pretrained line")],
    K14w=[CLAIM_FORBIDDEN["K14"][0]])
HEADLINE_TEMPLATE_V4 = HEADLINE_TEMPLATE.replace(
    "3. Federated (primary, K2). The matched-exposure retention of the registered FL recipe <{FL_RECIPE}> is {K2} % "
    "[{K2_CI}] of C_BIG ({K2_CLAIM})",
    "3. Federated from a random start (secondary: the random-start ablation, K2). The matched-exposure retention of the "
    "registered FL recipe <{FL_RECIPE}> is {K2} % [{K2_CI}] of C_BIG ({K2_CLAIM}); the pretrained start adds {K_INIT} "
    "[{K_INIT_CI}] (K_INIT)").replace(
    "7. Deployment branch (the Edge answer). Warm-started federation retains {K20b} % [{K20b_CI}] of its matched cloud "
    "reference C_FULL (K20b)",
    "7. Deployment branch (PRIMARY, the pair K20b + K21; the Edge answer). Federation from the pretrained model with the "
    "item tables frozen (FA_WARM* = S8) retains {K20b} % [{K20b_CI}] of its matched cloud reference C_FULL (K20b, "
    "{K20b_CLAIM})").replace("(K21); freezing", "(K21, {K21_CLAIM}); freezing").replace(
    "(number-free; 06L-C C12)", "(number-free; 06L-C C12; 06P hierarchy)", 1)
V4_PROFILE = {"HIERARCHY": HIERARCHY_V4, "CONFIRMATORY_RULE": CONFIRMATORY_RULE_V4, "REGISTERED_SEEDS": REGISTERED_SEEDS_V4,
              "CONTRAST_SHARED_DEN": CONTRAST_SHARED_DEN_V4, "PAIRING_NOTES": PAIRING_NOTES_V4,
              "READING_RULES": READING_RULES_V4, "CLAIM_FORBIDDEN": CLAIM_FORBIDDEN_V4,
              "HEADLINE_TEMPLATE": HEADLINE_TEMPLATE_V4, "CONTRAST_ORDER": CONTRAST_ORDER_V4,
              "BASE_BEST_ONLY": BASE_BEST_ONLY_V4, "EQUIVALENT_FLAG": EQUIVALENT_FLAG_V4}
BASE_BEST_ONLY = ()                                     # (v3: none)
_V3_PROFILE = {k: globals()[k] for k in V4_PROFILE}   # the v3 tables at import (documentation / tests)
_PROFILE_SAVED: list = []                              # the tables in force before a v4 switch


def set_registry_profile(fmt: Optional[str]) -> None:
    """ANA-06: the module-level registry tables of the loaded contrasts format. v4 = the 06P tables (the tables in
    force are saved first); any other format leaves the tables as they are, except that it restores the saved ones after
    a v4 switch. So a v1-v3 run never touches a table, and run() always ends with the pre-v4 tables."""
    if fmt == CONTRAST_FORMAT_V4:
        if not _PROFILE_SAVED:
            _PROFILE_SAVED.append({k: globals()[k] for k in V4_PROFILE})
        globals().update(V4_PROFILE)
    elif _PROFILE_SAVED:
        globals().update(_PROFILE_SAVED.pop())


def v4_items() -> list:
    """{id, contrast, question} of the v4 confirmatory registry (v3's 26 with K20 / K20b / K21 resolved, + 06P P3-P5)."""
    reg = registered_items()
    out = [{"id": k, "contrast": V4_RESOLVED.get(k, reg[k])[0], "question": V4_RESOLVED.get(k, reg[k])[1]}
           for k in V3_IDS]
    return out + [{"id": k, "contrast": c, "question": q} for k, c, q in V4_NEW]


def v4_multiplicity() -> str:
    n = len(V4_IDS)
    sec = list(HOLM_FAMILY_V4)
    return (f"{n} pre-registered contrasts (v3's 26 with K20 / K20b / K21 on FA_WARM_FROZEN, + K_INIT, P5, K4w, K5w, "
            f"K12w, K13w, K14w); primary = the pair K20b + K21, read unadjusted, outside Holm (06P P1 / P6); Holm "
            f"step-down over the secondary family of {len(sec)} ({', '.join(sec)}; 06P P6); tertiary unadjusted with a "
            "DESCRIPTIVE Holm band beside (step-down within the tertiary set); unadjusted 95% paired user-bootstrap "
            "percentile CIs (1,000 resamples, seed 20260926) for every contrast; K14 / K14w TOST on the unadjusted 90% CI; "
            "algebraic dependencies are not independent evidence; every other number is DESCRIPTIVE (06P)")


def v4_registry_content() -> dict:
    """The structured fields a PPSI_V4_ANA_CONTRASTS_4 file must carry verbatim (06P P1-P7)."""
    return {"confirmatory_contrasts": v4_items(),
            "confirmatory_rule_map": dict(CONFIRMATORY_RULE_V4), "default_confirmatory_rule": DEFAULT_CONFIRMATORY_RULE,
            "hierarchy": {"primary": sorted(k for k in V4_IDS if HIERARCHY_V4.get(k) == "primary"),
                          "secondary": sorted(k for k in V4_IDS if HIERARCHY_V4.get(k) == "secondary"),
                          "tertiary": sorted(k for k in V4_IDS if k not in HIERARCHY_V4)},
            "holm_family": list(HOLM_FAMILY_V4), "tertiary_band_family": list(TERTIARY_V4),
            "registered_seeds": {k: list(v) for k, v in sorted(REGISTERED_SEEDS_V4.items())},
            "populations": {k: CONTRAST_POPULATION.get(k, "MAIN") for k in V4_IDS + V4_DESCRIPTIVE_IDS},
            "pairing_notes": dict(PAIRING_NOTES_V4), "reading_rules": json.loads(json.dumps(READING_RULES_V4)),
            "shared_denominators": dict(CONTRAST_SHARED_DEN_V4), "base_best_only": list(BASE_BEST_ONLY_V4),
            "algebraic_dependencies": json.loads(json.dumps(list(ALGEBRAIC_DEPENDENCIES))),
            "registered_descriptive_contrasts": [{"id": k, "contrast": c, "question": q} for k, c, q in V4_DESCRIPTIVE],
            "multiplicity_statement": v4_multiplicity(),
            "post_validation_rule": POST_VALIDATION_NOTE, "amendment_06P_text": list(P06_TEXT),
            "amendment_06P_erratum_1": P06_ERRATUM_1,
            "amendment_06P_file_sha256": P06_FILE_SHA256}


def v4_registry_object() -> dict:
    """company/gates/ANA_CONTRASTS_v4.json (written by ANA-06 from contracts/PREG1_AMENDMENT_06P.md; its sha256 is
    registered by a later DL row before the ANA freeze, 06P P7)."""
    obj = {"format": CONTRAST_FORMAT_V4, "document": "ANA_CONTRASTS_v4", "draft": False,
           "status": ("BUILT BY ANA-06 (2026-10-01) FROM contracts/PREG1_AMENDMENT_06P.md (REGISTERED 2026-10-01, DL "
                      "316-317); registered by the DECISION_LOG row that records this file's sha256 (06P P7), before the "
                      "ANA freeze; used for the VALIDATION preflight and copied verbatim into ALL_METHODS_LOCK.json"),
           "prepared_utc": "2026-10-01", "default_seeds": list(SEEDS),
           "dp_disclosure": "K12/K13/K12w/K13w PRACTICAL_BEST ratios are DESCRIPTIVE: best-of-12 checkpoint selection "
                            "on the users' own validation data is not covered by the DP accountant (06D §3).",
           "supersedes": "company/gates/ANA_CONTRASTS_v3.json",
           "authority": ["contracts/PREG1_AMENDMENT_06N.md N1-N2", "contracts/PREG1_AMENDMENT_06P.md P1-P7",
                         "company/gates/ANA_CONTRASTS_v3.json (K1-K21, K9b, K9c, K11b, K20b, K20c)"],
           **v4_registry_content()}
    obj[SELF_KEY] = _canon_compact(obj)
    return obj


def load_contrasts_v4(obj: Mapping, raw: bytes, *, split: str, allow_draft: bool) -> dict:
    """ANA-06: a PPSI_V4_ANA_CONTRASTS_4 file = v4_registry_content() field by field (every id, text, rule, seed, tier,
    population, pairing note, reading rule, shared denominator, dependency and the Holm family); a draft loads only with
    --allow-draft-contrasts on VALIDATION, never on TEST."""
    if obj.get("draft") is not False:
        if split == "TEST" or not allow_draft:
            raise AnaRefused("--contrasts: a DRAFT PPSI_V4_ANA_CONTRASTS_4 file (draft not false); it loads only "
                             "with --split VALIDATION --allow-draft-contrasts, never for TEST")
    for key, want in v4_registry_content().items():
        got = obj.get(key)
        if key == "confirmatory_contrasts":
            got = [{k: it.get(k) for k in ("id", "contrast", "question")} for it in (got or [])]
            if sorted(_norm_text(json.dumps(x, sort_keys=True, ensure_ascii=False)) for x in got) != \
                    sorted(_norm_text(json.dumps(x, sort_keys=True, ensure_ascii=False)) for x in want):
                raise AnaRefused("--contrasts: the v4 confirmatory_contrasts are not the 06P P1-P5 set")
            continue
        if json.loads(json.dumps(got)) != json.loads(json.dumps(want)):
            raise AnaRefused(f"--contrasts: {key} is not the 06P content of this script")
    out = []
    for it in sorted(v4_items(), key=lambda x: CONTRAST_ORDER_V4.index(x["id"])):
        out.append(dict(it, population=CONTRAST_POPULATION.get(it["id"], "MAIN"), **parse_contrast(it["contrast"])))
    desc = [dict(id=k, contrast=c, question=q, population="MAIN", **parse_contrast(c)) for k, c, q in V4_DESCRIPTIVE]
    content = dict(v4_registry_content(), items=[{k: c[k] for k in ("id", "contrast", "question")} for c in out])
    content.pop("confirmatory_contrasts")
    return {"file_sha256": _sha_bytes(raw), "items": out, "descriptive_items": desc,
            "confirmatory_rule_map": dict(CONFIRMATORY_RULE_V4), "format": CONTRAST_FORMAT_V4, "n_contrasts": len(out),
            "multiplicity": v4_multiplicity(), "v1_dry_check": False, "draft": obj.get("draft") is not False,
            "holm_family": list(HOLM_FAMILY_V4), "tertiary_band_family": list(TERTIARY_V4),
            "superseded_note": ("a DRAFT v4 file: its CONFIRMATORY labels are provisional"
                                if obj.get("draft") is not False else None),
            "content_sha256": _sha_bytes(_canon_json(content)), "self_sha256_status": _self_ok(obj)}



LIB = SimpleNamespace()


class AnaRefused(RuntimeError):
    """A check failed: nothing is written."""


# ------------------------------------------------------------------------------------------------ small helpers
def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha_file(p) -> str:
    return _sha_bytes(Path(p).read_bytes())


def script_sha256() -> str:
    return _sha_file(Path(__file__).resolve())


def _read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _canon_compact(obj: Mapping) -> str:
    """The launchers' self-sha rule (v4int.launch_fl._canon / launch_device._canon)."""
    body = {k: v for k, v in obj.items() if k != SELF_KEY}
    return _sha_bytes(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def _canon_indent(obj: Mapping, key: str = SELF_KEY) -> str:
    """The indent-1 self-sha rule (v4base.strata._self_sha256; used for this script's own outputs)."""
    body = {k: v for k, v in obj.items() if k != key}
    text = json.dumps(body, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    return _sha_bytes(text.encode("utf-8"))


def is_undef(x) -> bool:
    return isinstance(x, LIB.V.Undefined) or (isinstance(x, Mapping) and x.get("status") == "UNDEFINED")


def undef(reason: str):
    return LIB.V.Undefined(reason)


def _jsonable(obj):
    if isinstance(obj, LIB.V.Undefined):
        return {"value": None, "status": "UNDEFINED", "reason": obj.reason}
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return _jsonable(obj.tolist())
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        if not math.isfinite(obj):
            raise ValueError(f"non-finite float {obj!r} in the result (use UNDEFINED)")
        return obj
    raise TypeError(f"cannot serialise {type(obj).__name__}")


def dumps(obj) -> str:
    return json.dumps(_jsonable(obj), sort_keys=True, allow_nan=False, indent=1, ensure_ascii=True) + "\n"


def with_self_sha(obj: dict) -> dict:
    obj = json.loads(dumps(obj))
    obj[SELF_KEY] = _canon_indent(obj)
    return obj


def verify_self_sha(obj: Mapping) -> bool:
    return obj.get(SELF_KEY) == _canon_indent(obj)


def _write_new(path: Path, data: bytes) -> str:
    """O_EXCL, binary (LF kept), fsync; returns the file sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    return _sha_bytes(data)


# ------------------------------------------------------------------------------------------------ library binding
def bind(code_root=DEFAULT_CODE_ROOT) -> SimpleNamespace:
    """--code-root first on sys.path; every library module used must come from that tree (as tools/g2_run.py)."""
    root = Path(code_root).resolve()
    if not (root / "v4train" / "bootstrap_ci.py").is_file():
        raise AnaRefused(f"{root} is not a V4 code tree (no v4train/bootstrap_ci.py)")
    while str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    if (root / "ppsi_env.py").is_file():
        import ppsi_env  # noqa: F401  (mandatory import isolation of the V4 tree)
    import numpy as np
    import v4base.identity as ID
    import v4base.strata as ST
    import v4eval.bootstrap as B
    import v4eval.metrics as M
    import v4eval.quality as Q
    import v4eval.values as V
    import v4train.bootstrap_ci as BC
    mods = {"v4train/bootstrap_ci.py": BC, "v4eval/bootstrap.py": B, "v4eval/metrics.py": M, "v4eval/quality.py": Q,
            "v4eval/values.py": V, "v4base/strata.py": ST, "v4base/identity.py": ID}
    for rel, mod in mods.items():
        origin = Path(mod.__file__).resolve()
        if root not in origin.parents:
            raise AnaRefused(f"library module {rel} was imported from {origin}, not from the code root {root}")
    files = {rel: _sha_file(root / rel) for rel in sorted(mods)}
    files["tools/g2_run.py"] = _sha_file(HERE / "g2_run.py")            # A-F14: the production loader it executes
    LIB.__dict__.update(root=root, np=np, BC=BC, B=B, M=M, Q=Q, V=V, ST=ST, ID=ID, files_sha256=files)
    return LIB


def _load_g2():
    got = _sha_file(HERE / "g2_run.py")
    if got != G2_RUN_SHA256:
        raise AnaRefused(f"tools/g2_run.py sha256 {got} is not the frozen {G2_RUN_SHA256} (G2_SCRIPT_FREEZE; A-F14)")
    spec = importlib.util.spec_from_file_location("g2_run_for_ana", HERE / "g2_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ role contexts
@dataclass
class RoleCtx:
    role: str
    manifest: object                      # v4eval.manifest.EvalManifest
    hex_by_euk: Optional[object] = None   # np object array: client_key_hex[eval_user_key]
    support_by_hex: Optional[dict] = None # {client_key_hex: n TRAIN-layout decisions}
    record: dict = field(default_factory=dict)


def hex_from_tables(tables: Mapping, root: Path):
    """client_key_hex[eval_user_key] from an EVAL manifest's registry tables (v4base.identity; file sha verified)."""
    return LIB.ID.eval_client_key_hex_by_eval_user_key(SimpleNamespace(verified=SimpleNamespace(tables=tables)),
                                                       manifest_dir=root)


def support_from_tables(tables: Mapping, root: Path) -> dict:
    """{client_key_hex: number of the client's decision rows in a TRAIN-layout manifest} (the TRAIN-period support
    count of 06A §2; the 5-100 pool is defined on the same count), via v4base.identity (file sha verified)."""
    np = LIB.np
    n_dec = int(tables["decisions"]["row_count"])
    fake = SimpleNamespace(verified=SimpleNamespace(tables=tables, canonical_rows=np.arange(n_dec, dtype=np.int64)))
    ords = LIB.ID.train_client_ord_by_canonical_position(fake, manifest_dir=root)
    hex_by_ord = LIB.ID.train_client_key_hex_by_ord(fake, manifest_dir=root)
    if ords.size != n_dec:
        raise AnaRefused(f"the decisions table has {ords.size} rows, the registry says {n_dec}")
    cnt = np.bincount(ords, minlength=hex_by_ord.size)
    return {str(h): int(c) for h, c in zip(hex_by_ord.tolist(), cnt.tolist())}


def production_context_loader():
    """The server path: g2_run's launcher manifest loading (DATA_PROTOCOL_LOCK_V2 -> REGISTRY_V2 -> verify ->
    build_eval_manifest) + v4base.identity over the registry's verified tables (client_key_hex joins)."""
    G = _load_g2()
    state = {}

    def load(role: str, spec: Mapping) -> RoleCtx:
        np = LIB.np
        if "pc" not in state:
            pc = G.production_context(SimpleNamespace(root=LIB.root))
            pinned = pc.ctx.get("manifest_registry_file_sha256")
            if pinned != pc.reg_file_sha256:              # g2_run.py:212-213, strict (a missing pin refuses too)
                raise AnaRefused(f"REGISTRY_V2 file sha256 {pc.reg_file_sha256} != the DATA_PROTOCOL_LOCK_V2 pin "
                                 f"{pinned} (A-F14)")
            state["pc"] = pc
        pc = state["pc"]
        if "TEST" in str(spec["eval_manifest_id"]).upper():
            raise AnaRefused("a TEST manifest is never loaded through the non-TEST production loader (E10.1)")
        mid = spec["eval_manifest_id"]
        m, binding = G.load_eval_manifest(SimpleNamespace(root=LIB.root), pc, mid)
        root = Path(pc.LC.V4)
        entry = pc.reg_obj["manifests"][mid]
        tid = spec.get("train_manifest_id")
        hex_by_euk = None
        if role in ("MAIN", "CALIBRATION") or tid:   # strata (MAIN), MAIN \ CALIB (06L-C C14) and support bins
            hex_by_euk = hex_from_tables(entry["tables"], root)
            if m.n and int(np.max(m.eval_user_key)) >= hex_by_euk.size:
                raise AnaRefused(f"role {role}: the users table does not cover every eval_user_key")
        support = support_from_tables(pc.reg_obj["manifests"][tid]["tables"], root) if tid else None
        rec = {"eval_manifest_id": mid, "train_manifest_id": tid, "registry_file_sha256": pc.reg_file_sha256,
               "registry_entry": {k: entry.get(k) for k in ("content_sha256", "header_sha256", "n_decisions",
                                                             "n_users", "eval_ordered_decision_id_sha256")},
               "binding": binding}
        if role == "CALIBRATION":
            if mid != CALIB_EVAL_ID:
                raise AnaRefused(f"role CALIBRATION is {CALIB_EVAL_ID} only (06L-C C14), not {mid}")
            rec["enforce_registered_counts"] = True
        return RoleCtx(role, m, hex_by_euk, support, rec)

    return load


# ------------------------------------------------------------------------------------------------ the TEST gate (A-F8)
def _canon_json(x) -> bytes:
    return json.dumps(x, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def _strict_json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"),
                          parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))
    except (OSError, ValueError) as e:
        raise AnaRefused(f"{p}: not readable strict JSON ({e})") from None


def check_test_gate(lock_path: Optional[Path], register_path: Optional[Path]) -> dict:
    """E10.1: the ALL_METHODS_LOCK file sha256 is the all_methods_lock_sha256 of a MANIFEST_BUILD / RETRO_TEST entry of
    a TEST_EXPOSURE_REGISTER whose hash chain verifies (tools/test_exposure_register.py `verify`, loaded by path — the
    same rule as v4data09.d9_gate.check_test_gate; the FIRST qualifying entry binds)."""
    if lock_path is None or register_path is None:
        raise AnaRefused("--split TEST requires --all-methods-lock and --test-exposure-register (E10.1)")
    lock_path, register_path = Path(lock_path), Path(register_path)
    if not lock_path.is_file():
        raise AnaRefused(f"ALL_METHODS_LOCK {lock_path} does not exist: the TEST look is refused (E10.1)")
    if not register_path.is_file():
        raise AnaRefused(f"TEST_EXPOSURE_REGISTER {register_path} does not exist (E10.1)")
    lock_sha = _sha_file(lock_path)
    reg = _strict_json(register_path)
    entries = reg.get("entries") if isinstance(reg, Mapping) else None
    if not isinstance(entries, list) or not entries:
        raise AnaRefused(f"{register_path}: no entries")
    if not REGISTER_TOOL.is_file():
        raise AnaRefused(f"{REGISTER_TOOL} is missing: the register chain cannot be verified")
    spec = importlib.util.spec_from_file_location("_ana_test_exposure_register_tool", str(REGISTER_TOOL))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    try:
        errs = tool.verify(reg)
    except Exception as e:                                   # a malformed entry breaks the chain verification itself
        raise AnaRefused(f"{register_path}: the hash chain cannot be verified ({type(e).__name__}: {e})") from None
    if errs:
        raise AnaRefused(f"{register_path}: the hash chain does not verify: {errs}")
    hits = [e for e in entries if isinstance(e, Mapping) and e.get("access_purpose") == TEST_GATE_PURPOSE
            and e.get("split") == TEST_GATE_SPLIT and e.get("all_methods_lock_sha256") == lock_sha]
    if not hits:
        raise AnaRefused(f"no TEST_EXPOSURE_REGISTER entry with access_purpose {TEST_GATE_PURPOSE}, split "
                         f"{TEST_GATE_SPLIT} and all_methods_lock_sha256 == sha256(ALL_METHODS_LOCK) {lock_sha}")
    e = min(hits, key=lambda x: int(x["entry_id"]))
    return {"all_methods_lock_sha256": lock_sha, "test_exposure_register_entry_id": int(e["entry_id"]),
            "test_exposure_register_entry_sha256": e["entry_sha256"], "register_file_sha256": _sha_file(register_path),
            "register_tool_sha256": _sha_file(REGISTER_TOOL), "n_register_entries": len(entries)}


def check_test_header(role: str, header: Mapping, gate: Mapping) -> None:
    """A TEST manifest is used only under the gate it was built under (06D §2: the register entry id in its header)."""
    if header.get("all_methods_lock_sha256") != gate["all_methods_lock_sha256"] or \
            header.get("test_exposure_register_entry_id") != gate["test_exposure_register_entry_id"]:
        raise AnaRefused(f"role {role}: the TEST manifest header is not bound to this gate (all_methods_lock_sha256 / "
                         "test_exposure_register_entry_id differ)")


def verify_test_ext_registry(ext: Mapping, base: Mapping, base_file_sha256: str) -> dict:
    """The REGISTRY_V2 extension rule (as v4data08.d8_registry.verify_ext_registry): every REGISTRY_V2 top-level key is
    canonical-JSON identical, except roles / manifests / population_bands whose every REGISTRY_V2 entry is identical and
    which gain only TEST roles (`*_TEST`) and RETRO_TEST manifests; `extends` names REGISTRY_V2's file and self sha256;
    both self hashes verify (the REGISTRY_V2 rule v4data.common.self_hash)."""
    from v4data08.d8_registry import canon, self_ok        # the DATA-08 extension rule's primitives (read-only)
    selfk = "self_sha256"
    c = {"ext.self_sha256": self_ok(ext), "base.self_sha256": self_ok(base)}
    ex = ext.get("extends") or {}
    c["extends.self_sha256_eq_base"] = ex.get("self_sha256") == base.get(selfk)
    c["extends.file_sha256_eq_base"] = ex.get("file_sha256") == base_file_sha256
    for k in sorted(set(base) - {selfk}):
        if k in ("roles", "manifests", "population_bands"):
            b, e = base.get(k) or {}, ext.get(k) or {}
            c[f"{k}.base_entries_identical"] = isinstance(e, Mapping) and all(
                sk in e and canon(e[sk]) == canon(v) for sk, v in b.items())
            added = set(e) - set(b) if isinstance(e, Mapping) else set()
            if k == "roles":
                c["roles.additions_are_TEST_roles"] = all(str(r).endswith("_TEST") for r in added)
            elif k == "manifests":
                c["manifests.additions_are_RETRO_TEST"] = all("RETRO_TEST" in str(m) for m in added)
            else:
                c["population_bands.no_additions"] = not added
        else:
            c[f"top.{k}.identical"] = k in ext and canon(ext[k]) == canon(base[k])
    bad = sorted(k for k, v in c.items() if not v)
    if bad:
        raise AnaRefused(f"--registry-ext does not extend REGISTRY_V2 by the rule: failed {bad}")
    return c


def test_context_loader(gate: Mapping, ext_path: Path, ext: Mapping, lock_path: Path, register_path: Path):
    """HOOK for the DATA-09 shim (A-F8 (ii)). The TEST manifests (MAIN_RETRO_TEST_5_100_131072,
    UNEW_HOLDOUT_RETRO_TEST_32768, TRUE_COLD_RETRO_TEST, ...) are read ONLY through the DATA-09 gated reader
    (`code/v4int/test_roles.py` over `code/v4data09`; E10.1 "the builder and the loader both refuse otherwise"):
      v4int.test_roles.gate(lock, register)                    -> its GateRecord (must equal this script's gate)
      v4int.test_roles.load_test_ext(ext, gate, base_path=REGISTRY_V2 pinned by DATA_PROTOCOL_LOCK_V2, base_file_sha256)
      v4int.test_roles.verify_test_manifest(mid, ext_path=, paths=, root=, lock_path=, register_path=) -> VerifiedManifest
      v4int.test_roles.test_eval_manifest(vm, lock_path=, register_path=)                      -> v4eval EvalManifest
    and client_key_hex / support tables from the extension's (verified) registry entries, as for VALIDATION. If the
    module or any of these names is missing, or DATA-09 refuses anything, this is a HARD refusal (exit 2), never an
    exit-5 KeyError. load_block additionally applies check_test_header to every TEST context. DATA-09 is unreviewed at
    the ANA-02 freeze: any change of its API changes this function and therefore the script sha (re-freeze)."""
    def refuse(what, e=None):
        raise AnaRefused(f"TEST manifests: {what}" + (f" ({type(e).__name__}: {e})" if e is not None else "")
                         + " — the DATA-09 gated loader (v4int.test_roles) is required; refusing")
    try:
        import v4int.test_roles as TR
    except Exception as e:                                   # not present / not importable
        refuse("the DATA-09 gated TEST loader is not importable", e)
    if LIB.root not in Path(TR.__file__).resolve().parents:
        refuse(f"v4int.test_roles was imported from {TR.__file__}, not from the code root")
    missing = [n for n in ("gate", "load_test_ext", "verify_test_manifest", "test_eval_manifest") if not hasattr(TR, n)]
    if missing:
        refuse(f"v4int.test_roles lacks {missing}")
    try:
        G = _load_g2()
        pc = G.production_context(SimpleNamespace(root=LIB.root))
        if pc.ctx.get("manifest_registry_file_sha256") != pc.reg_file_sha256:
            refuse("REGISTRY_V2 is not the DATA_PROTOCOL_LOCK_V2 pin (A-F14)")
        g = TR.gate(lock_path, register_path)
        if (g.lock_sha256, g.entry_id) != (gate["all_methods_lock_sha256"], gate["test_exposure_register_entry_id"]):
            refuse("the DATA-09 gate and this script's gate disagree (lock sha / register entry)")
        TR.load_test_ext(ext_path, g, base_path=pc.reg_path, base_file_sha256=pc.reg_file_sha256)
    except AnaRefused:
        raise
    except Exception as e:
        refuse("the DATA-09 gate / extension / production context refused", e)
    root = Path(pc.LC.V4)
    ext_sha = _sha_file(ext_path)

    def load(role: str, spec: Mapping) -> RoleCtx:
        mid, tid = spec["eval_manifest_id"], spec.get("train_manifest_id")
        if role == "CALIBRATION":                        # 06L-C C14: a VALIDATION manifest (users table only), read
            if mid != CALIB_EVAL_ID:                     # through the non-TEST registry path (REGISTRY_V2 = the pin)
                refuse(f"role CALIBRATION is {CALIB_EVAL_ID} only, not {mid}")
            try:
                m, binding = G.load_eval_manifest(SimpleNamespace(root=LIB.root), pc, mid)
                hex_by_euk = hex_from_tables(pc.reg_obj["manifests"][mid]["tables"], root)
            except AnaRefused:
                raise
            except Exception as e:
                refuse("the CALIBRATION users table could not be loaded", e)
            return RoleCtx(role, m, hex_by_euk, None, {"eval_manifest_id": mid, "binding": binding,
                                                       "registry_file_sha256": pc.reg_file_sha256})
        try:
            vm = TR.verify_test_manifest(mid, ext_path=ext_path, paths=pc.paths, root=root, lock_path=lock_path,
                                         register_path=register_path)
            m = TR.test_eval_manifest(vm, lock_path=lock_path, register_path=register_path)
            hex_by_euk = hex_from_tables(ext["manifests"][mid]["tables"], root)
            support = support_from_tables(ext["manifests"][tid]["tables"], root) if tid else None
        except AnaRefused:
            raise
        except Exception as e:
            refuse(f"role {role} ({mid}) could not be loaded through the gate", e)
        if m.n and int(LIB.np.max(m.eval_user_key)) >= hex_by_euk.size:
            raise AnaRefused(f"role {role}: the users table does not cover every eval_user_key")
        rec = {"eval_manifest_id": mid, "train_manifest_id": tid, "registry": "REGISTRY_V2_TEST_EXT",
               "registry_ext_file_sha256": ext_sha, "base_registry_file_sha256": pc.reg_file_sha256,
               "registry_entry": {k: ext["manifests"][mid].get(k) for k in ("content_sha256", "header_sha256",
                                                                            "n_decisions", "n_users")},
               "loader": "v4int.test_roles (DATA-09)", "loader_file_sha256": _sha_file(Path(TR.__file__))}
        return RoleCtx(role, m, hex_by_euk, support, rec)

    return load


# ------------------------------------------------------------------------------------------------ block index
@dataclass
class Entry:
    arm: str
    seed: Optional[int]
    rule: str
    role: str
    rows_rel: Optional[str]
    ranks_rel: Optional[str]
    row_filter: dict
    run_dir_rel: Optional[str]
    subset_rel: object                                # a path, {"rule": "EVAL_USER_KEY_LT", "n": N} or None
    historical: bool
    meta_declared: dict
    block: str = "MAIN_BLOCK"                         # or ONNX_BLOCK (A-F2) / BASELINE_BLOCK (06L-C §C3)
    bench_rel: Optional[str] = None
    ties_rel: Optional[str] = None                    # BASELINE_CPU: v4base.run n_tied_int32.npy (LAST -> REPEAT rows)
    ties: object = None
    start_run_id: Optional[str] = None                # device arms: the LAUNCH start run id (FT_PRE = theta_pre base)
    start_init: Optional[dict] = None                 # DEV-FTPRE: FT_PRE's LAUNCH start.init_from (theta_pre identity)
    start_init_record_sha256: Optional[str] = None    # DEV-FTPRE: FT_PRE's PRETRAIN_FOR_FL_INIT.json file sha256
    pre_init: Optional[dict] = None                   # DEV-FTPRE: PRE's REEVAL binding to its PRETRAIN_FOR_FL_INIT.json
    start_base: Optional[dict] = None                 # ANA-05: FT_FA_WARM(_FROZEN)'s LAUNCH start (the FA_WARM / S8 base)
    warm_selection: Optional[dict] = None             # ANA-05: FA_WARM / S8's LAUNCH warm_selection (the 2-D S7 / S8 file)
    warm_run: Optional[dict] = None                   # REV-ANA-05 F1: a run-folder-less warm entry's FL run (LAUNCH)
    rows: dict = field(default_factory=dict)          # (metric, k) -> the macro-E2E evidence row
    ranks: object = None
    subset_keys: object = None
    meta_auto: dict = field(default_factory=dict)
    marks: list = field(default_factory=list)
    rule_source: str = "ROWS"
    notes: list = field(default_factory=list)

    @property
    def eid(self) -> str:
        return f"{self.arm}|{self.seed}|{self.rule}|{self.role}"

    @property
    def ranked(self) -> bool:
        return self.ranks is not None

    @property
    def onnx(self) -> bool:
        return self.block == ONNX_BLOCK

    @property
    def subset_label(self):
        return self.subset_rel if isinstance(self.subset_rel, (str, type(None))) else json.dumps(self.subset_rel,
                                                                                                   sort_keys=True)

    def meta(self, key):
        if key in self.meta_declared and self.meta_declared[key] is not None:
            return self.meta_declared[key]
        return self.meta_auto.get(key)


def _inside(block: Path, rel) -> Path:
    if rel is None:
        return None
    p = (block / str(rel)).resolve()
    if block.resolve() not in p.parents and p != block.resolve():
        raise AnaRefused(f"index path {rel!r} leaves the block directory")
    if not p.exists():
        raise AnaRefused(f"index path {rel!r} does not exist")
    return p


def load_index(block: Path, split: Optional[str] = None) -> Tuple[dict, List[Entry]]:
    p = block / INDEX_NAME
    if not p.is_file():
        raise AnaRefused(f"{p} is missing")
    idx = _read_json(p)
    if idx.get("format") != INDEX_FORMAT:
        raise AnaRefused(f"{INDEX_NAME}: format {idx.get('format')!r} is not {INDEX_FORMAT}")
    if SELF_KEY in idx and idx[SELF_KEY] != _canon_indent(idx):
        raise AnaRefused(f"{INDEX_NAME}: self_sha256 does not verify (edited after tools/ana_block_index.py wrote it)")
    roles = idx.get("roles") or {}
    if "MAIN" not in roles:
        raise AnaRefused(f"{INDEX_NAME}: the MAIN role is required")
    for r, spec in roles.items():
        if not isinstance(spec, Mapping) or not spec.get("eval_manifest_id"):
            raise AnaRefused(f"{INDEX_NAME}: role {r} needs an eval_manifest_id")
        if split is not None:                          # the registered role map of the split (A-F8 (iii))
            want = ROLE_MAP[split].get(r)
            if want is None:
                raise AnaRefused(f"{INDEX_NAME}: role {r} is not a registered {split} role {sorted(ROLE_MAP[split])}")
            if spec.get("eval_manifest_id") != want[0] or spec.get("train_manifest_id") != want[1]:
                raise AnaRefused(f"{INDEX_NAME}: role {r} is {spec.get('eval_manifest_id')} / "
                                 f"{spec.get('train_manifest_id')}, the registered {split} pair is {want[0]} / {want[1]}")
    out, seen = [], set()
    for i, e in enumerate(idx.get("entries") or []):
        arm, rule, role = e.get("arm"), e.get("rule"), e.get("role", "MAIN")
        seed = e.get("seed")
        if arm not in ARMS:
            raise AnaRefused(f"entry {i}: unknown arm {arm!r} (allowed: {ARMS})")
        if rule not in RULES + (RULE_FREE,):
            raise AnaRefused(f"entry {i} ({arm}): rule {rule!r} is not one of {RULES + (RULE_FREE,)} "
                             "(PRACTICAL_BEST_EXTENDED never enters a block)")
        if rule == RULE_FREE and arm not in BASELINE_ARMS:
            raise AnaRefused(f"entry {i} ({arm}): RULE_FREE is allowed only for the baselines {BASELINE_ARMS}; device "
                             "arms and the ONNX arms are declared PRACTICAL_BEST, trained arms by their checkpoint rule "
                             "(A-F3)")
        if arm in PB_ONLY_ARMS and rule != "PRACTICAL_BEST":
            raise AnaRefused(f"entry {i} ({arm}): the device / ONNX arms exist under PRACTICAL_BEST only (A-F3)")
        if seed is not None:
            seed = int(seed)
            if seed not in SEEDS:
                raise AnaRefused(f"entry {i} ({arm}): seed {seed} is not registered {SEEDS}")
        elif rule != RULE_FREE:
            raise AnaRefused(f"entry {i} ({arm}): a checkpoint-rule entry needs a seed")
        if role not in roles:
            raise AnaRefused(f"entry {i} ({arm}): role {role!r} is not declared in roles")
        blk = e.get("comparison_block") or MAIN_BLOCK
        if blk not in BLOCKS:
            raise AnaRefused(f"entry {i} ({arm}): comparison_block {blk!r} is not {ONNX_BLOCK} / {BASELINE_BLOCK}")
        if (arm in ONNX_ARMS) != (blk == ONNX_BLOCK):
            raise AnaRefused(f"entry {i} ({arm}): comparison_block {ONNX_BLOCK} is required for exactly the ONNX arms "
                             f"{ONNX_ARMS} (onnxruntime CPU rows are never A40 rows; A-F2)")
        if blk == ONNX_BLOCK and (not e.get("ranks") or not e.get("bench_result")):
            raise AnaRefused(f"entry {i} ({arm}): an {ONNX_BLOCK} entry needs ranks and bench_result")
        if blk == BASELINE_BLOCK and (arm not in BASELINE_ARMS or rule != RULE_FREE or not e.get("ranks")
                                      or e.get("historical_reference")):
            raise AnaRefused(f"entry {i} ({arm}): a {BASELINE_BLOCK} entry is a baseline {BASELINE_ARMS} with rule "
                             "RULE_FREE and a per-row rank file (v4base.run ranks_int32.npy), never a historical "
                             "reference (06L-C §C3)")
        if e.get("row_ties") and blk != BASELINE_BLOCK:
            raise AnaRefused(f"entry {i} ({arm}): row_ties (a per-row tie count) is read for {BASELINE_BLOCK} entries only")
        if role == NO_FT_ROLE and arm in PREVIEW_REFERENCE_ARMS:        # 06O: a preview row is never a block entry
            raise AnaRefused(f"entry {i} ({arm}): {PREVIEW_REFERENCE_LABEL}")
        if role == NO_FT_ROLE and arm not in NO_FT_ARMS:
            raise AnaRefused(f"entry {i} ({arm}): role {NO_FT_ROLE} is the selection panel; only the no-FT references "
                             f"{NO_FT_ARMS} are admitted there (06L-C §C16; 06B §7(e))")
        sr, si, sirs = e.get("start_run_id"), e.get("start_init"), e.get("start_init_record_sha256")
        if arm == "FT_PRE":                                     # DEV-FTPRE: the init record, not a run-id prefix
            why = ft_pre_base_problem(si, sr, sirs)
            if why:
                raise AnaRefused(f"entry {i} ({arm}): ARM_IDENTITY — FT_PRE's base must be θ_pre of the PRE init record "
                                 f"(launch_device --arm FT_PRE start.init_from): {why} (06L-C §C7)")
        elif si is not None or sirs is not None or str(sr or "").startswith(FT_PRE_BASE_RUN_PREFIX):
            raise AnaRefused(f"entry {i} ({arm}): ARM_IDENTITY — a device run from θ_pre is FT_PRE, not {arm} (06L-C §C7)")
        if e.get("pre_init") is not None and arm != "PRE":
            raise AnaRefused(f"entry {i} ({arm}): pre_init (the θ_pre init-record binding) is PRE's only")
        sb, ws = e.get("start_base"), e.get("warm_selection")       # ANA-05 (06M_B B4 / B2)
        if arm in FT_WARM_BASE:
            why = ft_warm_base_problem(arm, sb, sr)
            if why:
                raise AnaRefused(f"entry {i} ({arm}): ARM_IDENTITY — {arm}'s base must be a registered "
                                 f"{FT_WARM_BASE[arm]} MAIN checkpoint (launch_device --base-kind {WARM_BASE_KIND} start): "
                                 f"{why} (06M_B B4)")
        elif sb is not None:
            raise AnaRefused(f"entry {i} ({arm}): start_base (an FA_WARM / S8 base) is FT_FA_WARM / FT_FA_WARM_FROZEN's "
                             "only (06M_B B4)")
        if ws is not None and (warm_stage_of(arm) is None or not isinstance(ws, Mapping)):
            raise AnaRefused(f"entry {i} ({arm}): warm_selection (the 2-D S7 / S8 binding) is FA_WARM / FA_WARM_FROZEN's "
                             "only (06M_B B2; ANA-06: and the FL-16 warm-line arms')")
        wr = e.get("warm_run")
        if wr is not None and (warm_stage_of(arm) is None or not isinstance(wr, Mapping)):
            raise AnaRefused(f"entry {i} ({arm}): warm_run (the FL run of a warm REEVAL entry) is FA_WARM / "
                             "FA_WARM_FROZEN's only (REV-ANA-05 F1; ANA-06: and the FL-16 warm-line arms')")
        why = warm_base_run_problem(arm, sr)                            # ANA-06 (DL 287 m3)
        if why:
            raise AnaRefused(f"entry {i} ({arm}): ARM_IDENTITY — {why}")
        if not e.get("rows") and blk != ONNX_BLOCK:
            raise AnaRefused(f"entry {i} ({arm}): rows path missing")
        if arm in DEVICE_ARMS and "device_cell" not in (e.get("row_filter") or {}):
            raise AnaRefused(f"entry {i} ({arm}): a device-arm entry needs row_filter.device_cell (the selected cell; "
                             "A-F10)")
        sub = e.get("users_subset")
        if isinstance(sub, Mapping) and (sub.get("rule") != SUBSET_RULE_V2 or not isinstance(sub.get("n"), int)
                                         or isinstance(sub.get("n"), bool) or sub["n"] < 1
                                         or not isinstance(sub.get("set_sha256"), str)):
            raise AnaRefused(f"entry {i} ({arm}): users_subset {sub!r} is not {{'rule': '{SUBSET_RULE_V2}', 'n': N, "
                             "'set_sha256': <rows.meta.json first_users_set_sha256>}")
        elif sub is not None and not isinstance(sub, (str, Mapping)):
            raise AnaRefused(f"entry {i} ({arm}): users_subset must be a .npy path or a {SUBSET_RULE_V2} object")
        en = Entry(arm, seed, rule, role, str(e["rows"]) if e.get("rows") else None, e.get("ranks"),
                   dict(e.get("row_filter") or {}), e.get("run_dir"), sub, bool(e.get("historical_reference", False)),
                   dict(e.get("meta") or {}), block=blk, bench_rel=e.get("bench_result"), ties_rel=e.get("row_ties"),
                   start_run_id=sr, start_init=dict(si) if isinstance(si, Mapping) else None,
                   start_init_record_sha256=sirs,
                   pre_init=dict(e["pre_init"]) if isinstance(e.get("pre_init"), Mapping) else None,
                   start_base=dict(sb) if isinstance(sb, Mapping) else None,
                   warm_selection=dict(ws) if isinstance(ws, Mapping) else None,
                   warm_run=dict(wr) if isinstance(wr, Mapping) else None)
        if en.eid in seen:
            raise AnaRefused(f"duplicate entry {en.eid}")
        seen.add(en.eid)
        out.append(en)
    if not any(e.arm == REF and e.role == "MAIN" and not e.historical for e in out):
        raise AnaRefused("the block has no C_BIG entry on MAIN: no retention reference")
    why = ft_pre_vs_pre_problem([(e.eid, e.start_init, e.start_init_record_sha256) for e in out if e.arm == "FT_PRE"],
                                [(e.eid, e.pre_init) for e in out if e.arm == "PRE" and not e.historical])
    if why:
        raise AnaRefused(f"ARM_IDENTITY — {why} (06L-C §C7; DEV-FTPRE)")
    return idx, out


def _hex64(v) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(v or "")))


def ft_pre_base_problem(init, start_run_id, init_record_sha256) -> Optional[str]:
    """DEV-FTPRE: why an FT_PRE entry's base is NOT theta_pre by its init record, or None: start_init must be the theta_pre
    identity (init_from_problem), its run id the entry's start_run_id, and the PRETRAIN_FOR_FL_INIT.json file sha256
    recorded (start_init_record_sha256). A run-id prefix alone never admits FT_PRE."""
    if not isinstance(init, Mapping):
        return (f"no start_init (the LAUNCH start.init_from theta_pre identity); start_run_id {start_run_id!r} alone "
                "is not a base check")
    why = init_from_problem(init)
    if why:
        return why
    if start_run_id is not None and start_run_id != init.get("run_id"):
        return f"start_run_id {start_run_id!r} is not the theta_pre identity's run id {init.get('run_id')!r}"
    if not _hex64(init_record_sha256):
        return f"start_init_record_sha256 {init_record_sha256!r} is not the PRETRAIN_FOR_FL_INIT.json file sha256"
    return None


def ft_warm_base_problem(arm: str, sb, start_run_id=None) -> Optional[str]:
    """ANA-05 (06M_B B4; FL-15 launch_device.load_warm_base): why an FT_FA_WARM / FT_FA_WARM_FROZEN start is NOT a
    registered FA_WARM / FA_WARM_FROZEN MAIN checkpoint, or None: kind = the base arm, base_kind fa_warm, checkpoint
    best.pt / endpoint_6.0.pt, file / arm-spec sha256, a run id that is not theta_pre (= the entry's start_run_id),
    init_from = the theta_pre identity, frozen tables exactly for the FROZEN arm."""
    if not isinstance(sb, Mapping):
        return "no start_base (the LAUNCH start of launch_device --base-kind fa_warm)"
    base = FT_WARM_BASE[arm]
    bad = {}
    if sb.get("kind") != base:
        bad["kind"] = sb.get("kind")
    if sb.get("base_kind") != WARM_BASE_KIND:
        bad["base_kind"] = sb.get("base_kind")
    if sb.get("checkpoint") not in WARM_BASE_FILES:
        bad["checkpoint"] = sb.get("checkpoint")
    for k in ("file_sha256", "base_arm_spec_sha256"):
        if not _hex64(sb.get(k)):
            bad[k] = sb.get(k)
    rid = str(sb.get("run_id") or "")
    if not rid or rid.startswith(PRETRAIN_RUN_PREFIX):
        bad["run_id"] = sb.get("run_id")
    if start_run_id is not None and start_run_id != sb.get("run_id"):
        bad["start_run_id"] = start_run_id
    if bool(sb.get("frozen_item_tables")) != (base == "FA_WARM_FROZEN"):
        bad["frozen_item_tables"] = sb.get("frozen_item_tables")
    why = init_from_problem(sb.get("init_from"))
    if why:
        bad["init_from"] = why
    return f"{bad}" if bad else None


def warm_selection_problem(arm: str, ws, spec: Mapping) -> Optional[str]:
    """ANA-05 (06M_B B2; FL-15 launch_fl.bind_warm_selection): why an FA_WARM / FA_WARM_FROZEN MAIN run is NOT bound
    to the registered 2-D warm selection of its stage, or None: format _2, the arm's stage, not a dry run, the
    registered file name, sha256 / self_sha256, and winner factors = the run's arm_spec factors (absent = 1)."""
    if not isinstance(ws, Mapping):
        return "no warm_selection (the LAUNCH binding to contracts/fl/FL_WARM_SELECTION_S7|S8.json)"
    st = warm_stage_of(arm)                                              # ANA-06: + the FL-16 warm-line arms
    bad = {}
    if ws.get("format") != WARM_SELECTION_FORMAT_2D:
        bad["format"] = ws.get("format")
    if ws.get("stage") != st:
        bad["stage"] = ws.get("stage")
    if ws.get("dry_run"):
        bad["dry_run"] = True
    if Path(str(ws.get("path") or "").replace("\\", "/")).name != f"FL_WARM_SELECTION_{st}.json":   # REV-ANA-05 F2
        bad["path"] = ws.get("path")
    for k in ("sha256", "self_sha256"):
        if not _hex64(ws.get(k)):
            bad[k] = ws.get(k)
    for k, sk in (("winner_factor", "warm_lr_factor"), ("winner_server_lr_factor", "warm_server_lr_factor")):
        try:
            if float(ws.get(k)) != float(spec.get(sk, 1.0)):
                bad[k] = (ws.get(k), spec.get(sk, 1.0))
        except (TypeError, ValueError):
            bad[k] = ws.get(k)
    return f"{bad}" if bad else None


def warm_stage_of(arm: str) -> Optional[str]:
    """ANA-06: the S7 / S8 stage whose registered 2-D warm selection an arm is bound to (FA_WARM / FA_WARM_FROZEN and
    every FL-16 warm-line arm through its variant), None for every other arm."""
    if arm in WARM_SELECTION_STAGE:
        return WARM_SELECTION_STAGE[arm]
    return WARM_SELECTION_STAGE[WARM_LINE_ARMS[arm][2]] if arm in WARM_LINE_ARMS else None


def warm_line_base_problem(arm: str, summ: Mapping) -> Optional[str]:
    """ANA-06 (FL-16): why an FL-16 warm-line run is NOT its label's semantics, or None: RUN_SUMMARY arm / arm_spec.arm =
    the launch label (DP8_FROZEN = DP_WARM_FROZEN, ...), cfg.method = the cold arm's method code, theta_pre init_from,
    frozen item tables exactly for the S8 variant, and the cold arm's knobs: FP mu > 0, PF cfg.pf, FA_Q8 upload codec,
    Poisson cfg.dp (clip set; z > 0 exactly for DP), no dp otherwise."""
    lab, base, variant, _eps = WARM_LINE_ARMS[arm]
    spec = summ.get("arm_spec") or {}
    cfg = spec.get("cfg") or {}
    bad = {}
    if summ.get("arm") != lab:
        bad["arm"] = summ.get("arm")
    if spec.get("arm", lab) != lab:
        bad["arm_spec.arm"] = spec.get("arm")
    if cfg.get("method", METHOD_CODE_OF[base]) != METHOD_CODE_OF[base]:
        bad["cfg.method"] = cfg.get("method")
    if bool(spec.get("frozen_item_tables")) != (variant == "FA_WARM_FROZEN"):
        bad["arm_spec.frozen_item_tables"] = spec.get("frozen_item_tables")
    why = init_from_problem(spec.get("init_from"))
    if why:
        bad["init_from"] = why
    dp = cfg.get("dp")
    if base in POISSON_BASE:
        if not isinstance(dp, Mapping) or dp.get("clip_norm") is None:
            bad["cfg.dp"] = dp
        elif (float(dp.get("noise_multiplier") or 0.0) > 0.0) != base.startswith("DP"):
            bad["cfg.dp.noise_multiplier"] = dp.get("noise_multiplier")
    elif dp is not None:
        bad["cfg.dp"] = dp
    try:
        mu = float(cfg.get("mu") or 0.0)
    except (TypeError, ValueError):
        mu = None
    if mu is None or (mu > 0.0) != (base == "FP"):
        bad["cfg.mu"] = cfg.get("mu")
    if (cfg.get("pf") is not None) != (base == "PF"):
        bad["cfg.pf"] = cfg.get("pf")
    if (spec.get("upload_codec") is not None) != (base == "FA_Q8"):
        bad["upload_codec"] = spec.get("upload_codec")
    return f"{bad}" if bad else None


def ft_pre_vs_pre_problem(ftp: Sequence[tuple], pres: Sequence[tuple]) -> Optional[str]:
    """DEV-FTPRE: every FT_PRE base = the ONE theta_pre the block's PRE entries were re-evaluated from, compared through
    the PRE init record: [(eid, start_init, start_init_record_sha256)] vs [(eid, pre_init)] where pre_init = the PRE
    REEVAL_RECORD's binding {checkpoint_sha256 (= fl_init.sha256), init_record_sha256, run_id}. None when it holds or
    the block has no FT_PRE entry."""
    if not ftp:
        return None
    if not pres:
        return ("the block has FT_PRE but no PRE entry: FT_PRE's base is checked against PRE's PRETRAIN_FOR_FL_INIT.json "
                "binding (and K9c = FT_PRE − PRE needs it)")
    ids = set()
    for eid, pi in pres:
        if not isinstance(pi, Mapping) or not _hex64(pi.get("checkpoint_sha256")) or \
                not _hex64(pi.get("init_record_sha256")):
            return (f"{eid} carries no pre_init (the PRE REEVAL_RECORD's PRETRAIN_FOR_FL_INIT.json binding: "
                    "checkpoint_sha256 + init_record_sha256), so FT_PRE's base cannot be checked")
        ids.add((pi["checkpoint_sha256"], pi["init_record_sha256"], pi.get("run_id")))
    if len(ids) != 1:
        return f"the PRE entries disagree on theta_pre {sorted(ids, key=str)} (PRE is ONE theta_pre, 06K §1)"
    ck, rs, rid = ids.pop()
    for eid, si, sirs in ftp:
        got = ((si or {}).get("checkpoint_sha256"), sirs, (si or {}).get("run_id"))
        if got[0] != ck or got[1] != rs or (rid is not None and got[2] != rid):
            return (f"{eid}: FT_PRE's base must be θ_pre of the PRE init record: FT_PRE starts from checkpoint "
                    f"{str(got[0])[:12]}… / init record {str(got[1])[:12]}… / run {got[2]!r}, PRE is "
                    f"{ck[:12]}… / {rs[:12]}… / {rid!r}")
    return None


def _rows_of(obj) -> list:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, Mapping) and isinstance(obj.get("rows"), list):
        return obj["rows"]
    raise AnaRefused("a rows file is neither a list of rows nor an object with a 'rows' list")


def _is_evidence(r: Mapping, metric: str, k: int) -> bool:
    return (r.get("aggregation") == "macro" and r.get("target_denominator") == "E2E" and r.get("metric") == metric
            and int(r.get("k", -1)) == k)


def _eq(a, b: float) -> bool:
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return False


def refuse_non_evidence(label: str, rows: Sequence[Mapping]) -> None:
    """A-F13: SMOKE / LR-screen / selection / search / CALIBRATION rows are never retention evidence."""
    for r in rows:
        if r.get("run_label") in REFUSED_RUN_LABELS or r.get("label_smoke") == "SMOKE_NOT_EVIDENCE":
            raise AnaRefused(f"{label}: a row is labelled {r.get('run_label') or r.get('label_smoke')} (not evidence)")
        if r.get("row_status") in REFUSED_ROW_STATUS:
            raise AnaRefused(f"{label}: a row has row_status {r.get('row_status')} (not retention evidence)")
        for f in ("role", "manifest_role", "run_role", "population_scope"):
            if r.get(f) == "CALIBRATION":
                raise AnaRefused(f"{label}: a row has {f} = CALIBRATION (the LR-screen / tuning population, not MAIN)")


def init_from_problem(ident) -> Optional[str]:
    """REV-DELTA-03 M2: why `ident` (a launch_fl arm_spec.init_from) is NOT the theta_pre identity of 06K §2 (the PRE
    run's endpoint_6.0 on PRETRAIN_TRAIN_5_100_NONMAIN, run id PRETRAIN_FOR_FL_*), or None when it is."""
    if not isinstance(ident, Mapping):
        return f"init_from {ident!r} is not an identity object"
    bad = {k: ident.get(k) for k, v in PRETRAIN_INIT.items() if ident.get(k) != v}
    if not str(ident.get("run_id") or "").startswith(PRETRAIN_RUN_PREFIX):
        bad["run_id"] = ident.get("run_id")
    if not re.fullmatch(r"[0-9a-f]{64}", str(ident.get("checkpoint_sha256") or "")):
        bad["checkpoint_sha256"] = ident.get("checkpoint_sha256")
    return f"init_from is not the theta_pre identity {PRETRAIN_INIT} ({bad})" if bad else None


def check_warm_init(arm: str, summary: Mapping, label: str) -> None:
    """REV-DELTA-03 M2 (the two-way rule of GATE-B B3's frozen tables): FA_WARM / FA_WARM_FROZEN / DP8_FROZEN need the
    run's arm_spec.init_from = the theta_pre identity; every other FL arm must not carry one (a cold-start FA copied
    into an FA_WARM_s<S> directory, or a warm run declared as FA, refuses)."""
    ident = (summary.get("arm_spec") or {}).get("init_from")
    if is_warm_start(arm):                                              # ANA-06: + the FL-16 warm-line arms
        why = init_from_problem(ident) if ident is not None else "no RUN_SUMMARY arm_spec.init_from"
        if why:
            raise AnaRefused(f"{label}: ARM_IDENTITY — {arm} starts from theta_pre, the run has {why} (06K §2)")
    elif ident is not None:
        raise AnaRefused(f"{label}: ARM_IDENTITY — arm {arm} carries arm_spec.init_from (a warm start from theta_pre is "
                         f"one of {WARM_START_ARMS}; 06K §2)")


def preview_row_problem(r: Mapping) -> Optional[str]:
    """06O O1: None for an ordinary row; the refusal reason for a row labelled preview_reference (v4int.reeval)."""
    pv = r.get("preview_reference")
    return None if pv is None else f"a row labelled preview_reference {pv!r}: {PREVIEW_REFERENCE_LABEL}"


def check_arm_identity(entry: Entry, r: Mapping) -> None:
    """A-F7: the rows' own train_manifest_id / method must be this arm's (a C_FULL / C_BIG swap refuses).
    REV-DELTA-03 M1: PRE / C_FULL rows must be v4int.reeval rows of that registered central source on every role
    (central_source = the arm, source_train_manifest_id = PRETRAIN_TRAIN_5_100_NONMAIN / FULL_TRAIN_5_100_TIEBREAK), PRE
    never carries method "C", and a row of a central source never enters under another arm. REV-DELTA-03 M2: a row
    labelled with a warm start (fl_init) enters only under FA_WARM / FA_WARM_FROZEN / DP8_FROZEN, from theta_pre.
    06O: a row labelled preview_reference (a C_FULL / PRE UNSEEN_VAL preview) never enters under any entry."""
    why = preview_row_problem(r)
    if why:
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — {why}")
    tm = r.get("train_manifest_id")
    if entry.role == "MAIN" and entry.arm in TRAIN_MANIFEST_OF and tm is not None and tm != TRAIN_MANIFEST_OF[entry.arm]:
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — the rows' train_manifest_id is {tm!r}, {entry.arm} trains on "
                         f"{TRAIN_MANIFEST_OF[entry.arm]} (A-F7)")
    if entry.role == "MAIN" and entry.arm == "PRE" and tm is not None and tm not in PRE_TRAIN_MANIFESTS:
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — PRE's rows carry train_manifest_id {tm!r}, not "
                         f"{PRE_TRAIN_MANIFESTS} (06K §1; A-F7)")
    cs, stm = r.get("central_source"), r.get("source_train_manifest_id")
    if entry.arm in SOURCE_TRAIN_MANIFEST:
        if cs != entry.arm or stm != SOURCE_TRAIN_MANIFEST[entry.arm]:
            raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — {entry.arm} rows must be v4int.reeval rows of the registered "
                             f"{entry.arm} source (central_source {entry.arm}, source_train_manifest_id "
                             f"{SOURCE_TRAIN_MANIFEST[entry.arm]}); the rows carry {cs!r} / {stm!r} (GATE-B B1; "
                             "REV-DELTA-03 M1)")
    elif cs is not None:
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — a row of the central source {cs!r} under arm {entry.arm} "
                         "(REV-DELTA-03 M1)")
    sst = r.get("source_seed_status")                                  # ANA-05: C_FULL s2027 = a DESCRIPTIVE source seed
    if entry.arm in DESCRIPTIVE_SEEDS and entry.seed in DESCRIPTIVE_SEEDS[entry.arm]:
        if sst != DESCRIPTIVE_SEED_STATUS or r.get("comparison_status") != DESCRIPTIVE_SEED_STATUS:
            raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — {entry.arm} seed {entry.seed} is a DESCRIPTIVE source seed; its "
                             f"rows must carry source_seed_status / comparison_status DESCRIPTIVE (v4int.reeval ANA-05), "
                             f"they carry {sst!r} / {r.get('comparison_status')!r} (06E §2; 06D §8)")
    elif sst is not None:
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — a row labelled source_seed_status {sst!r} under {entry.arm} seed "
                         f"{entry.seed} (only {DESCRIPTIVE_SEEDS} are DESCRIPTIVE source seeds; ANA-05)")
    bs = r.get("base_source")                                          # ANA-05 (06M_B B4): the FA_WARM / S8 base
    if entry.arm in FT_WARM_BASE:
        sb = entry.start_base or {}
        got = (bs, r.get("base_run_id"), r.get("base_checkpoint_sha256"))
        if got != (FT_WARM_BASE[entry.arm], sb.get("run_id"), sb.get("file_sha256")):
            raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — {entry.arm} rows must carry the base identity of the entry's "
                             f"start (base_source {FT_WARM_BASE[entry.arm]}, base_run_id, base_checkpoint_sha256); they "
                             f"carry {got!r} (launch_device ft_warm_row_labels; 06M_B B4)")
    elif bs in FT_WARM_BASE.values():
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — a row fine-tuned from a {bs} base under {entry.arm} (that is "
                         f"{'FT_FA_WARM' if bs == 'FA_WARM' else 'FT_FA_WARM_FROZEN'}; 06M_B B4)")
    elif bs in WARM_LAUNCH_LABELS or bs in WARM_LINE_ARMS:             # ANA-06 (DL 287 m3)
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — a row fine-tuned from a {bs} base under {entry.arm}: no device "
                         "arm is registered on a warm-line base (DL 287 m3)")
    why = warm_base_run_problem(entry.arm, r.get("base_run_id"))      # ANA-06 (DL 287 m3)
    if why:
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — the rows' base: {why}")
    fi = r.get("fl_init")
    if fi is not None and (not is_warm_start(entry.arm) or fi != PRETRAIN_INIT["kind"]
                           or not str(r.get("fl_init_run_id") or "").startswith(PRETRAIN_RUN_PREFIX)):
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — a row warm-started from {fi!r} / {r.get('fl_init_run_id')!r} "
                         f"under arm {entry.arm} (theta_pre warm starts are {WARM_START_ARMS}; REV-DELTA-03 M2)")
    fa = r.get("fl_arm")                                               # ANA-06: v4int.reeval's FL arm (the LAUNCH label)
    want_fa = WARM_LINE_ARMS[entry.arm][0] if entry.arm in WARM_LINE_ARMS else entry.arm
    if fa is not None and (fa in WARM_LAUNCH_LABELS or entry.arm in WARM_LINE_ARMS) and fa != want_fa:
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — the rows come from the FL arm {fa!r}, {entry.arm} is the launch_fl "
                         f"arm {want_fa} (FL-16 warm line; ANA-06)")
    meth = r.get("method")
    ok = ({entry.arm, FL_METHOD.get(entry.arm)} | set(FL_METHOD_ALT.get(entry.arm, ()))
          | ({"C"} if entry.arm in CENTRAL_ARMS and entry.arm != "PRE" else set()))
    if entry.arm in WARM_LINE_ARMS:                                    # ANA-06: rows say the label or the launch label
        ok = {entry.arm, WARM_LINE_ARMS[entry.arm][0]}
    if meth is not None and meth not in ok:
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — the rows carry method {meth!r} (A-F7)")


_LOCK_CACHE: dict = {}


def c_big_best_sha256(seed) -> Optional[str]:
    """The ARCHITECTURE_LOCK sha256 of the C_BIG seed's best.pt (PRACTICAL_BEST = best.pt = mark 3.0; the lock's
    `c_big.definition`), or None. Read from ARCH_LOCK_PATH (module attribute, so a test can point it elsewhere)."""
    path = Path(ARCH_LOCK_PATH)
    if not path.is_file():
        return None
    key = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    if key not in _LOCK_CACHE:
        try:
            lock = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            lock = {}
        _LOCK_CACHE.clear()
        ok = lock.get("document") == "ARCHITECTURE_LOCK"
        _LOCK_CACHE[key] = ((lock.get("c_big") or {}).get("checkpoints") or {}) if ok else {}
    return (((_LOCK_CACHE[key].get(str(seed)) or {}).get("best.pt")) or {}).get("sha256")


def deployed_c_big_best(entry: Entry, r: Mapping, cid: str) -> bool:
    """REV-DELTA-01 F9: the deployed C_BIG PRACTICAL_BEST checkpoint id is `02B_SASREC_MAIN_s<S>:best` (no `@3.0`). It is
    accepted iff the REEVAL row's checkpoint_sha256 is the ARCHITECTURE_LOCK sha256 of this seed's c_big best.pt."""
    if not cid.endswith(C_BIG_DEPLOYED_PB_ID):
        return False
    want, got = c_big_best_sha256(entry.seed), r.get("checkpoint_sha256")
    return bool(want) and got == want


def check_rule_vs_checkpoint(entry: Entry, r: Mapping) -> None:
    """A-F10: the declared checkpoint rule agrees with the row's efe / checkpoint_id / schedule_mark."""
    efe, cid, mark = r.get("efe"), str(r.get("checkpoint_id") or ""), r.get("schedule_mark")
    if efe is None and not cid and r.get("selection_rule") is None:
        raise AnaRefused(f"{entry.eid}: the evidence row has neither efe, checkpoint_id nor selection_rule: its "
                         "checkpoint rule cannot be verified (A-F10)")
    if entry.arm in NO_SELECTION_ARMS:              # PRE: theta_pre = the endpoint_6.0 checkpoint under either rule label
        if not (cid.endswith("endpoint_6.0") or _eq(efe, 6.0) or _eq(mark, 6.0)):
            raise AnaRefused(f"{entry.eid}: RULE_CHECKPOINT — {entry.arm} is the 6.0 endpoint (no selection; 06K §1), "
                             f"the row has checkpoint_id {cid!r} / efe {efe!r} (A-F10)")
        return
    if entry.rule == "CONTROLLED_BUDGET":
        if not (cid.endswith("endpoint_6.0") or _eq(efe, 6.0) or _eq(mark, 6.0)):
            raise AnaRefused(f"{entry.eid}: RULE_CHECKPOINT — CONTROLLED_BUDGET needs the 6.0 endpoint, the row has "
                             f"checkpoint_id {cid!r} / efe {efe!r} (A-F10)")
        return
    if "endpoint" in cid:
        raise AnaRefused(f"{entry.eid}: RULE_CHECKPOINT — a PRACTICAL_BEST row carries the endpoint checkpoint id "
                         f"{cid!r} (A-F10)")
    if entry.arm == REF and not (cid.endswith(f"best@{C_BIG_PB_MARK:.1f}") or _eq(efe, C_BIG_PB_MARK)
                                 or deployed_c_big_best(entry, r, cid)):
        raise AnaRefused(f"{entry.eid}: RULE_CHECKPOINT — C_BIG PRACTICAL_BEST is best@3.0 per the lock (or the deployed "
                         f"'...:best' id with the lock's best.pt sha256, F9), the row has checkpoint_id {cid!r} / "
                         f"efe {efe!r} / checkpoint_sha256 {r.get('checkpoint_sha256')!r} (A-F10)")
    best = entry.meta_auto.get("best_mark")
    if entry.run_dir_rel and best is not None and not (cid.endswith(f"best@{best:.1f}") or _eq(mark, best)
                                                       or _eq(efe, best)):
        raise AnaRefused(f"{entry.eid}: RULE_CHECKPOINT — the run's PRACTICAL_BEST mark is {best:.1f}, the row has "
                         f"checkpoint_id {cid!r} / efe {efe!r} (A-F10)")


def check_device_cell(entry: Entry, all_rows: Sequence[Mapping]) -> None:
    """A-F10 for device arms: the entry's device_cell is the run's selected cell whenever the rows mark one."""
    selected = sorted({str(r.get("device_cell")) for r in all_rows if r.get("device_selected") is True})
    cell = str(entry.row_filter.get("device_cell"))
    if selected and cell not in selected:
        raise AnaRefused(f"{entry.eid}: row_filter.device_cell {cell} is not the run's selected cell {selected} "
                         "(A-F10)")
    entry.rule_source = "DEVICE_SELECTION" if selected else "DEVICE_CELL_BY_INDEX"


def select_rows(entry: Entry, rows: list) -> dict:
    """The entry's three macro-E2E evidence rows after its row_filter and its declared rule (06A §7)."""
    all_rows = rows
    rows = [r for r in rows if all(r.get(f) == v for f, v in entry.row_filter.items())]
    refuse_non_evidence(entry.eid, rows)
    if entry.arm in DEVICE_ARMS:
        check_device_cell(entry, all_rows)
    out = {}
    for metric, k in METRICS:
        cand = [r for r in rows if _is_evidence(r, metric, k)]
        rules = sorted({str(r.get("selection_rule")) for r in cand if r.get("selection_rule") is not None})
        if entry.rule == RULE_FREE:
            if rules:
                raise AnaRefused(f"{entry.eid}: RULE_FREE entry but its rows carry selection_rule {rules}")
            hit = cand
        else:
            hit = [r for r in cand if r.get("selection_rule") == entry.rule]
            if not hit:
                if rules:
                    raise AnaRefused(f"{entry.eid}: RULE_MISMATCH — the rows carry selection_rule {rules}, the index "
                                     f"declares {entry.rule} (06A §7 pairing rule)")
                hit = [r for r in cand if r.get("selection_rule") is None]
                entry.rule_source = "INDEX"
        if len(hit) != 1:
            raise AnaRefused(f"{entry.eid}: {len(hit)} macro-E2E {metric}@{k} rows after the filter (need exactly 1)")
        r = hit[0]
        if r.get("value") is None or r.get("value_status") != "OK" or not math.isfinite(float(r["value"])):
            raise AnaRefused(f"{entry.eid}: the {metric}@{k} evidence row is not a defined value")
        if entry.seed is not None and r.get("seed") is not None and int(r["seed"]) != entry.seed:
            raise AnaRefused(f"{entry.eid}: SEED_MISMATCH — the rows carry seed {r['seed']}")
        check_arm_identity(entry, r)
        if entry.rule in RULES and entry.arm not in PB_ONLY_ARMS:
            check_rule_vs_checkpoint(entry, r)
        out[(metric, k)] = r
    return out


def _macro_of(manifest, ranks, metric: str, k: int):
    credits = LIB.M.credits_from_ranks(ranks, (int(k),))
    blk = LIB.M.population_block({f"{metric}@{k}": credits[f"{metric}@{k}"]}, manifest.eval_mask,
                                 manifest.eval_user_key)
    return blk["macro"][f"{metric}@{k}"]


def _subset_view(manifest, keys):
    np = LIB.np
    if keys is None:
        return manifest
    return dataclasses.replace(manifest, eval_mask=manifest.eval_mask & np.isin(manifest.eval_user_key, keys))


def check_rank_rows(label: str, manifest, ranks, rows: Mapping) -> float:
    """g2_run.check_bindings rule: each rank file reproduces its rows' macro-E2E values within 1e-9."""
    if ranks.shape != (manifest.n,):
        raise AnaRefused(f"{label}: rank file has {ranks.shape[0]} entries, the manifest {manifest.n}")
    worst = 0.0
    for (metric, k), r in rows.items():
        got = _macro_of(manifest, ranks, metric, k)
        if is_undef(got):
            raise AnaRefused(f"{label}: the recomputed {metric}@{k} is undefined")
        d = abs(float(got) - float(r["value"]))
        if not d <= ROW_VALUE_TOL:
            raise AnaRefused(f"{label}: the {metric}@{k} row value disagrees with its rank file (|d| = {d:.3g})")
        worst = max(worst, d)
    return worst


def dropped_visits(summary: Mapping):
    """(total dropped visits, source) of a FL run: the participation log's dropped ids (RUN_SUMMARY.participation_log
    = [{"round", "dropped": [ids]}], top level or inside `participation`) when the summary carries the log, else the
    run's own count RUN_SUMMARY.participation.dropped (06I 7b / audit S4). (None, reason) if neither exists."""
    part = summary.get("participation")
    log = summary.get("participation_log")
    if log is None and isinstance(part, Mapping):
        log = part.get("participation_log")
    if isinstance(log, list):
        if not all(isinstance(x, Mapping) and isinstance(x.get("dropped"), list) for x in log):
            raise AnaRefused("RUN_SUMMARY.participation_log entries need a `dropped` id list")
        n = sum(len(x["dropped"]) for x in log)
        if isinstance(part, Mapping) and part.get("dropped") is not None and int(part["dropped"]) != n:
            raise AnaRefused(f"RUN_SUMMARY participation.dropped {part['dropped']} != {n} dropped ids of its "
                             "participation_log")
        return n, "PARTICIPATION_LOG_IDS"
    if isinstance(part, Mapping) and part.get("dropped") is not None:
        return int(part["dropped"]), "PARTICIPATION_SUMMARY_COUNT"
    return None, "NO_PARTICIPATION_RECORD"


def charged_bytes(summary: Mapping) -> dict:
    """06I 7(b) with the variant registered in DECISION_LOG row 218 (FIX-DELTA-02 M4): bytes per client with the
    dropped visits charged their DOWNLOAD only (a dropped client receives the model and never uploads):
    bytes_per_client_charged_dropped = the run's registered mean (visits x (down + up), 06A 3) + dropped visits x down
    / n_clients. DESCRIPTIVE beside it: bytes_per_client_charged_dropped_up_down = the mean + dropped visits x (down +
    up) / n_clients. The per-client vector is not in RUN_SUMMARY, so only the mean is available."""
    b = summary.get("bytes") or {}
    pv, pc = b.get("per_visit") or {}, b.get("per_client") or {}
    n_clients = (summary.get("plan") or {}).get("n_clients")
    n_dropped, src = dropped_visits(summary)
    base = {"bytes_per_client_charged_dropped": None, "bytes_per_client_charged_dropped_up_down": None,
            "dropped_visits": n_dropped, "charged_bytes_source": src}
    if pc.get("mean") is None or pv.get("send_plus_receive_bytes") is None or pv.get("download_bytes") is None:
        return dict(base, charged_bytes_source="NO_BYTES_RECORD")
    if n_dropped is None:
        return base
    if not n_clients:
        return dict(base, charged_bytes_source="NO_N_CLIENTS")
    per_visit = float(pv["send_plus_receive_bytes"])
    down = float(pv["download_bytes"])
    return dict(base, bytes_per_client_charged_dropped=float(pc["mean"]) + n_dropped * down / float(n_clients),
                bytes_per_client_charged_dropped_up_down=float(pc["mean"]) + n_dropped * per_visit / float(n_clients))


def frozen_once_meta(summary: Mapping, label: str) -> dict:
    """GATE-B B3 (review 09 [8-C1]): S8's registered bytes per client = launch_fl's per_client_incl_frozen_once mean
    (the frozen item tables charged once per visited client); the charged-dropped variants add the same once component
    to the training-traffic charged values; the fleet-mean once component (incl - excl) is the first-visit step of
    curve_bytes. Refuses a run without the FL-12b frozen-once bytes record."""
    b = summary.get("bytes") or {}
    pv, pc, inc = b.get("per_visit") or {}, b.get("per_client") or {}, b.get("per_client_incl_frozen_once") or {}
    once = pv.get("frozen_bytes_once")
    if not summary.get("frozen_item_tables") or inc.get("mean") is None or once is None or pc.get("mean") is None:
        what = "a DP8_FROZEN (S8 + DP)" if str(label).startswith("DP8_FROZEN") else "an FA_WARM_FROZEN (S8)"
        raise AnaRefused(f"{label}: ARM_IDENTITY — {what} run needs RUN_SUMMARY frozen_item_tables, "
                         "bytes.per_visit.frozen_bytes_once and bytes.per_client_incl_frozen_once (launch_fl FL-12b; "
                         "06K §3)")
    step = float(inc["mean"]) - float(pc["mean"])
    if step < 0 or step > float(once) * (1 + 1e-12):
        raise AnaRefused(f"{label}: per_client_incl_frozen_once.mean - per_client.mean = {step} is not within "
                         f"[0, frozen_bytes_once = {once}] (06K §3)")
    ch = charged_bytes(summary)
    out = {"bytes_per_client": float(inc["mean"]), "bytes_per_client_excl_frozen_once": float(pc["mean"]),
           "frozen_bytes_once": int(once), "frozen_once_per_client_mean": step, "bytes_rule": FROZEN_ONCE_RULE}
    for k in ("bytes_per_client_charged_dropped", "bytes_per_client_charged_dropped_up_down"):
        if ch.get(k) is not None:
            out[k] = float(ch[k]) + step
    return out


def _fl_meta(summary: Mapping) -> dict:
    b = summary.get("bytes") or {}
    pv, pc, fl = b.get("per_visit") or {}, b.get("per_client") or {}, b.get("fleet") or {}
    cfg = ((summary.get("arm_spec") or {}).get("cfg") or {})
    dp = cfg.get("dp") or {}
    return {"bytes_per_visit": pv.get("send_plus_receive_bytes"), "bytes_up_per_visit": pv.get("upload_bytes"),
            "bytes_down_per_visit": pv.get("download_bytes"), "bytes_per_client": pc.get("mean"),
            "bytes_fleet": fl.get("total"), "param_count": pv.get("shared_numel"), "rounds": summary.get("rounds"),
            "consumed_efe": (summary.get("counters") or {}).get("efe"), "dp_z": dp.get("noise_multiplier"),
            "dp_clip": dp.get("clip_norm"), "lr_peak_local": cfg.get("lr_peak_local"), **charged_bytes(summary)}


def check_rank_coverage(label: str, view, ranks, hint: str = "") -> None:
    """A-F6: every RANKABLE row of the evaluated view has a rank >= 1 (an unscored -1 would be scored as a miss)."""
    np = LIB.np
    bad = int(np.count_nonzero(ranks[view.rankable] < 1))
    if bad:
        raise AnaRefused(f"{label}: {bad} RANKABLE row(s) of the evaluated view have no rank >= 1 (an unscored -1 is "
                         f"counted as a miss; A-F6){hint}")


def load_marks(block: Path, entry: Entry, manifest) -> list:
    rd = _inside(block, entry.run_dir_rel)
    summ = _read_json(rd / "RUN_SUMMARY.json")
    if summ.get("status") != "DONE_MAIN":
        raise AnaRefused(f"{entry.eid}: RUN_SUMMARY status {summ.get('status')!r} is not DONE_MAIN (A-F13)")
    want_arm = FL_METHOD.get(entry.arm, entry.arm)
    alt = FL_METHOD_ALT.get(entry.arm, ())
    if entry.arm in WARM_LINE_ARMS:                                    # ANA-06: the FL-16 launch label, never omitted
        want_arm, alt = WARM_LINE_ARMS[entry.arm][0], ()
    if (summ.get("arm") is not None or entry.arm in WARM_LINE_ARMS) and summ.get("arm") not in (want_arm, *alt):
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — the run dir is arm {summ.get('arm')!r}, not {want_arm} (A-F7)")
    check_warm_init(entry.arm, summ, entry.eid)                          # REV-DELTA-03 M2 (both directions)
    spec = summ.get("arm_spec") or {}
    if spec.get("role", "MAIN") != "MAIN":                              # ANA-05: a CALIB screen / WARM_PREVIEW run
        raise AnaRefused(f"{entry.eid}: the run is role {spec.get('role')!r}, not a MAIN run: "
                         + (PREVIEW_LABEL if spec.get("init_from") is not None else
                            "an LR-screen / tuning run is never a block entry (A-F13)"))
    if entry.arm in WARM_LINE_ARMS:                                    # ANA-06: the label's base semantics (FL-16)
        why = warm_line_base_problem(entry.arm, summ)
        if why:
            raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — the run is not {entry.arm} = launch_fl "
                             f"{WARM_LINE_ARMS[entry.arm][0]} (the {WARM_LINE_ARMS[entry.arm][1]} semantics from theta_pre, "
                             f"variant {WARM_LINE_ARMS[entry.arm][2]}): {why} (FL-16; ANA-06)")
    if warm_stage_of(entry.arm) is not None:                            # ANA-05 (06M_B B2): the 2-D S7 / S8 binding
        why = warm_selection_problem(entry.arm, entry.warm_selection, spec)
        if why:
            raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — {entry.arm} MAIN is not bound to the registered 2-D "
                             f"FL_WARM_SELECTION_{warm_stage_of(entry.arm)}.json: {why} (06M_B B2; FL-15)")
    entry.meta_auto.update(run_id=summ.get("run_id"), arm_spec_sha256=summ.get("arm_spec_sha256"),   # ANA-05 bindings
                           init_from=spec.get("init_from"),
                           warm_lr_factor=float(spec.get("warm_lr_factor", 1.0)) if is_warm_start(entry.arm) else None,
                           warm_server_lr_factor=(float(spec.get("warm_server_lr_factor", 1.0))
                                                  if is_warm_start(entry.arm) else None))
    entry.meta_auto.update(_fl_meta(summ))
    if entry.arm in FROZEN_ONCE_ARMS:                                   # GATE-B B3: S8 bytes incl. the tables once
        entry.meta_auto.update(frozen_once_meta(summ, entry.eid))
        pv = (summ.get("bytes") or {}).get("per_visit") or {}           # ANA-05: S8's own fl.comm per-visit record
        entry.meta_auto.update(uploaded_numel=pv.get("uploaded_numel", pv.get("shared_numel")),
                               frozen_numel=pv.get("frozen_numel"))
    elif (summ.get("bytes") or {}).get("per_client_incl_frozen_once") is not None or summ.get("frozen_item_tables"):
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — the run froze its item tables (S8) but the entry is arm "
                         f"{entry.arm} (use FA_WARM_FROZEN, or DP8_FROZEN for its DP variant; 06K §3)")
    z = entry.meta_auto.get("dp_z")
    if entry.arm in DP_Z and z is not None and not _eq(z, DP_Z[entry.arm]):
        raise AnaRefused(f"{entry.eid}: ARM_IDENTITY — the run's noise multiplier z = {z} is not {entry.arm}'s "
                         f"registered {DP_Z[entry.arm]} (A-F7)")
    if summ.get("best_half_efe") is None:
        raise AnaRefused(f"{entry.eid}: RUN_SUMMARY has no best_half_efe (the PRACTICAL_BEST mark)")
    best = float(summ["best_half_efe"]) / 2.0
    entry.meta_auto.update(best_mark=best, still_improving_at_6_0=bool(best >= 5.5))       # 06D §5 (descriptive)
    evals = {}
    mj = rd / "metrics.jsonl"
    if mj.is_file():
        for line in mj.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("kind") == "eval":
                    evals[f"{float(rec['mark']):.1f}"] = rec            # a resumed run: the last line of a mark wins
    if sorted(evals, key=float) != [f"{m:.1f}" for m in FL_MARKS]:
        raise AnaRefused(f"{entry.eid}: the run dir has marks {sorted(evals, key=float)}, not the 12 registered "
                         "0.5 .. 6.0 (06A §3)")
    if entry.ranks_rel is None:                    # PRACTICAL_BEST = the strictly-best mark, CONTROLLED_BUDGET = 6.0
        mark = 6.0 if entry.rule == "CONTROLLED_BUDGET" else best
        entry.ranks_rel = f"{entry.run_dir_rel}/evals/eval_{mark:.1f}.ranks_int32.npy"
        entry.notes.append(f"ranks derived from the run dir ({entry.rule} -> mark {mark:.1f})")
    marks = []
    for tag in sorted(evals, key=float):
        rec = evals[tag]
        rows_p = rd / "evals" / f"eval_{tag}.rows.json"
        ranks_p = rd / "evals" / f"eval_{tag}.ranks_int32.npy"
        if not (rows_p.is_file() and ranks_p.is_file()):
            raise AnaRefused(f"{entry.eid}: mark {tag} has no rows / ranks file")
        rows = [r for r in _rows_of(_read_json(rows_p)) if r.get("aggregation") == "macro"
                and r.get("target_denominator") == "E2E"]
        refuse_non_evidence(f"{entry.eid} mark {tag}", rows)
        ev = {}
        for metric, k in METRICS:
            hit = [r for r in rows if _is_evidence(r, metric, k)]
            if len(hit) != 1:
                raise AnaRefused(f"{entry.eid}: mark {tag} has {len(hit)} macro-E2E {metric}@{k} rows")
            check_arm_identity(entry, hit[0])
            ev[(metric, k)] = hit[0]
        ranks = LIB.np.load(ranks_p).astype(LIB.np.int64)
        check_rank_rows(f"{entry.eid} mark {tag}", manifest, ranks, ev)
        check_rank_coverage(f"{entry.eid} mark {tag}", manifest, ranks)
        marks.append({"mark": float(tag), "efe_raw": rec.get("efe"), "round": rec.get("round"), "rows": ev,
                      "ranks": ranks, "files": {"rows": _sha_file(rows_p), "ranks": _sha_file(ranks_p)}})
    return marks


def _entry_rowsets(e: Entry) -> list:
    return [("evidence", e.rows)] + [(f"mark {m['mark']:.1f}", m["rows"]) for m in e.marks]


def check_key(entries: Sequence[Entry], ctx: Mapping[str, RoleCtx], idx: Mapping, split: str) -> dict:
    """The block comparison key (ANA-01 + A-F5 / 06E §4): KEY_FIELDS, the manifest (eval_manifest_id / query_order_hash
    per role) over every evidence and FL-mark row of every non-historical entry of the main block; the pinned-class status
    and one device_pin_sha256; INTENDED_DIFFERENCES recorded per entry; the ONNX sub-block checked on its own."""
    if str(idx.get("split", "")).upper() != split:
        raise AnaRefused(f"{INDEX_NAME} declares split {idx.get('split')!r}, the CLI --split {split}")
    seen = {f: {} for f in KEY_FIELDS}
    hist, intended, pins, statuses = [], {}, {}, set()
    want_dev = idx.get("eval_device_class")
    for e in entries:
        if e.block != MAIN_BLOCK:                    # ONNX_CPU (A-F2) and BASELINE_CPU (06L-C §C3): own checks below
            continue
        m = ctx[e.role].manifest
        qhash = m.hashes["eval_ordered_decision_id_sha256"]
        qhash_ev = _subset_view(m, e.subset_keys).hashes["eval_ordered_decision_id_sha256"]   # users_subset rows
        mid = idx["roles"][e.role]["eval_manifest_id"]
        diffs = []
        for what, rows in _entry_rowsets(e):
            want_q = qhash_ev if what == "evidence" else qhash
            for r in rows.values():
                if r.get("split") not in SPLIT_VALUES[split]:
                    raise AnaRefused(f"{e.eid}: {what} row split {r.get('split')!r} is not a {split} split (a "
                                     "historical reference too must come from the --split split)")
                if r.get("query_order_hash") != want_q:
                    diffs.append(f"{what}: query_order_hash is not the loaded {mid} eval-ordered hash"
                                 + (" of the users_subset" if e.subset_keys is not None and what == "evidence" else ""))
                if r.get("eval_manifest_id") not in (None, mid):
                    diffs.append(f"{what}: eval_manifest_id {r.get('eval_manifest_id')!r} != {mid}")
                for f in KEY_FIELDS:
                    if r.get(f) in (None, ""):
                        diffs.append(f"{what}: {f} missing")
                    elif not e.historical:
                        seen[f].setdefault(str(r.get(f)), set()).add(e.eid)
                if not e.historical:
                    statuses.add(str(r.get("status")))
                    if want_dev == PINNED_CLASS and r.get("status") != PINNED_STATUS:
                        diffs.append(f"{what}: status {r.get('status')!r} is not {PINNED_STATUS} on the pinned class "
                                     f"{PINNED_CLASS} (06D §1 / 06E §4)")
                    pins.setdefault(str(r.get("device_pin_sha256")), set()).add(e.eid)
        intended[e.eid] = {f: sorted({str(r.get(f)) for _w, rows in _entry_rowsets(e) for r in rows.values()})
                           for f in INTENDED_DIFFERENCES}
        if e.historical:
            hist.append({"entry": e.eid, "label": "HISTORICAL_REFERENCE", "notes": sorted(set(diffs))})
            continue
        if diffs:
            raise AnaRefused(f"{e.eid}: comparison key — {sorted(set(diffs))[0]}")
    for f in KEY_FIELDS:
        if len(seen[f]) > 1:
            raise AnaRefused(f"comparison key: the block mixes {f} values {sorted(seen[f])} (HISTORICAL_REFERENCE only)")
    got = {f: next(iter(seen[f])) if seen[f] else None for f in KEY_FIELDS}
    if want_dev is not None and got.get("eval_device_class") != want_dev:
        raise AnaRefused(f"comparison key: rows carry eval_device_class {got.get('eval_device_class')!r}, the index "
                         f"declares {want_dev!r}")
    if got.get("split") not in SPLIT_VALUES[split]:
        raise AnaRefused(f"comparison key: rows carry split {got['split']!r}, not a {split} split")
    pin = None
    if want_dev == PINNED_CLASS:
        if len(pins) != 1 or "None" in pins:
            raise AnaRefused(f"comparison key: the pinned block needs ONE device_pin_sha256, the rows carry "
                             f"{sorted(pins)} (06E §4)")
        pin = next(iter(pins))
    return {"fields": list(KEY_FIELDS) + ["eval_manifest_id", "query_order_hash"], "values": got,
            "device_pin_sha256": pin, "intended_differences_fields": list(INTENDED_DIFFERENCES),
            "intended_differences": intended, "historical_reference": hist, "row_status_values": sorted(statuses),
            "onnx_block": check_onnx_block([e for e in entries if e.onnx], ctx, idx, split),
            "baseline_block": check_baseline_block([e for e in entries if e.block == BASELINE_BLOCK], ctx, idx, split,
                                                   got)}


def baseline_key_consistency(entries: Sequence[Entry], main_key: Mapping) -> None:
    """REV-DELTA-03 m3: the BASELINE_CPU rows share ONE value of every E9.1 key field among themselves (the exempt
    eval_device_class / precision_id / objective_id included), and the main block must carry every field they are
    checked against (otherwise the check against it would silently check nothing)."""
    seen = {f: {} for f in KEY_FIELDS}
    for e in entries:
        for r in e.rows.values():
            for f in KEY_FIELDS:
                seen[f].setdefault(str(r.get(f)), e.eid)
    split_ = {f: v for f, v in seen.items() if len(v) > 1}
    if split_:
        raise AnaRefused(f"{BASELINE_BLOCK} sub-block: the baseline rows do not share one key "
                         f"{ {f: sorted(v) for f, v in split_.items()} } (every E9.1 field, the exempt ones included, is "
                         "one value inside the sub-block; 06L-C §C3, REV-DELTA-03 m3)")
    missing = [f for f in BASELINE_KEY_FIELDS if main_key.get(f) is None]
    if missing:
        raise AnaRefused(f"{BASELINE_BLOCK} sub-block: the main block carries no value for {missing}, so the baseline "
                         "rows cannot be key-checked against it (06L-C §C3, REV-DELTA-03 m3)")


def check_baseline_block(entries: Sequence[Entry], ctx: Mapping[str, RoleCtx], idx: Mapping, split: str,
                         main_key: Mapping) -> Optional[dict]:
    """06L-C §C3: the BASELINE_CPU sub-block — baselines scored on the CPU through the frozen T3 path (v4base.scoring)
    with per-row rank files. Every row must be a --split row of the role's loaded manifest view (query_order_hash,
    eval_manifest_id) and carry the main block's key in every field except eval_device_class / precision_id /
    objective_id (CPU scoring; v4base placeholders), so the ranks are exact ranks of the same rows under the same
    evaluator, catalogue, label policy, data protocol and feature contract."""
    if not entries:
        return None
    devs = set()
    for e in entries:
        m = ctx[e.role].manifest
        want_q = m.hashes["eval_ordered_decision_id_sha256"]
        mid = idx["roles"][e.role]["eval_manifest_id"]
        for r in e.rows.values():
            if r.get("split") not in SPLIT_VALUES[split]:
                raise AnaRefused(f"{e.eid}: {BASELINE_BLOCK} row split {r.get('split')!r} is not a {split} split")
            if r.get("query_order_hash") != want_q or r.get("eval_manifest_id") not in (None, mid):
                raise AnaRefused(f"{e.eid}: {BASELINE_BLOCK} row is not on the loaded {mid} rows (query_order_hash / "
                                 "eval_manifest_id; 06L-C §C3)")
            for f in BASELINE_KEY_FIELDS:                  # check_key records the key values as strings
                if main_key.get(f) is not None and str(r.get(f)) != str(main_key.get(f)):
                    raise AnaRefused(f"{e.eid}: {BASELINE_BLOCK} row {f} {r.get(f)!r} is not the block's "
                                     f"{main_key.get(f)!r} (only {BASELINE_EXEMPT_KEY_FIELDS} may differ; 06L-C §C3)")
            devs.add(str(r.get("eval_device_class")))
    baseline_key_consistency(entries, main_key)          # REV-DELTA-03 m3 (after the per-row check against the block)
    return {"comparison_block": BASELINE_BLOCK, "entries": sorted(e.eid for e in entries),
            "key_fields_checked": list(BASELINE_KEY_FIELDS), "exempt_fields": list(BASELINE_EXEMPT_KEY_FIELDS),
            "eval_device_class_values": sorted(devs),
            "note": "CPU-scored baseline ranks of the same manifest rows (06L-C §C3); paired with A40 rows on the same "
                    "W as comparison BASELINE_CPU_EXACT_RANKS"}


def check_onnx_block(entries: Sequence[Entry], ctx: Mapping[str, RoleCtx], idx: Mapping, split: str) -> Optional[dict]:
    """A-F2 / 06E §3: the ONNX_CPU sub-block — one runtime identity (onnxruntime version, intra_op_num_threads, graph
    optimisation level, rows npz / meta sha256, manifest_n), one evaluator (all rows-free = values from the rank files by
    this script's library, or all with rows of one key) and one users view (the same subset query hash) per role."""
    if not entries:
        return None
    rt = {e.eid: e.meta_auto.get("onnx_runtime") for e in entries}
    if None in rt.values() or len({json.dumps(v, sort_keys=True) for v in rt.values()}) != 1:
        raise AnaRefused(f"{ONNX_BLOCK} sub-block: the entries do not share one runtime identity {rt} (A-F2)")
    kinds = {e.eid: e.rule_source for e in entries}
    if len(set(kinds.values())) != 1:
        raise AnaRefused(f"{ONNX_BLOCK} sub-block: rows-free and rows-carrying entries are mixed {kinds} (one evaluator)")
    views = {}
    for e in entries:
        m = ctx[e.role].manifest
        views.setdefault(e.role, set()).add(_subset_view(m, e.subset_keys).hashes["eval_ordered_decision_id_sha256"])
    if any(len(v) != 1 for v in views.values()):
        raise AnaRefused(f"{ONNX_BLOCK} sub-block: the entries of one role are on different user views (query hashes)")
    key = {}
    if next(iter(kinds.values())) != "RANKS":
        seen = {f: set() for f in ONNX_KEY_FIELDS}
        for e in entries:
            want_q = _subset_view(ctx[e.role].manifest, e.subset_keys).hashes["eval_ordered_decision_id_sha256"]
            for r in e.rows.values():
                if r.get("query_order_hash") != want_q or r.get("split") not in SPLIT_VALUES[split]:
                    raise AnaRefused(f"{e.eid}: {ONNX_BLOCK} row query_order_hash / split is not its view's")
                for f in ONNX_KEY_FIELDS:
                    seen[f].add(str(r.get(f)))
        bad = sorted(f for f, v in seen.items() if len(v) != 1)
        if bad:
            raise AnaRefused(f"{ONNX_BLOCK} sub-block: the rows differ in {bad}")
        key = {f: next(iter(v)) for f, v in seen.items()}
    return {"comparison_block": ONNX_BLOCK, "entries": sorted(rt), "runtime_identity": next(iter(rt.values())),
            "query_order_hash_by_role": {r: next(iter(v)) for r, v in sorted(views.items())},
            "evaluator": ("values computed from the rank files by this script's library (v4eval.metrics; "
                          "library_files_sha256)" if not key else "rows"), "key": key,
            "note": "exempt from the A40 device / status checks only (A-F2 / 06E §3); K16 is CONFIRMATORY only here"}


ONNX_IDENTITY_FIELDS = ("onnxruntime_version", "intra_op_num_threads", "graph_optimization_level", "rows_npz_sha256",
                        "rows_meta_sha256", "manifest_n")


ONNX_TAG_OF = {"C_SMALL_INT8": "onnx_int8", "C_SMALL_FP32": "onnx_fp32"}              # v4dev.bench tags (DEV-04)


def bench_retention_block(bench: Mapping, ranks_sha256: str, arm: str) -> Tuple[str, dict]:
    """The BENCH_RESULT.json `retention[tag]` block (v4dev.bench, DEV-04: a dict keyed by tag) of this arm's tag
    (onnx_int8 / onnx_fp32) whose ranks_sha256 is this rank file's; refuses otherwise (an INT8 entry can never point
    at the FP32 ranks, even when both rank files are byte-identical)."""
    ret = bench.get("retention")
    if not isinstance(ret, Mapping) or not ret:
        raise AnaRefused("BENCH_RESULT.json has no `retention` blocks (v4dev.bench --rows; A-F2)")
    tag = ONNX_TAG_OF.get(arm)
    blk = ret.get(tag) if tag else None
    if not isinstance(blk, Mapping) or blk.get("ranks_sha256") != ranks_sha256:
        raise AnaRefused(f"{arm}: the rank file sha256 is not BENCH_RESULT retention[{tag!r}].ranks_sha256 (tags "
                         f"{sorted(ret)}; A-F2)")
    return tag, blk


def onnx_runtime_identity(bench: Mapping, blk: Mapping) -> dict:
    top = bench.get("onnxruntime") or {}
    ident = {f: blk.get(f, top.get(f)) for f in ONNX_IDENTITY_FIELDS}
    missing = [f for f in ("onnxruntime_version", "intra_op_num_threads", "graph_optimization_level", "manifest_n")
               if ident.get(f) is None]
    if missing:
        raise AnaRefused(f"BENCH_RESULT retention block lacks the runtime identity fields {missing} (A-F2)")
    return ident


def users_set_sha256(hexes) -> str:
    """DEV-04 rows.meta.json first_users_set_sha256: sha256 of the sorted selected client_key_hex joined by '\\n'."""
    return _sha_bytes("\n".join(sorted(str(h) for h in hexes)).encode("ascii"))


def first_users_v2(ctx: RoleCtx, n: int):
    """06E §3 / DEV-04: the first n E2E users of the role in selection_key_v2 order (census_v2.core.selection_key_v2 over
    client_key_hex; ties by eval_user_key) -> (eval_user_keys sorted, set sha256)."""
    np = LIB.np
    from census_v2.core import selection_key_v2
    if ctx.hex_by_euk is None:
        raise AnaRefused(f"role {ctx.role}: users_subset SELECTION_KEY_V2_FIRST needs the users table (client_key_hex)")
    m = ctx.manifest
    keyed = sorted((selection_key_v2(bytes.fromhex(str(ctx.hex_by_euk[u]))), int(u))
                   for u in np.unique(m.eval_user_key[m.eval_mask]).tolist())
    chosen = [u for _k, u in keyed[:int(n)]]
    return np.asarray(sorted(chosen), dtype=np.int64), users_set_sha256(str(ctx.hex_by_euk[u]) for u in chosen)


def _resolve_subset(e: Entry, block: Path, c: RoleCtx, files: dict):
    np = LIB.np
    if e.subset_rel is None:
        return None
    m = c.manifest
    e2e = np.unique(m.eval_user_key[m.eval_mask])
    if isinstance(e.subset_rel, Mapping):                 # {"rule": "SELECTION_KEY_V2_FIRST", "n": N, "set_sha256": ..}
        keys, sha = first_users_v2(c, int(e.subset_rel["n"]))
        if sha != e.subset_rel["set_sha256"]:
            raise AnaRefused(f"{e.eid}: the first {e.subset_rel['n']} E2E users in selection_key_v2 order have set "
                             f"sha256 {sha}, the dumped rows' first_users_set_sha256 is {e.subset_rel['set_sha256']} "
                             "(06E §3; A-F6: the subset must be exactly the dumped users)")
    else:
        sp = _inside(block, e.subset_rel)
        files[str(e.subset_rel)] = _sha_file(sp)
        keys = np.unique(np.load(sp).astype(np.int64))
        if not np.all(np.isin(keys, e2e)):
            raise AnaRefused(f"{e.eid}: users_subset names users outside the {e.role} E2E user set")
    if keys.size == 0:
        raise AnaRefused(f"{e.eid}: users_subset is empty")
    e.meta_auto["users_subset_sha256_eval_user_key"] = _sha_bytes(np.ascontiguousarray(keys, dtype="<i8").tobytes())
    if c.hex_by_euk is not None:
        e.meta_auto["users_subset_set_sha256"] = users_set_sha256(str(c.hex_by_euk[u]) for u in keys.tolist())
    e.meta_auto["users_subset_n"] = int(keys.size)
    return keys


def load_block(block: Path, split: str, context_loader, gate: Optional[Mapping] = None) -> SimpleNamespace:
    np = LIB.np
    idx, entries = load_index(block, split)
    ctx = {}
    for role, spec in sorted(idx["roles"].items()):
        c = context_loader(role, spec)
        if c.manifest.manifest_id != spec["eval_manifest_id"]:
            raise AnaRefused(f"role {role}: loaded manifest {c.manifest.manifest_id!r} != {spec['eval_manifest_id']}")
        if split == "TEST":
            if gate is None:
                raise AnaRefused("a TEST block needs the open TEST gate (E10.1)")
            check_test_header(role, c.manifest.header or {}, gate)
        elif "TEST" in str(c.manifest.manifest_id).upper():
            raise AnaRefused(f"role {role}: a TEST manifest in a VALIDATION block")
        ctx[role] = c
    files = {INDEX_NAME: _sha_file(block / INDEX_NAME)}
    worst = 0.0
    for e in entries:
        m = ctx[e.role].manifest
        e.subset_keys = _resolve_subset(e, block, ctx[e.role], files)
        view = _subset_view(m, e.subset_keys)
        if e.run_dir_rel:                                             # first: best mark + still-improving (A-F10)
            e.marks = load_marks(block, e, m)
            for mk in e.marks:
                files[f"{e.run_dir_rel}/evals/eval_{mk['mark']:.1f}.rows.json"] = mk["files"]["rows"]
                files[f"{e.run_dir_rel}/evals/eval_{mk['mark']:.1f}.ranks_int32.npy"] = mk["files"]["ranks"]
            files[f"{e.run_dir_rel}/RUN_SUMMARY.json"] = _sha_file(_inside(block, e.run_dir_rel) / "RUN_SUMMARY.json")
        if e.rows_rel:
            rp = _inside(block, e.rows_rel)
            files[e.rows_rel] = _sha_file(rp)
            e.rows = select_rows(e, _rows_of(_read_json(rp)))
        if e.ranks_rel:
            kp = _inside(block, e.ranks_rel)
            files[e.ranks_rel] = _sha_file(kp)
            e.ranks = np.load(kp).astype(np.int64)
            if e.ranks.shape != (m.n,):
                raise AnaRefused(f"{e.eid}: rank file has {e.ranks.shape[0]} entries, the manifest {m.n}")
            if e.bench_rel:                                           # A-F2: BENCH_RESULT binds the rank file
                bp = _inside(block, e.bench_rel)
                files[str(e.bench_rel)] = _sha_file(bp)
                bench = _read_json(bp)
                tag, blk = bench_retention_block(bench, files[e.ranks_rel], e.arm)
                ident = onnx_runtime_identity(bench, blk)
                if int(ident["manifest_n"]) != m.n:
                    raise AnaRefused(f"{e.eid}: BENCH_RESULT manifest_n {ident['manifest_n']} != the {e.role} manifest "
                                     f"{m.n}")
                e.meta_auto.update(onnx_runtime=ident, bench_tag=tag,
                                   **{k: blk.get(k) for k in ("model_sha256", "n_rows_scored", "n_oov_skipped")})
            if not e.historical:
                hint = (" — an ONNX arm evaluated on a users subset must declare users_subset"
                        if e.arm in ONNX_ARMS and e.subset_keys is None else "")
                check_rank_coverage(e.eid, view, e.ranks, hint)
                if e.rows_rel:
                    worst = max(worst, check_rank_rows(e.eid, view, e.ranks, e.rows))
        if not e.rows_rel:                                            # ONNX rows-free: values from the ranks
            e.rows = {}
            for metric, k in METRICS:
                v = _macro_of(view, e.ranks, metric, k)
                if is_undef(v):
                    raise AnaRefused(f"{e.eid}: the {metric}@{k} value from its ranks is undefined")
                e.rows[(metric, k)] = {"metric": metric, "k": k, "value": float(v), "value_status": "OK",
                                       "source": "COMPUTED_FROM_RANKS", "seed": e.seed}
            e.rule_source = "RANKS"
        if e.historical and e.ranked:
            e.notes.append("historical reference: ranks ignored (point-only)")
            e.ranks = None
        if e.ties_rel:                                                # 06L-C §C3 / §C15: the per-row tie count
            tp = _inside(block, e.ties_rel)
            files[str(e.ties_rel)] = _sha_file(tp)
            e.ties = np.load(tp).astype(np.int64)
            check_row_ties(e, m)
    key = check_key(entries, ctx, idx, split)
    calib = load_calib_context(context_loader, ctx, split)
    return SimpleNamespace(block=block, idx=idx, entries=entries, ctx=ctx, files=files, key=key,
                           rank_row_max_abs_diff=worst, calib=calib)


def check_row_ties(e: Entry, manifest) -> None:
    """The per-row tie count of a BASELINE_CPU entry (v4base.run n_tied_int32.npy): one entry per manifest row, -1
    exactly where the rank is -1; for LAST (06L-C §C15 repeat / novel stratum) a tie count of 0 means the target is one
    of the window's distinct items, so its rank must be <= HIST_WINDOW (the recency order of <= 50 items)."""
    np = LIB.np
    if e.ties.shape != (manifest.n,):
        raise AnaRefused(f"{e.eid}: row_ties has {e.ties.shape[0]} entries, the manifest {manifest.n}")
    if not np.array_equal(e.ties < 0, e.ranks < 1):
        raise AnaRefused(f"{e.eid}: row_ties is -1 on other rows than the rank file's -1 (not the same artifact)")
    if e.arm in ANCHOR_ARMS:
        rep = (e.ranks > 0) & (e.ties == 0)
        if bool(np.any(e.ranks[rep] > HIST_WINDOW)):
            raise AnaRefused(f"{e.eid}: a row with tie count 0 has a LAST rank > {HIST_WINDOW}: the tie file is not "
                             "LAST's (the repeat / novel stratum needs LAST's own artifact)")


def load_calib_context(context_loader, ctx: Mapping[str, RoleCtx], split: str) -> dict:
    """06L-C §C14: MAIN \\ CALIB. The CALIB_VALIDATION users table (client_key_hex) through the same loader (role
    CALIBRATION); the MAIN E2E users whose client_key_hex is not a CALIB user form the CALIB-disjoint view. A loader
    that has no CALIBRATION context (returns None / KeyError) makes the diagnostic NOT_AVAILABLE; the production loader
    marks its context `enforce_registered_counts` and then the VALIDATION counts must be the registered ones."""
    np = LIB.np
    main = ctx.get("MAIN")
    if main is None or main.hex_by_euk is None:
        return {"status": "NOT_AVAILABLE", "reason": "no MAIN users table (client_key_hex)"}
    try:
        c = context_loader("CALIBRATION", {"eval_manifest_id": CALIB_EVAL_ID, "train_manifest_id": None})
    except KeyError:
        c = None
    if c is None:
        return {"status": "NOT_AVAILABLE", "reason": "the context loader has no CALIBRATION context"}
    if c.hex_by_euk is None:
        raise AnaRefused("CALIBRATION context without its users table (client_key_hex): MAIN \\ CALIB is undefined")
    cm = c.manifest
    calib_hex = {str(c.hex_by_euk[u]) for u in np.unique(cm.eval_user_key).tolist()}      # ALL CALIB manifest users
    mm = main.manifest
    m_users = np.unique(mm.eval_user_key)                                                  # ALL MAIN manifest users
    m_in = np.asarray([str(main.hex_by_euk[u]) in calib_hex for u in m_users.tolist()], dtype=bool)
    counts_manifest = {"main_users": int(m_users.size), "calib_users": int(m_in.sum()),
                       "disjoint_users": int(m_users.size - m_in.sum())}
    users = _euks(main)                                                                    # MAIN E2E users (>= 1 eval row)
    in_calib = np.asarray([str(main.hex_by_euk[u]) in calib_hex for u in users.tolist()], dtype=bool)
    c_e2e = np.unique(cm.eval_user_key[cm.eval_mask])
    counts_e2e = {"main_users": int(users.size), "calib_users": int(np.unique(
                      [str(c.hex_by_euk[u]) for u in c_e2e.tolist()]).size if c_e2e.size else 0),
                  "disjoint_users": int(users.size - in_calib.sum())}
    counts = {"manifest": counts_manifest, "e2e": counts_e2e}
    if split == "VALIDATION" and len(calib_hex) != counts_manifest["calib_users"]:   # at TEST: the same people, other rows
        raise AnaRefused(f"{len(calib_hex)} CALIB manifest users but {counts_manifest['calib_users']} of them among the "
                         "MAIN manifest users: CALIB is not a subset of MAIN")
    if c.record.get("enforce_registered_counts") and split == "VALIDATION":
        if counts_manifest != CALIB_REGISTERED_COUNTS:
            raise AnaRefused(f"MAIN / CALIB / disjoint manifest user counts {counts_manifest} are not the registered "
                             f"{CALIB_REGISTERED_COUNTS} (amendment 02 §2; G2 condition 5)")
        if {k: counts_e2e[k] for k in CALIB_E2E_REGISTERED} != CALIB_E2E_REGISTERED:
            raise AnaRefused(f"MAIN / CALIB E2E user counts {counts_e2e} are not the registered {CALIB_E2E_REGISTERED} "
                             "(DATA-06)")
    keys = users[~in_calib].astype(np.int64)
    return {"status": "COMPUTED", "counts": counts, "calib_eval_manifest_id": cm.manifest_id,
            "disjoint_keys": keys, "mapping": "CALIB_VALIDATION users -> MAIN by client_key_hex",
            "disjoint_users_sha256": _sha_bytes(np.ascontiguousarray(keys, dtype="<i8").tobytes())}


# ------------------------------------------------------------------------------------------------ the engine
class Engine:
    """Cached per-view plans (one shared W per user set, v4train.bootstrap_ci._plan) and per-entry statistics."""

    def __init__(self, ctx: Mapping[str, RoleCtx]):
        self.ctx = ctx
        self._views, self._stats, self._res = {}, {}, {}
        self.row_masks = {}                  # (role, name) -> bool per manifest row (06L-C C15 target strata)
        self.boot = {}                       # name -> the seed-mean resample array of a contrast (Holm band, C13)

    def view(self, vkey):
        """vkey = (role, subset_name, subset_keys_tuple_or_None); a registered row mask (role, name) restricts the rows
        instead (keys None). Returns (manifest view, plan) or None if empty."""
        if vkey not in self._views:
            role, name, keys = vkey
            m = self.ctx[role].manifest
            if keys is None and (role, name) in self.row_masks:
                mv = dataclasses.replace(m, eval_mask=m.eval_mask & self.row_masks[(role, name)])
            else:
                mv = m if keys is None else _subset_view(m, LIB.np.asarray(keys, dtype=LIB.np.int64))
            if not bool(LIB.np.any(mv.eval_mask)):
                self._views[vkey] = None
            else:
                self._views[vkey] = (mv, LIB.BC._plan(mv))
        return self._views[vkey]

    def stats(self, e: Entry, vkey, metric):
        key = (e.eid, vkey, metric)
        if key not in self._stats:
            mv, _plan = self.view(vkey)
            if metric == PRIMARY:
                st = LIB.BC._stats(mv, e.ranks, e.eid)                       # the T1d statistic (macro-E2E MRR@20)
            else:
                credit = LIB.M.credits_from_ranks(e.ranks, (metric[1],))[f"{metric[0]}@{metric[1]}"]
                st = LIB.B.user_stats(credit, mv.eval_mask, mv.rankable, mv.eval_user_key, e.eid)
            self._stats[key] = st
        return self._stats[key]

    def res(self, e: Entry, vkey, metric):
        key = (e.eid, vkey, metric)
        if key not in self._res:
            _mv, plan = self.view(vkey)
            self._res[key] = LIB.B.resampled(plan, self.stats(e, vkey, metric), "macro_e2e")
        return self._res[key]

    def point(self, e: Entry, vkey, metric):
        if self.view(vkey) is None:
            return undef("EMPTY_POPULATION")
        if e.ranked:
            return LIB.B.point(self.stats(e, vkey, metric), "macro_e2e")
        if vkey[1] == "ALL":
            return float(e.rows[metric]["value"])
        return undef("NO_RANKS")

    def meta(self, vkey) -> dict:
        v = self.view(vkey)
        if v is None:
            return {"n_users": 0, "n_decisions": 0, "W_sha256": None, "population": f"{vkey[0]}/{vkey[1]}"}
        return {"n_users": v[1].n_users, "n_decisions": int(LIB.np.count_nonzero(v[0].eval_mask)),   # E7.4 (A-F15)
                "W_sha256": v[1].W_sha256, "population": f"{vkey[0]}/{vkey[1]}"}

    def drop(self, role: str, prefix: str) -> None:
        """Free the sub-population views of one family (each W is 1,000 x n_users int32)."""
        gone = {k for k in self._views if k[0] == role and str(k[1]).startswith(prefix)}
        for k in gone:
            del self._views[k]
        for cache in (self._stats, self._res):
            for k in [k for k in cache if k[1] in gone]:
                del cache[k]


def full_view(role: str, keys=None):
    if keys is None:
        return (role, "ALL", None)
    return (role, "SUBSET:" + _sha_bytes(LIB.np.ascontiguousarray(keys, dtype="<i8").tobytes())[:16],
            tuple(int(x) for x in keys))


def require_nonzero_denominator(value, what: str) -> None:
    """E8.3 at block level: a full-population denominator must be > 0 (C < 0 is impossible for these credits)."""
    if is_undef(value):
        raise AnaRefused(f"{what}: the denominator is UNDEFINED ({value})")
    if float(value) < 0:
        raise AnaRefused(f"{what}: a negative denominator is impossible (credits >= 0)")
    if float(value) == 0.0:
        raise AnaRefused(f"{what}: zero denominator (E8.3) — retention undefined; refusing the block")


def pair_seed(num: Entry, den: Entry, shared: bool = False) -> Optional[int]:
    """The seed a (numerator, denominator) pair reports: the denominator's, except that a SHARED_SEED_ARMS denominator
    (PRE: one theta_pre serves both FA_WARM seeds; 06K / DL 206 (2)), a contrast-scoped shared denominator (`shared`:
    K20b's one C_FULL seed, 06L-C §C7) and a seed-free anchor (LAST) report the numerator's seed."""
    if shared or den.arm in SHARED_SEED_ARMS:
        return num.seed
    return den.seed if den.seed is not None else num.seed


def is_anchor(e: Entry) -> bool:
    """06L-C §C3: LAST with a per-row rank file in the BASELINE_CPU sub-block (seed-free, RULE_FREE)."""
    return e.arm in ANCHOR_ARMS and e.block == BASELINE_BLOCK and e.rule == RULE_FREE and e.ranks is not None


def check_pairs(pairs: Sequence[Tuple[Entry, Entry]], shared: bool = False) -> None:
    """06A §7: the numerator and the denominator use the same checkpoint rule and the same seed (a RULE_FREE /
    seed-free numerator pairs with any; a seed-free anchor denominator — LAST, 06L-C §C3 — pairs with any rule and seed;
    a `shared` denominator — K20b's one C_FULL seed, 06L-C §C7 — pairs with every numerator seed), the same role and
    the same user subset."""
    seeds = []
    for num, den in pairs:
        if den.rule == RULE_FREE and num.rule != RULE_FREE and not is_anchor(den):
            raise AnaRefused(f"{num.eid} / {den.eid}: a rule-free denominator cannot pair with a checkpoint rule")
        if num.rule != RULE_FREE and den.rule != RULE_FREE and num.rule != den.rule:
            raise AnaRefused(f"{num.eid} / {den.eid}: RULE_MISMATCH (06A §7: same checkpoint rule)")
        if (num.seed is not None and den.seed is not None and num.seed != den.seed
                and den.arm not in SHARED_SEED_ARMS and not shared):
            raise AnaRefused(f"{num.eid} / {den.eid}: SEED_MISMATCH (06A §7: same seed)")
        if num.role != den.role:
            raise AnaRefused(f"{num.eid} / {den.eid}: different roles")
        if den.subset_keys is not None and (num.subset_keys is None or
                                            not LIB.np.array_equal(num.subset_keys, den.subset_keys)):
            raise AnaRefused(f"{num.eid} / {den.eid}: the denominator is on another user subset")
        seeds.append(pair_seed(num, den, shared))
    if len(set(seeds)) != len(seeds):
        raise AnaRefused("a seed appears twice in one paired contrast")


def _comparison_label(pairs) -> dict:
    """A-F2: an ONNX-CPU row over an A40 row mixes runtime numerics (DEVICE_MIXED, DESCRIPTIVE only); a BASELINE_CPU
    rank file (06L-C §C3) over an A40 row pairs CPU-scored exact baseline ranks of the same rows (labelled, allowed)."""
    blocks = sorted({e.block for p in pairs for e in p})
    if ONNX_BLOCK in blocks and len(blocks) > 1:
        return {"comparison": "DEVICE_MIXED", "comparison_note": "numerator and denominator come from different "
                "comparison blocks (ONNX_CPU vs the A40 block): DESCRIPTIVE only"}
    if BASELINE_BLOCK in blocks and len(blocks) > 1:
        return {"comparison": "BASELINE_CPU_EXACT_RANKS", "comparison_note": "a BASELINE_CPU rank file (CPU-scored, the "
                "frozen T3 path, the same manifest rows and key except eval_device_class / precision_id / objective_id) "
                "paired with A40 rows on the same W (06L-C §C3)"}
    return {}


def _quantiles(du) -> dict:
    np = LIB.np
    q = np.percentile(du, [10, 25, 50, 75, 90])
    k = max(1, int(math.ceil(0.1 * du.size)))
    return {"per_user_delta_quantiles": {f"P{p}": float(v) for p, v in zip((10, 25, 50, 75, 90), q)},
            "worst_decile_mean_delta": float(np.mean(np.sort(du)[:k]))}


def _harm(du) -> dict:
    np = LIB.np
    return {"harmed_share": float(np.mean(du < 0)), "improved_share": float(np.mean(du > 0)),
            "tied_share": float(np.mean(du == 0))}


def _seed_stats(per: Sequence[Mapping], boot, point, ci) -> dict:
    """06L-C C13 (review 13 [12-E4]): per-seed points, the seed spread (max - min), SEED_SENSITIVE (spread > half the
    user-bootstrap CI width of the seed mean) and a DESCRIPTIVE user + seed interval (point ± 1.96 sqrt(var_boot +
    s²_seed / n_seeds))."""
    np = LIB.np
    pts = [p.get("point") for p in per]
    out = {"per_seed_points": {str(p["seed"]): p.get("point") for p in per}, "label": "DESCRIPTIVE",
           "rule": "SEED_SENSITIVE = (max - min of the per-seed points) > half the width of the seed-mean 95% CI"}
    if len(pts) < 2 or any(x is None or is_undef(x) for x in pts):
        return dict(out, status="SINGLE_SEED" if len(pts) < 2 else "MISSING_EVIDENCE", seed_sensitive=None)
    spread = float(max(pts) - min(pts))
    out.update(status="COMPUTED", seed_spread=spread)
    if isinstance(ci, list) and boot is not None:
        half = (ci[1] - ci[0]) / 2.0
        s2 = float(np.var(np.asarray(pts, dtype=np.float64), ddof=1))
        vb = float(np.var(boot, ddof=1))
        se = math.sqrt(vb + s2 / len(pts))
        out.update(user_ci_half_width=half, seed_sensitive=bool(spread > half), seed_variance=s2,
                   user_seed_interval=[float(point) - 1.96 * se, float(point) + 1.96 * se])
    else:
        out.update(seed_sensitive=None, note="no user-bootstrap CI (NO_RANKS / UNDEFINED)")
    return out


def ratio(eng: Engine, pairs, vkey, metric, *, strict: bool, name: str = "", shared: bool = False,
          boot_key=None) -> dict:
    """R = 100 M / C per seed (v4eval.quality.retained_percent), bootstrapped directly on the shared W (E7.5), and the
    seed mean (point = mean of the per-seed R; CI from the per-resample mean of the per-seed ratios). ANA-04 (06L-C
    C13 / C15): the harmed-client share (per-user macro M_u - C_u, seed mean, < 0) with per-user Δ quantiles, and the
    per-seed statistics (seed spread, SEED_SENSITIVE, user + seed interval); `boot_key` keeps the seed-mean resamples
    in eng.boot (the Holm band)."""
    np, B = LIB.np, LIB.B
    check_pairs(pairs, shared)
    per, arrays, du_all, ranked_all = [], [], [], True
    for num, den in pairs:
        pm, pc = eng.point(num, vkey, metric), eng.point(den, vkey, metric)
        seed = pair_seed(num, den, shared)
        row = {"seed": seed, "num": num.eid, "den": den.eid, "value_num": pm, "value_den": pc}
        if strict:
            require_nonzero_denominator(pc, f"{name} {den.eid}")
        if is_undef(pm) or is_undef(pc):
            row.update(point=undef("MISSING_EVIDENCE"), ci95=undef("MISSING_EVIDENCE"))
            ranked_all = False
            arrays.append(None)
        else:
            row["point"] = LIB.Q.retained_percent(pm, pc)
            if num.ranked and den.ranked:
                rm, rc = eng.res(num, vkey, metric), eng.res(den, vkey, metric)
                sm, sc = eng.stats(num, vkey, metric), eng.stats(den, vkey, metric)
                if np.array_equal(sm.user_keys, sc.user_keys):
                    du = sm.S / sm.N - sc.S / sc.N
                    du_all.append(du)
                    row.update(_harm(du))
                zero = int(np.count_nonzero(rc == 0))
                row["n_resamples_zero_denominator"] = zero
                if zero:
                    row["ci95"] = undef("ZERO_DENOMINATOR_IN_RESAMPLES")
                    arrays.append(None)
                else:
                    vals = 100.0 * rm / rc
                    row["ci95"] = B.ci95(vals)
                    arrays.append(vals)
            else:
                row["ci95"] = undef("NO_RANKS")
                ranked_all = False
                arrays.append(None)
        per.append(row)
    pts = [r["point"] for r in per]
    out = {"kind": "retained", "metric": f"{metric[0]}@{metric[1]}", "statistic": "macro_e2e",
           "seeds": [r["seed"] for r in per], "n_seeds": len(per), "per_seed": per}
    out.update(_comparison_label(pairs))
    out.update(eng.meta(vkey))
    if not per or any(is_undef(p) for p in pts):
        out.update(point=undef("MISSING_EVIDENCE"), ci95=undef("MISSING_EVIDENCE"))
        return out
    out["point"] = float(np.mean(pts))
    boot = None
    if ranked_all and all(a is not None for a in arrays):
        boot = np.mean(arrays, axis=0)
        out["ci95"] = B.ci95(boot)
        if boot_key is not None:
            eng.boot[boot_key] = boot
    else:
        out["ci95"] = next((r["ci95"] for r in per if is_undef(r["ci95"])), undef("NO_RANKS"))
    if du_all and len(du_all) == len(per):
        mdu = np.mean(du_all, axis=0)
        out.update(_harm(mdu), **_quantiles(mdu),
                   harmed_rule="per-user macro difference numerator - denominator (seed mean) < 0, over the paired "
                               "user set (06L-C C15)")
    out["seed_stats"] = _seed_stats(per, boot, out["point"], out["ci95"])
    return out


def delta(eng: Engine, pairs, vkey, metric, name: str = "", shared: bool = False, boot_key=None) -> dict:
    """a - b paired on the same W (v4eval.bootstrap.contrast / seed_contrast arithmetic) + the harmed-client share
    (users whose per-user macro delta, seed-mean, is < 0) with per-user Δ quantiles and the per-seed statistics
    (06L-C C13 / C15)."""
    np, B = LIB.np, LIB.B
    check_pairs(pairs, shared)
    per, diffs, du_all = [], [], []
    for a, b in pairs:
        seed = pair_seed(a, b, shared)
        if not (a.ranked and b.ranked) or eng.view(vkey) is None:
            per.append({"seed": seed, "a": a.eid, "b": b.eid, "point": undef("NO_RANKS"), "ci95": undef("NO_RANKS")})
            continue
        sa, sb = eng.stats(a, vkey, metric), eng.stats(b, vkey, metric)
        if not np.array_equal(sa.user_keys, sb.user_keys):
            raise AnaRefused(f"{name}: {a.eid} and {b.eid} cover different users")
        pa, pb = B.point(sa, "macro_e2e"), B.point(sb, "macro_e2e")
        d = eng.res(a, vkey, metric) - eng.res(b, vkey, metric)
        du = sa.S / sa.N - sb.S / sb.N
        diffs.append(d)
        du_all.append(du)
        per.append({"seed": seed, "a": a.eid, "b": b.eid, "value_a": pa, "value_b": pb, "point": pa - pb,
                    "ci95": B.ci95(d), **_harm(du)})
    out = {"kind": "delta", "metric": f"{metric[0]}@{metric[1]}", "statistic": "macro_e2e",
           "seeds": [r["seed"] for r in per], "n_seeds": len(per), "per_seed": per}
    out.update(_comparison_label(pairs))
    out.update(eng.meta(vkey))
    if not per or len(diffs) != len(per):
        out.update(point=undef("NO_RANKS"), ci95=undef("NO_RANKS"), harmed_share=undef("NO_RANKS"))
        return out
    mdu = np.mean(du_all, axis=0)
    boot = np.mean(diffs, axis=0)
    if boot_key is not None:
        eng.boot[boot_key] = boot
    out.update(point=float(np.mean([r["point"] for r in per])), ci95=B.ci95(boot), **_harm(mdu), **_quantiles(mdu),
               harmed_rule="per-user macro delta (seed mean) < 0, over the paired user set")
    out["seed_stats"] = _seed_stats(per, boot, out["point"], out["ci95"])
    return out


def seed_null(eng: Engine, ents: Mapping, vkey, metric=PRIMARY) -> dict:
    """06L-C C15 (review 13 [12-E6], 10 [9-R6]): the same-arm seed-vs-seed null of the harmed share — the share of
    users whose per-user macro value of seed 2027 is below seed 2026 of the SAME arm on the same users (a noise floor
    for the harmed-client share)."""
    np = LIB.np
    a, b = ents.get(2027), ents.get(2026)
    if a is None or b is None or not (a.ranked and b.ranked) or a.subset_keys is not None or eng.view(vkey) is None:
        return {"status": "NOT_AVAILABLE", "reason": "needs both seeds (2026, 2027) of the arm with rank files"}
    sa, sb = eng.stats(a, vkey, metric), eng.stats(b, vkey, metric)
    if not np.array_equal(sa.user_keys, sb.user_keys):
        return {"status": "NOT_AVAILABLE", "reason": "the two seeds cover different users"}
    du = sa.S / sa.N - sb.S / sb.N
    return {"arm": a.arm, "rule": a.rule, "seeds": [2027, 2026], **_harm(du), "label": "DESCRIPTIVE",
            "rule_text": "per-user macro (s2027 - s2026) of the same arm on the same users < 0 (a noise floor)"}


# ------------------------------------------------------------------------------------------------ pair building
class Book:
    """Entries by (role, arm, rule) -> {seed: entry} (non-historical) + the historical ones."""

    def __init__(self, entries: Sequence[Entry]):
        self.by = {}
        self.hist = [e for e in entries if e.historical]
        for e in entries:
            if not e.historical:
                self.by.setdefault((e.role, e.arm, e.rule), {})[e.seed] = e

    def get(self, role, arm, rule) -> dict:
        """{seed: entry} for `rule`, or the RULE_FREE entries of the arm (seed-free keyed None)."""
        return dict(self.by.get((role, arm, rule)) or self.by.get((role, arm, RULE_FREE)) or {})

    def arms(self, role) -> list:
        present = {a for (r, a, _rule) in self.by if r == role}
        return [a for a in ARMS if a in present]

    def pairs(self, role, num_arm, den_arm, rule, shared: bool = False) -> list:
        nums, dens = self.get(role, num_arm, rule), self.get(role, den_arm, rule)
        anchors = [d for d in dens.values() if is_anchor(d)]
        if anchors and any(n.rule != RULE_FREE for n in nums.values()):   # 06L-C §C3: LAST pairs with every seed
            return [(n, anchors[0]) for _s, n in sorted(nums.items(), key=lambda kv: kv[0] or 0)]
        if any(d.rule == RULE_FREE for d in dens.values()) and any(n.rule != RULE_FREE for n in nums.values()):
            return []                                      # A-F3: a rule-free denominator -> NOT_AVAILABLE, no refusal
        if shared:                                         # 06L-C §C7 (K20b): the one registered denominator seed
            den = dens.get(2026)
            return [] if den is None else [(n, den) for s, n in sorted(nums.items(), key=lambda kv: kv[0] or 0)
                                           if s is not None]
        out = []
        if den_arm in SHARED_SEED_ARMS:                    # one theta_pre pairs with every numerator seed (06K)
            den = dens.get(2026) or next((d for _s, d in sorted(dens.items(), key=lambda kv: kv[0] or 0)), None)
            return [(n, den) for _s, n in sorted(nums.items(), key=lambda kv: kv[0] or 0)
                    if den is not None and n.rule != RULE_FREE]
        for seed in sorted(s for s in dens if s is not None):
            n = nums.get(seed, nums.get(None))
            if n is not None:
                out.append((n, dens[seed]))
        if not out and None in dens:                       # rule-free denominator (baseline vs baseline)
            n = nums.get(None)
            if n is not None:
                out.append((n, dens[None]))
        return out


def _vkey_for(pairs, role, keys_override=None):
    keys = next((n.subset_keys for n, _d in pairs if n.subset_keys is not None), None)
    if keys_override is not None:                          # 06L-C C14: MAIN \ CALIB (never on a users_subset entry)
        if keys is not None:
            return None
        return (role, "CALIB_DISJOINT", tuple(int(x) for x in keys_override))
    return full_view(role, keys)


# ------------------------------------------------------------------------------------------------ contrasts input
def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).replace("−", "-").replace("–", "-")).strip()


def _contrast_items(obj) -> list:
    if isinstance(obj, Mapping):
        for key in ("confirmatory_contrasts", "contrasts"):
            if key in obj:
                return _contrast_items(obj[key])
        raise AnaRefused("--contrasts: no 'confirmatory_contrasts' / 'contrasts' list")
    if not isinstance(obj, list):
        raise AnaRefused("--contrasts: the contrasts are not a list")
    items = []
    for it in obj:
        if isinstance(it, str):
            cells = [c.strip() for c in it.strip().strip("|").split("|")]
            if len(cells) != 3:
                raise AnaRefused(f"--contrasts: row {it!r} is not '| id | contrast | question |'")
            items.append({"id": cells[0], "contrast": cells[1], "question": cells[2]})
        elif isinstance(it, Mapping):
            items.append({"id": it.get("id"), "contrast": it.get("contrast"), "question": it.get("question"),
                          **{k: it[k] for k in ("num", "den", "kind") if k in it}})
        else:
            raise AnaRefused("--contrasts: an item is neither a table-row string nor an object")
    return items


def parse_contrast(text: str) -> dict:
    t = _norm_text(text)
    m = re.fullmatch(r"R\((\S+) / (\S+)\)(?: for (\S+))?", t)
    if m:
        num, den, base = m.group(1), m.group(2), m.group(3)
        num, den = CONTRAST_ALIASES.get(num, num), CONTRAST_ALIASES.get(den, den)
        kind = "retained"
    else:
        m = re.fullmatch(r"(\S+) - (\S+)", t)
        if not m:
            raise AnaRefused(f"--contrasts: cannot parse {text!r}")
        num, den, base, kind = m.group(1), m.group(2), None, "delta"
    for a in (num, den):
        if a not in ARMS:
            raise AnaRefused(f"--contrasts: {text!r} names an unknown arm {a!r}")
    return {"kind": kind, "num": num, "den": den, "for": base}


def registered_items() -> dict:
    """{id: (contrast, question)} of every registered contrast (06B §2, 06I §1, 06K, 06L-C §C3 / §C7) + the registered
    DESCRIPTIVE S8-S7 (06L-C §C8)."""
    out = {k: (c, q) for k, c, q in REGISTERED_CONTRASTS}
    out.update({k: (c, q) for k, c, q in REGISTERED_CONTRASTS_06LC})
    out.update({k: (c, q) for k, c, q in REGISTERED_DESCRIPTIVE_06LC})
    return out


def v3_registry_content() -> dict:
    """The structured fields a PPSI_V4_ANA_CONTRASTS_3 file must carry verbatim (06L-C §C4): populations, pairing
    notes, reading rules, the algebraic dependency list and the registered DESCRIPTIVE contrast(s)."""
    reg = registered_items()
    return {"populations": {k: CONTRAST_POPULATION.get(k, "MAIN") for k in V3_IDS + V3_DESCRIPTIVE_IDS},
            "pairing_notes": dict(PAIRING_NOTES), "reading_rules": json.loads(json.dumps(READING_RULES)),
            "algebraic_dependencies": json.loads(json.dumps(list(ALGEBRAIC_DEPENDENCIES))),
            "registered_descriptive_contrasts": [{"id": k, "contrast": reg[k][0], "question": reg[k][1]}
                                                 for k in V3_DESCRIPTIVE_IDS]}


def load_contrasts(path: Path, *, split: str = "VALIDATION", allow_v1: bool = False, allow_draft: bool = False) -> dict:
    """The confirmatory registry. Format PPSI_V4_ANA_CONTRASTS_2 (K1-K19: 06B §2 verbatim + 06I §1) or
    PPSI_V4_ANA_CONTRASTS_3 (06L-C §C4: K1-K21 + K9b, K9c, K11b, K20b, K20c = 26 confirmatory, + the registered
    DESCRIPTIVE S8-S7; populations, pairing notes, the K14 reading rule and the algebraic dependency list verbatim);
    `confirmatory_rule_map` == CONFIRMATORY_RULE (06D §3; the map itself is checked against the embedded 06D §3
    sentence), `registered_seeds` == REGISTERED_SEEDS of its ids and `hierarchy` == HIERARCHY of its ids (06D §4, 06I
    §1, 06K, 06L-C). The v1 file (PPSI_V4_ANA_CONTRASTS_1, K1-K16) is accepted only for `--split VALIDATION
    --allow-v1-contrasts` (a dry check), never for a TEST run. Any deviation of an id, a text or a field refuses."""
    raw = path.read_bytes()
    obj = json.loads(raw.decode("utf-8"))
    if rule_map_from_06d() != CONFIRMATORY_RULE or hierarchy_from_06d() != HIERARCHY_06D:
        raise AnaRefused("the script's CONFIRMATORY_RULE / HIERARCHY differ from its embedded 06D §3 / §4 text")
    if not isinstance(obj, Mapping):
        raise AnaRefused("--contrasts: the file is not a JSON object")
    fmt = obj.get("format")
    if fmt == CONTRAST_FORMAT_V4:                                       # ANA-06: the 06P registry (ANA_CONTRASTS_v4)
        return load_contrasts_v4(obj, raw, split=split, allow_draft=allow_draft)
    if fmt not in CONTRAST_COUNT:
        raise AnaRefused(f"--contrasts: format {fmt!r} is not one of {sorted(CONTRAST_COUNT)}")
    if fmt == CONTRAST_FORMAT_V1 and not (allow_v1 and split == "VALIDATION"):
        raise AnaRefused(f"--contrasts: {CONTRAST_FORMAT_V1} (K1-K16) is superseded by {CONTRAST_FORMAT_V2} (K1-K19; "
                         "06I §1); it is accepted only with --split VALIDATION --allow-v1-contrasts (a dry check)")
    if split == "TEST" and fmt != CONTRAST_FORMAT_V3:
        raise AnaRefused(f"--contrasts: format {fmt} is superseded by {CONTRAST_FORMAT_V3} (06L-C §C4): a TEST run uses "
                         "the v3 registry only (a v2 file would silently turn K9b, K9c, K11b, K20b, K20c DESCRIPTIVE and "
                         "shrink the Holm family to 19; REV-DELTA-03 M5); v2 stays a VALIDATION dry check")
    n_reg = CONTRAST_COUNT[fmt]
    all_ids = list(CONTRAST_IDS[fmt])
    rmap = obj.get("confirmatory_rule_map")
    if rmap != CONFIRMATORY_RULE:
        raise AnaRefused(f"--contrasts: confirmatory_rule_map {rmap!r} is not the registered 06D §3 map "
                         f"{CONFIRMATORY_RULE} (ALL_METHODS_LOCK / the pre-lock contrasts JSON must carry it)")
    if obj.get("default_confirmatory_rule", DEFAULT_CONFIRMATORY_RULE) != DEFAULT_CONFIRMATORY_RULE:
        raise AnaRefused(f"--contrasts: default_confirmatory_rule is not {DEFAULT_CONFIRMATORY_RULE}")
    hier = obj.get("hierarchy")
    if hier is not None or fmt != CONTRAST_FORMAT_V1:
        want = {"primary": sorted(k for k in all_ids if HIERARCHY.get(k) == "primary"),
                "secondary": sorted(k for k in all_ids if HIERARCHY.get(k) == "secondary"),
                "tertiary": sorted(k for k in all_ids if k not in HIERARCHY)}
        if not isinstance(hier, Mapping) or {k: sorted(v) for k, v in hier.items()} != want:
            raise AnaRefused(f"--contrasts: hierarchy {hier!r} is not the registered hierarchy {want} (06D §4; 06I §1; "
                             "06K; 06L-C)")
    if fmt != CONTRAST_FORMAT_V1:
        want_seeds = {k: list(v) for k, v in REGISTERED_SEEDS.items() if k in all_ids}
        got_seeds = obj.get("registered_seeds")
        if not isinstance(got_seeds, Mapping) or {k: list(v) for k, v in got_seeds.items()} != want_seeds:
            raise AnaRefused(f"--contrasts: registered_seeds {got_seeds!r} is not the registered {want_seeds} "
                             "(06E §2; 06I §1; 06L-C §C7)")
    if fmt == CONTRAST_FORMAT_V3:
        for key, want_v in v3_registry_content().items():
            if json.loads(json.dumps(obj.get(key))) != want_v:
                raise AnaRefused(f"--contrasts: {key} is not the registered 06L-C §C4 content")
        if "co-headline" in raw.decode("utf-8").lower():
            raise AnaRefused("--contrasts: the v3 registry carries the word 'co-headline' (06L-C §C4 removes it)")
    items = _contrast_items(obj)
    reg = registered_items()
    want = {k: reg[k] for k in all_ids}
    ids = [str(i["id"]) for i in items]
    if sorted(ids) != sorted(want) or len(set(ids)) != len(ids):
        raise AnaRefused(f"--contrasts: ids {ids} are not exactly {ids_label(all_ids)} (format {fmt})")
    out = []
    for it in sorted(items, key=lambda x: order_key(str(x["id"]))):
        c, q = want[it["id"]]
        if _norm_text(it["contrast"]) != _norm_text(c) or _norm_text(it["question"]) != _norm_text(q):
            raise AnaRefused(f"--contrasts: {it['id']} differs from the registered text "
                             f"({it['contrast']!r} / {it['question']!r})")
        p = parse_contrast(it["contrast"])
        for k in ("num", "den", "kind"):
            if k in it and it[k] != p[k]:
                raise AnaRefused(f"--contrasts: {it['id']} explicit {k} {it[k]!r} != parsed {p[k]!r}")
        out.append(dict(id=it["id"], contrast=it["contrast"], question=it["question"],
                        population=CONTRAST_POPULATION.get(it["id"], "MAIN"), **p))
    desc = []
    if fmt == CONTRAST_FORMAT_V3:
        for k in V3_DESCRIPTIVE_IDS:
            c, q = reg[k]
            desc.append(dict(id=k, contrast=c, question=q, population=CONTRAST_POPULATION.get(k, "MAIN"),
                             **parse_contrast(c)))
    got_ids = [c["id"] for c in out]
    if fmt != CONTRAST_FORMAT_V1 and obj.get("multiplicity_statement") is not None \
            and not str(obj["multiplicity_statement"]).startswith(f"{n_reg} pre-registered contrasts"):
        raise AnaRefused(f"--contrasts: multiplicity_statement does not open with '{n_reg} pre-registered contrasts'")
    content = {"items": [{k: c[k] for k in ("id", "contrast", "question")} for c in out],
               "confirmatory_rule_map": CONFIRMATORY_RULE, "default_confirmatory_rule": DEFAULT_CONFIRMATORY_RULE,
               "hierarchy": {k: HIERARCHY.get(k, DEFAULT_HIERARCHY) for k in got_ids},
               "registered_seeds": {k: list(v) for k, v in sorted(REGISTERED_SEEDS.items()) if k in got_ids}}
    if fmt == CONTRAST_FORMAT_V3:
        content.update(v3_registry_content())
    return {"file_sha256": _sha_bytes(raw), "items": out, "descriptive_items": desc,
            "confirmatory_rule_map": dict(CONFIRMATORY_RULE), "format": fmt, "n_contrasts": len(out),
            "multiplicity": multiplicity_line(got_ids), "v1_dry_check": fmt == CONTRAST_FORMAT_V1,
            "superseded_note": (None if fmt in (CONTRAST_FORMAT_V3, CONTRAST_FORMAT_V1) else
                                f"{fmt} is superseded by {CONTRAST_FORMAT_V3} (06L-C §C4): the rows it does not carry "
                                "(K20, K21, K9b, K9c, K11b, K20b, K20c, S8-S7) are computed and labelled DESCRIPTIVE"),
            "content_sha256": _sha_bytes(_canon_json(content)), "self_sha256_status": _self_ok(obj)}


# ------------------------------------------------------------------------------------------------ the tables
def _meta_row(e: Entry) -> dict:
    keys = ("param_count", "model_bytes", "bytes_per_visit", "bytes_up_per_visit", "bytes_down_per_visit",
            "bytes_per_client", "bytes_per_client_charged_dropped", "bytes_per_client_charged_dropped_up_down",
            "dropped_visits", "charged_bytes_source", "bytes_fleet", "epsilon", "dp_z", "rounds", "consumed_efe",
            "best_mark", "still_improving_at_6_0")
    out = {k: e.meta(k) for k in keys}
    out["still_improving_at_6.0"] = out.pop("still_improving_at_6_0")        # 06D §5 (DESCRIPTIVE)
    if out["epsilon"] is None and e.arm in EPS_DEFAULT:
        out["epsilon"] = EPS_DEFAULT[e.arm]
    r = e.rows.get(PRIMARY) or {}
    out["efe_at_checkpoint"] = r.get("efe")
    out["checkpoint_id"] = r.get("checkpoint_id")
    out["comparison_block"] = e.block
    for k in ("users_subset_n", "users_subset_set_sha256", "bench_tag", "onnx_runtime", "bytes_rule",
              "bytes_per_client_excl_frozen_once", "frozen_bytes_once", "frozen_once_per_client_mean"):
        if e.meta_auto.get(k) is not None:
            out[k] = e.meta_auto[k]
    conflicts = {k: {"declared": e.meta_declared[k], "auto": e.meta_auto[k]} for k in e.meta_declared
                 if k in e.meta_auto and e.meta_auto[k] is not None and e.meta_declared[k] != e.meta_auto[k]}
    if conflicts:
        out["meta_conflicts_declared_wins"] = conflicts
    return out


def still_improving_flags(book: Book, role: str, arm: str) -> dict:
    """06D §5: {seed: still_improving_at_6.0} from every entry of the arm with FL marks (either rule)."""
    out = {}
    for (r, a, _rule), ents in book.by.items():
        if r == role and a == arm:
            for seed, e in ents.items():
                if e.meta_auto.get("still_improving_at_6_0") is not None:
                    out[str(seed)] = bool(e.meta_auto["still_improving_at_6_0"])
    return dict(sorted(out.items()))


def retention_table(eng: Engine, book: Book, role: str = "MAIN") -> dict:
    """arm x seed x rule: values, R vs C_BIG and vs C_SMALL (paired, CI), 2-seed mean; NDCG@10 / HR@10."""
    out = {}
    for rule in RULES:
        rows = []
        for arm in book.arms(role):
            ents = book.get(role, arm, rule)
            if not ents:
                continue
            rec = {"arm": arm, "rule": rule, "label": "DESCRIPTIVE", "entries": {}}
            for seed, e in sorted(ents.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
                rec["entries"][str(seed)] = {"entry": e.eid, "rule_source": e.rule_source, "meta": _meta_row(e),
                                             "values": {f"{m}@{k}": eng.point(e, full_view(role, e.subset_keys), (m, k))
                                                        for m, k in METRICS}}
            dsd = sorted(s for s in ents if s in DESCRIPTIVE_SEEDS.get(arm, ()))
            if dsd:                                                    # ANA-05: C_FULL s2027 (06E §2)
                rec["descriptive_seeds"] = {"seeds": dsd, "rule": DESCRIPTIVE_SEED_RULE}
            flags = still_improving_flags(book, role, arm)
            if flags:
                rec["still_improving_at_6.0"] = {"per_seed": flags, "any_seed": any(flags.values()),
                                                 "rule": "PRACTICAL_BEST mark >= 5.5 (06D §5)", "label": "DESCRIPTIVE"}
            for ref, tag in ((REF, "vs_C_BIG"), (REF_SMALL, "vs_C_SMALL")):
                pairs = book.pairs(role, arm, ref, rule)
                if not pairs:
                    rec[tag] = {"status": "NOT_AVAILABLE"}
                    continue
                vk = _vkey_for(pairs, role)
                metrics = METRICS if tag == "vs_C_BIG" else (PRIMARY,)
                rec[tag] = {f"{m}@{k}": ratio(eng, pairs, vk, (m, k), strict=(vk[1] == "ALL"),
                                              name=f"{arm}/{ref} {rule}") for m, k in metrics}
            rows.append(rec)
        out[rule] = rows
    return out


def _pair_seed(p, shared: bool = False) -> Optional[int]:
    return pair_seed(p[0], p[1], shared)


def _contrast_stat(eng: Engine, c: Mapping, pairs, name: str, *, shared: bool = False, boot_key=None,
                   keys_override=None) -> dict:
    """One contrast statistic on the contrast's population (MAIN; K9b: UNSEEN_HOLDOUT, 06L-C §C7), on the numerator's
    users_subset if it has one, or on MAIN \\ CALIB (`keys_override`, 06L-C §C14; strict denominators off there)."""
    role = c.get("population") or "MAIN"
    vk = _vkey_for(pairs, role, keys_override)
    if vk is None:
        return {"status": "NOT_AVAILABLE", "reason": "a users_subset (ONNX) entry has no CALIB-disjoint view"}
    if eng.view(vk) is None:
        return {"status": "NOT_AVAILABLE", "reason": f"empty population {vk[0]}/{vk[1]}"}
    st = (ratio(eng, pairs, vk, PRIMARY, strict=keys_override is None, name=name, shared=shared, boot_key=boot_key)
          if c["kind"] == "retained" else delta(eng, pairs, vk, PRIMARY, name=name, shared=shared, boot_key=boot_key))
    if vk[1] not in ("ALL", "CALIB_DISJOINT"):
        st["population_note"] = f"users_subset of {c['num']} ({vk[1]})"
    return st


def confirmatory(eng: Engine, book: Book, contrasts: Mapping) -> list:
    """The loaded registry's contrasts (K1-K19 with the v2 file; K1-K21 + K9b, K9c, K11b, K20b, K20c with the v3 file,
    06L-C §C4). Each K's CONFIRMATORY row uses its registered rule (CONFIRMATORY_RULE, 06D §3) on BOTH sides and exactly
    its registered seeds (REGISTERED_SEEDS, 06E §2; 06I §1; 06L-C §C7); the other rule is DESCRIPTIVE (K12/K13
    PRACTICAL_BEST with the 06D §3 disclosure); extra seeds -> DESCRIPTIVE `extra_seeds`; a missing registered seed ->
    INCOMPLETE_SEEDS beside the value. K16 is CONFIRMATORY only inside the ONNX_CPU sub-block; the A40 C_SMALL fallback
    is DESCRIPTIVE, DEVICE_MIXED. The C_FULL denominators (K15, K17-K19) pair like every other denominator: the C_FULL
    entry of the same rule and seed (2026), on the same MAIN view and W; K20b pairs both FA_WARM seeds with C_FULL s2026
    (06L-C §C7); K11b pairs every FA seed with the one LAST rank file (06L-C §C3); K9b lives on UNSEEN_HOLDOUT."""
    return [contrast_record(eng, book, c) for c in contrasts["items"]]


def k14_equivalence(eng: Engine, key, b: Mapping) -> dict:
    """06L-C §C15: the K14 equivalence reading (TOST at alpha 0.05 <=> the 90% percentile CI inside [99, 101])."""
    np = LIB.np
    rr = READING_RULES["K14"]
    boot = eng.boot.get(key)
    if boot is None or not isinstance(b.get("ci95"), list):
        return {"status": "NOT_AVAILABLE", "reason": "no bootstrap distribution for K14", "rule": rr["rule"]}
    lo90, hi90 = (float(x) for x in np.percentile(boot, [5.0, 95.0]))
    lo, hi = rr["interval"]
    return {"status": "COMPUTED", "margin": [lo, hi], "ci90": [lo90, hi90], "equivalent": bool(lo <= lo90 and hi90 <= hi),
            "ci95_inside_margin": bool(lo <= b["ci95"][0] and b["ci95"][1] <= hi), "rule": rr["rule"],
            "registered": "06L-C §C15 (before any FA_Q8 read)"}


def contrast_record(eng: Engine, book: Book, c: Mapping, reg_as: Optional[str] = None) -> dict:
    kid = c["id"]
    pk = reg_as or kid                 # ANA-06: a 06N analog takes the shared-den / rule / seeds / reading rule of pk
    role = c.get("population") or "MAIN"
    shared = pk in CONTRAST_SHARED_DEN and c["den"] == CONTRAST_SHARED_DEN[pk]
    crule = CONFIRMATORY_RULE.get(pk, DEFAULT_CONFIRMATORY_RULE)
    reg = tuple(REGISTERED_SEEDS.get(pk, SEEDS))
    rec = {"id": kid, "contrast": c["contrast"], "question": c["question"], "label": "CONFIRMATORY",
           "kind": c["kind"], "num": c["num"], "den": c["den"], "population": role,
           "confirmatory_rule": crule, "hierarchy": HIERARCHY.get(pk, DEFAULT_HIERARCHY),
           "registered_seeds": list(reg), "by_rule": {}}
    if pk in PAIRING_NOTES:
        rec["pairing_note"] = PAIRING_NOTES[pk]
    for rule in RULES:
        label = "CONFIRMATORY" if rule == crule else "DESCRIPTIVE"
        all_pairs = book.pairs(role, c["num"], c["den"], rule, shared=shared)
        if pk in BASE_BEST_ONLY:                                       # ANA-06 (06P P5): base = the same-seed best.pt
            all_pairs = [p for p in all_pairs if (p[0].start_base or {}).get("checkpoint") == "best.pt"]
        if c["den"] in CONTRAST_FALLBACK:                              # K16: both sides must be ONNX_CPU
            all_pairs = [p for p in all_pairs if p[0].onnx and p[1].onnx]
            fb = book.pairs(role, c["num"], CONTRAST_FALLBACK[c["den"]], rule)
            if fb and not all_pairs:
                fpairs = [p for p in fb if _pair_seed(p) in reg]
                if fpairs:
                    st = _contrast_stat(eng, c, fpairs, f"{kid} fallback")
                    st.update(label="DESCRIPTIVE", comparison="DEVICE_MIXED",
                              note=f"{c['num']} (ONNX_CPU) over the A40 PyTorch {CONTRAST_FALLBACK[c['den']]}: "
                                   "mixes quantisation with runtime numerics (06E §3); never confirmatory")
                    rec.setdefault("descriptive_fallback", {})[rule] = st
        pairs = [p for p in all_pairs if _pair_seed(p, shared) in reg]
        extra = [p for p in all_pairs if _pair_seed(p, shared) not in reg]
        got = sorted({_pair_seed(p, shared) for p in pairs})
        missing = [s for s in reg if s not in got] if (rule == crule or all_pairs) else []
        if pairs:
            b = _contrast_stat(eng, c, pairs, kid, shared=shared, boot_key=(kid, rule))
            vk = _vkey_for(pairs, role)
            b["harmed_share_seed_null"] = {"numerator": seed_null(eng, book.get(role, c["num"], rule), vk),
                                           "denominator": seed_null(eng, book.get(role, c["den"], rule), vk)}
        else:
            b = {"status": "NOT_AVAILABLE"}
            if c["den"] in CONTRAST_FALLBACK:
                b["reason"] = f"no {c['num']} / {c['den']} pair inside the {ONNX_BLOCK} sub-block"
            if c["den"] in ANCHOR_ARMS and not any(is_anchor(e) for e in book.get(role, c["den"], rule).values()):
                b["reason"] = (f"no {c['den']} rank file (a {BASELINE_BLOCK} entry of v4base.run ranks_int32.npy; "
                               "06L-C §C3)")
        b["label"] = label
        b["seeds_status"] = ("INCOMPLETE_SEEDS" if (missing and all_pairs) else "NOT_RUN" if missing else
                             "COMPLETE" if pairs else "NOT_AVAILABLE")
        if missing:
            b["missing_registered_seeds"] = missing
        if pk in CONFIRMATORY_RULE and rule != crule:
            b["disclosure"] = DISCLOSURE_06D_3
        if pk in READING_RULES and rule == crule:
            b["equivalence"] = k14_equivalence(eng, (kid, rule), b)
        rec["by_rule"][rule] = b
        if extra:
            st = _contrast_stat(eng, c, all_pairs, f"{kid} all seeds", shared=shared)
            st.update(label="DESCRIPTIVE", note="mean over every available seed, including seeds outside the "
                                                f"registered set {list(reg)} (06E §2)")
            rec.setdefault("extra_seeds", {})[rule] = st
    return rec


EXTRA_REGISTERED_IDS = ("K20", "K21") + tuple(k for k, _c, _q in REGISTERED_CONTRASTS_06LC)   # 06K; 06L-C §C3 / §C7
WARM_ROWS = ("K20", "K20b", "K20c", "K21", "K9c")      # the deployment branch (06K §2; 06L-C §C7: the Edge answer)
REGISTERED_BY = {"K20": "06K §2", "K21": "06K §2", "K9b": "06L-C §C7", "K9c": "06L-C §C7", "K11b": "06L-C §C3",
                 "K20b": "06L-C §C7", "K20c": "06L-C §C7", "S8-S7": "06L-C §C8"}


def _parsed_item(kid: str, contrast: str, question: str) -> dict:
    return dict(id=kid, contrast=contrast, question=question, population=CONTRAST_POPULATION.get(kid, "MAIN"),
                **parse_contrast(contrast))


def extra_items(contrasts: Mapping) -> list:
    """The registered confirmatory rows the loaded file does not carry (v2: K20, K21 and the 06L-C rows; v3: none)."""
    have = {c["id"] for c in contrasts["items"]}
    reg = registered_items()
    return [_parsed_item(k, *reg[k]) for k in sorted(EXTRA_REGISTERED_IDS, key=order_key) if k not in have]


def descriptive_items(contrasts: Mapping) -> list:
    """The registered DESCRIPTIVE contrasts (06L-C §C8 S8-S7): from the v3 file, else from the script's registry."""
    if contrasts.get("descriptive_items"):
        return list(contrasts["descriptive_items"])
    reg = registered_items()
    return [_parsed_item(k, *reg[k]) for k in V3_DESCRIPTIVE_IDS]


def _make_descriptive(rec: dict, note: str) -> dict:
    rec.update(label="DESCRIPTIVE", hierarchy="-", note=note)
    for b in rec["by_rule"].values():
        b["label"] = "DESCRIPTIVE"
    return rec


def extra_registered(eng: Engine, book: Book, contrasts: Mapping) -> list:
    return [_make_descriptive(contrast_record(eng, book, c),
                              f"{c['id']} is registered by {REGISTERED_BY[c['id']]} but the loaded contrasts file "
                              f"(format {contrasts.get('format')}) does not carry it: computed and DESCRIPTIVE")
            for c in extra_items(contrasts)]


def bytes_ratio(book: Book, num_arm: str, den_arm: str, rule: str) -> dict:
    """06L-C §C8: bytes per client of S8 over S7, per seed and seed mean (the registered definition; the charged-dropped
    variant beside it)."""
    nums, dens = book.get("MAIN", num_arm, rule), book.get("MAIN", den_arm, rule)
    per = {}
    for s in sorted(x for x in nums if x is not None and x in dens):
        rec = {}
        for key in ("bytes_per_client", "bytes_per_client_charged_dropped"):
            a, b = nums[s].meta(key), dens[s].meta(key)
            rec[key.replace("bytes_per_client", "ratio")] = (float(a) / float(b) if a is not None and b else None)
        per[str(s)] = rec
    vals = [v["ratio"] for v in per.values() if v["ratio"] is not None]
    return {"per_seed": per, "mean_ratio": (sum(vals) / len(vals) if vals else None), "label": "DESCRIPTIVE",
            "rule": f"bytes per client {num_arm} / {den_arm} (S8 incl. the frozen tables once; 06L-C §C8)"}


def registered_descriptive(eng: Engine, book: Book, contrasts: Mapping) -> list:
    out = []
    for c in descriptive_items(contrasts):
        rec = _make_descriptive(contrast_record(eng, book, c), f"registered DESCRIPTIVE contrast "
                                f"({REGISTERED_BY.get(c['id'], '06L-C')}; never confirmatory)")
        rec["bytes_ratio"] = {rule: bytes_ratio(book, c["num"], c["den"], rule) for rule in RULES}
        out.append(rec)
    return out


N06_IDS = ("K20b", "K21", "K20", "K_INIT", "P5", "K4w", "K5w", "K12w", "K13w", "K14w")   # 06P P1 / P3-P5


def n06_items(star_arm: Optional[str]) -> list:
    """The registered 06P set (P1 K20b / K21 / K20 on FA_WARM_FROZEN; P3 K_INIT; P5; P4 K4w ... K14w) as items with
    ids '<id>*', computed (under the v4 tables, v4_tables) as DESCRIPTIVE when v1-v3 is loaded: (item, 06P tier, the
    06P id). [] when FA_WARM* is undetermined."""
    if star_arm is None:
        return []
    by = {it["id"]: it for it in v4_items()}
    return [(_parsed_item(f"{k}*", by[k]["contrast"], by[k]["question"]), HIERARCHY_V4.get(k, DEFAULT_HIERARCHY), k)
            for k in N06_IDS]


@contextlib.contextmanager
def v4_tables():
    """The 06P registry tables for a block of code (no-op when a v4 file is already loaded)."""
    if _PROFILE_SAVED:
        yield
        return
    set_registry_profile(CONTRAST_FORMAT_V4)
    try:
        yield
    finally:
        set_registry_profile(None)


def ana06(eng: Engine, book: Book, bind: Mapping, conf: Sequence[Mapping], a5: Mapping, fmt: Optional[str] = None) -> dict:
    """ANA-06 (PREG1_AMENDMENT_06N N1-N2): the pretrained-line contrasts on FA_WARM* (every row DESCRIPTIVE with the
    06N tier beside it; the v3 confirmatory set, hierarchy and Holm family are unchanged), FA_WARM*'s mechanical choice,
    the warm-line bindings and the needs-registration list."""
    star = dict(bind.get("fa_warm_star") or {"status": "UNDETERMINED", "stage": None, "arm": None})
    A = star.get("arm")
    if fmt == CONTRAST_FORMAT_V4:                                      # 06P: the loaded file carries the 06N set
        ids = [c["id"] for c in conf]
        return {"status": "IN_LOADED_FILE", "fa_warm_star": star, "registry_fa_warm_star": S8A,
                "contrasts": [], "confirmatory_ids": ids,
                "references": [{"id": k, "tier_06N": HIERARCHY.get(k, DEFAULT_HIERARCHY), "see": f"confirmatory {k}"}
                               for k in N06_IDS + ("K2",) if k in ids],
                "hierarchy_06N": {t: sorted(k for k in ids if HIERARCHY.get(k, DEFAULT_HIERARCHY) == t)
                                  for t in ("primary", "secondary", "tertiary")},
                "warm_line": bind.get("warm_line") or {}, "s_record_warm": bind.get("s_record_warm") or {},
                "needs_registration": list(N06_NEEDS_REGISTRATION),
                "label": "the loaded v4 registry's labels (06P)",
                "authority": "PREG1_AMENDMENT_06N N1-N3; PREG1_AMENDMENT_06P P1-P7; FL-16; DL 287 m3; DL 317"}
    rows = []
    with v4_tables():                                                  # the 06P rules / seeds / shared dens / P5 base
        for c, tier, analog in n06_items(A):
            rec = contrast_record(eng, book, c, reg_as=analog)
            rec.update(label="DESCRIPTIVE", status_06N=N06_LABEL, tier_06N=tier, analog_of=analog,
                       hierarchy="-", fa_warm_star=A, registry_fa_warm_star=S8A)
            for b in rec["by_rule"].values():
                b["label"] = "DESCRIPTIVE"
            rows.append(rec)
    k2 = next((c for c in conf if c["id"] == "K2"), None)
    refs = [{"id": "K2", "tier_06N": "secondary (the random-start ablation, 06P P1)", "see": "confirmatory K2"
             + (" (the loaded file's label / hierarchy; v4 makes it secondary)" if k2 else " (not in the loaded file)")},
            {"id": "S8-S7", "tier_06N": "descriptive (06P P2 / P6)",
             "see": "registered_descriptive S8-S7 (FA_WARM_FROZEN − FA_WARM)"}]
    return {"status": ("NOT_AVAILABLE" if A is None else "COMPUTED" if A == S8A else "COMPUTED_STAR_CONFLICT"),
            "reason": (star.get("reason") or "FA_WARM* undetermined" if A is None else None if A == S8A else
                       f"the selection files choose {A}; 06P P1 registers FA_WARM* = {S8A} (a v4 run refuses)"),
            "fa_warm_star": star, "contrasts": rows, "references": refs,
            "hierarchy_06N": {"primary": ["K20b", "K21"], "secondary": list(HOLM_FAMILY_V4),
                              "tertiary": list(TERTIARY_V4)},
            "warm_line": bind.get("warm_line") or {}, "s_record_warm": bind.get("s_record_warm") or {},
            "needs_registration": list(N06_NEEDS_REGISTRATION), "label": "DESCRIPTIVE",
            "authority": "PREG1_AMENDMENT_06N N1-N3; PREG1_AMENDMENT_06P; FL-16 (warm labels); DL 287 m3"}


def _has_warm_start(book: Book) -> bool:
    return any(arm in ("FA_WARM", "PRE", "FT_PRE", "FA_WARM_FROZEN") for (_r, arm, _x) in book.by)


def warm_start(eng: Engine, book: Book, contrasts: Mapping, conf: Sequence[Mapping], extra: Sequence[Mapping]) -> dict:
    """06K / DL 206 / 06L-C §C7: the deployment branch — K20 = R(FA_WARM / C_BIG), K20b = R(FA_WARM / C_FULL), K20c =
    R(PRE / C_FULL), K21 = FA_WARM - PRE, K9c = FT_PRE - PRE (secondary confirmatory with the v3 file; PRE = one θ_pre
    seed 2026 shared by every numerator seed; K20b pairs both FA_WARM seeds with C_FULL s2026). The rows reference the
    confirmatory records (v3) or the DESCRIPTIVE extra_registered records (a file that does not carry them)."""
    if not _has_warm_start(book):
        return {"status": "NOT_AVAILABLE", "reason": "no FA_WARM / PRE / FT_PRE entry in the block", "rows": [],
                "disclosures": list(DISCLOSURES_06K)}
    in_conf, in_extra = {c["id"] for c in conf}, {c["id"] for c in extra}
    rows = []
    for k in WARM_ROWS:
        if k in in_conf:
            rows.append({"id": k, "in_confirmatory": True})
        elif k in in_extra:
            rows.append({"id": k, "in_extra_registered": True})
    return {"status": "COMPUTED", "rows": rows, "disclosures": list(DISCLOSURES_06K)}


def personalisation(eng: Engine, book: Book) -> list:
    out = []
    for a, b in PERSONALISATION:
        rec = {"contrast": f"{a} - {b}", "label": "DESCRIPTIVE", "by_rule": {},
               "confirmatory_counterpart": {"FT_C": "K9", "FT_FA": "K10"}.get(a)}
        for rule in RULES:
            pairs = book.pairs("MAIN", a, b, rule)
            rec["by_rule"][rule] = (delta(eng, pairs, full_view("MAIN"), PRIMARY, name=rec["contrast"]) if pairs
                                    else {"status": "NOT_AVAILABLE"})
        out.append(rec)
    return out


def _seed_mean_R(tab: Mapping, arm: str, rule: str = CURVE_RULE, tag: str = "vs_C_BIG"):
    for r in tab.get(rule, []):
        if r["arm"] == arm:
            return (r.get(tag) or {}).get("mrr@20") or {"status": "NOT_AVAILABLE"}
    return {"status": "NOT_AVAILABLE"}


def _first_entry(book: Book, arm: str, rule: str = CURVE_RULE) -> Optional[Entry]:
    ents = book.get("MAIN", arm, rule)
    return ents[min(ents, key=lambda s: (s is None, s or 0))] if ents else None


def privacy_ladder(book: Book, tab: Mapping) -> list:
    out = []
    for level, arms, train, infer, guarantee in PRIVACY_LADDER:
        rows = []
        for arm in arms:
            e = _first_entry(book, arm)
            if e is None:
                continue
            g = guarantee
            if arm in ("DP8", "DP4", "DP1"):
                eps = e.meta("epsilon") if e.meta("epsilon") is not None else EPS_DEFAULT[arm]
                g = f"user-level (ε = {eps}, δ = 5e-6)-DP of the released model"
            rows.append({"arm": arm, "R_vs_C_BIG": _seed_mean_R(tab, arm), "bytes_per_client": e.meta("bytes_per_client"),
                         "bytes_per_client_charged_dropped": e.meta("bytes_per_client_charged_dropped"),
                         "bytes_per_client_charged_dropped_up_down": e.meta("bytes_per_client_charged_dropped_up_down"),
                         "bytes_per_visit": e.meta("bytes_per_visit"), "guarantee": g})
        out.append({"level": level, "leaves_training": train, "leaves_inference": infer, "guarantee": guarantee,
                    "arms": rows, "label": "DESCRIPTIVE"})
    return out


# ------------------------------------------------------------------------------------------------ ANA-04 (06L-C)
def _ci_side(ci, ref: float) -> Optional[str]:
    if not (isinstance(ci, list) and len(ci) == 2):
        return None
    return "ABOVE" if ci[0] > ref else "BELOW" if ci[1] < ref else "INCLUDES"


def anchor_entry(book: Book, role: str = "MAIN") -> Optional[Entry]:
    """06L-C §C3: the LAST rank file of the role (a BASELINE_CPU entry), or None."""
    return next((e for e in book.get(role, "LAST", CURVE_RULE).values() if is_anchor(e)), None)


def headroom(eng: Engine, units: Sequence[Tuple[Entry, Entry]], last: Entry, vk) -> dict:
    """06L-C §C3: R* = 100 (arm - LAST) / (C_ref - LAST) per (arm entry, C_ref entry) unit, per resample on the same
    W (seed mean of the per-unit R*; CI from the per-resample mean). A C_ref not above LAST (point or any resample)
    makes the value / CI UNDEFINED (the headroom is not defined there)."""
    np, B = LIB.np, LIB.B
    per, arrays = [], []
    for num, cref in units:
        pa, pl, pc = (eng.point(x, vk, PRIMARY) for x in (num, last, cref))
        row = {"num": num.eid, "C_ref": cref.eid, "seed": num.seed if num.seed is not None else cref.seed}
        if any(is_undef(x) for x in (pa, pl, pc)):
            row.update(point=undef("MISSING_EVIDENCE"), ci95=undef("MISSING_EVIDENCE"))
            arrays.append(None)
        elif float(pc) - float(pl) <= 0:
            row.update(point=undef("C_REF_NOT_ABOVE_LAST"), ci95=undef("C_REF_NOT_ABOVE_LAST"))
            arrays.append(None)
        else:
            row["point"] = 100.0 * (float(pa) - float(pl)) / (float(pc) - float(pl))
            if num.ranked and cref.ranked and last.ranked:
                ra, rl, rc = (eng.res(x, vk, PRIMARY) for x in (num, last, cref))
                den = rc - rl
                bad = int(np.count_nonzero(den <= 0))
                row["n_resamples_c_ref_not_above_last"] = bad
                if bad:
                    row["ci95"] = undef("C_REF_NOT_ABOVE_LAST_IN_RESAMPLES")
                    arrays.append(None)
                else:
                    vals = 100.0 * (ra - rl) / den
                    row["ci95"] = B.ci95(vals)
                    arrays.append(vals)
            else:
                row["ci95"] = undef("NO_RANKS")
                arrays.append(None)
        per.append(row)
    out = {"per_seed": per, "statistic": "R* = 100 (arm - LAST) / (C_ref - LAST), macro-E2E MRR@20",
           "label": "DESCRIPTIVE"}
    pts = [r["point"] for r in per]
    if not per or any(is_undef(p) for p in pts):
        return dict(out, point=next((p for p in pts if is_undef(p)), undef("MISSING_EVIDENCE")),
                    ci95=undef("MISSING_EVIDENCE"))
    out["point"] = float(np.mean(pts))
    out["ci95"] = (B.ci95(np.mean(arrays, axis=0)) if all(a is not None for a in arrays)
                   else next((r["ci95"] for r in per if is_undef(r["ci95"])), undef("NO_RANKS")))
    return out


def last_anchor(eng: Engine, book: Book, role: str = "MAIN") -> dict:
    """06L-C §C3: for every arm of the role, per rule: the paired Δ(arm - LAST) with CI (every seed of the arm against
    the one LAST rank file, same W), the headroom R* vs C_BIG (same rule and seed) — and vs C_FULL s2026 for the
    deployment-branch arms — and the flag BELOW_DEVICE_HEURISTIC (the CI of Δ lies below 0). Baselines (RULE_FREE) are
    listed once, under their own rule label."""
    last = anchor_entry(book, role)
    if last is None:
        return {"status": "NOT_AVAILABLE", "reason": f"no LAST rank file on {role} (a {BASELINE_BLOCK} entry; 06L-C §C3: "
                                                     "baseline runs also write per-row rank files)", "rows": []}
    rows = []
    for arm in book.arms(role):
        if arm == "LAST":
            continue
        for rule in RULES:
            ents = book.get(role, arm, rule)
            if not ents:
                continue
            rule_free = all(e.rule == RULE_FREE for e in ents.values())
            if rule_free and rule != CURVE_RULE:
                continue
            pairs = book.pairs(role, arm, "LAST", rule)
            if not pairs:
                continue
            vk = _vkey_for(pairs, role)
            rec = {"arm": arm, "rule": RULE_FREE if rule_free else rule, "role": role, "label": "DESCRIPTIVE"}
            d = delta(eng, pairs, vk, PRIMARY, name=f"{arm} - LAST {rule}")
            rec["delta_vs_LAST"] = d
            side = _ci_side(d.get("ci95"), 0.0)
            rec["BELOW_DEVICE_HEURISTIC"] = None if side is None else side == "BELOW"
            crefs = book.get(role, REF, CURVE_RULE if rule_free else rule)
            units = ([(n, c) for n in ents.values() for s, c in sorted(crefs.items(), key=lambda kv: kv[0] or 0)
                      if s is not None] if rule_free else
                     [(n, crefs[n.seed]) for _s, n in sorted(ents.items(), key=lambda kv: kv[0] or 0) if n.seed in crefs])
            rec["headroom_vs_C_BIG"] = (headroom(eng, units, last, vk) if units and arm != REF else
                                        {"status": "NOT_APPLICABLE" if arm == REF else "NO_C_BIG"})
            if arm in DEPLOYMENT_ARMS:
                cf = book.get(role, "C_FULL", rule).get(2026)
                rec["headroom_vs_C_FULL"] = (headroom(eng, [(n, cf) for _s, n in sorted(ents.items())], last, vk)
                                             if cf is not None else {"status": "NO_C_FULL"})
            rows.append(rec)
    return {"status": "COMPUTED", "anchor": last.eid, "rows": rows, "label": "DESCRIPTIVE",
            "rule": "BELOW_DEVICE_HEURISTIC = the 95% CI of Δ(arm - LAST) lies below 0 (06L-C §C3)"}


def device_envelope(eng: Engine, book: Book, tab: Mapping, anchor: Mapping) -> dict:
    """06L-C §C3: the device-only envelope {LO, LAST, RULE} as one row group (PRACTICAL_BEST; the baselines are
    rule-free): value (seed mean), R vs C_BIG (seed mean, CI), Δ vs LAST; `best_point` names the largest point value
    (DESCRIPTIVE, never a selection)."""
    rows, best = [], None
    by_arm = {(r["arm"], r["rule"]): r for r in anchor.get("rows", [])}
    for arm in ("LO", "LAST", "RULE"):
        ents = book.get("MAIN", arm, CURVE_RULE)
        if not ents:
            rows.append({"arm": arm, "status": "NOT_AVAILABLE"})
            continue
        vals = [eng.point(e, full_view("MAIN", e.subset_keys), PRIMARY) for e in ents.values()]
        v = None if any(is_undef(x) for x in vals) else float(sum(vals) / len(vals))
        a = by_arm.get((arm, CURVE_RULE)) or by_arm.get((arm, RULE_FREE)) or {}
        rows.append({"arm": arm, "value": v, "R_vs_C_BIG": _seed_mean_R(tab, arm),
                     "delta_vs_LAST": (a.get("delta_vs_LAST") if arm != "LAST" else {"status": "ANCHOR"}),
                     "seeds": sorted((s for s in ents if s is not None)) or [None]})
        if v is not None and (best is None or v > best[1]):
            best = (arm, v)
    return {"group": "device-only envelope {LO, LAST, RULE} (06L-C §C3)", "rows": rows, "label": "DESCRIPTIVE",
            "best_point": None if best is None else {"arm": best[0], "value": best[1]},
            "note": "LO is an untuned from-θ0 control, not the device-only ceiling (review 10 [9-R1 (d)])"}


def target_strata(eng: Engine, book: Book, role: str = "MAIN") -> dict:
    """06L-C §C15 (review 03 [2-D5]): REPEAT = RANKABLE rows whose target is in the user's L50 window (LAST's per-row tie
    count 0), NOVEL = every other E2E row; per arm (PRACTICAL_BEST, no users_subset) R vs C_BIG (same seed) on each
    stratum, paired on that stratum's own W. Row-level strata (a user can be in both)."""
    np = LIB.np
    last = anchor_entry(book, role)
    if last is None or last.ties is None:
        return {"status": "NOT_AVAILABLE", "reason": "needs the LAST rank file with its per-row tie count "
                                                     "(v4base.run n_tied_int32.npy; row_ties)", "rows": []}
    m = eng.ctx[role].manifest
    rep = m.eval_mask & (last.ranks > 0) & (last.ties == 0)
    nov = m.eval_mask & ~rep
    n_e2e = int(np.count_nonzero(m.eval_mask))
    counts = {"E2E_rows": n_e2e, "REPEAT_rows": int(np.count_nonzero(rep)), "NOVEL_rows": int(np.count_nonzero(nov)),
              "repeat_share_of_E2E_rows": (float(np.count_nonzero(rep)) / n_e2e) if n_e2e else None}
    rows = []
    for name, mask in (("TARGET:REPEAT", rep), ("TARGET:NOVEL", nov)):
        eng.row_masks[(role, name)] = mask
        vk = (role, name, None)
        if eng.view(vk) is None:
            rows.append({"stratum": name.split(":")[1], "status": "EMPTY_POPULATION"})
            continue
        for arm in book.arms(role):
            pairs = [(n, d) for n, d in book.pairs(role, arm, REF, CURVE_RULE) if n.ranked and d.ranked
                     and n.subset_keys is None]
            if not pairs:
                continue
            r = ratio(eng, pairs, vk, PRIMARY, strict=False, name=f"{name}:{arm}")
            rows.append({"stratum": name.split(":")[1], "arm": arm, "n_users": r["n_users"],
                         "n_decisions": r["n_decisions"], "R_vs_C_BIG": r["point"], "ci95": r["ci95"],
                         "per_seed": [{"seed": p["seed"], "value": p["value_num"], "C_ref": p["value_den"],
                                       "R": p["point"]} for p in r["per_seed"]], "label": "DESCRIPTIVE"})
        eng.drop(role, name)
        eng.row_masks.pop((role, name), None)
    return {"status": "COMPUTED", "counts": counts, "rows": rows, "label": "DESCRIPTIVE",
            "rule": "REPEAT = RANKABLE and LAST tie count 0 (target in the L50 window); NOVEL = the other E2E rows"}


def calib_disjoint(eng: Engine, book: Book, items: Sequence[Mapping], calib: Mapping, tab: Mapping) -> dict:
    """06L-C §C14: MAIN \\ CALIB for every CALIB-tuned arm (R vs C_BIG, PRACTICAL_BEST, same seed, beside the full-MAIN
    value) and for every registered contrast that uses a CALIB-tuned arm (its registered rule and seeds). DESCRIPTIVE,
    report only, never selection."""
    if not calib or calib.get("status") != "COMPUTED":
        return {"status": (calib or {}).get("status", "NOT_AVAILABLE"), "reason": (calib or {}).get("reason"),
                "arms": [], "contrasts": []}
    keys = calib["disjoint_keys"]
    arms, cons = [], []
    for arm in book.arms("MAIN"):
        if arm not in CALIB_TUNED_ARMS:
            continue
        pairs = [(n, d) for n, d in book.pairs("MAIN", arm, REF, CURVE_RULE) if n.ranked and d.ranked
                 and n.subset_keys is None]
        if not pairs:
            continue
        r = ratio(eng, pairs, _vkey_for(pairs, "MAIN", keys), PRIMARY, strict=False, name=f"CALIB_DISJOINT:{arm}")
        arms.append({"arm": arm, "rule": CURVE_RULE, "R_vs_C_BIG_disjoint": {k: r.get(k) for k in (
            "point", "ci95", "seeds", "n_users", "n_decisions", "per_seed")},
                     "R_vs_C_BIG_full_MAIN": {k: _seed_mean_R(tab, arm).get(k) for k in ("point", "ci95")},
                     "label": "DESCRIPTIVE"})
    for c in items:
        if (c.get("population") or "MAIN") != "MAIN" or not ({c["num"], c["den"]} & set(CALIB_TUNED_ARMS)):
            continue
        kid = c["id"]
        rule = CONFIRMATORY_RULE.get(kid, DEFAULT_CONFIRMATORY_RULE)
        shared = kid in CONTRAST_SHARED_DEN
        reg = tuple(REGISTERED_SEEDS.get(kid, SEEDS))
        pairs = [p for p in book.pairs("MAIN", c["num"], c["den"], rule, shared=shared) if _pair_seed(p, shared) in reg]
        if c["den"] in CONTRAST_FALLBACK:
            pairs = [p for p in pairs if p[0].onnx and p[1].onnx]
        if not pairs:
            continue
        st = _contrast_stat(eng, c, pairs, f"CALIB_DISJOINT:{kid}", shared=shared, keys_override=keys)
        st["label"] = "DESCRIPTIVE"
        cons.append({"id": kid, "contrast": c["contrast"], "rule": rule, "stat": st})
    eng.drop("MAIN", "CALIB_DISJOINT")
    return {"status": "COMPUTED", "counts": calib["counts"], "disjoint_users_sha256": calib["disjoint_users_sha256"],
            "mapping": calib["mapping"], "calib_eval_manifest_id": calib["calib_eval_manifest_id"], "arms": arms,
            "contrasts": cons, "label": "DESCRIPTIVE", "note": "report only, never selection (06L-C §C14)"}


def holm_band(eng: Engine, conf: Sequence[dict], contrasts: Mapping, alpha: float = 0.05) -> dict:
    """06P P6 (v4): the Holm step-down over the registered secondary family; the tertiary set gets its own DESCRIPTIVE
    band (step-down within it, "Holm band beside"); the primary pair is unadjusted. v1-v3: _holm_band unchanged."""
    fam = contrasts.get("holm_family")
    if not fam:
        return _holm_band(eng, conf, contrasts, alpha)
    out = _holm_band(eng, conf, dict(contrasts, holm_family=fam), alpha)
    ter = contrasts.get("tertiary_band_family") or []
    tconf = [c for c in conf if c["id"] in ter]
    tb = _holm_band(eng, tconf, dict(contrasts, holm_family=list(ter)), alpha)
    for c in tconf:
        b = c["by_rule"][c["confirmatory_rule"]]
        if (b.get("holm") or {}).get("status") == "COMPUTED":
            b["holm"].update(family="TERTIARY_BAND", holm_reject=None, decision="none (tertiary: unadjusted, 06P P6)")
    out["tertiary_band"] = {k: tb[k] for k in ("family", "family_size", "n_computed")}
    return out


def _holm_band(eng: Engine, conf: Sequence[dict], contrasts: Mapping, alpha: float = 0.05) -> dict:
    """06L-C §C13 (review 13 [12-E5]): Holm step-down on normal intervals from the bootstrap SE over the loaded
    registry (family size = its registered count), beside the unadjusted percentile CIs; DESCRIPTIVE. Each computed K
    gets {se, z, p, holm_rank, holm_level, p_holm, holm_reject, holm_band, bonferroni_band} under its confirmatory
    rule."""
    from statistics import NormalDist
    nd = NormalDist()
    fam = contrasts.get("holm_family")                                   # ANA-06 (06P P6): v4 names its Holm family
    m = len(fam) if fam else int(contrasts["n_contrasts"])
    comp = []
    for rec in conf:
        rule = rec["confirmatory_rule"]
        b = rec["by_rule"][rule]
        if fam and rec["id"] not in fam:
            if "holm" not in b:
                b["holm"] = {"status": "NOT_IN_HOLM_FAMILY", "tier": HIERARCHY.get(rec["id"], DEFAULT_HIERARCHY),
                             "rule": "06P P6: the primary pair is read unadjusted, outside Holm"}
            continue
        boot = eng.boot.get((rec["id"], rule))
        if boot is None or not isinstance(b.get("ci95"), list) or not isinstance(b.get("point"), float):
            b["holm"] = {"status": "NOT_COMPUTED", "reason": "no bootstrap distribution under the confirmatory rule"}
            continue
        ref = 100.0 if rec["kind"] == "retained" else 0.0
        se = float(LIB.np.std(boot, ddof=1))
        if se > 0:
            z = (b["point"] - ref) / se
            p = 2.0 * (1.0 - nd.cdf(abs(z)))
        else:
            z, p = (None, 0.0 if b["point"] != ref else 1.0)
        comp.append((p, order_key(rec["id"]), rec, b, se, z, ref))
    comp.sort(key=lambda x: (x[0], x[1]))
    running, stop, n_rej = 0.0, False, 0
    zb = nd.inv_cdf(1.0 - alpha / (2.0 * m))
    for j, (p, _o, rec, b, se, z, ref) in enumerate(comp, start=1):
        level = alpha / (m - j + 1)
        running = max(running, min(1.0, (m - j + 1) * p))
        reject = (not stop) and p <= level
        if reject:
            n_rej += 1
        stop = stop or not reject
        # REV-DELTA-03 M3: after the first non-rejection every later K gets the band at the level of that first
        # non-rejection, alpha / (m - r) with r = the rejections so far (the compatible simultaneous bound), so the band
        # excludes the reference exactly when holm_reject is True; a rejected K keeps its own step level
        band_level = level if reject else alpha / (m - n_rej)
        zq = nd.inv_cdf(1.0 - band_level / 2.0)
        b["holm"] = {"status": "COMPUTED", "se_bootstrap": se, "z": z, "p_normal_two_sided": p, "holm_rank": j,
                     "holm_level": level, "holm_band_level": band_level, "p_holm": running,
                     "holm_reject": bool(reject), "holm_band": [b["point"] - zq * se, b["point"] + zq * se],
                     "bonferroni_band": [b["point"] - zb * se, b["point"] + zb * se], "reference": ref,
                     "label": "DESCRIPTIVE"}
    return {"family": list(fam) if fam else [c["id"] for c in conf], "family_size": m, "alpha": alpha,
            "n_computed": len(comp),
            "n_rejected": n_rej,
            "method": "Holm step-down on normal intervals from the bootstrap SE (06L-C C13); after the first "
                      "non-rejection every non-rejected K's band is at alpha / (m - r), r = the number of rejections "
                      "(REV-DELTA-03 M3: band and decision agree); Bonferroni band beside it",
            "label": "DESCRIPTIVE"}


def no_ft_references(eng: Engine, book: Book, records: Mapping) -> dict:
    """06L-C §C16 (review 10 [9-R3]): C_SMALL / FA re-evaluated on UNSEEN_VAL (no fine-tuning) beside the SELECTION
    record's value of the FT arm's selected cell on the same panel. DESCRIPTIVE (a selection panel, never a headline)."""
    B = LIB.B
    if NO_FT_ROLE not in eng.ctx:
        return {"status": "NOT_AVAILABLE", "reason": f"no {NO_FT_ROLE} role in the block", "rows": []}
    rows = []
    for ft, base in NO_FT_PAIRS:
        sel = next((s for s in records.get("SELECTION") or [] if s.get("arm") == ft), None)
        for rule in RULES:
            for seed, e in sorted(book.get(NO_FT_ROLE, base, rule).items(), key=lambda kv: kv[0] or 0):
                vk = full_view(NO_FT_ROLE, e.subset_keys)
                v = eng.point(e, vk, PRIMARY)
                ci = B.ci95(eng.res(e, vk, PRIMARY)) if e.ranked and eng.view(vk) is not None else undef("NO_RANKS")
                sv = None
                if sel and isinstance(sel.get("values_primary"), Mapping) and sel.get("cell") is not None:
                    sv = sel["values_primary"].get(str(sel["cell"]))
                rows.append({"ft_arm": ft, "base_arm": base, "rule": rule, "seed": seed, "no_ft_value": v, "no_ft_ci95": ci,
                             "selection_file": (sel or {}).get("file"), "selection_seed": (sel or {}).get("seed"),
                             "selected_cell": (sel or {}).get("cell"), "selected_cell_value": sv,
                             "selected_minus_no_ft": (float(sv) - float(v) if isinstance(sv, (int, float))
                                                      and not is_undef(v) else None), "label": "DESCRIPTIVE"})
    return {"status": "COMPUTED" if rows else "NOT_AVAILABLE", "role": NO_FT_ROLE, "rows": rows, "label": "DESCRIPTIVE",
            "note": DISCLOSURES_ANA04[7]}


def family_comparison(eng: Engine, book: Book) -> dict:
    """06L-C §C11: GRU_D128 vs C_SMALL on the A40 block, PRACTICAL_BEST and CONTROLLED_BUDGET (paired R and Δ, same
    seed); DESCRIPTIVE, never a downstream family."""
    a, b = FAMILY_PAIR
    if not any(arm == a for (_r, arm, _x) in book.by):
        return {"status": "NOT_AVAILABLE", "reason": f"no {a} entry", "by_rule": {}, "text": REGISTERED_06LC["C11"]}
    out = {}
    for rule in RULES:
        pairs = book.pairs("MAIN", a, b, rule)
        if not pairs:
            out[rule] = {"status": "NOT_AVAILABLE"}
            continue
        vk = _vkey_for(pairs, "MAIN")
        ea, eb = pairs[0]
        out[rule] = {"R": ratio(eng, pairs, vk, PRIMARY, strict=True, name=f"{a}/{b} {rule}"),
                     "delta": delta(eng, pairs, vk, PRIMARY, name=f"{a} - {b} {rule}"),
                     "param_count": {a: ea.meta("param_count"), b: eb.meta("param_count")}, "label": "DESCRIPTIVE"}
    return {"status": "COMPUTED", "pair": [a, b], "by_rule": out, "label": "DESCRIPTIVE", "text": REGISTERED_06LC["C11"]}


def frozen_branch(eng: Engine, book: Book) -> dict:
    """06L-C §C16-C17: DP8_FROZEN (S8 + DP ε = 8, seed 2026) against its z = 0 twin S8 (FA_WARM_FROZEN), both rules
    (CONTROLLED_BUDGET = the data-independent endpoint); DESCRIPTIVE."""
    if not any(arm == "DP8_FROZEN" for (_r, arm, _x) in book.by):
        return {"status": "NOT_AVAILABLE", "reason": "no DP8_FROZEN entry", "by_rule": {}}
    out = {}
    for rule in RULES:
        pairs = book.pairs("MAIN", "DP8_FROZEN", "FA_WARM_FROZEN", rule)
        out[rule] = (ratio(eng, pairs, _vkey_for(pairs, "MAIN"), PRIMARY, strict=True, name=f"DP8_FROZEN/S8 {rule}")
                     if pairs else {"status": "NOT_AVAILABLE"})
        if pairs:
            out[rule]["label"] = "DESCRIPTIVE"
            if rule == "PRACTICAL_BEST":
                out[rule]["disclosure"] = DISCLOSURE_06D_3
    e = _first_entry(book, "DP8_FROZEN")
    return {"status": "COMPUTED", "contrast": "R(DP8_FROZEN / FA_WARM_FROZEN)", "by_rule": out, "label": "DESCRIPTIVE",
            "epsilon": (e.meta("epsilon") if e and e.meta("epsilon") is not None else EPS_DEFAULT["DP8_FROZEN"]),
            "dp_z": e.meta("dp_z") if e else None, "text": REGISTERED_06LC["C16-C17"]}


def descriptive_seeds(book: Book) -> dict:
    """ANA-05: the entries of a DESCRIPTIVE source seed (C_FULL s2027) and where they appear."""
    rows = []
    for (role, arm, rule), ents in sorted(book.by.items()):
        for seed in sorted(s for s in ents if s in DESCRIPTIVE_SEEDS.get(arm, ())):
            rows.append({"arm": arm, "seed": seed, "rule": rule, "role": role, "entry": ents[seed].eid,
                         "label": "DESCRIPTIVE"})
    return {"status": "COMPUTED" if rows else "NOT_AVAILABLE", "rows": rows, "rule": DESCRIPTIVE_SEED_RULE,
            "registered_seeds": {k: list(v) for k, v in sorted(REGISTERED_SEEDS.items())}}


def production_stack(eng: Engine, book: Book, bases: Sequence[Mapping]) -> dict:
    """06M_B B4 (DESCRIPTIVE): the paired delta FT_FA_WARM - FA_WARM (and FT_FA_WARM_FROZEN - FA_WARM_FROZEN), per
    seed on MAIN, PRACTICAL_BEST (the device arm vs its base's best.pt), with the harmed-client share and per-user delta
    quantiles (06L-C C13-C15). A base whose checkpoint is endpoint_6.0.pt is reported, not paired (BASE_NOT_BEST)."""
    rule_of = {b["entry"]: b.get("base_checkpoint_rule") for b in bases}
    out = []
    for ft, base in PRODUCTION_STACK:
        fts = book.get("MAIN", ft, CURVE_RULE)
        if not fts:
            out.append({"contrast": f"{ft} − {base}", "status": "NOT_AVAILABLE", "reason": f"no {ft} entry"})
            continue
        pairs, skipped = [], []
        bents = book.get("MAIN", base, CURVE_RULE)
        for seed, e in sorted(fts.items(), key=lambda kv: kv[0] or 0):
            if rule_of.get(e.eid) != "PRACTICAL_BEST":
                skipped.append({"seed": seed, "reason": f"BASE_NOT_BEST (base checkpoint rule {rule_of.get(e.eid)})"})
            elif seed in bents:
                pairs.append((e, bents[seed]))
            else:
                skipped.append({"seed": seed, "reason": f"no {base} PRACTICAL_BEST entry of this seed"})
        rec = {"contrast": f"{ft} − {base}", "label": "DESCRIPTIVE", "rule": CURVE_RULE,
               "authority": "PREG1_AMENDMENT_06M_B B4; 06L-C C13-C15 (harmed-client share)", "skipped": skipped}
        if pairs:
            rec["delta"] = delta(eng, pairs, _vkey_for(pairs, "MAIN"), PRIMARY, name=f"{ft} - {base}")
            rec["status"] = "COMPUTED"
        else:
            rec["status"] = "NOT_AVAILABLE"
        out.append(rec)
    return {"status": "COMPUTED" if any(r.get("status") == "COMPUTED" for r in out) else "NOT_AVAILABLE",
            "rows": out, "label": "DESCRIPTIVE"}


def rollout_quality(eng: Engine, book: Book) -> dict:
    """06M_B B5 (DESCRIPTIVE): quality during rollout of FA, FA_WARM and S8 (ROLLOUT_RULE)."""
    np, B = LIB.np, LIB.B
    vk = full_view("MAIN")
    pre_ents = book.get("MAIN", "PRE", CURVE_RULE)
    refs = {"PRE": pre_ents.get(2026) or next((x for _s, x in sorted(pre_ents.items(), key=lambda kv: kv[0] or 0)),
                                              None),
            "LAST": anchor_entry(book, "MAIN")}
    refs = {k: v for k, v in refs.items() if v is not None and v.ranked and v.subset_keys is None}
    ref_val = {k: eng.point(v, vk, PRIMARY) for k, v in refs.items()}
    runs, curve = [], []
    for arm in ROLLOUT_ARMS:
        for seed, e in sorted(book.get("MAIN", arm, CURVE_RULE).items(), key=lambda kv: kv[0] or 0):
            if not e.marks or e.subset_keys is not None:
                continue
            vals, boots, first, first_ci = [], [], {k: None for k in ROLLOUT_REFS}, {k: None for k in ROLLOUT_REFS}
            for mk in e.marks:
                pseudo = dataclasses.replace(e, ranks=mk["ranks"], rows=mk["rows"], run_dir_rel=None, marks=[],
                                             subset_keys=None)
                pe = _Marked(pseudo, f"{e.eid}|mark={mk['mark']:.1f}")
                v = eng.point(pe, vk, PRIMARY)
                vals.append(float(v))
                boots.append(eng.res(pe, vk, PRIMARY))
                row = {"arm": arm, "seed": seed, "mark": mk["mark"], "round": mk["round"], "value": float(v),
                       "label": "DESCRIPTIVE"}
                for k, ref in refs.items():
                    d = delta(eng, [(pe, ref)], vk, PRIMARY, name=f"{pe.eid} - {k}")
                    row[f"delta_vs_{k}"] = {"point": d["point"], "ci95": d["ci95"]}
                    if first[k] is None and float(v) >= float(ref_val[k]):
                        first[k] = mk["mark"]
                    if first_ci[k] is None and isinstance(d["ci95"], list) and d["ci95"][0] > 0:
                        first_ci[k] = mk["mark"]
                curve.append(row)
            marks = [mk["mark"] for mk in e.marks]
            span = marks[-1] - marks[0]
            auc = float(sum(0.5 * (vals[i] + vals[i + 1]) * (marks[i + 1] - marks[i]) for i in range(len(vals) - 1)))
            bauc = sum(0.5 * (boots[i] + boots[i + 1]) * (marks[i + 1] - marks[i]) for i in range(len(vals) - 1)) / span
            reach = {}
            for k in ROLLOUT_REFS:
                if k not in refs:
                    reach[k] = {"status": "NO_REFERENCE"}
                else:
                    reach[k] = {"first_mark_point_rule": first[k] if first[k] is not None else "NOT_REACHED",
                                "first_mark_ci_rule": first_ci[k] if first_ci[k] is not None else "NOT_REACHED",
                                "reference_entry": refs[k].eid, "reference_value": float(ref_val[k])}
            runs.append({"arm": arm, "seed": seed, "entry": e.eid, "n_marks": len(marks), "auc": auc,
                         "auc_mean": auc / span, "auc_mean_ci95": B.ci95(np.asarray(bauc)), "_boot": bauc,
                         "reach": reach, "label": "DESCRIPTIVE"})
    by_arm = {}
    for arm in ROLLOUT_ARMS:
        rs = [r for r in runs if r["arm"] == arm]
        if rs:
            by_arm[arm] = {"seeds": [r["seed"] for r in rs], "auc_mean": float(np.mean([r["auc_mean"] for r in rs])),
                           "auc_mean_ci95": B.ci95(np.mean([r["_boot"] for r in rs], axis=0)), "label": "DESCRIPTIVE"}
    for r in runs:
        r.pop("_boot")
    return {"status": "COMPUTED" if runs else "NOT_AVAILABLE", "runs": runs, "by_arm": by_arm, "curve": curve,
            "references": {k: v.eid for k, v in refs.items()}, "rule": ROLLOUT_RULE, "label": "DESCRIPTIVE",
            "authority": "PREG1_AMENDMENT_06M_B B5"}


def s8_comm(book: Book) -> dict:
    """06L-C C8 / ANA-05: S8's (and DP8_FROZEN's) per-visit communication from the run's own fl.comm record (RUN_SUMMARY
    bytes.per_visit), beside FA_WARM's of the same seed; the measured uploaded parameter count beside the registered
    265,162 (to be re-measured). DESCRIPTIVE."""
    rows = []
    for arm in FROZEN_ONCE_ARMS:
        for seed, e in sorted(book.get("MAIN", arm, CURVE_RULE).items(), key=lambda kv: kv[0] or 0):
            if not e.marks:
                continue
            warm = book.get("MAIN", "FA_WARM", CURVE_RULE).get(seed)
            bpv, wpv = e.meta("bytes_per_visit"), (warm.meta("bytes_per_visit") if warm is not None else None)
            up = e.meta_auto.get("uploaded_numel")
            rows.append({"arm": arm, "seed": seed, "entry": e.eid, "bytes_per_visit": bpv,
                         "bytes_up_per_visit": e.meta("bytes_up_per_visit"),
                         "bytes_down_per_visit": e.meta("bytes_down_per_visit"),
                         "uploaded_numel_measured": up, "frozen_numel": e.meta_auto.get("frozen_numel"),
                         "frozen_bytes_once": e.meta_auto.get("frozen_bytes_once"),
                         "bytes_per_client_incl_frozen_once": e.meta("bytes_per_client"),
                         "uploaded_numel_registered_text": S8_REGISTERED_UPLOADED,
                         "uploaded_numel_equals_registered_text": (None if up is None else int(up) == S8_REGISTERED_UPLOADED),
                         "FA_WARM_bytes_per_visit_same_seed": wpv,
                         "per_visit_ratio_vs_FA_WARM": (float(bpv) / float(wpv) if bpv is not None and wpv else None),
                         "source": "RUN_SUMMARY bytes.per_visit (fl.comm visit_bytes of the run itself)",
                         "label": "DESCRIPTIVE"})
    return {"status": "COMPUTED" if rows else "NOT_AVAILABLE", "rows": rows, "label": "DESCRIPTIVE",
            "text": REGISTERED_06LC["C8"]}


def ana05(eng: Engine, book: Book, bind: Mapping) -> dict:
    """ANA-05 (06M, 06M_B B2 / B4 / B5, C_FULL s2027, S8 comm) - every item DESCRIPTIVE; the v3 confirmatory set is
    unchanged (26 contrasts)."""
    return {"descriptive_seeds": descriptive_seeds(book),
            "warm_selection": {"status": "CHECKED", "stages": dict(bind.get("warm_selection") or {}),
                               "rule": bind.get("rule")},
            "ft_warm_bases": list(bind.get("ft_warm_bases") or []),
            "production_stack": production_stack(eng, book, bind.get("ft_warm_bases") or []),
            "rollout_quality": rollout_quality(eng, book),
            "s8_comm": s8_comm(book),
            "warm_preview": {"status": "REFUSED_AS_BLOCK_ENTRY", "rule": PREVIEW_LABEL},
            "label": "DESCRIPTIVE", "disclosures": list(DISCLOSURES_ANA05)}


def bytes_axis(book: Book, ctx: Mapping[str, RoleCtx], records: Mapping, tab: Mapping) -> list:
    """06L-C §C8 (review 09 [8-C3]): one bytes-per-client axis for every privacy level — raw events (the per-event size
    x mean TRAIN decisions per client) -> model download (C_SMALL FP32 / INT8) -> FL traffic (+ the final download) ->
    S8 (the frozen tables once + the encoder rounds) — with R vs C_BIG (PRACTICAL_BEST, seed mean) beside it. Design
    quantities, not results; DESCRIPTIVE."""
    np = LIB.np
    out = []
    ev = records.get("EVENT_SIZE") or {}
    main = ctx.get("MAIN")
    sup = list((main.support_by_hex or {}).values()) if main is not None and main.support_by_hex else []
    per_client = float(np.mean(sup)) if sup else None
    bpe = ev.get("bytes_per_event")
    out.append({"step": 1, "level": "raw events to the server (L0 / L1 cloud training)", "arm": "C_BIG / C_SMALL / C_FULL",
                "bytes_per_client": (float(bpe) * per_client if bpe is not None and per_client is not None else None),
                "train_decisions_per_client_mean": per_client, "bytes_per_event": bpe,
                "status": "COMPUTED" if bpe is not None and per_client is not None else
                ("NOT_REGISTERED (06L-C §C8: the per-event size convention needs a DL row; records.EVENT_SIZE)"
                 if bpe is None else "NO_TRAIN_SUPPORT_TABLE"),
                "rule": "bytes_per_event (records.EVENT_SIZE) x mean TRAIN decisions per MAIN client (TRAIN manifest)"})
    for arm, what in (("C_SMALL", "model download FP32 (L1 device inference; L5 / FT base; PRE-alone)"),
                      ("C_SMALL_INT8", "model download INT8 (ONNX dynamic quantisation)")):
        e = _first_entry(book, arm) or (_first_entry(book, "C_SMALL_FP32") if arm == "C_SMALL" else None)
        mb = e.meta("model_bytes") if e is not None else None
        out.append({"step": 2, "level": what, "arm": arm, "bytes_per_client": mb, "status": "COMPUTED" if mb else
                    "NO_MODEL_BYTES", "rule": "the model file bytes (entry meta model_bytes), downloaded once"})
    for arm in FL_ARMS:
        e = _first_entry(book, arm)
        if e is None:
            continue
        train, down = e.meta("bytes_per_client"), e.meta("bytes_down_per_visit")
        out.append({"step": 4 if arm in FROZEN_ONCE_ARMS else 3,
                    "level": ("S8: frozen tables once + encoder rounds" if arm in FROZEN_ONCE_ARMS else
                              "FL training traffic (+ the final download)"), "arm": arm, "seed": e.seed,
                    "bytes_per_client": train, "bytes_per_client_charged_dropped": e.meta("bytes_per_client_charged_dropped"),
                    "final_download_bytes": down,
                    "bytes_per_client_incl_final_download": (float(train) + float(down) if train is not None and down
                                                             is not None else None),
                    "R_vs_C_BIG": _seed_mean_R(tab, arm), "status": "COMPUTED" if train is not None else "NO_BYTES_RECORD",
                    "rule": (FROZEN_ONCE_RULE if arm in FROZEN_ONCE_ARMS else
                             "visits x (down + up) per client (06A §3), registered mean") + "; + one final model download"})
    for arm in ("LO", "LAST", "RULE"):
        if _first_entry(book, arm) is None:
            continue
        out.append({"step": 0, "level": "device-only" if arm != "RULE" else "device + published aggregate", "arm": arm,
                    "bytes_per_client": 0 if arm != "RULE" else None, "R_vs_C_BIG": _seed_mean_R(tab, arm),
                    "status": "COMPUTED" if arm != "RULE" else "AGGREGATE_SIZE_NOT_MEASURED",
                    "rule": "no upload; only the shared bootstrap block (catalogue / θ0)" if arm != "RULE" else
                            "the published MAIN-TRAIN popularity / category aggregate is downloaded (size not measured)"})
    for r in out:
        r["label"] = "DESCRIPTIVE"
    return sorted(out, key=lambda r: (r["step"], ARMS.index(r["arm"]) if r["arm"] in ARMS else -1))


def privacy_matrix(book: Book, tab: Mapping, axis: Sequence[Mapping]) -> list:
    """06L-C §C1: the four-column privacy matrix (+ secure aggregation) per arm present in the block, with R vs C_BIG
    (PRACTICAL_BEST, seed mean) and the bytes per client of the bytes axis. DESCRIPTIVE."""
    present = {a for (_r, a, _x) in book.by} | {e.arm for e in book.hist}
    bytes_of = {r["arm"]: r.get("bytes_per_client") for r in axis}
    out = []
    for level, arms, train, infer, guarantee, whom, secagg in PRIVACY_MATRIX:
        rows = [{"arm": a, "R_vs_C_BIG": _seed_mean_R(tab, a), "bytes_per_client": bytes_of.get(a)}
                for a in arms if a in present]
        out.append({"level": level, "server_sees_training": train, "server_sees_inference": infer,
                    "formal_guarantee": guarantee, "against_whom": whom, "secure_aggregation": secagg, "arms": rows,
                    "label": "DESCRIPTIVE"})
    return out


def _claim_fill(text: str, c: Mapping, tier_rule: str, seeds) -> str:
    for k, v in (("{num}", c["num"]), ("{den}", c["den"]), ("{population}", c.get("population") or "MAIN"),
                 ("{rule}", tier_rule), ("{seeds}", ", ".join(str(s) for s in seeds)), ("{id}", c["id"]),
                 ("{question}", c["question"])):
        text = text.replace(k, str(v))
    return text


def claim_register(contrasts: Mapping) -> dict:
    """06L-C §C12 (review 15 [14-W2, W10]): allowed / forbidden sentence templates per contrast (number-free; the
    placeholders {point} {ci_lo} {ci_hi} {abs_point} {status} are the only slots for numbers) + the headline template."""
    items = list(contrasts["items"]) + extra_items(contrasts) + descriptive_items(contrasts)
    have = {c["id"] for c in contrasts["items"]}
    out = []
    for c in sorted(items, key=lambda x: order_key(x["id"])):
        kid = c["id"]
        rule = CONFIRMATORY_RULE.get(kid, DEFAULT_CONFIRMATORY_RULE)
        seeds = REGISTERED_SEEDS.get(kid, SEEDS)
        tier = (HIERARCHY.get(kid, DEFAULT_HIERARCHY) if kid in have else "descriptive")
        allowed = {k: _claim_fill(t, c, rule, seeds) for k, t in CLAIM_TEMPLATES[c["kind"]].items()}
        allowed.update({k: _claim_fill(CLAIM_TEMPLATES[k], c, rule, seeds) for k in ("NOT_AVAILABLE", "UNDEFINED")})
        if tier == "descriptive":
            allowed = {k: "DESCRIPTIVE: " + v for k, v in allowed.items()}
        out.append({"id": kid, "contrast": c["contrast"], "question": c["question"], "kind": c["kind"],
                    "population": c.get("population") or "MAIN", "tier": tier, "confirmatory_rule": rule,
                    "registered_seeds": list(seeds), "allowed": allowed, "forbidden": list(CLAIM_FORBIDDEN.get(kid, [])),
                    "pairing_note": PAIRING_NOTES.get(kid)})
    return {"format": CLAIM_REGISTER_FORMAT, "registry_format": contrasts["format"],
            "registry_content_sha256": contrasts["content_sha256"], "general_rules": list(CLAIM_GENERAL_RULES),
            "flags": {"SEED_SENSITIVE": "06L-C C13", "BELOW_DEVICE_HEURISTIC": "06L-C C3", "INCOMPLETE_SEEDS": "A-F4",
                      "EQUIVALENT": EQUIVALENT_FLAG, "DEVICE_MIXED": "never confirmatory (06E §3)",
                      "POST_VALIDATION_EXPLORATORY": "06L-C C5"},
            "contrasts": out, "headline_template": HEADLINE_TEMPLATE, "authority": REGISTERED_06LC["C12"]}


def stamp_claims(*groups: Sequence[dict]) -> None:
    """The claim-register template key for each registered contrast under its confirmatory rule (ABOVE / BELOW /
    INCLUDES / NOT_AVAILABLE / UNDEFINED) + its flags; no number is added."""
    for grp in groups:
        for rec in grp:
            rule = rec.get("confirmatory_rule", DEFAULT_CONFIRMATORY_RULE)
            b = rec["by_rule"].get(rule) or {}
            ref = 100.0 if rec["kind"] == "retained" else 0.0
            if b.get("status") == "NOT_AVAILABLE":
                key = "NOT_AVAILABLE"
            else:
                key = _ci_side(b.get("ci95"), ref) or "UNDEFINED"
            flags = []
            if (b.get("seed_stats") or {}).get("seed_sensitive"):
                flags.append("SEED_SENSITIVE")
            if b.get("seeds_status") == "INCOMPLETE_SEEDS":
                flags.append("INCOMPLETE_SEEDS")
            if b.get("comparison") == "DEVICE_MIXED":
                flags.append("DEVICE_MIXED")
            if (b.get("equivalence") or {}).get("equivalent"):
                flags.append("EQUIVALENT")
            b["claim"] = {"template_key": key, "flags": flags, "label_of_the_row": b.get("label"),
                          "register": CLAIM_REGISTER_FORMAT}


def curve_bytes(eng: Engine, book: Book) -> list:
    """R vs cumulative bytes per client from the 12 in-run marks (06A §3); denominator C_BIG PRACTICAL_BEST, same
    seed. Bytes at a mark = the run's per-client mean x round / rounds (PRORATED_BY_ROUND, INFERENCE). S8
    (FA_WARM_FROZEN; GATE-B B3): the one-time frozen-table component is a step charged in full from the first visit and
    only the training traffic is prorated (FROZEN_CURVE_RULE)."""
    out = []
    for (role, arm, rule), ents in sorted(book.by.items(), key=lambda kv: (kv[0][0], ARMS.index(kv[0][1]), kv[0][2])):
        if role != "MAIN" or rule != CURVE_RULE:
            continue
        for seed, e in sorted(ents.items(), key=lambda kv: (kv[0] is None, kv[0] or 0)):
            den = book.get("MAIN", REF, CURVE_RULE).get(seed)
            if not e.marks or den is None:
                continue
            total_rounds, per_client = e.meta("rounds"), e.meta("bytes_per_client")
            charged = e.meta("bytes_per_client_charged_dropped")
            charged_ud = e.meta("bytes_per_client_charged_dropped_up_down")
            step = e.meta_auto.get("frozen_once_per_client_mean") if e.arm in FROZEN_ONCE_ARMS else None

            def cum_at(total, rnd, _step=step, _rounds=total_rounds):
                if _step is None:
                    return float(total) * float(rnd) / float(_rounds)
                return _step + (float(total) - _step) * float(rnd) / float(_rounds)   # B3: step + prorated training
            for mk in e.marks:
                pseudo = dataclasses.replace(e, ranks=mk["ranks"], rows=mk["rows"], run_dir_rel=None, marks=[],
                                             subset_keys=None)
                pseudo_eid = f"{e.eid}|mark={mk['mark']:.1f}"
                r = ratio(eng, [(_Marked(pseudo, pseudo_eid), den)], full_view("MAIN"), PRIMARY, strict=False,
                          name=pseudo_eid)
                cum, cum_charged, cum_charged_ud = None, None, None
                if total_rounds and per_client is not None and mk["round"] is not None:
                    cum = cum_at(per_client, mk["round"])
                    if charged is not None:
                        cum_charged = cum_at(charged, mk["round"])
                    if charged_ud is not None:
                        cum_charged_ud = cum_at(charged_ud, mk["round"])
                out.append({"arm": arm, "seed": seed, "mark": mk["mark"], "efe_raw": mk["efe_raw"], "round": mk["round"],
                            "cum_bytes_per_client": cum, "cum_bytes_per_client_charged_dropped": cum_charged,
                            "cum_bytes_per_client_charged_dropped_up_down": cum_charged_ud,
                            "bytes_rule": "PRORATED_BY_ROUND (INFERENCE); the registered series is cum_bytes_per_client "
                                          "(06A 3), the charged-dropped series (dropped visits charged their download, "
                                          "06I 7b / DL 218) is the second series, the up + down variant is DESCRIPTIVE",
                            "value": r["per_seed"][0]["value_num"], "R_vs_C_BIG_PB": r["per_seed"][0]["point"],
                            "ci95": r["per_seed"][0]["ci95"], "label": "DESCRIPTIVE"})
                if step is not None:
                    out[-1].update(bytes_rule=FROZEN_CURVE_RULE, frozen_once_step_bytes_per_client=step)
    return out


class _Marked:
    """An Entry proxy with its own eid (one FL mark) so the engine caches marks separately."""

    def __init__(self, e: Entry, eid: str):
        self._e, self._eid = e, eid

    def __getattr__(self, k):
        return getattr(self._e, k)

    @property
    def eid(self):
        return self._eid


def curve_epsilon(book: Book, tab: Mapping, dp_wor: Optional[dict]) -> list:
    out = []
    for arm in EPS_ARMS:
        e = _first_entry(book, arm)
        if e is None:
            continue
        z = e.meta("dp_z")
        rec = {"arm": arm, "epsilon_registered_accountant": e.meta("epsilon") if e.meta("epsilon") is not None
               else EPS_DEFAULT[arm], "dp_z": z, "label": "DESCRIPTIVE",
               "by_rule": {rule: {"R_vs_C_BIG": _seed_mean_R(tab, arm, rule),
                                  "R_vs_FA_1024": _seed_mean_R_pair(tab, arm, rule)} for rule in RULES}}
        if dp_wor is not None and z:
            rec["epsilon_receipt_fields"] = dp_wor["by_z"].get(_zkey(z), "Z_NOT_IN_RECEIPT")
        out.append(rec)
    return out


def _seed_mean_R_pair(tab, arm, rule):
    return (tab.get("_vs_FA_1024", {}).get(rule, {}).get(arm)) or {"status": "NOT_AVAILABLE"}


def vs_fa1024(eng: Engine, book: Book) -> dict:
    out = {}
    for rule in RULES:
        out[rule] = {}
        for arm in EPS_ARMS[1:]:
            pairs = book.pairs("MAIN", arm, "FA_1024", rule)
            if pairs:
                out[rule][arm] = ratio(eng, pairs, full_view("MAIN"), PRIMARY, strict=True, name=f"{arm}/FA_1024")
    return out


def curve_size(book: Book, tab: Mapping) -> list:
    out = []
    for arm in SIZE_ARMS:
        e = _first_entry(book, arm)
        if e is None:
            continue
        out.append({"arm": arm, "param_count": e.meta("param_count"), "model_bytes": e.meta("model_bytes"),
                    "by_rule": {rule: _seed_mean_R(tab, arm, rule) for rule in RULES}, "label": "DESCRIPTIVE",
                    "note": "INT8 / ONNX FP32 on its users_subset when given (06A §6)" if e.subset_keys is not None
                    else None})
    return out


def _group_users(eng: Engine, role: str, labels_by_euk: Mapping[int, str]) -> Dict[str, object]:
    np = LIB.np
    groups = {}
    for k, lab in labels_by_euk.items():
        groups.setdefault(lab, []).append(int(k))
    return {lab: np.asarray(sorted(v), dtype=np.int64) for lab, v in groups.items()}


def _subgroup_rows(eng: Engine, book: Book, role: str, family: str, groups: Mapping, arms=None) -> list:
    out = []
    ref = book.get(role, REF, CURVE_RULE)
    if not ref:
        return out
    n_role = int(_euks(eng.ctx[role]).size)
    for lab in sorted(groups):
        keys = groups[lab]
        vk = full_view(role) if int(keys.size) == n_role else (role, f"{family}:{lab}", tuple(int(x) for x in keys))
        for arm in (arms or book.arms(role)):
            pairs = [(n, d) for n, d in book.pairs(role, arm, REF, CURVE_RULE) if n.ranked and d.ranked
                     and n.subset_keys is None]
            if not pairs:
                continue
            r = ratio(eng, pairs, vk, PRIMARY, strict=False, name=f"{family}:{lab}:{arm}")
            out.append({"role": role, "family": family, "group": lab, "arm": arm, "n_users": r["n_users"],
                        "R_vs_C_BIG": r["point"], "ci95": r["ci95"],
                        "per_seed": [{"seed": p["seed"], "value": p["value_num"], "C_ref": p["value_den"],
                                      "R": p["point"]} for p in r["per_seed"]],
                        "study": "A" if arm in STUDY_A else "B" if arm in STUDY_B else "OTHER", "label": "DESCRIPTIVE"})
        eng.drop(role, f"{family}:{lab}")
    return out


def _euks(ctx: RoleCtx):
    np = LIB.np
    m = ctx.manifest
    return np.unique(m.eval_user_key[m.eval_mask])


def curve_strata(eng: Engine, book: Book, strata_obj: Optional[dict]) -> list:
    if strata_obj is None:
        return []
    ctx = eng.ctx["MAIN"]
    if ctx.hex_by_euk is None:
        raise AnaRefused("strata need the MAIN users table (client_key_hex)")
    fam = {"category_diversity": {}, "brand_loyalty": {}, "persona": {}, "activity": {}}
    for k in _euks(ctx).tolist():
        s = LIB.ST.assign_stratum_for_user(strata_obj, str(ctx.hex_by_euk[k]))
        if s["status"] != "BUCKETED":
            for f in fam:
                fam[f][k] = LIB.ST.NO_TRAIN_STRATUM
            continue
        fam["category_diversity"][k] = f"T{s['category_diversity_tercile']}"
        fam["brand_loyalty"][k] = f"T{s['brand_loyalty_tercile']}"
        fam["persona"][k] = f"d{s['category_diversity_tercile']}_l{s['brand_loyalty_tercile']}"
        fam["activity"][k] = f"Q{s['activity_quantile']}"
    out = []
    for f in ("category_diversity", "brand_loyalty", "persona", "activity"):
        out += _subgroup_rows(eng, book, "MAIN", f, _group_users(eng, "MAIN", fam[f]))
    return out


def support_bin(n: int) -> str:
    for lo, hi, lab in SUPPORT_BINS:
        if lo <= n <= hi:
            return lab
    return "OUT_OF_REGISTERED_BINS:" + ("1-4" if 1 <= n <= 4 else ">100")


def curve_bins(eng: Engine, book: Book) -> Tuple[list, list, dict]:
    rows, counts = [], {}
    for role in sorted(eng.ctx):
        c = eng.ctx[role]
        if role not in BIN_PANELS + (COLD_ROLE,) or role not in {r for (r, _a, _x) in book.by}:
            continue
        euks = _euks(c).tolist()
        if role == COLD_ROLE and c.support_by_hex is None:
            labels = {k: "0" for k in euks}
        else:
            if c.hex_by_euk is None or c.support_by_hex is None:
                raise AnaRefused(f"support bins on {role} need the users table and the TRAIN-layout manifest")
            labels = {k: support_bin(int(c.support_by_hex.get(str(c.hex_by_euk[k]), 0))) for k in euks}
        groups = _group_users(eng, role, labels)
        counts[role] = {lab: int(v.size) for lab, v in sorted(groups.items())}
        reg = {lab: v for lab, v in groups.items() if not lab.startswith("OUT_OF")}
        rows += _subgroup_rows(eng, book, role, "support_bin", reg)
    ev = events_to_x_bins(rows)
    return rows, ev, counts


def events_to_x_bins(rows: list) -> list:
    """06A §2: per arm and seed, the smallest bin b with M(b) >= X * C_ref(b) (C_ref = C_BIG PRACTICAL_BEST, same bin,
    same seed); bin 0 comes from TRUE_COLD; reported only if reached; no interpolation."""
    order = [lab for _lo, _hi, lab in SUPPORT_BINS]
    val = {}
    for r in rows:
        for p in r["per_seed"]:
            val[(r["role"], r["arm"], p["seed"], r["group"])] = (p["value"], p["C_ref"])
    out = []
    arms_seeds = sorted({(a, s) for (_r, a, s, _g) in val}, key=lambda x: (ARMS.index(x[0]), x[1] or 0))
    for panel in BIN_PANELS:
        for arm, seed in arms_seeds:
            seq = []
            for lab in order:
                role = COLD_ROLE if lab == "0" else panel
                if (role, arm, seed, lab) in val:
                    seq.append((lab, role) + val[(role, arm, seed, lab)])
            if not any(role == panel for _l, role, _m, _c in seq):
                continue
            rec = {"panel": panel, "arm": arm, "seed": seed, "bins": [
                {"bin": lab, "role": role, "M": m, "C_ref": c} for lab, role, m, c in seq], "label": "DESCRIPTIVE"}
            for x in X_LEVELS:
                hit = next((lab for lab, _role, m, c in seq if not is_undef(m) and not is_undef(c)
                            and float(m) >= x * float(c)), None)
                rec[f"events_to_{int(round(x * 100))}"] = hit if hit is not None else "NOT_REACHED"
            out.append(rec)
    return out


# ------------------------------------------------------------------------------------------------ DATA-07 per-h rows
def _first_hit_or_status(seq: list, x: float):
    """GATE-B B4 (review 11 [10-R4]): the first-hit rule over the registered h in order. A missing C_ref row makes the
    value undecidable, so it reads NO_C_REF (never NOT_REACHED, which is a substantive negative finding): at the
    first h without C_ref before any hit, the result is NO_C_REF; NOT_REACHED only when every h had a C_ref."""
    for h, m, c, _R in seq:
        if c is None:
            return "NO_C_REF"
        if m >= x * c:
            return h
    return "NOT_REACHED"


TRUNC_H_BY_TYPE = {"FIXED_50": (0, 1, 5, 10, 20, 50), "FIXED_20": (0, 1, 5, 10, 20),   # = v4trunc.rules
                   "ALL_AVAILABLE": (0, 1, 5, 10, 20, 50)}                            # FIXED_BUDGETS / BUDGETS (E11.1)
TRUNC_PER_USER_FORMAT = "PPSI_V4_TRUNC_PER_USER_1"
PERSISTENCE_RULE = (
    "06L-C §C10 persistence rule (beside the registered first-hit rule): the smallest registered h of the series' panel "
    "type such that at h AND at the next registered h the point ratio is >= X and the 95% paired user-bootstrap "
    "percentile CI lower bound of the ratio (1,000 resamples, seed 20260926, one W per user set, the ratio per resample) "
    "is >= 100 X; the largest registered h is never a persistent hit (it has no next h). Statuses: NO_C_REF (a reference "
    "row is missing before a hit), NO_NEXT_H (the point holds at h but the next registered h has no row), NO_CI (the "
    "points hold at h and at the next h but a CI is not computable: no per-user evidence for a cell, or UNDEFINED), "
    "NOT_REACHED (every h decided and none persists). Applied to R(h) = M(h) / C_ref (C_BIG FULL history) and to the "
    "matched-h parity P(h) = M(h) / C_BIG(h). DESCRIPTIVE; an 'events to parity' sentence needs a persistent hit.")


def trunc_cell_key(arm: str, seed, panel: str, ptype: str, qh: int, full: bool) -> str:
    """The per-user evidence key of one DATA-07 cell (PPSI_V4_TRUNC_PER_USER_1: arrays `u__<key>` = the sorted int64
    eval_user_key of the cell's E2E users, `m__<key>` = their per-user macro-E2E MRR@20 in the same order)."""
    return f"{arm}__{seed}__{panel}__{ptype}__{int(qh)}__{'F' if full else 'H'}"


def load_trunc_per_user(p: Path) -> dict:
    """{cell key: (users int64 sorted unique, per-user MRR@20 float64)} of a PPSI_V4_TRUNC_PER_USER_1 .npz (no pickle)."""
    np = LIB.np
    out = {}
    with np.load(p, allow_pickle=False) as z:
        names = set(z.files)
        if "format" not in names or str(z["format"]) != TRUNC_PER_USER_FORMAT:
            raise AnaRefused(f"{p.name}: not a {TRUNC_PER_USER_FORMAT} file (array `format`)")
        for n in sorted(names - {"format"}):
            pair = ("m__" + n[3:]) if n.startswith("u__") else ("u__" + n[3:]) if n.startswith("m__") else None
            if pair is None or pair not in names:
                raise AnaRefused(f"{p.name}: array {n!r} is not one of a u__ / m__ pair ({TRUNC_PER_USER_FORMAT})")
            if n.startswith("m__"):
                continue
            u, m = z[n], z[pair]
            if (u.ndim != 1 or u.dtype.kind not in "iu" or m.shape != u.shape or m.dtype.kind != "f" or u.size == 0
                    or (u.size > 1 and not bool(np.all(u[1:] > u[:-1]))) or not bool(np.all(np.isfinite(m)))):
                raise AnaRefused(f"{p.name}: cell {n[3:]!r} needs sorted unique integer user keys and finite per-user "
                                 "values of the same length")
            out[n[3:]] = (u.astype(np.int64), m.astype(np.float64))
    return out


_TRUNC_W: dict = {}


def _trunc_ratio_ci(per_user: Optional[Mapping], km: str, kc: str, v_m: float, v_c: float) -> tuple:
    """([lo, hi] of 100 M / C by the paired user bootstrap, None) or (None, reason). Both cells must carry per-user
    evidence on the SAME users (a matched pair is one truncation panel cell); each cell's per-user mean must reproduce
    its row value within ROW_VALUE_TOL (else refuse). W = v4eval.bootstrap.make_W(n) over the sorted users (one W per
    user set); a zero denominator in any resample makes the CI UNDEFINED (no resample dropped, E7.5)."""
    np = LIB.np
    if not per_user or km not in per_user or kc not in per_user:
        return None, "NO_PER_USER_EVIDENCE"
    (um, mm), (uc, mc) = per_user[km], per_user[kc]
    for k, m, v in ((km, mm, v_m), (kc, mc, v_c)):
        if abs(float(m.mean()) - float(v)) > ROW_VALUE_TOL:
            raise AnaRefused(f"TRUNC per-user evidence {k}: the per-user mean {float(m.mean())!r} does not reproduce "
                             f"the row value {float(v)!r} (06L-C §C10)")
    if um.shape != uc.shape or not bool(np.array_equal(um, uc)):
        raise AnaRefused(f"TRUNC per-user evidence: {km} and {kc} are not on the same users (a matched pair is one "
                         "truncation panel cell; 06L-C §C10)")
    n = int(um.shape[0])
    if n not in _TRUNC_W:
        _TRUNC_W.clear()
        _TRUNC_W[n] = LIB.B.make_W(n)
    W = _TRUNC_W[n]
    rm = np.concatenate([mm[W[a:a + 100]].mean(axis=1) for a in range(0, W.shape[0], 100)])
    rc = np.concatenate([mc[W[a:a + 100]].mean(axis=1) for a in range(0, W.shape[0], 100)])
    if bool(np.any(rc == 0)):
        return None, "CI_UNDEFINED_ZERO_DENOMINATOR_IN_RESAMPLES"
    return LIB.B.ci95(100.0 * rm / rc), None


def _next_registered_h(ptype: str, h: int):
    later = [g for g in TRUNC_H_BY_TYPE.get(ptype, TRUNC_H) if g > int(h)]
    return later[0] if later else None


def _persistent_hit_or_status(seq: list, x: float, ptype: str):
    """06L-C §C10 persistence rule over seq = [(h, M, ref, CI lower bound in percent or None)] in registered order
    (PERSISTENCE_RULE)."""
    by_h = {h: (m, c, lo) for h, m, c, lo in seq}
    for h, m, c, lo in seq:
        if c is None:
            return "NO_C_REF"
        nh = _next_registered_h(ptype, h)
        if nh is None:
            break
        if m < x * c:
            continue
        if nh not in by_h:
            return "NO_NEXT_H"
        m2, c2, lo2 = by_h[nh]
        if c2 is None:
            return "NO_C_REF"
        if m2 < x * c2:
            continue
        if lo is None or lo2 is None:
            return "NO_CI"
        if lo >= 100.0 * x and lo2 >= 100.0 * x:
            return h
    return "NOT_REACHED"


def trunc_curves(rows: list, key: Optional[Mapping] = None, per_user: Optional[Mapping] = None) -> Tuple[list, list]:
    """Controlled cold start (06B §6.4; contract §11.7 / E11.10; A-F1) from DATA-07 rows (v4int.launch_trunc.trunc_labels
    fields trunc_panel, trunc_panel_type, trunc_h, trunc_full_history, trunc_query_h): R(h) = M(h) / C_ref, C_ref = the
    C_BIG (method) FULL-history row of the same panel / panel type / seed — per query_h for ALL_AVAILABLE, the one FULL
    cell for a FIXED sub-panel; M's own FULL-history value is reported beside each series; events-to-X = the smallest
    registered h with M(h) >= X * C_ref(h); no interpolation; NO_C_REF when a C_ref row is missing before the first hit
    (GATE-B B4), NOT_REACHED only when every h had its C_ref. Study B rows keep only the selected device cell. Duplicate
    keys refuse; the block key (eval_device_class, split, evaluator sha) applies to every row. Point values (DESCRIPTIVE).
    06L-C §C10 (REV-DELTA-03 m8): beside the first-hit rule, the matched-h parity P(h) = M(h) / C_BIG(h) (C_BIG's own
    H_h row of the same panel / panel type / seed / h) and the persistence rule (PERSISTENCE_RULE) for both R and P; the
    CI lower bounds come from `per_user` (records.TRUNC_ROWS[i].per_user, PPSI_V4_TRUNC_PER_USER_1); without it the
    persistent result reads NO_CI wherever the points would persist (never a hit)."""
    pts, fulls = {}, {}
    for r in rows:
        if not _is_evidence(r, *PRIMARY):
            continue
        if "device_selected" in r and r.get("device_selected") is not True:
            continue                                                   # Study B: the selected cell only
        meth, arm_f = r.get("method"), r.get("arm")
        if meth is not None and arm_f is not None and meth != arm_f:
            raise AnaRefused(f"TRUNC row: method {meth!r} != arm {arm_f!r}")
        arm = meth if meth is not None else arm_f
        for f in ("trunc_panel", "trunc_panel_type", "trunc_query_h", "trunc_full_history"):
            if r.get(f) is None:
                raise AnaRefused(f"TRUNC row ({arm}): field {f} missing (DATA-07 trunc_labels format; A-F1)")
        if key is not None:
            for f in ("eval_device_class", "split", "evaluator_code_sha256"):
                if r.get(f) != key.get(f):
                    raise AnaRefused(f"TRUNC row ({arm} {r.get('trunc_panel')}): {f} {r.get(f)!r} is not the block's "
                                     f"{key.get(f)!r} (A-F1)")
        v = r.get("value")
        if v is None or r.get("value_status", "OK") != "OK":
            continue
        panel, ptype, seed = r["trunc_panel"], r["trunc_panel_type"], r.get("seed")
        qh, full = int(r["trunc_query_h"]), bool(r["trunc_full_history"])
        ck = trunc_cell_key(arm, seed, panel, ptype, qh, full)
        if full:
            k = (arm, panel, ptype, seed, qh if ptype == "ALL_AVAILABLE" else None)
            if k in fulls:
                raise AnaRefused(f"TRUNC: duplicate FULL-history row {k} (A-F1)")
            fulls[k] = (float(v), ck)
        else:
            h = r.get("trunc_h")
            if h is None or int(h) != qh:
                raise AnaRefused(f"TRUNC row ({arm} {panel} {ptype}): trunc_h {h!r} != trunc_query_h {qh}")
            k = (arm, panel, ptype, seed, int(h))
            if k in pts:
                raise AnaRefused(f"TRUNC: duplicate row {k} (A-F1)")
            pts[k] = (float(v), r.get("trunc_study"), ck)
    curve, ev = [], []
    series = sorted({k[:4] for k in pts}, key=lambda x: tuple(map(str, x)))
    for arm, panel, ptype, seed in series:
        seq, rseq, pseq, m_full = [], [], [], {}
        for h in TRUNC_H:
            if (arm, panel, ptype, seed, h) not in pts:
                continue
            m, study, km = pts[(arm, panel, ptype, seed, h)]
            fk = h if ptype == "ALL_AVAILABLE" else None
            c, kc = fulls.get((REF, panel, ptype, seed, fk), (None, None))
            mf = fulls.get((arm, panel, ptype, seed, fk), (None, None))[0]
            m_full[str(h) if fk is not None else "FULL"] = mf
            R = undef("NO_C_REF") if c is None else LIB.Q.retained_percent(m, c)
            r_ci, r_why = (None, "NO_C_REF") if c is None else _trunc_ratio_ci(per_user, km, kc, m, c)
            seq.append((h, m, c, R))
            rseq.append((h, m, c, None if r_ci is None else float(r_ci[0])))
            row = {"arm": arm, "panel": panel, "panel_type": ptype, "seed": seed, "h": h, "study": study, "M": m,
                   "C_ref": c, "R": R, "R_ci95": r_ci if r_ci is not None else undef(r_why)}
            if arm == REF:
                row.update(C_BIG_h=m, parity_matched_h=undef("SELF_REFERENCE"), parity_ci95=undef("SELF_REFERENCE"))
            else:
                ch, _st, kch = pts.get((REF, panel, ptype, seed, h), (None, None, None))
                if ch is None:
                    P, p_ci, p_why = undef("NO_C_BIG_AT_H"), None, "NO_C_BIG_AT_H"
                else:
                    P = LIB.Q.retained_percent(m, ch)
                    p_ci, p_why = _trunc_ratio_ci(per_user, km, kch, m, ch)
                pseq.append((h, m, ch, None if p_ci is None else float(p_ci[0])))
                row.update(C_BIG_h=ch, parity_matched_h=P, parity_ci95=p_ci if p_ci is not None else undef(p_why))
            row.update(M_full_history=mf, label="DESCRIPTIVE")
            curve.append(row)
        rec = {"arm": arm, "panel": panel, "panel_type": ptype, "seed": seed, "label": "DESCRIPTIVE",
               "C_ref_method": REF, "M_full_history": (m_full.get("FULL") if ptype != "ALL_AVAILABLE" else m_full)}
        for x in X_LEVELS:
            tag = int(round(x * 100))
            rec[f"events_to_{tag}"] = _first_hit_or_status(seq, x)
            rec[f"events_to_{tag}_persistent"] = _persistent_hit_or_status(rseq, x, ptype)
            if arm == REF:
                rec[f"parity_events_to_{tag}"] = rec[f"parity_events_to_{tag}_persistent"] = "SELF_REFERENCE"
            else:
                rec[f"parity_events_to_{tag}"] = _first_hit_or_status([(h, m, c, None) for h, m, c, _lo in pseq], x)
                rec[f"parity_events_to_{tag}_persistent"] = _persistent_hit_or_status(pseq, x, ptype)
        ev.append(rec)
    return curve, ev


# ------------------------------------------------------------------------------------------------ records
def _zkey(z) -> str:
    return f"{float(z):.4f}"


def parse_dp_wor(obj) -> dict:
    """Tolerant: every dict with a numeric z; all numeric fields whose name contains 'eps' reported for that z."""
    by_z = {}

    def walk(o):
        if isinstance(o, Mapping):
            z = o.get("z")
            if isinstance(z, (int, float)) and not isinstance(z, bool):
                eps = {k: v for k, v in o.items() if "eps" in k.lower() and isinstance(v, (int, float))
                       and not isinstance(v, bool)}
                if eps:
                    by_z.setdefault(_zkey(z), {}).update(eps)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)
    return {"by_z": by_z, "self_sha256_ok": _self_ok(obj)}


def _self_ok(obj) -> str:
    if not isinstance(obj, Mapping) or SELF_KEY not in obj:
        return "NO_SELF_SHA256"
    body = {k: v for k, v in obj.items() if k != SELF_KEY}
    utf8 = _sha_bytes(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                 allow_nan=False).encode("utf-8"))   # how company/gates/*.json were written (non-ASCII text)
    if obj[SELF_KEY] in (_canon_compact(obj), _canon_indent(obj), utf8):
        return "VERIFIED"
    return "MISMATCH"


def _scalars(obj: Mapping) -> dict:
    return {k: v for k, v in sorted(obj.items()) if isinstance(v, (int, float, str, bool)) or v is None}


def load_records(block: Path, idx: Mapping, files: dict) -> dict:
    rec = idx.get("records") or {}
    out = {}

    def rd(rel):
        p = _inside(block, rel)
        files[str(rel)] = _sha_file(p)
        return _read_json(p)

    if rec.get("STRATA"):
        p = _inside(block, rec["STRATA"])
        files[str(rec["STRATA"])] = _sha_file(p)
        out["STRATA"] = LIB.ST.load_registry(p)                        # verifies its self-hash before any use
        want = (idx.get("roles") or {}).get("MAIN", {}).get("train_manifest_id")
        if out["STRATA"].get("source_manifest_id") != want:              # A-F16
            raise AnaRefused(f"STRATA source_manifest_id {out['STRATA'].get('source_manifest_id')!r} is not "
                             f"roles.MAIN.train_manifest_id {want!r} (A-F16)")
    if rec.get("S_RECORD"):
        s = rd(rec["S_RECORD"])
        if s.get(SELF_KEY) != _canon_compact(s):
            raise AnaRefused("S_RECORD.json: self_sha256 mismatch")
        out["S_RECORD"] = {k: s.get(k) for k in ("format", "S", "n_norms", "rule", "seed", "label", "disclosure",
                                                 SELF_KEY)}
    sels = rec.get("SELECTION") or []
    out["SELECTION"] = []
    for rel in ([sels] if isinstance(sels, str) else sels):
        s = rd(rel)
        if s.get(SELF_KEY) != _canon_compact(s):
            raise AnaRefused(f"{rel}: SELECTION self_sha256 mismatch")
        out["SELECTION"].append({"file": str(rel), **{k: s.get(k) for k in ("format", "arm", "seed", "role",
                                                                             "selection_role", "cell",
                                                                             "values_primary", "rule", SELF_KEY)}})
    if rec.get("LR_SELECTION"):
        s = rd(rec["LR_SELECTION"])
        out["LR_SELECTION"] = {"label": "CALIB (not MAIN)", "self_sha256_status": _self_ok(s), "fields": _scalars(s),
                               "values": s.get("values") or s.get("candidates") or s.get("runs")}
    if rec.get("DP_WOR"):
        out["DP_WOR"] = parse_dp_wor(rd(rec["DP_WOR"]))
    if rec.get("DP_Z_EPS4"):
        s = rd(rec["DP_Z_EPS4"])
        out["DP_Z_EPS4"] = {"self_sha256_status": _self_ok(s), "fields": _scalars(s)}
    if rec.get("EVENT_SIZE"):                                  # 06L-C §C8: the registered per-event size convention
        s = rd(rec["EVENT_SIZE"])
        bpe = s.get("bytes_per_event")
        if _self_ok(s) != "VERIFIED" or not isinstance(bpe, (int, float)) or isinstance(bpe, bool) or bpe <= 0:
            raise AnaRefused("records.EVENT_SIZE: needs a self-verified record with a positive bytes_per_event (the "
                             "per-event size convention of 06L-C §C8, registered by a DECISION_LOG row)")
        out["EVENT_SIZE"] = {"bytes_per_event": bpe, "rule": s.get("rule"), "decision_log_row": s.get("decision_log_row"),
                             SELF_KEY: s.get(SELF_KEY)}
    for st, key in sorted(WARM_SELECTION_RECORD.items()):      # ANA-05 (06M_B B2): the registered 2-D warm selection
        if rec.get(key):
            p = _inside(block, rec[key])
            fsha = files[str(rec[key])] = _sha_file(p)
            wobj = _read_json(p)
            out[key] = warm_selection_record(wobj, st, fsha, str(rec[key]))
            out.setdefault("_WARM_WINNER_CALIB", {})[st] = warm_winner_calib(wobj)   # ANA-06 (06N N2), kept internal
    for variant, key in sorted(S_RECORD_WARM_KEY.items()):     # ANA-06 (FL-16): the warm DP chain's S records
        if rec.get(key):
            p = _inside(block, rec[key])
            fsha = files[str(rec[key])] = _sha_file(p)
            out[key] = s_record_warm(_read_json(p), variant, fsha, str(rec[key]))
    trunc = rec.get("TRUNC_ROWS") or []
    out["TRUNC_ROWS"], out["TRUNC_FILES"], out["TRUNC_PER_USER"] = [], [], {}
    for item in ([trunc] if not isinstance(trunc, list) else trunc):
        if not isinstance(item, Mapping) or not item.get("rows"):
            raise AnaRefused("records.TRUNC_ROWS: each item must be {'rows': <file>, 'rule': 'PRACTICAL_BEST'} (the "
                             "DATA-07 rows carry selection_rule = None; A-F1)")
        if item.get("rule") != CURVE_RULE:
            raise AnaRefused(f"records.TRUNC_ROWS {item.get('rows')}: declared rule {item.get('rule')!r} is not "
                             f"{CURVE_RULE} (curves use PRACTICAL_BEST; A-F1)")
        rows = _rows_of(rd(item["rows"]))
        refuse_non_evidence(f"TRUNC_ROWS {item['rows']}", rows)
        bad = sorted({str(r.get("selection_rule")) for r in rows if r.get("selection_rule") not in (None, CURVE_RULE)})
        if bad:
            raise AnaRefused(f"records.TRUNC_ROWS {item['rows']}: rows carry selection_rule {bad}")
        out["TRUNC_ROWS"] += rows
        tf = {"rows": str(item["rows"]), "rule": item["rule"], "n_rows": len(rows)}
        if item.get("per_user"):                      # 06L-C §C10: per-user evidence for the persistence CI
            pp = _inside(block, item["per_user"])
            files[str(item["per_user"])] = tf["per_user_sha256"] = _sha_file(pp)
            tf["per_user"] = str(item["per_user"])
            pu = load_trunc_per_user(pp)
            dup = sorted(set(pu) & set(out["TRUNC_PER_USER"]))
            if dup:
                raise AnaRefused(f"records.TRUNC_ROWS: two per-user files carry cell {dup[0]!r}")
            out["TRUNC_PER_USER"].update(pu)
            tf["per_user_cells"] = len(pu)
        out["TRUNC_FILES"].append(tf)
    if rec.get("VALIDATION_RESULT"):
        s = rd(rec["VALIDATION_RESULT"])
        if s.get("schema") != SCHEMA or not verify_self_sha(s):
            raise AnaRefused("VALIDATION_RESULT: not a self-verified RETENTION_RESULT")
        if s.get("split") != "VALIDATION":
            raise AnaRefused("VALIDATION_RESULT: its split is not VALIDATION")
        out["VALIDATION_RESULT"] = s
    return out


def warm_selection_record(obj: Mapping, stage: str, file_sha256: str, rel: str) -> dict:
    """ANA-05 (06M_B B2): the registered contracts/fl/FL_WARM_SELECTION_<stage>.json (lr_screen select-warm, FL-15):
    format PPSI_V4_FL_WARM_SELECTION_2, self_sha256 (the launch_fl canon rule), the stage's arm, a winner consistent
    with winner_factor / winner_server_lr_factor and theta_pre = the theta_pre identity. The 06J decision itself was
    re-verified by launch_fl.load_warm_selection_2d at launch; its CALIB values are kept out of the result."""
    arm = {v: k for k, v in WARM_SELECTION_STAGE.items()}[stage]
    w = obj.get("winner") or {}
    bad = {}
    if obj.get("format") != WARM_SELECTION_FORMAT_2D:
        bad["format"] = obj.get("format")
    if obj.get(SELF_KEY) != _canon_compact(obj):
        bad["self_sha256"] = "does not verify"
    if obj.get("stage") != stage or obj.get("arm") != arm:
        bad["stage/arm"] = (obj.get("stage"), obj.get("arm"))
    try:
        if float(w.get("warm_lr_factor")) != float(obj.get("winner_factor")) or \
                float(w.get("warm_server_lr_factor")) != float(obj.get("winner_server_lr_factor")):
            bad["winner"] = w
    except (TypeError, ValueError):
        bad["winner"] = w
    why = init_from_problem(obj.get("init_from"))
    if why:
        bad["init_from"] = why
    if bad:
        raise AnaRefused(f"records.{WARM_SELECTION_RECORD[stage]} {rel}: not a verified {WARM_SELECTION_FORMAT_2D} "
                         f"{stage} file: {bad} (06M_B B2)")
    return {"file": rel, "file_sha256": file_sha256, "self_sha256": obj[SELF_KEY], "stage": stage, "arm": arm,
            "format": obj["format"], "winner_factor": float(obj["winner_factor"]),
            "winner_server_lr_factor": float(obj["winner_server_lr_factor"]), "winner_label": w.get("label"),
            "no_measurable_gain": obj.get("no_measurable_gain"), "init_from": dict(obj["init_from"]),
            "tuning_selection_sha256": (obj.get("tuning_selection") or {}).get("sha256"),
            "label": "CALIB selection record (values not carried here)"}


def warm_winner_calib(obj: Mapping) -> dict:
    """ANA-06 (06N N2): the CALIB value of a verified 2-D warm selection's selected cell (values[winner.label]); a value
    that is missing or not finite gives value None with the reason (FA_WARM* then stays UNDETERMINED)."""
    label = (obj.get("winner") or {}).get("label")
    v = (obj.get("values") or {}).get(label) if label is not None else None
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))
    return {"winner_label": label, "value": float(v) if ok else None,
            "reason": None if ok else f"values[{label!r}] is {v!r} (no finite CALIB value of the selected cell)"}


def s_record_warm(obj: Mapping, variant: str, file_sha256: str, rel: str) -> dict:
    """ANA-06 (FL-16 S_CAL_WARM*; REV-FL-16 m5 on the ANA side): a registered warm S record (contracts/dp/
    S_RECORD_WARM[_FROZEN].json) — format PPSI_V4_DP_S_RECORD_WARM_1, self_sha256 (the launch_fl canon rule), written by
    the variant's S_CAL arm at seed 2026 as S_CALIBRATION, theta_pre init_from, frozen keys exactly for S8, and an S
    CONTENT check mirroring launch_fl.check_s_content: S finite real > 0, n_norms an int > 0, a median rule, rounds_0based = 0..19."""
    bad = {}
    if obj.get("format") != S_FORMAT_WARM:
        bad["format"] = obj.get("format")
    if obj.get(SELF_KEY) != _canon_compact(obj):
        bad["self_sha256"] = "does not verify"
    if obj.get("arm") != S_CAL_WARM_ARM[variant] or obj.get("warm_variant") != variant:
        bad["arm / warm_variant"] = (obj.get("arm"), obj.get("warm_variant"))
    if obj.get("stage") != WARM_SELECTION_STAGE[variant]:
        bad["stage"] = obj.get("stage")
    if obj.get("seed") != 2026 or obj.get("label") != "S_CALIBRATION":
        bad["seed / label"] = (obj.get("seed"), obj.get("label"))
    S, n, rounds, rule = obj.get("S"), obj.get("n_norms"), obj.get("rounds_0based"), obj.get("rule")
    if isinstance(S, bool) or not isinstance(S, (int, float)) or not math.isfinite(S) or S <= 0:
        bad["S"] = S
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        bad["n_norms"] = n
    if not isinstance(rule, str) or not rule.startswith("median"):
        bad["rule"] = rule
    if not isinstance(rounds, list) or rounds != list(range(S_CAL_ROUNDS)) or \
            any(isinstance(r, bool) or not isinstance(r, int) for r in rounds):
        bad["rounds_0based"] = rounds
    why = init_from_problem(obj.get("init_from"))
    if why:
        bad["init_from"] = why
    if bool(obj.get("frozen_item_tables")) != (variant == "FA_WARM_FROZEN"):
        bad["frozen_item_tables"] = obj.get("frozen_item_tables")
    if not _hex64(obj.get("init_sha256")):
        bad["init_sha256"] = obj.get("init_sha256")
    if bad:
        raise AnaRefused(f"records.{S_RECORD_WARM_KEY[variant]} {rel}: not a verified {S_FORMAT_WARM} record of "
                         f"{S_CAL_WARM_ARM[variant]}: {bad} (FL-16; REV-FL-16 m5)")
    return {"file": rel, "file_sha256": file_sha256, "format": obj["format"], "S": float(S), "n_norms": n,
            "rule": obj.get("rule"), "seed": 2026, "label": obj["label"], "arm": obj["arm"], "warm_variant": variant,
            "stage": obj["stage"], "init_from": dict(obj["init_from"]), SELF_KEY: obj[SELF_KEY]}


def warm_run_problem(arm: str, e) -> Optional[str]:
    """REV-ANA-05 F1: why a VALIDATION warm entry WITHOUT a run folder (a REEVAL / rows-only entry, no same-seed run
    folder of its role in the block) is NOT bound to the registered 2-D warm selection, or None. It needs the index's
    warm_selection (the FL run's LAUNCH binding, warm_selection_problem against that run's arm_spec factors) and
    warm_run (the FL run's LAUNCH run_id / arm_spec_sha256 / seed), and its own rows must name that run (fl_run_id,
    arm_spec_sha256; v4int.reeval FL rows)."""
    wr, ws = e.warm_run, e.warm_selection
    if not isinstance(wr, Mapping) or not isinstance(ws, Mapping):
        return "no warm_selection / warm_run (the FL run's LAUNCH binding, recorded by tools/ana_block_index.py)"
    why = warm_selection_problem(arm, ws, {"warm_lr_factor": wr.get("warm_lr_factor", 1.0),
                                           "warm_server_lr_factor": wr.get("warm_server_lr_factor", 1.0)})
    if why:
        return why
    r = e.rows.get(PRIMARY) or {}
    bad = {}
    if not wr.get("run_id") or not _hex64(wr.get("arm_spec_sha256")):
        bad["warm_run"] = {k: wr.get(k) for k in ("run_id", "arm_spec_sha256")}
    if wr.get("seed") is not None and int(wr["seed"]) != e.seed:
        bad["warm_run.seed"] = wr.get("seed")
    if (r.get("fl_run_id"), r.get("arm_spec_sha256")) != (wr.get("run_id"), wr.get("arm_spec_sha256")):
        bad["rows fl_run_id / arm_spec_sha256"] = (r.get("fl_run_id"), r.get("arm_spec_sha256"))
    return f"{bad}" if bad else None


def require_warm_record(arm: str, st: str, entries: Sequence, rec: Optional[Mapping]) -> None:
    """REV-ANA-05 F3: a VALIDATION block with entries of a warm stage needs that stage's registered file in records."""
    if entries and rec is None:
        raise AnaRefused(f"ARM_IDENTITY - LAUNCH_BINDING_ONLY: the block has {arm} entries but not the registered "
                         f"FL_WARM_SELECTION_{st}.json in records ({WARM_SELECTION_RECORD[st]}); the VALIDATION block "
                         "needs the file (06M_B B2; REV-ANA-05 F3)")


def runs_bound_on_validation(validation: Optional[Mapping], st: str) -> set:
    """REV-ANA-05 F1: {(seed, run_id, arm_spec_sha256)} of the warm runs the VALIDATION_RESULT bound for stage st."""
    vst = ((((validation or {}).get("ana05") or {}).get("warm_selection") or {}).get("stages") or {}).get(st) or {}
    return {(r.get("seed"), r.get("run_id"), r.get("arm_spec_sha256")) for r in vst.get("bound_runs") or []}


def run_rows_problem(e, src_eid: str, rid, asha) -> Optional[str]:
    """REV-06O-ANA05 MINOR 1: why a warm entry bound through a run folder (its own, or the same-seed run folder of its
    role) is NOT that run: its evidence rows must name the folder's RUN_SUMMARY run_id / arm_spec_sha256 (launch_fl
    fl_rows fl_run_id / arm_spec_sha256); an equal warm_selection alone does not bind. None when they do."""
    r = e.rows.get(PRIMARY) or {}
    if (r.get("fl_run_id"), r.get("arm_spec_sha256")) == (rid, asha):
        return None
    return (f"its rows name the FL run {r.get('fl_run_id')!r} / {str(r.get('arm_spec_sha256'))[:12]}, not its run "
            f"folder {src_eid} ({rid!r} / {str(asha)[:12]}); an equal warm_selection alone does not bind "
            "(REV-06O-ANA05 MINOR 1)")


def warm_base_run_problem(arm: str, *names) -> Optional[str]:
    """DL 287 m3 (REV-FL-16 m3, ANA side): why a device arm's start / base is a *_WARM* FL run it may not take, or None.
    Only FT_FA_WARM / FT_FA_WARM_FROZEN take a warm base (FA_WARM / FA_WARM_FROZEN, checked by ft_warm_base_problem)."""
    if arm not in DEVICE_ARMS or arm in FT_WARM_BASE:
        return None
    hit = next((str(n) for n in names if "_WARM" in str(n or "").upper()), None)
    return None if hit is None else (f"{arm} starts from the warm FL run {hit!r}: an FT_FA / FT_C base is never a "
                                     "*_WARM* run (only FT_FA_WARM / FT_FA_WARM_FROZEN take an FA_WARM / "
                                     "FA_WARM_FROZEN base; DL 287 m3)")


def _bind_warm_arm(ents: Sequence, arm: str, st: str, records: Mapping, split: str, vruns: set, vwhere: str) -> dict:
    """The (a) binding of check_ana05 for one warm arm of stage st (FA_WARM / FA_WARM_FROZEN; ANA-06: every FL-16
    warm-line arm too): ONE registered file per arm, a run's other entries = its run folder's binding, the file in records
    = the binding (sha256 / self_sha256 / factors / theta_pre); on VALIDATION the file is required (F3) and every entry
    is bound to a run (F1): a run folder whose RUN_SUMMARY run_id / arm_spec_sha256 the entry's OWN rows must name
    (fl_run_id / arm_spec_sha256; REV-06O-ANA05 MINOR 1), or a REEVAL entry's warm_run (warm_run_problem); on TEST every
    entry's rows name a run bound on VALIDATION (vruns)."""
    allw = [e for e in ents if e.arm == arm]
    bound = [e for e in allw if e.warm_selection is not None]
    shas = sorted({str(e.warm_selection.get("sha256")) for e in bound})
    if len(shas) > 1:
        raise AnaRefused(f"ARM_IDENTITY - the {arm} entries are bound to {len(shas)} different warm selections "
                         f"{[x[:12] for x in shas]} (one registered FL_WARM_SELECTION_{st}.json; 06M_B B2)")
    runs = {(e.role, e.seed): e for e in allw if e.run_dir_rel}
    for e in allw:
        r = runs.get((e.role, e.seed))
        if r is not None and e is not r and e.warm_selection != r.warm_selection:
            raise AnaRefused(f"{e.eid}: ARM_IDENTITY - its warm_selection differs from its run's ({r.eid}; 06M_B B2)")
    rec = records.get(WARM_SELECTION_RECORD[st])
    if rec is not None:
        for e in bound:
            ws = e.warm_selection
            got = (ws.get("sha256"), ws.get("self_sha256"), float(ws.get("winner_factor")),
                   float(ws.get("winner_server_lr_factor")))
            want = (rec["file_sha256"], rec["self_sha256"], rec["winner_factor"], rec["winner_server_lr_factor"])
            if got != want:
                raise AnaRefused(f"{e.eid}: ARM_IDENTITY - the run's warm selection {got} is not the registered "
                                 f"records.{WARM_SELECTION_RECORD[st]} {want} (06M_B B2)")
            init = e.meta_auto.get("init_from")
            if init is not None and dict(init) != rec["init_from"]:
                raise AnaRefused(f"{e.eid}: ARM_IDENTITY - the run's theta_pre is not the warm selection's (06K §2)")
    bound_runs = {}
    if split == "VALIDATION":
        require_warm_record(arm, st, allw, rec)                    # REV-ANA-05 F3
        for e in allw:                                             # REV-ANA-05 F1
            src = e if e.run_dir_rel else runs.get((e.role, e.seed))
            if src is not None:
                rid, asha = src.meta_auto.get("run_id"), src.meta_auto.get("arm_spec_sha256")
                if not rid or not _hex64(asha):
                    raise AnaRefused(f"{e.eid}: ARM_IDENTITY - its run folder {src.eid} records no RUN_SUMMARY "
                                     "run_id / arm_spec_sha256: the warm run cannot be bound (REV-ANA-05 F1)")
                why = run_rows_problem(e, src.eid, rid, asha)     # REV-06O-ANA05 MINOR 1: the rows name that run
                if why:
                    raise AnaRefused(f"{e.eid}: ARM_IDENTITY - {why}")
            else:
                why = warm_run_problem(arm, e)
                if why:
                    raise AnaRefused(f"{e.eid}: ARM_IDENTITY - an {arm} entry without a run folder is not bound to "
                                     f"the registered 2-D FL_WARM_SELECTION_{st}.json: {why} (06M_B B2; REV-ANA-05 "
                                     "F1)")
                rid, asha = (e.warm_run or {}).get("run_id"), (e.warm_run or {}).get("arm_spec_sha256")
            bound_runs[(e.seed, rid, asha)] = None
    else:                                                          # TEST: the runs bound on VALIDATION
        for e in allw:
            r = e.rows.get(PRIMARY) or {}
            key = (e.seed, r.get("fl_run_id"), r.get("arm_spec_sha256"))
            if key not in vruns:
                raise AnaRefused(f"{e.eid}: ARM_IDENTITY - on TEST an {arm} entry must be the run bound on "
                                 f"VALIDATION (records.VALIDATION_RESULT {vwhere}); its rows name "
                                 f"{key[1]!r} / {str(key[2])[:12]} (REV-ANA-05 F1)")
            bound_runs[key] = None
    return {"arm": arm, "entries_bound": sorted(e.eid for e in allw),
            "warm_selection_sha256": shas[0] if shas else None,
            "bound_runs": [{"seed": k[0], "run_id": k[1], "arm_spec_sha256": k[2]}
                           for k in sorted(bound_runs, key=lambda k: (k[0], str(k[1])))],
            "record": (None if rec is None else {k: rec[k] for k in ("file", "file_sha256", "self_sha256",
                                                                     "winner_factor", "winner_server_lr_factor",
                                                                     "winner_label")}),
            "status": ("NO_ENTRY" if not allw else "BOUND_AND_FILE_VERIFIED" if split == "VALIDATION" else
                       "BOUND_TO_VALIDATION_RUNS" + (" + FILE_VERIFIED" if rec is not None else ""))}


def fa_warm_star(records: Mapping, split: str, validation: Optional[Mapping]) -> dict:
    """06N N2 (mechanical, CALIB only): FA_WARM* from the two registered FL_WARM_SELECTION_S7|S8.json files — the higher
    CALIB value of the selected cell, |S8 - S7| <= 0.0005 -> S8. UNDETERMINED (every 06N row NOT_AVAILABLE) unless both
    files carry a finite winner value. On TEST the VALIDATION_RESULT's choice is used; files given there must agree."""
    cal = records.get("_WARM_WINNER_CALIB") or {}
    out = {"rule": STAR_RULE, "tie_band": STAR_TIE, "calib_winner": {st: dict(cal[st]) for st in sorted(cal)},
           "label": "CALIB values of the registered S7 / S8 selections (never MAIN)"}
    v7, v8 = (cal.get("S7") or {}).get("value"), (cal.get("S8") or {}).get("value")
    if v7 is not None and v8 is not None:
        st = "S8" if (abs(v8 - v7) <= STAR_TIE or v8 > v7) else "S7"
        out.update(status="DETERMINED_FROM_FILES", stage=st, tie=abs(v8 - v7) <= STAR_TIE,
                   files={s: (records.get(WARM_SELECTION_RECORD[s]) or {}).get("file_sha256") for s in ("S7", "S8")})
    else:
        out.update(status="UNDETERMINED", stage=None,
                   reason="needs both registered FL_WARM_SELECTION_S7.json and _S8.json (records WARM_SELECTION_S7 / "
                          "_S8) with a finite CALIB value of the selected cell: "
                          + "; ".join(f"{s}: {(cal.get(s) or {}).get('reason') or 'file not in records'}"
                                      for s in ("S7", "S8") if (cal.get(s) or {}).get("value") is None))
    if split == "TEST":
        vst = (((validation or {}).get("ana06") or {}).get("fa_warm_star") or {})
        if out["status"] == "DETERMINED_FROM_FILES" and vst.get("stage") != out["stage"]:
            raise AnaRefused(f"FA_WARM*: the TEST records choose {out['stage']}, the VALIDATION_RESULT chose "
                             f"{vst.get('stage')!r} (06N N2)")
        if out["status"] != "DETERMINED_FROM_FILES":
            out.update(status=("FROM_VALIDATION_RESULT" if vst.get("stage") else "UNDETERMINED"),
                       stage=vst.get("stage"))
    out["arm"] = {"S7": "FA_WARM", "S8": "FA_WARM_FROZEN"}.get(out["stage"])
    return out


def check_ana06(blk: SimpleNamespace, records: Mapping, split: str = "VALIDATION",
                validation: Optional[Mapping] = None) -> dict:
    """ANA-06 cross-entry bindings (both --check-only and the full run; refusal = exit 2): every FL-16 warm-line entry is
    bound like FA_WARM(_FROZEN) (_bind_warm_arm) to its variant's registered S7 / S8 file and to its FL run (TEST: a run
    bound on VALIDATION, records.VALIDATION_RESULT ana06.warm_line.arms[arm].bound_runs); every warm entry of a stage
    names ONE file; a registered warm S record binds the S (cfg.dp.clip_norm) of every Poisson warm-line run of its
    variant; FA_WARM* (06N N2) is chosen mechanically from the two files."""
    ents = [e for e in blk.entries if not e.historical]
    arms = {}
    for arm in WARM_LINE_ARMS:
        if not any(e.arm == arm for e in ents):
            continue
        st = warm_stage_of(arm)
        vr = {(r.get("seed"), r.get("run_id"), r.get("arm_spec_sha256")) for r in
              ((((((validation or {}).get("ana06") or {}).get("warm_line") or {}).get("arms") or {}).get(arm) or {})
               .get("bound_runs") or [])}
        arms[arm] = dict(_bind_warm_arm(ents, arm, st, records, split, vr, f"ana06 warm_line bound_runs of {arm}"),
                         stage=st, launch_arm=WARM_LINE_ARMS[arm][0], base_arm=WARM_LINE_ARMS[arm][1],
                         variant=WARM_LINE_ARMS[arm][2], frozen=WARM_LINE_ARMS[arm][2] == "FA_WARM_FROZEN")
    for st in ("S7", "S8"):                                    # one registered file per stage across every warm arm
        shas = sorted({str(e.warm_selection.get("sha256")) for e in ents
                       if warm_stage_of(e.arm) == st and e.warm_selection is not None})
        if len(shas) > 1:
            raise AnaRefused(f"ARM_IDENTITY - the {st} warm entries are bound to {len(shas)} different warm selections "
                             f"{[x[:12] for x in shas]} (one registered FL_WARM_SELECTION_{st}.json; ANA-06)")
    srec = {}
    for variant, key in sorted(S_RECORD_WARM_KEY.items()):
        rec = records.get(key)
        runs = [e for e in ents if e.arm in WARM_LINE_ARMS and WARM_LINE_ARMS[e.arm][2] == variant
                and WARM_LINE_ARMS[e.arm][1] in POISSON_BASE and e.run_dir_rel]
        for e in runs if rec is not None else []:
            clip = e.meta_auto.get("dp_clip")
            if clip is None or not _eq(clip, rec["S"]):
                raise AnaRefused(f"{e.eid}: ARM_IDENTITY - the run's clip norm {clip!r} is not the registered "
                                 f"records.{key} S {rec['S']} (FL-16 warm S record)")
            if dict(e.meta_auto.get("init_from") or {}) != rec["init_from"]:
                raise AnaRefused(f"{e.eid}: ARM_IDENTITY - the run's theta_pre is not the warm S record's (FL-16)")
        srec[key] = {"status": ("NO_RECORD" if rec is None else "BOUND" if runs else "RECORD_ONLY"),
                     "runs_bound": sorted(e.eid for e in runs) if rec is not None else [],
                     "record": None if rec is None else {k: rec[k] for k in ("file", "file_sha256", "S", "n_norms",
                                                                              "arm", SELF_KEY)}}
    return {"fa_warm_star": fa_warm_star(records, split, validation),
            "warm_line": {"arms": arms, "rule": "FL-16 labels -> block arms (WARM_LINE_ARMS); each bound like "
                                                "FA_WARM(_FROZEN) to its variant's S7 / S8 file and FL run"},
            "s_record_warm": srec}


def check_ana05(blk: SimpleNamespace, records: Mapping, split: str = "VALIDATION",
                validation: Optional[Mapping] = None) -> dict:
    """ANA-05 cross-entry bindings (both --check-only and the full run; refusal = exit 2):
    (a) 06M_B B2: every FA_WARM / FA_WARM_FROZEN entry (every role) that carries a warm_selection agrees with the
        other entries of its stage (ONE registered file per stage) and a run's other entries with its run-folder entry;
        when the file is in records, the file's sha256 / self_sha256 / winner factors / theta_pre equal the binding.
        REV-ANA-05 F3: on VALIDATION the stage's file is REQUIRED in records whenever the block has a warm entry
        (LAUNCH_BINDING_ONLY refuses). REV-ANA-05 F1: on VALIDATION every warm entry is bound to a run - a run folder
        (RUN_SUMMARY run_id / arm_spec_sha256), the same-seed run folder of its role, or (a REEVAL / rows-only entry)
        warm_run_problem None; on TEST every warm entry's rows (fl_run_id / arm_spec_sha256, seed) must be a run bound
        on VALIDATION (records.VALIDATION_RESULT ana05.warm_selection.stages[st].bound_runs).
    (b) 06M_B B4: every FT_FA_WARM(_FROZEN) entry's base (start_base) is the block's base-arm entry of the SAME seed and
        role MAIN: run id + arm-spec sha256 of its run dir (RUN_SUMMARY) or of its REEVAL rows (fl_run_id /
        arm_spec_sha256), the same theta_pre as that run (and as PRE's pre_init when the block has one); an FT_FA_WARM
        without its base entry refuses (the paired delta and the base identity need it)."""
    ents = [e for e in blk.entries if not e.historical]
    stages = {}
    for arm, st in WARM_SELECTION_STAGE.items():
        stages[st] = _bind_warm_arm(ents, arm, st, records, split, runs_bound_on_validation(validation, st)
                                    if split != "VALIDATION" else set(), "ana05 bound_runs of " + st)
    pre = [e.pre_init for e in ents if e.arm == "PRE" and isinstance(e.pre_init, Mapping)]
    bases = []
    for e in ents:
        if e.arm not in FT_WARM_BASE:
            continue
        sb, barm = e.start_base or {}, FT_WARM_BASE[e.arm]
        cands = [b for b in ents if b.arm == barm and b.seed == e.seed and b.role == "MAIN"]
        ident = None
        for b in sorted(cands, key=lambda x: x.run_dir_rel is None):
            if b.run_dir_rel and b.meta_auto.get("arm_spec_sha256"):
                ident = (b, b.meta_auto.get("run_id"), b.meta_auto.get("arm_spec_sha256"), b.meta_auto.get("init_from"))
                break
            r = b.rows.get(PRIMARY) or {}
            if r.get("fl_run_id") and r.get("arm_spec_sha256"):
                ident = (b, r.get("fl_run_id"), r.get("arm_spec_sha256"), None)
                break
        if ident is None:
            raise AnaRefused(f"{e.eid}: ARM_IDENTITY - no {barm} MAIN entry of seed {e.seed} with its run identity (run "
                             f"dir RUN_SUMMARY run_id / arm_spec_sha256, or REEVAL rows fl_run_id / arm_spec_sha256): "
                             f"{e.arm}'s base cannot be checked and FT_FA_WARM - FA_WARM is paired per seed (06M_B B4)")
        b, rid, asha, init = ident
        if (sb.get("run_id"), sb.get("base_arm_spec_sha256")) != (rid, asha):
            raise AnaRefused(f"{e.eid}: ARM_IDENTITY - its base {sb.get('run_id')!r} / "
                             f"{str(sb.get('base_arm_spec_sha256'))[:12]} is not the block's {b.eid} run {rid!r} / "
                             f"{str(asha)[:12]} (06M_B B4)")
        if init is not None and dict(init) != dict(sb.get("init_from") or {}):
            raise AnaRefused(f"{e.eid}: ARM_IDENTITY - its base's theta_pre differs from {b.eid}'s init_from (06K §2)")
        if pre and any((sb.get("init_from") or {}).get("checkpoint_sha256") != p.get("checkpoint_sha256") for p in pre):
            raise AnaRefused(f"{e.eid}: ARM_IDENTITY - its base did not start from the block's PRE theta_pre (06K §2)")
        bases.append({"entry": e.eid, "base_entry": b.eid, "base_run_id": rid, "base_checkpoint": sb.get("checkpoint"),
                      "base_checkpoint_rule": WARM_BASE_FILES.get(sb.get("checkpoint"))})
    return {"warm_selection": stages, "ft_warm_bases": bases, "rule": WARM_SELECTION_RULE}


def check_validation_result(val: Optional[Mapping], script_sha: str, contrasts: Mapping) -> dict:
    """A-F8 (iv): the TEST run needs the VALIDATION preflight result of the SAME frozen script, library and contrasts
    (the contrast CONTENT: the VALIDATION run reads the pre-lock JSON, the TEST run the ALL_METHODS_LOCK file)."""
    if val is None:
        raise AnaRefused("--split TEST needs records.VALIDATION_RESULT (the VALIDATION preflight of the same script)")
    if (val.get("script") or {}).get("sha256") != script_sha:
        raise AnaRefused("VALIDATION_RESULT was written by another retention_report.py (script sha256 differs)")
    if val.get("library_files_sha256") != LIB.files_sha256:
        raise AnaRefused("VALIDATION_RESULT used other library files (library_files_sha256 differs)")
    if (val.get("inputs") or {}).get("contrasts_content_sha256") != contrasts["content_sha256"]:
        raise AnaRefused("VALIDATION_RESULT used another confirmatory registry (contrasts_content_sha256 differs)")
    return {"self_sha256": val.get(SELF_KEY), "script_sha256": script_sha,
            "contrasts_content_sha256": contrasts["content_sha256"], "library_files_sha256_equal": True}


def shift_indicator(tab: Mapping, split: str, val: Optional[dict]) -> dict:
    """06B §7(b): the VALIDATION -> TEST change of C_BIG (descriptive; different decision sets, no paired CI)."""
    if split != "TEST" or val is None:
        return {"status": "NOT_APPLICABLE" if split != "TEST" else "NO_VALIDATION_RESULT"}
    out = {"label": "DESCRIPTIVE (shift indicator, 06B §7b)", "by_rule": {}}
    for rule in RULES:
        now = next((r for r in tab.get(rule, []) if r["arm"] == REF), None)
        was = next((r for r in val.get("retention", {}).get("MAIN", {}).get(rule, []) if r["arm"] == REF), None)
        if not now or not was:
            out["by_rule"][rule] = {"status": "NOT_AVAILABLE"}
            continue
        per = {}
        for seed, e in now["entries"].items():
            t = e["values"]["mrr@20"]
            v = (was["entries"].get(seed) or {}).get("values", {}).get("mrr@20")
            if isinstance(v, (int, float)) and isinstance(t, (int, float)) and v:
                per[seed] = {"validation": v, "test": t, "delta": t - v, "test_over_validation_pct": 100.0 * t / v}
        out["by_rule"][rule] = per
    return out


def sanity_trigger(eng: Engine, book: Book, tab: Mapping) -> dict:
    """Amendment 04 §1 / 06E §6: "C_SMALL 6.0 value < 0.5 x C_BIG" = C_SMALL CONTROLLED_BUDGET vs C_BIG PRACTICAL_BEST
    (per seed, point; the seed mean of the per-seed ratios), printed beside the PB / PB ratio (a trigger, not a verdict)."""
    cb, pb = book.get("MAIN", REF_SMALL, "CONTROLLED_BUDGET"), book.get("MAIN", REF, "PRACTICAL_BEST")
    per = {}
    for seed in sorted(s for s in cb if s is not None and s in pb):
        m = eng.point(cb[seed], full_view("MAIN", cb[seed].subset_keys), PRIMARY)
        c = eng.point(pb[seed], full_view("MAIN"), PRIMARY)
        if not (is_undef(m) or is_undef(c)) and float(c) > 0:
            per[str(seed)] = {"C_SMALL_CB": m, "C_BIG_PB": c, "R": LIB.Q.retained_percent(m, c)}
    pbpb = _seed_mean_R(tab, REF_SMALL, CURVE_RULE, "vs_C_BIG").get("point")
    if not per:
        return {"status": "NOT_AVAILABLE", "R_C_SMALL_PB_vs_C_BIG_PB": pbpb}
    R = float(sum(v["R"] for v in per.values()) / len(per))
    return {"R_C_SMALL_CB_vs_C_BIG_PB": R, "per_seed": per, "fired": bool(R < 50.0),
            "R_C_SMALL_PB_vs_C_BIG_PB": pbpb,
            "rule": "C_SMALL 6.0 (CONTROLLED_BUDGET) value < 0.5 x C_BIG (PRACTICAL_BEST) (amendment 04 §1; 06E §6)"}


def panels(eng: Engine, book: Book) -> dict:
    return {role: retention_table(eng, book, role) for role in sorted(eng.ctx) if role != "MAIN"
            and any(r == role for (r, _a, _x) in book.by) and book.get(role, REF, CURVE_RULE)}


def historical(book: Book, tab_ctx: Mapping) -> list:
    """Point-only R of the historical references vs C_BIG: a seeded entry against the same seed, a rule-free baseline
    against the seed mean of the per-seed ratios over every C_BIG seed (the seeds are printed; A-F17)."""
    out = []
    for e in book.hist:
        den = book.get(e.role, REF, e.rule if e.rule != RULE_FREE else CURVE_RULE)
        v = float(e.rows[PRIMARY]["value"])
        if e.seed is not None:
            ds = {e.seed: den[e.seed]} if e.seed in den else {}
        else:
            ds = {s: d for s, d in den.items() if s is not None}
        per = {str(s): LIB.Q.retained_percent(v, float(d.rows[PRIMARY]["value"])) for s, d in sorted(ds.items())}
        R = (undef("NO_REFERENCE") if not per else float(sum(per.values()) / len(per))
             if not any(is_undef(x) for x in per.values()) else undef("UNDEFINED_REFERENCE"))
        out.append({"entry": e.eid, "label": "HISTORICAL_REFERENCE", "value": v, "R_vs_C_BIG_point": R,
                    "C_BIG_seeds": [int(s) for s in per], "R_per_seed": per,
                    "C_BIG_rule": e.rule if e.rule != RULE_FREE else CURVE_RULE,
                    "note": "comparison key differs (E9.1): never a same-conditions effect; seed mean of per-seed R"})
    return out


def check_bootstrap_w(eng: Engine, entries: Sequence[Entry]) -> dict:
    """E7.2 (A-F15): a row's non-null bootstrap_W_sha256 must be the W of the entry's own view (v4train.bootstrap_ci
    plan of the role's E2E users, or of its users_subset)."""
    seen = {}
    for e in entries:
        if e.historical or not e.ranked:
            continue
        ws = {r.get("bootstrap_W_sha256") for _w, rows in _entry_rowsets(e) for r in rows.values()} - {None}
        if not ws:
            continue
        want = eng.meta(full_view(e.role, e.subset_keys))["W_sha256"]
        if ws != {want}:
            raise AnaRefused(f"{e.eid}: bootstrap_W_sha256 {sorted(ws)} is not the W of its view {want} (E7.2; A-F15)")
        seen[e.eid] = want
    return seen


def plan_pairs(eng: Engine, book: Book, contrasts: Mapping, star_arm: Optional[str] = None) -> dict:
    """A-F11: every pairing the analysis will make (retention tables of every role, the loaded registry's contrasts and
    the registered rows it does not carry (06K, 06L-C) with their registered seeds, populations and shared
    denominators, the K16 fallback, the registered DESCRIPTIVE S8-S7, the LAST anchor pairs (06L-C §C3), the FA_1024
    ratios, the personalisation deltas, the strata / bin sub-populations) through check_pairs, and every block-level
    denominator through require_nonzero_denominator (E8.3). Prints nothing numeric."""
    n_pairs, n_den = 0, 0

    def one(pairs, vk, strict, name, shared=False):
        nonlocal n_pairs, n_den
        if not pairs:
            return
        check_pairs(pairs, shared)
        n_pairs += len(pairs)
        if strict:
            for _num, den in pairs:
                require_nonzero_denominator(eng.point(den, vk, PRIMARY), f"{name} {den.eid}")
                n_den += 1

    roles = sorted({r for (r, _a, _x) in book.by})
    for role in roles:
        for rule in RULES:
            for arm in book.arms(role):
                for ref in (REF, REF_SMALL):
                    pairs = book.pairs(role, arm, ref, rule)
                    if pairs:
                        vk = _vkey_for(pairs, role)
                        one(pairs, vk, vk[1] == "ALL", f"{arm}/{ref} {rule}")
                        sub = [(n, d) for n, d in pairs if n.ranked and d.ranked and n.subset_keys is None]
                        one(sub, vk, False, "sub-population")
                if arm != "LAST":
                    anc = book.pairs(role, arm, "LAST", rule)
                    one([p for p in anc if is_anchor(p[1])], _vkey_for(anc, role) if anc else None, False,
                        f"{arm} - LAST {rule}")
    for c in list(contrasts["items"]) + extra_items(contrasts) + descriptive_items(contrasts):
        reg = tuple(REGISTERED_SEEDS.get(c["id"], SEEDS))
        role = c.get("population") or "MAIN"
        shared = c["id"] in CONTRAST_SHARED_DEN
        for rule in RULES:
            dens = [c["den"]] + ([CONTRAST_FALLBACK[c["den"]]] if c["den"] in CONTRAST_FALLBACK else [])
            for den in dens:
                sh = shared and den == CONTRAST_SHARED_DEN.get(c["id"])
                pairs = book.pairs(role, c["num"], den, rule, shared=sh)
                for sel in ([p for p in pairs if _pair_seed(p, sh) in reg], pairs):
                    if sel:
                        one(sel, _vkey_for(sel, role), c["kind"] == "retained", c["id"], sh)
    for rule in RULES:
        for arm in EPS_ARMS[1:]:
            pairs = book.pairs("MAIN", arm, "FA_1024", rule)
            one(pairs, full_view("MAIN"), True, f"{arm}/FA_1024")
        for a, b in PERSONALISATION + PRODUCTION_STACK:                # + ANA-05 (06M_B B4)
            one(book.pairs("MAIN", a, b, rule), full_view("MAIN"), False, f"{a} - {b}")
        one(book.pairs("MAIN", *FAMILY_PAIR, rule), full_view("MAIN"), True, "GRU_D128/C_SMALL")
        one(book.pairs("MAIN", "DP8_FROZEN", "FA_WARM_FROZEN", rule), full_view("MAIN"), True, "DP8_FROZEN/S8")
    with v4_tables():                                                  # ANA-06: the 06P set (v1-v3 loaded)
        for c, _tier, an in ([] if contrasts.get("format") == CONTRAST_FORMAT_V4 else n06_items(star_arm)):
            pk = an or c["id"]
            reg = tuple(REGISTERED_SEEDS.get(pk, SEEDS))
            for rule in RULES:
                sh = pk in CONTRAST_SHARED_DEN and c["den"] == CONTRAST_SHARED_DEN[pk]
                pairs = book.pairs("MAIN", c["num"], c["den"], rule, shared=sh)
                for sel in ([p for p in pairs if _pair_seed(p, sh) in reg], pairs):
                    if sel:
                        one(sel, _vkey_for(sel, "MAIN"), c["kind"] == "retained", c["id"], sh)
    return {"n_pairs_checked": n_pairs, "n_block_denominators_checked": n_den,
            "bootstrap_W_checked": len(check_bootstrap_w(eng, [e for ents in book.by.values() for e in ents.values()]))}


# ------------------------------------------------------------------------------------------------ assemble
def analyse(blk: SimpleNamespace, contrasts: Mapping, records: Mapping, split: str) -> dict:
    eng = Engine(blk.ctx)
    book = Book(blk.entries)
    w_checked = check_bootstrap_w(eng, blk.entries)
    tab = retention_table(eng, book, "MAIN")
    tab_all = dict(tab, _vs_FA_1024=vs_fa1024(eng, book))
    conf = confirmatory(eng, book, contrasts)
    extra = extra_registered(eng, book, contrasts)
    desc = registered_descriptive(eng, book, contrasts)
    holm = holm_band(eng, conf, contrasts)
    warm = warm_start(eng, book, contrasts, conf, extra)
    anchor = last_anchor(eng, book, "MAIN")
    axis = bytes_axis(book, blk.ctx, records, tab)
    bins, ev_bins, bin_counts = curve_bins(eng, book)
    trunc, trunc_ev = trunc_curves(records.get("TRUNC_ROWS") or [], blk.key["values"], records.get("TRUNC_PER_USER"))
    calib = calib_disjoint(eng, book, list(contrasts["items"]) + extra_items(contrasts), getattr(blk, "calib", None),
                           tab)
    stamp_claims(conf, extra, desc)
    bind = getattr(blk, "ana05", None)
    if bind is None:
        bind = check_ana05(blk, records, split, records.get("VALIDATION_RESULT"))
    a5 = ana05(eng, book, bind)
    b6 = getattr(blk, "ana06", None)
    if b6 is None:
        b6 = check_ana06(blk, records, split, records.get("VALIDATION_RESULT"))
    a6 = ana06(eng, book, b6, conf, a5, contrasts.get("format"))
    return {
        "retention": {"MAIN": tab},
        "retention_vs_FA_1024": tab_all["_vs_FA_1024"],
        "confirmatory": conf,
        "holm_band": holm,
        "extra_registered": extra,
        "registered_descriptive": desc,
        "warm_start": warm,
        "last_anchor": anchor,
        "device_only_envelope": device_envelope(eng, book, tab, anchor),
        "personalisation": personalisation(eng, book),
        "privacy_ladder": privacy_ladder(book, tab),
        "privacy_matrix": privacy_matrix(book, tab, axis),
        "calib_disjoint": calib,
        "target_strata": target_strata(eng, book, "MAIN"),
        "no_ft_unseen_val": no_ft_references(eng, book, records),
        "family_comparison": family_comparison(eng, book),
        "frozen_branch": frozen_branch(eng, book),
        "claim_register": claim_register(contrasts),
        "curves": {"bytes": curve_bytes(eng, book), "epsilon": curve_epsilon(book, tab_all, records.get("DP_WOR")),
                   "model_size": curve_size(book, tab), "strata": curve_strata(eng, book, records.get("STRATA")),
                   "support_bins": bins, "support_bin_counts": bin_counts, "events_to_x_bins": ev_bins,
                   "trunc": trunc, "trunc_events_to_x": trunc_ev, "trunc_persistence_rule": PERSISTENCE_RULE,
                   "bytes_axis": axis, "rollout": a5["rollout_quality"]["curve"]},
        "panels": panels(eng, book),
        "historical_reference": historical(book, tab),
        "shift_indicator": shift_indicator(tab, split, records.get("VALIDATION_RESULT")),
        "sanity_trigger": sanity_trigger(eng, book, tab),
        "ana05": dict(a5, rollout_quality={k: v for k, v in a5["rollout_quality"].items() if k != "curve"}),
        "ana06": a6,
        "_bootstrap_W_checked": w_checked,
    }


def build_result(blk, contrasts, records, split, analysis, *, script_sha: str, test_gate: Optional[Mapping] = None,
                 validation_check: Optional[Mapping] = None) -> dict:
    analysis = dict(analysis)
    w_checked = analysis.pop("_bootstrap_W_checked", {})
    rec_out = {k: v for k, v in records.items() if k not in ("STRATA", "TRUNC_ROWS", "TRUNC_FILES",
                                                              "TRUNC_PER_USER", "VALIDATION_RESULT")
               and not k.startswith("_")}                          # ANA-06: internal keys (FA_WARM* inputs -> ana06)
    if "STRATA" in records:
        rec_out["STRATA"] = {"registry_self_sha256": records["STRATA"].get("registry_self_sha256"),
                             "source_manifest_id": records["STRATA"].get("source_manifest_id"),
                             "cuts": records["STRATA"].get("cuts"), "n_users": records["STRATA"].get("n_users")}
    if records.get("VALIDATION_RESULT"):
        rec_out["VALIDATION_RESULT"] = {SELF_KEY: records["VALIDATION_RESULT"][SELF_KEY]}
    rec_out["TRUNC_ROWS"] = {"n_rows": len(records.get("TRUNC_ROWS") or []), "files": records.get("TRUNC_FILES") or []}
    return {
        "schema": SCHEMA, "split": split, "task": "ANA-01 (+06B addendum) + ANA-02 (REV-ANA-01 fixes, 06D §3-§5) + ANA-03 (K17-K21, 06I §7b, F9) + "
                "GATE-B (B3, B4) + ANA-04 (06L-C C3, C4, C8, C11-C17) + ANA-05 (C_FULL s2027, 06M, 06M_B B2 / B4 / B5, "
                "S8 comm)",
        "script": {"file": "tools/retention_report.py", "sha256": script_sha},
        "library_files_sha256": LIB.files_sha256,
        "inputs": {"files_sha256": dict(sorted(blk.files.items())), "contrasts_sha256": contrasts["file_sha256"],
                   "contrasts_format": contrasts["format"], "contrasts_v1_dry_check": contrasts["v1_dry_check"],
                   "contrasts_content_sha256": contrasts["content_sha256"],
                   "contrasts_self_sha256_status": contrasts.get("self_sha256_status"),
                   "roles": {r: dict(c.record, manifest_id=c.manifest.manifest_id, n_rows=c.manifest.n,
                                     eval_ordered_decision_id_sha256=c.manifest.hashes["eval_ordered_decision_id_sha256"])
                             for r, c in sorted(blk.ctx.items())},
                   "test_gate": test_gate, "validation_result_check": validation_check},
        "checks": {"comparison_key": blk.key, "rank_row_max_abs_diff": blk.rank_row_max_abs_diff,
                   "rank_row_tolerance": ROW_VALUE_TOL, "pairing": "same checkpoint rule + same seed (06A §7)",
                   "zero_denominator": "refused at block level (E8.3); UNDEFINED in sub-populations",
                   "rank_coverage": "every RANKABLE row of every evaluated view has rank >= 1 (A-F6)",
                   "bootstrap_W_sha256_checked": w_checked,
                   "entries": [{"entry": e.eid, "rule_source": e.rule_source, "ranked": e.ranked,
                                "historical_reference": e.historical, "n_marks": len(e.marks),
                                "comparison_block": e.block, "users_subset": e.subset_label,
                                "notes": e.notes} for e in blk.entries]},
        "bootstrap": {"n_resamples": LIB.B.N_RESAMPLES, "seed": LIB.B.SEED, "unit": "user (eval_user_key)",
                      "statistic": "macro_e2e", "ratios": "bootstrapped directly per resample (E7.5/E8.4)",
                      "seed_mean": "mean over seeds of the per-seed statistic, per resample on the same W"},
        "labels": {"CONFIRMATORY": [c["id"] for c in contrasts["items"]], "default": "DESCRIPTIVE",
                   "confirmatory_rule_map": dict(CONFIRMATORY_RULE),
                   "default_confirmatory_rule": DEFAULT_CONFIRMATORY_RULE,
                   "hierarchy": {c["id"]: HIERARCHY.get(c["id"], DEFAULT_HIERARCHY) for c in contrasts["items"]},
                   "registered_seeds": {c["id"]: list(REGISTERED_SEEDS.get(c["id"], SEEDS)) for c in contrasts["items"]},
                   "multiplicity": contrasts["multiplicity"], "dp_disclosure": DISCLOSURE_06D_3,
                   "registered_descriptive": [c["id"] for c in descriptive_items(contrasts)],
                   "registered_not_in_file": [c["id"] for c in extra_items(contrasts)],
                   "superseded_note": contrasts.get("superseded_note"),
                   "algebraic_dependencies": [dict(d, **({"note": DEPENDENCY_NOTES[d["id"]]} if d["id"] in
                                                          DEPENDENCY_NOTES else {})) for d in ALGEBRAIC_DEPENDENCIES],
                   "post_validation": POST_VALIDATION_NOTE},
        "records": rec_out,
        **analysis,
        "disclosures": {"06A_9_verbatim": DISCLOSURES_06A_9, "06B_7_verbatim": DISCLOSURES_06B_7,
                        "06B_1": DISCLOSURE_06B_1, "06D_3_verbatim": REGISTERED_06D_3,
                        "06D_4_verbatim": REGISTERED_06D_4, "notes": list(DISCLOSURE_NOTES),
                        "ladder_note": LADDER_NOTE, "06K_DL206": list(DISCLOSURES_06K),
                        "bytes_charged_06I_7b": DISCLOSURE_BYTES_06I_7B, "06L_C_verbatim": dict(REGISTERED_06LC),
                        "ana04_methods": list(DISCLOSURES_ANA04), "ana05_methods": list(DISCLOSURES_ANA05)},
    }


# ------------------------------------------------------------------------------------------------ markdown / csv
def _num(x, nd=6):
    if isinstance(x, Mapping) and x.get("status") == "UNDEFINED":
        return f"UNDEF({x.get('reason')})"
    if isinstance(x, Mapping) and "status" in x:
        return str(x["status"])
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if x.is_integer() and abs(x) >= 1000:
            return str(int(x))
        return f"{x:.{nd}f}"
    return str(x)


def _ci(x, nd=2):
    if isinstance(x, list) and len(x) == 2:
        return f"[{x[0]:.{nd}f}, {x[1]:.{nd}f}]"
    return _num(x, nd)


def _side(stat: Mapping, ref: float) -> str:
    ci = stat.get("ci95") if isinstance(stat, Mapping) else None
    if not (isinstance(ci, list) and len(ci) == 2):
        return "-"
    if ci[0] > ref:
        return "above"
    if ci[1] < ref:
        return "below"
    return "includes"


def _seed_R(stat: Mapping, seed: str) -> str:
    """The per-seed R [CI] of a seed row (a seed-free baseline shows every paired seed)."""
    per = stat.get("per_seed") or []
    hit = [p for p in per if str(p["seed"]) == seed] or (per if seed == "None" else [])
    return "; ".join(f"{_num(p.get('point'), 2)} {_ci(p.get('ci95'))}" for p in hit) or "-"


def _table(head: Sequence[str], rows: Sequence[Sequence]) -> List[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


CONF_HEAD = ["id", "contrast", "question", "hierarchy", "rule", "label", "seeds", "seeds status", "point", "95% CI",
             "CI vs 100/0", "users", "decisions", "harmed share", "seed spread (SEED_SENSITIVE)", "Holm band",
             "claim key"]


def _conf_row(c: Mapping, rule: str, b: Mapping, label: str, bold: bool) -> list:
    ref = 100.0 if c["kind"] == "retained" else 0.0
    nd = 2 if c["kind"] == "retained" else 6
    ss = b.get("seed_stats") or {}
    spread = ("-" if ss.get("seed_spread") is None else
              f"{_num(ss.get('seed_spread'), nd)} ({'SEED_SENSITIVE' if ss.get('seed_sensitive') else 'no'})")
    holm = (b.get("holm") or {}).get("holm_band")
    cells = [c["id"], c["contrast"], c["question"], c.get("hierarchy", "-"), rule, label,
             _num(b.get("seeds", b.get("status"))), _num(b.get("seeds_status", "-")), _num(b.get("point"), nd),
             _ci(b.get("ci95"), nd), _side(b, ref), _num(b.get("n_users")), _num(b.get("n_decisions")),
             _num(b.get("harmed_share"), 4), spread, _ci(holm, nd) if holm else "-",
             (b.get("claim") or {}).get("template_key", "-")]
    return [f"**{x}**" if bold and x not in ("", "-") else x for x in cells]


def _find_rec(res: Mapping, kid: str) -> Optional[Mapping]:
    for key in ("confirmatory", "extra_registered", "registered_descriptive"):
        for c in res.get(key) or []:
            if c["id"] == kid:
                return c
    return None


def _stat_cell(st: Mapping, nd: int = 2) -> str:
    if not isinstance(st, Mapping):
        return "-"
    if "point" not in st:
        return _num(st.get("status", "-"))
    return f"{_num(st.get('point'), nd)} {_ci(st.get('ci95'), nd)}"


def render_ana04_sections(res: Mapping) -> List[str]:
    """Report sections 8-12 (06L-C): the LAST anchor and the device-only envelope, the privacy matrix and the bytes
    axis, the labelled rows (S8 / DP8_FROZEN, no-FT UNSEEN_VAL, GRU_D128 vs C_SMALL), the CALIB-disjoint diagnostic
    and the repeat / novel target strata, the claim register and the headline template. All DESCRIPTIVE."""
    L = ["## 8. LAST anchor and the device-only envelope (06L-C C3) — DESCRIPTIVE (K11b is in section 1)", ""]
    an = res.get("last_anchor") or {}
    if an.get("status") != "COMPUTED":
        L += [f"- {an.get('status')}: {an.get('reason')}", ""]
    else:
        L += [f"- anchor: `{an['anchor']}`; {an['rule']}.", ""]
        rows = [[r["arm"], r["rule"], _stat_cell(r["delta_vs_LAST"], 6), _num(r["BELOW_DEVICE_HEURISTIC"]),
                 _stat_cell(r["headroom_vs_C_BIG"]), _stat_cell(r.get("headroom_vs_C_FULL")) if r.get(
                     "headroom_vs_C_FULL") else "-", _num((r["delta_vs_LAST"] or {}).get("harmed_share"), 4)]
                for r in an["rows"]]
        L += _table(["arm", "rule", "Δ(arm − LAST) [CI]", "BELOW_DEVICE_HEURISTIC", "R* vs C_BIG % [CI]",
                     "R* vs C_FULL % [CI]", "harmed share vs LAST"], rows) + [""]
    env = res.get("device_only_envelope") or {}
    L += [f"### {env.get('group', 'device-only envelope')}", ""]
    rows = [[r["arm"], _num(r.get("value")), _stat_cell(r.get("R_vs_C_BIG")),
             _stat_cell(r.get("delta_vs_LAST"), 6) if isinstance(r.get("delta_vs_LAST"), Mapping) else "-"]
            for r in env.get("rows", [])]
    L += _table(["arm", "MRR@20 (seed mean)", "R% vs C_BIG [CI]", "Δ vs LAST [CI]"], rows)
    L += ["", f"- best point (DESCRIPTIVE, never a selection): {json.dumps(env.get('best_point'), sort_keys=True)}; "
              f"{env.get('note', '')}.", ""]
    L += ["## 9. Privacy matrix (06L-C C1) and one bytes axis (06L-C C8) — DESCRIPTIVE", ""]
    rows = []
    for lv in res.get("privacy_matrix") or []:
        arms = lv["arms"] or [{"arm": "-", "R_vs_C_BIG": {}, "bytes_per_client": None}]
        for a in arms:
            rows.append([lv["level"], a["arm"], _stat_cell(a["R_vs_C_BIG"]), _num(a["bytes_per_client"], 0),
                         lv["server_sees_training"], lv["server_sees_inference"], lv["formal_guarantee"],
                         lv["against_whom"], lv["secure_aggregation"]])
    L += _table(["level", "arm", "R% vs C_BIG (PB, seed mean)", "bytes/client", "server sees (training)",
                 "server sees (inference)", "formal guarantee", "against whom", "secure aggregation"], rows)
    L += ["", "### Bytes axis (raw events → model download → FL traffic → S8)", ""]
    rows = [[b["step"], b["level"], b["arm"], _num(b.get("bytes_per_client"), 0),
             _num(b.get("bytes_per_client_incl_final_download"), 0), _num(b.get("status")), b["rule"]]
            for b in res["curves"].get("bytes_axis") or []]
    L += _table(["step", "level", "arm", "bytes/client", "bytes/client incl. final download", "status", "rule"], rows)
    L += ["", "## 10. Labelled rows (06L-C C11, C16-C17) — DESCRIPTIVE", "", "### 10a. DP8_FROZEN vs its z = 0 twin S8", ""]
    fb = res.get("frozen_branch") or {}
    L += ([f"- {fb.get('status')}: {fb.get('reason')}"] if fb.get("status") != "COMPUTED" else
          [f"- {fb['contrast']} {rule}: {_stat_cell(fb['by_rule'][rule])} (ε = {_num(fb.get('epsilon'))}, z = "
           f"{_num(fb.get('dp_z'), 4)})" for rule in RULES]) + [""]
    L += ["### 10b. No-FT references on UNSEEN_VAL (the selection panel; never a headline)", ""]
    nf = res.get("no_ft_unseen_val") or {}
    rows = [[r["ft_arm"], r["base_arm"], r["rule"], r["seed"], _num(r["no_ft_value"]), _ci(r["no_ft_ci95"], 6),
             _num(r["selected_cell"]), _num(r["selected_cell_value"]), _num(r["selected_minus_no_ft"])]
            for r in nf.get("rows", [])]
    L += (_table(["FT arm", "base (no FT)", "rule", "seed", "no-FT MRR@20", "95% CI", "selected cell",
                  "selected cell value (SELECTION)", "selected − no-FT"], rows) if rows else
          [f"- {nf.get('status')}: {nf.get('reason')}"]) + ["", f"- {nf.get('note', '')}", ""]
    L += ["### 10c. Deployed-size family comparison GRU_D128 vs C_SMALL (06L-C C11)", ""]
    fc_ = res.get("family_comparison") or {}
    if fc_.get("status") != "COMPUTED":
        L += [f"- {fc_.get('status')}: {fc_.get('reason')}"]
    else:
        for rule in RULES:
            b = fc_["by_rule"][rule]
            L += [f"- {rule}: R(GRU_D128 / C_SMALL) = {_stat_cell(b.get('R'))}; Δ = {_stat_cell(b.get('delta'), 6)}; "
                  f"params {json.dumps(b.get('param_count'), sort_keys=True)}" if "R" in b else f"- {rule}: NOT_AVAILABLE"]
    L += ["", f"- {REGISTERED_06LC['C11']}", ""]
    L += ["## 11. CALIB-disjoint diagnostic (MAIN \\ CALIB; 06L-C C14) and repeat / novel target strata (C15) — "
          "DESCRIPTIVE", ""]
    cd = res.get("calib_disjoint") or {}
    if cd.get("status") != "COMPUTED":
        L += [f"- {cd.get('status')}: {cd.get('reason')}", ""]
    else:
        _cm, _ce = cd["counts"]["manifest"], cd["counts"]["e2e"]
        L += [f"- {_cm['disjoint_users']:,} CALIB-disjoint users, of whom {_ce['disjoint_users']:,} have ≥ 1 eval row "
              f"(manifest users: MAIN {_cm['main_users']:,}, CALIB {_cm['calib_users']:,}; E2E users: MAIN "
              f"{_ce['main_users']:,}, CALIB {_ce['calib_users']:,}); counts {json.dumps(cd['counts'], sort_keys=True)}; "
              f"{cd['mapping']}; {cd['note']}.", ""]
        rows = [[a["arm"], _stat_cell(a["R_vs_C_BIG_disjoint"]), _stat_cell(a["R_vs_C_BIG_full_MAIN"])]
                for a in cd["arms"]]
        L += _table(["arm", "R% vs C_BIG on MAIN \\ CALIB [CI]", "R% vs C_BIG on MAIN [CI]"], rows) + [""]
        rows = [[k["id"], k["contrast"], k["rule"], _stat_cell(k["stat"], 2 if "R(" in k["contrast"] else 6)]
                for k in cd["contrasts"]]
        L += _table(["id", "contrast", "rule", "on MAIN \\ CALIB [CI]"], rows) + [""]
    ts = res.get("target_strata") or {}
    if ts.get("status") != "COMPUTED":
        L += [f"- target strata {ts.get('status')}: {ts.get('reason')}", ""]
    else:
        L += [f"- {ts['rule']}; counts {json.dumps(ts['counts'], sort_keys=True)}.", ""]
        rows = [[t.get("stratum"), t.get("arm", "-"), _num(t.get("n_users")), _num(t.get("R_vs_C_BIG"), 2),
                 _ci(t.get("ci95"))] for t in ts["rows"]]
        L += _table(["stratum", "arm", "users", "R% vs C_BIG", "95% CI"], rows) + [""]
    cr = res.get("claim_register") or {}
    L += ["## 12. Claim register and headline template (06L-C C12) — DESCRIPTIVE (text, frozen with the script)", "",
          f"- CLAIM_REGISTER.json ({cr.get('format')}; {len(cr.get('contrasts') or [])} contrasts) and HEADLINE_TEMPLATE.md "
          "are written beside this report; the template key of each K is the `claim key` column of section 1.", ""]
    L += [f"- {r}" for r in cr.get("general_rules") or []] + [""]
    return L


def render_ana05_sections(res: Mapping) -> List[str]:
    """Report section 12b (ANA-05): C_FULL s2027, the S7 / S8 warm-selection binding, FT_FA_WARM - FA_WARM (06M_B B4),
    quality during rollout (06M_B B5), S8 communication; all DESCRIPTIVE."""
    a = res.get("ana05") or {}
    L = ["## 12b. ANA-05 additions (C_FULL s2027; 06M; 06M_B B2 / B4 / B5; S8 comm) — DESCRIPTIVE", ""]
    ds = a.get("descriptive_seeds") or {}
    L += [f"- Descriptive source seeds: {json.dumps([(r['arm'], r['seed'], r['rule'], r['role']) for r in ds.get('rows', [])])}"
          f"; {ds.get('rule', '')}.", ""]
    for st, w in sorted(((a.get("warm_selection") or {}).get("stages") or {}).items()):
        L += [f"- {st} ({w['arm']}): {w['status']}; warm selection sha256 {w.get('warm_selection_sha256')}; record "
              f"{json.dumps(w.get('record'), sort_keys=True)}."]
    L += ["", "### FT_FA_WARM − FA_WARM (06M_B B4; the full production stack)", ""]
    rows = []
    for r in (a.get("production_stack") or {}).get("rows", []):
        d = r.get("delta") or {}
        rows.append([r["contrast"], r.get("status"), _num(d.get("seeds")), _num(d.get("point")), _ci(d.get("ci95"), 6),
                     _side(d, 0.0) if d else "-", _num(d.get("harmed_share"), 4),
                     json.dumps(r.get("skipped") or []) if r.get("skipped") else "-"])
    L += _table(["contrast", "status", "seeds", "Δ MRR@20", "95% CI", "CI vs 0", "harmed share", "skipped"], rows) + [""]
    rq = a.get("rollout_quality") or {}
    L += ["### Quality during rollout (06M_B B5)", "", f"- {rq.get('rule', '')}", ""]
    rows = [[r["arm"], r["seed"], _num(r["auc"]), _num(r["auc_mean"]), _ci(r["auc_mean_ci95"], 6),
             _num((r["reach"].get("PRE") or {}).get("first_mark_point_rule",
                                                    (r["reach"].get("PRE") or {}).get("status"))),
             _num((r["reach"].get("PRE") or {}).get("first_mark_ci_rule")),
             _num((r["reach"].get("LAST") or {}).get("first_mark_point_rule",
                                                     (r["reach"].get("LAST") or {}).get("status"))),
             _num((r["reach"].get("LAST") or {}).get("first_mark_ci_rule"))] for r in rq.get("runs", [])]
    L += _table(["arm", "seed", "AUC (MRR x EFE)", "AUC / 5.5", "95% CI", "first mark ≥ PRE (point)",
                 "first mark > PRE (CI)", "first mark ≥ LAST (point)", "first mark > LAST (CI)"], rows) + [""]
    sc = a.get("s8_comm") or {}
    rows = [[r["arm"], r["seed"], _num(r["bytes_per_visit"]), _num(r["bytes_up_per_visit"]),
             _num(r["bytes_down_per_visit"]), _num(r["uploaded_numel_measured"]), _num(r["frozen_numel"]),
             _num(r["frozen_bytes_once"]), _num(r["per_visit_ratio_vs_FA_WARM"], 4)] for r in sc.get("rows", [])]
    L += ["### S8 communication from the run's own record (06L-C C8)", ""]
    L += (_table(["arm", "seed", "bytes/visit", "up", "down", "uploaded params (measured)", "frozen params",
                  "frozen bytes once", "per-visit ratio vs FA_WARM"], rows) if rows else [f"- {sc.get('status')}"])
    L += ["", f"- WARM_PREVIEW: {(a.get('warm_preview') or {}).get('rule', '')}", ""]
    return L


def render_ana06_sections(res: Mapping) -> List[str]:
    """Report section 12c (ANA-06): 06N on the FA_WARM* line, DESCRIPTIVE with the 06N tier; needs registration."""
    a = res.get("ana06") or {}
    if not a:
        return []
    st = a.get("fa_warm_star") or {}
    L = ["## 12c. ANA-06: the pretrained line (PREG1_AMENDMENT_06N) — DESCRIPTIVE until registered", "",
         f"- FA_WARM* = {st.get('arm')} ({st.get('stage')}; {st.get('status')}); CALIB winners "
         f"{json.dumps({k: v.get('value') for k, v in (st.get('calib_winner') or {}).items()}, sort_keys=True)}; "
         f"{st.get('rule', '')}.", ""]
    rows = []
    for c in a.get("contrasts", []):
        for rule in RULES:
            b = c["by_rule"].get(rule) or {}
            rows.append([c["id"], c["contrast"], c.get("tier_06N"), rule, _num(b.get("seeds")), _num(b.get("point")),
                         _ci(b.get("ci95")), b.get("seeds_status", b.get("status", "-"))])
    L += (_table(["id", "contrast", "06N tier", "rule", "seeds", "point", "95% CI", "seeds status"], rows)
          if rows else [f"- {a.get('status')}: {a.get('reason')}"]) + [""]
    for r in a.get("references", []):
        L.append(f"- {r['id']} ({r['tier_06N']}): see {r['see']}.")
    L += ["", "Needs registration:"] + [f"- {x}" for x in a.get("needs_registration", [])] + [""]
    return L


def render_report(res: Mapping) -> str:
    L = [f"# PPSI V4 retention report — split {res['split']}", "",
         f"- RETENTION_RESULT.json self_sha256: `{res[SELF_KEY]}`",
         f"- script sha256: `{res['script']['sha256']}`; contrasts sha256: `{res['inputs']['contrasts_sha256']}` "
         f"(content `{res['inputs']['contrasts_content_sha256']}`)",
         f"- block index sha256: `{res['inputs']['files_sha256'].get(INDEX_NAME)}`",
         f"- comparison key: `{json.dumps(res['checks']['comparison_key']['values'], sort_keys=True)}`; device pin "
         f"`{res['checks']['comparison_key'].get('device_pin_sha256')}`",
         f"- bootstrap: paired user, {res['bootstrap']['n_resamples']} resamples, seed {res['bootstrap']['seed']}; "
         "ratios bootstrapped directly; seed mean on the same W.",
         f"- contrasts: format {res['inputs']['contrasts_format']}"
         + (" — **DRY CHECK ONLY (v1 contrasts, VALIDATION): not evidence for the thesis**"
            if res["inputs"]["contrasts_v1_dry_check"] else "") + ".",
         f"- multiplicity (06D §4; 06I §1; 06L-C C4 / C13): {res['labels']['multiplicity']}.",
         (f"- **{res['labels']['superseded_note']}.**" if res["labels"].get("superseded_note") else
          "- contrasts registry: company/gates/ANA_CONTRASTS_v3.json (06L-C §C4)."
          if res["inputs"]["contrasts_format"] == CONTRAST_FORMAT_V3 else
          "- contrasts registry: company/gates/ANA_CONTRASTS_v4.json (06P)."
          if res["inputs"]["contrasts_format"] == CONTRAST_FORMAT_V4 else "- contrasts registry: v1 (dry check)."),
         f"- {POST_VALIDATION_NOTE}",
         f"- **Only section 1 ({ids_label([c['id'] for c in res['confirmatory']])}) is CONFIRMATORY (each K under its "
         "registered rule and seeds, bold); every other number is DESCRIPTIVE.**", ""]
    st = res["sanity_trigger"]
    if "fired" in st:
        L += [f"- sanity trigger (amendment 04 §1 / 06E §6, not a verdict): R(C_SMALL CB / C_BIG PB) = "
              f"{_num(st['R_C_SMALL_CB_vs_C_BIG_PB'], 2)} -> fired = {st['fired']}; R(C_SMALL PB / C_BIG PB) = "
              f"{_num(st.get('R_C_SMALL_PB_vs_C_BIG_PB'), 2)}", ""]
    L += ["## 1. Confirmatory contrasts (06B §2; rule map 06D §3; hierarchy 06D §4; seeds 06E §2) — CONFIRMATORY", ""]
    rows, notes = [], []
    for c in res["confirmatory"]:
        for rule in RULES:
            b = c["by_rule"][rule]
            lab = b.get("label", "DESCRIPTIVE")
            rows.append(_conf_row(c, rule, b, lab, lab == "CONFIRMATORY"))
            if b.get("disclosure"):
                notes.append(f"- {c['id']} {rule} (DESCRIPTIVE): {b['disclosure']}.")
            if b.get("missing_registered_seeds"):
                notes.append(f"- {c['id']} {rule}: {b['seeds_status']} — registered seeds "
                             f"{c['registered_seeds']}, missing {b['missing_registered_seeds']}.")
        for rule, b in sorted((c.get("extra_seeds") or {}).items()):
            rows.append(_conf_row(c, rule, b, "DESCRIPTIVE (all seeds)", False))
        for rule, b in sorted((c.get("descriptive_fallback") or {}).items()):
            rows.append(_conf_row(c, rule, b, "DESCRIPTIVE (DEVICE_MIXED fallback)", False))
    L += _table(CONF_HEAD, rows)
    L += [""] + notes + ([""] if notes else [])
    ws = res.get("warm_start") or {}
    if ws.get("rows"):
        L += ["### 1b. Deployment branch (06K; DL 206; 06L-C C7: the Edge answer and the L5 row lead with these) — "
              "K20, K20b, K20c, K21, K9c", ""]
        rows, wnotes = [], []
        for w in ws["rows"]:
            c = _find_rec(res, w["id"])
            if c is None:
                continue
            for rule in RULES:
                b = c["by_rule"][rule]
                lab = b.get("label", "DESCRIPTIVE")
                rows.append(_conf_row(c, rule, b, lab, lab == "CONFIRMATORY"))
            if c.get("note"):
                wnotes.append(f"- {c['id']}: {c['note']}.")
        L += _table(CONF_HEAD, rows) + [""] + wnotes + [""]
        L += [f"- {d}" for d in ws["disclosures"]] + [""]
    L += ["### 1c. Per-K statistics (06L-C C13-C15) — DESCRIPTIVE beside each K", "",
          f"Holm band: {res['holm_band']['method']}; family = {res['holm_band']['family_size']} registered contrasts, "
          f"{res['holm_band']['n_computed']} computed.", ""]
    rows = []
    for c in res["confirmatory"]:
        b = c["by_rule"][c["confirmatory_rule"]]
        h, ss, eq = b.get("holm") or {}, b.get("seed_stats") or {}, b.get("equivalence") or {}
        nulls = b.get("harmed_share_seed_null") or {}
        nd = 2 if c["kind"] == "retained" else 6
        rows.append([c["id"], c["confirmatory_rule"], json.dumps(ss.get("per_seed_points"), sort_keys=True)
                     if ss.get("per_seed_points") else "-", _num(ss.get("seed_spread"), nd),
                     _num(ss.get("seed_sensitive")), _ci(ss.get("user_seed_interval"), nd), _num(h.get("p_holm"), 4),
                     _num(h.get("holm_reject")), _ci(h.get("holm_band"), nd), _ci(h.get("bonferroni_band"), nd),
                     _num(b.get("harmed_share"), 4), _num((nulls.get("numerator") or {}).get("harmed_share"), 4),
                     (f"{eq.get('equivalent')} (90% CI {_ci(eq.get('ci90'))})" if eq.get("status") == "COMPUTED" else
                      "-")])
    L += _table(["id", "rule", "per-seed points", "seed spread", "SEED_SENSITIVE", "user + seed interval", "Holm p",
                 "Holm reject", "Holm band", "Bonferroni band", "harmed share", "harmed share, seed-vs-seed null",
                 "K14 equivalence (±1 % rel.)"], rows) + [""]
    if res.get("extra_registered"):
        L += ["### 1d. Registered rows the loaded file does not carry — DESCRIPTIVE", ""]
        rows = []
        for c in res["extra_registered"]:
            for rule in RULES:
                rows.append(_conf_row(c, rule, c["by_rule"][rule], "DESCRIPTIVE", False))
        L += _table(CONF_HEAD, rows) + [""]
    if res.get("registered_descriptive"):
        L += ["### 1e. Registered DESCRIPTIVE contrasts (06L-C C8: S8 − S7 with the bytes ratio)", ""]
        rows = []
        for c in res["registered_descriptive"]:
            for rule in RULES:
                br = (c.get("bytes_ratio") or {}).get(rule) or {}
                rows.append(_conf_row(c, rule, c["by_rule"][rule], "DESCRIPTIVE", False)
                            + [_num(br.get("mean_ratio"), 4)])
        L += _table(CONF_HEAD + ["bytes ratio S8 / S7"], rows) + [""]
    L += ["## 2. Retention table (MAIN) — DESCRIPTIVE", ""]
    for rule in RULES:
        L += [f"### {rule}", ""]
        rows = []
        for r in res["retention"]["MAIN"][rule]:
            vb, vs = r.get("vs_C_BIG") or {}, r.get("vs_C_SMALL") or {}
            pb, ps = vb.get("mrr@20") or vb, vs.get("mrr@20") or vs
            si = r.get("still_improving_at_6.0") or {}
            for seed, e in r["entries"].items():
                m = e["meta"]
                rows.append([r["arm"], seed, _num(e["values"]["mrr@20"]), _num(e["values"]["ndcg@10"]),
                             _num(e["values"]["hr@10"]), _seed_R(pb, seed), _seed_R(ps, seed),
                             _num(m.get("bytes_per_visit")),
                             _num(m.get("bytes_per_client"), 0), _num(m.get("bytes_per_client_charged_dropped"), 0),
                             _num(m.get("bytes_per_client_charged_dropped_up_down"), 0),
                             _num(m.get("bytes_fleet")), _num(m.get("epsilon"), 4),
                             _num(m.get("param_count")), _num(m.get("model_bytes")), _num(m.get("rounds")),
                             _num(m.get("consumed_efe"), 3), _num((si.get("per_seed") or {}).get(seed))])
            mixed = " DEVICE_MIXED" if pb.get("comparison") == "DEVICE_MIXED" else ""
            rows.append([f"**{r['arm']}**", f"mean({_num(pb.get('n_seeds'))})", "", "", "",
                         f"{_num(pb.get('point'), 2)} {_ci(pb.get('ci95'))}{mixed}",
                         f"{_num(ps.get('point'), 2)} {_ci(ps.get('ci95'))}",
                         "", "", "", "", "", "", "", "", "", "", f"any={_num(si.get('any_seed'))}" if si else ""])
        L += _table(["arm", "seed", "MRR@20", "NDCG@10", "HR@10", "R% vs C_BIG [CI]", "R% vs C_SMALL [CI]",
                     "bytes/visit", "bytes/client", "bytes/client (dropped visits charged, 06I 7b)",
                     "bytes/client (dropped visits charged down + up; DESCRIPTIVE)", "bytes fleet",
                     "ε", "params", "model bytes", "rounds",
                     "consumed EFE", "still improving at 6.0 (PB mark >= 5.5; DESCRIPTIVE)"], rows) + [""]
        rows = []
        for r in res["retention"]["MAIN"][rule]:
            vb = r.get("vs_C_BIG") or {}
            if "ndcg@10" in vb:
                rows.append([r["arm"], f"{_num(vb['ndcg@10'].get('point'), 2)} {_ci(vb['ndcg@10'].get('ci95'))}",
                             f"{_num(vb['hr@10'].get('point'), 2)} {_ci(vb['hr@10'].get('ci95'))}"])
        L += _table(["arm", "R% NDCG@10 vs C_BIG", "R% HR@10 vs C_BIG"], rows) + [""]
    L += ["## 3. Personalisation gains (paired) — DESCRIPTIVE", ""]
    rows = []
    for p in res["personalisation"]:
        for rule in RULES:
            b = p["by_rule"][rule]
            rows.append([p["contrast"], rule, _num(b.get("seeds", b.get("status"))), _num(b.get("point")),
                         _ci(b.get("ci95"), 6), _side(b, 0.0), _num(b.get("harmed_share"), 4)])
    L += _table(["contrast", "rule", "seeds", "delta MRR@20", "95% CI", "CI vs 0", "harmed-client share"], rows)
    L += ["", "## 4. Privacy-level ladder (06B §3) — DESCRIPTIVE", ""]
    rows = []
    for lv in res["privacy_ladder"]:
        if not lv["arms"]:
            rows.append([lv["level"], "-", "-", "-", lv["leaves_training"], lv["leaves_inference"], lv["guarantee"]])
        for a in lv["arms"]:
            R = a["R_vs_C_BIG"]
            rows.append([lv["level"], a["arm"], f"{_num(R.get('point'), 2)} {_ci(R.get('ci95'))}",
                         _num(a["bytes_per_client"], 0), lv["leaves_training"], lv["leaves_inference"], a["guarantee"]])
    L += _table(["level", "arm", "R% vs C_BIG (PB, seed mean)", "bytes/client", "leaves device (training)",
                 "leaves device (inference)", "guarantee"], rows) + ["", LADDER_NOTE, ""]
    L += ["## 5. Curves — DESCRIPTIVE (PRACTICAL_BEST; full series in curves/*.csv)", "", "### 5a. R vs ε", ""]
    rows = []
    for e in res["curves"]["epsilon"]:
        for rule in RULES:
            b = e["by_rule"][rule]
            rows.append([e["arm"], _num(e["epsilon_registered_accountant"], 4), _num(e["dp_z"], 4), rule,
                         f"{_num(b['R_vs_C_BIG'].get('point'), 2)} {_ci(b['R_vs_C_BIG'].get('ci95'))}",
                         f"{_num(b['R_vs_FA_1024'].get('point'), 2)} {_ci(b['R_vs_FA_1024'].get('ci95'))}",
                         json.dumps(e.get("epsilon_receipt_fields", "-"), sort_keys=True)])
    L += _table(["arm", "ε (registered)", "z", "rule", "R% vs C_BIG", "R% vs FA_1024", "ε fields (DP_WOR receipt)"],
                rows)
    L += ["", "### 5b. R vs model size", ""]
    rows = [[s["arm"], _num(s["param_count"]), _num(s["model_bytes"]),
             f"{_num(s['by_rule']['PRACTICAL_BEST'].get('point'), 2)} {_ci(s['by_rule']['PRACTICAL_BEST'].get('ci95'))}",
             f"{_num(s['by_rule']['CONTROLLED_BUDGET'].get('point'), 2)} {_ci(s['by_rule']['CONTROLLED_BUDGET'].get('ci95'))}",
             s.get("note") or ""] for s in res["curves"]["model_size"]]
    L += _table(["arm", "params", "model bytes", "R% PB", "R% CB", "note"], rows)
    L += ["", "### 5c. R vs cumulative bytes per client (FL marks)", ""]
    rows = [[b["arm"], b["seed"], _num(b["mark"], 1), _num(b["round"]), _num(b["cum_bytes_per_client"], 0),
             _num(b.get("cum_bytes_per_client_charged_dropped"), 0),
             _num(b.get("cum_bytes_per_client_charged_dropped_up_down"), 0), _num(b["value"]),
             _num(b["R_vs_C_BIG_PB"], 2), _ci(b["ci95"])] for b in res["curves"]["bytes"]]
    L += _table(["arm", "seed", "mark", "round", "cum bytes/client (prorated; registered series)",
                 "cum bytes/client (dropped visits charged their download; second series)",
                 "cum bytes/client (dropped visits charged down + up; DESCRIPTIVE)", "MRR@20", "R%", "95% CI"], rows)
    L += ["", DISCLOSURE_BYTES_06I_7B]
    L += ["", "### 5d. R per stratum (MAIN)", ""]
    rows = [[s["family"], s["group"], s["arm"], s["n_users"], _num(s["R_vs_C_BIG"], 2), _ci(s["ci95"])]
            for s in res["curves"]["strata"]]
    L += _table(["family", "group", "arm", "users", "R% vs C_BIG", "95% CI"], rows)
    L += ["", "### 5e. R per support-count bin (06A §2, observational)", "",
          f"User counts per bin: `{json.dumps(res['curves']['support_bin_counts'], sort_keys=True)}`", ""]
    rows = [[s["role"], s["group"], s["arm"], s["study"], s["n_users"], _num(s["R_vs_C_BIG"], 2), _ci(s["ci95"])]
            for s in res["curves"]["support_bins"]]
    L += _table(["role", "bin", "arm", "study", "users", "R% vs C_BIG", "95% CI"], rows)
    L += ["", "Events-to-90% / 95% (smallest bin with M >= X C_ref; bin 0 from TRUE_COLD; no interpolation):", ""]
    rows = [[e["panel"], e["arm"], e["seed"], e["events_to_90"], e["events_to_95"]] for e in res["curves"]["events_to_x_bins"]]
    L += _table(["panel", "arm", "seed", "events-to-90%", "events-to-95%"], rows)
    L += ["", "### 5f. Controlled cold start (DATA-07 per-h rows; contract §11.7; C_ref = C_BIG FULL history)", ""]
    rows = [[t["arm"], t["panel"], t["panel_type"], t["seed"], t["h"], _num(t["M"]), _num(t["C_ref"]), _num(t["R"], 2),
             _num(t["M_full_history"])] for t in res["curves"]["trunc"]]
    L += _table(["arm", "panel", "panel type", "seed", "h", "M", "C_ref", "R%", "M (FULL history)"], rows) + [""]
    rows = [[t["arm"], t["panel"], t["panel_type"], t["seed"], t["events_to_90"], t["events_to_95"],
             json.dumps(t["M_full_history"], sort_keys=True) if isinstance(t["M_full_history"], Mapping)
             else _num(t["M_full_history"])] for t in res["curves"]["trunc_events_to_x"]]
    L += _table(["arm", "panel", "panel type", "seed", "events-to-90% (h)", "events-to-95% (h)",
                 "M (FULL history, E11.10)"], rows)
    L += ["", "06L-C §C10 beside the registered first-hit rule: matched-h parity P(h) = M(h) / C_BIG(h) and the "
          "persistence rule. " + res["curves"]["trunc_persistence_rule"], ""]
    rows = [[t["arm"], t["panel"], t["panel_type"], t["seed"], t["h"], _num(t["R"], 2), _ci(t.get("R_ci95")),
             _num(t.get("C_BIG_h")), _num(t.get("parity_matched_h"), 2), _ci(t.get("parity_ci95"))]
            for t in res["curves"]["trunc"]]
    L += _table(["arm", "panel", "panel type", "seed", "h", "R%", "R 95% CI", "C_BIG(h)", "P% (matched h)",
                 "P 95% CI"], rows) + [""]
    rows = [[t["arm"], t["panel"], t["panel_type"], t["seed"], t["events_to_90_persistent"],
             t["events_to_95_persistent"], t["parity_events_to_90"], t["parity_events_to_95"],
             t["parity_events_to_90_persistent"], t["parity_events_to_95_persistent"]]
            for t in res["curves"]["trunc_events_to_x"]]
    L += _table(["arm", "panel", "panel type", "seed", "R: persistent 90%", "R: persistent 95%", "P: first-hit 90%",
                 "P: first-hit 95%", "P: persistent 90%", "P: persistent 95%"], rows)
    L += ["", "## 6. Other panels — DESCRIPTIVE", ""]
    for role, tab in sorted(res["panels"].items()):
        rows = []
        for rule in RULES:
            for r in tab.get(rule, []):
                pb = (r.get("vs_C_BIG") or {}).get("mrr@20") or r.get("vs_C_BIG") or {}
                rows.append([role, rule, r["arm"], f"{_num(pb.get('point'), 2)} {_ci(pb.get('ci95'))}"])
        L += _table(["panel", "rule", "arm", "R% vs C_BIG (seed mean)"], rows) + [""]
    if res["historical_reference"]:
        L += ["### HISTORICAL_REFERENCE rows (comparison key differs; point only)", ""]
        L += _table(["entry", "MRR@20", "R% vs C_BIG (point; seed mean)", "C_BIG seeds", "C_BIG rule"],
                    [[h["entry"], _num(h["value"]), _num(h["R_vs_C_BIG_point"], 2), _num(h.get("C_BIG_seeds")),
                      h.get("C_BIG_rule", "-")] for h in res["historical_reference"]]) + [""]
    L += ["## 7. VALIDATION -> TEST shift indicator (06B §7b) — DESCRIPTIVE", "",
          "```", json.dumps(res["shift_indicator"], sort_keys=True, indent=1), "```", ""]
    L += render_ana04_sections(res)
    L += render_ana05_sections(res)
    L += render_ana06_sections(res)
    L += ["## 13. Appendix — records (LR screen = CALIB, not MAIN), inputs, checks", "", "```",
          json.dumps(res["records"], sort_keys=True, indent=1, ensure_ascii=False), "```", "",
          "Library files sha256:", "", "```", json.dumps(res["library_files_sha256"], sort_keys=True, indent=1), "```", ""]
    L += ["## 14. Registered disclosures (verbatim)", "", "Amendment 06A §9:", "", DISCLOSURES_06A_9, "",
          "Amendment 06B §7:", "", DISCLOSURES_06B_7, "", "Amendment 06B §1:", "", DISCLOSURE_06B_1, "",
          "Amendment 06D §3:", "", REGISTERED_06D_3, "", "Amendment 06D §4:", "", REGISTERED_06D_4, "", "Notes:", ""]
    L += [f"- {n}" for n in DISCLOSURE_NOTES] + [""]
    L += ["Amendment 06K / DL 206:", ""] + [f"- {d}" for d in DISCLOSURES_06K] + [""]
    L += ["Amendment 06I §7(b):", "", DISCLOSURE_BYTES_06I_7B, ""]
    L += ["Amendment 06L-C (verbatim):", ""] + [f"- {k}: {v}" for k, v in REGISTERED_06LC.items()] + [""]
    L += ["How the 06L-C items are computed (ANA-04):", ""] + [f"- {d}" for d in DISCLOSURES_ANA04] + [""]
    L += ["How the ANA-05 items are computed:", ""] + [f"- {d}" for d in DISCLOSURES_ANA05] + [""]
    return "\n".join(L)


CURVE_COLUMNS = {
    "bytes": ("arm", "seed", "mark", "efe_raw", "round", "cum_bytes_per_client", "cum_bytes_per_client_charged_dropped",
              "cum_bytes_per_client_charged_dropped_up_down", "value", "R_vs_C_BIG_PB", "ci95", "bytes_rule", "label"),
    "strata": ("role", "family", "group", "arm", "study", "n_users", "R_vs_C_BIG", "ci95", "label"),
    "support_bins": ("role", "family", "group", "arm", "study", "n_users", "R_vs_C_BIG", "ci95", "label"),
    "events_to_x_bins": ("panel", "arm", "seed", "events_to_90", "events_to_95", "label"),
    "trunc": ("arm", "panel", "panel_type", "seed", "study", "h", "M", "C_ref", "R", "R_ci95", "C_BIG_h",
              "parity_matched_h", "parity_ci95", "M_full_history", "label"),
    "trunc_events_to_x": ("panel", "panel_type", "arm", "seed", "events_to_90", "events_to_95",
                          "events_to_90_persistent", "events_to_95_persistent", "parity_events_to_90",
                          "parity_events_to_95", "parity_events_to_90_persistent", "parity_events_to_95_persistent",
                          "M_full_history", "C_ref_method", "label"),
    "epsilon": ("arm", "epsilon_registered_accountant", "dp_z", "by_rule", "epsilon_receipt_fields", "label"),
    "model_size": ("arm", "param_count", "model_bytes", "by_rule", "note", "label"),
    "bytes_axis": ("step", "level", "arm", "seed", "bytes_per_client", "bytes_per_client_charged_dropped",
                   "final_download_bytes", "bytes_per_client_incl_final_download", "status", "rule", "label"),
    "rollout": ("arm", "seed", "mark", "round", "value", "delta_vs_PRE", "delta_vs_LAST", "label"),   # ANA-05 (B5)
}


def _cell(v) -> str:
    v = _jsonable(v)
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    if isinstance(v, float):
        return repr(v)
    return "" if v is None else str(v)


def render_csv(rows: Sequence[Mapping], cols: Sequence[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow([_cell(r.get(c)) for c in cols])
    return buf.getvalue()


def write_outputs(out: Path, res: dict) -> dict:
    shas = {}
    for name, cols in CURVE_COLUMNS.items():
        rows = res["curves"][name]
        shas[f"curves/{name}.csv"] = _write_new(out / "curves" / f"{name}.csv", render_csv(rows, cols).encode("utf-8"))
        shas[f"curves/{name}.json"] = _write_new(out / "curves" / f"{name}.json", dumps(rows).encode("utf-8"))
    shas[RESULT_NAME] = _write_new(out / RESULT_NAME, dumps(res).encode("utf-8"))
    shas[REPORT_NAME] = _write_new(out / REPORT_NAME, render_report(res).encode("utf-8"))
    shas.update(write_claims(out, res["claim_register"]))                  # 06L-C C12
    man = with_self_sha({"schema": "PPSI_V4_ANA_OUTPUTS_1", "result_self_sha256": res[SELF_KEY],
                         "files_sha256": dict(sorted(shas.items()))})
    _write_new(out / MANIFEST_NAME, dumps(man).encode("utf-8"))
    return man



CLAIM_NAME, HEADLINE_NAME = "CLAIM_REGISTER.json", "HEADLINE_TEMPLATE.md"


def write_claims(out: Path, register: Mapping) -> dict:
    """06L-C C12: CLAIM_REGISTER.json (self_sha256) + HEADLINE_TEMPLATE.md (number-free), O_EXCL."""
    obj = with_self_sha(dict(register))
    return {CLAIM_NAME: _write_new(out / CLAIM_NAME, dumps(obj).encode("utf-8")),
            HEADLINE_NAME: _write_new(out / HEADLINE_NAME, register["headline_template"].encode("utf-8"))}


# ------------------------------------------------------------------------------------------------ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--block", type=Path, default=None, help="required unless --claims-only")
    ap.add_argument("--split", choices=("VALIDATION", "TEST"), default=None, help="required unless --claims-only")
    ap.add_argument("--contrasts", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--code-root", type=Path, default=DEFAULT_CODE_ROOT)
    ap.add_argument("--expect-script-sha256", default=None)
    ap.add_argument("--check-only", action="store_true", help="bind + all checks; no value printed, nothing written")
    ap.add_argument("--all-methods-lock", type=Path, default=None, help="TEST only: company/gates/ALL_METHODS_LOCK.json")
    ap.add_argument("--test-exposure-register", type=Path, default=None,
                    help="TEST only: company/registers/TEST_EXPOSURE_REGISTER.json")
    ap.add_argument("--registry-ext", type=Path, default=None, help="TEST only: REGISTRY_V2_TEST_EXT.json (DATA-09)")
    ap.add_argument("--allow-v1-contrasts", action="store_true",
                    help="VALIDATION dry check only: accept company/gates/ANA_CONTRASTS.json (v1, K1-K16); never for TEST")
    ap.add_argument("--allow-draft-contrasts", action="store_true",
                    help="VALIDATION only: load a PPSI_V4_ANA_CONTRASTS_4 file whose draft flag is not false; never TEST")
    ap.add_argument("--claims-only", action="store_true",
                    help="06L-C C12: write CLAIM_REGISTER.json + HEADLINE_TEMPLATE.md for --contrasts into --out; no block")
    args = ap.parse_args(argv)
    if not args.claims_only and (args.block is None or args.split is None):
        ap.error("--block and --split are required (unless --claims-only)")
    return args


def check_ext_roles(idx: Mapping, ext: Mapping) -> dict:
    """Every TEST role of the index is the extension's `<role>_TEST` pair and the registered ROLE_MAP pair."""
    out = {}
    for role in sorted(idx.get("roles") or {}):
        ev, tr = ROLE_MAP["TEST"][role]
        want = {"train": tr, "eval": ev} if tr else {"eval": ev}
        got = (ext.get("roles") or {}).get(f"{role}_TEST")
        if got != want:
            raise AnaRefused(f"--registry-ext maps {role}_TEST to {got!r}, the registered pair is {want!r} (A-F8 (iii))")
        if ev not in (ext.get("manifests") or {}):
            raise AnaRefused(f"--registry-ext has no manifest entry {ev}")
        out[role] = want
    return out


def open_test_gate(args, block: Path) -> Tuple[dict, dict]:
    """A-F8 (i)-(iii): the lock / register gate, --contrasts IS the lock, the extension rule, the TEST role map."""
    gate = check_test_gate(args.all_methods_lock, args.test_exposure_register)
    if _sha_file(args.contrasts) != gate["all_methods_lock_sha256"]:
        raise AnaRefused("--split TEST: --contrasts must be the ALL_METHODS_LOCK file itself (sha256 differs; A-F8)")
    if args.registry_ext is None or not Path(args.registry_ext).is_file():
        raise AnaRefused("--split TEST requires --registry-ext <REGISTRY_V2_TEST_EXT.json> (A-F8 (ii))")
    ext = _strict_json(args.registry_ext)
    if not REGISTRY_V2_PATH.is_file():
        raise AnaRefused(f"{REGISTRY_V2_PATH} is missing: the extension cannot be verified")
    checks = verify_test_ext_registry(ext, _strict_json(REGISTRY_V2_PATH), _sha_file(REGISTRY_V2_PATH))
    idx, _entries = load_index(block, "TEST")
    roles = check_ext_roles(idx, ext)
    return gate, {"registry_ext_file_sha256": _sha_file(args.registry_ext), "registry_ext_self_sha256":
                  ext.get("self_sha256"), "base_registry_file_sha256": _sha_file(REGISTRY_V2_PATH),
                  "extension_rule_checks": sorted(checks), "roles": roles, "_ext": ext}


def run(args, *, context_loader=None, say=print) -> int:
    """The CLI body (_run) with the registry tables of the loaded contrasts format; always back to v3 on exit."""
    try:
        return _run(args, context_loader=context_loader, say=say)
    finally:
        set_registry_profile(None)


def _run(args, *, context_loader=None, say=print) -> int:
    sha = script_sha256()
    if args.expect_script_sha256 and args.expect_script_sha256 != sha:
        raise AnaRefused(f"script sha256 {sha} is not the frozen {args.expect_script_sha256}")
    out = Path(args.out)
    if not args.check_only and out.exists() and any(out.iterdir()):
        raise AnaRefused(f"{out} exists and is not empty: outputs are never overwritten (O_EXCL); use a new --out")
    bind(args.code_root)
    if args.claims_only:                                           # 06L-C C12: the register frozen with the script
        if args.block is not None or args.split is not None or args.check_only:
            raise AnaRefused("--claims-only takes only --contrasts and --out (no block, no split, no --check-only)")
        contrasts = load_contrasts(Path(args.contrasts), split="VALIDATION", allow_v1=False,
                                   allow_draft=getattr(args, "allow_draft_contrasts", False))
        set_registry_profile(contrasts["format"])                     # ANA-06: v4 tables (v3: unchanged)
        reg = claim_register(contrasts)
        out.mkdir(parents=True, exist_ok=True)
        shas = write_claims(out, reg)
        say(f"ANA CLAIMS: written {out / CLAIM_NAME} ({len(reg['contrasts'])} contrasts) and {out / HEADLINE_NAME}; "
            f"contrasts {contrasts['format']} content {contrasts['content_sha256']}; files "
            f"{json.dumps(shas, sort_keys=True)}; script sha256 {sha}")
        return 0
    if args.allow_v1_contrasts and args.split == "TEST":
        raise AnaRefused("--allow-v1-contrasts is a VALIDATION dry check only; a TEST run needs the registered "
                         "ALL_METHODS_LOCK (K1-K19 / K1-K21)")
    if getattr(args, "allow_draft_contrasts", False) and args.split == "TEST":
        raise AnaRefused("--allow-draft-contrasts is a VALIDATION dry check only (06P is not registered)")
    contrasts = load_contrasts(Path(args.contrasts), split=args.split, allow_v1=args.allow_v1_contrasts,
                               allow_draft=getattr(args, "allow_draft_contrasts", False))
    set_registry_profile(contrasts["format"])                         # ANA-06: v4 tables (v3: unchanged)
    gate, ext_rec = None, None
    if args.split == "TEST":
        gate, ext_rec = open_test_gate(args, Path(args.block))
        loader = context_loader or test_context_loader(gate, Path(args.registry_ext), ext_rec["_ext"],
                                                       Path(args.all_methods_lock), Path(args.test_exposure_register))
    else:
        if args.all_methods_lock or args.test_exposure_register or args.registry_ext:
            raise AnaRefused("--all-methods-lock / --test-exposure-register / --registry-ext are TEST-only options")
        loader = context_loader or production_context_loader()
    blk = load_block(Path(args.block), args.split, loader, gate=gate)
    records = load_records(Path(args.block), blk.idx, blk.files)
    vcheck = check_validation_result(records.get("VALIDATION_RESULT"), sha, contrasts) if args.split == "TEST" else None
    blk.ana05 = check_ana05(blk, records, args.split,                # ANA-05 bindings (06M_B B2 / B4; REV-ANA-05 F1 /
                            records.get("VALIDATION_RESULT"))        # F3), check-only too
    blk.ana06 = check_ana06(blk, records, args.split, records.get("VALIDATION_RESULT"))   # ANA-06 (06N; FL-16)
    star_st = (blk.ana06.get("fa_warm_star") or {}).get("stage")
    if contrasts["format"] == CONTRAST_FORMAT_V4 and star_st not in (None, "S8"):
        raise AnaRefused(f"--contrasts v4 (06P P1) resolves FA_WARM* = FA_WARM_FROZEN (S8), but the registered selection "
                         f"files choose {star_st} (06N N2): the loaded registry does not fit this block")
    gate_rec = None if gate is None else dict(gate, **{k: v for k, v in ext_rec.items() if k != "_ext"})
    if args.check_only:
        eng = Engine(blk.ctx)
        book = Book(blk.entries)
        plan = plan_pairs(eng, book, contrasts, (blk.ana06.get("fa_warm_star") or {}).get("arm"))
        trunc_curves(records.get("TRUNC_ROWS") or [], blk.key["values"],          # its refusals, nothing kept
                     records.get("TRUNC_PER_USER"))
        say(f"ANA CHECK-ONLY: PASS — {len(blk.entries)} entries, roles {sorted(blk.ctx)}, comparison key "
            f"{json.dumps(blk.key['values'], sort_keys=True)}, historical {len(blk.key['historical_reference'])}, "
            f"onnx sub-block {'yes' if blk.key.get('onnx_block') else 'no'}, pairs checked {plan['n_pairs_checked']}, "
            f"block denominators checked {plan['n_block_denominators_checked']}, "
            f"FA_WARM* {(blk.ana06.get('fa_warm_star') or {}).get('stage')}, "
            f"records {sorted(k for k, v in records.items() if v)}; script sha256 {sha}")
        return 0
    analysis = analyse(blk, contrasts, records, args.split)
    res = with_self_sha(build_result(blk, contrasts, records, args.split, analysis, script_sha=sha,
                                     test_gate=gate_rec, validation_check=vcheck))
    man = write_outputs(out, res)
    say(f"ANA: written {out / RESULT_NAME} self_sha256 {res[SELF_KEY]}; {out / REPORT_NAME}; "
        f"{len(man['files_sha256'])} files in {MANIFEST_NAME}; script sha256 {sha}")
    return 0


def main(argv=None, *, context_loader=None, say=print) -> int:
    args = parse_args(argv)
    try:
        return run(args, context_loader=context_loader, say=say)
    except AnaRefused as e:
        say(f"ANA REFUSED: {e}")
        return REFUSED
    except Exception as e:                       # a library refusal (PairingError, ...) or a malformed input
        import traceback
        say(f"ANA ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return ERROR


if __name__ == "__main__":
    sys.exit(main())
