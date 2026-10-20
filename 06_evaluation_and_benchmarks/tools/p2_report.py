"""tools/p2_report.py — the PPSI V4 Phase-2 analysis (work packet P2-12; frozen by P2_SCRIPT_FREEZE, Fable touch point 3).

  python -B tools/p2_report.py --split TEST --block <P2 TEST block> --contrasts company/gates/P2_CONTRASTS_v3.json
         --out <new dir p2_out_test_*> --p2-lock company/gates/P2_METHODS_LOCK.json
         --p2-freeze company/gates/P2_SCRIPT_FREEZE.json --all-methods-lock company/gates/ALL_METHODS_LOCK.json
         --test-exposure-register company/registers/TEST_EXPOSURE_REGISTER.json --registry-ext <REGISTRY_V2_TEST_EXT.json>
         --phase1-result <the sitting's Phase-1 RETENTION_RESULT.json> [--code-root <V4>/code] [--check-only]
  python -B tools/p2_report.py --split VALIDATION --block <P2 VALIDATION block> --contrasts ... --out <new dir>
         --p2-lock ... --p2-freeze ... --all-methods-lock ... --phase1-result <Phase-1 sitting RETENTION_RESULT.json>
         --p2-test-result <p2_out_test_*/RETENTION_RESULT.json> [--check-only]

Binding spec: contracts/PREG2.md (REGISTERED, DL 411, sha256 be0561ab…) §6.3 (T0a sitting S1-S8), §6.4 (labels), §9
(contrasts, statistics, multiplicity, the §9.4 reading and interpretation grid), §11 (disclosures D-a … D-k); contrasts
of record company/gates/P2_CONTRASTS_v3.json (self_sha256 e98580564b…, DL 427: O3 = YES -> Holm over Q2-Q8; owner O2 =
NO -> no TEST2 data version, Q1-replication and Q14 are TEST2_NOT_AVAILABLE); PHASE2_IMPLEMENTATION_PLAN P2-12; Fable
gate reviews/FABLE_GATE_ENDGAME_SCRIPTS_20261003 §9-§10 (Q14 is reported only in the TEST2 family, never in the RETRO_TEST
band; NO_MEASURABLE_CALIB_GAIN at 0.001).

LIBRARY (PREG2 F6: frozen Phase-1 tools are imported, never edited). tools/retention_report.py is imported as the module
`retention_report` only after its file sha256 equals RR_SHA256 (9a3b2c2a…; asserted at import and again at run time,
together with tools/ana_block_index.py = ABI_SHA256, 6e7043bb…). Reused unchanged: bind (the --code-root library),
load_index / load_block (every Phase-1 block check: E9.1 comparison key, pinned device, rank files reproduce rows,
rank coverage, rule vs checkpoint, arm identity, TEST manifest headers bound to the gate), check_test_gate /
verify_test_ext_registry / check_ext_roles / test_context_loader (the DATA-09 TEST path), Engine (subclassed: P2Engine
only replaces the resample plan's seed), Book (pairs), ratio / delta (paired user bootstrap, seed means, harmed share,
per-seed statistics via _seed_stats), seed_null, holm_band (Holm step-down on normal intervals from the bootstrap SE,
tertiary DESCRIPTIVE band), k14_equivalence (TOST on the 90% percentile CI inside [99, 101] = Q13's registered margin),
with_self_sha / dumps / _write_new (outputs), _self_ok (registry self-sha rules).
The Phase-2 arm labels enter the library's tables only inside `p2_profile()` — the library's own pattern
(retention_report.set_registry_profile) — which extends ARMS / FL_ARMS / DEVICE_ARMS / CENTRAL_ARMS / WARM_INIT_ARMS /
FL_METHOD_ALT / DP_Z / EPS_DEFAULT / FROZEN_ONCE_ARMS / SOURCE_TRAIN_MANIFEST + TRAIN_MANIFEST_OF (C2 =
P2_C2_CFULL_SMALL, a central source trained on FULL_TRAIN_5_100_TIEBREAK), sets the block anchor REF = PRE (load_index's reference
requirement; a P2 block has no C_BIG) and restores every table, by identity, on exit (tested).

STATISTICS (PREG2 §9.2): paired user bootstrap, 1,000 resamples, seed 20261004 (P2_BOOTSTRAP_SEED; one shared W per
user set = v4eval.bootstrap.make_plan(seed=20261004)), percentile 95% CIs, ratios bootstrapped directly, seed means on
the same W, per-seed points, seed spread / SEED_SENSITIVE, harmed-client share with the same-arm seed-vs-seed null. E2E
client-macro MRR@20 is the statistic of every Q; NDCG@10 / HR@10 appear in the DESCRIPTIVE value table only.
Each Q uses its registered rule (PRACTICAL_BEST; CONTROLLED_BUDGET for Q6 / Q7 / Q11 / Q12a / Q12b) on both sides and
exactly its registered seeds; the other rule is DESCRIPTIVE (the DP PRACTICAL_BEST ratios with the registered
dp_disclosure). Shared denominators: PRE (one theta_pre, seed 2026) for Q1 / Q2 / Q7 / Q8 / Q12b, C_FULL s2026 for Q9.
Q5 pairs only an FT_P2_WSTAR whose base is the same-seed P2_WSTAR best.pt (base_best_only; index entry `p2_base`).
MULTIPLICITY (§9.3): Q1 alone primary, unadjusted at alpha 0.05, outside Holm; Holm step-down at alpha 0.05 over the
secondary family Q2-Q8 (family size 7, P2_CONTRASTS_v3); tertiary Q9 / Q11 / Q12a / Q12b / Q13 unadjusted with a
DESCRIPTIVE Holm band (Q14 is in the TEST2 family only: TEST2_NOT_AVAILABLE). Q10 = Q1 / Q2 per seed and per resample
on the same W (derived, DESCRIPTIVE). Equivalence: Q1 by TOST on the 90% CI inside [-0.0005, +0.0005] (absolute MRR@20);
Q13 by TOST on the 90% CI inside [99, 101] (k14_equivalence).
READING (§9.4, registered): SUCCESS iff Q1's 95% CI lower bound > 0 and both per-seed points > 0; EQUIVALENT iff Q1's 90%
CI lies inside ±0.0005; FAILURE iff the 95% CI upper bound < 0; INCONCLUSIVE otherwise. When two conditions hold at once
(possible: a significant gain smaller than the margin) the listed order decides and `conditions_true` / `overlap` record
both (open question for the freeze). DEPLOYMENT-GRADE = SUCCESS + Q5 > 0 (Holm-rejected) + Q7 > 0 (Holm-rejected). The
interpretation grid (Q2 x Q1) is evaluated with the definitions in GRID_DEFINITIONS. D-h is printed beside Q1.
LABELS (§6.4): Q1-Q8 on RETRO_TEST = CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP), each with D-a verbatim; Q9-Q13 =
TERTIARY_P2 (unadjusted; DESCRIPTIVE Holm band); every number of a VALIDATION run = POST_VALIDATION_EXPLORATORY; Q1 on
TEST2 and Q14 = TEST2_NOT_AVAILABLE (O2 = NO); E1-E5, devices, D1-ext = DESCRIPTIVE (tertiary), NOT_RUN when absent.

GATES (refusal = exit 2, nothing written):
  always   retention_report / ana_block_index file shas = the frozen values; P2_CONTRASTS = format
           PPSI_V4_P2_CONTRASTS_1, draft false, self_sha256 (utf8-compact sort_keys rule) = the pinned e98580564b…, every
           structural field = this script's tables; P2_METHODS_LOCK exists and verifies (P2_LOCK_FORMAT: self_sha256,
           preg2_sha256, p2_contrasts_self_sha256, bootstrap_seed 20261004, sealed_arms, phase1_reused,
           declared_not_run, selection_files_sha256); P2_SCRIPT_FREEZE verifies and binds THIS file, tools/
           p2_block_index.py, retention_report and ana_block_index; the sitting's Phase-1 RETENTION_RESULT exists, is
           split TEST, self-sha verified, written by retention_report 9a3b2c2a and bound to the ALL_METHODS_LOCK (S3: the
           Phase-1 analysis precedes the Phase-2 analysis).
  TEST     the Phase-1 TEST gate (check_test_gate: ALL_METHODS_LOCK in a MANIFEST_BUILD RETRO_TEST entry of a verified
           chain) + the extension rule + the TEST role map; TEST_EXPOSURE_REGISTER LOCKED_EVALUATION entries for BOTH
           locks (the Phase-2 one naming P2_METHODS_LOCK in observations_description; S2, §13 item 6); --out named
           p2_out_test_*; every P2 entry sealed in the lock with its checkpoint sha256 (S4); no declared-not-run arm in
           the block (S4); the reused Phase-1 entries (PRE, FA_WARM_FROZEN, FT_PRE, C_FULL) are exactly the lock's
           phase1_reused list, carry its checkpoint sha256, and their rows / ranks files have the sha256 the Phase-1
           RETENTION_RESULT recorded for the same block paths (S5: produced once, hash-bound into both blocks).
  VALID.   S8: Phase-2 MAIN VALIDATION values are analysed only after BOTH sitting analyses exist: the P2 lock + the
           Phase-1 sitting RETENTION_RESULT + the Phase-2 sitting RETENTION_RESULT (--p2-test-result: schema
           PPSI_V4_P2_RETENTION_1, split TEST, this script's sha, the same P2 lock). Every number is
           POST_VALIDATION_EXPLORATORY.
  block    only P2 arms + the four reused Phase-1 arms, role MAIN only, PRE present (the anchor), P2 rows carry
           study PREG2 (launch_fl.p2_row_labels; the P2-13 reeval_p2 rows must too) and Phase-1 rows do not; a P2_WSTAR
           run dir (VALIDATION) carries the P2_WARM_SELECTION_FINAL winner label; the selection record (when bound by the
           lock) has the bound sha256.
OUTPUT (--out new / empty; O_EXCL; deterministic): RETENTION_RESULT.json (schema PPSI_V4_P2_RETENTION_1, self_sha256),
REPORT.md, OUTPUT_MANIFEST.json. stdout carries paths, shas and counts only. --check-only runs every gate, the block load
and every pairing / denominator check, prints no value and writes nothing. Exit: 0 ok, 2 refused, 5 error.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
V4 = HERE.parent
RR_PATH = HERE / "retention_report.py"
ABI_PATH = HERE / "ana_block_index.py"
P2_INDEX_SCRIPT = HERE / "p2_block_index.py"
RR_SHA256 = "9a3b2c2a918ef2baacc5299426941f1eb94c366f5bb6aa7c29b4bfdbca0afe58"     # PREG2 F6 / §9.2 / P2-00 freeze
ABI_SHA256 = "6e7043bbe42967b0de313206496de38fb0cf20cc85aaa34945dfdd19863cb662"    # tools/ana_block_index.py (frozen)
PREG2_SHA256 = "be0561abe5ceec450d3a75bef49853a3c480d3847f75559dd94e9e0545df18a8"  # contracts/PREG2.md (DL 411)
P2_CONTRASTS_SELF_SHA256 = "e98580564b618f411db7e7e328fbcedb6e1134f4be5b65806b34e12229373799"   # P2_CONTRASTS_v3, DL 427
P2_CONTRASTS_FORMAT = "PPSI_V4_P2_CONTRASTS_1"
P2_CONTRASTS_DOCUMENT = "P2_CONTRASTS_v3"
P2_BOOTSTRAP_SEED = 20261004                                                         # PREG2 F2 / §9.2
P2_N_RESAMPLES = 1000
P2_LOCK_FORMAT = "PPSI_V4_P2_METHODS_LOCK_1"                                         # P2-00 tools/p2_lock.py (interface)
P2_FREEZE_FORMAT = "PPSI_V4_P2_SCRIPT_FREEZE_1"                                      # P2-00 (interface)
P2_SCHEMA = "PPSI_V4_P2_RETENTION_1"
P2_STUDY = "PREG2"                                                                    # launch_fl.p2_row_labels "study"
TEST_OUT_PREFIX = "p2_out_test_"                                                      # PREG2 §6.3 S3
FREEZE_FILES = ("tools/p2_report.py", "tools/p2_block_index.py", "tools/retention_report.py", "tools/ana_block_index.py")
REFUSED, ERROR = 2, 5


def _file_sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _load_library():
    """tools/retention_report.py, unchanged, as the module `retention_report` (the same object tools/ana_block_index.py
    resolves), imported only when its file sha256 is the frozen RR_SHA256."""
    got = _file_sha(RR_PATH)
    if got != RR_SHA256:
        raise ImportError(f"tools/retention_report.py sha256 {got} is not the frozen {RR_SHA256} (PREG2 F6): the "
                          "Phase-2 analysis imports the frozen library only")
    mod = sys.modules.get("retention_report")
    if mod is not None:
        if Path(mod.__file__).resolve() != RR_PATH.resolve():
            raise ImportError(f"a module 'retention_report' from {mod.__file__} is loaded, not {RR_PATH}")
        return mod
    spec = importlib.util.spec_from_file_location("retention_report", str(RR_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["retention_report"] = mod                       # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


RR = _load_library()


class P2Refused(RR.AnaRefused):
    """A Phase-2 gate or check failed: nothing is written."""


def assert_frozen_library() -> dict:
    """Run-time re-check (the import-time check may be stale in a long-lived process): both frozen Phase-1 tools still
    have their registered sha256 and the imported module is that file."""
    rr = _file_sha(RR_PATH)
    if rr != RR_SHA256 or RR.script_sha256() != RR_SHA256:
        raise P2Refused(f"tools/retention_report.py sha256 {rr} is not the frozen {RR_SHA256} (PREG2 F6)")
    abi = _file_sha(ABI_PATH) if ABI_PATH.is_file() else None
    if abi != ABI_SHA256:
        raise P2Refused(f"tools/ana_block_index.py sha256 {abi} is not the frozen {ABI_SHA256} (PREG2 F6)")
    return {"tools/retention_report.py": rr, "tools/ana_block_index.py": abi}


def script_sha256() -> str:
    return _file_sha(Path(__file__).resolve())


# ------------------------------------------------------------------------------------------------ Phase-2 arm table
P2_FL_ARMS = ("P2_WSTAR", "P2_WSTAR_FA1024", "P2_WSTAR_DP8", "P2_WSTAR_DP1", "P2_WSTAR_DPFTRL8", "P2_WSTAR_SECAGG")
P2_CENTRAL_ARMS = ("P2_C1_FT_CENTRAL_FULL", "P2_C1_FT_CENTRAL_FROZEN", "P2_C2_CFULL_SMALL")   # PREG2 §4.2 / §4.3
P2_DEVICE_ARMS = ("FT_P2_WSTAR",)                          # the registered FT_C procedure on top of W* (PREG2 §2)
P2_ARMS = P2_FL_ARMS + P2_CENTRAL_ARMS + P2_DEVICE_ARMS
PHASE1_REUSED_ARMS = ("PRE", "FA_WARM_FROZEN", "FT_PRE", "C_FULL")   # PREG2 §6.3 S5 (Fable §10 E1)
P2_BLOCK_ARMS = PHASE1_REUSED_ARMS + P2_ARMS
P2_ANCHOR = "PRE"
P2_RUN_ARM = "P2_W"                                        # launch_fl.P2_ARMS: the RUN_SUMMARY / LAUNCH arm of a P2 FL run
P2_METHOD_ALT = {                                          # rows / RUN_SUMMARY method or arm labels admitted per block arm
    "P2_WSTAR": (P2_RUN_ARM, "FA", "PF"), "P2_WSTAR_FA1024": (P2_RUN_ARM, "FA", "PF"),
    "P2_WSTAR_DP8": (P2_RUN_ARM, "DP", "FA", "PF"), "P2_WSTAR_DP1": (P2_RUN_ARM, "DP", "FA", "PF"),
    "P2_WSTAR_DPFTRL8": (P2_RUN_ARM, "DP", "FA", "PF"), "P2_WSTAR_SECAGG": (P2_RUN_ARM, "FA", "PF"),
    "FT_P2_WSTAR": ("FT_C",)}
P2_DP_Z = {"P2_WSTAR_DP8": 0.5642, "P2_WSTAR_DP1": 1.3094}   # PREG2 §5.5 / O7: the registered z at N = 131,072
P2_EPS = {"P2_WSTAR_FA1024": "inf", "P2_WSTAR_DP8": 8.0, "P2_WSTAR_DP1": 1.0, "P2_WSTAR_DPFTRL8": 8.0}
P2_SINGLE_SEED = ("P2_C2_CFULL_SMALL", "P2_WSTAR_DP1", "P2_WSTAR_DPFTRL8", "P2_WSTAR_SECAGG")   # PREG2 §9.5 {2026}
P2_SOURCE_TRAIN = {"P2_C2_CFULL_SMALL": "FULL_TRAIN_5_100_TIEBREAK"}   # C2 = a registered central source (PREG2 §4.3,
#   O3 = YES): its REEVAL_RECORD / rows carry central_source P2_C2_CFULL_SMALL + source_train_manifest_id (GATE-B B1
#   pattern, v4int.reeval_p2) and its MAIN rows the TIEBREAK train side (A-F7)
_PROFILE_KEYS = ("ARMS", "FL_ARMS", "DEVICE_ARMS", "PB_ONLY_ARMS", "CENTRAL_ARMS", "WARM_INIT_ARMS", "FL_METHOD_ALT",
                 "DP_Z", "EPS_DEFAULT", "FROZEN_ONCE_ARMS", "SOURCE_TRAIN_MANIFEST", "TRAIN_MANIFEST_OF", "REF")
_PROFILE_STATE = {"depth": 0}


@contextlib.contextmanager
def p2_profile(frozen_fl_arms: Sequence[str] = ()):
    """The Phase-2 arm labels inside the frozen library's tables (read at call time by load_index / load_block /
    check_arm_identity / load_marks / parse_contrast / tools/ana_block_index), restored by identity on exit. Not
    re-entrant (a nested profile would extend twice). `frozen_fl_arms`: the P2 FL arms whose runs froze the item tables
    (their bytes charge the tables once, GATE-B B3; decided from the runs' own RUN_SUMMARY)."""
    if _PROFILE_STATE["depth"]:
        raise P2Refused("p2_profile is already active (not re-entrant)")
    bad = [a for a in frozen_fl_arms if a not in P2_FL_ARMS]
    if bad:
        raise P2Refused(f"frozen_fl_arms {bad} are not P2 FL arms")
    saved = {k: getattr(RR, k) for k in _PROFILE_KEYS}
    _PROFILE_STATE["depth"] = 1
    try:
        RR.ARMS = tuple(saved["ARMS"]) + P2_ARMS
        RR.FL_ARMS = tuple(saved["FL_ARMS"]) + P2_FL_ARMS
        RR.DEVICE_ARMS = tuple(saved["DEVICE_ARMS"]) + P2_DEVICE_ARMS
        RR.PB_ONLY_ARMS = RR.DEVICE_ARMS + tuple(RR.ONNX_ARMS)
        RR.CENTRAL_ARMS = tuple(saved["CENTRAL_ARMS"]) + P2_CENTRAL_ARMS
        RR.WARM_INIT_ARMS = tuple(saved["WARM_INIT_ARMS"]) + P2_FL_ARMS          # theta_pre starts (init_from required)
        RR.FL_METHOD_ALT = dict(saved["FL_METHOD_ALT"], **P2_METHOD_ALT)
        RR.DP_Z = dict(saved["DP_Z"], **P2_DP_Z)
        RR.EPS_DEFAULT = dict(saved["EPS_DEFAULT"], **P2_EPS)
        RR.FROZEN_ONCE_ARMS = tuple(saved["FROZEN_ONCE_ARMS"]) + tuple(frozen_fl_arms)
        RR.SOURCE_TRAIN_MANIFEST = dict(saved["SOURCE_TRAIN_MANIFEST"], **P2_SOURCE_TRAIN)
        RR.TRAIN_MANIFEST_OF = dict(saved["TRAIN_MANIFEST_OF"], **P2_SOURCE_TRAIN)
        RR.REF = P2_ANCHOR                                   # load_index: the block's reference entry on MAIN = PRE
        yield
    finally:
        for k, v in saved.items():
            setattr(RR, k, v)
        _PROFILE_STATE["depth"] = 0


# ------------------------------------------------------------------------------------------------ registered content
P2_ITEMS = (   # P2_CONTRASTS_v3 confirmatory_contrasts (id, contrast); PREG2 §9.1
    ("Q1", "P2_WSTAR − PRE"), ("Q2", "P2_C1_FT_CENTRAL_FULL − PRE"), ("Q3", "P2_WSTAR − P2_C1_FT_CENTRAL_FROZEN"),
    ("Q4", "P2_WSTAR − FA_WARM_FROZEN"), ("Q5", "FT_P2_WSTAR − FT_PRE"), ("Q6", "R(P2_WSTAR_DP8 / P2_WSTAR_FA1024)"),
    ("Q7", "P2_WSTAR_DP8 − PRE"), ("Q8", "P2_C2_CFULL_SMALL − PRE"), ("Q9", "R(P2_WSTAR / C_FULL)"),
    ("Q11", "R(P2_WSTAR_DP1 / P2_WSTAR_FA1024)"), ("Q12a", "R(P2_WSTAR_DPFTRL8 / P2_WSTAR)"),
    ("Q12b", "P2_WSTAR_DPFTRL8 − PRE"), ("Q13", "R(P2_WSTAR_SECAGG / P2_WSTAR)"), ("Q14", "P2_WSTAR − PRE"))
P2_ORDER = tuple(k for k, _c in P2_ITEMS)
P2_PRIMARY = ("Q1",)
P2_HOLM_FAMILY = ("Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8")                     # O3 = YES (DL 427)
P2_TERTIARY = ("Q9", "Q11", "Q12a", "Q12b", "Q13", "Q14")
P2_TERTIARY_BAND_FAMILY = P2_TERTIARY                                             # as registered
P2_TEST2_FAMILY = ("Q14",)                                                         # Fable §10 (ii): TEST2 family only
P2_RETRO_TEST_BAND = tuple(k for k in P2_TERTIARY if k not in P2_TEST2_FAMILY)    # the sitting's DESCRIPTIVE band
P2_DEFAULT_RULE = "PRACTICAL_BEST"
P2_RULE_MAP = {"Q6": "CONTROLLED_BUDGET", "Q7": "CONTROLLED_BUDGET", "Q11": "CONTROLLED_BUDGET",
               "Q12a": "CONTROLLED_BUDGET", "Q12b": "CONTROLLED_BUDGET"}
P2_DEFAULT_SEEDS = (2026, 2027)
P2_REGISTERED_SEEDS = {"Q8": (2026,), "Q11": (2026,), "Q12a": (2026,), "Q12b": (2026,), "Q13": (2026,)}
P2_POPULATIONS = dict({k: "MAIN" for k in P2_ORDER}, Q10="MAIN", Q14="TEST2_UNEW_HOLDOUT")
P2_SHARED_DEN = {"Q1": "PRE", "Q2": "PRE", "Q7": "PRE", "Q8": "PRE", "Q9": "C_FULL", "Q12b": "PRE", "Q14": "PRE"}
P2_BASE_BEST_ONLY = ("Q5",)
Q1_MARGIN = 0.0005                                                                # §9.4 / D-h: TEST equivalence margin
Q1_INTERVAL = (-Q1_MARGIN, Q1_MARGIN)
Q13_INTERVAL = (99.0, 101.0)                                                      # §7.4 / §9.1: TOST [99, 101]
P2_DEPENDENCIES = {"Q10": ["Q1", "Q2"], "Q3": ["Q1"], "Q7": ["Q1"]}
DP_QS = ("Q6", "Q7", "Q11", "Q12a", "Q12b")
P2_RETRO_TEST_IDS = tuple(k for k in P2_ORDER if k not in P2_TEST2_FAMILY)

# §6.4 labels (Fable S6)
L_CONF = "CONFIRMATORY_P2 (PREREGISTERED_FOLLOWUP)"
L_TERT = "TERTIARY_P2 (unadjusted; DESCRIPTIVE Holm band)"
L_PVE = "POST_VALIDATION_EXPLORATORY"
L_DESC = "DESCRIPTIVE"
L_DESC_TERT = "DESCRIPTIVE (tertiary)"
L_T2NA = "TEST2_NOT_AVAILABLE"
LABELS_6_4 = (
    ("Phase-1 K1…K21 (all registered K)", "CONFIRMATORY (PREG1), unchanged"),
    ("Phase-2 Q1…Q8 on RETRO_TEST (the sitting)", L_CONF + ", always printed with disclosure D-a verbatim"),
    ("Phase-2 Q9…Q14 on RETRO_TEST", L_TERT),
    ("Phase-2 arms on MAIN VALIDATION", L_PVE),
    ("Q1 on TEST2", "REPLICATED_FRESH (same sign, 95 % CI excludes 0) / NOT_REPLICATED / TEST2_NOT_AVAILABLE"),
    ("Phase-1 locked arms on TEST2", "FRESH_PERIOD_AUDIT (DESCRIPTIVE)"),
    ("Fallback FB (§6.5)", "HOLDOUT_USERS_SAME_PERIOD, never a Q1 confirmation"),
    ("audits E1–E5, devices §8, D1-ext", L_DESC_TERT))
TEST2_REASON = ("owner decision O2 = NO (P2_CONTRASTS_v3 Q14 'only if TEST2 exists'; PREG2 §6.2 / §12): no December-2019 "
                "data version and no TEST2 register exist, so the Q1 replication family and Q14 are not scored")

# §11 disclosures, verbatim (whitespace folded)
DISCLOSURES = {
    "D-a": ("Phase-2 hypotheses were generated after the Phase-1 VALIDATION read; arms were selected on CALIB; everything "
            "was locked before any TEST value existed."),
    "D-b": "CALIB selection data are a subset of the MAIN users and of the VALIDATION window that was read.",
    "D-c": "θ_pre is one seed (no second seed was run). The PRE recipe was not re-tuned at its scale (Phase-1 note 3).",
    "D-d": "C_FULL differs from PRE in size and recipe. C2 (if run) is the matched reference.",
    "D-e": ("Central DP trusts the aggregator. SecAgg is simulated (costs counted, cryptography not executed). DP-FTRL / "
            "LDP are the untrusted-server variants."),
    "D-f": ("Device numbers come from the listed devices only. On-device training cost is a PC proxy unless stated "
            "otherwise."),
    "D-g": "RETRO_TEST was inspected at period level by earlier v1 campaigns (Phase-1 E12(d)).",
    "D-h": ("Phase 2 can confirm a gain ≥ ≈ 0.0005 MRR@20 or equivalence within ±0.0005; a true gain between them is "
            "likely to read INCONCLUSIVE."),
    "D-i": ("The DP arms keep T = 384 and the registered z. A small-step recipe over 384 rounds may move θ little; that "
            "is the registered mechanism's answer, not a defect to tune away."),
    "D-j": ("Futility uses CALIB headroom (32,768 users); MAIN has 131,072 users, so FUTILITY_SKIP_WB_WC may under-read "
            "the headroom available on MAIN."),
    "D-k": ("The Phase-1 TEST result was held for the Phase-2 runs (owner decision, DL 406). No Phase-1 arm, contrast or "
            "rule changed."),
}
READING_RULE_Q1 = ("PREG2 §9.4 (registered): SUCCESS iff the Q1 95% CI lower bound > 0 and both per-seed points > 0; "
                   "EQUIVALENT iff the Q1 90% paired user-bootstrap percentile CI lies inside ±0.0005 MRR@20 (TOST, alpha "
                   "0.05; the practical-relevance margin, distinct from the CALIB tie band 0.001); FAILURE iff the 95% CI "
                   "upper bound < 0; INCONCLUSIVE otherwise. Precedence when several hold: the listed order (SUCCESS, "
                   "EQUIVALENT, FAILURE); every condition that holds is recorded in conditions_true.")
DEPLOYMENT_RULE = "PREG2 §9.4: DEPLOYMENT-GRADE = SUCCESS + Q5 > 0 (Holm-rejected) + Q7 > 0 (Holm-rejected); not a test"
GRID = (   # PREG2 §9.4 interpretation grid (Q2 central headroom x Q1 FL), in the registered row order
    ("Q2_POSITIVE", "Q1_POSITIVE", "FL captures part of the available headroom (Q10 = how much)"),
    ("Q2_POSITIVE", "Q1_NONPOSITIVE_OR_EQUIVALENT", "the FL recipe fails to capture available headroom (an FL problem)"),
    ("Q2_EQUIVALENT", "Q1_EQUIVALENT", "MAIN data adds no measurable information over θ_pre at D64; FL cannot help here "
                                       "(a data / capacity limit)"),
    ("Q2_EQUIVALENT", "Q1_POSITIVE", "flagged for review (check for a central-recipe problem)"))
GRID_DEFINITIONS = {
    "Q1_POSITIVE": "the §9.4 reading of Q1 is SUCCESS",
    "Q1_NONPOSITIVE_OR_EQUIVALENT": "the §9.4 reading of Q1 is FAILURE or EQUIVALENT",
    "Q1_EQUIVALENT": "EQUIVALENT holds for Q1 (90% CI inside ±0.0005)",
    "Q2_POSITIVE": "Q2 (secondary) is Holm-rejected with a positive point (PRACTICAL_BEST)",
    "Q2_EQUIVALENT": "Q2's 90% paired user-bootstrap CI lies inside ±0.0005 (the Q1 margin applied to the grid's "
                     "'EQUIVALENT to 0'; DESCRIPTIVE TOST)",
    "NO_REGISTERED_CELL": "none of the four registered rows holds (e.g. Q1 INCONCLUSIVE): no grid reading is claimed"}
DESCRIPTIVE_RECORDS = (   # records key -> what it is (PREG2 §7, §8, §3.3); DESCRIPTIVE, NOT_RUN when absent
    ("E1_MIA", "E1 user-level membership inference on final models (PREG2 §7.1)"),
    ("E2_CANARY", "E2 canary exposure and one-run audit on FL training (§7.2)"),
    ("E3_UPDATE_LEAKAGE", "E3 leakage from individual FL updates (§7.3)"),
    ("E4_SECAGG", "E4 secure-aggregation simulation: bytes, rounds, unmasking time, dropout success (§7.4)"),
    ("E5_DPFTRL_LDP", "E5 DP-FTRL accountant / LDP feasibility (§7.5)"),
    ("DEVICES", "real-device measurements, thresholds DESCRIPTIVE design targets (§8)"),
    ("D1_EXT", "D1-ext update-norm summaries of the P2 FL runs (§3.3)"))
SELECTION_RECORDS = ("P2_WARM_SELECTION_FINAL", "P2_WARM_SELECTION_A", "P2_C1_SELECTION", "S_RECORD_P2")
P2_RECORD_KEYS = SELECTION_RECORDS + tuple(k for k, _w in DESCRIPTIVE_RECORDS)
FINAL_FORMAT = "PPSI_V4_P2_WARM_SELECTION_FINAL_1"                                # = v4int.p2_select FORMATS["FINAL"]
E1_EPS_RULE = ("PREG2 §7.1 (a) / Fable E-S3: an ε_LB is printed only when the record states pre_null_valid = true (PRE's "
               "AUC CI includes 0.5); otherwise every key containing 'eps' is withheld")


# ------------------------------------------------------------------------------------------------ small helpers
def _norm(s) -> str:
    return RR._norm_text(s)


def _jl(x):
    return json.loads(json.dumps(x))


def _utf8_compact_self(obj: Mapping) -> str:
    body = {k: v for k, v in obj.items() if k != RR.SELF_KEY}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _strict(path, what: str) -> dict:
    p = Path(path)
    if not p.is_file():
        raise P2Refused(f"{what} {p} does not exist")
    obj = RR._strict_json(p)
    if not isinstance(obj, Mapping):
        raise P2Refused(f"{what} {p} is not a JSON object")
    return obj


def _hex64(v) -> bool:
    return RR._hex64(v)


def tier_of(kid: str) -> str:
    return "primary" if kid in P2_PRIMARY else "secondary" if kid in P2_HOLM_FAMILY else "tertiary"


def p2_label(kid: str, split: str) -> str:
    """PREG2 §6.4: the label of contrast `kid` on `split` (TEST = the RETRO_TEST sitting)."""
    if kid in P2_TEST2_FAMILY or kid == "Q1_TEST2":
        return L_T2NA
    if split == "VALIDATION":
        return L_PVE
    if kid in P2_PRIMARY or kid in P2_HOLM_FAMILY:
        return L_CONF
    if kid in P2_TERTIARY:
        return L_TERT
    return L_DESC


# ------------------------------------------------------------------------------------------------ the contrasts registry
def load_p2_contrasts(path) -> dict:
    """company/gates/P2_CONTRASTS_v3.json: format, draft false, the utf8-compact self_sha256 = the pinned registered
    value, and every structural field = this script's tables (ids, formulas, hierarchy, Holm family, tertiary band,
    rule map, seeds, populations, shared denominators, base_best_only, reading rules, dependencies, PREG2 sha)."""
    p = Path(path)
    if not p.is_file():
        raise P2Refused(f"--contrasts {p} does not exist")
    raw = p.read_bytes()
    obj = _strict(p, "--contrasts")
    if obj.get("format") != P2_CONTRASTS_FORMAT:
        raise P2Refused(f"--contrasts: format {obj.get('format')!r} is not {P2_CONTRASTS_FORMAT}")
    if obj.get("draft") is not False:
        raise P2Refused("--contrasts: a DRAFT P2 contrasts file (draft not false) never enters the analysis")
    got_self = _utf8_compact_self(obj)
    if obj.get(RR.SELF_KEY) != got_self:
        raise P2Refused(f"--contrasts: self_sha256 does not verify under the utf8-compact sort_keys rule ({got_self})")
    if got_self != P2_CONTRASTS_SELF_SHA256:
        raise P2Refused(f"--contrasts: self_sha256 {got_self} is not the registered P2_CONTRASTS_v3 "
                        f"{P2_CONTRASTS_SELF_SHA256} (DL 427; v1 / v2 are superseded)")
    want = {
        "document": P2_CONTRASTS_DOCUMENT,
        "hierarchy": {"primary": list(P2_PRIMARY), "secondary": list(P2_HOLM_FAMILY), "tertiary": list(P2_TERTIARY)},
        "holm_family": list(P2_HOLM_FAMILY), "tertiary_band_family": list(P2_TERTIARY_BAND_FAMILY),
        "confirmatory_rule_map": dict(P2_RULE_MAP), "default_confirmatory_rule": P2_DEFAULT_RULE,
        "default_seeds": list(P2_DEFAULT_SEEDS),
        "registered_seeds": {k: list(v) for k, v in P2_REGISTERED_SEEDS.items()},
        "populations": dict(P2_POPULATIONS), "shared_denominators": dict(P2_SHARED_DEN),
        "base_best_only": list(P2_BASE_BEST_ONLY)}
    for key, w in want.items():
        g = obj.get(key)
        if key == "hierarchy" and isinstance(g, Mapping):
            g = {k: sorted(v, key=P2_ORDER.index) for k, v in g.items()}
        if _jl(g) != _jl(w):
            raise P2Refused(f"--contrasts: {key} {g!r} is not this script's registered {w!r}")
    rr_ = obj.get("reading_rules") or {}
    if [float(x) for x in (rr_.get("Q1") or {}).get("interval", [])] != list(Q1_INTERVAL) or \
            float((rr_.get("Q1") or {}).get("equivalence_margin_absolute", -1)) != Q1_MARGIN:
        raise P2Refused("--contrasts: reading_rules.Q1 is not the ±0.0005 TOST margin of PREG2 §9.4")
    if [float(x) for x in (rr_.get("Q13") or {}).get("interval", [])] != list(Q13_INTERVAL):
        raise P2Refused("--contrasts: reading_rules.Q13 is not the [99, 101] TOST interval of PREG2 §7.4")
    deps = {d.get("id"): list(d.get("of") or []) for d in obj.get("algebraic_dependencies") or []}
    if deps != P2_DEPENDENCIES:
        raise P2Refused(f"--contrasts: algebraic_dependencies {deps} are not {P2_DEPENDENCIES}")
    auth = obj.get("authority") or []
    if not auth or auth[0] != f"contracts/PREG2.md sha {PREG2_SHA256}":
        raise P2Refused("--contrasts: authority[0] does not name the registered contracts/PREG2.md sha256")
    if not str(obj.get("o3_decision", "")).startswith("YES"):
        raise P2Refused("--contrasts: o3_decision is not YES (the registered Holm family Q2-Q8 needs O3 = YES)")
    items = obj.get("confirmatory_contrasts")
    if not isinstance(items, list):
        raise P2Refused("--contrasts: no confirmatory_contrasts list")
    ids = [str(i.get("id")) for i in items]
    if sorted(ids, key=lambda k: P2_ORDER.index(k) if k in P2_ORDER else 99) != list(P2_ORDER) or \
            len(set(ids)) != len(ids):
        raise P2Refused(f"--contrasts: ids {ids} are not exactly {list(P2_ORDER)}")
    formula = dict(P2_ITEMS)
    by_id = {str(i["id"]): i for i in items}
    out = []
    with p2_profile():
        for kid in P2_ORDER:
            it = by_id[kid]
            if _norm(it.get("contrast")) != _norm(formula[kid]) or not str(it.get("question") or "").strip():
                raise P2Refused(f"--contrasts: {kid} is {it.get('contrast')!r}, the registered formula is "
                                f"{formula[kid]!r}")
            out.append(dict(id=kid, contrast=it["contrast"], question=it["question"], population=P2_POPULATIONS[kid],
                            **RR.parse_contrast(it["contrast"])))
    return {"file_sha256": RR._sha_bytes(raw), "self_sha256": got_self, "format": P2_CONTRASTS_FORMAT,
            "document": obj.get("document"), "items": out, "n_contrasts": len(out),
            "holm_family": list(P2_HOLM_FAMILY), "tertiary_band_family": list(P2_TERTIARY_BAND_FAMILY),
            "retro_test_band_family": list(P2_RETRO_TEST_BAND),
            "multiplicity_statement": obj.get("multiplicity_statement"), "dp_disclosure": obj.get("dp_disclosure"),
            "post_validation_rule": obj.get("post_validation_rule"),
            "pairing_notes": dict(obj.get("pairing_notes") or {}),
            "reading_rules": _jl(rr_), "algebraic_dependencies": _jl(obj.get("algebraic_dependencies"))}


# ------------------------------------------------------------------------------------------------ the gates
def check_p2_lock(path) -> dict:
    """PREG2 §6.3 S1 / S2 / S4 / S5: company/gates/P2_METHODS_LOCK.json (written by P2-00 tools/p2_lock.py). Without it
    every Phase-2 analysis refuses (TEST: S2; VALIDATION: S8)."""
    if path is None:
        raise P2Refused("--p2-lock is required: no Phase-2 analysis exists before P2_METHODS_LOCK (PREG2 §6.3 S2 / S8)")
    p = Path(path)
    if not p.is_file():
        raise P2Refused(f"P2_METHODS_LOCK {p} does not exist: no Phase-2 analysis exists before the lock (PREG2 §6.3 "
                        "S2 / S8)")
    obj = _strict(p, "P2_METHODS_LOCK")
    if obj.get("format") != P2_LOCK_FORMAT:
        raise P2Refused(f"P2_METHODS_LOCK: format {obj.get('format')!r} is not {P2_LOCK_FORMAT}")
    if obj.get("draft") is not False:
        raise P2Refused("P2_METHODS_LOCK: draft is not false")
    if RR._self_ok(obj) != "VERIFIED":
        raise P2Refused("P2_METHODS_LOCK: self_sha256 does not verify")
    for key, want in (("preg2_sha256", PREG2_SHA256), ("p2_contrasts_self_sha256", P2_CONTRASTS_SELF_SHA256),
                      ("bootstrap_seed", P2_BOOTSTRAP_SEED)):
        if obj.get(key) != want:
            raise P2Refused(f"P2_METHODS_LOCK: {key} {obj.get(key)!r} is not {want!r}")

    def triples(key, arms):
        out = {}
        for i, it in enumerate(obj.get(key) or []):
            if not isinstance(it, Mapping):
                raise P2Refused(f"P2_METHODS_LOCK: {key}[{i}] is not an object")
            k = (it.get("arm"), it.get("seed"), it.get("rule"))
            if k[0] not in arms or k[1] not in RR.SEEDS or k[2] not in RR.RULES:
                raise P2Refused(f"P2_METHODS_LOCK: {key}[{i}] {k} is not a registered (arm, seed, rule)")
            if k in out:
                raise P2Refused(f"P2_METHODS_LOCK: {key} lists {k} twice")
            out[k] = it
        return out

    sealed = {}
    for k, it in triples("sealed_arms", P2_ARMS).items():
        if k[0] in P2_DEVICE_ARMS:              # a device arm = the registered FT_C procedure on a sealed base (S4)
            if k[2] != "PRACTICAL_BEST" or not _hex64(it.get("base_checkpoint_sha256")):
                raise P2Refused(f"P2_METHODS_LOCK: sealed {k} (device arm) needs rule PRACTICAL_BEST and the "
                                "base_checkpoint_sha256 of its W* best.pt")
        elif not _hex64(it.get("checkpoint_sha256")):
            raise P2Refused(f"P2_METHODS_LOCK: sealed {k} has no checkpoint_sha256")
        sealed[k] = {"checkpoint_sha256": it.get("checkpoint_sha256"),
                     "base_checkpoint_sha256": it.get("base_checkpoint_sha256")}
    reused = {}
    for k, it in triples("phase1_reused", PHASE1_REUSED_ARMS).items():
        ck = it.get("checkpoint_sha256")
        if ck is not None and not _hex64(ck):
            raise P2Refused(f"P2_METHODS_LOCK: phase1_reused {k} checkpoint_sha256 {ck!r} is not a sha256")
        reused[k] = ck
    dnr = []
    for i, it in enumerate(obj.get("declared_not_run") or []):
        if not isinstance(it, Mapping) or it.get("arm") not in P2_ARMS or not str(it.get("reason") or "").strip():
            raise P2Refused(f"P2_METHODS_LOCK: declared_not_run[{i}] needs a P2 arm and a reason (06D §8 wording)")
        seed = it.get("seed")
        if any(a == it["arm"] and (seed is None or s == seed) for a, s, _r in sealed):
            raise P2Refused(f"P2_METHODS_LOCK: {it['arm']} seed {seed} is both sealed and declared not run")
        dnr.append({"arm": it["arm"], "seed": seed, "reason": it["reason"]})
    sel = obj.get("selection_files_sha256") or {}
    if not isinstance(sel, Mapping) or any(k not in SELECTION_RECORDS or not _hex64(v) for k, v in sel.items()):
        raise P2Refused(f"P2_METHODS_LOCK: selection_files_sha256 must map {SELECTION_RECORDS} to file sha256 values")
    return {"path": str(p), "file_sha256": _file_sha(p), "self_sha256": obj.get(RR.SELF_KEY), "sealed": sealed,
            "phase1_reused": reused, "declared_not_run": dnr, "selection_files_sha256": dict(sel),
            "n_sealed": len(sealed), "n_phase1_reused": len(reused)}


def check_p2_freeze(path) -> dict:
    """PREG2 §6.3 S2: company/gates/P2_SCRIPT_FREEZE.json binds this file, tools/p2_block_index.py and the two frozen
    Phase-1 tools they import; the running files must be exactly those."""
    if path is None:
        raise P2Refused("--p2-freeze is required (P2_SCRIPT_FREEZE, PREG2 §6.3 S2)")
    obj = _strict(path, "P2_SCRIPT_FREEZE")
    if obj.get("format") != P2_FREEZE_FORMAT:
        raise P2Refused(f"P2_SCRIPT_FREEZE: format {obj.get('format')!r} is not {P2_FREEZE_FORMAT}")
    if RR._self_ok(obj) != "VERIFIED":
        raise P2Refused("P2_SCRIPT_FREEZE: self_sha256 does not verify")
    files = obj.get("files_sha256") or {}
    running = {"tools/p2_report.py": script_sha256(),
               "tools/p2_block_index.py": _file_sha(P2_INDEX_SCRIPT) if P2_INDEX_SCRIPT.is_file() else None,
               "tools/retention_report.py": _file_sha(RR_PATH),
               "tools/ana_block_index.py": _file_sha(ABI_PATH) if ABI_PATH.is_file() else None}
    if files.get("tools/retention_report.py") != RR_SHA256 or files.get("tools/ana_block_index.py") != ABI_SHA256:
        raise P2Refused("P2_SCRIPT_FREEZE: the retention_report / ana_block_index sha256 are not the frozen Phase-1 "
                        f"values {RR_SHA256[:8]} / {ABI_SHA256[:8]}")
    bad = sorted(k for k in FREEZE_FILES if files.get(k) != running[k])
    if bad:
        raise P2Refused(f"P2_SCRIPT_FREEZE: the running {bad} differ from the frozen sha256 (S2: the analysis runs the "
                        "registered scripts only)")
    return {"path": str(path), "file_sha256": _file_sha(path), "self_sha256": obj.get(RR.SELF_KEY),
            "files_sha256": {k: files[k] for k in FREEZE_FILES}}


def check_phase1_result(path, p1_lock_sha: Optional[str]) -> dict:
    """PREG2 §6.3 S3: the sitting's Phase-1 RETENTION_RESULT (retention_report, split TEST, bound to ALL_METHODS_LOCK)
    exists before any Phase-2 analysis; its inputs.files_sha256 is the S5 binding of the reused Phase-1 rows."""
    if path is None:
        raise P2Refused("--phase1-result is required: the Phase-2 analysis runs only after the sitting's Phase-1 "
                        "RETENTION_RESULT exists (PREG2 §6.3 S3 / S8)")
    obj = _strict(path, "the Phase-1 RETENTION_RESULT")
    if obj.get("schema") != RR.SCHEMA or obj.get("split") != "TEST":
        raise P2Refused(f"the Phase-1 RETENTION_RESULT is schema {obj.get('schema')!r} / split {obj.get('split')!r}, "
                        f"not {RR.SCHEMA} / TEST (the sitting's Phase-1 analysis)")
    if not RR.verify_self_sha(obj):
        raise P2Refused("the Phase-1 RETENTION_RESULT self_sha256 does not verify")
    if (obj.get("script") or {}).get("sha256") != RR_SHA256:
        raise P2Refused("the Phase-1 RETENTION_RESULT was not written by the frozen retention_report (script sha256)")
    gate = (obj.get("inputs") or {}).get("test_gate") or {}
    if p1_lock_sha is None or gate.get("all_methods_lock_sha256") != p1_lock_sha:
        raise P2Refused("the Phase-1 RETENTION_RESULT is not bound to this ALL_METHODS_LOCK (inputs.test_gate)")
    files = (obj.get("inputs") or {}).get("files_sha256") or {}
    return {"path": str(path), "file_sha256": _file_sha(path), "self_sha256": obj.get(RR.SELF_KEY),
            "all_methods_lock_sha256": p1_lock_sha, "files_sha256": dict(files)}


def check_p2_test_result(path, lock: Mapping, script_sha: str, p1_lock_sha: str) -> dict:
    """PREG2 §6.3 S8: a VALIDATION analysis needs the Phase-2 sitting result written by this script under this lock."""
    if path is None:
        raise P2Refused("--p2-test-result is required for --split VALIDATION: Phase-2 MAIN VALIDATION values are "
                        "opened only after both sitting analyses are written (PREG2 §6.3 S8)")
    obj = _strict(path, "the Phase-2 sitting RETENTION_RESULT")
    if obj.get("schema") != P2_SCHEMA or obj.get("split") != "TEST":
        raise P2Refused(f"--p2-test-result is schema {obj.get('schema')!r} / split {obj.get('split')!r}, not "
                        f"{P2_SCHEMA} / TEST")
    if not RR.verify_self_sha(obj):
        raise P2Refused("--p2-test-result self_sha256 does not verify")
    if (obj.get("script") or {}).get("sha256") != script_sha:
        raise P2Refused("--p2-test-result was written by another tools/p2_report.py (script sha256 differs)")
    if ((obj.get("gates") or {}).get("p2_lock") or {}).get("file_sha256") != lock["file_sha256"]:
        raise P2Refused("--p2-test-result was written under another P2_METHODS_LOCK")
    if (obj.get("gates") or {}).get("all_methods_lock_sha256") != p1_lock_sha:
        raise P2Refused("--p2-test-result was written under another ALL_METHODS_LOCK")
    return {"path": str(path), "file_sha256": _file_sha(path), "self_sha256": obj.get(RR.SELF_KEY)}


def check_locked_evaluations(register_path, p1_lock_sha: str, p2_lock_sha: str) -> dict:
    """PREG2 §6.3 S2 / §13 item 6: the (already chain-verified) TEST_EXPOSURE_REGISTER holds a LOCKED_EVALUATION
    RETRO_TEST entry for the Phase-1 lock and one whose all_methods_lock_sha256 is the P2 lock (naming
    P2_METHODS_LOCK in observations_description); the register tool is never edited."""
    reg = RR._strict_json(Path(register_path))
    ents = [e for e in reg.get("entries") or [] if isinstance(e, Mapping) and e.get("access_purpose") ==
            "LOCKED_EVALUATION" and e.get("split") == "RETRO_TEST"]
    p1 = [e for e in ents if e.get("all_methods_lock_sha256") == p1_lock_sha]
    p2 = [e for e in ents if e.get("all_methods_lock_sha256") == p2_lock_sha]
    if not p1:
        raise P2Refused("TEST_EXPOSURE_REGISTER has no LOCKED_EVALUATION RETRO_TEST entry for the Phase-1 "
                        "ALL_METHODS_LOCK (PREG2 §6.3 S2)")
    if not p2:
        raise P2Refused("TEST_EXPOSURE_REGISTER has no LOCKED_EVALUATION RETRO_TEST entry whose all_methods_lock_sha256 "
                        "is the P2_METHODS_LOCK sha256 (PREG2 §6.3 S2): Phase-2 TEST scoring is refused")
    e2 = min(p2, key=lambda x: int(x["entry_id"]))
    if "P2_METHODS_LOCK" not in str(e2.get("observations_description") or ""):
        raise P2Refused("the Phase-2 LOCKED_EVALUATION entry does not name P2_METHODS_LOCK in observations_description "
                        "(PREG2 §13 item 6)")
    e1 = min(p1, key=lambda x: int(x["entry_id"]))
    return {"phase1_entry_id": int(e1["entry_id"]), "phase1_entry_sha256": e1.get("entry_sha256"),
            "p2_entry_id": int(e2["entry_id"]), "p2_entry_sha256": e2.get("entry_sha256")}


# ------------------------------------------------------------------------------------------------ the block
def _raw_entries(idx: Mapping) -> dict:
    out = {}
    for e in idx.get("entries") or []:
        seed = e.get("seed")
        out[f"{e.get('arm')}|{None if seed is None else int(seed)}|{e.get('rule')}|{e.get('role', 'MAIN')}"] = e
    return out


def _frozen_fl_arms(block: Path, idx: Mapping) -> tuple:
    """The P2 FL arms whose MAIN run dirs froze the item tables (RUN_SUMMARY frozen_item_tables); all seeds agree."""
    seen = {}
    for e in idx.get("entries") or []:
        if e.get("arm") in P2_FL_ARMS and e.get("run_dir"):
            p = RR._inside(block, f"{e['run_dir']}/RUN_SUMMARY.json")
            fz = bool(RR._read_json(p).get("frozen_item_tables"))
            seen.setdefault(e["arm"], set()).add(fz)
    mixed = sorted(a for a, v in seen.items() if len(v) > 1)
    if mixed:
        raise P2Refused(f"P2 FL arms {mixed}: some seeds froze the item tables and some did not (one recipe per arm)")
    return tuple(sorted(a for a, v in seen.items() if True in v))


def precheck_index(block: Path, split: str) -> dict:
    """Phase-2 block shape before the library load: study PREG2, only the admitted arms, role MAIN only, PRE present,
    phase1_reused exactly on the Phase-1 arms."""
    p = block / RR.INDEX_NAME
    if not p.is_file():
        raise P2Refused(f"{p} is missing")
    idx = RR._read_json(p)
    if idx.get("study") != P2_STUDY:
        raise P2Refused(f"{RR.INDEX_NAME}: study {idx.get('study')!r} is not {P2_STUDY} (tools/p2_block_index.py)")
    if str(idx.get("split", "")).upper() != split:
        raise P2Refused(f"{RR.INDEX_NAME} declares split {idx.get('split')!r}, the CLI --split {split}")
    if set(idx.get("roles") or {}) != {"MAIN"}:
        raise P2Refused(f"{RR.INDEX_NAME}: roles {sorted(idx.get('roles') or {})}; every Phase-2 contrast of the "
                        "sitting is on MAIN (Q14's TEST2 panel is TEST2_NOT_AVAILABLE)")
    for i, e in enumerate(idx.get("entries") or []):
        arm = e.get("arm")
        if arm not in P2_BLOCK_ARMS:
            raise P2Refused(f"entry {i}: arm {arm!r} is not a Phase-2 block arm {P2_BLOCK_ARMS}")
        if bool(e.get("phase1_reused")) != (arm in PHASE1_REUSED_ARMS):
            raise P2Refused(f"entry {i} ({arm}): phase1_reused must be true exactly for the reused Phase-1 arms "
                            f"{PHASE1_REUSED_ARMS} (S5)")
        if e.get("role", "MAIN") != "MAIN":
            raise P2Refused(f"entry {i} ({arm}): role {e.get('role')!r} is not MAIN")
    if not any(e.get("arm") == P2_ANCHOR for e in idx.get("entries") or []):
        raise P2Refused(f"the block has no {P2_ANCHOR} entry: Q1 / Q2 / Q7 / Q8 / Q12b have no anchor")
    return idx


def check_rows_study(entries) -> None:
    """P2 rows are labelled study PREG2 (launch_fl.p2_row_labels; the P2-13 reeval_p2 rows must carry it); a reused
    Phase-1 row never is."""
    for e in entries:
        studies = {r.get("study") for r in e.rows.values()}
        if e.arm in P2_ARMS and studies != {P2_STUDY}:
            raise P2Refused(f"{e.eid}: P2 rows must carry study {P2_STUDY}, they carry {sorted(map(str, studies))}")
        if e.arm in PHASE1_REUSED_ARMS and P2_STUDY in studies:
            raise P2Refused(f"{e.eid}: a reused Phase-1 row labelled study {P2_STUDY} (S5: Phase-1 rows come from the "
                            "Phase-1 plan only)")


def check_sealed(entries, raw: Mapping, lock: Mapping) -> dict:
    """S4 on TEST: every P2 entry is a sealed (arm, seed, rule) of the lock with the same checkpoint sha256; no
    declared-not-run arm enters the block."""
    dnr = lock["declared_not_run"]
    n = 0
    for e in entries:
        if e.arm not in P2_ARMS:
            continue
        for d in dnr:
            if d["arm"] == e.arm and (d["seed"] is None or d["seed"] == e.seed):
                raise P2Refused(f"{e.eid}: {e.arm} is declared not run in P2_METHODS_LOCK ({d['reason']}); S4: no "
                                "second sitting, it never enters the block")
        k = (e.arm, e.seed, e.rule)
        if k not in lock["sealed"]:
            raise P2Refused(f"{e.eid}: not a sealed arm of P2_METHODS_LOCK (S4: an arm without a sealed checkpoint at "
                            "lock time is declared not run)")
        if e.arm in P2_DEVICE_ARMS:
            what, got = "p2_base.checkpoint_sha256", (raw[e.eid].get("p2_base") or {}).get("checkpoint_sha256")
            want = lock["sealed"][k]["base_checkpoint_sha256"]
        else:
            what, got, want = "checkpoint_sha256", raw[e.eid].get("checkpoint_sha256"), lock["sealed"][k]["checkpoint_sha256"]
        if got != want:
            raise P2Refused(f"{e.eid}: {what} {str(got)[:12]}… is not the sealed {str(want)[:12]}… (P2_METHODS_LOCK)")
        n += 1
    return {"n_sealed_entries_checked": n}


def check_phase1_binding(entries, raw: Mapping, files: Mapping, lock: Mapping, p1: Mapping) -> dict:
    """S5 on TEST: the reused Phase-1 entries are exactly the lock's phase1_reused list (no arm scored twice), carry
    its checkpoint sha256 where the lock names one, and their rows / ranks files are byte-identical (sha256) to the
    files the sitting's Phase-1 RETENTION_RESULT recorded under the same block paths."""
    want = lock["phase1_reused"]
    got = {(e.arm, e.seed, e.rule): e for e in entries if e.arm in PHASE1_REUSED_ARMS}
    extra = sorted(set(got) - set(want), key=str)
    missing = sorted(set(want) - set(got), key=str)
    if extra or missing:
        raise P2Refused(f"S5: the reused Phase-1 entries differ from P2_METHODS_LOCK phase1_reused (extra {extra}, "
                        f"missing {missing})")
    p1files = p1["files_sha256"]
    out = {}
    for k, e in sorted(got.items(), key=lambda kv: str(kv[0])):
        ck = want[k]
        if ck is not None and raw[e.eid].get("checkpoint_sha256") != ck:
            raise P2Refused(f"S5: {e.eid} checkpoint_sha256 is not the P2_METHODS_LOCK phase1_reused binding")
        rec = {}
        for what, rel in (("rows", e.rows_rel), ("ranks", e.ranks_rel)):
            if rel is None:
                raise P2Refused(f"S5: {e.eid} has no {what} file")
            if rel not in p1files:
                raise P2Refused(f"S5: {e.eid} {what} {rel} is not a file of the Phase-1 RETENTION_RESULT block")
            if files.get(rel) != p1files[rel]:
                raise P2Refused(f"S5: {e.eid} {what} {rel} sha256 differs from the Phase-1 sitting's file (a reused "
                                "Phase-1 row is produced once and hash-bound into both blocks)")
            rec[what] = {"path": rel, "sha256": p1files[rel]}
        out[e.eid] = rec
    return out


def base_problem(e, raw: Mapping, lock: Optional[Mapping], split: str, entries) -> Optional[str]:
    """Q5 base_best_only: an FT_P2_WSTAR entry pairs only when its index `p2_base` names the same-seed P2_WSTAR best.pt
    (TEST: the sealed PRACTICAL_BEST checkpoint; VALIDATION: the block's P2_WSTAR PRACTICAL_BEST checkpoint when
    recorded). None when it does."""
    b = raw.get(e.eid, {}).get("p2_base")
    if not isinstance(b, Mapping):
        return "no p2_base (the LAUNCH start of the W* base)"
    if b.get("arm") != "P2_WSTAR" or b.get("seed") != e.seed or b.get("checkpoint") != "best.pt" or \
            not _hex64(b.get("checkpoint_sha256")):
        return f"p2_base {dict(b)} is not the same-seed P2_WSTAR best.pt"
    want = None
    if split == "TEST" and lock is not None:
        want = (lock["sealed"].get(("P2_WSTAR", e.seed, "PRACTICAL_BEST")) or {}).get("checkpoint_sha256")
    else:
        w = next((x for x in entries if (x.arm, x.seed, x.rule) == ("P2_WSTAR", e.seed, "PRACTICAL_BEST")), None)
        want = raw.get(w.eid, {}).get("checkpoint_sha256") if w is not None else None
    if want is not None and b.get("checkpoint_sha256") != want:
        return "p2_base checkpoint_sha256 is not the same-seed P2_WSTAR PRACTICAL_BEST checkpoint"
    return None


def load_p2_block(block: Path, split: str, loader, gate: Optional[Mapping]):
    """Phase-2 block = the frozen library's load_block inside p2_profile (every Phase-1 check applies to every entry),
    after the Phase-2 shape check."""
    idx = precheck_index(block, split)
    frozen = _frozen_fl_arms(block, idx) if split == "VALIDATION" else ()
    with p2_profile(frozen):
        blk = RR.load_block(block, split, loader, gate=gate)
    blk.raw = _raw_entries(blk.idx)
    blk.frozen_fl_arms = list(frozen)
    check_rows_study(blk.entries)
    return blk


def selection_info(block: Path, idx: Mapping, lock: Mapping) -> dict:
    """The CALIB selection records named in the block (records.<KEY>), bound to the lock's selection_files_sha256; the
    FINAL record's CALIB_GAIN / NO_MEASURABLE_CALIB_GAIN / EDGE_UNRESOLVED / FUTILITY_SKIP_WB_WC flags (PREG2 §5.6 -
    §5.8) printed beside every Q, D-j beside the futility flag."""
    recs = idx.get("records") or {}
    bad = sorted(k for k in recs if k not in P2_RECORD_KEYS)
    if bad:
        raise P2Refused(f"{RR.INDEX_NAME}: records {bad} are not Phase-2 records {P2_RECORD_KEYS}")
    out = {"files": {}}
    for key in SELECTION_RECORDS:
        bound = lock["selection_files_sha256"].get(key)
        rel = recs.get(key)
        if rel is None:
            if bound is not None:
                raise P2Refused(f"P2_METHODS_LOCK binds {key} but the block carries no records.{key}")
            continue
        sha = _file_sha(RR._inside(block, rel))
        if bound is not None and sha != bound:
            raise P2Refused(f"records.{key} sha256 {sha[:12]}… is not the P2_METHODS_LOCK binding {bound[:12]}…")
        out["files"][key] = {"path": rel, "sha256": sha, "bound_by_lock": bound is not None}
    fin = recs.get("P2_WARM_SELECTION_FINAL")
    if fin is None:
        out["beside"] = {"status": "NOT_AVAILABLE", "reason": "no P2_WARM_SELECTION_FINAL record in the block"}
        return out
    obj = RR._strict_json(RR._inside(block, fin))
    if obj.get("format") != FINAL_FORMAT or RR._self_ok(obj) != "VERIFIED":
        raise P2Refused(f"records.P2_WARM_SELECTION_FINAL is not a self-verified {FINAL_FORMAT} file")
    flags = list(obj.get("flags") or [])
    beside = {"W_STAR": obj.get("winner"), "CALIB_GAIN": obj.get("calib_gain"),
              "NO_MEASURABLE_CALIB_GAIN": bool(obj.get("no_measurable_calib_gain")),
              "EDGE_UNRESOLVED": "EDGE_UNRESOLVED" in flags, "FUTILITY_SKIP_WB_WC": "FUTILITY_SKIP_WB_WC" in flags,
              "flags": flags, "label": "SELECTION (CALIB values; PREG2 §5.6-§5.8)",
              "rule": "CALIB_GAIN = W*_CALIB - PRE_CALIB, NO_MEASURABLE_CALIB_GAIN iff < 0.001 (W* is carried to MAIN "
                      "regardless); EDGE_UNRESOLVED disclosed beside every Q (§5.7)"}
    if beside["FUTILITY_SKIP_WB_WC"]:
        beside["D-j"] = DISCLOSURES["D-j"]
    out["beside"] = beside
    return out


def check_wstar_runs(block: Path, entries, sel: Mapping) -> int:
    """VALIDATION: a P2_WSTAR run dir carries the registered W* recipe label (RUN_SUMMARY arm_spec.p2_label = the
    P2_WARM_SELECTION_FINAL winner), when both are present."""
    win = (sel.get("beside") or {}).get("W_STAR")
    n = 0
    for e in entries:
        if e.arm == "P2_WSTAR" and e.run_dir_rel and win is not None:
            summ = RR._read_json(RR._inside(block, f"{e.run_dir_rel}/RUN_SUMMARY.json"))
            lab = (summ.get("arm_spec") or {}).get("p2_label")
            if lab != win:
                raise P2Refused(f"{e.eid}: the run's p2_label {lab!r} is not the registered W* {win!r} "
                                "(P2_WARM_SELECTION_FINAL)")
            n += 1
    return n


# ------------------------------------------------------------------------------------------------ the engine
class P2Engine(RR.Engine):
    """retention_report.Engine with the Phase-2 resample plan: one shared W per user set from
    v4eval.bootstrap.make_plan(seed=20261004, 1,000 resamples) (PREG2 F2 / §9.2). Everything else is inherited."""

    def view(self, vkey):
        if vkey not in self._views:
            role, name, keys = vkey
            np = RR.LIB.np
            m = self.ctx[role].manifest
            if keys is None and (role, name) in self.row_masks:
                mv = dataclasses.replace(m, eval_mask=m.eval_mask & self.row_masks[(role, name)])
            else:
                mv = m if keys is None else RR._subset_view(m, np.asarray(keys, dtype=np.int64))
            if not bool(np.any(mv.eval_mask)):
                self._views[vkey] = None
            else:
                plan = RR.LIB.B.make_plan(np.unique(mv.eval_user_key[mv.eval_mask]), seed=P2_BOOTSTRAP_SEED,
                                          n_resamples=P2_N_RESAMPLES)
                if plan.seed != P2_BOOTSTRAP_SEED or plan.n_resamples != P2_N_RESAMPLES:
                    raise P2Refused("the resample plan is not seed 20261004 / 1,000 resamples")
                self._views[vkey] = (mv, plan)
        return self._views[vkey]


# ------------------------------------------------------------------------------------------------ readings
def tost(boot, interval) -> dict:
    """TOST at alpha 0.05 <=> the 90% percentile CI of the bootstrap distribution lies inside `interval`."""
    if boot is None:
        return {"status": "NOT_AVAILABLE", "reason": "no bootstrap distribution"}
    lo90, hi90 = (float(x) for x in RR.LIB.np.percentile(boot, [5.0, 95.0]))
    lo, hi = (float(x) for x in interval)
    return {"status": "COMPUTED", "margin": [lo, hi], "ci90": [lo90, hi90], "equivalent": bool(lo <= lo90 and hi90 <= hi)}


def q1_equivalence(eng, key, b: Mapping) -> dict:
    out = tost(eng.boot.get(key), Q1_INTERVAL)
    if out["status"] == "COMPUTED":
        out.update(ci95_inside_margin=bool(isinstance(b.get("ci95"), list) and Q1_INTERVAL[0] <= b["ci95"][0]
                                            and b["ci95"][1] <= Q1_INTERVAL[1]),
                   rule="TOST at alpha 0.05: Q1 is EQUIVALENT iff its 90% paired user-bootstrap percentile CI lies "
                        "inside [-0.0005, 0.0005] (absolute MRR@20; PREG2 §9.4, D-h)",
                   registered="PREG2 §9.4 / P2_CONTRASTS_v3 reading_rules.Q1")
    return out


def q13_equivalence(eng, key, b: Mapping, rule_text: str) -> dict:
    """Q13: the library's k14_equivalence (TOST, 90% CI inside READING_RULES['K14'].interval) — asserted to be the
    registered [99, 101] — with Q13's own rule text."""
    if [float(x) for x in RR.READING_RULES["K14"]["interval"]] != list(Q13_INTERVAL):
        raise P2Refused("retention_report K14 interval is not [99, 101]: k14_equivalence cannot serve Q13")
    out = dict(RR.k14_equivalence(eng, key, b))
    out.update(rule=rule_text or out.get("rule"), registered="PREG2 §7.4 / §9.1; P2_CONTRASTS_v3 reading_rules.Q13")
    chk = tost(eng.boot.get(key), Q13_INTERVAL)
    if out.get("status") == "COMPUTED" and chk.get("equivalent") != out.get("equivalent"):
        raise P2Refused("Q13: k14_equivalence and the generic TOST disagree")
    return out


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def q1_reading(rec: Mapping) -> dict:
    """PREG2 §9.4 on Q1's confirmatory block."""
    b = (rec.get("by_rule") or {}).get(rec.get("confirmatory_rule"), {})
    out = {"rule": READING_RULE_Q1, "D-h": DISCLOSURES["D-h"]}
    ci = b.get("ci95")
    if not (isinstance(ci, list) and _is_num(b.get("point"))):
        return dict(out, reading="NOT_AVAILABLE", reason=b.get("status") or "no Q1 statistic", conditions_true=[])
    per = [p.get("point") for p in b.get("per_seed") or []]
    seeds_ok = len(per) == 2 and all(_is_num(x) for x in per) and b.get("seeds_status") == "COMPLETE"
    success = ci[0] > 0 and seeds_ok and all(x > 0 for x in per)
    eq = b.get("equivalence") or {}
    equivalent = bool(eq.get("equivalent"))
    failure = ci[1] < 0
    cond = [n for n, v in (("SUCCESS", success), ("EQUIVALENT", equivalent), ("FAILURE", failure)) if v]
    out.update(reading=cond[0] if cond else "INCONCLUSIVE", conditions_true=cond, overlap=len(cond) > 1,
               ci95=ci, ci90=eq.get("ci90"), per_seed_points=per, both_seeds_positive=bool(seeds_ok and
                                                                                          all(x > 0 for x in per)))
    if ci[0] > 0 and not seeds_ok:
        out["success_blocked"] = "the 95% CI lies above 0 but both per-seed points are not available / positive"
    return out


def holm_positive(rec: Optional[Mapping]) -> bool:
    if not rec:
        return False
    b = (rec.get("by_rule") or {}).get(rec.get("confirmatory_rule"), {})
    h = b.get("holm") or {}
    return bool(h.get("holm_reject")) and _is_num(b.get("point")) and float(b["point"]) > float(h.get("reference", 0.0))


def deployment_grade(q1r: Mapping, recs: Mapping) -> dict:
    q5, q7 = holm_positive(recs.get("Q5")), holm_positive(recs.get("Q7"))
    ok = q1r.get("reading") == "SUCCESS" and q5 and q7
    return {"rule": DEPLOYMENT_RULE, "Q1_SUCCESS": q1r.get("reading") == "SUCCESS", "Q5_holm_positive": q5,
            "Q7_holm_positive": q7, "deployment_grade": bool(ok)}


def interpretation_grid(q1r: Mapping, q2: Optional[Mapping], eng) -> dict:
    """The PREG2 §9.4 grid with GRID_DEFINITIONS (registered row order decides)."""
    st = {"Q1_POSITIVE": q1r.get("reading") == "SUCCESS",
          "Q1_NONPOSITIVE_OR_EQUIVALENT": q1r.get("reading") in ("FAILURE", "EQUIVALENT"),
          "Q1_EQUIVALENT": "EQUIVALENT" in (q1r.get("conditions_true") or [])}
    q2eq = {"status": "NOT_AVAILABLE"}
    if q2:
        q2eq = tost(eng.boot.get(("Q2", q2["confirmatory_rule"])), Q1_INTERVAL)
    st["Q2_POSITIVE"] = holm_positive(q2)
    st["Q2_EQUIVALENT"] = bool(q2eq.get("equivalent"))
    hit = next(((a, b, t) for a, b, t in GRID if st[a] and st[b]), None)
    return {"states": st, "q2_equivalence": q2eq, "definitions": dict(GRID_DEFINITIONS),
            "cell": "NO_REGISTERED_CELL" if hit is None else f"{hit[0]} x {hit[1]}",
            "reading": GRID_DEFINITIONS["NO_REGISTERED_CELL"] if hit is None else hit[2],
            "grid": [{"Q2": a, "Q1": b, "reading": t} for a, b, t in GRID], "label": L_DESC + " (registered reading grid)"}


# ------------------------------------------------------------------------------------------------ contrasts
def _pairs_for(book, c: Mapping, rule: str, raw: Mapping, lock, split, entries):
    kid = c["id"]
    shared = kid in P2_SHARED_DEN and c["den"] == P2_SHARED_DEN[kid]
    pairs = book.pairs("MAIN", c["num"], c["den"], rule, shared=shared)
    dropped = []
    if kid in P2_BASE_BEST_ONLY:
        keep = []
        for p in pairs:
            why = base_problem(p[0], raw, lock, split, entries)
            if why is None:
                keep.append(p)
            else:
                dropped.append({"entry": p[0].eid, "reason": why})
        pairs = keep
    return pairs, shared, dropped


def _declared_for(c: Mapping, lock: Optional[Mapping]) -> list:
    if not lock:
        return []
    return [d for d in lock["declared_not_run"] if d["arm"] in (c["num"], c["den"])]


def q_record(eng, book, c: Mapping, ctx: SimpleNamespace) -> dict:
    """One Phase-2 contrast: both rules (the registered rule labelled per §6.4, the other DESCRIPTIVE), exactly its
    registered seeds (others -> DESCRIPTIVE extra_seeds), the library's ratio / delta with the harmed share, the seed
    statistics and the seed-vs-seed null; Q1 / Q13 equivalence under the registered rule."""
    kid = c["id"]
    crule = P2_RULE_MAP.get(kid, P2_DEFAULT_RULE)
    reg = tuple(P2_REGISTERED_SEEDS.get(kid, P2_DEFAULT_SEEDS))
    label = p2_label(kid, ctx.split)
    rec = {"id": kid, "contrast": c["contrast"], "question": c["question"], "kind": c["kind"], "num": c["num"],
           "den": c["den"], "population": c["population"], "tier": tier_of(kid), "label": label,
           "confirmatory_rule": crule, "registered_seeds": list(reg), "by_rule": {},
           "pairing_note": ctx.contrasts["pairing_notes"].get(kid)}
    if label == L_CONF:
        rec["disclosure_D_a"] = DISCLOSURES["D-a"]
    if kid in ("Q6", "Q7", "Q11"):
        rec["disclosure_D_i"] = DISCLOSURES["D-i"]
    if kid in P2_DEPENDENCIES:
        rec["dependency"] = P2_DEPENDENCIES[kid]
    dnr = _declared_for(c, ctx.lock)
    if dnr:
        rec["declared_not_run"] = dnr
    for rule in RR.RULES:
        rlabel = label if rule == crule else (L_PVE if ctx.split == "VALIDATION" else L_DESC)
        all_pairs, shared, dropped = _pairs_for(book, c, rule, ctx.raw, ctx.lock, ctx.split, ctx.entries)
        pairs = [p for p in all_pairs if RR._pair_seed(p, shared) in reg]
        extra = [p for p in all_pairs if RR._pair_seed(p, shared) not in reg]
        got = sorted({RR._pair_seed(p, shared) for p in pairs})
        missing = [s for s in reg if s not in got] if (rule == crule or all_pairs) else []
        if pairs:
            b = RR._contrast_stat(eng, c, pairs, kid, shared=shared, boot_key=(kid, rule))
            vk = RR._vkey_for(pairs, "MAIN")
            b["harmed_share_seed_null"] = {"numerator": RR.seed_null(eng, book.get("MAIN", c["num"], rule), vk),
                                           "denominator": RR.seed_null(eng, book.get("MAIN", c["den"], rule), vk)}
        else:
            b = {"status": "DECLARED_NOT_RUN" if dnr else "NOT_AVAILABLE"}
            if dnr:
                b["reason"] = "; ".join(f"{d['arm']}: {d['reason']}" for d in dnr)
        if dropped:
            b["base_best_only_excluded"] = dropped
        b["label"] = rlabel
        b["seeds_status"] = ("INCOMPLETE_SEEDS" if (missing and all_pairs) else "NOT_RUN" if missing else
                             "COMPLETE" if pairs else "NOT_AVAILABLE")
        if missing:
            b["missing_registered_seeds"] = missing
        if kid in DP_QS and rule != crule:
            b["disclosure"] = ctx.contrasts.get("dp_disclosure")
        if rule == crule and pairs:
            if kid == "Q1":
                b["equivalence"] = q1_equivalence(eng, (kid, rule), b)
            elif kid == "Q13":
                b["equivalence"] = q13_equivalence(eng, (kid, rule), b,
                                                   (ctx.contrasts["reading_rules"].get("Q13") or {}).get("rule"))
        rec["by_rule"][rule] = b
        if extra:
            st = RR._contrast_stat(eng, c, all_pairs, f"{kid} all seeds", shared=shared)
            st.update(label=L_PVE if ctx.split == "VALIDATION" else L_DESC,
                      note=f"mean over every available seed, including seeds outside the registered set {list(reg)}")
            rec.setdefault("extra_seeds", {})[rule] = st
    rec["decision"] = q_decision(rec)
    return rec


def q_decision(rec: Mapping) -> dict:
    b = rec["by_rule"].get(rec["confirmatory_rule"], {})
    ref = 100.0 if rec["kind"] == "retained" else 0.0
    side = RR._side(b, ref) if isinstance(b.get("ci95"), list) else (b.get("status") or "NOT_AVAILABLE")
    return {"reference": ref, "unadjusted_ci95_side": side, "label": rec["label"]}


def test2_record(c: Mapping) -> dict:
    return {"id": c["id"], "contrast": c["contrast"], "question": c["question"], "kind": c["kind"], "num": c["num"],
            "den": c["den"], "population": c["population"], "tier": tier_of(c["id"]), "label": L_T2NA,
            "family": "TEST2 replication (separate family; never in the RETRO_TEST band)", "status": L_T2NA,
            "reason": TEST2_REASON, "by_rule": {}, "confirmatory_rule": P2_RULE_MAP.get(c["id"], P2_DEFAULT_RULE),
            "registered_seeds": list(P2_REGISTERED_SEEDS.get(c["id"], P2_DEFAULT_SEEDS))}


def p2_holm(eng, conf: Sequence[dict], split: str) -> dict:
    """PREG2 §9.3 through retention_report.holm_band: Holm step-down over Q2-Q8 (alpha 0.05, family size 7), the
    DESCRIPTIVE tertiary band over Q9 / Q11 / Q12a / Q12b / Q13, Q1 outside Holm; labels rewritten in Phase-2 terms."""
    fam_recs = [r for r in conf if r["id"] in P2_RETRO_TEST_IDS]
    out = RR.holm_band(eng, fam_recs, {"holm_family": list(P2_HOLM_FAMILY),
                                       "tertiary_band_family": list(P2_RETRO_TEST_BAND),
                                       "n_contrasts": len(P2_RETRO_TEST_IDS)})
    for r in fam_recs:
        b = r["by_rule"][r["confirmatory_rule"]]
        h = b.get("holm") or {}
        if r["id"] in P2_PRIMARY:
            b["holm"] = {"status": "NOT_IN_HOLM_FAMILY", "tier": "primary",
                         "rule": "PREG2 §9.3: Q1 alone is primary, read unadjusted at alpha 0.05, outside Holm"}
        elif r["id"] in P2_HOLM_FAMILY and h.get("status") == "COMPUTED":
            h["label"] = (L_PVE if split == "VALIDATION" else
                          "CONFIRMATORY_P2 multiplicity decision (Holm step-down over Q2-Q8, PREG2 §9.3)")
            h["family"] = "SECONDARY_HOLM_Q2_Q8"
        elif r["id"] in P2_RETRO_TEST_BAND and h.get("status") == "COMPUTED":
            h["decision"] = "none (TERTIARY_P2: unadjusted, DESCRIPTIVE Holm band; PREG2 §9.3)"
            h["label"] = L_DESC
        if r["id"] in P2_HOLM_FAMILY:
            r["decision"]["holm_reject"] = (b.get("holm") or {}).get("holm_reject")
    out.update(primary=list(P2_PRIMARY), secondary_family=list(P2_HOLM_FAMILY),
               tertiary_band_family=list(P2_RETRO_TEST_BAND), test2_family=list(P2_TEST2_FAMILY),
               method=out.get("method", "") + " (reused unchanged from retention_report.holm_band; PREG2 §9.3)")
    return out


def q10_headroom(eng, book, ctx) -> dict:
    """Q10 = Q1 / Q2 (PREG2 §9.1 / §9.3; P2_CONTRASTS_v3 algebraic_dependencies): per seed s, Q10_s = (W*_s - PRE) /
    (C1_FULL_s - PRE); the seed mean, and its percentile CI from the per-resample seed mean of the same ratio on the
    same W. DERIVED, DESCRIPTIVE. Undefined unless every per-seed Q2 point (and, for the CI, every resample) is > 0."""
    rule = P2_DEFAULT_RULE
    out = {"id": "Q10", "relation": "Q10_s = Q1_s / Q2_s (per seed and per resample, same W)",
           "label": (L_PVE if ctx.split == "VALIDATION" else L_DESC) + " (derived; not a registered contrast item)",
           "rule": rule}
    q1p = book.pairs("MAIN", "P2_WSTAR", "PRE", rule, shared=True)
    q2p = book.pairs("MAIN", "P2_C1_FT_CENTRAL_FULL", "PRE", rule, shared=True)
    s1 = {n.seed: (n, d) for n, d in q1p if n.seed in P2_DEFAULT_SEEDS}
    s2 = {n.seed: (n, d) for n, d in q2p if n.seed in P2_DEFAULT_SEEDS}
    seeds = sorted(set(s1) & set(s2))
    if not seeds or not all(n.ranked and d.ranked for s in seeds for n, d in (s1[s], s2[s])):
        return dict(out, status="NOT_AVAILABLE", reason="needs Q1 and Q2 pairs with rank files on a common seed")
    np = RR.LIB.np
    vk = RR.full_view("MAIN")
    per, ratios, bad = [], [], 0
    for s in seeds:
        (w, pre), (c1, pre2) = s1[s], s2[s]
        p1 = float(eng.point(w, vk, RR.PRIMARY)) - float(eng.point(pre, vk, RR.PRIMARY))
        p2 = float(eng.point(c1, vk, RR.PRIMARY)) - float(eng.point(pre2, vk, RR.PRIMARY))
        d1 = eng.res(w, vk, RR.PRIMARY) - eng.res(pre, vk, RR.PRIMARY)
        d2 = eng.res(c1, vk, RR.PRIMARY) - eng.res(pre2, vk, RR.PRIMARY)
        bad = max(bad, int(np.count_nonzero(d2 <= 0)))
        per.append({"seed": s, "Q1_point": p1, "Q2_point": p2,
                    "point": (p1 / p2) if p2 > 0 else RR.undef("Q2_POINT_NOT_POSITIVE")})
        ratios.append(np.where(d2 > 0, d1 / np.where(d2 > 0, d2, 1.0), np.nan))
    out.update(seeds=seeds, per_seed=per, n_resamples_q2_not_positive=bad)
    if any(RR.is_undef(p["point"]) for p in per):
        return dict(out, status="UNDEFINED", point=RR.undef("Q2_POINT_NOT_POSITIVE"),
                    ci95=RR.undef("Q2_POINT_NOT_POSITIVE"))
    point = float(np.mean([p["point"] for p in per]))
    out.update(status="COMPUTED", point=point, headroom_captured_percent=100.0 * point)
    if bad:
        out["ci95"] = RR.undef("Q2_RESAMPLE_NOT_POSITIVE")
    else:
        out["ci95"] = RR.LIB.B.ci95(np.mean(ratios, axis=0))
    return out


def value_table(eng, book, split: str) -> list:
    """DESCRIPTIVE (VALIDATION: POST_VALIDATION_EXPLORATORY) values of every block arm: macro-E2E MRR@20, NDCG@10,
    HR@10 per seed and rule."""
    rows = []
    for arm in P2_BLOCK_ARMS:
        for rule in RR.RULES:
            ents = book.get("MAIN", arm, rule)
            for seed, e in sorted(ents.items(), key=lambda kv: kv[0] or 0):
                rows.append({"arm": arm, "seed": seed, "rule": rule, "entry": e.eid,
                             "phase1_reused": arm in PHASE1_REUSED_ARMS,
                             "label": L_PVE if split == "VALIDATION" else L_DESC,
                             "values": {f"{m}@{k}": eng.point(e, RR.full_view("MAIN", e.subset_keys), (m, k))
                                        for m, k in RR.METRICS}})
    return rows


def descriptive_sections(block: Path, idx: Mapping) -> dict:
    """E1-E5, devices, D1-ext: DESCRIPTIVE (tertiary) summaries of the records the block carries (sha256, self-sha
    status, top-level scalars); NOT_RUN when absent. E1's ε_LB keys are withheld unless pre_null_valid is true."""
    recs = idx.get("records") or {}
    out = {}
    for key, what in DESCRIPTIVE_RECORDS:
        rel = recs.get(key)
        if rel is None:
            out[key] = {"status": "NOT_RUN", "what": what, "label": L_DESC_TERT}
            continue
        p = RR._inside(block, rel)
        obj = RR._strict_json(p)
        sc = RR._scalars(obj) if isinstance(obj, Mapping) else {}
        sc.pop(RR.SELF_KEY, None)
        rec = {"status": "REPORTED", "what": what, "label": L_DESC_TERT, "path": rel, "sha256": _file_sha(p),
               "self_sha256_status": RR._self_ok(obj), "scalars": sc}
        if key == "E1_MIA":
            rec["eps_rule"] = E1_EPS_RULE
            if not (isinstance(obj, Mapping) and obj.get("pre_null_valid") is True):
                rec["scalars"] = {k: v for k, v in sc.items() if "eps" not in k.lower()}
                rec["eps_lb"] = "WITHHELD (PRE-null validity not recorded as passed)"
        out[key] = rec
    return out


def plan_pairs_p2(eng, book, ctx) -> dict:
    """--check-only: every pairing of every Q under both rules through check_pairs and every full-population ratio
    denominator through require_nonzero_denominator; prints nothing numeric."""
    n_pairs = n_den = 0
    for c in ctx.contrasts["items"]:
        if c["id"] in P2_TEST2_FAMILY:
            continue
        for rule in RR.RULES:
            pairs, shared, _dropped = _pairs_for(book, c, rule, ctx.raw, ctx.lock, ctx.split, ctx.entries)
            if not pairs:
                continue
            RR.check_pairs(pairs, shared)
            n_pairs += len(pairs)
            if c["kind"] == "retained":
                vk = RR._vkey_for(pairs, "MAIN")
                for _n, d in pairs:
                    RR.require_nonzero_denominator(eng.point(d, vk, RR.PRIMARY), f"{c['id']} {d.eid}")
                    n_den += 1
    return {"n_pairs_checked": n_pairs, "n_denominators_checked": n_den}


# ------------------------------------------------------------------------------------------------ assemble
def evaluator_w_check(blk) -> dict:
    """E7.2 (A-F15) unchanged: a row's non-null bootstrap_W_sha256 is the EVALUATOR's registered W of the entry's view
    (v4train.bootstrap_ci, seed 20260926) — checked with the library's own Engine; the Phase-2 statistics then use their
    own plan (seed 20261004), recorded separately."""
    has_w = any(r.get("bootstrap_W_sha256") for e in blk.entries for _w, rows in RR._entry_rowsets(e)
                for r in rows.values())
    return RR.check_bootstrap_w(RR.Engine(blk.ctx), blk.entries) if has_w else {}


def analyse(blk, ctx: SimpleNamespace) -> dict:
    eng = P2Engine(blk.ctx)
    book = RR.Book(blk.entries)
    w_checked = evaluator_w_check(blk)
    conf, t2 = [], []
    for c in ctx.contrasts["items"]:
        if c["id"] in P2_TEST2_FAMILY:
            t2.append(test2_record(c))
        else:
            conf.append(q_record(eng, book, c, ctx))
    holm = p2_holm(eng, conf, ctx.split)
    recs = {r["id"]: r for r in conf}
    q1r = q1_reading(recs["Q1"])
    beside = ctx.selection.get("beside")
    for r in conf:
        r["beside"] = beside
    main_view = eng.meta(RR.full_view("MAIN"))
    return {"contrasts": conf + t2, "holm": holm, "q1_reading": q1r,
            "interpretation_grid": interpretation_grid(q1r, recs.get("Q2"), eng),
            "deployment_grade": deployment_grade(q1r, recs), "q10_headroom_capture": q10_headroom(eng, book, ctx),
            "test2": {"status": L_T2NA, "reason": TEST2_REASON, "Q1_replication": L_T2NA,
                      "labels_if_available": ["REPLICATED_FRESH", "NOT_REPLICATED"], "Q14": L_T2NA},
            "values": value_table(eng, book, ctx.split),
            "_bootstrap": {"seed": P2_BOOTSTRAP_SEED, "n_resamples": P2_N_RESAMPLES, "main_view": main_view,
                           "evaluator_rows_bootstrap_W_checked_E7_2": w_checked}}


def build_result(blk, ctx: SimpleNamespace, analysis: dict, *, script_sha: str, gates: Mapping) -> dict:
    analysis = dict(analysis)
    boot = analysis.pop("_bootstrap")
    return {
        "schema": P2_SCHEMA, "split": ctx.split, "study": P2_STUDY,
        "task": "P2-12 tools/p2_report.py (PREG2 §6.3, §6.4, §9, §11)",
        "script": {"file": "tools/p2_report.py", "sha256": script_sha},
        "library": {"tools/retention_report.py": RR_SHA256, "tools/ana_block_index.py": ABI_SHA256,
                    "library_files_sha256": RR.LIB.files_sha256,
                    "reused": ["bind", "load_index", "load_block", "check_test_gate", "verify_test_ext_registry",
                               "check_ext_roles", "test_context_loader", "Engine", "Book", "ratio", "delta",
                               "_seed_stats", "seed_null", "holm_band", "k14_equivalence", "check_pairs",
                               "require_nonzero_denominator", "with_self_sha", "_write_new"]},
        "gates": dict(gates),
        "inputs": {"files_sha256": dict(sorted(blk.files.items())),
                   "contrasts": {k: ctx.contrasts[k] for k in ("file_sha256", "self_sha256", "format", "document",
                                                               "n_contrasts")},
                   "roles": {r: dict(c.record, manifest_id=c.manifest.manifest_id, n_rows=c.manifest.n,
                                     eval_ordered_decision_id_sha256=c.manifest.hashes["eval_ordered_decision_id_sha256"])
                             for r, c in sorted(blk.ctx.items())},
                   "frozen_fl_arms": blk.frozen_fl_arms},
        "checks": {"comparison_key": blk.key, "rank_row_max_abs_diff": blk.rank_row_max_abs_diff,
                   "rank_row_tolerance": RR.ROW_VALUE_TOL, "sealed": ctx.sealed_check,
                   "phase1_binding_S5": ctx.phase1_binding, "wstar_runs_checked": ctx.wstar_checked,
                   "entries": [{"entry": e.eid, "rule_source": e.rule_source, "ranked": e.ranked, "n_marks": len(e.marks),
                                "phase1_reused": e.arm in PHASE1_REUSED_ARMS, "notes": e.notes} for e in blk.entries]},
        "bootstrap": dict(boot, unit="user (eval_user_key)", statistic="macro_e2e (MRR@20)",
                          ratios="bootstrapped directly per resample", plan="v4eval.bootstrap.make_plan(seed=20261004)",
                          seed_mean="mean over seeds of the per-seed statistic, per resample on the same W"),
        "labels": {"table_6_4": [{"claim": a, "label": b} for a, b in LABELS_6_4],
                   "by_contrast": {c["id"]: c["label"] for c in analysis["contrasts"]},
                   "hierarchy": {"primary": list(P2_PRIMARY), "secondary": list(P2_HOLM_FAMILY),
                                 "tertiary": list(P2_TERTIARY)},
                   "rule_map": dict(P2_RULE_MAP), "default_rule": P2_DEFAULT_RULE,
                   "registered_seeds": {c["id"]: list(P2_REGISTERED_SEEDS.get(c["id"], P2_DEFAULT_SEEDS))
                                        for c in ctx.contrasts["items"]},
                   "multiplicity_statement": ctx.contrasts.get("multiplicity_statement"),
                   "post_validation_rule": ctx.contrasts.get("post_validation_rule"),
                   "dp_disclosure": ctx.contrasts.get("dp_disclosure"),
                   "algebraic_dependencies": ctx.contrasts.get("algebraic_dependencies")},
        "declared_not_run": list(ctx.lock["declared_not_run"]),
        "selection": ctx.selection,
        **analysis,
        "descriptive_sections": ctx.descriptive,
        "disclosures": dict(DISCLOSURES),
    }


# ------------------------------------------------------------------------------------------------ report
def _n(x, nd=6):
    return RR._num(x, nd)


def _cell_ci(x, nd=6):
    return RR._ci(x, nd)


def render_report(res: Mapping) -> str:
    L = [f"# Phase-2 analysis (PREG2) — {res['split']}", "",
         f"script tools/p2_report.py {res['script']['sha256']}; retention_report {RR_SHA256[:12]}…; bootstrap seed "
         f"{res['bootstrap']['seed']} ({res['bootstrap']['n_resamples']} resamples); MAIN users "
         f"{res['bootstrap']['main_view'].get('n_users')}; W {res['bootstrap']['main_view'].get('W_sha256')}", ""]
    if res["split"] == "VALIDATION":
        L += [f"**Every number in this report is {L_PVE}** (PREG2 §6.4; opened after both sitting analyses, S8).", ""]
    else:
        L += [f"**{L_CONF}** claims (Q1-Q8) are printed with D-a: \"{DISCLOSURES['D-a']}\"", ""]
    L += ["## Contrasts (registered rule; E2E client-macro MRR@20)", ""]
    head = ["id", "contrast", "tier", "label", "rule", "seeds", "seeds status", "point", "95% CI", "vs ref",
            "Holm", "harmed share", "equivalence (90% CI)"]
    rows = []
    for c in res["contrasts"]:
        if c.get("status") == L_T2NA:
            rows.append([c["id"], c["contrast"], c["tier"], L_T2NA, "-", "-", "-", "-", "-", "-", "-", "-", "-"])
            continue
        b = c["by_rule"].get(c["confirmatory_rule"], {})
        h = b.get("holm") or {}
        hcell = ("reject" if h.get("holm_reject") else "not rejected" if h.get("holm_reject") is False else
                 h.get("status", "-"))
        eq = b.get("equivalence") or {}
        eqc = (f"{'EQUIVALENT' if eq.get('equivalent') else 'not equivalent'} {_cell_ci(eq.get('ci90'))}"
               if eq.get("status") == "COMPUTED" else "-")
        nd = 2 if c["kind"] == "retained" else 6
        rows.append([c["id"], c["contrast"], c["tier"], c["label"], c["confirmatory_rule"],
                     ",".join(str(s) for s in c["registered_seeds"]), b.get("seeds_status", "-"),
                     _n(b.get("point"), nd) if "point" in b else b.get("status", "-"), _cell_ci(b.get("ci95"), nd),
                     c["decision"].get("unadjusted_ci95_side", "-"), hcell, _n(b.get("harmed_share"), 4)
                     if "harmed_share" in b else "-", eqc])
    L += RR._table(head, rows) + [""]
    sel = (res.get("selection") or {}).get("beside") or {}
    if sel.get("status") != "NOT_AVAILABLE":
        L += [f"Beside every Q (PREG2 §5.6-§5.8): W* {sel.get('W_STAR')}; CALIB_GAIN {_n(sel.get('CALIB_GAIN'))}; flags "
              f"{sel.get('flags')}" + (f"; D-j: {sel.get('D-j')}" if sel.get("D-j") else ""), ""]
    q1 = res["q1_reading"]
    L += ["## Q1 — the registered reading (PREG2 §9.4)", "", f"**{q1.get('reading')}** (conditions true: "
          f"{q1.get('conditions_true')}; overlap {q1.get('overlap')}); per-seed points {q1.get('per_seed_points')}",
          "", q1["rule"], "", f"D-h: {q1['D-h']}", ""]
    g = res["interpretation_grid"]
    L += ["## Interpretation grid (Q2 x Q1)", "", f"cell {g['cell']}: {g['reading']}", ""]
    L += RR._table(["state", "holds", "definition"], [[k, v, g["definitions"].get(k, "")] for k, v in g["states"].items()])
    d = res["deployment_grade"]
    L += ["", f"DEPLOYMENT-GRADE: **{d['deployment_grade']}** (Q1 SUCCESS {d['Q1_SUCCESS']}, Q5 Holm-positive "
          f"{d['Q5_holm_positive']}, Q7 Holm-positive {d['Q7_holm_positive']}) — {d['rule']}", ""]
    q10 = res["q10_headroom_capture"]
    L += ["## Q10 headroom capture (derived, DESCRIPTIVE)", "",
          f"{q10.get('status')}: point {_n(q10.get('point'), 4)} CI {_cell_ci(q10.get('ci95'), 4)}; per seed "
          f"{[(p['seed'], _n(p['point'], 4)) for p in q10.get('per_seed', [])]}", ""]
    h = res["holm"]
    L += ["## Multiplicity", "", f"Holm step-down family {h['secondary_family']} (size {h['family_size']}, alpha "
          f"{h['alpha']}; rejected {h['n_rejected']}); tertiary DESCRIPTIVE band {h['tertiary_band_family']}; Q1 outside "
          f"Holm; TEST2 family {h['test2_family']}: {L_T2NA}.", ""]
    L += ["## TEST2", "", f"{res['test2']['status']}: {res['test2']['reason']}", ""]
    L += ["## Declared not run (P2_METHODS_LOCK)", ""]
    L += [f"- {x['arm']} seed {x['seed']}: {x['reason']}" for x in res["declared_not_run"]] or ["- none"]
    L += ["", "## Descriptive sections (E1-E5, devices, D1-ext)", ""]
    for k, v in res["descriptive_sections"].items():
        L.append(f"- {k} ({v['what']}): {v['status']} [{v['label']}]")
    L += ["", "## Values (" + (L_PVE if res["split"] == "VALIDATION" else L_DESC) + ")", ""]
    L += RR._table(["arm", "seed", "rule", "MRR@20", "NDCG@10", "HR@10"],
                   [[v["arm"], v["seed"], v["rule"], _n(v["values"]["mrr@20"]), _n(v["values"]["ndcg@10"]),
                     _n(v["values"]["hr@10"])] for v in res["values"]])
    L += ["", "## Labels (PREG2 §6.4)", ""] + RR._table(["claim", "label"], [[x["claim"], x["label"]]
                                                                            for x in res["labels"]["table_6_4"]])
    L += ["", "## Disclosures (PREG2 §11, verbatim)", ""] + [f"- {k}. {v}" for k, v in res["disclosures"].items()]
    return "\n".join(L) + "\n"


def write_outputs(out: Path, res: dict) -> dict:
    shas = {RR.RESULT_NAME: RR._write_new(out / RR.RESULT_NAME, RR.dumps(res).encode("utf-8")),
            RR.REPORT_NAME: RR._write_new(out / RR.REPORT_NAME, render_report(res).encode("utf-8"))}
    man = RR.with_self_sha({"schema": "PPSI_V4_P2_OUTPUTS_1", "result_self_sha256": res[RR.SELF_KEY],
                            "files_sha256": dict(sorted(shas.items()))})
    RR._write_new(out / RR.MANIFEST_NAME, RR.dumps(man).encode("utf-8"))
    return man


# ------------------------------------------------------------------------------------------------ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--block", required=True, type=Path)
    ap.add_argument("--split", required=True, choices=("VALIDATION", "TEST"))
    ap.add_argument("--contrasts", required=True, type=Path, help="company/gates/P2_CONTRASTS_v3.json")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--code-root", type=Path, default=RR.DEFAULT_CODE_ROOT)
    ap.add_argument("--expect-script-sha256", default=None)
    ap.add_argument("--check-only", action="store_true", help="every gate and check; no value printed, nothing written")
    ap.add_argument("--p2-lock", type=Path, default=None, help="company/gates/P2_METHODS_LOCK.json")
    ap.add_argument("--p2-freeze", type=Path, default=None, help="company/gates/P2_SCRIPT_FREEZE.json")
    ap.add_argument("--all-methods-lock", type=Path, default=None, help="company/gates/ALL_METHODS_LOCK.json")
    ap.add_argument("--phase1-result", type=Path, default=None, help="the sitting's Phase-1 RETENTION_RESULT.json")
    ap.add_argument("--test-exposure-register", type=Path, default=None, help="TEST only")
    ap.add_argument("--registry-ext", type=Path, default=None, help="TEST only: REGISTRY_V2_TEST_EXT.json")
    ap.add_argument("--p2-test-result", type=Path, default=None, help="VALIDATION only: p2_out_test_*/RETENTION_RESULT.json")
    return ap.parse_args(argv)


def _open_test(args, block: Path) -> tuple:
    """The Phase-1 TEST gate the manifests were built under (retention_report.check_test_gate), the extension rule and
    the TEST role map; the manifests are then loaded only through test_context_loader (DATA-09)."""
    if args.test_exposure_register is None or args.registry_ext is None:
        raise P2Refused("--split TEST requires --test-exposure-register and --registry-ext (E10.1)")
    gate = RR.check_test_gate(args.all_methods_lock, args.test_exposure_register)
    if not Path(args.registry_ext).is_file():
        raise P2Refused(f"--registry-ext {args.registry_ext} does not exist")
    ext = RR._strict_json(args.registry_ext)
    if not RR.REGISTRY_V2_PATH.is_file():
        raise P2Refused(f"{RR.REGISTRY_V2_PATH} is missing: the extension cannot be verified")
    checks = RR.verify_test_ext_registry(ext, RR._strict_json(RR.REGISTRY_V2_PATH), RR._sha_file(RR.REGISTRY_V2_PATH))
    idx = RR._read_json(block / RR.INDEX_NAME)
    roles = RR.check_ext_roles(idx, ext)
    return gate, ext, {"registry_ext_file_sha256": RR._sha_file(args.registry_ext),
                       "registry_ext_self_sha256": ext.get("self_sha256"), "extension_rule_checks": sorted(checks),
                       "roles": roles}


def _run(args, *, context_loader=None, say=print) -> int:
    sha = script_sha256()
    if args.expect_script_sha256 and args.expect_script_sha256 != sha:
        raise P2Refused(f"script sha256 {sha} is not the frozen {args.expect_script_sha256}")
    lib = assert_frozen_library()
    out = Path(args.out)
    if not args.check_only and out.exists() and any(out.iterdir()):
        raise P2Refused(f"{out} exists and is not empty: outputs are never overwritten (O_EXCL)")
    if args.split == "TEST" and not out.name.startswith(TEST_OUT_PREFIX):
        raise P2Refused(f"--out {out.name}: a Phase-2 TEST output directory is named {TEST_OUT_PREFIX}* (PREG2 §6.3 S3)")
    if args.split == "VALIDATION" and out.name.startswith(TEST_OUT_PREFIX.rstrip("_")):
        raise P2Refused(f"--out {out.name}: {TEST_OUT_PREFIX}* is the TEST output name")
    if args.split == "VALIDATION" and (args.test_exposure_register or args.registry_ext):
        raise P2Refused("--test-exposure-register / --registry-ext are TEST-only options")
    if args.split == "TEST" and args.p2_test_result is not None:
        raise P2Refused("--p2-test-result is a VALIDATION-only option")
    RR.bind(args.code_root)
    contrasts = load_p2_contrasts(args.contrasts)
    lock = check_p2_lock(args.p2_lock)
    freeze = check_p2_freeze(args.p2_freeze)
    if args.all_methods_lock is None or not Path(args.all_methods_lock).is_file():
        raise P2Refused("--all-methods-lock (the Phase-1 ALL_METHODS_LOCK file) is required")
    p1_lock_sha = RR._sha_file(args.all_methods_lock)
    p1 = check_phase1_result(args.phase1_result, p1_lock_sha)
    gates = {"library": lib, "p2_lock": _gate_rec(lock, "sealed", "phase1_reused"), "p2_freeze": _gate_rec(freeze),
             "phase1_result": _gate_rec(p1, "files_sha256"), "all_methods_lock_sha256": p1_lock_sha}
    block = Path(args.block)
    if args.split == "TEST":
        precheck_index(block, "TEST")
        gate, ext, ext_rec = _open_test(args, block)
        gates["phase1_test_gate"] = dict(gate, **ext_rec)
        gates["locked_evaluations"] = check_locked_evaluations(args.test_exposure_register, p1_lock_sha,
                                                               lock["file_sha256"])
        loader = context_loader or RR.test_context_loader(gate, Path(args.registry_ext), ext,
                                                          Path(args.all_methods_lock), Path(args.test_exposure_register))
    else:
        gates["p2_test_result"] = _gate_rec(check_p2_test_result(args.p2_test_result, lock, sha, p1_lock_sha))
        gate = None
        loader = context_loader or RR.production_context_loader()
    blk = load_p2_block(block, args.split, loader, gate)
    ctx = SimpleNamespace(split=args.split, contrasts=contrasts, lock=lock, raw=blk.raw, entries=blk.entries)
    ctx.sealed_check = check_sealed(blk.entries, blk.raw, lock) if args.split == "TEST" else None
    ctx.phase1_binding = (check_phase1_binding(blk.entries, blk.raw, blk.files, lock, p1) if args.split == "TEST"
                          else {"status": "NOT_APPLICABLE", "reason": "S5 binds the sitting's TEST rows"})
    ctx.selection = selection_info(block, blk.idx, lock)
    ctx.wstar_checked = check_wstar_runs(block, blk.entries, ctx.selection)
    ctx.descriptive = descriptive_sections(block, blk.idx)
    if args.check_only:
        with p2_profile(blk.frozen_fl_arms):
            plan = plan_pairs_p2(P2Engine(blk.ctx), RR.Book(blk.entries), ctx)
        say(f"P2 CHECK-ONLY: PASS — split {args.split}, {len(blk.entries)} entries, P2 lock {lock['file_sha256'][:12]}…, "
            f"pairs checked {plan['n_pairs_checked']}, denominators checked {plan['n_denominators_checked']}, "
            f"declared not run {len(lock['declared_not_run'])}; script sha256 {sha}")
        return 0
    with p2_profile(blk.frozen_fl_arms):
        analysis = analyse(blk, ctx)
        res = RR.with_self_sha(build_result(blk, ctx, analysis, script_sha=sha, gates=gates))
    man = write_outputs(out, res)
    say(f"P2: written {out / RR.RESULT_NAME} self_sha256 {res[RR.SELF_KEY]}; {out / RR.REPORT_NAME}; "
        f"{len(man['files_sha256'])} files in {RR.MANIFEST_NAME}; script sha256 {sha}")
    return 0


def _gate_rec(d: Mapping, *drop: str) -> dict:
    """A gate record for the result: no absolute path (outputs are machine-independent), no tuple-keyed tables."""
    return {k: v for k, v in d.items() if k != "path" and k not in drop}


def run(args, *, context_loader=None, say=print) -> int:
    """The CLI body. The library's registry profile (set_registry_profile) is never touched; p2_profile restores its
    own tables on every exit path."""
    return _run(args, context_loader=context_loader, say=say)


def main(argv=None, *, context_loader=None, say=print) -> int:
    args = parse_args(argv)
    try:
        return run(args, context_loader=context_loader, say=say)
    except RR.AnaRefused as e:
        say(f"P2 REFUSED: {e}")
        return REFUSED
    except Exception as e:
        import traceback
        say(f"P2 ERROR: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        return ERROR


if __name__ == "__main__":
    sys.exit(main())
