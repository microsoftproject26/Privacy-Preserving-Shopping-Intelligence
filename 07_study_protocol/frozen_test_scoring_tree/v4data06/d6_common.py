"""v4data06.d6_common (owner: Data engineer, DATA-06) - band ids, expected counts and the header/emitter helpers for
the v2 population build (ADR-005 R2+R4 binding synthesis: company/reviews/POPULATION_5_100_SCIENTIFIC_SYNTHESIS.md,
company/decisions/ADR-005-population-5-100-and-fast-feature-search.md addendum 2).

Nothing here reads the dataset. Every generic utility (hashing, encode/write, header content hash, self-hash) is
IMPORTED UNCHANGED from v4fs_data.fs_common (FC), which itself only calls the LOCKED v4data.manifests /
v4data.common. This module adds only: the DATA-06 band/manifest ids, the expected-count table from the task packet,
and a header re-labeller (d6_header) that mirrors v4fs_data.fs_build.fs_header exactly, one level more generic (it
takes the manifest's ROLE name instead of a single fixed FS status).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent

from census_v2 import core as C2  # noqa: E402
from v4data.manifests import BuildBlocked  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

TASK = "DATA-06"
DATA_VERSION = "CLEAN_V2_POP5100_V2"

# the CENSUS-02 5-100 pool: identical to FS-01's (same salt v2, same stratum-C definition) - reused unchanged, not
# recomputed as a new constant, so a byte-for-byte pool mismatch is impossible to miss.
POOL_N, POOL_EXAMPLES, POOL_SHA_V2 = FC.POOL_N, FC.POOL_EXAMPLES, FC.POOL_SHA_V2
PREFIX_SHA_V2 = FC.PREFIX_SHA_V2
BANDS = FC.BANDS                      # MAIN/CALIB/TIEBREAK/U_NEW_VAL/U_NEW_TEST/PACKET_REGION_0_294912/FS
MAIN_SHA_V2 = "563e76637a384c2985fff5e272d4290f4f3e52c99f57eebec4c20cde0628b8b3"
CALIB_SHA_V2 = "9be38de146f5069a88d57c867dfa021cf1c67f2d67ae1e58df63f7c6ac9c1f02"

N_MAIN, N_CALIB, N_PANEL = 131_072, 32_768, 16_384

MAIN_TRAIN_ID, MAIN_EVAL_ID = "MAIN_TRAIN_5_100_131072", "MAIN_VALIDATION_5_100_131072"
CALIB_TRAIN_ID, CALIB_EVAL_ID = "CALIB_TRAIN_5_100_32768", "CALIB_VALIDATION_5_100_32768"
UNEW_VAL_TRAIN_ID, UNEW_VAL_EVAL_ID = "UNEW_VAL_SUPPORT_32768", "UNEW_VAL_VALIDATION_32768"
UNEW_TEST_TRAIN_ID, UNEW_TEST_EVAL_ID = "UNEW_HOLDOUT_SUPPORT_32768", "UNEW_HOLDOUT_VALIDATION_32768"
SPARSE_TRAIN_ID, SPARSE_EVAL_ID = "SPARSE_PANEL_16384_SUPPORT", "SPARSE_PANEL_16384_VALIDATION"
HEAVY_TRAIN_ID, HEAVY_EVAL_ID = "HEAVY_PANEL_16384_SUPPORT", "HEAVY_PANEL_16384_VALIDATION"
TRUE_COLD_ID = "TRUE_COLD_VALIDATION"
COLD_H_ID = "COLD_H_16384"
WARM_TRUNCATION_ID = "WARM_TRUNCATION_16384"

ALL_IDS = (MAIN_TRAIN_ID, MAIN_EVAL_ID, CALIB_TRAIN_ID, CALIB_EVAL_ID, UNEW_VAL_TRAIN_ID, UNEW_VAL_EVAL_ID,
           UNEW_TEST_TRAIN_ID, UNEW_TEST_EVAL_ID, SPARSE_TRAIN_ID, SPARSE_EVAL_ID, HEAVY_TRAIN_ID, HEAVY_EVAL_ID,
           TRUE_COLD_ID, COLD_H_ID, WARM_TRUNCATION_ID)

ROLES = {
    "MAIN": {"train": MAIN_TRAIN_ID, "eval": MAIN_EVAL_ID},
    "CALIBRATION": {"train": CALIB_TRAIN_ID, "eval": CALIB_EVAL_ID},
    "UNSEEN_VAL": {"train": UNEW_VAL_TRAIN_ID, "eval": UNEW_VAL_EVAL_ID},
    "UNSEEN_HOLDOUT": {"train": UNEW_TEST_TRAIN_ID, "eval": UNEW_TEST_EVAL_ID},
    "SPARSE_PANEL": {"train": SPARSE_TRAIN_ID, "eval": SPARSE_EVAL_ID},
    "HEAVY_PANEL": {"train": HEAVY_TRAIN_ID, "eval": HEAVY_EVAL_ID},
    "TRUE_COLD": {"eval": TRUE_COLD_ID},
    "COLD_H": {"panel": COLD_H_ID},
    "WARM_TRUNCATION": {"panel": WARM_TRUNCATION_ID},
}

# the manifest naming decision this build owns (task DATA-06.md gives one id per row for SPARSE_PANEL_16384 /
# HEAVY_PANEL_16384 / TRUE_COLD_VALIDATION / COLD_H_16384 / WARM_TRUNCATION_16384): SPARSE and HEAVY are TRAIN+
# VALIDATION pairs (support = M01, VALIDATION = M02), so they get the same _SUPPORT/_VALIDATION suffixing as
# UNEW_VAL/UNEW_TEST; TRUE_COLD_VALIDATION is eval-only (M03 shape); COLD_H_16384 and WARM_TRUNCATION_16384 are
# single panel manifests (M04/M04W shape: panel_users/panel_queries/panel_support tables), documented in the handoff.
NAMING_DECISION = (
    "SPARSE_PANEL_16384 and HEAVY_PANEL_16384 (task table: 'support = their own TRAIN examples; VALIDATION = the M02 "
    "definition') are built and registered as two physical manifests each, suffixed _SUPPORT / _VALIDATION exactly "
    "like UNEW_VAL / UNEW_TEST; TRUE_COLD_VALIDATION is eval-only (M03 shape, no TRAIN counterpart, cold users have "
    "zero eligible TRAIN examples); COLD_H_16384 and WARM_TRUNCATION_16384 are single panel manifests (M04 / M04W "
    "shape). DEVIATION: the task packet names this band's physical manifests UNEW_TEST_SUPPORT_32768 / "
    "UNEW_TEST_VALIDATION_32768, but v4data.manifests.refuse_test_manifest (LOCKED, not edited) refuses any manifest "
    "id containing the substring 'TEST' unconditionally, as a blunt defense-in-depth gate against ever building a "
    "TEST-split manifest before ALL_METHODS_LOCK - it would refuse these files outright. The physical manifest ids "
    "are therefore UNEW_HOLDOUT_SUPPORT_32768 / UNEW_HOLDOUT_VALIDATION_32768. The registry role key was first kept "
    "as the task's own 'UNSEEN_TEST', but the Operations Director corrected this (v4train's registry loader refuses "
    "any manifest id OR ROLE containing 'TEST' too): the role is 'UNSEEN_HOLDOUT' everywhere on disk (header "
    "status/run_label and the registry roles key); no string 'TEST' appears anywhere in DATA-06's output."
)

# KNOWN, EXPECTED assertion deviation: the locked _train_manifest's 'calib_prefix_is_first_65536' assertion checks
# that in_calib_prefix=True clients occupy the first rows of the client_ord order (ascending client digest /
# selection_key_v1). That coincidence held in the ORIGINAL DATA-04 design because BOTH the cohort/calib selection
# and the client_ord storage order used the same v1 key. DATA-06 selects CALIB by selection_key_v2 (nested in MAIN,
# also chosen by v2 key) while client_ord is still assigned by the locked, unedited digest/v1-key order (unchanged,
# not selection-related) - so CALIB's 32,768 members are scattered through MAIN's v1-ordered client table, not a
# contiguous prefix. This is expected and harmless: CALIB_TRAIN_5_100_32768 is a complete, independent manifest;
# no consumer needs to slice MAIN's client_ord to reach it. Affects MAIN_TRAIN_5_100_131072 only (every other TRAIN
# manifest's in_calib_prefix column is either all-False or, for CALIB_TRAIN itself, all-True - both trivially
# satisfy the assertion regardless of order).
KNOWN_ASSERTION_DEVIATIONS = {MAIN_TRAIN_ID: {"calib_prefix_is_first_65536"}}

SELECTION_REFUSAL = (
    "DATA_VERSION CLEAN_V2_POP5100_V2 (ADR-005 R2+R4 binding synthesis, DECISION_LOG 42/43): MAIN/CALIB/UNSEEN_VAL/"
    "UNSEEN_TEST/panel manifests over selection_key_v2 rank bands and CENSUS-02 strata; every read is "
    "v4data.manifests.read_non_test; TEST is never touched and no RETRO_TEST row is built (refused until "
    "ALL_METHODS_LOCK)."
)

# expected counts (company/tasks/DATA-06.md "Checks"; CENSUS-02 measurements quoted verbatim, not recomputed here)
EXPECTED_MAIN = {"n_users": 131_072, "n_train_examples": 2_587_776, "n_validation_users": 40_764,
                 "n_eval_rows": 519_714, "n_rankable_rows": 515_636}
EXPECTED_CALIB = {"n_users": 32_768, "n_train_examples": 646_067, "n_validation_users": 10_070, "n_eval_rows": 127_530}
EXPECTED_TRUE_COLD = {"n_users": 381_295, "n_eval_rows": 2_857_665, "n_users_A0": 299_035, "n_eval_rows_A0": 2_270_516,
                      "n_users_A1": 82_260, "n_eval_rows_A1": 587_149}
LOCKED_MAIN_POP_SHA256 = MAIN_SHA_V2          # DATA-06.md "Population hashes equal CENSUS-02"
LOCKED_CALIB_POP_SHA256 = CALIB_SHA_V2
DATA04_BUILDER_CODE_SHA256 = FC.DATA04_BUILDER_CODE_SHA256
TRANSFORM_STATE_SELF_SHA256 = FC.TRANSFORM_STATE_SELF_SHA256   # Clean V2, unchanged; DATA-06 fits no new statistics


# ------------------------------------------------------------------------------------------------ code identity
BUILD_FILES = ("__init__.py", "d6_common.py", "d6_build.py")


def d6_code_hashes() -> dict:
    files = [HERE / n for n in BUILD_FILES]
    return {"d6_code_sha256": FC.code_tree_sha256(files, HERE),
            "d6_code_sha256_rule": "v4data.common.code_tree_sha256 over code/v4data06/{__init__,d6_common,d6_build}.py",
            "d6_code_files_sha256": {p.name: FC.sha256_file(p) for p in files},
            "census_v2_core_sha256": FC.sha256_file(CODE / "census_v2" / "core.py"),
            "v4fs_data_fs_common_sha256": FC.sha256_file(CODE / "v4fs_data" / "fs_common.py"),
            "v4data_builder_code_sha256": FC.builder_code_sha256()}


# ------------------------------------------------------------------------------------------------ header
def d6_header(b, m: dict, tables_meta: list, *, out_id: str, role: str, parent, lock: dict, code: dict,
              population: dict, expected: list, users_keys_v2: np.ndarray, extra: dict | None = None) -> dict:
    """The builder's own header for the LOCKED definition manifest (m['manifest_id'] must already be one of the
    MANIFEST_SPEC pre-lock ids so b._header's spec lookup succeeds), re-labelled to the DATA-06 band id; the
    header_content_sha256 is recomputed with the builder's rule. Mirrors v4fs_data.fs_build.fs_header exactly."""
    h = b._header(dict(m, expected=[]), tables_meta)
    definition = h["manifest_id"]
    h.pop("header_content_sha256")
    h["manifest_id"] = out_id
    h["status"] = role if b.full else "PARTIAL_TEST_BUILD"
    h["run_label"] = role
    h["data_version"] = DATA_VERSION
    h["task"] = TASK
    h["definition"] = {"manifest_spec_definition": definition, "spec_id": h["spec_id"],
                       "spec_version": h["spec_version"],
                       "note": f"tables, row filter, order and assertions of {definition} (built by the locked "
                               "v4data.manifests functions, unedited) over the DATA-06 v2 population band instead "
                               "of the original M0x cohort; target_class_cohort_train_positives and client_ord / "
                               "eval_user_key are computed within this band's own population unless noted otherwise"}
    h["parent_manifests"] = [parent] if parent else []
    h["data_protocol_lock_sha256"] = lock["lock_self_sha256"]
    h["data_protocol_lock_file_sha256"] = lock["file_sha256"]
    h["population_lock_sha256"] = ("NOT_APPLICABLE: ADR-005 v2 population (R2+R4 binding synthesis), not a "
                                   "POPULATION_PLAN cohort")
    h["population_sha256_rule"] = ("MANIFEST_SPEC_V4 hashing.population_sha256 over the C1.2 selection_key of this "
                                   "manifest's users, ascending")
    h["population_sha256_selection_key_v2"] = FC.population_sha256(np.sort(np.asarray(users_keys_v2, dtype=np.uint64)))
    h["population_sha256_selection_key_v2_rule"] = ("CENSUS-02 rule: v4data.common.population_sha256 over each "
                                                    "user's 8-byte big-endian selection_key_v2, ascending")
    h["population_v2"] = population
    h["code"] = code
    h["expected_counts_check"] = [{"field": f, "expected": FC.jsonable(e), "expected_source": s,
                                   "observed": FC.jsonable(o), "match": bool(o == e)} for f, o, e, s in expected]
    h["selection_refusal"] = SELECTION_REFUSAL
    if extra:
        h.update(extra)
    h = FC.jsonable(h)
    h["header_content_sha256"] = FC.header_content_sha256(h)
    return h


def chk(name, observed, expected, source, hard=True) -> dict:
    return {"name": name, "observed": FC.jsonable(observed), "expected": FC.jsonable(expected), "source": source,
            "pass": bool(observed == expected), "hard": bool(hard)}


__all__ = [n for n in dir() if not n.startswith("_")]
