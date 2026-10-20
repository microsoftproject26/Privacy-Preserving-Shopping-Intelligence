"""v4data08.d8_common (owner: Data engineer, DATA-08) - ids, the REUSED TIEBREAK population definition, expected counts
and the header re-labeller of the C_FULL training manifest.

Population (REUSED, not invented): the TIEBREAK rank band already defined in code as
    v4fs_data.fs_common.BANDS["TIEBREAK"] = (0, 1_348_682)            (code/v4fs_data/fs_common.py:48)
i.e. ranks r < 1,348,682 of the ascending selection_key_v2 order of the CENSUS-02 5-100 pool - the LOCKED D1 ruling
(DECISION_LOG 41/42, ADR-005 "TB r < 1,348,682", POPULATION_5_100_FABLE_POSITION_FINAL.md). It is the pool minus the
three top bands U_NEW_VAL [1,348,682, 1,381,450), U_NEW_TEST [1,381,450, 1,414,218) and FS [1,414,218, 1,446,986).
The packet / amendment 06 §5 figure 1,315,914 is the SUPERSEDED first Fable ruling (DECISION_LOG 40: U_NEW bands of
65,536 each; 1,446,986 - 131,072); it is recorded below as a discrepancy, never used as a check.
Nothing here reads the dataset.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent

from v4data.manifests import BuildBlocked  # noqa: E402
from v4data06 import d6_common as DC  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

TASK = "DATA-08"
DATA_VERSION = DC.DATA_VERSION                 # CLEAN_V2_POP5100_V2: same data version, one more manifest
OUT_REL = "data_cfull"                         # server output root (relative to ppsi_v4)
TRAIN_ID = "FULL_TRAIN_5_100_TIEBREAK"
ROLE = "C_FULL"
EVAL_ID = DC.MAIN_EVAL_ID                      # MAIN_VALIDATION_5_100_131072, unchanged (amendment 06 §5)
DEFINITION = "M01_COHORT_TRAIN"                # the locked M01 TRAIN layout: clients / decisions / central_order
BAND = "TIEBREAK"
BANDS = FC.BANDS                               # the locked rank bands (code/v4fs_data/fs_common.py:48-50)
EXCLUDED_BANDS = ("U_NEW_VAL", "U_NEW_TEST", "FS")
POOL_N, POOL_EXAMPLES, POOL_SHA_V2 = FC.POOL_N, FC.POOL_EXAMPLES, FC.POOL_SHA_V2

# expected values, each with its measured source (none recomputed here)
TB_N_USERS = BANDS[BAND][1] - BANDS[BAND][0]   # 1,348,682
TB_SHA_V2 = "df5d0c7e82bc0842f5c8435fafaa67f947d4fe951d5f9e6aed9ca58d10a46cdf"   # FS-01 handoff §1 band table
EXCLUDED_BAND_DECISIONS = {"U_NEW_VAL": (650_203, "DATA-06 UNEW_VAL_SUPPORT_32768 n_decisions (REGISTRY_V2)"),
                           "U_NEW_TEST": (649_980, "DATA-06 UNEW_HOLDOUT_SUPPORT_32768 n_decisions (REGISTRY_V2)"),
                           "FS": (645_603, "FS-01 FS_TRAIN_5_100_32768 n_decisions (REGISTRY_FS)")}
# every pool user's M01 row_count == n_eligible (locked assertion), so a band's decisions = its users' eligible counts
TB_N_DECISIONS = POOL_EXAMPLES - sum(v for v, _ in EXCLUDED_BAND_DECISIONS.values())    # 26,677,600
TB_N_DECISIONS_SOURCE = ("pool 28,623,386 (CENSUS-02) - UNEW_VAL 650,203 - UNEW_HOLDOUT 649,980 - FS 645,603 "
                         "(measured manifests); DERIVED, exact iff every band is a rank slice of the same pool")
AMENDMENT_06_STATED_USERS = 1_315_914          # superseded DL40 figure (U_NEW 65,536 each); NOT a check
DISCREPANCY_NOTE = (
    "PREG1_AMENDMENT_06 §5 and the DATA-08 packet state 1,315,914 users. That is the SUPERSEDED first Fable "
    "POPULATION_5_100 ruling (DECISION_LOG 40: U_NEW_VAL / U_NEW_TEST 65,536 each, TIEBREAK = 1,446,986 - 131,072). "
    "The LOCKED D1 (DECISION_LOG 41/42, ADR-005, v4fs_data.fs_common.BANDS['TIEBREAK'] = (0, 1,348,682)) has U_NEW "
    "bands of 32,768 each and TB = r < 1,348,682, which also excludes the FS band (the top 32,768). The packet says "
    "'if a TIEBREAK population already exists, REUSE it exactly', so this manifest is the locked TB band: 1,348,682 "
    "users. (A literal 'pool minus the two U_NEW bands' would be 1,381,450 and would include the FS screen band.)")

SELECTION_REFUSAL = (
    "C_FULL (PREG1_AMENDMENT_06 §5 / 06B §6.3): a secondary 'cloud' reference trained on the locked TIEBREAK band "
    "(TRAIN-period rows only); never architecture-selection, contest, tie-break or calibration evidence; evaluated on "
    "the unchanged MAIN_VALIDATION_5_100_131072 rows. Every read is v4data.manifests.read_non_test; TEST is never "
    "touched.")

BUILD_FILES = ("__init__.py", "d8_common.py", "d8_build.py")


def d8_code_hashes() -> dict:
    files = [HERE / n for n in BUILD_FILES]
    return {"d8_code_sha256": FC.code_tree_sha256(files, HERE),
            "d8_code_sha256_rule": "v4data.common.code_tree_sha256 over code/v4data08/{__init__,d8_common,d8_build}.py",
            "d8_code_files_sha256": {p.name: FC.sha256_file(p) for p in files},
            "v4data06_files_sha256": {n: FC.sha256_file(CODE / "v4data06" / n)
                                      for n in ("d6_common.py", "d6_build.py")},
            "census_v2_core_sha256": FC.sha256_file(CODE / "census_v2" / "core.py"),
            "v4fs_data_fs_common_sha256": FC.sha256_file(CODE / "v4fs_data" / "fs_common.py"),
            "v4data_builder_code_sha256": FC.builder_code_sha256()}


# ------------------------------------------------------------------------------------------------ band selection
def tb_band(pool_sorted: np.ndarray, bands: dict = BANDS, band: str = BAND,
            excluded: tuple = EXCLUDED_BANDS) -> dict:
    """The TB users (index array, pool rank order) and the band checks. `pool_sorted` = pool user indices in ascending
    selection_key_v2 order (rank 0 first). Pure: no dataset access."""
    P = int(np.asarray(pool_sorted).size)
    x, y = bands[band]
    if not (0 <= x < y <= P):
        raise BuildBlocked(f"band {band} ranks [{x}, {y}) do not fit the pool of {P} users")
    tb = np.asarray(pool_sorted[x:y])
    checks = {}
    cover = np.zeros(P, dtype=np.int64)
    cover[x:y] += 1
    for name in excluded:
        a, b = bands[name]
        a, b = min(a, P), min(b, P)
        cover[a:b] += 1
        ov = int(np.intersect1d(tb, pool_sorted[a:b]).size)
        checks[f"disjoint.{band}_vs_{name}"] = (ov, 0)
        checks[f"rank_ranges_disjoint.{band}_vs_{name}"] = (bool(y <= a or b <= x), True)
    checks["pool_covered_exactly_once_by_TB_and_excluded_bands"] = (bool(np.all(cover == 1)), True)
    for name in ("MAIN", "CALIB"):
        if name in bands and bands[name][1] <= P:
            a, b = bands[name]
            checks[f"{name}.subset_of_{band}"] = (bool(np.all(np.isin(pool_sorted[a:b], tb))), True)
    return {"tb": tb, "ranks": [int(x), int(y)], "pool_size": P, "checks": checks}


# ------------------------------------------------------------------------------------------------ header
def d8_header(b, m: dict, tables_meta: list, *, lock: dict, code: dict, population: dict, expected: list,
              users_keys_v2: np.ndarray, extra: dict | None = None) -> dict:
    """v4data06.d6_common.d6_header (itself a mirror of v4fs_data.fs_build.fs_header) re-labelled to DATA-08: status /
    run_label = the registry role C_FULL, task DATA-08, the C_FULL selection refusal; content hash recomputed by the
    builder rule."""
    h = DC.d6_header(b, m, tables_meta, out_id=TRAIN_ID, role=ROLE, parent=None, lock=lock, code=code,
                     population=population, expected=expected, users_keys_v2=users_keys_v2, extra=extra)
    h.pop("header_content_sha256")
    h["task"] = TASK
    h["selection_refusal"] = SELECTION_REFUSAL
    h["definition"]["note"] = (f"tables, row filter, order and assertions of {DEFINITION} (built by the locked "
                               "v4data.manifests._train_manifest, unedited) over the locked TIEBREAK band "
                               "(v4fs_data.fs_common.BANDS['TIEBREAK']) instead of the original M01 cohort; client_ord "
                               "is the band's own dense digest order; no calibration subset (in_calib_prefix all False)")
    h = FC.jsonable(h)
    h["header_content_sha256"] = FC.header_content_sha256(h)
    return h


chk = DC.chk

__all__ = [n for n in dir() if not n.startswith("_")]
