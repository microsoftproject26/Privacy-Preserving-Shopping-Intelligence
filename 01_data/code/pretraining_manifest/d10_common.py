"""v4data10.d10_common (owner: Data engineer, DATA-10) - ids, the population definition, expected counts and the header
re-labeller of the FA_WARM pretraining manifest PRETRAIN_TRAIN_5_100_NONMAIN.

Population (DERIVED from two registered bands, nothing new invented):
    PRETRAIN_NONMAIN = BANDS["TIEBREAK"] minus BANDS["MAIN"] = pool ranks [131,072, 1,348,682)
over the ascending selection_key_v2 order of the CENSUS-02 5-100 pool (v4fs_data.fs_common.BANDS, code/v4fs_data/
fs_common.py:48). MAIN [0, 131,072) contains CALIB [0, 32,768), so both are excluded; U_NEW_VAL, U_NEW_TEST and FS lie
above the TIEBREAK band, so they are excluded too; the DATA-06 panels (SPARSE: stratum B, HEAVY: stratum D, TRUE_COLD /
COLD_H: strata A0/A1, WARM_TRUNCATION subset of MAIN) are outside the band by construction (checked at build).
Owner decision 2026-09-29 (DECISION_LOG 195): FA_WARM = FL warm-started from a SMALL model pretrained centrally on
these OTHER users of the same pool (Gboard-style public/opted-in pretraining, then FL on the private MAIN users).
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
from v4data08 import d8_common as D8  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

TASK = "DATA-10"
DATA_VERSION = DC.DATA_VERSION                 # CLEAN_V2_POP5100_V2: same data version, one more manifest
OUT_REL = "data_pretrain"                      # server output root (relative to ppsi_v4)
TRAIN_ID = "PRETRAIN_TRAIN_5_100_NONMAIN"
ROLE = "PRETRAIN"
EVAL_ID = DC.UNEW_VAL_EVAL_ID                  # UNEW_VAL_VALIDATION_32768 (disjoint unseen users; DESCRIPTIVE marks only)
DEFINITION = "M01_COHORT_TRAIN"                # the locked M01 TRAIN layout: clients / decisions / central_order
BAND = "PRETRAIN_NONMAIN"
BANDS = FC.BANDS                               # the locked rank bands (code/v4fs_data/fs_common.py:48-50)
PARENT_BAND, MINUS_BAND = "TIEBREAK", "MAIN"
EXCLUDED_BANDS = ("MAIN", "CALIB", "U_NEW_VAL", "U_NEW_TEST", "FS")    # rank bands of the same pool
COVER_BANDS = ("MAIN", "U_NEW_VAL", "U_NEW_TEST", "FS")                # with PRETRAIN: the pool exactly once
PANELS = ("SPARSE", "HEAVY", "TRUE_COLD", "COLD_H", "WARM_TRUNCATION")  # DATA-06 panels (other strata / inside MAIN)
POOL_N, POOL_EXAMPLES, POOL_SHA_V2 = FC.POOL_N, FC.POOL_EXAMPLES, FC.POOL_SHA_V2


def band_ranks(bands: dict = BANDS) -> tuple:
    """[MAIN end, TIEBREAK end): the TIEBREAK band minus its MAIN prefix (MAIN must be a prefix of TIEBREAK)."""
    tx, ty = bands[PARENT_BAND]
    mx, my = bands[MINUS_BAND]
    if not (tx == mx == 0 and my < ty):
        raise BuildBlocked(f"MAIN {bands[MINUS_BAND]} is not a proper prefix of TIEBREAK {bands[PARENT_BAND]}")
    return (int(my), int(ty))


RANKS = band_ranks()                           # (131,072, 1,348,682)
# expected values, each with its measured source (none recomputed from data here)
N_USERS = RANKS[1] - RANKS[0]                  # 1,217,610
MAIN_TRAIN_N_DECISIONS = DC.EXPECTED_MAIN["n_train_examples"]          # 2,587,776 (DATA-06, REGISTRY_V2, measured)
N_DECISIONS = D8.TB_N_DECISIONS - MAIN_TRAIN_N_DECISIONS               # 26,677,600 - 2,587,776 = 24,089,824
N_DECISIONS_SOURCE = ("FULL_TRAIN_5_100_TIEBREAK 26,677,600 (DATA-08, its full build hard-checked it) - "
                      "MAIN_TRAIN_5_100_131072 2,587,776 (DATA-06 / REGISTRY_V2); DERIVED, exact because MAIN is the "
                      "rank prefix of TIEBREAK and every pool user's M01 row_count == n_eligible (locked assertion)")
TB_SHA_V2 = D8.TB_SHA_V2                       # FS-01 §1 band table (measured): TIEBREAK = PRETRAIN u MAIN
MAIN_SHA_V2 = DC.MAIN_SHA_V2                   # CENSUS-02 / DATA-06 (measured)
# PRETRAIN_NONMAIN's own selection-key-v2 hash: MEASURED 2026-09-29 on the frozen local v1 copy by the DATA-10
# band-only pass over all 64 buckets (d10_build --band-only; 30/30 checks incl. the pool / prefix / TIEBREAK / MAIN
# hashes and the exact DATA-06 panels; n_users 1,217,610; sum n_eligible 24,089,824 = N_DECISIONS; receipt
# _pytest_tmp/DATA-10/local_band_only/receipts/DATA10_BAND_RECEIPT.json, receipt_sha256 e814a88b...f256).
PRETRAIN_SHA_V2 = "716beeb961633a9d0918df6bc981cbaafa03bbf9a36b2e2614f4ee4f916989c4"

SELECTION_REFUSAL = (
    "PRETRAIN (FA_WARM, owner decision 2026-09-29, DECISION_LOG 195): the central SMALL pretraining of the FL warm "
    "start on the non-MAIN users of the locked TIEBREAK band (TRAIN-period rows only); its UNEW_VAL_VALIDATION_32768 "
    "marks are DESCRIPTIVE and never select anything (the FL init is the endpoint_6.0 checkpoint, no checkpoint "
    "selection); never architecture-selection, contest, tie-break, calibration or retention evidence. Every read is "
    "v4data.manifests.read_non_test; TEST is never touched.")

BUILD_FILES = ("__init__.py", "d10_common.py", "d10_build.py")


def d10_code_hashes() -> dict:
    files = [HERE / n for n in BUILD_FILES]
    return {"d10_code_sha256": FC.code_tree_sha256(files, HERE),
            "d10_code_sha256_rule": "v4data.common.code_tree_sha256 over code/v4data10/{__init__,d10_common,d10_build}.py",
            "d10_code_files_sha256": {p.name: FC.sha256_file(p) for p in files},
            "v4data08_files_sha256": {n: FC.sha256_file(CODE / "v4data08" / n) for n in ("d8_common.py", "d8_build.py")},
            "v4data06_files_sha256": {n: FC.sha256_file(CODE / "v4data06" / n)
                                      for n in ("d6_common.py", "d6_build.py")},
            "census_v2_core_sha256": FC.sha256_file(CODE / "census_v2" / "core.py"),
            "v4fs_data_fs_common_sha256": FC.sha256_file(CODE / "v4fs_data" / "fs_common.py"),
            "v4data_builder_code_sha256": FC.builder_code_sha256()}


# ------------------------------------------------------------------------------------------------ band selection
def scaled_bands(pool_size: int, bands: dict = BANDS) -> dict:
    """PARTIAL (bucket-subset) smoke builds only: every rank boundary v mapped to v * P // POOL_N, which keeps the
    locked layout (CALIB prefix of MAIN prefix of TIEBREAK; U_NEW_VAL / U_NEW_TEST / FS above it; FS ends at P).
    A full build never uses this (its pool is the full CENSUS-02 pool and the bands are the locked ones)."""
    P = int(pool_size)
    return {k: (int(a) * P // POOL_N, int(b) * P // POOL_N) for k, (a, b) in bands.items()}


def pretrain_band(pool_sorted: np.ndarray, bands: dict = BANDS) -> dict:
    """The PRETRAIN users (index array, pool rank order) and the band checks. `pool_sorted` = pool user indices in
    ascending selection_key_v2 order (rank 0 first). Pure: no dataset access."""
    pool_sorted = np.asarray(pool_sorted)
    P = int(pool_sorted.size)
    x, y = band_ranks(bands)
    if not (0 <= x < y <= P):
        raise BuildBlocked(f"band {BAND} ranks [{x}, {y}) do not fit the pool of {P} users")
    users = pool_sorted[x:y]
    checks = {}
    for name in EXCLUDED_BANDS:
        a, b = bands[name]
        a, b = min(a, P), min(b, P)
        checks[f"disjoint.{BAND}_vs_{name}"] = (int(np.intersect1d(users, pool_sorted[a:b]).size), 0)
        checks[f"rank_ranges_disjoint.{BAND}_vs_{name}"] = (bool(y <= a or b <= x), True)
    cover = np.zeros(P, dtype=np.int64)
    cover[x:y] += 1
    for name in COVER_BANDS:
        a, b = bands[name]
        cover[min(a, P):min(b, P)] += 1
    checks["pool_covered_exactly_once_by_PRETRAIN_MAIN_UNEW_FS"] = (bool(np.all(cover == 1)), True)
    tx, ty = bands[PARENT_BAND]
    mx, my = bands[MINUS_BAND]
    tb = np.sort(pool_sorted[tx:ty])
    un = np.sort(np.concatenate([users, pool_sorted[mx:my]]))
    checks[f"{BAND}_union_MAIN_eq_TIEBREAK"] = (bool(un.size == tb.size and np.array_equal(un, tb)), True)
    checks["CALIB.subset_of_MAIN"] = (bool(np.all(np.isin(pool_sorted[slice(*bands["CALIB"])],
                                                          pool_sorted[slice(*bands["MAIN"])]))), True)
    return {"users": users, "ranks": [int(x), int(y)], "pool_size": P, "checks": checks}


def panel_checks(users: np.ndarray, panels: dict, pool_mask: np.ndarray | None = None) -> dict:
    """Explicit set disjointness of the PRETRAIN users from every DATA-06 panel (index arrays), plus (when given) that
    every PRETRAIN user is in stratum C (the pool) - the panels' strata B / D / A0 / A1 are then excluded wholesale."""
    out = {}
    for name in PANELS:
        if name in panels:
            out[f"disjoint.{BAND}_vs_{name}"] = (int(np.intersect1d(users, np.asarray(panels[name])).size), 0)
    if pool_mask is not None:
        out[f"{BAND}.all_in_pool_stratum_C"] = (bool(np.all(pool_mask[users])), True)
    return out


# ------------------------------------------------------------------------------------------------ header
def d10_header(b, m: dict, tables_meta: list, *, lock: dict, code: dict, population: dict, expected: list,
               users_keys_v2: np.ndarray, extra: dict | None = None) -> dict:
    """v4data06.d6_common.d6_header (a mirror of v4fs_data.fs_build.fs_header) re-labelled to DATA-10: status /
    run_label = the registry role PRETRAIN (PARTIAL_TEST_BUILD for a partial build), task DATA-10, the PRETRAIN selection
    refusal; content hash recomputed by the builder rule."""
    h = DC.d6_header(b, m, tables_meta, out_id=TRAIN_ID, role=ROLE, parent=None, lock=lock, code=code,
                     population=population, expected=expected, users_keys_v2=users_keys_v2, extra=extra)
    h.pop("header_content_sha256")
    h["task"] = TASK
    h["selection_refusal"] = SELECTION_REFUSAL
    h["definition"]["note"] = (f"tables, row filter, order and assertions of {DEFINITION} (built by the locked "
                               "v4data.manifests._train_manifest, unedited) over the band TIEBREAK minus MAIN "
                               "(pool ranks [131,072, 1,348,682)) instead of the original M01 cohort; client_ord is the "
                               "band's own dense digest order; no calibration subset (in_calib_prefix all False)")
    h = FC.jsonable(h)
    h["header_content_sha256"] = FC.header_content_sha256(h)
    return h


chk = DC.chk

__all__ = [n for n in dir() if not n.startswith("_")]
