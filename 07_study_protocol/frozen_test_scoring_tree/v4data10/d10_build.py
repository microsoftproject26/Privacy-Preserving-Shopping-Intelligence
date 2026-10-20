"""v4data10.d10_build (owner: Data engineer, DATA-10) - the FA_WARM pretraining manifest PRETRAIN_TRAIN_5_100_NONMAIN.

  users      the locked TIEBREAK band minus the MAIN band = pool ranks [131,072, 1,348,682) of the ascending
             selection_key_v2 order of the CENSUS-02 5-100 pool (v4data10.d10_common; 1,217,610 users);
  rows       the locked M01 definition: TRAIN-split, loss-eligible decisions whose target lies before TRAIN_END
             (v4data.manifests.ManifestBuilder.pass_extract + _train_manifest, unedited);
  layout     clients / decisions / central_order - exactly MAIN_TRAIN_5_100_131072's tables and column types;
  eval       none built: role PRETRAIN evaluates on UNEW_VAL_VALIDATION_32768 unchanged (d10_finalize / REGISTRY ext).
Mechanics = DATA-08 (v4data08.d8_build.D8Builder = v4data06.D6Builder, a subclass of the locked ManifestBuilder, reused
unchanged): one pass_extract() with self.cohort = the PRETRAIN band alone; the VALIDATION rows the locked pass collects
for COHORT users are released right after the pass; the M01 row arrays are released once the decisions table exists.
Disjointness from MAIN / CALIB / U_NEW_VAL / U_NEW_TEST / FS (rank bands) and from the DATA-06 panels (explicit sets in a
full build via v4data06.d6_build.compute_bands, unedited) is checked before the extract.
`--band-only`: prepare + band + panel checks + the band's population facts (no extract, no manifest) -> a small
DATA10_BAND_RECEIPT.json (the local measurement of n_users / n_decisions / hashes; see the DATA-10 handoff).
TEST: every read is v4data.manifests.read_non_test; the manifest id is checked by refuse_test_manifest first.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import ppsi_env  # noqa: E402  (import isolation first, T43)

import numpy as np  # noqa: E402

from census_v2 import core as C2  # noqa: E402
from v4data.common import builder_code_sha256, population_sha256, utc_now  # noqa: E402
from v4data.manifests import BuildBlocked, refuse_test_manifest  # noqa: E402
from v4data08.d8_build import D8Builder, pool_order  # noqa: E402
from v4data10 import d10_common as D10  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

V4 = D10.V4


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class D10Builder(D8Builder):
    """D8Builder (= D6Builder: the locked ManifestBuilder + an injected population per pass), unchanged; DATA-10 adds
    only the driver below."""


def panels_of(b, po: dict, bands: dict) -> dict:
    """The DATA-06 panels. Full build: the exact DATA-06 sets (v4data06.d6_build.compute_bands, unedited - it raises
    on a population smaller than a panel). Partial / synthetic build: the panels' whole source strata (a superset of
    each panel: SPARSE <- stratum B, HEAVY <- D, TRUE_COLD / COLD_H <- A0 u A1, WARM_TRUNCATION <- MAIN)."""
    if b.full:
        from v4data06.d6_build import compute_bands
        cb = compute_bands(b)
        if not np.array_equal(cb["pool_sorted"], po["pool_sorted"]):
            raise BuildBlocked("v4data06 compute_bands pool order differs from v4data08 pool_order")
        return {k: cb[k] for k in D10.PANELS}
    strat = C2.stratum_codes(b.u_nelig, b.u_ntrain)
    cold = np.flatnonzero((strat == C2.S_A0) | (strat == C2.S_A1))
    return {"SPARSE": np.flatnonzero(strat == C2.S_B), "HEAVY": np.flatnonzero(strat == C2.S_D),
            "TRUE_COLD": cold, "COLD_H": cold, "WARM_TRUNCATION": po["pool_sorted"][slice(*bands["MAIN"])]}


def band_checks(b, po: dict, sel: dict, panels: dict, bands: dict) -> list:
    """Structural band / panel checks always; the real-population constants only in a full build (hard there)."""
    ks = po["pool_keys_sorted"]
    kv2 = po["keys_v2"]
    users = sel["users"]
    out = []
    if b.full:
        out += [D10.chk("pool.n_users", int(ks.size), D10.POOL_N, "CENSUS-02 / FS-01 / DATA-06"),
                D10.chk("pool.population_sha256_selection_key_v2", population_sha256(ks), D10.POOL_SHA_V2,
                        "CENSUS-02 / FS-01 / DATA-06")]
        for k, ref in FC.PREFIX_SHA_V2.items():
            out.append(D10.chk(f"pool.prefix{k}.population_sha256_selection_key_v2", population_sha256(ks[:k]), ref,
                               "CENSUS-02 candidate-N table"))
        tx, ty = bands[D10.PARENT_BAND]
        out += [D10.chk("TIEBREAK.population_sha256_selection_key_v2", population_sha256(np.sort(ks[tx:ty])),
                        D10.TB_SHA_V2, "FS-01 handoff §1 band table (measured)"),
                D10.chk("MAIN.population_sha256_selection_key_v2", population_sha256(np.sort(ks[slice(*bands["MAIN"])])),
                        D10.MAIN_SHA_V2, "CENSUS-02 / DATA-06"),
                D10.chk("PRETRAIN.n_users", int(users.size), D10.N_USERS, "BANDS['TIEBREAK'] - BANDS['MAIN']"),
                D10.chk("PRETRAIN.n_eligible_examples", int(b.u_nelig[users].sum()), D10.N_DECISIONS,
                        D10.N_DECISIONS_SOURCE)]
        if D10.PRETRAIN_SHA_V2 is not None:
            out.append(D10.chk("PRETRAIN.population_sha256_selection_key_v2", population_sha256(np.sort(kv2[users])),
                               D10.PRETRAIN_SHA_V2, "DATA-10 local band-only measurement (v1 copy, 64 buckets)"))
    for name, (obs, exp) in sel["checks"].items():
        out.append(D10.chk(name, obs, exp, "rank-band construction over the shared pool"))
    strat = C2.stratum_codes(b.u_nelig, b.u_ntrain)
    for name, (obs, exp) in D10.panel_checks(users, panels, strat == C2.S_C).items():
        out.append(D10.chk(name, obs, exp, "DATA-06 panels: other strata (B / D / A0-A1) or inside MAIN"))
    out.append(D10.chk("PRETRAIN.every_user_has_ge1_eligible_example", bool(np.all(b.u_nelig[users] >= 1)), True,
                       "pool definition (5 <= n_eligible <= 100)"))
    return out


def population_block(sel: dict, keys_sorted: np.ndarray, bands: dict) -> dict:
    return {"band": D10.BAND, "rank_range": sel["ranks"], "pool_size": sel["pool_size"],
            "n_users": int(sel["users"].size), "population_sha256_selection_key_v2": population_sha256(keys_sorted),
            "definition": "BANDS['TIEBREAK'] minus BANDS['MAIN'] (v4fs_data.fs_common.BANDS, code/v4fs_data/"
                          "fs_common.py:48); owner decision 2026-09-29 (DECISION_LOG 195)",
            "excluded_bands": {n: list(bands[n]) for n in D10.EXCLUDED_BANDS},
            "excluded_panels": list(D10.PANELS)}


def select(b, bands: dict) -> tuple:
    po = pool_order(b)
    if not b.full and bands is D10.BANDS:                   # partial smoke on real data: proportionally scaled bands
        bands = D10.scaled_bands(po["pool_sorted"].size)
    sel = D10.pretrain_band(po["pool_sorted"], bands)
    panels = panels_of(b, po, bands)
    checks = band_checks(b, po, sel, panels, bands)
    return po, sel, checks, bands


def run_band_only(b: D10Builder, out_root: Path, *, bands: dict | None = None, write: bool = True) -> dict:
    """prepare + band + panels + checks, no extract: the band's facts (population hash, n_users, sum n_eligible)."""
    bands = D10.BANDS if bands is None else bands
    refuse_test_manifest(D10.TRAIN_ID)
    t0 = time.time()
    b.prepare()
    po, sel, checks, bands = select(b, bands)
    users = np.sort(np.asarray(sel["users"], dtype=np.int64))
    keys = np.sort(po["keys_v2"][users])
    rec = {"receipt": "DATA10_BAND_RECEIPT", "task": D10.TASK, "full_build": b.full, "buckets": len(b.buckets),
           "python_utc_end": utc_now(), "population": population_block(sel, keys, bands),
           "n_eligible_examples": int(b.u_nelig[users].sum()), "expected": {"n_users": D10.N_USERS,
                                                                            "n_decisions": D10.N_DECISIONS},
           "checks": checks, "n_checks": len(checks), "n_failed": len([c for c in checks if not c["pass"]]),
           "timings_s": {"total": round(time.time() - t0, 1), **b.timings}, "test_split_read": False,
           "no_raw_ids_written": True, "code": D10.d10_code_hashes()}
    if write:
        FC.write_json_self_hashed(out_root, "receipts/DATA10_BAND_RECEIPT.json", rec)
    return rec


def run_build(b: D10Builder, out_root: Path, *, bands: dict | None = None, write: bool = True) -> dict:
    bands = D10.BANDS if bands is None else bands
    refuse_test_manifest(D10.TRAIN_ID)
    t0 = time.time()
    lock = FC.lock_info()
    code = D10.d10_code_hashes()
    checks = [D10.chk("code.v4data_builder_code_sha256_is_DATA04", b.code_sha, FC.DATA04_BUILDER_CODE_SHA256,
                      "DATA-04 handoff", hard=b.full)]
    b.prepare()
    t_prep = time.time()
    if b.full:
        checks += b.universe_checks()
    po, sel, bchecks, bands = select(b, bands)
    checks += bchecks
    hard = [c for c in checks if c["hard"] and not c["pass"]]
    if hard:
        raise BuildBlocked(f"DATA-10 band checks failed: {[c['name'] for c in hard]}")
    kv2 = po["keys_v2"]

    tp0 = time.time()
    b.set_one(cohort=sel["users"])                          # COHORT bit = PRETRAIN only; no CALIB / cold / panel bits
    b.extract()
    t_extract = time.time() - tp0
    if b.full:
        checks += b.recon_checks()
    n_val_rows_released = int(b.rows_val["g"].size)
    b.rows_val, b.rows_ev = None, None                      # no eval manifest is built (memory)
    gc.collect()

    users = np.sort(np.asarray(sel["users"], dtype=np.int64))
    m = b.rows_m01
    in_band = np.zeros(b.u_dig.size, dtype=bool)
    in_band[users] = True
    checks.append(D10.chk("M01_rows.all_in_PRETRAIN", bool(np.all(in_band[m["g"]])), True,
                          "pass_extract COHORT bit = PRETRAIN"))
    main_idx = po["pool_sorted"][slice(*bands["MAIN"])]
    checks.append(D10.chk("M01_rows.none_from_MAIN", bool(not np.any(np.isin(m["g"], main_idx))), True,
                          "band excludes MAIN (and CALIB)"))
    ord_of = np.full(b.u_dig.size, -1, dtype=np.int64)
    ord_of[users] = b._dense_rank_by_digest(users)
    tm = b._train_manifest(D10.DEFINITION, m, users, ord_of, None)
    b.rows_m01 = None                                       # the decisions table holds its own reordered copies
    del m
    gc.collect()

    cli, dec = tm["tables"]["clients"], tm["tables"]["decisions"]
    key_sorted = np.sort(kv2[users])
    split_train = int(b.E.c("old_split", "TRAIN"))
    if b.full:
        checks += [D10.chk("PRETRAIN_TRAIN.n_users_eq_band", tm["n_users"], D10.N_USERS,
                           "BANDS['TIEBREAK'] - BANDS['MAIN']"),
                   D10.chk("PRETRAIN_TRAIN.n_decisions", tm["n_decisions"], D10.N_DECISIONS, D10.N_DECISIONS_SOURCE)]
    checks += [
        D10.chk("PRETRAIN_TRAIN.n_users", tm["n_users"], int(users.size), "band construction"),
        D10.chk("PRETRAIN_TRAIN.n_decisions_eq_sum_n_eligible", tm["n_decisions"], int(b.u_nelig[users].sum()),
                "locked M01 row_count == n_eligible"),
        D10.chk("PRETRAIN_TRAIN.old_split_all_TRAIN", bool(np.all(dec["old_split"].codes == split_train)), True,
                "TRAIN-period rows only"),
        D10.chk("PRETRAIN_TRAIN.no_calib_subset", int(cli["in_calib_prefix"].sum()), 0, "D10: no calibration subset"),
    ]
    for k, v in tm["assertions"].items():
        checks.append(D10.chk(f"PRETRAIN_TRAIN.assertion.{k}", bool(v), True, "locked MANIFEST_SPEC M01 assertion"))
    hard = [c for c in checks if c["hard"] and not c["pass"]]
    if hard:
        raise BuildBlocked(f"DATA-10 manifest checks failed: {[c['name'] for c in hard]}")

    population = population_block(sel, key_sorted, bands)
    expected = [("n_users", tm["n_users"], D10.N_USERS, "BANDS['TIEBREAK'] - BANDS['MAIN']"),
                ("n_decisions", tm["n_decisions"], D10.N_DECISIONS, D10.N_DECISIONS_SOURCE)]
    if D10.PRETRAIN_SHA_V2 is not None:
        expected.append(("population_sha256_selection_key_v2", population_sha256(key_sorted), D10.PRETRAIN_SHA_V2,
                         "DATA-10 local band-only measurement"))
    enc = FC.encode_tables(b, D10.TRAIN_ID, tm)
    meta = FC.tables_meta_of(enc)
    h = D10.d10_header(b, tm, meta, lock=lock, code=code, population=population, expected=expected,
                       users_keys_v2=kv2[users])
    written = FC.write_manifest(out_root, D10.TRAIN_ID, enc, h) if write else {}
    if write:
        log(f"wrote {D10.TRAIN_ID}: " + ", ".join(f"{t['table']}={t['row_count']}" for t in meta))
        FC.atomic_write_bytes(out_root, "population/PRETRAIN_NONMAIN_SELECTION_KEY_V2.u64be",
                              np.ascontiguousarray(key_sorted.astype(">u8")).tobytes())
    del enc
    entry = FC.hashes_entry(h, meta, (written or {}).get("header_file_sha256"))
    rec = {"receipt": "DATA10_BUILD_RECEIPT", "task": D10.TASK, "status": "PASS", "full_build": b.full,
           "buckets": len(b.buckets), "python_utc_end": utc_now(),
           "isolation_modules_checked": len(ppsi_env.assert_isolated()), "lock": lock, "code": code,
           "checks": checks, "n_checks": len(checks), "n_failed": len([c for c in checks if not c["pass"]]),
           "manifests": {D10.TRAIN_ID: entry}, "role": D10.ROLE, "eval_manifest": D10.EVAL_ID,
           "population": population, "n_validation_rows_collected_and_released": n_val_rows_released,
           "timings_s": {"prepare": round(t_prep - t0, 1), "extract": round(t_extract, 1),
                        "total": round(time.time() - t0, 1), **b.timings},
           "test_split_read": False, "no_raw_ids_written": True}
    if write:
        FC.write_json_self_hashed(out_root, "receipts/DATA10_BUILD_RECEIPT.json", rec)
    rec.update(_tm=tm, _header=h, _meta=meta, _users=users, _keys_v2=kv2, _pool_sorted=po["pool_sorted"])
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DATA-10 FA_WARM pretraining manifest (TIEBREAK minus MAIN)")
    ap.add_argument("--out", default=str(V4 / D10.OUT_REL))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--buckets", default="all", help="'all' or a comma list (partial smoke; disables hard checks)")
    ap.add_argument("--band-only", action="store_true", help="no extract / manifest: the band receipt only")
    a = ap.parse_args(argv)
    refuse_test_manifest(D10.TRAIN_ID)
    buckets = None if a.buckets == "all" else [int(x) for x in a.buckets.split(",")]
    b = D10Builder(ds_root=Path(a.dataset) if a.dataset else None, out_root=Path(a.out), buckets=buckets, log=log)
    if builder_code_sha256() != FC.DATA04_BUILDER_CODE_SHA256:
        log("WARNING: v4data builder_code_sha256 differs from DATA-04 (hard failure in a full build)")
    if a.band_only:
        rec = run_band_only(b, Path(a.out))
        log(f"DATA-10 band-only: n_failed={rec['n_failed']} n_users={rec['population']['n_users']} "
            f"n_eligible={rec['n_eligible_examples']} sha={rec['population']['population_sha256_selection_key_v2']}")
        return 0 if rec["n_failed"] == 0 else 3
    rec = run_build(b, Path(a.out))
    log(f"DATA-10 done: n_failed={rec['n_failed']} n_users={rec['manifests'][D10.TRAIN_ID]['n_users']} "
        f"n_decisions={rec['manifests'][D10.TRAIN_ID]['n_decisions']}")
    return 0 if rec["n_failed"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
