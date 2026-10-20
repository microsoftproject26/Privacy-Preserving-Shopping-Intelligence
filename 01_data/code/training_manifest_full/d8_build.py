"""v4data08.d8_build (owner: Data engineer, DATA-08) - the C_FULL training manifest FULL_TRAIN_5_100_TIEBREAK.

  users      the locked TIEBREAK band (v4fs_data.fs_common.BANDS['TIEBREAK'] = ranks [0, 1,348,682) of the ascending
             selection_key_v2 order of the CENSUS-02 5-100 pool; see v4data08.d8_common for the 1,315,914 note);
  rows       the locked M01 definition: TRAIN-split, loss-eligible decisions whose target lies before TRAIN_END
             (v4data.manifests.ManifestBuilder.pass_extract + _train_manifest, unedited);
  layout     clients / decisions / central_order - exactly MAIN_TRAIN_5_100_131072's tables and column types;
  eval       none built: role C_FULL evaluates on MAIN_VALIDATION_5_100_131072 unchanged (d8_finalize / REGISTRY ext).
One pass_extract() with self.cohort = TB alone (D6Builder.set_one; no CALIB / cold / panel bits). Memory: the
VALIDATION rows the locked pass collects for COHORT users are released right after the pass (no eval manifest is
built), and the M01 row arrays are released as soon as the decisions table exists; see the DATA-08 handoff for the
peak-RSS estimate (the job runs under the 16 GiB guard, v4data.run_bounded).
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
from v4data.common import builder_code_sha256, digest_rows, population_sha256, utc_now  # noqa: E402
from v4data.manifests import BuildBlocked, refuse_test_manifest  # noqa: E402
from v4data06.d6_build import D6Builder  # noqa: E402
from v4data08 import d8_common as D8  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

V4 = D8.V4


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class D8Builder(D6Builder):
    """D6Builder (the locked ManifestBuilder + an injected population per pass), unchanged; DATA-08 adds only the
    driver below."""


def pool_order(b) -> dict:
    """The CENSUS-02 5-100 pool in ascending selection_key_v2 order (the same computation as d6_build.compute_bands)."""
    strat = C2.stratum_codes(b.u_nelig, b.u_ntrain)
    keys_v2 = C2.selection_keys_v2(digest_rows(b.u_dig))
    if np.unique(keys_v2).size != keys_v2.size:
        raise BuildBlocked("two users share a selection_key_v2: BLOCK and escalate (CLIENT_CONTRACT C2.5)")
    pool_c = np.flatnonzero(strat == C2.S_C)
    pool_sorted = pool_c[np.argsort(keys_v2[pool_c], kind="stable")]
    return {"keys_v2": keys_v2, "pool_sorted": pool_sorted, "pool_keys_sorted": keys_v2[pool_sorted]}


def band_checks(b, po: dict, sel: dict, bands: dict) -> list:
    """Structural band checks always; the real-population constants (pool / prefix / TB hashes and counts) only in a
    full build, where they are hard (a partial or synthetic build has another population by construction)."""
    ks = po["pool_keys_sorted"]
    out = []
    if b.full:
        out += [D8.chk("pool.n_users", int(ks.size), D8.POOL_N, "CENSUS-02 / FS-01 / DATA-06"),
                D8.chk("pool.population_sha256_selection_key_v2", population_sha256(ks), D8.POOL_SHA_V2,
                       "CENSUS-02 / FS-01 / DATA-06")]
        for k, ref in FC.PREFIX_SHA_V2.items():
            out.append(D8.chk(f"pool.prefix{k}.population_sha256_selection_key_v2", population_sha256(ks[:k]), ref,
                              "CENSUS-02 candidate-N table"))
        tb_keys = np.sort(po["keys_v2"][sel["tb"]])
        out += [D8.chk("TB.n_users", int(sel["tb"].size), D8.TB_N_USERS, "v4fs_data.fs_common.BANDS['TIEBREAK']"),
                D8.chk("TB.population_sha256_selection_key_v2", population_sha256(tb_keys), D8.TB_SHA_V2,
                       "FS-01 handoff §1 band table (measured)"),
                D8.chk("TB.n_eligible_examples", int(b.u_nelig[sel["tb"]].sum()), D8.TB_N_DECISIONS,
                       D8.TB_N_DECISIONS_SOURCE)]
    for name, (obs, exp) in sel["checks"].items():
        out.append(D8.chk(name, obs, exp, "rank-band construction over the shared pool"))
    out.append(D8.chk("TB.every_user_has_ge1_eligible_example", bool(np.all(b.u_nelig[sel["tb"]] >= 1)), True,
                      "pool definition (5 <= n_eligible <= 100)"))
    return out


def run_build(b: D8Builder, out_root: Path, *, bands: dict | None = None, write: bool = True) -> dict:
    bands = D8.BANDS if bands is None else bands
    refuse_test_manifest(D8.TRAIN_ID)
    t0 = time.time()
    lock = FC.lock_info()
    code = D8.d8_code_hashes()
    checks = [D8.chk("code.v4data_builder_code_sha256_is_DATA04", b.code_sha, FC.DATA04_BUILDER_CODE_SHA256,
                     "DATA-04 handoff", hard=b.full)]
    b.prepare()
    t_prep = time.time()
    if b.full:
        checks += b.universe_checks()
    po = pool_order(b)
    sel = D8.tb_band(po["pool_sorted"], bands)
    checks += band_checks(b, po, sel, bands)
    hard = [c for c in checks if c["hard"] and not c["pass"]]
    if hard:
        raise BuildBlocked(f"DATA-08 band checks failed: {[c['name'] for c in hard]}")
    tb = sel["tb"]
    kv2 = po["keys_v2"]

    tp0 = time.time()
    b.set_one(cohort=tb)                                    # COHORT bit = TB only; no CALIB / cold / panel bits
    b.extract()
    t_extract = time.time() - tp0
    if b.full:
        checks += b.recon_checks()
    n_val_rows_released = int(b.rows_val["g"].size)
    b.rows_val, b.rows_ev = None, None                      # no eval manifest is built (memory)
    gc.collect()

    users = np.sort(np.asarray(tb, dtype=np.int64))
    m = b.rows_m01
    in_tb = np.zeros(b.u_dig.size, dtype=bool)
    in_tb[users] = True
    checks.append(D8.chk("M01_rows.all_in_TB", bool(np.all(in_tb[m["g"]])), True, "pass_extract COHORT bit = TB"))
    ord_of = np.full(b.u_dig.size, -1, dtype=np.int64)
    ord_of[users] = b._dense_rank_by_digest(users)
    tm = b._train_manifest(D8.DEFINITION, m, users, ord_of, None)
    b.rows_m01 = None                                       # the decisions table holds its own reordered copies
    del m
    gc.collect()

    cli, dec = tm["tables"]["clients"], tm["tables"]["decisions"]
    tb_key_sorted = np.sort(kv2[users])
    split_train = int(b.E.c("old_split", "TRAIN"))
    if b.full:
        checks += [D8.chk("C_FULL_TRAIN.n_users_eq_TB", tm["n_users"], D8.TB_N_USERS,
                          "v4fs_data.fs_common.BANDS['TIEBREAK']"),
                   D8.chk("C_FULL_TRAIN.n_decisions", tm["n_decisions"], D8.TB_N_DECISIONS, D8.TB_N_DECISIONS_SOURCE)]
    checks += [
        D8.chk("C_FULL_TRAIN.n_users", tm["n_users"], int(users.size), "band construction"),
        D8.chk("C_FULL_TRAIN.n_decisions_eq_sum_n_eligible", tm["n_decisions"], int(b.u_nelig[users].sum()),
               "locked M01 row_count == n_eligible"),
        D8.chk("C_FULL_TRAIN.old_split_all_TRAIN", bool(np.all(dec["old_split"].codes == split_train)), True,
               "TRAIN-period rows only"),
        D8.chk("C_FULL_TRAIN.no_calib_subset", int(cli["in_calib_prefix"].sum()), 0, "D8: no calibration subset"),
    ]
    for k, v in tm["assertions"].items():
        checks.append(D8.chk(f"C_FULL_TRAIN.assertion.{k}", bool(v), True, "locked MANIFEST_SPEC M01 assertion"))
    hard = [c for c in checks if c["hard"] and not c["pass"]]
    if hard:
        raise BuildBlocked(f"DATA-08 manifest checks failed: {[c['name'] for c in hard]}")

    population = {"band": D8.BAND, "rank_range": sel["ranks"], "pool_size": sel["pool_size"],
                  "n_users": int(users.size), "population_sha256_selection_key_v2": population_sha256(tb_key_sorted),
                  "definition_source": "v4fs_data.fs_common.BANDS['TIEBREAK'] (code/v4fs_data/fs_common.py:48); "
                                       "DECISION_LOG 41/42; ADR-005",
                  "excluded_bands": {n: list(bands[n]) for n in D8.EXCLUDED_BANDS},
                  "amendment_06_stated_users": D8.AMENDMENT_06_STATED_USERS, "discrepancy_note": D8.DISCREPANCY_NOTE}
    expected = [("n_users", tm["n_users"], D8.TB_N_USERS, "v4fs_data.fs_common.BANDS['TIEBREAK']"),
                ("n_decisions", tm["n_decisions"], D8.TB_N_DECISIONS, D8.TB_N_DECISIONS_SOURCE),
                ("population_sha256_selection_key_v2", population_sha256(tb_key_sorted), D8.TB_SHA_V2,
                 "FS-01 handoff §1 band table")]
    enc = FC.encode_tables(b, D8.TRAIN_ID, tm)
    meta = FC.tables_meta_of(enc)
    h = D8.d8_header(b, tm, meta, lock=lock, code=code, population=population, expected=expected,
                     users_keys_v2=kv2[users])
    written = FC.write_manifest(out_root, D8.TRAIN_ID, enc, h) if write else {}
    if write:
        log(f"wrote {D8.TRAIN_ID}: " + ", ".join(f"{t['table']}={t['row_count']}" for t in meta))
        FC.atomic_write_bytes(out_root, "population/TIEBREAK_SELECTION_KEY_V2.u64be",
                              np.ascontiguousarray(tb_key_sorted.astype(">u8")).tobytes())
    del enc
    entry = FC.hashes_entry(h, meta, (written or {}).get("header_file_sha256"))
    rec = {"receipt": "DATA08_BUILD_RECEIPT", "task": D8.TASK, "status": "PASS", "full_build": b.full,
           "buckets": len(b.buckets), "python_utc_end": utc_now(),
           "isolation_modules_checked": len(ppsi_env.assert_isolated()), "lock": lock, "code": code,
           "checks": checks, "n_checks": len(checks), "n_failed": len([c for c in checks if not c["pass"]]),
           "manifests": {D8.TRAIN_ID: entry}, "role": D8.ROLE, "eval_manifest": D8.EVAL_ID,
           "population": population, "n_validation_rows_collected_and_released": n_val_rows_released,
           "timings_s": {"prepare": round(t_prep - t0, 1), "extract": round(t_extract, 1),
                        "total": round(time.time() - t0, 1), **b.timings},
           "test_split_read": False, "no_raw_ids_written": True}
    if write:
        FC.write_json_self_hashed(out_root, "receipts/DATA08_BUILD_RECEIPT.json", rec)
    rec["_tm"] = tm
    rec["_header"] = h
    rec["_meta"] = meta
    rec["_tb"] = users
    rec["_keys_v2"] = kv2
    rec["_pool_sorted"] = po["pool_sorted"]
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DATA-08 C_FULL training manifest (TIEBREAK band)")
    ap.add_argument("--out", default=str(V4 / D8.OUT_REL))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--buckets", default="all", help="'all' or a comma list (partial smoke; disables hard checks)")
    a = ap.parse_args(argv)
    refuse_test_manifest(D8.TRAIN_ID)
    buckets = None if a.buckets == "all" else [int(x) for x in a.buckets.split(",")]
    b = D8Builder(ds_root=Path(a.dataset) if a.dataset else None, out_root=Path(a.out), buckets=buckets, log=log)
    if builder_code_sha256() != FC.DATA04_BUILDER_CODE_SHA256:
        log("WARNING: v4data builder_code_sha256 differs from DATA-04 (hard failure in a full build)")
    rec = run_build(b, Path(a.out))
    log(f"DATA-08 done: n_failed={rec['n_failed']} n_users={rec['manifests'][D8.TRAIN_ID]['n_users']} "
        f"n_decisions={rec['manifests'][D8.TRAIN_ID]['n_decisions']}")
    return 0 if rec["n_failed"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
