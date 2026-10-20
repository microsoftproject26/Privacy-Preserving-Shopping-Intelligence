"""v4fs_data.fs_build (owner: Data engineer, FS-01) - the FAST FEATURE SEARCH sample manifests (DECISION_LOG 39).

The LOCKED DATA-04 builder (v4data.manifests.ManifestBuilder, builder_code_sha256 443254e9...) is driven with an
explicit warm population by SUBCLASSING it; v4data is never edited. Inherited and run unchanged: pass_support (P0),
pass_aggregate (PA), select_populations, pass_extract (PB, incl. the target-metadata join), _row_arrays, _censor,
_train_manifest, _eval_manifest, _header, _check_no_raw_ids, _columns, _arrow, _assert_manifest. The subclass only
(1) injects the population (set_population: role bits + role_of, exactly the structures select_populations fills),
(2) assembles one TRAIN + one VALIDATION manifest from the extracted rows with the builder's table functions.

Modes
  fs           FS_TRAIN_5_100_32768 (the M01_COHORT_TRAIN definition: tables clients / decisions / central_order,
               row filter, order, assertions) and FS_VALIDATION_5_100_32768 (the M02_VALIDATION_WARM definition:
               users / decisions, eval_mask, censor classes, OOV causes, target_status, VALIDATION = VS u VC) over the
               FS sample = the 32,768 users of the CENSUS-02 5-100 pool with the LARGEST selection_key_v2. Written to
               data_fs/ (parquet, headers, .OK markers, receipts, the pool / FS key lists as selection_key_v2 only).
  equivalence  the same code path with the OLD M05 population (the 65,536 smallest C1.2 selection keys of U, chosen by
               the locked select_populations): (a) with the M05/M06 definitions every MANIFEST_HASHES.json value of
               M05_CALIB_TRAIN / M06_CALIB_VALIDATION_WARM must be reproduced (table content hashes, ordered/eval/
               rankable decision-id hashes, population, counts, header_content_sha256); (b) with the FS configuration
               (M01/M02 definitions, no calibration prefix, FS header path) M06 must be reproduced exactly and M05 up
               to the in_calib_prefix column only; (c) the server M05/M06 parquet files are re-hashed. Only the
               receipt is written.
TEST: refused. Every ledger / feature read is v4data.manifests.read_non_test (scan filter split IN TRAIN / VS / VC, so no
TEST row is materialised); a manifest id naming TEST is refused before any file is opened.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import ppsi_env  # noqa: E402  (import isolation first, T43)

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from census_v2 import core as C2  # noqa: E402
from v4data import reference_counts as REF  # noqa: E402
from v4data.common import (N_BUCKETS, TRAIN_END_US, VALIDATION_END_US, atomic_write_bytes, builder_code_sha256,  # noqa: E402
                           digest_rows, json_bytes_strict, population_sha256, sha256_bytes, sha256_file,
                           table_content_sha256, utc_now)
from v4data.manifests import CALIB, COHORT, BuildBlocked, ManifestBuilder, refuse_test_manifest, reread_columns  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

V4 = FC.V4


class PopulationBuilder(ManifestBuilder):
    """The locked ManifestBuilder with an injected warm population (see the module docstring)."""

    def prepare(self) -> None:
        self.pass_support()
        self.pass_aggregate()
        self._recon0 = dict(self.recon)

    def set_population(self, users, *, calib_prefix: bool) -> None:
        P = np.unique(np.asarray(users, dtype=np.int64))
        if P.size == 0:
            raise BuildBlocked("empty warm population")
        if np.any(self.u_nelig[P] < 1):
            raise BuildBlocked("a warm-population user has no eligible TRAIN example (M01 needs row_count >= 1)")
        z = np.zeros(0, dtype=np.int64)
        self.cohort = P
        self.calib = P if calib_prefix else z
        self.cold_sorted = self.m04 = self.m04w = self.m04s = z
        self.role = np.zeros(self.u_dig.size, dtype=np.int64)
        self.role[P] |= COHORT
        if calib_prefix:
            self.role[P] |= CALIB
        self.role_of = dict(zip(digest_rows(self.u_dig[P]), zip(P.tolist(), self.role[P].tolist())))

    def extract(self) -> None:
        self.recon = dict(self._recon0)
        self.pass_extract()

    def assemble_pair(self, definition: str) -> tuple[dict, dict]:
        """definition M05_M06 = the builder's calibration pair; M01_M02 = the cohort pair (the FS configuration).
        Mirrors ManifestBuilder.assemble for exactly these two manifests; every table comes from the locked
        _train_manifest / _eval_manifest."""
        if definition == "M05_M06":
            tid, eid, tpar, epar = ("M05_CALIB_TRAIN", "M06_CALIB_VALIDATION_WARM", "M01_COHORT_TRAIN",
                                    "M02_VALIDATION_WARM")
        elif definition == "M01_M02":
            tid, eid, tpar, epar = "M01_COHORT_TRAIN", "M02_VALIDATION_WARM", None, "M01_COHORT_TRAIN"
        else:
            raise ValueError(definition)
        m = self.rows_m01
        users = np.sort(self.cohort)
        ord_of = np.full(self.u_dig.size, -1, dtype=np.int64)
        ord_of[users] = self._dense_rank_by_digest(users)
        tm = self._train_manifest(tid, m, users, ord_of, tpar)
        pos = np.bincount(m["tc"], minlength=self.K).astype(np.int64)
        v = self.rows_val
        keep = (v["role"] & COHORT) != 0
        em = self._eval_manifest(eid, {k: a[keep] for k, a in v.items()}, pos, "warm", epar)
        return tm, em

    # ------------------------------------------------------------------ reproduction of the full-population counts
    def universe_checks(self) -> list:
        U = np.flatnonzero(self.u_nelig >= 1)
        ks = np.sort(self.u_key[U])
        got = {"universe_size": int(U.size), "s5_total_examples": int(self.u_nelig.sum()),
               "cohortALL.population_sha256": population_sha256(ks)}
        return [chk(f"DATA04.{k}", v, REF.POPULATION[k][0], REF.POPULATION[k][1]) for k, v in got.items()]

    def recon_checks(self) -> list:
        out = [chk(f"DATA04.{k}", self.recon.get(k), val, src) for k, (val, src) in REF.FULL_POPULATION.items()]
        out += [chk(f"DATA04.{k}", v, 0, "DATA-04 integrity check") for k, v in self.recon.items()
                if k.startswith("check.")]
        return out


def chk(name, observed, expected, source, hard=True) -> dict:
    return {"name": name, "observed": FC.jsonable(observed), "expected": FC.jsonable(expected), "source": source,
            "pass": bool(observed == expected), "hard": bool(hard)}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ------------------------------------------------------------------------------------------------ FS header
def fs_header(b: PopulationBuilder, m: dict, tables_meta: list, *, fs_id: str, parent: str | None, lock: dict,
              code: dict, population: dict, expected: list, users_keys_v2: np.ndarray, transform_state: dict) -> dict:
    """The builder's own header for the definition manifest (M01 / M02), re-labelled for the FS manifest; the
    header_content_sha256 is recomputed with the builder's rule."""
    h = b._header(dict(m, expected=[]), tables_meta)
    definition = h["manifest_id"]
    h.pop("header_content_sha256")
    h["manifest_id"] = fs_id
    h["status"] = FC.FS_STATUS if b.full else "PARTIAL_TEST_BUILD"
    h["run_label"] = FC.FS_STATUS
    h["data_version"] = FC.DATA_VERSION
    h["task"] = FC.TASK
    h["definition"] = {"manifest_spec_definition": definition, "spec_id": h["spec_id"],
                       "spec_version": h["spec_version"],
                       "note": f"tables, row filter, order and assertions of {definition} (built by the locked "
                               "v4data.manifests functions) over the FS population instead of the M01 cohort; "
                               "target_class_cohort_train_positives and client_ord / eval_user_key are computed "
                               "within the FS population"}
    h["parent_manifests"] = [parent] if parent else []
    h["data_protocol_lock_sha256"] = lock["lock_self_sha256"]
    h["data_protocol_lock_file_sha256"] = lock["file_sha256"]
    h["population_lock_sha256"] = ("NOT_APPLICABLE: FEATURE_SEARCH_SCREEN sample (DECISION_LOG 39), not a "
                                   "POPULATION_PLAN cohort")
    h["population_sha256_rule"] = ("MANIFEST_SPEC_V4 hashing.population_sha256 over the C1.2 selection_key of this "
                                   "manifest's users, ascending")
    h["population_sha256_selection_key_v2"] = population_sha256(np.sort(users_keys_v2))
    h["population_sha256_selection_key_v2_rule"] = ("CENSUS-02 rule: v4data.common.population_sha256 over each user's "
                                                    "8-byte big-endian selection_key_v2, ascending")
    h["fs_population"] = population
    h["transform_state"] = transform_state
    h["code"] = code
    h["expected_counts_check"] = [{"field": f, "expected": FC.jsonable(e), "expected_source": s,
                                   "observed": FC.jsonable(o), "match": bool(o == e)} for f, o, e, s in expected]
    h["selection_refusal"] = FC.SELECTION_REFUSAL
    h = FC.jsonable(h)
    h["header_content_sha256"] = FC.header_content_sha256(h)
    return h


def transform_state_ref(path: Path, required: bool) -> dict:
    base = {"rule": "shared statistics = full TRAIN as fitted, identical for every arm (final Fable ruling relayed by "
                    "the Operations Director); no FS-specific statistic is fitted",
            "path": "data_v2/TRANSFORM_STATE_CLEAN_V2.json",
            "transform_state_sha256_expected": FC.TRANSFORM_STATE_SELF_SHA256}
    if not Path(path).is_file():
        if required:
            raise BuildBlocked(f"TRANSFORM_STATE_CLEAN_V2 not found at {path}")
        return dict(base, verified=False, note="not present on this host (partial / synthetic build)")
    from v4data.transform_state import load_state
    st = load_state(Path(path))
    ok = st["transform_state_sha256"] == FC.TRANSFORM_STATE_SELF_SHA256
    if required and not ok:
        raise BuildBlocked(f"TRANSFORM_STATE self-hash {st['transform_state_sha256']} != {FC.TRANSFORM_STATE_SELF_SHA256}")
    return dict(base, verified=bool(ok), transform_state_sha256=st["transform_state_sha256"],
                file_sha256=sha256_file(Path(path)))


# ------------------------------------------------------------------------------------------------ FS mode
def run_fs(b: PopulationBuilder, out_root: Path, *, lo=C2.C_LO, hi=C2.C_HI, n_fs=FC.FS_N, bands=None,
           disjoint_from=FC.DISJOINT_FROM, train_id=FC.FS_TRAIN_ID, eval_id=FC.FS_EVAL_ID, write=True,
           transform_state_path: Path | None = None) -> dict:
    for mid in (train_id, eval_id):
        refuse_test_manifest(mid)
    t0 = time.time()
    lock = FC.lock_info()
    code = FC.fs_code_hashes()
    ts = transform_state_ref(transform_state_path or (V4 / "data_v2" / "TRANSFORM_STATE_CLEAN_V2.json"), b.full)
    b.prepare()
    t_prep = time.time()
    sel = FC.select_fs(b.u_nelig, b.u_dig, lo=lo, hi=hi, n_fs=n_fs, bands=bands, disjoint_from=disjoint_from)
    log(f"pool {sel['pool_size']} users, FS ranks {sel['fs_ranks']}, all_disjoint={sel['all_disjoint']}")
    b.set_population(sel["fs"], calib_prefix=False)
    b.extract()
    t_ext = time.time()
    tm, em = b.assemble_pair("M01_M02")
    full = b.full
    fs = sel["fs"]
    checks = [chk("code.v4data_builder_code_sha256_is_DATA04", b.code_sha, FC.DATA04_BUILDER_CODE_SHA256,
                  "DATA-04 handoff", hard=full)]
    if full:
        checks += b.universe_checks() + b.recon_checks()
        checks += [chk("pool.n_users", sel["pool_size"], FC.POOL_N, "CENSUS-02 / Fable D1"),
                   chk("pool.supervised_examples", sel["pool_examples"], FC.POOL_EXAMPLES, "CENSUS-02 / Fable D1"),
                   chk("pool.population_sha256_selection_key_v2", sel["pool_population_sha256_v2"], FC.POOL_SHA_V2,
                       "CENSUS-02 CANDIDATE_N_PROJECTIONS pool_population_sha256")]
        for k, ref in FC.PREFIX_SHA_V2.items():
            checks.append(chk(f"pool.prefix{k}.population_sha256_selection_key_v2",
                              population_sha256(sel["keys_sorted"][:k]), ref, "CENSUS-02 candidate-N table"))
        checks.append(chk("fs.ranks", sel["fs_ranks"], list(FC.BANDS["FS"]), "FS-01 packet + final Fable ruling"))
    for name, d in sel["disjointness"].items():            # full-scale bands cannot hold on a partial smoke pool
        checks.append(chk(f"fs.disjoint_from_{name}", d["disjoint"], True, "FS-01 packet + final Fable ruling",
                          hard=full or bands is not None))
    cli = tm["tables"]["clients"]
    usr = em["tables"]["users"]
    n_cli = cli["row_count"]
    kv2 = FC.keys_v2_of_hex(cli["client_key_hex"])
    checks += [
        chk("train.n_users", tm["n_users"], n_fs, "FS-01 packet"),
        chk("train.assertions_all_true", sorted(k for k, v in tm["assertions"].items() if v is not True), [],
            "MANIFEST_SPEC M01 assertions"),
        chk("eval.assertions_all_true", sorted(k for k, v in em["assertions"].items() if v is not True), [],
            "MANIFEST_SPEC M02 assertions"),
        chk("train.every_client_n_in_pool_bounds", bool(np.all((n_cli >= lo) & (n_cli <= hi))), True, "CENSUS-02 pool"),
        chk("train.keys_v2_recomputed_from_client_key_hex_equal_fs", bool(np.array_equal(np.sort(kv2), sel["fs_keys_v2"])),
            True, "independent recompute"),
        chk("train.n_decisions_equals_aggregate", tm["n_decisions"], int(b.u_nelig[fs].sum()), "PA aggregates"),
        chk("train.in_calib_prefix_all_false", int(cli["in_calib_prefix"].sum()), 0, "no calibration prefix in FS"),
        chk("eval.users_subset_of_train", bool(set(usr["client_key_hex"].tolist()) <= set(cli["client_key_hex"].tolist())),
            True, "M02 assertion 'every client_key_hex is present in M01.clients'"),
        chk("eval.n_users_equals_fs_users_with_validation", em["n_users"], int((b.u_nval[fs] >= 1).sum()), "PA aggregates"),
        chk("eval.n_decisions_equals_aggregate", em["n_decisions"], int(b.u_nval[fs].sum()), "PA aggregates"),
        chk("eval.n_eval_rows_equals_aggregate", em["n_eval_rows"], int(b.u_neval[fs].sum()), "PA aggregates"),
        chk("eval.n_rankable_rows_equals_aggregate", em["n_rankable_rows"], int(b.u_nrank[fs].sum()), "PA aggregates"),
        chk("eval.eval_rows_equal_rankable_plus_oov", em["n_eval_rows"],
            em["counts_by_cause"]["n_rankable_rows"] + em["counts_by_cause"]["n_oov_rows"], "identity"),
        chk("train.max_decision_ts_lt_train_end",
            bool(np.all(tm["tables"]["decisions"]["decision_ts_us"] < TRAIN_END_US)), True, "split bound"),
        chk("eval.max_decision_ts_lt_validation_end",
            bool(np.all(em["tables"]["decisions"]["decision_ts_us"] < VALIDATION_END_US)), True, "TEST gate"),
        chk("target_meta_join_mismatches", b.meta_checks["target_pid_mismatch"] + b.meta_checks["target_ts_mismatch"], 0,
            "DATA-04 target-metadata join"),
    ]
    hard_fail = [c for c in checks if c["hard"] and not c["pass"]]
    if hard_fail:
        raise BuildBlocked(f"FS build checks failed: {[c['name'] for c in hard_fail]}")
    population = {
        "definition": ("the n_fs users of the CENSUS-02 5-100 pool (5 <= n_eligible_TRAIN_examples <= 100, catalogue A, "
                       "DATA-04 eligibility) with the LARGEST selection_key_v2 = int.from_bytes(sha256(b'"
                       "PPSI_V4_POP_5_100_SEL_20260926|' + digest32).digest()[:8], 'big'), digest32 = the CLIENT_CONTRACT "
                       "C1.2 digest; ranks in the ascending key order (rank 0 = smallest key)"),
        "n_fs": int(n_fs), "pool_size": sel["pool_size"], "pool_examples": sel["pool_examples"],
        "pool_population_sha256_selection_key_v2": sel["pool_population_sha256_v2"], "fs_ranks": sel["fs_ranks"],
        "fs_population_sha256_selection_key_v2": sel["fs_population_sha256_v2"],
        "fs_key_v2_min_hex": sel["fs_key_v2_min_hex"], "fs_key_v2_max_hex": sel["fs_key_v2_max_hex"],
        "key_v2_just_below_fs_hex": sel["key_v2_below_fs_max_hex"], "disjointness": sel["disjointness"],
        "bands": {k: list(v) for k, v in (bands or FC.BANDS).items()}}
    exp_t = [("n_users", tm["n_users"], n_fs, "FS-01 packet / final Fable ruling"),
             ("fs_ranks", sel["fs_ranks"], [sel["pool_size"] - n_fs, sel["pool_size"]], "top n_fs of the pool"),
             ("all_bands_disjoint", sel["all_disjoint"], True, "FS-01 packet / final Fable ruling")]
    if full:
        exp_t += [("pool_size", sel["pool_size"], FC.POOL_N, "CENSUS-02"),
                  ("pool_population_sha256_selection_key_v2", sel["pool_population_sha256_v2"], FC.POOL_SHA_V2,
                   "CENSUS-02")]
    exp_e = [("users_subset_of_parent", True, True, "M02 assertion"),
             ("n_eval_rows_equals_rankable_plus_oov", em["n_eval_rows"],
              em["counts_by_cause"]["n_rankable_rows"] + em["counts_by_cause"]["n_oov_rows"], "identity")]
    man = {}
    for m, fs_id, parent, exp, keys in ((tm, train_id, None, exp_t, kv2),
                                        (em, eval_id, train_id, exp_e, FC.keys_v2_of_hex(usr["client_key_hex"]))):
        enc = FC.encode_tables(b, fs_id, m)
        meta = FC.tables_meta_of(enc)
        h = fs_header(b, m, meta, fs_id=fs_id, parent=parent, lock=lock, code=code, population=population,
                      expected=exp, users_keys_v2=keys, transform_state=ts)
        man[fs_id] = {"m": m, "enc": enc, "meta": meta, "header": h}
    t_asm = time.time()
    written = {}
    if write:
        for fs_id, x in man.items():
            w = FC.write_manifest(out_root, fs_id, x["enc"], x["header"])
            written[fs_id] = w
            log(f"wrote {fs_id}: " + ", ".join(f"{t['table']}={t['row_count']}" for t in x["meta"]))
        pk = FC.atomic_write_bytes(out_root, "population/POOL_5_100_SELECTION_KEY_V2.u64be",
                                   np.ascontiguousarray(sel["keys_sorted"].astype(">u8")).tobytes())
        fk = FC.atomic_write_bytes(out_root, "population/FS_SELECTION_KEY_V2.u64be",
                                   np.ascontiguousarray(sel["fs_keys_v2"].astype(">u8")).tobytes())
        pop_files = {"population/POOL_5_100_SELECTION_KEY_V2.u64be": sha256_file(pk),
                     "population/FS_SELECTION_KEY_V2.u64be": sha256_file(fk)}
        if pop_files["population/POOL_5_100_SELECTION_KEY_V2.u64be"] != sel["pool_population_sha256_v2"] or \
                pop_files["population/FS_SELECTION_KEY_V2.u64be"] != sel["fs_population_sha256_v2"]:
            raise BuildBlocked("key-list file sha256 differs from its population_sha256 (they are the same bytes)")
    else:
        pop_files = {}
    hashes = {fs_id: FC.hashes_entry(x["header"], x["meta"], (written.get(fs_id) or {}).get("header_file_sha256"))
              for fs_id, x in man.items()}
    stats = {
        "train": {"n_users": tm["n_users"], "n_decisions": tm["n_decisions"],
                  "n_eligible_quantiles_linear": C2.quantiles(n_cli), "counts_by_cause": tm["counts_by_cause"]},
        "eval": {"n_users_with_ge1_validation_decision": em["n_users"],
                 "n_users_with_ge1_eval_row": int((usr["n_eval_rows"] >= 1).sum()),
                 "n_decisions": em["n_decisions"], "n_eval_rows": em["n_eval_rows"],
                 "n_rankable_rows": em["n_rankable_rows"], "counts_by_cause": em["counts_by_cause"]}}
    rec = {"receipt": "FS_BUILD_RECEIPT", "task": FC.TASK, "status": "PASS", "mode": "fs", "full_build": full,
           "buckets": len(b.buckets), "python_utc_end": utc_now(), "host": FC.host_info(),
           "isolation_modules_checked": len(ppsi_env.assert_isolated()), "lock": lock, "code": code,
           "transform_state": ts, "population": population, "band_population_sha256_selection_key_v2": sel["band_hashes"],
           "population_files_sha256": pop_files, "manifests": hashes, "stats": stats, "checks": checks,
           "n_checks": len(checks), "n_failed": len([c for c in checks if not c["pass"]]),
           "target_metadata_join": b.meta_checks, "timings_s": {"prepare": round(t_prep - t0, 1),
                                                                "extract": round(t_ext - t_prep, 1),
                                                                "assemble_encode": round(t_asm - t_ext, 1),
                                                                "total": round(time.time() - t0, 1),
                                                                **b.timings},
           "test_split_read": False, "no_raw_ids_written": True}
    if write:
        FC.write_json_self_hashed(out_root, "receipts/MANIFEST_HASHES_FS.json",
                                  {"receipt": "MANIFEST_HASHES_FS", "data_version": FC.DATA_VERSION,
                                   "builder_code_sha256": b.code_sha, "fs_code_sha256": code["fs_code_sha256"],
                                   "candidate_order_hash": b.candidate_order_hash, "manifests": hashes})
        FC.write_json_self_hashed(out_root, "receipts/FS_BUILD_RECEIPT.json", rec)
    rec["_man"] = man
    return rec


# ------------------------------------------------------------------------------------------------ equivalence
def run_equivalence(b: PopulationBuilder, out_root: Path, *, manifest_hashes: Path, server_manifest_dir: Path | None,
                    write=True) -> dict:
    t0 = time.time()
    lock = FC.lock_info()
    code = FC.fs_code_hashes()
    mh_bytes = Path(manifest_hashes).read_bytes()
    mh = json.loads(mh_bytes)
    b.prepare()
    b.select_populations()                            # the LOCKED DATA-04 selection of the old M05 population
    P_old = np.asarray(b.calib, dtype=np.int64).copy()
    pop_old = population_sha256(np.sort(b.u_key[P_old]))
    checks = [chk("old_M05.population_sha256", pop_old, mh["manifests"]["M05_CALIB_TRAIN"]["population_sha256"],
                  "MANIFEST_HASHES.json"),
              chk("code.v4data_builder_code_sha256_is_DATA04", b.code_sha, mh["builder_code_sha256"],
                  "MANIFEST_HASHES.json builder_code_sha256")]
    if b.full:
        checks.append(chk("old_M05.population_sha256_is_DATA04", pop_old, FC.OLD_M05_POP_SHA, "DATA-04 handoff"))
        for k, v in b.pop_hash.items():
            if k != "universe_size":
                checks.append(chk(f"cohort_key_hashes.{k}", v, mh["cohort_key_hashes"][k], "MANIFEST_HASHES.json"))
    b.set_population(P_old, calib_prefix=True)
    b.extract()
    if b.full:
        checks += b.recon_checks()
    # (a) the M05 / M06 definitions: must reproduce MANIFEST_HASHES exactly, header included
    tm, em = b.assemble_pair("M05_M06")
    part_a = {}
    for m in (tm, em):
        b._assert_manifest(m)                         # the builder's assertions and (full build) expected counts
        enc = FC.encode_tables(b, m["manifest_id"], m)
        meta = FC.tables_meta_of(enc)
        hdr = b._header(m, meta)
        got = FC.hashes_entry(hdr, meta, sha256_bytes(json_bytes_strict(hdr)))
        part_a[m["manifest_id"]] = FC.compare_entries(got, mh["manifests"][m["manifest_id"]])
        del enc
    # (b) the FS configuration (M01/M02 definitions, no calibration prefix, FS header) on the same population
    b.calib = np.zeros(0, dtype=np.int64)
    tm2, em2 = b.assemble_pair("M01_M02")
    ref5, ref6 = mh["manifests"]["M05_CALIB_TRAIN"], mh["manifests"]["M06_CALIB_VALIDATION_WARM"]
    enc_t = {t: (c, n) for t, c, d, n in FC.encode_tables(b, "FS_EQUIV_TRAIN_OLD_M05", tm2)}
    enc_e = FC.encode_tables(b, "FS_EQUIV_VALIDATION_OLD_M05", em2)
    meta_e = FC.tables_meta_of(enc_e)
    cli_flip = dict(tm2["tables"]["clients"])
    cli_flip["in_calib_prefix"] = np.ones(len(cli_flip["client_ord"]), dtype=bool)
    flip_hash = table_content_sha256(b._columns(cli_flip))
    pop_info = {"definition": "EQUIVALENCE ONLY: the old M05 population (65,536 smallest C1.2 selection keys of U)",
                "population_sha256": pop_old}
    hdr_e = fs_header(b, em2, meta_e, fs_id="FS_EQUIV_VALIDATION_OLD_M05", parent="FS_EQUIV_TRAIN_OLD_M05", lock=lock,
                      code=code, population=pop_info, expected=[], users_keys_v2=FC.keys_v2_of_hex(
                          em2["tables"]["users"]["client_key_hex"]), transform_state={"rule": "not used"})
    got_e = FC.hashes_entry(hdr_e, meta_e, None)
    part_b_fields = {
        "train.decisions.table_content_sha256": (enc_t["decisions"][0], ref5["tables"]["decisions"]["table_content_sha256"]),
        "train.central_order.table_content_sha256": (enc_t["central_order"][0],
                                                     ref5["tables"]["central_order"]["table_content_sha256"]),
        "train.clients.table_content_sha256_with_in_calib_prefix_set_true": (
            flip_hash, ref5["tables"]["clients"]["table_content_sha256"]),
        "train.clients.in_calib_prefix_true_count": (int(tm2["tables"]["clients"]["in_calib_prefix"].sum()), 0),
        "train.n_users": (tm2["n_users"], ref5["n_users"]), "train.n_decisions": (tm2["n_decisions"], ref5["n_decisions"]),
        "train.population_sha256": (tm2["population_sha256"], ref5["population_sha256"]),
        "train.counts_by_cause_equal_M05_definition": (FC.jsonable(tm2["counts_by_cause"]),
                                                       FC.jsonable(tm["counts_by_cause"])),
        "train.assertions_all_true": (all(v is True for v in tm2["assertions"].values()), True),
        "eval.users.table_content_sha256": (got_e["tables"]["users"]["table_content_sha256"],
                                            ref6["tables"]["users"]["table_content_sha256"]),
        "eval.decisions.table_content_sha256": (got_e["tables"]["decisions"]["table_content_sha256"],
                                                ref6["tables"]["decisions"]["table_content_sha256"]),
        "eval.counts_by_cause_equal_M06_definition": (FC.jsonable(em2["counts_by_cause"]),
                                                      FC.jsonable(em["counts_by_cause"])),
        "eval.assertions_all_true": (all(v is True for v in em2["assertions"].values()), True),
    }
    for k in ("ordered_decision_id_sha256", "eval_ordered_decision_id_sha256", "rankable_ordered_decision_id_sha256",
              "population_sha256", "n_users", "n_decisions", "n_eval_rows", "n_rankable_rows"):
        part_b_fields[f"eval.{k}"] = (got_e[k], ref6[k])
    part_b = {k: {"got": FC.jsonable(g) if not isinstance(g, dict) else "(dict)", "ref": FC.jsonable(r) if not isinstance(r, dict) else "(dict)",
                  "equal": bool(g == r)} for k, (g, r) in part_b_fields.items()}
    # (c) the server M05 / M06 parquet files re-hashed
    part_c = {}
    if server_manifest_dir is not None and Path(server_manifest_dir).is_dir():
        for mid in ("M05_CALIB_TRAIN", "M06_CALIB_VALIDATION_WARM"):
            for t, e in mh["manifests"][mid]["tables"].items():
                p = Path(server_manifest_dir) / f"{mid}.{t}.parquet"
                c = table_content_sha256(reread_columns(pq.ParquetFile(p).read()))
                part_c[f"{mid}.{t}"] = {"table_content_sha256": c, "file_sha256": sha256_file(p),
                                        "content_equal": c == e["table_content_sha256"],
                                        "file_equal": sha256_file(p) == e["file_sha256"]}
    all_a = all(v["all_equal"] for v in part_a.values())
    all_b = all(v["equal"] for v in part_b.values())
    all_c = bool(part_c) and all(v["content_equal"] and v["file_equal"] for v in part_c.values())
    hard_fail = [c for c in checks if c["hard"] and not c["pass"]]
    ok = all_a and all_b and (all_c or not b.full) and not hard_fail
    file_eq = {mid: {k: v["equal"] for k, v in r["fields"].items() if not v["in_verdict"]} for mid, r in part_a.items()}
    rec = {"receipt": "EQUIVALENCE_M05_M06", "task": FC.TASK, "status": "PASS" if ok else "FAIL", "all_equal": bool(ok),
           "full_build": b.full, "buckets": len(b.buckets), "python_utc_end": utc_now(), "host": FC.host_info(),
           "manifest_hashes_path": str(manifest_hashes), "manifest_hashes_file_sha256": sha256_bytes(mh_bytes),
           "code": code, "lock": lock, "old_M05_population_sha256": pop_old, "old_M05_n_users": int(P_old.size),
           "a_M05_M06_definitions_vs_MANIFEST_HASHES": part_a, "a_all_equal": all_a,
           "a_parquet_file_sha256_equal_informative": file_eq,
           "b_FS_configuration_vs_M05_M06": part_b, "b_all_equal": all_b,
           "c_server_files_rehashed": part_c, "c_all_equal": all_c, "checks": checks,
           "n_checks": len(checks), "n_failed": len(hard_fail),
           "verdict_rule": ("a: every table content hash, ordered/eval/rankable decision-id hash, population, count and "
                            "header_content_sha256 equal MANIFEST_HASHES (Parquet file bytes are informative only); "
                            "b: the FS configuration reproduces M06 exactly and M05 exactly except in_calib_prefix "
                            "(False in FS; content equal once set True); c: the server files hash to MANIFEST_HASHES"),
           "timings_s": {"total": round(time.time() - t0, 1), **b.timings}, "test_split_read": False}
    if write:
        FC.write_json_self_hashed(out_root, "receipts/EQUIVALENCE_M05_M06.json", rec)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="FS-01 FAST FEATURE SEARCH manifests")
    ap.add_argument("--mode", choices=("fs", "equivalence"), required=True)
    ap.add_argument("--out", default=str(V4 / "data_fs"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--buckets", default="all", help="'all' or a comma list (partial smoke; disables full checks)")
    ap.add_argument("--manifest-hashes", default=str(V4 / "data_v2" / "receipts" / "MANIFEST_HASHES.json"))
    ap.add_argument("--server-manifest-dir", default=str(V4 / "data_v2" / "manifests"))
    ap.add_argument("--transform-state", default=str(V4 / "data_v2" / "TRANSFORM_STATE_CLEAN_V2.json"))
    a = ap.parse_args(argv)
    for mid in (FC.FS_TRAIN_ID, FC.FS_EVAL_ID):
        refuse_test_manifest(mid)
    buckets = None if a.buckets == "all" else [int(x) for x in a.buckets.split(",")]
    b = PopulationBuilder(ds_root=Path(a.dataset) if a.dataset else None, out_root=Path(a.out), buckets=buckets,
                          log=log)
    if builder_code_sha256() != FC.DATA04_BUILDER_CODE_SHA256:
        log("WARNING: v4data builder_code_sha256 differs from DATA-04 (hard failure in a full build)")
    if a.mode == "fs":
        rec = run_fs(b, Path(a.out), transform_state_path=Path(a.transform_state))
        log(f"FS done: {json.dumps({k: rec['manifests'][k]['n_users'] for k in rec['manifests']})}")
        return 0
    rec = run_equivalence(b, Path(a.out), manifest_hashes=Path(a.manifest_hashes),
                          server_manifest_dir=Path(a.server_manifest_dir))
    log(f"equivalence: all_equal={rec['all_equal']} a={rec['a_all_equal']} b={rec['b_all_equal']} c={rec['c_all_equal']}")
    return 0 if rec["all_equal"] else 3


if __name__ == "__main__":
    sys.exit(main())
