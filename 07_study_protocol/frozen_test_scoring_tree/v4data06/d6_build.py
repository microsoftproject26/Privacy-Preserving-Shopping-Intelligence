"""v4data06.d6_build (owner: Data engineer, DATA-06) - the data version v2 manifests (ADR-005 R2+R4 binding
synthesis) over the LOCKED DATA-04 builder (v4data.manifests.ManifestBuilder, builder_code_sha256 443254e9...),
mechanical re-use of FS-01's v4fs_data pattern.

D6Builder subclasses ManifestBuilder and injects an EXPLICIT population per pass (set_one: role bits + role_of,
exactly the structures the locked select_populations fills), then reuses the locked, unedited table builders:
_train_manifest (M01/M05 shape), _eval_manifest (M02/M06/M03 shape), _panel_manifest (M04/M04W shape), _header.

Why five extract() passes instead of one: the locked pass_extract accumulates TRAIN counts_by_cause under exactly
two fixed tags, "M01" (role & COHORT) and "M05" (role & CALIB) - see v4data.manifests.ManifestBuilder.pass_extract.
Packing every COHORT-shaped v2 band (MAIN, UNEW_VAL, UNEW_TEST, SPARSE_PANEL, HEAVY_PANEL) into one COHORT-bit pass
would merge their counts_by_cause under the single "M01" tag. Each COHORT-shaped band is therefore built in its own
pass_extract() pass (self.cohort = that band alone). MAIN nests CALIB (COHORT+CALIB, exactly like the locked M01/M05
pair) and additionally carries TRUE_COLD (COLD bit), COLD_H (M04 bit) and WARM_TRUNCATION (M04W bit), which do not
depend on the "M01" tag, so they share MAIN's pass. Measured cost (FS-01 timings): pass_support 17.5s, pass_aggregate
38.6s (run once via prepare()), pass_extract ~57-60s per pass (run five times) - about 6-7 minutes total.

TEST: refused, unconditionally. Every ledger / feature read is v4data.manifests.read_non_test (scan filter
split IN TRAIN / VS / VC); a manifest id naming TEST is refused before any file is opened.
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

from census_v2 import core as C2  # noqa: E402
from v4data.common import builder_code_sha256, digest_rows, population_sha256, utc_now  # noqa: E402
from v4data.manifests import (CALIB as BIT_CALIB, COHORT as BIT_COHORT, COLD as BIT_COLD, P_M04 as BIT_M04,  # noqa: E402
                              P_M04W as BIT_M04W, BuildBlocked, ManifestBuilder, refuse_test_manifest)
from v4data06 import d6_common as DC  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

V4 = DC.V4


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class D6Builder(ManifestBuilder):
    """The locked ManifestBuilder with an injected population per pass_extract() call (see the module docstring)."""

    def prepare(self) -> None:
        self.pass_support()
        self.pass_aggregate()
        self._recon0 = dict(self.recon)

    def set_one(self, *, cohort, calib=None, cold=None, m04=None, m04w=None) -> None:
        """cohort/calib/cold/m04/m04w are index arrays into u_dig/u_nelig/... . Order is PRESERVED (not
        np.unique-sorted): self.cold_sorted must stay in the caller's selection_key_v2 ascending order, because
        the locked _panel_manifest's COLD_START assertion compares a panel's users to self.cold_sorted[:N_PANEL]
        by position, not by value."""
        z = np.zeros(0, dtype=np.int64)

        def _prep(a, name):
            if a is None or len(a) == 0:
                return z
            x = np.asarray(a, dtype=np.int64)
            if np.unique(x).size != x.size:
                raise BuildBlocked(f"duplicate user index in {name}")
            return x

        coh, cal, cld, m4, m4w = (_prep(cohort, "cohort"), _prep(calib, "calib"), _prep(cold, "cold"),
                                  _prep(m04, "m04"), _prep(m04w, "m04w"))
        if coh.size == 0:
            raise BuildBlocked("empty cohort population for this pass")
        if np.any(self.u_nelig[coh] < 1):
            raise BuildBlocked("a COHORT-bit user has no eligible TRAIN example (M01 needs row_count >= 1)")
        if cld.size and np.any(self.u_nelig[cld] != 0):
            raise BuildBlocked("a COLD-bit user has an eligible TRAIN example")
        if cal.size and not np.all(np.isin(cal, coh)):
            raise BuildBlocked("CALIB is not a subset of this pass's cohort")
        if m4.size and not np.all(np.isin(m4, cld)):
            raise BuildBlocked("the M04-bit panel is not a subset of this pass's cold population")
        if m4w.size and not np.all(np.isin(m4w, coh)):
            raise BuildBlocked("the M04W-bit panel is not a subset of this pass's cohort")
        self.cohort, self.calib, self.cold_sorted, self.m04, self.m04w, self.m04s = coh, cal, cld, m4, m4w, z
        self.role = np.zeros(self.u_dig.size, dtype=np.int64)
        self.role[coh] |= BIT_COHORT
        if cal.size:
            self.role[cal] |= BIT_CALIB
        if cld.size:
            self.role[cld] |= BIT_COLD
        if m4.size:
            self.role[m4] |= BIT_M04
        if m4w.size:
            self.role[m4w] |= BIT_M04W
        if np.any((self.role & BIT_COHORT) & (self.role & BIT_COLD)):
            raise BuildBlocked("a user is both cohort (warm) and cold")
        rel = np.flatnonzero(self.role != 0)
        self.role_of = dict(zip(digest_rows(self.u_dig[rel]), zip(rel.tolist(), self.role[rel].tolist())))

    def extract(self) -> None:
        self.recon = dict(self._recon0)
        self.pass_extract()

    def universe_checks(self) -> list:
        from v4data import reference_counts as REF
        U = np.flatnonzero(self.u_nelig >= 1)
        ks = np.sort(self.u_key[U])
        got = {"universe_size": int(U.size), "s5_total_examples": int(self.u_nelig.sum()),
               "cohortALL.population_sha256": population_sha256(ks)}
        return [DC.chk(f"DATA04.{k}", v, REF.POPULATION[k][0], REF.POPULATION[k][1]) for k, v in got.items()]

    def recon_checks(self) -> list:
        from v4data import reference_counts as REF
        out = [DC.chk(f"DATA04.{k}", self.recon.get(k), val, src) for k, (val, src) in REF.FULL_POPULATION.items()]
        out += [DC.chk(f"DATA04.{k}", v, 0, "DATA-04 integrity check") for k, v in self.recon.items()
                if k.startswith("check.")]
        return out


# ------------------------------------------------------------------------------------------------ band selection
def compute_bands(b: D6Builder) -> dict:
    strat = C2.stratum_codes(b.u_nelig, b.u_ntrain)
    keys_v2 = C2.selection_keys_v2(digest_rows(b.u_dig))
    if np.unique(keys_v2).size != keys_v2.size:
        raise BuildBlocked("two users share a selection_key_v2: BLOCK and escalate (CLIENT_CONTRACT C2.5)")
    pool_c = np.flatnonzero(strat == C2.S_C)
    o = np.argsort(keys_v2[pool_c], kind="stable")
    pool_sorted = pool_c[o]
    pool_keys_sorted = keys_v2[pool_sorted]
    P = int(pool_sorted.size)

    x1, y1 = DC.BANDS["U_NEW_VAL"]
    x2, y2 = DC.BANDS["U_NEW_TEST"]
    main = pool_sorted[:DC.N_MAIN]
    calib = pool_sorted[:DC.N_CALIB]
    unew_val = pool_sorted[x1:y1]
    unew_test = pool_sorted[x2:y2]

    def smallest_key_with_eval(idx: np.ndarray, want: int) -> np.ndarray:
        idx = idx[b.u_neval[idx] >= 1]
        if idx.size < want:
            raise BuildBlocked(f"only {idx.size} eligible users, need {want}")
        oo = np.argsort(keys_v2[idx], kind="stable")
        return idx[oo][:want]

    pool_b = np.flatnonzero(strat == C2.S_B)
    pool_d = np.flatnonzero(strat == C2.S_D)
    sparse = smallest_key_with_eval(pool_b, DC.N_PANEL)
    heavy = smallest_key_with_eval(pool_d, DC.N_PANEL)

    cold_idx = np.flatnonzero((strat == C2.S_A0) | (strat == C2.S_A1))
    cold_eval = cold_idx[b.u_neval[cold_idx] >= 1]
    true_cold_sorted = cold_eval[np.argsort(keys_v2[cold_eval], kind="stable")]
    if true_cold_sorted.size < DC.N_PANEL:
        raise BuildBlocked("fewer TRUE_COLD users than the COLD_H panel size")
    cold_h = true_cold_sorted[:DC.N_PANEL]

    warm_trunc = smallest_key_with_eval(main, DC.N_PANEL)

    return {"strat": strat, "keys_v2": keys_v2, "pool_sorted": pool_sorted, "pool_keys_sorted": pool_keys_sorted,
            "pool_size": P, "MAIN": main, "CALIB": calib, "U_NEW_VAL": unew_val, "U_NEW_TEST": unew_test,
            "SPARSE": sparse, "HEAVY": heavy, "TRUE_COLD": true_cold_sorted, "COLD_H": cold_h,
            "WARM_TRUNCATION": warm_trunc}


def band_integrity_checks(b: D6Builder, bands: dict) -> list:
    checks = []
    full = b.full
    checks.append(DC.chk("pool.n_users", bands["pool_size"], DC.POOL_N, "CENSUS-02 / FS-01", hard=full))
    checks.append(DC.chk("pool.population_sha256_selection_key_v2", population_sha256(bands["pool_keys_sorted"]),
                         DC.POOL_SHA_V2, "CENSUS-02 / FS-01 pool_population_sha256", hard=full))
    for k, ref in DC.PREFIX_SHA_V2.items():
        checks.append(DC.chk(f"pool.prefix{k}.population_sha256_selection_key_v2",
                             population_sha256(bands["pool_keys_sorted"][:k]), ref, "CENSUS-02 candidate-N table",
                             hard=full))
    checks.append(DC.chk("MAIN.population_sha256_selection_key_v2",
                         population_sha256(np.sort(bands["keys_v2"][bands["MAIN"]])), DC.MAIN_SHA_V2,
                         "DATA-06.md / CENSUS-02", hard=full))
    checks.append(DC.chk("CALIB.population_sha256_selection_key_v2",
                         population_sha256(np.sort(bands["keys_v2"][bands["CALIB"]])), DC.CALIB_SHA_V2,
                         "DATA-06.md / CENSUS-02", hard=full))
    # subset relationships
    checks.append(DC.chk("CALIB.subset_of_MAIN", bool(np.all(np.isin(bands["CALIB"], bands["MAIN"]))), True,
                         "band construction (nested prefix)"))
    checks.append(DC.chk("WARM_TRUNCATION.subset_of_MAIN", bool(np.all(np.isin(bands["WARM_TRUNCATION"],
                                                                                bands["MAIN"]))), True,
                         "band construction"))
    checks.append(DC.chk("COLD_H.subset_of_TRUE_COLD", bool(np.all(np.isin(bands["COLD_H"], bands["TRUE_COLD"]))),
                         True, "band construction"))
    # pairwise disjointness of the stratum-C bands (rank arithmetic, verified by explicit set overlap)
    for a_name, b_name in (("MAIN", "U_NEW_VAL"), ("MAIN", "U_NEW_TEST"), ("U_NEW_VAL", "U_NEW_TEST")):
        ov = int(np.intersect1d(bands[a_name], bands[b_name]).size)
        checks.append(DC.chk(f"disjoint.{a_name}_vs_{b_name}", ov, 0, "rank-band construction over the shared pool"))
    # cross-stratum disjointness (structurally guaranteed; checked defensively)
    for a_name in ("MAIN", "U_NEW_VAL", "U_NEW_TEST"):
        for b_name in ("SPARSE", "HEAVY", "TRUE_COLD", "COLD_H"):
            ov = int(np.intersect1d(bands[a_name], bands[b_name]).size)
            checks.append(DC.chk(f"disjoint.{a_name}_vs_{b_name}", ov, 0, "mutually exclusive strata (C vs B/D/A0-A1)"))
    ov = int(np.intersect1d(bands["SPARSE"], bands["HEAVY"]).size)
    checks.append(DC.chk("disjoint.SPARSE_vs_HEAVY", ov, 0, "mutually exclusive strata (B vs D)"))
    return checks


# ------------------------------------------------------------------------------------------------ manifest assembly
def build_pair(b: D6Builder, *, tid: str, eid: str, band_users: np.ndarray, class_pos_override=None):
    users = np.sort(np.asarray(band_users, dtype=np.int64))
    ord_of = np.full(b.u_dig.size, -1, dtype=np.int64)
    ord_of[users] = b._dense_rank_by_digest(users)
    umask = np.zeros(b.u_dig.size, dtype=bool)
    umask[users] = True
    m = b.rows_m01
    m_band = {k: v[umask[m["g"]]] for k, v in m.items()}
    tm = b._train_manifest(tid, m_band, users, ord_of, None)
    pos = (class_pos_override if class_pos_override is not None
           else np.bincount(m_band["tc"], minlength=b.K).astype(np.int64))
    v = b.rows_val
    v_band = {k: a[umask[v["g"]]] for k, a in v.items()}
    em = b._eval_manifest(eid, v_band, pos, "warm", tid)
    return tm, em, pos


def build_cold(b: D6Builder, *, cold_users: np.ndarray, pos_ref: np.ndarray) -> dict:
    users = np.asarray(cold_users, dtype=np.int64)
    umask = np.zeros(b.u_dig.size, dtype=bool)
    umask[users] = True
    v = b.rows_val
    v_band = {k: a[umask[v["g"]]] for k, a in v.items()}
    return b._eval_manifest("M03_COLD_PANEL_VAL", v_band, pos_ref, "cold", None)


# ------------------------------------------------------------------------------------------------ header + write
def finish_and_write(b: D6Builder, m: dict, *, out_id: str, role: str, parent, lock: dict, code: dict,
                     population: dict, expected: list, users_keys_v2: np.ndarray, out_root: Path, write: bool,
                     extra: dict | None = None) -> dict:
    enc = FC.encode_tables(b, out_id, m)
    meta = FC.tables_meta_of(enc)
    h = DC.d6_header(b, m, meta, out_id=out_id, role=role, parent=parent, lock=lock, code=code, population=population,
                     expected=expected, users_keys_v2=users_keys_v2, extra=extra)
    written = FC.write_manifest(out_root, out_id, enc, h) if write else {}
    return {"m": m, "enc": enc, "meta": meta, "header": h, "written": written}


def band_pop(name: str, keys_sorted, **extra) -> dict:
    """keys_sorted may be None for a stub built before the band's own extract() pass; the caller then fills
    n_users / population_sha256_selection_key_v2 in directly once the band's keys are known."""
    d = {"band": name}
    if keys_sorted is not None:
        d["n_users"] = int(np.asarray(keys_sorted).size)
        d["population_sha256_selection_key_v2"] = population_sha256(np.sort(np.asarray(keys_sorted, dtype=np.uint64)))
    d.update(extra)
    return d


# ------------------------------------------------------------------------------------------------ driver
def run_build(b: D6Builder, out_root: Path, *, write: bool = True) -> dict:
    for out_id in DC.ALL_IDS:
        refuse_test_manifest(out_id)
    t0 = time.time()
    lock = FC.lock_info()
    code = DC.d6_code_hashes()
    checks = [DC.chk("code.v4data_builder_code_sha256_is_DATA04", b.code_sha, DC.DATA04_BUILDER_CODE_SHA256,
                     "DATA-04 handoff", hard=b.full)]
    b.prepare()
    t_prep = time.time()
    if b.full:
        checks += b.universe_checks()
    bands = compute_bands(b)
    checks += band_integrity_checks(b, bands)
    kv2 = bands["keys_v2"]
    man: dict = {}
    written = {}
    pass_timings = []

    def emit(x, out_id, role, parent, population, expected, users_idx, extra=None):
        r = finish_and_write(b, x, out_id=out_id, role=role, parent=parent, lock=lock, code=code,
                             population=population, expected=expected, users_keys_v2=kv2[users_idx],
                             out_root=out_root, write=write, extra=extra)
        man[out_id] = r
        if write:
            written[out_id] = r["written"]
            log(f"wrote {out_id}: " + ", ".join(f"{t['table']}={t['row_count']}" for t in r["meta"]))
        return r

    # ---------------------------------------------------------------- pass 1: MAIN + CALIB + TRUE_COLD + panels
    tp0 = time.time()
    b.set_one(cohort=bands["MAIN"], calib=bands["CALIB"], cold=bands["TRUE_COLD"], m04=bands["COLD_H"],
              m04w=bands["WARM_TRUNCATION"])
    b.extract()
    if b.full:
        checks += b.recon_checks()
    tm_main, em_main, pos_main = build_pair(b, tid="M01_COHORT_TRAIN", eid="M02_VALIDATION_WARM",
                                            band_users=bands["MAIN"])
    tm_calib, em_calib, pos_calib = build_pair(b, tid="M05_CALIB_TRAIN", eid="M06_CALIB_VALIDATION_WARM",
                                               band_users=bands["CALIB"])
    em_cold = build_cold(b, cold_users=bands["TRUE_COLD"], pos_ref=pos_main)
    pm_coldh = b._panel_manifest("M04_COLD_H_PANEL", bands["COLD_H"], "COLD_START", None)
    pm_wt = b._panel_manifest("M04W_WARM_TRUNCATION_PANEL", bands["WARM_TRUNCATION"], "WARM_TRUNCATION", None)
    pass_timings.append({"pass": "MAIN+CALIB+TRUE_COLD+panels", "s": round(time.time() - tp0, 1)})

    def n_e2e(em: dict) -> int:
        """Users with >= 1 EVAL ROW (an observed target), the task packet's 'VALIDATION users' - a strict subset of
        the manifest's own n_users (every user with >= 1 VALIDATION-split decision, the M02 definition; matches the
        FS-01 precedent: 'FS_VALIDATION_5_100_32768 | 12,885 (10,269 with >= 1 eval row)')."""
        return int((em["tables"]["users"]["n_eval_rows"] >= 1).sum())

    n_cli_main = tm_main["tables"]["clients"]["row_count"]
    full = b.full
    checks += [
        DC.chk("MAIN_TRAIN.n_users", tm_main["n_users"], DC.EXPECTED_MAIN["n_users"], "DATA-06.md", hard=full),
        DC.chk("MAIN_TRAIN.n_train_examples", tm_main["n_decisions"], DC.EXPECTED_MAIN["n_train_examples"],
              "DATA-06.md", hard=full),
        DC.chk("MAIN_VALIDATION.n_validation_users", n_e2e(em_main), DC.EXPECTED_MAIN["n_validation_users"],
              "DATA-06.md", hard=full),
        DC.chk("MAIN_VALIDATION.n_eval_rows", em_main["n_eval_rows"], DC.EXPECTED_MAIN["n_eval_rows"], "DATA-06.md",
              hard=full),
        DC.chk("MAIN_VALIDATION.n_rankable_rows", em_main["n_rankable_rows"], DC.EXPECTED_MAIN["n_rankable_rows"],
              "DATA-06.md", hard=full),
        DC.chk("CALIB_TRAIN.n_users", tm_calib["n_users"], DC.EXPECTED_CALIB["n_users"], "DATA-06.md", hard=full),
        DC.chk("CALIB_TRAIN.n_train_examples", tm_calib["n_decisions"], DC.EXPECTED_CALIB["n_train_examples"],
              "DATA-06.md", hard=full),
        DC.chk("CALIB_VALIDATION.n_validation_users", n_e2e(em_calib), DC.EXPECTED_CALIB["n_validation_users"],
              "DATA-06.md", hard=full),
        DC.chk("CALIB_VALIDATION.n_eval_rows", em_calib["n_eval_rows"], DC.EXPECTED_CALIB["n_eval_rows"], "DATA-06.md",
              hard=full),
        DC.chk("CALIB_TRAIN.in_calib_prefix_all_true", int(tm_calib["tables"]["clients"]["in_calib_prefix"].sum()),
              tm_calib["n_users"], "structural: CALIB is the calib prefix itself"),
        DC.chk("MAIN_TRAIN.in_calib_prefix_count", int(tm_main["tables"]["clients"]["in_calib_prefix"].sum()),
              int(bands["CALIB"].size), "structural: CALIB nested in MAIN"),
        DC.chk("MAIN_TRAIN.known_deviation.calib_prefix_is_first_65536",
              tm_main["assertions"]["calib_prefix_is_first_65536"], False,
              "KNOWN, EXPECTED (v4data06.d6_common.KNOWN_ASSERTION_DEVIATIONS): CALIB is chosen by "
              "selection_key_v2 (nested in MAIN by v2 key); client_ord is the locked, unedited digest/"
              "selection_key_v1 order, so the calib subset is not a client_ord-contiguous prefix. This does not "
              "affect any table's correctness; consumers use CALIB_TRAIN_5_100_32768 directly.", hard=False),
        DC.chk("TRUE_COLD.n_users", em_cold["n_users"], DC.EXPECTED_TRUE_COLD["n_users"], "DATA-06.md", hard=full),
        DC.chk("TRUE_COLD.n_eval_rows", em_cold["n_eval_rows"], DC.EXPECTED_TRUE_COLD["n_eval_rows"], "DATA-06.md",
              hard=full),
        DC.chk("TRUE_COLD.A0.n_users", em_cold["counts_by_cause"]["stratum_COLD_NO_TRAIN_EVENTS"]["users_with_ge1_eval_row"],
              DC.EXPECTED_TRUE_COLD["n_users_A0"], "DATA-06.md", hard=full),
        DC.chk("TRUE_COLD.A0.n_eval_rows", em_cold["counts_by_cause"]["stratum_COLD_NO_TRAIN_EVENTS"]["eval_rows"],
              DC.EXPECTED_TRUE_COLD["n_eval_rows_A0"], "DATA-06.md", hard=full),
        DC.chk("TRUE_COLD.A1.n_users", em_cold["counts_by_cause"]["stratum_COLD_TRAIN_EVENTS_NO_LABEL"]["users_with_ge1_eval_row"],
              DC.EXPECTED_TRUE_COLD["n_users_A1"], "DATA-06.md", hard=full),
        DC.chk("TRUE_COLD.A1.n_eval_rows", em_cold["counts_by_cause"]["stratum_COLD_TRAIN_EVENTS_NO_LABEL"]["eval_rows"],
              DC.EXPECTED_TRUE_COLD["n_eval_rows_A1"], "DATA-06.md", hard=full),
        DC.chk("COLD_H.n_users", pm_coldh["n_users"], int(bands["COLD_H"].size), "structural: requested panel size"),
        DC.chk("WARM_TRUNCATION.n_users", pm_wt["n_users"], int(bands["WARM_TRUNCATION"].size),
              "structural: requested panel size"),
    ]
    hard_fail = [c for c in checks if c["hard"] and not c["pass"]]
    if hard_fail:
        raise BuildBlocked(f"DATA-06 pass-1 checks failed: {[c['name'] for c in hard_fail]}")

    emit(tm_main, DC.MAIN_TRAIN_ID, "MAIN", None,
        band_pop("MAIN", kv2[bands["MAIN"]], rank_range=list(DC.BANDS["MAIN"]), pool_size=bands["pool_size"]),
        [("n_users", tm_main["n_users"], DC.EXPECTED_MAIN["n_users"], "DATA-06.md"),
         ("n_decisions", tm_main["n_decisions"], DC.EXPECTED_MAIN["n_train_examples"], "DATA-06.md"),
         ("population_sha256_selection_key_v2", population_sha256(np.sort(kv2[bands["MAIN"]])), DC.MAIN_SHA_V2,
          "CENSUS-02")], bands["MAIN"])
    emit(em_main, DC.MAIN_EVAL_ID, "MAIN", DC.MAIN_TRAIN_ID,
        band_pop("MAIN", kv2[bands["MAIN"]], rank_range=list(DC.BANDS["MAIN"]), pool_size=bands["pool_size"]),
        [("n_users_with_ge1_eval_row", n_e2e(em_main), DC.EXPECTED_MAIN["n_validation_users"], "DATA-06.md"),
         ("n_eval_rows", em_main["n_eval_rows"], DC.EXPECTED_MAIN["n_eval_rows"], "DATA-06.md"),
         ("n_rankable_rows", em_main["n_rankable_rows"], DC.EXPECTED_MAIN["n_rankable_rows"], "DATA-06.md")],
        bands["MAIN"], extra={"n_users_with_ge1_eval_row": n_e2e(em_main)})
    emit(tm_calib, DC.CALIB_TRAIN_ID, "CALIBRATION", None,
        band_pop("CALIB", kv2[bands["CALIB"]], rank_range=list(DC.BANDS["CALIB"]), nested_in=DC.MAIN_TRAIN_ID),
        [("n_users", tm_calib["n_users"], DC.EXPECTED_CALIB["n_users"], "DATA-06.md"),
         ("n_decisions", tm_calib["n_decisions"], DC.EXPECTED_CALIB["n_train_examples"], "DATA-06.md"),
         ("population_sha256_selection_key_v2", population_sha256(np.sort(kv2[bands["CALIB"]])), DC.CALIB_SHA_V2,
          "CENSUS-02")], bands["CALIB"])
    emit(em_calib, DC.CALIB_EVAL_ID, "CALIBRATION", DC.CALIB_TRAIN_ID,
        band_pop("CALIB", kv2[bands["CALIB"]], rank_range=list(DC.BANDS["CALIB"]), nested_in=DC.MAIN_TRAIN_ID),
        [("n_users_with_ge1_eval_row", n_e2e(em_calib), DC.EXPECTED_CALIB["n_validation_users"], "DATA-06.md"),
         ("n_eval_rows", em_calib["n_eval_rows"], DC.EXPECTED_CALIB["n_eval_rows"], "DATA-06.md")], bands["CALIB"],
        extra={"n_users_with_ge1_eval_row": n_e2e(em_calib)})
    emit(em_cold, DC.TRUE_COLD_ID, "TRUE_COLD", None,
        band_pop("TRUE_COLD", kv2[bands["TRUE_COLD"]],
                selection_rule="every A0/A1 (n_eligible_TRAIN_examples == 0) user with >= 1 E2E VALIDATION row, "
                               "the M03 definition; not truncated",
                class_positive_reference=DC.MAIN_TRAIN_ID),
        [("n_users", em_cold["n_users"], DC.EXPECTED_TRUE_COLD["n_users"], "DATA-06.md"),
         ("n_eval_rows", em_cold["n_eval_rows"], DC.EXPECTED_TRUE_COLD["n_eval_rows"], "DATA-06.md")],
        bands["TRUE_COLD"])
    emit(pm_coldh, DC.COLD_H_ID, "COLD_H", DC.TRUE_COLD_ID,
        band_pop("COLD_H", kv2[bands["COLD_H"]],
                selection_rule="the 16,384 smallest selection_key_v2 among TRUE_COLD_VALIDATION; M04 candidate-A "
                               "semantics, rebuilt"),
        [("n_users", pm_coldh["n_users"], DC.N_PANEL, "DATA-06.md")], bands["COLD_H"])
    emit(pm_wt, DC.WARM_TRUNCATION_ID, "WARM_TRUNCATION", DC.MAIN_EVAL_ID,
        band_pop("WARM_TRUNCATION", kv2[bands["WARM_TRUNCATION"]],
                selection_rule="the 16,384 smallest selection_key_v2 among MAIN users with >= 1 E2E VALIDATION row; "
                               "M04W semantics, rebuilt"),
        [("n_users", pm_wt["n_users"], DC.N_PANEL, "DATA-06.md")], bands["WARM_TRUNCATION"])

    # ---------------------------------------------------------------- passes 2-5: the isolated COHORT-shaped bands
    isolated = ((bands["U_NEW_VAL"], DC.UNEW_VAL_TRAIN_ID, DC.UNEW_VAL_EVAL_ID, "UNSEEN_VAL",
                band_pop("U_NEW_VAL", None, rank_range=list(DC.BANDS["U_NEW_VAL"]))),
                (bands["U_NEW_TEST"], DC.UNEW_TEST_TRAIN_ID, DC.UNEW_TEST_EVAL_ID, "UNSEEN_HOLDOUT",
                band_pop("U_NEW_TEST", None, rank_range=list(DC.BANDS["U_NEW_TEST"]))),
                (bands["SPARSE"], DC.SPARSE_TRAIN_ID, DC.SPARSE_EVAL_ID, "SPARSE_PANEL",
                band_pop("SPARSE", None, selection_rule="16,384 smallest-key stratum-B (1-4) users with >= 1 E2E "
                                                        "VALIDATION row")),
                (bands["HEAVY"], DC.HEAVY_TRAIN_ID, DC.HEAVY_EVAL_ID, "HEAVY_PANEL",
                band_pop("HEAVY", None, selection_rule="16,384 smallest-key stratum-D (> 100) users with >= 1 E2E "
                                                       "VALIDATION row")))
    for band_users, tid_out, eid_out, role, pop_stub in isolated:
        tp0 = time.time()
        b.set_one(cohort=band_users)
        b.extract()
        tm, em, pos = build_pair(b, tid="M01_COHORT_TRAIN", eid="M02_VALIDATION_WARM", band_users=band_users)
        pass_timings.append({"pass": role, "s": round(time.time() - tp0, 1)})
        pop = dict(pop_stub)
        pop["n_users"] = int(band_users.size)
        pop["population_sha256_selection_key_v2"] = population_sha256(np.sort(kv2[band_users]))
        checks.append(DC.chk(f"{role}.n_users", tm["n_users"], int(band_users.size), "band construction"))
        emit(tm, tid_out, role, None, pop,
            [("n_users", tm["n_users"], int(band_users.size), "band construction")], band_users)
        emit(em, eid_out, role, tid_out, pop,
            [("n_users", em["n_users"], em["n_users"], "measured (no external target)")], band_users)

    hard_fail = [c for c in checks if c["hard"] and not c["pass"]]
    if hard_fail:
        raise BuildBlocked(f"DATA-06 build checks failed: {[c['name'] for c in hard_fail]}")

    if write:
        # file-name note: the internal bands dict key stays "U_NEW_TEST" (in-memory only); the written file name
        # uses "U_NEW_HOLDOUT" throughout, like the manifest ids, so no on-disk path ever contains "TEST".
        for name, file_tag in (("MAIN", "MAIN"), ("CALIB", "CALIB"), ("U_NEW_VAL", "U_NEW_VAL"),
                               ("U_NEW_TEST", "U_NEW_HOLDOUT"), ("SPARSE", "SPARSE"), ("HEAVY", "HEAVY"),
                               ("TRUE_COLD", "TRUE_COLD"), ("COLD_H", "COLD_H"),
                               ("WARM_TRUNCATION", "WARM_TRUNCATION")):
            FC.atomic_write_bytes(out_root, f"population/{file_tag}_SELECTION_KEY_V2.u64be",
                                  np.ascontiguousarray(np.sort(kv2[bands[name]]).astype(">u8")).tobytes())
        FC.atomic_write_bytes(out_root, "population/POOL_5_100_SELECTION_KEY_V2.u64be",
                              np.ascontiguousarray(bands["pool_keys_sorted"].astype(">u8")).tobytes())

    hashes = {out_id: FC.hashes_entry(x["header"], x["meta"], (x["written"] or {}).get("header_file_sha256"))
             for out_id, x in man.items()}
    rec = {"receipt": "DATA06_BUILD_RECEIPT", "task": DC.TASK, "status": "PASS", "full_build": b.full,
           "buckets": len(b.buckets), "python_utc_end": utc_now(), "isolation_modules_checked": len(ppsi_env.assert_isolated()),
           "lock": lock, "code": code, "checks": checks, "n_checks": len(checks),
           "n_failed": len([c for c in checks if not c["pass"]]), "manifests": hashes,
           "timings_s": {"prepare": round(t_prep - t0, 1), "passes": pass_timings, "total": round(time.time() - t0, 1),
                        **b.timings},
           "test_split_read": False, "no_raw_ids_written": True,
           "naming_decision": DC.NAMING_DECISION}
    if write:
        FC.write_json_self_hashed(out_root, "receipts/DATA06_MANIFEST_HASHES.json",
                                  {"receipt": "DATA06_MANIFEST_HASHES", "data_version": DC.DATA_VERSION,
                                   "builder_code_sha256": b.code_sha, "d6_code_sha256": code["d6_code_sha256"],
                                   "candidate_order_hash": b.candidate_order_hash, "manifests": hashes})
        FC.write_json_self_hashed(out_root, "receipts/DATA06_BUILD_RECEIPT.json", rec)
    rec["_man"] = man
    rec["_bands"] = bands
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DATA-06 v2 population manifests")
    ap.add_argument("--out", default=str(V4 / "data_v3"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--buckets", default="all", help="'all' or a comma list (partial smoke; disables hard checks)")
    a = ap.parse_args(argv)
    for out_id in DC.ALL_IDS:
        refuse_test_manifest(out_id)
    buckets = None if a.buckets == "all" else [int(x) for x in a.buckets.split(",")]
    b = D6Builder(ds_root=Path(a.dataset) if a.dataset else None, out_root=Path(a.out), buckets=buckets, log=log)
    if builder_code_sha256() != DC.DATA04_BUILDER_CODE_SHA256:
        log("WARNING: v4data builder_code_sha256 differs from DATA-04 (hard failure in a full build)")
    rec = run_build(b, Path(a.out))
    log(f"DATA-06 done: n_failed={rec['n_failed']} manifests={list(rec['manifests'])}")
    return 0 if rec["n_failed"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
