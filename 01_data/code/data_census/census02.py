"""census_v2.census02 (owner: Data Lead, CENSUS-02) - exact TRAIN-only census for the owner's 5-100 amendment,
candidate-N projections, FL local work, catalogue option B, the optional v1 precision prior, and the receipt.

Authority: company/owner/OWNER_AMENDMENT_POP_5_100_2026-09-26.md (binding) and company/tasks/CENSUS-02.md.

Rules enforced here:
  * eligibility and every stratum come from TRAIN only; VALIDATION rows are read only for coverage counts;
  * TEST is never read: every ledger scan goes through v4data.manifests.read_non_test (split IN TRAIN /
    VALIDATION_SELECT / VALIDATION_CONFIRM as a scan filter, so no TEST row is materialised); the optional prior reads
    only VALIDATION_CONFIRM rank files and refuses any path naming TEST;
  * the frozen v1 dataset is opened read-only; nothing is written outside --out;
  * no raw user_id leaves memory: users are held as the C1.2 salted digest (v4data.common.digests_of) and outputs are
    aggregate counts and one-way key hashes only.

Row definitions are the DATA-04 ones, executed by the DATA-04 functions themselves (ManifestBuilder._row_arrays and
_censor bound to a minimal shim), so elig / eval / rankable / OOV / censor classes are identical to M01/M02/M03.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from v4data import reference_counts as REF  # noqa: E402
from v4data.common import (K as K_LOCK, N_BUCKETS, TEST_END_US, TRAIN_END_US, VALIDATION_END_US, VERSION_HASH,  # noqa: E402
                           atomic_write_bytes, builder_code_sha256, canonical_json_bytes, digest_rows, digests_of,
                           json_bytes_strict, load_manifest_spec, population_sha256, selection_keys_of,
                           sha256_bytes, sha256_file)
from v4data.manifests import Enums, ManifestBuilder, read_non_test  # noqa: E402

from census_v2 import core  # noqa: E402
from census_v2.core import S_A0, S_A1, S_B, S_C, S_D, STRATA  # noqa: E402

TASK = "CENSUS-02"
EVID = "PROJECT_MEASURED"
DATA04_BUILDER_CODE_SHA256 = "443254e9ed95e41a48076f91643c83fbc77355dc80bb6b111d8432703d3e34a5"
COLS = ["user_id", "decision_id", "split", "decision_ts", "target_ts", "target_status", "next_split",
        "next_block_type", "has_observed_target", "loss_mask", "eval_mask", "target_class", "target_oov",
        "target_product_id", "product_id"]
OPS_PREVIEW = {"B": (1197003, 2503629), "C": (1446986, 28623386), "D": (51009, 9053593)}
G0_S5 = {"users_with_ge1_example": 2694998, "total_examples": 40180608, "A1_users_in_train_zero_eligible": 1509381}
CENSOR = ("CEN_NO_NEXT", "CEN_LEGACY_BARRIER", "CEN_RIGHT_EDGE", "CEN_AMBIGUOUS_NEXT_BARRIER")
STRATUM_DEF = {"A0": "TRUE COLD: no TRAIN decision at all (appears only in VALIDATION)",
               "A1": "TRUE COLD: has TRAIN decisions but n = 0", "A": "TRUE COLD: n = 0 (A0 + A1)",
               "B": "SPARSE / FEW-SHOT: 1 <= n <= 4", "C": "MAIN: 5 <= n <= 100 (inclusive)",
               "D": "HEAVY: n > 100", "TOTAL": "every user with >= 1 TRAIN or VALIDATION decision"}


class CensusBlocked(AssertionError):
    pass


class RowDefs:
    """Minimal shim so the DATA-04 row functions run unchanged: they only use self.E and self._censor."""
    _row_arrays = ManifestBuilder._row_arrays
    _censor = ManifestBuilder._censor

    def __init__(self, spec: dict):
        self.E = Enums(spec)


def date_u() -> str:
    """UTC timestamp from `date -u` only (never hand-typed)."""
    try:
        r = subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True, timeout=10)
        s = r.stdout.strip()
        return s if r.returncode == 0 and s else "UNAVAILABLE"
    except Exception:
        return "UNAVAILABLE"


def peak_rss_self_mib():
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)   # Linux: KiB
    except Exception:
        return None


def log(msg: str) -> None:
    print(f"[census02 {time.strftime('%H:%M:%S', time.gmtime())}Z] {msg}", flush=True)


def _merge_counts(acc: dict, vc) -> None:
    for pid, c in zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()):
        acc[pid] = acc.get(pid, 0) + int(c)


# ------------------------------------------------------------------------------------------------ per bucket
def process_bucket(defs: RowDefs, path: Path, offset: int) -> dict:
    """One bucket (a user's rows never cross a bucket, CEN/CENSUS_README 'User->bucket partitioning'), so every
    per-user count - and therefore the stratum - is complete inside the bucket."""
    E = defs.E
    t = read_non_test(path, COLS)
    if t["user_id"].null_count:
        raise CensusBlocked("NULL user_id in the ledger")
    r = defs._row_arrays(t)
    uniq = pc.unique(t["user_id"])
    ui = np.asarray(pc.index_in(t["user_id"], value_set=uniq).to_numpy(zero_copy_only=False), dtype=np.int64)
    nu = len(uniq)
    tr, vs, vc, val, ev = r["tr"], r["vs"], r["vc"], r["val"], r["eval"]
    tc, oov, elig, has_obs, st, tts = r["tc"], r["oov"], r["elig"], r["has_obs"], r["status"], r["tts"]

    def bc(m):
        return np.bincount(ui[m], minlength=nu).astype(np.int64)

    u = {"ntrain": bc(tr), "n": bc(elig), "nval": bc(val), "nval_vs": bc(vs), "nval_vc": bc(vc),
         "neval": bc(val & ev), "neval_vs": bc(vs & ev), "neval_vc": bc(vc & ev),
         "nrank": bc(val & ev & (tc >= 0)), "noov": bc(val & ev & oov)}
    strat_u = core.stratum_codes(u["n"], u["ntrain"])
    srow = strat_u[ui].astype(np.int64)
    cnt = {}

    def sc(name, m):
        cnt[name] = np.bincount(srow[m], minlength=5).astype(np.int64)

    sc("train_rows", tr)
    sc("train_supervised", elig)
    sc("train_label_time_excluded", tr & r["loss"] & (tc >= 0) & ((tts < 0) | (tts >= TRAIN_END_US)))
    for s in ("NO_OBSERVED_NEXT", "AMBIGUOUS_BOUNDARY", "BOUNDARY_UNRESOLVED", "OBSERVED_IN_PARTITION"):
        sc(f"train_status_{s}", tr & (st == E.c("target_status", s)))
        sc(f"val_status_{s}", val & (st == E.c("target_status", s)))
    sc("train_target_oov", tr & oov)
    sc("val_decisions", val)
    sc("val_decisions_VAL_A", vs)
    sc("val_decisions_VAL_B", vc)
    sc("val_eval_rows", val & ev)
    sc("val_eval_rows_VAL_A", vs & ev)
    sc("val_eval_rows_VAL_B", vc & ev)
    sc("val_rankable", val & ev & (tc >= 0))
    sc("val_oov", val & ev & oov)
    for c in CENSOR:
        sc(f"val_censor_{c}", val & (r["censor"] == E.c("censor_class", c)))
    sc("val_censor_NONE", val & (r["censor"] == E.c("censor_class", "NONE")))

    sup: dict = {}
    _merge_counts(sup, pc.value_counts(t["product_id"].filter(pa.array(tr))))
    supC: dict = {}
    trC = tr & (srow == S_C)
    if trC.any():
        _merge_counts(supC, pc.value_counts(t["product_id"].filter(pa.array(trC))))

    g = (ui + offset).astype(np.int64)
    ncm = has_obs & (tc < 0) & (tr | val)          # observed target outside the catalogue (OOV candidates)
    nci = np.flatnonzero(ncm)
    vce = np.flatnonzero(vc & ev)
    rec = {
        "TRAIN.decisions": int(tr.sum()), "VALIDATION_SELECT.decisions": int(vs.sum()),
        "VALIDATION_CONFIRM.decisions": int(vc.sum()),
        "TRAIN.loss_mask_true": int((tr & r["loss"]).sum()), "TRAIN.target_oov_true": int((tr & oov).sum()),
        "VALIDATION.target_oov_true": int((val & oov).sum()),
        "VALIDATION.target_class_ge0": int((val & ev & (tc >= 0)).sum()),
        "label_time.train_loss_target_ts_ge_train_end": int((tr & r["loss"] & (tts >= TRAIN_END_US)).sum()),
        "label_time.train_loss_target_ts_null": int((tr & r["loss"] & (tts < 0)).sum()),
        "label_time.val_eval_target_ts_ge_val_end": int((val & ev & (tts >= VALIDATION_END_US)).sum()),
        "label_time.decisions_ts_ge_2019_11_30_non_test": int((r["dts"] >= TEST_END_US).sum()),
        "check.m01_filter_vs_census_eligibility_mismatch": int(
            ((tr & has_obs & (tts >= 0) & (tts < TRAIN_END_US) & (tc >= 0)) != elig).sum()),
        "check.val_eval_not_rankable_or_oov": int((val & ev & ~((tc >= 0) | oov)).sum()),
        "check.oov_not_eval_on_val": int((val & oov & ~ev).sum()),
    }
    return {
        "nu": nu, "dig": digests_of(uniq.to_pylist()), "u": u, "strat": strat_u, "cnt": cnt, "sup": sup,
        "supC": supC, "rec": rec,
        "elig_g": g[elig].astype(np.int32), "elig_tc": tc[elig].astype(np.int32),
        "val_g": g[val & ev].astype(np.int32), "val_tc": tc[val & ev].astype(np.int32),
        "nc_g": g[nci], "nc_val": val[nci], "nc_eval": ev[nci], "nc_oov": oov[nci],
        "nc_pid": t["target_product_id"].take(pa.array(nci)).to_pylist() if nci.size else [],
        "vc_did": np.asarray(t["decision_id"].take(pa.array(vce)).to_numpy(zero_copy_only=False), dtype=np.int64),
        "vc_g": g[vce].astype(np.int32), "vc_tc": tc[vce].astype(np.int32),
    }


# ------------------------------------------------------------------------------------------------ main census
class Census:
    def __init__(self, ds: Path, out: Path, buckets, *, prior_root: Path | None, inventory_dir: Path):
        self.ds, self.out = Path(ds), Path(out)
        self.buckets = list(buckets)
        self.full = self.buckets == list(range(N_BUCKETS))
        self.prior_root = prior_root
        self.inventory_dir = inventory_dir
        self.t0 = time.time()
        self.utc = {"census_start_date_u": date_u()}
        self.checks: list = []
        ver = json.loads((self.ds / "VERSION.json").read_text(encoding="utf-8"))
        if ver.get("version_hash") != VERSION_HASH:
            raise CensusBlocked(f"dataset version_hash {ver.get('version_hash')} != {VERSION_HASH}")
        self.spec = load_manifest_spec()
        self.defs = RowDefs(self.spec)

    def check(self, name, got, expected, source, *, hard=True):
        ok = got == expected
        self.checks.append({"name": name, "expected": expected, "got": got, "pass": bool(ok), "source": source,
                            "hard": bool(hard)})
        return ok

    # -------------------------------------------------------------------------- pass
    def run_pass(self):
        dig_l, strat_l, uc = [], [], {k: [] for k in ("ntrain", "n", "nval", "nval_vs", "nval_vc", "neval",
                                                      "neval_vs", "neval_vc", "nrank", "noov")}
        self.cnt = {}
        self.sup, self.supC, self.rec = {}, {}, {}
        rows = {k: [] for k in ("elig_g", "elig_tc", "val_g", "val_tc", "nc_g", "nc_val", "nc_eval", "nc_oov",
                                "vc_did", "vc_g", "vc_tc")}
        nc_pid = []
        offset = 0
        for b in self.buckets:
            p = self.ds / "decisions" / f"user_bucket={b}" / "ledger.parquet"
            part = process_bucket(self.defs, p, offset)
            offset += part["nu"]
            dig_l.append(part["dig"])
            strat_l.append(part["strat"])
            for k in uc:
                uc[k].append(part["u"][k])
            for k, v in part["cnt"].items():
                self.cnt[k] = self.cnt.get(k, np.zeros(5, dtype=np.int64)) + v
            for k in rows:
                rows[k].append(part[k])
            nc_pid.extend(part["nc_pid"])
            for pid, c in part["sup"].items():
                self.sup[pid] = self.sup.get(pid, 0) + c
            for pid, c in part["supC"].items():
                self.supC[pid] = self.supC.get(pid, 0) + c
            for k, v in part["rec"].items():
                self.rec[k] = self.rec.get(k, 0) + v
            if b % 8 == 7 or b == self.buckets[-1]:
                log(f"bucket {b} done; users so far {offset}; elapsed {time.time() - self.t0:.0f}s; "
                    f"peak_rss_self {peak_rss_self_mib()} MiB")
        self.dig = np.concatenate(dig_l)
        self.strat = np.concatenate(strat_l).astype(np.int8)
        self.U = {k: np.concatenate(v) for k, v in uc.items()}
        self.R = {k: np.concatenate(v) for k, v in rows.items()}
        self.nc_pid = nc_pid
        self.nusers = int(self.dig.size)
        if np.unique(self.dig).size != self.dig.size:
            raise CensusBlocked("client digest collision or a user spanning two buckets (fatal, C1.2)")
        self.pass_s = round(time.time() - self.t0, 1)
        log(f"pass done: {self.nusers} users in {self.pass_s}s")

    # -------------------------------------------------------------------------- catalogue + OOV causes
    def catalogue(self):
        cat = pq.ParquetFile(self.ds / "mappings" / "catalogue.parquet").read(
            columns=["product_id", "output_class", "representative_views"])
        pid = cat["product_id"].to_pylist()
        oc = np.asarray(cat["output_class"].to_numpy(zero_copy_only=False), dtype=np.int64)
        rv = np.asarray(cat["representative_views"].to_numpy(zero_copy_only=False), dtype=np.int64)
        self.K = len(pid)
        if not np.array_equal(np.sort(oc), np.arange(self.K)):
            raise CensusBlocked("catalogue output_class is not 0..K-1")
        self.pid_of_class = [None] * self.K
        for p, c in zip(pid, oc.tolist()):
            self.pid_of_class[c] = p
        self.class_of_pid = dict(zip(pid, oc.tolist()))
        mism = sum(1 for p, v in zip(pid, rv.tolist()) if self.sup.get(p, 0) != v)
        ge5 = {p for p, n in self.sup.items() if n >= 5}
        self.cat_check = {"K": self.K, "support_mismatch": mism, "K1_products_with_train_support": len(self.sup),
                          "support_ge5_set_equals_catalogue": ge5 == set(pid)}
        if self.full:
            self.check("catalogue.n_rows", self.K, REF.value("catalogue.n_rows"), "v4data.reference_counts")
            self.check("catalogue.K_equals_lock", self.K, K_LOCK, "v4data.common.K")
            self.check("catalogue.support_mismatch", mism, 0, "support == catalogue.representative_views")
            self.check("catalogue.K1_products_with_train_support", len(self.sup),
                       REF.value("catalogue.K1_products_with_train_support"), "v4data.reference_counts")
            self.check("catalogue.support_ge5_set_equals_catalogue", ge5 == set(pid), True, "G0 CATALOGUE_SUPPORT")
        # OOV causes of the observed-but-uncatalogued targets (TRAIN and VALIDATION)
        sup_nc = np.fromiter((self.sup.get(p, 0) for p in self.nc_pid), dtype=np.int64, count=len(self.nc_pid))
        R = self.R
        cause = np.full(sup_nc.size, 0, dtype=np.int8)      # 0 other, 1 ABSENT_FROM_TRAIN, 2 BELOW_SUPPORT
        cause[R["nc_oov"] & (sup_nc == 0)] = 1
        cause[R["nc_oov"] & (sup_nc >= 1) & (sup_nc <= 4)] = 2
        bad = int((R["nc_oov"] & (sup_nc >= 5)).sum())
        other = int((cause == 0).sum())
        self.check("oov.target_with_support_ge5_marked_oov", bad, 0, "EVAL M02 oov_cause_consistent")
        self.check("oov.observed_uncatalogued_not_oov", other, 0, "M02 oov_cause: has_obs & tc<0 implies oov")
        isv = R["nc_val"]
        self.val_oov_absent_row_g = R["nc_g"][isv & R["nc_eval"] & (cause == 1)]
        self.val_oov_below_row_g = R["nc_g"][isv & R["nc_eval"] & (cause == 2)]
        self.U["noov_absent"] = np.bincount(self.val_oov_absent_row_g, minlength=self.nusers).astype(np.int64)
        self.U["noov_below"] = np.bincount(self.val_oov_below_row_g, minlength=self.nusers).astype(np.int64)
        tr_abs = R["nc_g"][~isv & (cause == 1)]
        tr_bel = R["nc_g"][~isv & (cause == 2)]
        st = self.strat.astype(np.int64)
        self.cnt["train_oov_absent_from_train"] = np.bincount(st[tr_abs], minlength=5).astype(np.int64)
        self.cnt["train_oov_below_support"] = np.bincount(st[tr_bel], minlength=5).astype(np.int64)
        self.val_oov_absent_s1_all = int((isv & (sup_nc == 0)).sum())      # DATA-04 definition (val & has_obs)
        if not np.array_equal(self.U["noov_absent"] + self.U["noov_below"], self.U["noov"]):
            raise CensusBlocked("VALIDATION OOV rows do not split exactly into ABSENT + BELOW")

    # -------------------------------------------------------------------------- class positives
    def class_positives(self):
        st_e = self.strat[self.R["elig_g"]]
        self.cls_pos = {s: np.bincount(self.R["elig_tc"][st_e == s], minlength=self.K).astype(np.int64)
                        for s in (S_B, S_C, S_D)}
        self.cls_pos_all = self.cls_pos[S_B] + self.cls_pos[S_C] + self.cls_pos[S_D]
        self.st_v = self.strat[self.R["val_g"]]

    # -------------------------------------------------------------------------- reproduction
    def reproduction(self):
        U, st = self.U, self.strat
        n = U["n"]
        s5_users = int((n >= 1).sum())
        s5_ex = int(n.sum())
        a1 = int(((U["ntrain"] > 0) & (n == 0)).sum())
        if not self.full:
            return
        self.check("G0.s5.users_with_ge1_example", s5_users, G0_S5["users_with_ge1_example"],
                   "g0/data_census/CLIENT_CENSUS.json s5.users_with_ge1_example")
        self.check("G0.s5.total_examples", s5_ex, G0_S5["total_examples"], "CLIENT_CENSUS.json s5.total_examples")
        self.check("G0.s5.A1_users_in_train_zero_eligible", a1, G0_S5["A1_users_in_train_zero_eligible"],
                   "CLIENT_CENSUS.json s5.users_in_train_with_zero_eligible_examples")
        for s, (eu, ee) in OPS_PREVIEW.items():
            code = STRATA.index(s)
            m = st == code
            self.check(f"OPS_PREVIEW.{s}.users", int(m.sum()), eu, "CENSUS-02 packet (from the G0 frequency table)")
            self.check(f"OPS_PREVIEW.{s}.examples", int(n[m].sum()), ee, "CENSUS-02 packet (from the G0 frequency table)")
        # full-population reconciliation vs the DATA-04 reference counts
        rec = dict(self.rec)
        rec["TRAIN.distinct_users"] = int((U["ntrain"] > 0).sum())
        rec["VALIDATION_SELECT.eval_mask_true"] = int(U["neval_vs"].sum())
        rec["VALIDATION_SELECT.distinct_users"] = int((U["nval_vs"] > 0).sum())
        rec["VALIDATION_CONFIRM.eval_mask_true"] = int(U["neval_vc"].sum())
        rec["VALIDATION_CONFIRM.distinct_users"] = int((U["nval_vc"] > 0).sum())
        rec["VALIDATION.decisions"] = rec["VALIDATION_SELECT.decisions"] + rec["VALIDATION_CONFIRM.decisions"]
        rec["VALIDATION.eval_mask_true"] = int(U["neval"].sum())
        rec["VALIDATION.distinct_users"] = int((U["nval"] > 0).sum())
        for s in ("OBSERVED_IN_PARTITION", "NO_OBSERVED_NEXT", "AMBIGUOUS_BOUNDARY", "BOUNDARY_UNRESOLVED"):
            rec[f"TRAIN.status.{s}"] = int(self.cnt[f"train_status_{s}"].sum())
            rec[f"VALIDATION.status.{s}"] = int(self.cnt[f"val_status_{s}"].sum())
        for c in CENSOR:
            rec[f"VALIDATION.censor.{c}"] = int(self.cnt[f"val_censor_{c}"].sum())
        rec["VALIDATION.oov_absent_from_train_s1_excluded"] = self.val_oov_absent_s1_all
        for k, (v, src) in REF.FULL_POPULATION.items():
            if k in rec:
                self.check(f"DATA04.{k}", rec[k], v, f"v4data.reference_counts ({src})")
        for k in ("check.m01_filter_vs_census_eligibility_mismatch", "check.val_eval_not_rankable_or_oov",
                  "check.oov_not_eval_on_val"):
            self.check(k, rec[k], 0, "DATA-04 integrity check (must be 0)")
        # population reproduction under the OLD (historical) G0/DATA-04 key - proves the M01/M02 definitions
        P = REF.POPULATION
        Uidx = np.flatnonzero(n >= 1)
        kold = selection_keys_of(self.dig[Uidx])
        o = np.argsort(kold, kind="stable")
        Us, ks = Uidx[o], kold[o]
        self.check("DATA04.universe_size", int(Uidx.size), P["universe_size"][0], P["universe_size"][1])
        self.check("DATA04.users_in_train_zero_eligible", a1, P["users_in_train_zero_eligible"][0], "REF")
        ne = U["neval"]
        self.check("DATA04.val_users_with_ge1_eval", int((ne >= 1).sum()), P["val_users_with_ge1_eval"][0], "REF")
        self.check("DATA04.val_users_warm", int(((ne >= 1) & (n >= 1)).sum()), P["val_users_warm"][0], "REF")
        self.check("DATA04.val_users_cold", int(((ne >= 1) & (n == 0)).sum()), P["val_users_cold"][0], "REF")
        for lab, k in (("16384", 16384), ("65536", 65536), ("262144", 262144), ("ALL", Us.size)):
            self.check(f"DATA04.cohort{lab}.population_sha256", population_sha256(ks[:k]),
                       P[f"cohort{lab}.population_sha256"][0], P[f"cohort{lab}.population_sha256"][1])
        for lab, k in (("262144", 262144), ("65536", 65536)):
            users = Us[:k]
            mem = np.zeros(self.nusers, dtype=bool)
            mem[users] = True
            pos = np.bincount(self.R["elig_tc"][mem[self.R["elig_g"]]], minlength=self.K)
            vm = mem[self.R["val_g"]]
            vt = self.R["val_tc"][vm]
            rk = np.unique(vt[vt >= 0])
            got = {"eligible_examples": int(n[users].sum()), "val_eval_rows": int(ne[users].sum()),
                   "users_ge1_eval": int((ne[users] >= 1).sum()), "rankable": int(U["nrank"][users].sum()),
                   "oov": int(U["noov"][users].sum()), "classes_ge1_positive": int((pos >= 1).sum()),
                   "classes_ge5_positive": int((pos >= 5).sum()),
                   "val_classes_zero_positive": int((pos[rk] == 0).sum())}
            for kk, vv in got.items():
                self.check(f"DATA04.cohort{lab}.{kk}", vv, P[f"cohort{lab}.{kk}"][0], P[f"cohort{lab}.{kk}"][1])
            if lab == "262144":
                self.check("DATA04.cohort262144.per_client_min", int(n[users].min()), P["cohort262144.per_client_min"][0], "REF")
                self.check("DATA04.cohort262144.per_client_p50", float(np.percentile(n[users], 50, method="linear")),
                           float(P["cohort262144.per_client_p50"][0]), "REF")
                self.check("DATA04.cohort262144.per_client_max", int(n[users].max()), P["cohort262144.per_client_max"][0], "REF")
        cold = (ne >= 1) & (n == 0)
        for name, m in (("COLD_NO_TRAIN_EVENTS", cold & (U["ntrain"] == 0)),
                        ("COLD_TRAIN_EVENTS_NO_LABEL", cold & (U["ntrain"] > 0))):
            self.check(f"DATA04.cold.{name}.users", int(m.sum()), P[f"cold.{name}.users"][0], P[f"cold.{name}.users"][1])
            self.check(f"DATA04.cold.{name}.eval_rows", int(ne[m].sum()), P[f"cold.{name}.eval_rows"][0], "REF")

    # -------------------------------------------------------------------------- strata
    def strata(self) -> dict:
        U, st, n = self.U, self.strat, self.U["n"]
        tot_users = self.nusers
        tot_ex = int(n.sum())
        tot_eval_users = int((U["neval"] >= 1).sum())
        tot_eval = int(U["neval"].sum())
        train_users = int((U["ntrain"] > 0).sum())
        groups = {"A0": [S_A0], "A1": [S_A1], "A": [S_A0, S_A1], "B": [S_B], "C": [S_C], "D": [S_D],
                  "TOTAL": [S_A0, S_A1, S_B, S_C, S_D]}
        out = {}
        for name, codes in groups.items():
            m = np.isin(st, codes)
            cs = np.array(codes)
            pos = np.zeros(self.K, dtype=np.int64)
            for c in codes:
                if c in self.cls_pos:
                    pos += self.cls_pos[c]
            vm = np.isin(self.st_v, cs)
            vt = self.R["val_tc"][vm]
            rkt = vt[vt >= 0]
            rk_cls = np.unique(rkt)
            cs_sum = lambda key: int(sum(int(self.cnt[key][c]) for c in codes))  # noqa: E731
            users = int(m.sum())
            ex = int(n[m].sum())
            blk = {
                "definition": STRATUM_DEF[name],
                "users": users, "supervised_examples": ex,
                "share_of_users": core.share(users, tot_users),
                "share_of_train_users": (core.share(int((m & (U["ntrain"] > 0)).sum()), train_users)
                                         if name != "A0" else 0.0),
                "share_of_supervised_examples": core.share(ex, tot_ex),
                "n_quantiles_linear": core.quantiles(n[m]),
                "n_frequency_table": (np.stack(np.unique(n[m], return_counts=True), axis=1).tolist()
                                      if name in ("B", "C", "D") else None),
                "users_with_ge1_train_decision": int((m & (U["ntrain"] > 0)).sum()),
                "train_decisions": cs_sum("train_rows"),
                "train_rows_by_cause": {
                    "supervised_examples": cs_sum("train_supervised"),
                    "oov_below_support": cs_sum("train_oov_below_support"),
                    "oov_absent_from_train": cs_sum("train_oov_absent_from_train"),
                    "status_NO_OBSERVED_NEXT": cs_sum("train_status_NO_OBSERVED_NEXT"),
                    "status_AMBIGUOUS_BOUNDARY": cs_sum("train_status_AMBIGUOUS_BOUNDARY"),
                    "status_BOUNDARY_UNRESOLVED_train_fence": cs_sum("train_status_BOUNDARY_UNRESOLVED"),
                    "label_time_excluded": cs_sum("train_label_time_excluded")},
                "validation": {
                    "users_with_ge1_validation_decision": int((m & (U["nval"] >= 1)).sum()),
                    "validation_decisions": cs_sum("val_decisions"),
                    "users_with_ge1_eval_row": int((m & (U["neval"] >= 1)).sum()),
                    "share_of_validation_eval_users": core.share(int((m & (U["neval"] >= 1)).sum()), tot_eval_users),
                    "eval_rows": cs_sum("val_eval_rows"),
                    "share_of_validation_eval_rows": core.share(cs_sum("val_eval_rows"), tot_eval),
                    "rankable_targets": cs_sum("val_rankable"),
                    "oov_targets": cs_sum("val_oov"),
                    "oov_by_cause": {"ABSENT_FROM_TRAIN": int(U["noov_absent"][m].sum()),
                                     "BELOW_SUPPORT": int(U["noov_below"][m].sum())},
                    "rankable_share_of_eval_rows": core.share(cs_sum("val_rankable"), cs_sum("val_eval_rows")),
                    "by_subperiod": {
                        "VAL_A_20191116_20191120": {"decisions": cs_sum("val_decisions_VAL_A"),
                                                    "eval_rows": cs_sum("val_eval_rows_VAL_A")},
                        "VAL_B_20191120_20191123": {"decisions": cs_sum("val_decisions_VAL_B"),
                                                    "eval_rows": cs_sum("val_eval_rows_VAL_B")}},
                    "censored_rows_by_class": {c: cs_sum(f"val_censor_{c}") for c in CENSOR},
                    "rankable_targets_whose_class_has_ge1_train_positive_in_stratum": int((pos[rkt] >= 1).sum()),
                    "share_rankable_targets_class_has_ge1_train_positive_in_stratum": core.share(
                        int((pos[rkt] >= 1).sum()), int(rkt.size)),
                    "distinct_rankable_target_classes": int(rk_cls.size),
                    "distinct_rankable_target_classes_with_zero_train_positive_in_stratum": int((pos[rk_cls] == 0).sum()),
                },
                "catalogue_classes_with_ge1_train_positive": int((pos >= 1).sum()),
                "catalogue_classes_with_ge5_train_positives": int((pos >= 5).sum()),
                "source": f"census_v2/census02.py Census.strata; {EVID}; TRAIN-only strata, VALIDATION coverage only",
            }
            v = blk["validation"]
            if v["eval_rows"] != v["rankable_targets"] + v["oov_targets"]:
                raise CensusBlocked(f"{name}: eval rows != rankable + OOV")
            tb = blk["train_rows_by_cause"]
            if blk["train_decisions"] != sum(tb.values()):
                raise CensusBlocked(f"{name}: TRAIN rows by cause do not sum to the TRAIN decisions")
            if ex != blk["train_rows_by_cause"]["supervised_examples"]:
                raise CensusBlocked(f"{name}: supervised examples mismatch")
            out[name] = blk
        # identities
        for key in ("users", "supervised_examples"):
            if out["A"][key] != out["A0"][key] + out["A1"][key]:
                raise CensusBlocked(f"A != A0 + A1 on {key}")
            if out["TOTAL"][key] != sum(out[s][key] for s in ("A0", "A1", "B", "C", "D")):
                raise CensusBlocked(f"TOTAL != sum of strata on {key}")
        self.shares_sum = {"users": sum(out[s]["share_of_users"] for s in ("A0", "A1", "B", "C", "D")),
                           "supervised_examples": sum(out[s]["share_of_supervised_examples"]
                                                      for s in ("A0", "A1", "B", "C", "D"))}
        self.check("strata.user_shares_sum_to_1", abs(self.shares_sum["users"] - 1.0) < 1e-12, True, "identity")
        self.check("strata.example_shares_sum_to_1", abs(self.shares_sum["supervised_examples"] - 1.0) < 1e-12,
                   True, "identity")
        return out

    # -------------------------------------------------------------------------- pool C
    def build_pool(self):
        pool = np.flatnonzero(self.strat == S_C)
        keys = core.selection_keys_v2(digest_rows(self.dig[pool]))
        n_unique = int(np.unique(keys).size)
        self.check("poolC.no_two_users_share_selection_key_v2", n_unique == keys.size, True, "CENSUS-02 deliverable 2")
        if n_unique != keys.size:
            raise CensusBlocked("two stratum-C users share a selection_key_v2: BLOCK and escalate")
        o = np.argsort(keys, kind="stable")
        self.pool_sorted = pool[o]
        self.keys_sorted = keys[o]
        self.P = int(pool.size)
        pos_of = np.full(self.nusers, -1, dtype=np.int64)
        pos_of[self.pool_sorted] = np.arange(self.P)
        rp = pos_of[self.R["elig_g"]]
        m = rp >= 0
        oe = np.argsort(rp[m], kind="stable")
        self.C_tc = self.R["elig_tc"][m][oe]
        rv = pos_of[self.R["val_g"]]
        mv = rv >= 0
        ov = np.argsort(rv[mv], kind="stable")
        self.Cv_tc = self.R["val_tc"][mv][ov]
        self.pos_of = pos_of
        n = self.U["n"][self.pool_sorted]
        self.cum_n = np.concatenate([[0], np.cumsum(n)])
        self.cum_e = np.concatenate([[0], np.cumsum(self.U["neval"][self.pool_sorted])])
        if self.cum_n[-1] != self.C_tc.size or self.cum_e[-1] != self.Cv_tc.size:
            raise CensusBlocked("pool row prefix arithmetic does not match the extracted rows")
        self.sizes = {}
        for N in core.CANDIDATE_N:
            if N <= self.P:
                self.sizes[str(N)] = N
        self.sizes["ALL_C"] = self.P
        calib = {}
        for lab, N in self.sizes.items():
            calib[lab] = [(cl, s) for cl, s in core.calib_sizes(N) if s >= 1]
        self.calib = calib
        all_sizes = sorted(set(self.sizes.values()) | {s for v in calib.values() for _, s in v})
        self.nested_ok = core.assert_nested(self.keys_sorted, all_sizes)
        self.check("poolC.prefixes_nested_and_keys_strictly_ascending", self.nested_ok, True, "construction")
        self.all_sizes = all_sizes

    def prefix_stats(self, k: int, inB: np.ndarray | None = None) -> dict:
        U = self.U
        users = self.pool_sorted[:k]
        n = U["n"][users]
        tc = self.C_tc[:self.cum_n[k]]
        pos = np.bincount(tc, minlength=self.K)
        vt = self.Cv_tc[:self.cum_e[k]]
        rk = vt[vt >= 0]
        rk_cls = np.unique(rk)
        ne = U["neval"][users]
        out = {
            "users": int(k), "supervised_examples": int(n.sum()), "n_quantiles_linear": core.quantiles(n),
            "validation": {
                "users_with_ge1_validation_decision": int((U["nval"][users] >= 1).sum()),
                "validation_decisions": int(U["nval"][users].sum()),
                "users_with_ge1_eval_row": int((ne >= 1).sum()),
                "share_of_cohort_users_with_ge1_eval_row": core.share(int((ne >= 1).sum()), k),
                "eval_rows": int(ne.sum()),
                "rankable_targets": int(U["nrank"][users].sum()),
                "oov_targets": int(U["noov"][users].sum()),
                "oov_by_cause": {"ABSENT_FROM_TRAIN": int(U["noov_absent"][users].sum()),
                                 "BELOW_SUPPORT": int(U["noov_below"][users].sum())},
                "users_with_ge1_eval_row_VAL_A": int((U["neval_vs"][users] >= 1).sum()),
                "users_with_ge1_eval_row_VAL_B": int((U["neval_vc"][users] >= 1).sum()),
                "eval_rows_per_eval_user_mean": core.share(int(ne.sum()), int((ne >= 1).sum())),
            },
            "catalogue_classes_with_ge1_train_positive_in_cohort": int((pos >= 1).sum()),
            "catalogue_classes_with_ge5_train_positives_in_cohort": int((pos >= 5).sum()),
            "catalogue_classes_with_zero_train_positive_in_cohort": int((pos == 0).sum()),
            "share_rankable_validation_targets_class_has_ge1_train_positive_in_cohort": core.share(
                int((pos[rk] >= 1).sum()), int(rk.size)),
            "distinct_rankable_validation_target_classes": int(rk_cls.size),
            "distinct_rankable_validation_target_classes_with_zero_cohort_train_positive": int((pos[rk_cls] == 0).sum()),
            "population_sha256": population_sha256(self.keys_sorted[:k]),
            "selection_key_v2_min_max_hex": [format(int(self.keys_sorted[0]), "016x"),
                                             format(int(self.keys_sorted[k - 1]), "016x")],
        }
        if out["validation"]["eval_rows"] != out["validation"]["rankable_targets"] + out["validation"]["oov_targets"]:
            raise CensusBlocked("cohort eval rows != rankable + OOV")
        if inB is not None:
            posB = pos * inB
            out["_B"] = {"classes_B_with_zero_train_positive_in_cohort": int(((posB == 0) & inB).sum()),
                         "classes_A_with_zero_train_positive_in_cohort": int((pos == 0).sum()),
                         "rankable_under_A": int(rk.size), "rankable_under_B": int(inB[rk].sum()),
                         "rankable_share_of_eval_rows_under_A": core.share(int(rk.size), int(vt.size)),
                         "rankable_share_of_eval_rows_under_B": core.share(int(inB[rk].sum()), int(vt.size)),
                         "supervised_examples_under_A": int(tc.size),
                         "supervised_examples_under_B": int(inB[tc].sum())}
        return out

    def candidate_projections(self) -> dict:
        cache = {s: self.prefix_stats(s) for s in self.all_sizes}
        self.prefix_cache = cache
        cohorts = {}
        for lab, N in self.sizes.items():
            blk = dict(cache[N])
            blk["requested_N"] = None if lab == "ALL_C" else N
            blk["calibration_prefix_candidates"] = {
                cl: {**cache[s], "fraction_of_N": s / N, "size_rule": cl,
                     "source": f"census_v2/census02.py Census.prefix_stats (first {s} keys of the same order); {EVID}"}
                for cl, s in self.calib[lab]}
            blk["source"] = f"census_v2/census02.py Census.prefix_stats; {EVID}"
            cohorts[lab] = blk
        return {
            "_definition": {
                "pool": "stratum C: users with 5 <= n_eligible_TRAIN_examples <= 100 (catalogue A, K = 158,486)",
                "selection_key_v2": "int.from_bytes(sha256(b'PPSI_V4_POP_5_100_SEL_20260926|' + digest32)"
                                    ".digest()[:8], 'big'); digest32 = sha256(b'PPSI_V4_POPULATION_20260926' + b'|' "
                                    "+ user_id) (CLIENT_CONTRACT C1.2, v4data.common.client_digest, SALT unchanged)",
                "cohort": "the N users of the pool with the smallest selection_key_v2 (one ascending order: every "
                          "cohort and every calibration prefix is a prefix of it, so all are nested)",
                "population_sha256": "v4data.common.population_sha256: sha256 over each selected user's 8-byte "
                                     "big-endian selection_key_v2, ascending",
                "calibration_prefixes": "N/16, N/8, N/4 (floor) and the fixed sizes 8192, 16384, 32768, 65536 "
                                        "where <= N/2; each is the first k users of the same order",
                "validation": "v4 VALIDATION = old VALIDATION_SELECT u VALIDATION_CONFIRM, decision_ts in "
                              "[2019-11-16, 2019-11-23); eval row = eval_mask (== target_status "
                              "OBSERVED_IN_PARTITION); rankable = eval & target_class >= 0; OOV = eval & target_oov "
                              "(cause by global TRAIN support: 0 ABSENT_FROM_TRAIN, 1-4 BELOW_SUPPORT); censor "
                              "classes as DATA-04 M02 (ManifestBuilder._censor, run unchanged)",
                "train_positives": "per class: count of the cohort's eligible TRAIN examples with that target_class",
                "quantile_method": "numpy.percentile(method='linear') as in G0",
                "evidence": EVID,
            },
            "pool_size": self.P,
            "selection_salt": core.SEL_PREFIX_V2.decode("ascii"),
            "pool_population_sha256": population_sha256(self.keys_sorted),
            "no_two_pool_users_share_a_key": True,
            "nested_prefix_check": {"checked_sizes": self.all_sizes, "all_nested": bool(self.nested_ok)},
            "cohorts": cohorts,
            "source": f"census_v2/census02.py Census.candidate_projections; {EVID}",
        }

    # -------------------------------------------------------------------------- FL local work
    def fl_local_work(self) -> dict:
        out = {}
        for lab, N in self.sizes.items():
            n = self.U["n"][self.pool_sorted[:N]]
            configs = {}
            for B in (8, 16, 32):
                for P in (1, 2):
                    configs[f"B{B}_passes{P}"] = core.fl_local_work(n, B, P, 64)
            lo = core.fl_local_work(n, 16, 6, 64)
            ex = int(n.sum())
            out[lab] = {
                "clients": int(N), "examples_per_EFE": ex,
                "primary_B16_passes2": configs["B16_passes2"],
                "comparison": {k: v for k, v in configs.items() if k != "B16_passes2"},
                "local_only_reference_B16_6passes": {
                    "per_client_steps_quantiles": lo["per_client_steps_per_visit_quantiles"],
                    "total_client_steps": lo["total_client_steps_per_sweep"],
                    "note": "LO recipe: 6 passes at local batch 16 (fl/gpu_receipt.py LO recipe, eval at 2/4/6); "
                            "one trajectory per client, no rounds"},
                "central_reference_B256": {
                    "updates_per_EFE_analytic": ex / 256.0,
                    "updates_to_6_EFE_analytic_ceil": int(-(-6 * ex // 256)),
                    "note": "analytic 6*examples/256; the planned v4train table rounds per update and may differ by a "
                            "few updates (TRAIN-01: 91,572 planned vs 91,569 analytic at N=3,906,929)"},
                "weight_concentration": core.weight_concentration(n),
                "source": f"census_v2/core.py fl_local_work over this run's measured per-client n; {EVID} "
                          "(analytic arithmetic only)",
            }
        return {"_definition": {
            "local_steps": "passes * ceil(n / B) per visit (each pass a fresh permutation, the last partial batch at "
                           "its own size: fl/client.py LocalSolver)",
            "EFE": "exposures / N_decisions with exposures = passes * n per visit (fl/participation.py); one full "
                   "sweep (every client once, groups of 64, SMALLER_FINAL_GROUP) = `passes` EFE",
            "rounds_per_EFE_at_64_clients_per_round": "ceil(N / 64) / passes under full-sweep participation",
            "evidence": EVID + " (analytic; no model, no timing)"}, "cohorts": out,
            "source": f"census_v2/census02.py Census.fl_local_work; {EVID}"}

    # -------------------------------------------------------------------------- catalogue option B
    def catalogue_option_b(self) -> dict:
        B_pids = [p for p, c in self.supC.items() if c >= 5]
        not_in_A = [p for p in B_pids if p not in self.class_of_pid]
        inB = np.zeros(self.K, dtype=bool)
        for p in B_pids:
            c = self.class_of_pid.get(p)
            if c is not None:
                inB[c] = True
        K_B = len(B_pids)
        A_and_B = int(inB.sum())
        self.inB = inB
        Ks_C = {str(s): int(sum(1 for c in self.supC.values() if c >= s)) for s in (1, 2, 5)}
        # zero-positive classes
        posC = self.cls_pos[S_C]
        zero = {"C_pool": {"classes_A_with_zero_train_positive": int((posC == 0).sum()),
                           "classes_B_with_zero_train_positive": int(((posC == 0) & inB).sum())}}
        cov = {}
        for lab, N in self.sizes.items():
            st = self.prefix_stats(N, inB)["_B"]
            zero[lab] = {"classes_A_with_zero_train_positive": st["classes_A_with_zero_train_positive_in_cohort"],
                         "classes_B_with_zero_train_positive": st["classes_B_with_zero_train_positive_in_cohort"]}
            cov[lab] = {k: st[k] for k in ("rankable_under_A", "rankable_under_B", "rankable_share_of_eval_rows_under_A",
                                           "rankable_share_of_eval_rows_under_B", "supervised_examples_under_A",
                                           "supervised_examples_under_B")}
            cov[lab]["eval_rows"] = self.prefix_cache[N]["validation"]["eval_rows"]
        # circularity: n under B for every user, strata transitions
        eg, et = self.R["elig_g"], self.R["elig_tc"]
        nB = np.bincount(eg[inB[et]], minlength=self.nusers).astype(np.int64)
        stB = core.stratum_codes(nB, self.U["ntrain"])
        trans = {}
        for a in range(5):
            ma = self.strat == a
            trans[STRATA[a]] = {STRATA[b]: int((ma & (stB == b)).sum()) for b in range(5)}
        C_A = self.strat == S_C
        C_B = stB == S_C
        circ = {
            "transition_matrix_users_rows_A_to_columns_B": trans,
            "C_users_leaving_C_under_B": int((C_A & ~C_B).sum()),
            "C_users_to_B_sparse": int((C_A & (stB == S_B)).sum()),
            "C_users_to_A1_cold": int((C_A & (stB == S_A1)).sum()),
            "D_users_entering_C_under_B": int(((self.strat == S_D) & C_B).sum()),
            "C_users_whose_n_changes_under_B": int((C_A & (nB != self.U["n"])).sum()),
            "C_pool_under_A": {"users": int(C_A.sum()), "examples": int(self.U["n"][C_A].sum())},
            "C_pool_under_B": {"users": int(C_B.sum()), "examples_under_B": int(nB[C_B].sum())},
            "B_catalogue_is_subset_of_A": len(not_in_A) == 0,
            "source": f"census_v2/census02.py Census.catalogue_option_b (n recounted with targets in B); {EVID}",
            "note": "B counts support only over the stratum-C universe defined under A; under B, n (targets in B only) "
                    "can only fall (B is a subset of A), so users move C->B/A1 and D->C. One step only; no fixed "
                    "point is iterated",
        }
        head = {}
        for fam, fn in (("GRU", "GRU_CLEAN_V2_MANIFEST_INVENTORY.json"), ("SASREC", "SASREC_CLEAN_V2_MANIFEST_INVENTORY.json")):
            p = self.inventory_dir / fn
            inv = json.loads(p.read_text(encoding="utf-8"))
            self.inv_hashes[fn] = sha256_file(p)
            own = core.head_sizes(int(inv["K"]), inv)
            a = core.head_sizes(self.K, inv)
            bb = core.head_sizes(K_B, inv)
            head[fam] = {"A": a, "B": bb, "delta_B_minus_A": {
                "W_out_rows": bb["W_out"]["rows"] - a["W_out"]["rows"],
                "head_params": bb["head_params_W_out_plus_bias"]["params"] - a["head_params_W_out_plus_bias"]["params"],
                "head_MB_fp32": bb["head_params_W_out_plus_bias"]["MB_fp32"] - a["head_params_W_out_plus_bias"]["MB_fp32"],
                "total_params_head_only": bb["total_shared_params_head_only_change"]["params"]
                - a["total_shared_params_head_only_change"]["params"],
                "total_params_head_and_input": bb["total_shared_params_head_and_input_vocab_change"]["params"]
                - a["total_shared_params_head_and_input_vocab_change"]["params"]},
                "inventory_total_check": own["total_shared_params_head_and_input_vocab_change"]["params"]
                == int(inv["parameter_inventory"]["total"]) and int(inv["K"]) == K_LOCK,
                "inventory": f"code/v4models/inventory/{fn}",
                "source": f"census_v2/core.py head_sizes over the MODEL-01 inventory {fn}; {EVID} (arithmetic)"}
            if not head[fam]["inventory_total_check"]:
                raise CensusBlocked(f"{fam}: the head arithmetic does not reproduce the inventory total at K")
        rkC_A = cov["ALL_C"]["rankable_under_A"]
        zero["source"] = f"census_v2/census02.py Census.catalogue_option_b / prefix_stats; {EVID}"
        cov["source"] = f"census_v2/census02.py Census.prefix_stats(inB); {EVID}"
        return {
            "_definition": {
                "A": "global TRAIN support >= 5 (K = 158,486), the current catalogue",
                "B": "same rule, support counted only over TRAIN rows (representative views, the context product_id "
                     "column) of stratum-C users (the exact G0 pass1_support / DATA-04 pass_support definition, "
                     "restricted to C)",
                "positives": "eligible TRAIN examples (targets); a class can have C-support >= 5 and still 0 "
                             "positives",
                "evidence": EVID},
            "K_A": self.K, "K_B": K_B, "A_and_B": A_and_B, "A_minus_B": self.K - A_and_B, "B_minus_A": len(not_in_A),
            "K_s_over_C_universe": Ks_C,
            "zero_positive_classes": zero,
            "validation_target_coverage": cov,
            "headline": {"C_pool_rankable_share_under_A": cov["ALL_C"]["rankable_share_of_eval_rows_under_A"],
                         "C_pool_rankable_share_under_B": cov["ALL_C"]["rankable_share_of_eval_rows_under_B"],
                         "C_pool_rankable_targets_lost_under_B": rkC_A - cov["ALL_C"]["rankable_under_B"]},
            "circularity": circ,
            "model_head": head,
            "source": f"census_v2/census02.py Census.catalogue_option_b; {EVID}",
        }

    # -------------------------------------------------------------------------- precision prior
    def precision_prior(self, cand: dict) -> dict:
        base = {"label": "PRIOR_EVIDENCE_FROM_V1_MODELS", "used_for_selection": False,
                "analytic_bound": {"per_user_rr_sd_upper_bound": 0.5,
                                   "se_upper_bound_by_N": {
                                       lab: (0.5 / np.sqrt(c["validation"]["users_with_ge1_eval_row"])
                                             if c["validation"]["users_with_ge1_eval_row"] else None)
                                       for lab, c in cand["cohorts"].items()},
                                   "definition": "per-user RR@20 lies in [0, 1], so its sd is <= 0.5; SE of the "
                                                 "client-macro mean <= 0.5 / sqrt(users with >= 1 eval row)"},
                "source": f"census_v2/census02.py Census.precision_prior"}
        if self.prior_root is None:
            base["status"] = "NOT_COMPUTED_NO_PRIOR_ROOT (analytic bound only)"
            return base
        R = self.prior_root
        files = {
            "GRU_F0_locked_v1": {"ranks": R / "gru_v1" / "final_eval" / "vc_ranks_F0_item_only_h256.npy",
                                 "headline": R / "gru_v1" / "final_eval" / "VC_RESULTS.json", "key": "F0_item_only_h256"},
            "SASREC_A1_RICH256_s2027_locked_v1": {
                "ranks": R / "sasrec_v1" / "retro_eval" / "VALIDATION_CONFIRM" / "A1_RICH256_s2027.ranks.npz",
                "headline": R / "sasrec_v1" / "retro_eval" / "VALIDATION_CONFIRM" / "A1_RICH256_s2027.json",
                "key": None},
        }
        for f in files.values():
            for k in ("ranks", "headline"):
                s = str(f[k])
                if "TEST" in s.upper().replace("VALIDATION_CONFIRM", "") or not f[k].is_file():
                    base["status"] = f"NOT_COMPUTED: refused or missing {s} (analytic bound only)"
                    return base
        # the v1 VALIDATION_CONFIRM eval dataset: every VC eval_mask row, ascending decision_id (CompactStore
        # decision_set(kind='eval'))
        o = np.argsort(self.R["vc_did"], kind="stable")
        did = self.R["vc_did"][o]
        g = self.R["vc_g"][o].astype(np.int64)
        tc = self.R["vc_tc"][o]
        if did.size > 1 and not np.all(did[1:] > did[:-1]):
            raise CensusBlocked("VC decision ids are not unique")
        ds_sha = sha256_bytes(np.ascontiguousarray(did, dtype="<i8").tobytes())
        per_model = {}
        rr_user = {}
        cnt_u = np.bincount(g, minlength=self.nusers)
        has = cnt_u > 0
        for name, f in files.items():
            self.prior_hashes[str(f["ranks"])] = sha256_file(f["ranks"])
            self.prior_hashes[str(f["headline"])] = sha256_file(f["headline"])
            r = np.load(f["ranks"])
            ranks = np.asarray(r["ranks"] if hasattr(r, "files") else r, dtype=np.int64)
            hd = json.loads(Path(f["headline"]).read_text(encoding="utf-8"))
            h = hd[f["key"]]["end_to_end"] if f["key"] else hd["end_to_end"]
            arch_users = int(h["macro"]["user_count"]) if "user_count" in h["macro"] else int(h["users"])
            align = {"n_rows_equal": int(ranks.size) == int(did.size),
                     "unranked_rows_are_exactly_the_oov_rows": bool(ranks.size == did.size
                                                                    and np.array_equal(ranks == -1, tc < 0)),
                     "ranks_in_1_to_K": bool(ranks.size == did.size and np.all((ranks[tc >= 0] >= 1)
                                                                               & (ranks[tc >= 0] <= self.K)))}
            if not all(align.values()):
                per_model[name] = {"status": "ALIGNMENT_FAILED", "alignment": align}
                continue
            rr = np.where((ranks >= 1) & (ranks <= 20), 1.0 / np.maximum(ranks, 1), 0.0)
            su = np.bincount(g, weights=rr, minlength=self.nusers)
            pu = np.where(has, su / np.maximum(cnt_u, 1), np.nan)
            macro_all = float(np.nanmean(pu[has]))
            micro_all = float(rr.mean())
            align["archived_n"] = int(h["n"]) == int(did.size)
            align["archived_user_count"] = arch_users == int(has.sum())
            align["archived_macro_mrr20_abs_diff"] = abs(macro_all - float(h["macro"]["mrr@20"]))
            align["archived_micro_mrr20_abs_diff"] = abs(micro_all - float(h["micro"]["mrr@20"]))
            ok = (align["archived_n"] and align["archived_user_count"] and align["archived_macro_mrr20_abs_diff"] < 1e-5
                  and align["archived_micro_mrr20_abs_diff"] < 1e-5)
            if not ok:
                per_model[name] = {"status": "ALIGNMENT_FAILED", "alignment": align}
                continue
            rr_user[name] = pu
            per_model[name] = {"status": "ALIGNED", "alignment": align,
                               "files": {"ranks": str(f["ranks"]), "headline": str(f["headline"])}}
        base["vc_eval_dataset"] = {"rows": int(did.size), "users": int(has.sum()), "decision_id_sha256": ds_sha,
                                   "definition": "VALIDATION_CONFIRM eval_mask rows, ascending decision_id (the v1 "
                                                 "CompactStore eval DecisionSet order)"}
        base["models"] = per_model
        if not rr_user:
            base["status"] = "NOT_COMPUTED: no rank file aligned (analytic bound only)"
            return base
        inC = (self.strat == S_C) & has

        def sd_block(vals):
            v = vals[~np.isnan(vals)]
            return {"users": int(v.size), "sd_per_user_rr20": float(v.std(ddof=1))}      # means deliberately withheld

        for name, pu in rr_user.items():
            blk = {"all_vc_users": sd_block(pu[has]), "stratum_C_vc_users": sd_block(pu[inC])}
            by_n = {}
            for lab, N in self.sizes.items():
                users = self.pool_sorted[:N]
                uu = users[has[users]]
                sdN = sd_block(pu[uu]) if uu.size > 1 else None
                sdC = blk["stratum_C_vc_users"]["sd_per_user_rr20"]
                nval = cand["cohorts"][lab]["validation"]["users_with_ge1_eval_row"]
                by_n[lab] = {"cohort_vc_users": int(uu.size), "cohort_sd_per_user_rr20": sdN,
                             "v4_validation_users_with_ge1_eval_row": nval,
                             "implied_se_client_macro_mrr20_v4_users": sdC / np.sqrt(nval),
                             "implied_ci95_halfwidth_v4_users": 1.96 * sdC / np.sqrt(nval),
                             "implied_se_client_macro_mrr20_vc_only_users": sdC / np.sqrt(max(uu.size, 1))}
            blk["by_candidate_N"] = by_n
            per_model[name]["prior"] = blk
        if len(rr_user) == 2:
            a, b = (rr_user[k] for k in files)
            d = a[inC] - b[inC]
            sd_d = float(d.std(ddof=1))
            base["paired_difference_GRU_minus_SASREC_stratum_C"] = {
                "users": int(d.size), "sd_per_user_difference": sd_d,
                "per_user_rr_correlation": float(np.corrcoef(a[inC], b[inC])[0, 1]),
                "implied_se_of_macro_difference_by_N": {
                    lab: sd_d / np.sqrt(cand["cohorts"][lab]["validation"]["users_with_ge1_eval_row"])
                    for lab in self.sizes}}
        base["status"] = "COMPUTED"
        base["withheld"] = ("mean RR levels of the v1 models restricted to stratum C / to a candidate cohort, and the "
                            "mean GRU-SASREC difference, are deliberately NOT reported: only dispersion is needed for "
                            "the SE, and owner section 4 freezes N before model outcomes are observed")
        base["caveats"] = ["v1 models, v1 features, v1 VALIDATION_CONFIRM window (3 days) only; the v4 window is 7 days "
                           "(more rows per user, so the per-user sd there is expected to be smaller: INFERENCE)",
                           "restricted to stratum-C users by this run's TRAIN-only n; not used for any selection",
                           "no TEST file is opened (paths naming TEST are refused)"]
        return base

    # -------------------------------------------------------------------------- inputs
    def input_hashes(self) -> dict:
        fj = json.loads((self.ds / "FILES.json").read_text(encoding="utf-8"))
        ent = {e["path"]: e for e in fj["files"]}
        paths = [f"decisions/user_bucket={b}/ledger.parquet" for b in self.buckets] + ["mappings/catalogue.parquet"]
        rows = []
        for rel in paths:
            e = ent.get(rel)
            got = sha256_file(self.ds / rel)
            rows.append({"path": rel, "files_json_sha256": e["sha256"] if e else None, "measured_sha256": got,
                         "match": bool(e and e["sha256"] == got)})
        other = {}
        for rel in ("VERSION.json", "FILES.json", "splits/splits.json"):
            p = self.ds / rel
            if p.is_file():
                other[rel] = {"measured_sha256": sha256_file(p),
                              "files_json_sha256": ent[rel]["sha256"] if rel in ent else None}
        return {"dataset_root": str(self.ds), "version_hash": VERSION_HASH, "files_read": rows,
                "all_files_json_entries_match": all(r["match"] for r in rows), "other": other}

    # -------------------------------------------------------------------------- run
    def run(self, tests: dict) -> int:
        self.inv_hashes, self.prior_hashes = {}, {}
        self.run_pass()
        self.catalogue()
        self.class_positives()
        self.reproduction()
        strata = self.strata()
        self.build_pool()
        cand = self.candidate_projections()
        fl = self.fl_local_work()
        catb = self.catalogue_option_b()
        prior = self.precision_prior(cand)
        inputs = self.input_hashes()
        if self.full:
            self.check("inputs.all_files_json_entries_match", inputs["all_files_json_entries_match"], True, "FILES.json")
            self.check("code.v4data_builder_code_sha256_is_DATA04", builder_code_sha256(), DATA04_BUILDER_CODE_SHA256,
                       "DATA-04 handoff builder_code_sha256")
        self.utc["census_end_date_u"] = date_u()
        strata_out = {"_definition": {
            "n": "n_eligible_TRAIN_examples: split = TRAIN, loss_mask, target observed with target_class >= 0 under "
                 "catalogue A (global TRAIN support >= 5, K = 158,486), target_ts < 2019-11-16T00:00Z (DATA-04 "
                 "_row_arrays 'elig', identical to the G0 s5 definition)",
            "strata": STRATUM_DEF,
            "shares": "share_of_users over TOTAL (every user with a TRAIN or VALIDATION decision; users seen only in "
                      "TEST are not visible because TEST is never read); share_of_train_users over users with >= 1 "
                      "TRAIN decision; share_of_supervised_examples over all eligible examples",
            "validation": "v4 VALIDATION = VALIDATION_SELECT u VALIDATION_CONFIRM, [2019-11-16, 2019-11-23), coverage "
                          "only (never used for selection)",
            "quantile_method": "numpy.percentile(method='linear') over the stratum's per-user n (G0 convention)",
            "evidence": EVID},
            "strata": strata, "shares_sum": self.shares_sum,
            "source": f"census_v2/census02.py Census.strata; {EVID}"}
        outputs = {"STRATA_CENSUS.json": strata_out, "CANDIDATE_N_PROJECTIONS.json": cand,
                   "FL_LOCAL_WORK.json": fl, "CATALOGUE_OPTION_B.json": catb, "PRECISION_PRIOR.json": prior}
        hard_fail = [c for c in self.checks if c["hard"] and not c["pass"]]
        self.out.mkdir(parents=True, exist_ok=True)
        written = {}
        if not hard_fail:
            for name, obj in outputs.items():
                b = json_bytes_strict(_jsonable(obj))
                atomic_write_bytes(self.out, name, b)
                written[name] = sha256_bytes(b)
        receipt = {
            "task": TASK, "evidence": EVID, "status": "PASS" if not hard_fail else "BLOCKED_REPRODUCTION_FAILED",
            "utc": dict(self.utc), "full_population": self.full, "buckets": len(self.buckets),
            "host": {"platform": platform.platform(), "python": platform.python_version(), "numpy": np.__version__,
                     "pyarrow": pa.__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                     "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
            "inputs": inputs, "other_inputs_sha256": {**self.inv_hashes, **self.prior_hashes},
            "code": code_hashes(),
            "resources": {"peak_rss_self_mib": peak_rss_self_mib(), "wall_s": round(time.time() - self.t0, 1),
                          "pass_s": self.pass_s},
            "rows_materialised_by_split": {"TRAIN": self.rec.get("TRAIN.decisions"),
                                           "VALIDATION_SELECT": self.rec.get("VALIDATION_SELECT.decisions"),
                                           "VALIDATION_CONFIRM": self.rec.get("VALIDATION_CONFIRM.decisions"),
                                           "TEST": 0},
            "test_split_read": False,
            "reproduction_checks": self.checks,
            "reproduction_all_pass": not hard_fail,
            "n_checks": len(self.checks), "n_failed": len(hard_fail),
            "tests": tests,
            "outputs_sha256": written,
            "source": f"census_v2/census02.py; {EVID}",
        }
        receipt["receipt_sha256"] = sha256_bytes(canonical_json_bytes(_jsonable(receipt)))
        atomic_write_bytes(self.out, "CENSUS02_RECEIPT.json", json_bytes_strict(_jsonable(receipt)))
        log(f"checks {len(self.checks)}, failed {len(hard_fail)}; outputs {list(written)}")
        for c in hard_fail:
            log(f"FAILED CHECK {c['name']}: expected {c['expected']!r} got {c['got']!r}")
        return 0 if not hard_fail else 3


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    return o


def code_hashes() -> dict:
    files = sorted(p for p in HERE.glob("*") if p.is_file() and p.suffix in (".py", ".sh"))
    per = {p.name: sha256_file(p) for p in files}
    h = hashlib.sha256()
    for name in sorted(per):
        h.update(name.encode() + b"\0" + per[name].encode() + b"\n")
    imported = {}
    for rel in ("v4data/common.py", "v4data/manifests.py", "v4data/reference_counts.py"):
        imported[rel] = sha256_file(CODE / rel)
    return {"census_v2_files_sha256": per, "census_v2_tree_sha256": h.hexdigest(),
            "imported_v4data_files_sha256": imported, "v4data_builder_code_sha256": builder_code_sha256()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--buckets", default="all", help="'all' or a comma list (debug only; disables full checks)")
    ap.add_argument("--prior-root", default=None, help=".local_ai root holding gru_v1/ and sasrec_v1/ (read-only)")
    ap.add_argument("--inventory-dir", default=str(CODE / "v4models" / "inventory"))
    ap.add_argument("--tests-receipt", default=None)
    a = ap.parse_args(argv)
    buckets = list(range(N_BUCKETS)) if a.buckets == "all" else [int(x) for x in a.buckets.split(",")]
    tests = {"tests_passed": None, "note": "no test receipt given"}
    if a.tests_receipt and Path(a.tests_receipt).is_file():
        tests = json.loads(Path(a.tests_receipt).read_text(encoding="utf-8"))
    c = Census(Path(a.dataset), Path(a.out), buckets, prior_root=Path(a.prior_root) if a.prior_root else None,
               inventory_dir=Path(a.inventory_dir))
    return c.run(tests)


if __name__ == "__main__":
    sys.exit(main())
