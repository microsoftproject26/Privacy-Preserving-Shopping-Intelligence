"""v4data.manifests (owner: Data Lead, DATA-04) - the pre-TEST manifests of MANIFEST_SPEC_V4 1.2.0.

Builds M01, M05, M02, M06, M03, M04, M04W, M04S and the bootstrap W registrations for the required user sets.
TEST manifests (M07, M08, M09) are REFUSED: this builder has no TEST code path, and every ledger/feature read
applies a scan filter split IN (TRAIN, VALIDATION_SELECT, VALIDATION_CONFIRM) so no RETRO_TEST row is ever
materialised (MANIFEST_SPEC global_rules.test_gate; EVALUATION_CONTRACT_V4 E10.1).

Passes over the frozen v1 ledger (decisions/user_bucket=*/ledger.parquet), one bucket at a time (a user's rows never
cross a bucket, CEN/CENSUS_README.md "User->bucket partitioning"):
  P0  TRAIN support per product_id (the census support definition; verified against catalogue.representative_views)
  PA  per-user aggregates + full-population reconciliation counters (all users)
  PB  row extraction for the selected populations + target metadata states (model_ready join) + panel events
Raw user_id / product_id strings exist only transiently in memory. Every written column is an integer handle, the
decision_id surrogate, an enum, or the salted pseudonym client_key_hex (global_rules.no_raw_ids, checked at write).
"""
from __future__ import annotations

import io
import json
import platform
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from . import reference_counts as REF
from .common import (DATASET_CODE_HASH, GRACE_END_US, K, N_BUCKETS, NON_TEST_SPLITS, SALT_STR, SCHEMA_SELF_SHA256,
                     SCHEMA_VERSION, TEST_END_US, TRAIN_END_US, TRAIN_START_US, VALIDATION_END_US, VERSION_HASH,
                     VS_END_US, EnumValues, TestGateError, atomic_write_bytes, atomic_write_json,
                     builder_code_sha256, candidate_order_hash, canonical_json_bytes, data_v2_root, dataset_root,
                     digest_rows, digests_of, hex_of, load_manifest_spec, load_schema, ordered_ids_sha256,
                     population_sha256, selection_keys_of, sha256_bytes, sha256_file, table_content_sha256,
                     user_set_sha256, utc_now, write_marker)

PRE_LOCK_IDS = ("M01_COHORT_TRAIN", "M05_CALIB_TRAIN", "M02_VALIDATION_WARM", "M06_CALIB_VALIDATION_WARM",
                "M03_COLD_PANEL_VAL", "M04_COLD_H_PANEL", "M04W_WARM_TRUNCATION_PANEL",
                "M04S_FEW_SHOT_SELECTION_PANEL")
TEST_MANIFEST_IDS = ("M07_TEST_WARM", "M08_TEST_COLD_PANEL", "M09_TEST_COLD_H_PANEL")
N_COHORT, N_CALIB, N_PANEL = 262_144, 65_536, 16_384
H_ALL = (0, 1, 5, 10, 20, 50)
H_FIXED = {"FIXED_50": (0, 1, 5, 10, 20, 50), "FIXED_20": (0, 1, 5, 10, 20)}
FIXED_MIN = {"FIXED_50": 50, "FIXED_20": 20}
BOOT_SEED, BOOT_R = 20260926, 1000
HEX_RE = re.compile(r"^[0-9a-f]{64}$")

COHORT, CALIB, COLD, P_M04, P_M04W, P_M04S = 1, 2, 4, 8, 16, 32

LEDGER_A = ["user_id", "split", "decision_ts", "target_ts", "target_status", "next_split", "next_block_type",
            "has_observed_target", "loss_mask", "eval_mask", "target_class", "target_oov"]
LEDGER_B = ["user_id", "decision_id", "decision_ts", "split", "target_ts", "target_status", "next_split",
            "next_block_type", "has_observed_target", "loss_mask", "eval_mask", "target_class", "target_oov",
            "target_product_id", "segment_index", "position_in_segment"]
FEATURES_B = ["user_id", "split", "segment_index", "position_in_segment", "product_id", "block_ts",
              "category_idx", "brand_idx", "main_category_idx"]

# field -> (dtype, enum name or None); the MANIFEST_SPEC field_dictionary plus two table-internal fields
FIELD_TYPES = {
    "decision_id": ("int64", None), "client_key_hex": ("string", "HEX"), "selection_key": ("uint64", None),
    "client_ord": ("int64", None), "eval_user_key": ("int64", None), "decision_ts_us": ("int64", None),
    "target_ts_us": ("int64", None), "old_split": ("string", "old_split"), "v4_split": ("string", "v4_split"),
    "subperiod": ("string", "subperiod"), "target_status": ("string", "target_status"),
    "next_split": ("string", "next_split"), "next_block_type": ("string", "next_block_type"),
    "loss_mask": ("bool", None), "eval_mask": ("bool", None), "target_class": ("int64", None),
    "target_oov": ("bool", None), "target_train_support": ("int64", None), "oov_cause": ("string", "oov_cause"),
    "rankable": ("bool", None), "target_class_cohort_train_positives": ("int64", None),
    "censor_class": ("string", "censor_class"), "target_category_state": ("string", "meta_token_state"),
    "target_brand_state": ("string", "meta_token_state"), "target_main_category_state": ("string", "meta_token_state"),
    "cold_stratum": ("string", "cold_stratum"), "n_train_events": ("int64", None),
    "n_eligible_train_examples": ("int64", None), "in_calib_prefix": ("bool", None), "row_start": ("int64", None),
    "row_count": ("int64", None), "n_validation_decisions": ("int64", None), "n_eval_rows": ("int64", None),
    "n_rankable_rows": ("int64", None), "fixed50_eligible": ("bool", None), "fixed20_eligible": ("bool", None),
    "panel_type": ("string", "panel_type"), "h": ("int32", None), "query_decision_id": ("int64", None),
    "query_decision_ts_us": ("int64", None), "n_prior_events_available": ("int64", None),
    "n_permitted_events": ("int64", None), "permitted_first_decision_id": ("int64", None),
    "n_support_labels": ("int64", None), "query_rankable": ("bool", None), "query_target_class": ("int64", None),
    "support_decision_id": ("int64", None), "support_target_ts_us": ("int64", None),
    "support_target_class": ("int64", None), "row": ("int64", None),
}
M02_DECISION_FIELDS = ["decision_id", "eval_user_key", "decision_ts_us", "target_ts_us", "old_split", "v4_split",
                       "subperiod", "target_status", "next_split", "next_block_type", "eval_mask", "target_class",
                       "target_oov", "target_train_support", "oov_cause", "rankable",
                       "target_class_cohort_train_positives", "censor_class", "target_category_state",
                       "target_brand_state", "target_main_category_state"]


class BuildBlocked(AssertionError):
    """An expected count, hash or assertion failed: the build stops and writes no success marker."""


def refuse_test_manifest(manifest_id: str, all_methods_lock_path: Optional[Path] = None) -> None:
    """Every TEST manifest is refused before ALL_METHODS_LOCK. This builder cannot satisfy the gate at all: it has
    no TEST read path, so it refuses even if a lock file is named (the gate belongs to a later, separate builder)."""
    if manifest_id in TEST_MANIFEST_IDS or "TEST" in manifest_id.upper():
        raise TestGateError(
            f"{manifest_id}: TEST manifests are built only after ALL_METHODS_LOCK.json exists, its sha256 verifies "
            "and a TEST_EXPOSURE_REGISTER MANIFEST_BUILD entry is appended (MANIFEST_SPEC_V4 global_rules.test_gate). "
            "The DATA-04 builder has no TEST code path and refuses unconditionally.")


def read_non_test(path: Path, columns: Sequence[str]) -> pa.Table:
    """Scan with split IN (non-TEST); TEST rows are never materialised into the returned table."""
    cols = list(columns) if "split" in columns else list(columns) + ["split"]
    t = pads.dataset(str(path), format="parquet", partitioning=None).to_table(
        columns=cols, filter=pc.field("split").isin(list(NON_TEST_SPLITS)))
    seen = set(pc.unique(t["split"]).to_pylist())
    if not seen <= set(NON_TEST_SPLITS):
        raise TestGateError(f"scan of {path} returned non-permitted split values {sorted(seen - set(NON_TEST_SPLITS))}")
    return t


def _np(a, dtype=None):
    x = a.to_numpy(zero_copy_only=False) if hasattr(a, "to_numpy") else np.asarray(a)
    return x.astype(dtype, copy=False) if dtype is not None else x


def _ts_us(col) -> np.ndarray:
    """timestamp[us] (UTC instants, tz-naive storage) -> int64 microseconds; NULL -> -1; exact round trip asserted."""
    if col.type != pa.timestamp("us"):
        raise BuildBlocked(f"timestamp column has type {col.type}, expected timestamp[us]")
    ints = pc.cast(col, pa.int64())
    back = pc.cast(ints, pa.timestamp("us"))
    if not pc.all(pc.or_kleene(pc.equal(back, col), pc.is_null(col))).as_py() in (True, None):
        raise BuildBlocked("timestamp -> int64 microsecond round trip failed")
    return _np(pc.fill_null(ints, -1), np.int64)


class Enums:
    def __init__(self, spec: dict):
        self.v = {k: list(v) for k, v in spec["enums"].items() if isinstance(v, list)}

    def code(self, name: str, arr, *, null_as: Optional[str] = None) -> np.ndarray:
        vocab = self.v[name]
        a = arr if null_as is None else pc.fill_null(arr, null_as)
        idx = pc.fill_null(pc.index_in(a, value_set=pa.array(vocab)), -1)
        out = _np(idx, np.int64)
        if np.any(out < 0):
            vals = a.to_pylist()
            bad = sorted({str(vals[i]) for i in np.flatnonzero(out < 0)[:50].tolist()})
            raise BuildBlocked(f"unregistered {name} value(s) {bad[:5]}: the build stops (MANIFEST_SPEC enums)")
        return out

    def c(self, name: str, value: str) -> int:
        return self.v[name].index(value)


def _bincount(idx: np.ndarray, mask: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(idx[mask], minlength=n).astype(np.int64)


class ManifestBuilder:
    def __init__(self, ds_root: Optional[Path] = None, out_root: Optional[Path] = None,
                 buckets: Optional[Sequence[int]] = None, *, expect: Optional[bool] = None,
                 log: Callable = print):
        self.ds = Path(ds_root) if ds_root is not None else dataset_root()
        self.out = Path(out_root) if out_root is not None else data_v2_root()
        self.buckets = list(range(N_BUCKETS)) if buckets is None else sorted(int(b) for b in buckets)
        self.full = self.buckets == list(range(N_BUCKETS))
        self.expect = self.full if expect is None else bool(expect)
        self.log = log
        self.t0 = time.time()
        self.spec = load_manifest_spec()
        self.schema = load_schema()
        self.E = Enums(self.spec)
        ver = json.loads((self.ds / "VERSION.json").read_text(encoding="utf-8"))
        if ver.get("version_hash") != VERSION_HASH:
            raise BuildBlocked(f"dataset version_hash {ver.get('version_hash')} != {VERSION_HASH}")
        self.recon: dict = {}
        self.timings: dict = {}
        self.code_sha = builder_code_sha256()

    # ------------------------------------------------------------------------------------------ utils
    def _say(self, msg: str) -> None:
        self.log(f"[v4data {time.time() - self.t0:7.1f}s] {msg}")

    def _ledger(self, b: int) -> Path:
        return self.ds / "decisions" / f"user_bucket={b}" / "ledger.parquet"

    def _features(self, b: int) -> Path:
        return self.ds / "model_ready" / f"user_bucket={b}" / "features.parquet"

    def _add(self, key: str, n) -> None:
        self.recon[key] = self.recon.get(key, 0) + int(n)

    # --------------------------------------------------------------------------------------- pass 0
    def pass_support(self) -> None:
        t0 = time.time()
        sup: dict[str, int] = {}
        for b in self.buckets:
            t = read_non_test(self._ledger(b), ["split", "product_id"])
            t = t.filter(pc.equal(t["split"], "TRAIN"))
            vc = pc.value_counts(t["product_id"])
            for pid, n in zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()):
                sup[pid] = sup.get(pid, 0) + int(n)
        self.support = sup
        cat = pq.ParquetFile(self.ds / "mappings" / "catalogue.parquet").read(
            columns=["product_id", "output_class", "product_idx", "representative_views"])
        pid = cat["product_id"].to_pylist()
        rv = _np(cat["representative_views"], np.int64)
        oc = _np(cat["output_class"], np.int64)
        pix = _np(cat["product_idx"], np.int64)
        self.K = len(pid)
        self.candidate_order_hash = candidate_order_hash(oc, pid)
        if not np.array_equal(pix[np.argsort(oc)], np.arange(3, 3 + self.K)):
            raise BuildBlocked("catalogue product_idx != output_class + 3")
        mism = sum(1 for p, r in zip(pid, rv.tolist()) if sup.get(p, 0) != r)
        ge5 = {p for p, n in sup.items() if n >= 5}
        self.support_check = {
            "K": self.K, "n_catalogue_support_mismatch": mism, "n_products_with_train_support": len(sup),
            "support_ge5_set_equals_catalogue": ge5 == set(pid), "candidate_order_hash": self.candidate_order_hash}
        self.recon["catalogue.n_rows"] = self.K
        self.recon["catalogue.support_mismatch"] = mism
        self.recon["catalogue.K1_products_with_train_support"] = len(sup)
        self.recon["check.support_ge5_set_differs_from_catalogue"] = 0 if ge5 == set(pid) else 1
        if self.full and (mism != 0 or ge5 != set(pid) or self.K != K):
            raise BuildBlocked(f"support definition does not reproduce the catalogue: {self.support_check}")
        self.timings["pass0_support_s"] = round(time.time() - t0, 1)
        self._say(f"P0 support: {len(sup)} TRAIN products, catalogue K={self.K}, mismatches={mism}")

    # --------------------------------------------------------------------------------------- pass A
    def _row_arrays(self, t: pa.Table) -> dict:
        E = self.E
        r = {
            "split": E.code("old_split", t["split"]),
            "dts": _ts_us(t["decision_ts"]),
            "tts": _ts_us(t["target_ts"]),
            "status": E.code("target_status", t["target_status"]),
            "nsplit": E.code("next_split", t["next_split"], null_as="NONE"),
            "nblock": E.code("next_block_type", t["next_block_type"], null_as="NONE"),
            "has_obs": _np(t["has_observed_target"], bool),
            "loss": _np(t["loss_mask"], bool),
            "eval": _np(t["eval_mask"], bool),
            "tc": _np(t["target_class"], np.int64),
            "oov": _np(t["target_oov"], bool),
        }
        for k in ("has_observed_target", "loss_mask", "eval_mask", "target_class", "target_oov", "target_status"):
            if t[k].null_count:
                raise BuildBlocked(f"NULL in ledger column {k}")
        tr = r["split"] == E.c("old_split", "TRAIN")
        vs = r["split"] == E.c("old_split", "VALIDATION_SELECT")
        vc = r["split"] == E.c("old_split", "VALIDATION_CONFIRM")
        r.update(tr=tr, vs=vs, vc=vc, val=vs | vc)
        r["elig"] = tr & r["loss"] & (r["tc"] >= 0) & (r["tts"] >= 0) & (r["tts"] < TRAIN_END_US)
        r["censor"] = self._censor(r)
        return r

    def _censor(self, r: dict) -> np.ndarray:
        """EVALUATION_CONTRACT_V4 section 2.2 / E2.1 (R2) for VALIDATION rows; -1 elsewhere."""
        E = self.E
        st, ns = r["status"], r["nsplit"]
        out = np.full(st.shape, -1, dtype=np.int64)
        v = r["val"]
        bu = st == E.c("target_status", "BOUNDARY_UNRESOLVED")
        ab = st == E.c("target_status", "AMBIGUOUS_BOUNDARY")
        nn = st == E.c("target_status", "NO_OBSERVED_NEXT")
        obs = st == E.c("target_status", "OBSERVED_IN_PARTITION")
        legacy = bu & r["vs"] & (ns == E.c("next_split", "VALIDATION_CONFIRM"))
        right = bu & ((ns == E.c("next_split", "TEST")) | (ns == E.c("next_split", "LABEL_GRACE")))
        sel = [(r["eval"], "NONE"), (legacy, "CEN_LEGACY_BARRIER"), (right, "CEN_RIGHT_EDGE"),
               (ab, "CEN_AMBIGUOUS_NEXT_BARRIER"), (nn, "CEN_NO_NEXT")]
        for m, name in sel:                                             # precedence = list order
            out[v & m & (out < 0)] = E.c("censor_class", name)
        if np.any(v & (out < 0)):
            raise BuildBlocked("a VALIDATION row matches no censor rule: the build stops (E2.1)")
        if np.any(v & (r["eval"] != obs)):
            raise BuildBlocked("eval_mask != (target_status == OBSERVED_IN_PARTITION) on a VALIDATION row")
        return out

    def pass_aggregate(self) -> None:
        t0 = time.time()
        E = self.E
        dig_l, cnt_l, bkt_l = [], [], []
        for b in self.buckets:
            t = read_non_test(self._ledger(b), LEDGER_A)
            if t["user_id"].null_count:
                raise BuildBlocked("NULL user_id in the ledger")
            r = self._row_arrays(t)
            uniq = pc.unique(t["user_id"])
            ui = _np(pc.index_in(t["user_id"], value_set=uniq), np.int64)
            nu = len(uniq)
            tr, val, ev = r["tr"], r["val"], r["eval"]
            cnts = np.stack([_bincount(ui, tr, nu), _bincount(ui, r["elig"], nu), _bincount(ui, val, nu),
                             _bincount(ui, val & ev, nu), _bincount(ui, val & ev & (r["tc"] >= 0), nu)], axis=1)
            dig_l.append(digests_of(uniq.to_pylist()))
            cnt_l.append(cnts)
            bkt_l.append(np.full(nu, b, dtype=np.int16))
            self._reconcile_bucket(r, ui, nu)
        self.u_dig = np.concatenate(dig_l)
        cnt = np.concatenate(cnt_l)
        self.u_ntrain, self.u_nelig, self.u_nval, self.u_neval, self.u_nrank = (cnt[:, i] for i in range(5))
        self.u_bucket = np.concatenate(bkt_l)
        if np.unique(self.u_dig).size != self.u_dig.size:
            raise BuildBlocked("client_key_hex collision or a user spanning two buckets (fatal, C1.2)")
        self.u_key = selection_keys_of(self.u_dig)
        self.timings["passA_aggregate_s"] = round(time.time() - t0, 1)
        self._say(f"PA aggregates: {self.u_dig.size} users (non-TEST rows)")

    def _reconcile_bucket(self, r: dict, ui: np.ndarray, nu: int) -> None:
        E = self.E
        for name, m in (("TRAIN", r["tr"]), ("VALIDATION_SELECT", r["vs"]), ("VALIDATION_CONFIRM", r["vc"]),
                        ("VALIDATION", r["val"])):
            self._add(f"{name}.decisions", m.sum())
            self._add(f"{name}.distinct_users", np.unique(ui[m]).size)
            if name != "TRAIN":
                self._add(f"{name}.eval_mask_true", (m & r["eval"]).sum())
        self._add("TRAIN.loss_mask_true", (r["tr"] & r["loss"]).sum())
        self._add("TRAIN.target_oov_true", (r["tr"] & r["oov"]).sum())
        self._add("VALIDATION.target_oov_true", (r["val"] & r["oov"]).sum())
        self._add("VALIDATION.target_class_ge0", (r["val"] & r["eval"] & (r["tc"] >= 0)).sum())
        for s in E.v["target_status"]:
            m = r["status"] == E.c("target_status", s)
            if (r["tr"] & m).any() or f"TRAIN.status.{s}" in self.recon:
                self._add(f"TRAIN.status.{s}", (r["tr"] & m).sum())
            if (r["val"] & m).any() or f"VALIDATION.status.{s}" in self.recon:
                self._add(f"VALIDATION.status.{s}", (r["val"] & m).sum())
        for c in E.v["censor_class"]:
            if c != "NONE":
                self._add(f"VALIDATION.censor.{c}", (r["censor"] == E.c("censor_class", c)).sum())
        self._add("label_time.train_loss_target_ts_ge_train_end", (r["tr"] & r["loss"] & (r["tts"] >= TRAIN_END_US)).sum())
        self._add("label_time.train_loss_target_ts_null", (r["tr"] & r["loss"] & (r["tts"] < 0)).sum())
        self._add("label_time.val_eval_target_ts_ge_val_end", (r["val"] & r["eval"] & (r["tts"] >= VALIDATION_END_US)).sum())
        self._add("label_time.decisions_ts_ge_2019_11_30_non_test", (r["dts"] >= TEST_END_US).sum())
        d = r["dts"]
        bad_bounds = ((r["tr"] & ~((d >= TRAIN_START_US) & (d < TRAIN_END_US)))
                      | (r["vs"] & ~((d >= TRAIN_END_US) & (d < VS_END_US)))
                      | (r["vc"] & ~((d >= VS_END_US) & (d < VALIDATION_END_US))))
        self._add("check.decision_ts_outside_split_bounds", bad_bounds.sum())
        cen = r["tr"] & r["has_obs"] & (r["tts"] >= 0) & (r["tts"] < TRAIN_END_US) & (r["tc"] >= 0)
        self._add("check.m01_filter_vs_census_eligibility_mismatch", (cen != r["elig"]).sum())
        self._add("check.loss_mask_with_negative_class", (r["loss"] & (r["tc"] < 0)).sum())
        self._add("check.target_oov_not_observed_or_classed", (r["oov"] & (~r["has_obs"] | (r["tc"] != -1))).sum())
        ab = r["status"] == E.c("target_status", "AMBIGUOUS_BOUNDARY")
        bu = r["status"] == E.c("target_status", "BOUNDARY_UNRESOLVED")
        self._add("check.AB_next_block_not_ambiguous_barrier",
                  (ab & (r["nblock"] != E.c("next_block_type", "AMBIGUOUS_TIME_BLOCK_BARRIER"))).sum())
        own = np.select([r["tr"], r["vs"], r["vc"]], [E.c("next_split", "TRAIN"), E.c("next_split", "VALIDATION_SELECT"),
                                                       E.c("next_split", "VALIDATION_CONFIRM")], -9)
        self._add("check.BU_not_single_item_other_split",
                  (bu & ((r["nblock"] != E.c("next_block_type", "SINGLE_ITEM")) | (r["nsplit"] == own))).sum())
        self._add("check.has_obs_without_target_ts", (r["has_obs"] & (r["tts"] < 0)).sum())

    # ------------------------------------------------------------------------------------ selection
    def select_populations(self) -> None:
        U = np.flatnonzero(self.u_nelig >= 1)
        keysU = self.u_key[U]
        if np.unique(keysU).size != keysU.size:
            raise BuildBlocked("two users of U share a selection_key: BLOCK and escalate (CLIENT_CONTRACT C2.5)")
        U_sorted = U[np.argsort(keysU, kind="stable")]
        self.U_sorted = U_sorted
        ncoh = min(N_COHORT, U_sorted.size)
        self.cohort = U_sorted[:ncoh]
        self.calib = U_sorted[:min(N_CALIB, ncoh)]
        self.pop_hash = {
            "universe_size": int(U.size),
            "cohort16384.population_sha256": population_sha256(self.u_key[U_sorted[:16384]]),
            "cohort65536.population_sha256": population_sha256(self.u_key[self.calib]),
            "cohort262144.population_sha256": population_sha256(self.u_key[self.cohort]),
            "cohortALL.population_sha256": population_sha256(self.u_key[U_sorted]),
        }
        cold = np.flatnonzero((self.u_neval >= 1) & (self.u_nelig == 0))
        self.cold_sorted = cold[np.argsort(self.u_key[cold], kind="stable")]
        self.m04 = self.cold_sorted[:N_PANEL]
        self.m04s = self.cold_sorted[N_PANEL:2 * N_PANEL]
        coh_e2e = self.cohort[self.u_neval[self.cohort] >= 1]
        self.m04w = coh_e2e[np.argsort(self.u_key[coh_e2e], kind="stable")][:N_PANEL]
        n = self.u_dig.size
        self.role = np.zeros(n, dtype=np.int64)
        for idx, bit in ((self.cohort, COHORT), (self.calib, CALIB), (self.cold_sorted, COLD), (self.m04, P_M04),
                         (self.m04w, P_M04W), (self.m04s, P_M04S)):
            self.role[idx] |= bit
        if np.any((self.role & COHORT) & (self.role & COLD)):
            raise BuildBlocked("a user is both cohort (warm) and cold")
        rel = np.flatnonzero(self.role != 0)
        self.role_of = dict(zip(digest_rows(self.u_dig[rel]), zip(rel.tolist(), self.role[rel].tolist())))
        self.pop_recon = {
            "universe_size": int(U.size),
            "users_in_train_zero_eligible": int(((self.u_ntrain > 0) & (self.u_nelig == 0)).sum()),
            "val_users_with_ge1_eval": int((self.u_neval >= 1).sum()),
            "val_users_warm": int(((self.u_neval >= 1) & (self.u_nelig >= 1)).sum()),
            "val_users_cold": int(cold.size),
            "s5_total_examples": int(self.u_nelig.sum()),
            "cohort262144.eligible_examples": int(self.u_nelig[self.cohort].sum()),
            "cohort262144.val_eval_rows": int(self.u_neval[self.cohort].sum()),
            "cohort262144.users_ge1_eval": int((self.u_neval[self.cohort] >= 1).sum()),
            "cohort65536.eligible_examples": int(self.u_nelig[self.calib].sum()),
            "cohort65536.val_eval_rows": int(self.u_neval[self.calib].sum()),
            "cohort65536.users_ge1_eval": int((self.u_neval[self.calib] >= 1).sum()),
            "cold.COLD_NO_TRAIN_EVENTS.users": int((self.u_ntrain[cold] == 0).sum()),
            "cold.COLD_NO_TRAIN_EVENTS.eval_rows": int(self.u_neval[cold][self.u_ntrain[cold] == 0].sum()),
            "cold.COLD_TRAIN_EVENTS_NO_LABEL.users": int((self.u_ntrain[cold] > 0).sum()),
            "cold.COLD_TRAIN_EVENTS_NO_LABEL.eval_rows": int(self.u_neval[cold][self.u_ntrain[cold] > 0].sum()),
        }
        self.pop_recon.update({k: v for k, v in self.pop_hash.items() if k != "universe_size"})
        if self.full:
            self._clients_parquet_check()
        self._say(f"selection: |U|={U.size} cohort={self.cohort.size} cold={cold.size} "
                  f"M04={self.m04.size} M04W={self.m04w.size} M04S={self.m04s.size}")

    def _clients_parquet_check(self) -> None:
        """COLD_NO_TRAIN_EVENTS iff absent from mappings/clients.parquet (MANIFEST_SPEC M03 assertion)."""
        cl = pq.ParquetFile(self.ds / "mappings" / "clients.parquet").read(columns=["user_id"])
        cd = np.sort(digests_of(cl["user_id"].to_pylist()))
        del cl
        td = np.sort(self.u_dig[self.u_ntrain > 0])
        ok = cd.size == td.size and bool(np.array_equal(cd, td))
        self.recon["check.clients_parquet_equals_train_users"] = 0 if ok else 1
        self.recon["clients_parquet_rows"] = int(cd.size)

    # --------------------------------------------------------------------------------------- pass B
    def pass_extract(self) -> None:
        t0 = time.time()
        E = self.E
        m01, val, ev = [], [], []
        self.meta_checks = {"target_rows_joined": 0, "target_pid_mismatch": 0, "target_ts_mismatch": 0}
        for b in self.buckets:
            t = read_non_test(self._ledger(b), LEDGER_B)
            r = self._row_arrays(t)
            uniq = pc.unique(t["user_id"])
            ui = _np(pc.index_in(t["user_id"], value_set=uniq), np.int64)
            ud = digest_rows(digests_of(uniq.to_pylist()))
            g_u = np.full(len(uniq), -1, dtype=np.int64)
            role_u = np.zeros(len(uniq), dtype=np.int64)
            for i, d in enumerate(ud):
                hit = self.role_of.get(d)
                if hit is not None:
                    g_u[i], role_u[i] = hit
            g, role = g_u[ui], role_u[ui]
            did = _np(t["decision_id"], np.int64)
            sel = ((role & COHORT) != 0) & r["elig"]
            m01.append({"g": g[sel], "decision_id": did[sel], "dts": r["dts"][sel], "tts": r["tts"][sel],
                        "tc": r["tc"][sel], "status": r["status"][sel], "split": r["split"][sel]})
            # TRAIN support of the observed target, for every VALIDATION row (full-population s=1 check) and for
            # the cohort's TRAIN OOV targets (M01/M05 counts by cause)
            coh_tr = r["tr"] & ((role & COHORT) != 0)
            need_sup = r["has_obs"] & (r["val"] | (coh_tr & r["oov"]))
            sup_row = np.full(r["tc"].size, -1, dtype=np.int64)
            nr = np.flatnonzero(need_sup)
            if nr.size:
                pids = t["target_product_id"].take(pa.array(nr)).to_pylist()
                sup_row[nr] = np.fromiter((self.support.get(p, 0) for p in pids), dtype=np.int64, count=len(pids))
            self._add("VALIDATION.oov_absent_from_train_s1_excluded", (r["val"] & r["has_obs"] & (sup_row == 0)).sum())
            for tag, bit in (("M01", COHORT), ("M05", CALIB)):
                ct = r["tr"] & ((role & bit) != 0)
                self._add(f"{tag}.n_label_time_excluded",
                          (ct & r["loss"] & (r["tc"] >= 0) & ((r["tts"] < 0) | (r["tts"] >= TRAIN_END_US))).sum())
                self._add(f"{tag}.n_oov_absent_from_train", (ct & r["has_obs"] & r["oov"] & (sup_row == 0)).sum())
                self._add(f"{tag}.n_oov_below_support",
                          (ct & r["has_obs"] & r["oov"] & (sup_row >= 1) & (sup_row <= 4)).sum())
                for s in ("NO_OBSERVED_NEXT", "AMBIGUOUS_BOUNDARY", "BOUNDARY_UNRESOLVED"):
                    self._add(f"{tag}.n_status_{s}", (ct & (r["status"] == E.c("target_status", s))).sum())
                self._add(f"{tag}.n_train_rows", ct.sum())
            vm = r["val"] & ((role & (COHORT | COLD)) != 0)
            rows = np.flatnonzero(vm)
            sup = np.where(r["has_obs"][rows], sup_row[rows], -1).astype(np.int64)
            meta = self._target_meta(b, t, r, ui, uniq, rows)
            val.append({"g": g[rows], "role": role[rows], "decision_id": did[rows], "dts": r["dts"][rows],
                        "tts": r["tts"][rows], "split": r["split"][rows], "status": r["status"][rows],
                        "nsplit": r["nsplit"][rows], "nblock": r["nblock"][rows], "eval": r["eval"][rows],
                        "tc": r["tc"][rows], "oov": r["oov"][rows], "has_obs": r["has_obs"][rows],
                        "support": sup, "censor": r["censor"][rows], **meta})
            pm = (role & (P_M04 | P_M04W | P_M04S)) != 0
            ev.append({"g": g[pm], "decision_id": did[pm], "dts": r["dts"][pm], "tts": r["tts"][pm],
                       "split": r["split"][pm], "eval": r["eval"][pm], "loss": r["loss"][pm],
                       "has_obs": r["has_obs"][pm], "tc": r["tc"][pm]})
        cat = lambda parts: {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}  # noqa: E731
        self.rows_m01, self.rows_val, self.rows_ev = cat(m01), cat(val), cat(ev)
        self.timings["passB_extract_s"] = round(time.time() - t0, 1)
        self._say(f"PB extract: M01 rows {self.rows_m01['g'].size}, VAL rows {self.rows_val['g'].size}, "
                  f"panel events {self.rows_ev['g'].size}; target meta joins {self.meta_checks}")

    def _target_meta(self, b: int, t: pa.Table, r: dict, ui: np.ndarray, uniq, rows: np.ndarray) -> dict:
        """Token state of the TARGET's own representative event row in model_ready (the next representative of
        the same segment; the frozen end-of-TRAIN metadata snapshot applies to future partitions,
        metadata.py:167-169). NOT_APPLICABLE when there is no observed target. Target identity is verified."""
        E = self.E
        na = E.c("meta_token_state", "NOT_APPLICABLE")
        out = {k: np.full(rows.size, na, dtype=np.int64) for k in ("cat_state", "brand_state", "main_state")}
        need = rows[r["has_obs"][rows]]
        if need.size == 0:
            return out
        F = read_non_test(self._features(b), FEATURES_B)
        fu = _np(pc.fill_null(pc.index_in(F["user_id"], value_set=uniq), -1), np.int64)
        seg = _np(F["segment_index"], np.float64).astype(np.int64)
        pos = _np(F["position_in_segment"], np.int64)
        if np.any(fu < 0) or seg.max(initial=0) >= (1 << 18) or pos.max(initial=0) >= (1 << 17) \
                or len(uniq) >= (1 << 27):
            raise BuildBlocked("features key out of range for the (user, segment, position) join")
        fkey = (fu << 36) | (seg << 18) | pos
        order = np.argsort(fkey, kind="stable")
        fks = fkey[order]
        lseg = _np(t["segment_index"], np.float64).astype(np.int64)
        lpos = _np(t["position_in_segment"], np.int64)
        want = (ui[need] << 36) | (lseg[need] << 18) | (lpos[need] + 1)
        j = np.clip(np.searchsorted(fks, want), 0, fks.size - 1)
        if not np.array_equal(fks[j], want):
            raise BuildBlocked("an observed target has no next-representative feature row")
        frow = order[j]
        pid_ok = pc.equal(F["product_id"].take(pa.array(frow)), t["target_product_id"].take(pa.array(need)))
        ts_ok = pc.equal(F["block_ts"].take(pa.array(frow)), t["target_ts"].take(pa.array(need)))
        n_pid_bad = int(len(need) - pc.sum(pc.cast(pid_ok, pa.int64())).as_py())
        n_ts_bad = int(len(need) - pc.sum(pc.cast(ts_ok, pa.int64())).as_py())
        self.meta_checks["target_rows_joined"] += int(need.size)
        self.meta_checks["target_pid_mismatch"] += n_pid_bad
        self.meta_checks["target_ts_mismatch"] += n_ts_bad
        if n_pid_bad or n_ts_bad:
            raise BuildBlocked("target feature row does not carry the target product/time")
        pos_in_rows = np.searchsorted(rows, need)
        for key, col in (("cat_state", "category_idx"), ("brand_state", "brand_idx"),
                         ("main_state", "main_category_idx")):
            tok = _np(F[col], np.int64)[frow]
            st = np.full(tok.shape, -1, dtype=np.int64)
            st[tok >= 3] = E.c("meta_token_state", "KNOWN")
            st[tok == 1] = E.c("meta_token_state", "MISSING")
            st[tok == 2] = E.c("meta_token_state", "OOV")
            if np.any(st < 0):
                raise BuildBlocked(f"PAD token on a real target event ({col})")
            out[key][pos_in_rows] = st
        return out

    # ------------------------------------------------------------------------------------- assembly
    def _dense_rank_by_digest(self, gidx: np.ndarray) -> np.ndarray:
        """Dense rank of client_key_hex (= digest bytes = selection_key order) among the given users."""
        order = np.argsort(self.u_dig[gidx], kind="stable")
        rank = np.empty(gidx.size, dtype=np.int64)
        rank[order] = np.arange(gidx.size)
        return rank

    def assemble(self) -> dict:
        E = self.E
        man: dict = {}
        # ---------------------------------------------------------------- M01 / M05
        coh = np.sort(self.cohort)
        ord_of = np.full(self.u_dig.size, -1, dtype=np.int64)
        coh_rank = self._dense_rank_by_digest(coh)
        ord_of[coh] = coh_rank
        m = self.rows_m01
        man["M01_COHORT_TRAIN"] = self._train_manifest("M01_COHORT_TRAIN", m, coh, ord_of, None)
        calib_set = np.zeros(self.u_dig.size, dtype=bool)
        calib_set[self.calib] = True
        keep = calib_set[m["g"]]
        m5 = {k: v[keep] for k, v in m.items()}
        cal = np.sort(self.calib)
        ord5 = np.full(self.u_dig.size, -1, dtype=np.int64)
        ord5[cal] = self._dense_rank_by_digest(cal)
        man["M05_CALIB_TRAIN"] = self._train_manifest("M05_CALIB_TRAIN", m5, cal, ord5, "M01_COHORT_TRAIN")
        pos01 = np.bincount(m["tc"], minlength=self.K).astype(np.int64)
        pos05 = np.bincount(m5["tc"], minlength=self.K).astype(np.int64)
        self.class_pos = {"M01": pos01, "M05": pos05}
        # ---------------------------------------------------------------- M02 / M06 / M03
        v = self.rows_val
        coh_rows = (v["role"] & COHORT) != 0
        cold_rows = (v["role"] & COLD) != 0
        man["M02_VALIDATION_WARM"] = self._eval_manifest("M02_VALIDATION_WARM", {k: a[coh_rows] for k, a in v.items()},
                                                         pos01, "warm", "M01_COHORT_TRAIN")
        cal_rows = coh_rows & calib_set[v["g"]]
        man["M06_CALIB_VALIDATION_WARM"] = self._eval_manifest(
            "M06_CALIB_VALIDATION_WARM", {k: a[cal_rows] for k, a in v.items()}, pos05, "warm", "M02_VALIDATION_WARM")
        man["M03_COLD_PANEL_VAL"] = self._eval_manifest("M03_COLD_PANEL_VAL", {k: a[cold_rows] for k, a in v.items()},
                                                        pos01, "cold", None)
        # ---------------------------------------------------------------- panels
        for mid, users, label, parent in (("M04_COLD_H_PANEL", self.m04, "COLD_START", "M03_COLD_PANEL_VAL"),
                                          ("M04W_WARM_TRUNCATION_PANEL", self.m04w, "WARM_TRUNCATION",
                                           "M02_VALIDATION_WARM"),
                                          ("M04S_FEW_SHOT_SELECTION_PANEL", self.m04s, "RECIPE_SELECTION",
                                           "M03_COLD_PANEL_VAL")):
            man[mid] = self._panel_manifest(mid, users, label, parent)
        self._cross_manifest_checks(man)
        return man

    # ------------------------------------------------------------------------------ train manifests
    def _train_manifest(self, mid: str, m: dict, users: np.ndarray, ord_of: np.ndarray, parent) -> dict:
        E = self.E
        co = ord_of[m["g"]]
        if np.any(co < 0):
            raise BuildBlocked(f"{mid}: a row belongs to no manifest client")
        o = np.lexsort((m["decision_id"], co))
        dec = {"decision_id": m["decision_id"][o], "client_ord": co[o], "decision_ts_us": m["dts"][o],
               "target_ts_us": m["tts"][o], "target_class": m["tc"][o],
               "target_status": EnumValues(m["status"][o], E.v["target_status"]),
               "old_split": EnumValues(m["split"][o], E.v["old_split"])}
        n_cli = users.size
        by_ord = np.empty(n_cli, dtype=np.int64)
        by_ord[ord_of[users]] = users
        cnt = np.bincount(dec["client_ord"], minlength=n_cli).astype(np.int64)
        start = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
        in_cal = np.zeros(self.u_dig.size, dtype=bool)
        in_cal[self.calib] = True
        cli = {"client_ord": np.arange(n_cli, dtype=np.int64), "client_key_hex": hex_of(self.u_dig[by_ord]),
               "selection_key": self.u_key[by_ord], "n_eligible_train_examples": self.u_nelig[by_ord],
               "row_start": start, "row_count": cnt, "in_calib_prefix": in_cal[by_ord]}
        # assertions (MANIFEST_SPEC M01)
        a = {}
        a["target_ts_lt_train_end"] = bool(np.all((dec["target_ts_us"] >= 0) & (dec["target_ts_us"] < TRAIN_END_US)))
        a["target_class_in_range"] = bool(np.all((dec["target_class"] >= 0) & (dec["target_class"] < self.K)))
        a["status_all_observed"] = bool(np.all(m["status"] == E.c("target_status", "OBSERVED_IN_PARTITION")))
        a["every_client_row_count_ge1"] = bool(np.all(cnt >= 1))
        a["row_count_equals_n_eligible"] = bool(np.array_equal(cnt, cli["n_eligible_train_examples"]))
        a["client_ranges_contiguous_cover"] = bool(
            np.array_equal(np.repeat(np.arange(n_cli), cnt), dec["client_ord"]) and cnt.sum() == dec["decision_id"].size)
        a["selection_key_ascending_with_client_ord"] = bool(np.all(cli["selection_key"][1:] > cli["selection_key"][:-1]))
        a["calib_prefix_is_first_65536"] = bool(np.array_equal(np.flatnonzero(cli["in_calib_prefix"]),
                                                               np.arange(int(cli["in_calib_prefix"].sum()))))
        a["decision_ts_in_train"] = bool(np.all((dec["decision_ts_us"] >= TRAIN_START_US)
                                                & (dec["decision_ts_us"] < TRAIN_END_US)))
        a["decision_id_unique"] = bool(np.unique(dec["decision_id"]).size == dec["decision_id"].size)
        central = np.argsort(dec["decision_id"], kind="stable").astype(np.int64)
        per = cnt.astype(np.float64)
        expected = []
        if mid == "M01_COHORT_TRAIN":
            expected = [("n_users", n_cli, 262144), ("n_decisions", int(cnt.sum()), 3906929),
                        ("population_sha256", population_sha256(np.sort(cli["selection_key"])),
                         "54e1d3c2539fd4cd7fe22d2b8ef91bed1a786d69be27867406bb19141ac48184"),
                        ("per_client_examples_min", int(cnt.min()) if n_cli else None, 1),
                        ("per_client_examples_p50", float(np.percentile(per, 50, method="linear")) if n_cli else None, 6),
                        ("per_client_examples_max", int(cnt.max()) if n_cli else None, 3014),
                        ("n_in_calib_prefix", int(cli["in_calib_prefix"].sum()), 65536)]
        else:
            expected = [("n_users", n_cli, 65536), ("n_decisions", int(cnt.sum()), 990362),
                        ("population_sha256", population_sha256(np.sort(cli["selection_key"])),
                         "3d680aa7222fb5c9246d03fe6088464c78095d90a344e3453acb68e98d53a6f4")]
        tables = {"clients": cli, "decisions": dec, "central_order": {"row": central}}
        tag = "M01" if mid == "M01_COHORT_TRAIN" else "M05"
        g = lambda k: int(self.recon.get(f"{tag}.{k}", 0))  # noqa: E731
        counts = {
            "n_label_time_excluded": g("n_label_time_excluded"),
            "n_oov_below_support": g("n_oov_below_support"),
            "n_oov_absent_from_train": g("n_oov_absent_from_train"),
            "n_activity_filtered": 0, "n_split_reassigned": 0,
            "n_censor_CEN_NO_NEXT": g("n_status_NO_OBSERVED_NEXT"),
            "n_censor_CEN_LEGACY_BARRIER": 0, "n_censor_CEN_RIGHT_EDGE": 0,
            "n_censor_CEN_AMBIGUOUS_NEXT_BARRIER": g("n_status_AMBIGUOUS_BOUNDARY"),
            "n_status_BOUNDARY_UNRESOLVED_train_fence": g("n_status_BOUNDARY_UNRESOLVED"),
            "n_invalid_source_quarantined": 0,
            "n_train_rows_of_manifest_users": g("n_train_rows"),
            "n_supervised_examples": int(cnt.sum()),
            "notes": {"censor_fields_for_TRAIN": "TRAIN rows of the manifest's users without a label, by ledger status "
                                                 "(NO_OBSERVED_NEXT, AMBIGUOUS_BOUNDARY); the VALIDATION-only classes "
                                                 "LEGACY_BARRIER and RIGHT_EDGE do not exist in TRAIN; the TRAIN-fence "
                                                 "BOUNDARY_UNRESOLVED rows are reported separately",
                      "n_invalid_source_quarantined": "rows quarantined by this build (0); upstream v1 quarantines are "
                                                      "recorded in the v1 artifacts and were not re-read",
                      "identity": "n_train_rows = n_supervised_examples + n_oov_below_support + n_oov_absent_from_train"
                                  " + n_censor_CEN_NO_NEXT + n_censor_CEN_AMBIGUOUS_NEXT_BARRIER"
                                  " + n_status_BOUNDARY_UNRESOLVED_train_fence + n_label_time_excluded"},
        }
        a["train_rows_by_cause_sum"] = counts["n_train_rows_of_manifest_users"] == (
            counts["n_supervised_examples"] + counts["n_oov_below_support"] + counts["n_oov_absent_from_train"]
            + counts["n_censor_CEN_NO_NEXT"] + counts["n_censor_CEN_AMBIGUOUS_NEXT_BARRIER"]
            + counts["n_status_BOUNDARY_UNRESOLVED_train_fence"] + counts["n_label_time_excluded"])
        return {"manifest_id": mid, "tables": tables, "assertions": a, "expected": expected, "parent": parent,
                "population_sha256": population_sha256(np.sort(cli["selection_key"])), "n_users": n_cli,
                "n_decisions": int(cnt.sum()), "n_eval_rows": 0, "n_rankable_rows": 0, "counts_by_cause": counts,
                "decision_table": "decisions"}

    # ------------------------------------------------------------------------------- eval manifests
    def _eval_manifest(self, mid: str, v: dict, class_pos: np.ndarray, kind: str, parent) -> dict:
        E = self.E
        o = np.argsort(v["decision_id"], kind="stable")
        v = {k: a[o] for k, a in v.items()}
        users = np.unique(v["g"])
        rank = self._dense_rank_by_digest(users)
        key_of = np.full(self.u_dig.size, -1, dtype=np.int64)
        key_of[users] = rank
        euk = key_of[v["g"]]
        rankable = v["eval"] & (v["tc"] >= 0)
        # oov_cause
        oc = np.full(v["g"].size, E.c("oov_cause", "NOT_APPLICABLE"), dtype=np.int64)
        oc[v["has_obs"] & (v["tc"] >= 0)] = E.c("oov_cause", "NONE")
        oc[v["oov"] & (v["support"] == 0)] = E.c("oov_cause", "ABSENT_FROM_TRAIN")
        oc[v["oov"] & (v["support"] >= 1) & (v["support"] <= 4)] = E.c("oov_cause", "BELOW_SUPPORT")
        bad_oc = v["has_obs"] & ~(v["tc"] >= 0) & ~v["oov"] | (v["oov"] & (v["support"] >= 5))
        v4 = np.full(v["g"].size, E.c("v4_split", "VALIDATION"), dtype=np.int64)
        sub = np.where(v["split"] == E.c("old_split", "VALIDATION_SELECT"),
                       E.c("subperiod", "VAL_A_20191116_20191120"), E.c("subperiod", "VAL_B_20191120_20191123"))
        tpos = np.where(rankable, class_pos[np.clip(v["tc"], 0, None)], -1).astype(np.int64)
        tsup = np.where(v["has_obs"], v["support"], -1).astype(np.int64)
        dec = {"decision_id": v["decision_id"], "eval_user_key": euk, "decision_ts_us": v["dts"],
               "target_ts_us": v["tts"], "old_split": EnumValues(v["split"], E.v["old_split"]),
               "v4_split": EnumValues(v4, E.v["v4_split"]), "subperiod": EnumValues(sub, E.v["subperiod"]),
               "target_status": EnumValues(v["status"], E.v["target_status"]),
               "next_split": EnumValues(v["nsplit"], E.v["next_split"]),
               "next_block_type": EnumValues(v["nblock"], E.v["next_block_type"]), "eval_mask": v["eval"],
               "target_class": v["tc"], "target_oov": v["oov"], "target_train_support": tsup,
               "oov_cause": EnumValues(oc, E.v["oov_cause"]), "rankable": rankable,
               "target_class_cohort_train_positives": tpos, "censor_class": EnumValues(v["censor"], E.v["censor_class"]),
               "target_category_state": EnumValues(v["cat_state"], E.v["meta_token_state"]),
               "target_brand_state": EnumValues(v["brand_state"], E.v["meta_token_state"]),
               "target_main_category_state": EnumValues(v["main_state"], E.v["meta_token_state"])}
        n_u = users.size
        by_key = np.empty(n_u, dtype=np.int64)
        by_key[rank] = users
        nd = np.bincount(euk, minlength=n_u).astype(np.int64)
        ne = np.bincount(euk[v["eval"]], minlength=n_u).astype(np.int64)
        nr = np.bincount(euk[rankable], minlength=n_u).astype(np.int64)
        usr = {"eval_user_key": np.arange(n_u, dtype=np.int64), "client_key_hex": hex_of(self.u_dig[by_key])}
        if kind == "cold":
            strat = np.where(self.u_ntrain[by_key] == 0, E.c("cold_stratum", "COLD_NO_TRAIN_EVENTS"),
                             E.c("cold_stratum", "COLD_TRAIN_EVENTS_NO_LABEL"))
            usr.update({"selection_key": self.u_key[by_key],
                        "cold_stratum": EnumValues(strat, E.v["cold_stratum"]),
                        "n_train_events": self.u_ntrain[by_key]})
        usr.update({"n_validation_decisions": nd, "n_eval_rows": ne, "n_rankable_rows": nr})
        cen = v["censor"]
        counts = {
            "n_label_time_excluded": int((v["eval"] & (v["tts"] >= VALIDATION_END_US)).sum()),
            "n_oov_below_support": int((oc == E.c("oov_cause", "BELOW_SUPPORT")).sum()),
            "n_oov_absent_from_train": int((oc == E.c("oov_cause", "ABSENT_FROM_TRAIN")).sum()),
            "n_activity_filtered": 0,
            "n_split_reassigned": int(v["g"].size),
            **{f"n_censor_{c}": int((cen == E.c("censor_class", c)).sum())
               for c in ("CEN_NO_NEXT", "CEN_LEGACY_BARRIER", "CEN_RIGHT_EDGE", "CEN_AMBIGUOUS_NEXT_BARRIER")},
            "n_invalid_source_quarantined": 0,
            "n_eval_rows": int(v["eval"].sum()), "n_rankable_rows": int(rankable.sum()),
            "n_oov_rows": int((v["eval"] & v["oov"]).sum()),
            "n_rankable_rows_zero_cohort_positive_class": int((rankable & (tpos == 0)).sum()),
            "by_subperiod": {s: {"decisions": int((sub == E.c("subperiod", s)).sum()),
                                 "eval_rows": int(((sub == E.c("subperiod", s)) & v["eval"]).sum()),
                                 "users": int(np.unique(euk[sub == E.c("subperiod", s)]).size)}
                             for s in ("VAL_A_20191116_20191120", "VAL_B_20191120_20191123")},
            "target_metadata_state_counts": {
                side: {state: {"rankable": int(((v[k] == E.c("meta_token_state", state)) & rankable).sum()),
                               "oov": int(((v[k] == E.c("meta_token_state", state)) & v["eval"] & v["oov"]).sum())}
                       for state in ("KNOWN", "MISSING", "OOV")}
                for side, k in (("category", "cat_state"), ("brand", "brand_state"), ("main_category", "main_state"))},
        }
        if kind == "cold":
            for sname in ("COLD_NO_TRAIN_EVENTS", "COLD_TRAIN_EVENTS_NO_LABEL"):
                ms = usr["cold_stratum"].codes == E.c("cold_stratum", sname)
                counts[f"stratum_{sname}"] = {"users_with_ge1_eval_row": int((ms & (ne >= 1)).sum()),
                                              "eval_rows": int(ne[ms].sum()), "rankable_rows": int(nr[ms].sum())}
        a = {}
        a["decision_ts_in_validation"] = bool(np.all((v["dts"] >= TRAIN_END_US) & (v["dts"] < VALIDATION_END_US)))
        a["eval_mask_equals_observed_status"] = bool(np.array_equal(
            v["eval"], v["status"] == E.c("target_status", "OBSERVED_IN_PARTITION")))
        a["censor_none_iff_eval"] = bool(np.array_equal(cen == E.c("censor_class", "NONE"), v["eval"]))
        a["rankable_definition"] = bool(np.array_equal(rankable, v["eval"] & (v["tc"] >= 0)))
        a["oov_implies_eval_and_unranked"] = bool(np.all(~v["oov"] | (v["eval"] & (v["tc"] == -1))))
        a["no_eval_target_at_or_after_validation_end"] = counts["n_label_time_excluded"] == 0
        a["subperiod_matches_old_split"] = bool(np.array_equal(
            sub == E.c("subperiod", "VAL_A_20191116_20191120"), v["split"] == E.c("old_split", "VALIDATION_SELECT"))
            and np.array_equal(sub == E.c("subperiod", "VAL_A_20191116_20191120"), v["dts"] < VS_END_US))
        a["oov_cause_consistent"] = not bool(np.any(bad_oc))
        a["decision_id_unique"] = bool(np.unique(v["decision_id"]).size == v["decision_id"].size)
        if kind == "warm":
            a["every_user_in_M01"] = bool(np.all((self.role[users] & COHORT) != 0))
        else:
            a["no_user_in_M01"] = bool(np.all((self.role[users] & COHORT) == 0))
            a["every_user_zero_eligible_examples"] = bool(np.all(self.u_nelig[users] == 0))
            a["cold_stratum_never_warm"] = bool(np.all(usr["cold_stratum"].codes != E.c("cold_stratum", "WARM")))
        pos = class_pos
        rk_classes = np.unique(v["tc"][rankable])
        cls = {"n_classes_with_ge1_cohort_train_positive": int((pos >= 1).sum()),
               "n_classes_with_ge5_cohort_train_positive": int((pos >= 5).sum()),
               "n_validation_target_classes_with_zero_cohort_train_positives": int((pos[rk_classes] == 0).sum())}
        counts.update(cls)
        exp = []
        e2e_users = int((ne >= 1).sum())
        if mid == "M02_VALIDATION_WARM":
            exp = [("n_eval_rows", counts["n_eval_rows"], 808610), ("n_users_with_ge1_eval_row", e2e_users, 62368),
                   ("n_rankable_rows", counts["n_rankable_rows"], 801762), ("n_oov_rows", counts["n_oov_rows"], 6848),
                   ("n_classes_with_ge1_cohort_train_positive", cls["n_classes_with_ge1_cohort_train_positive"], 133154),
                   ("n_validation_target_classes_with_zero_cohort_train_positives",
                    cls["n_validation_target_classes_with_zero_cohort_train_positives"], 5158)]
        elif mid == "M06_CALIB_VALIDATION_WARM":
            exp = [("n_eval_rows", counts["n_eval_rows"], 200985), ("n_users_with_ge1_eval_row", e2e_users, 15487),
                   ("n_rankable_rows", counts["n_rankable_rows"], 199276), ("n_oov_rows", counts["n_oov_rows"], 1709),
                   ("n_classes_with_ge1_cohort_train_positive", cls["n_classes_with_ge1_cohort_train_positive"], 90545),
                   ("n_validation_target_classes_with_zero_cohort_train_positives",
                    cls["n_validation_target_classes_with_zero_cohort_train_positives"], 7619)]
        else:
            s1, s2 = counts["stratum_COLD_NO_TRAIN_EVENTS"], counts["stratum_COLD_TRAIN_EVENTS_NO_LABEL"]
            exp = [("n_users_with_ge1_eval_row", e2e_users, 381295),
                   ("n_users_COLD_NO_TRAIN_EVENTS", s1["users_with_ge1_eval_row"], 299035),
                   ("n_eval_rows_COLD_NO_TRAIN_EVENTS", s1["eval_rows"], 2270516),
                   ("n_users_COLD_TRAIN_EVENTS_NO_LABEL", s2["users_with_ge1_eval_row"], 82260),
                   ("n_eval_rows_COLD_TRAIN_EVENTS_NO_LABEL", s2["eval_rows"], 587149)]
        e2e_idx = np.flatnonzero(ne >= 1)
        return {"manifest_id": mid, "tables": {"users": usr, "decisions": dec}, "assertions": a, "expected": exp,
                "parent": parent, "population_sha256": population_sha256(np.sort(self.u_key[users])),
                "n_users": int(n_u), "n_decisions": int(v["g"].size), "n_eval_rows": counts["n_eval_rows"],
                "n_rankable_rows": counts["n_rankable_rows"], "counts_by_cause": counts, "decision_table": "decisions",
                "e2e_user_digests": np.sort(self.u_dig[by_key][e2e_idx]), "users_g": users}

    # ---------------------------------------------------------------------------------- panels
    def _panel_manifest(self, mid: str, users: np.ndarray, label: str, parent: str) -> dict:
        E = self.E
        ev = self.rows_ev
        users = np.asarray(users, dtype=np.int64)
        sel = np.isin(ev["g"], users)
        e = {k: a[sel] for k, a in ev.items()}
        o = np.lexsort((e["decision_id"], e["dts"], e["g"]))
        e = {k: a[o] for k, a in e.items()}
        rank = self._dense_rank_by_digest(users)
        key_of = np.full(self.u_dig.size, -1, dtype=np.int64)
        key_of[users] = rank
        bounds = np.flatnonzero(np.concatenate([[True], e["g"][1:] != e["g"][:-1]])) if e["g"].size else np.zeros(0, int)
        ends = np.concatenate([bounds[1:], [e["g"].size]]).astype(np.int64)
        q_rows, s_rows = [], []
        f50 = np.zeros(self.u_dig.size, dtype=bool)
        f20 = np.zeros(self.u_dig.size, dtype=bool)
        is_val = (e["split"] == E.c("old_split", "VALIDATION_SELECT")) | (e["split"] == E.c("old_split",
                                                                                           "VALIDATION_CONFIRM"))
        cand_all = is_val & e["eval"]
        support_ok = e["has_obs"] & (e["tc"] >= 0) & (e["loss"] | e["eval"])
        ptc = {p: E.c("panel_type", p) for p in ("FIXED_50", "FIXED_20", "ALL_AVAILABLE")}
        for lo, hi in zip(bounds.tolist(), ends.tolist()):
            gu = int(e["g"][lo])
            cand = np.flatnonzero(cand_all[lo:hi])              # prior events available = index within user
            if cand.size == 0:
                raise BuildBlocked(f"{mid}: a panel user has no E2E VALIDATION decision")
            queries = []
            for p, hmin in FIXED_MIN.items():
                c = cand[cand >= hmin]
                if c.size:
                    (f50 if p == "FIXED_50" else f20)[gu] = True
                    queries += [(p, h, int(c[0])) for h in H_FIXED[p]]
            for h in H_ALL:
                c = cand[cand >= h]
                if c.size:
                    queries.append(("ALL_AVAILABLE", h, int(c[0])))
            for p, h, qi in queries:
                qa = lo + qi
                perm = np.arange(qa - h, qa)
                sup = perm[support_ok[perm] & (e["tts"][perm] < e["dts"][qa])]
                q_rows.append((ptc[p], h, key_of[gu], int(e["decision_id"][qa]), int(e["dts"][qa]), qi, h,
                               int(e["decision_id"][qa - h]) if h > 0 else -1, int(sup.size),
                               bool(e["tc"][qa] >= 0), int(e["tc"][qa])))
                for j in sup.tolist():
                    s_rows.append((ptc[p], h, key_of[gu], int(e["decision_id"][j]), int(e["tts"][j]), int(e["tc"][j])))
        qa = np.array(q_rows, dtype=np.int64).reshape(-1, 11)
        sa = np.array(s_rows, dtype=np.int64).reshape(-1, 6)
        qo = np.lexsort((qa[:, 2], qa[:, 1], qa[:, 0])) if qa.size else np.zeros(0, int)
        so = np.lexsort((sa[:, 3], sa[:, 2], sa[:, 1], sa[:, 0])) if sa.size else np.zeros(0, int)
        qa, sa = qa[qo], sa[so]
        n_u = users.size
        by_key = np.empty(n_u, dtype=np.int64)
        by_key[rank] = users
        if label == "WARM_TRUNCATION":
            strat = np.full(n_u, E.c("cold_stratum", "WARM"), dtype=np.int64)
        else:
            strat = np.where(self.u_ntrain[by_key] == 0, E.c("cold_stratum", "COLD_NO_TRAIN_EVENTS"),
                             E.c("cold_stratum", "COLD_TRAIN_EVENTS_NO_LABEL"))
        pu = {"eval_user_key": np.arange(n_u, dtype=np.int64), "client_key_hex": hex_of(self.u_dig[by_key]),
              "selection_key": self.u_key[by_key], "cold_stratum": EnumValues(strat, E.v["cold_stratum"]),
              "fixed50_eligible": f50[by_key], "fixed20_eligible": f20[by_key]}
        pq_t = {"panel_type": EnumValues(qa[:, 0], E.v["panel_type"]), "h": qa[:, 1].astype(np.int32),
                "eval_user_key": qa[:, 2], "query_decision_id": qa[:, 3], "query_decision_ts_us": qa[:, 4],
                "n_prior_events_available": qa[:, 5], "n_permitted_events": qa[:, 6],
                "permitted_first_decision_id": qa[:, 7], "n_support_labels": qa[:, 8],
                "query_rankable": qa[:, 9].astype(bool), "query_target_class": qa[:, 10]}
        ps_t = {"panel_type": EnumValues(sa[:, 0], E.v["panel_type"]), "h": sa[:, 1].astype(np.int32),
                "eval_user_key": sa[:, 2], "support_decision_id": sa[:, 3], "support_target_ts_us": sa[:, 4],
                "support_target_class": sa[:, 5]}
        # assertions (MANIFEST_SPEC M04)
        a = {}
        kq = (qa[:, 0] * 64 + qa[:, 1]) * (1 << 24) + qa[:, 2] if qa.size else np.zeros(0, np.int64)
        ks = (sa[:, 0] * 64 + sa[:, 1]) * (1 << 24) + sa[:, 2] if sa.size else np.zeros(0, np.int64)
        a["query_keys_unique"] = bool(np.unique(kq).size == kq.size)
        if sa.size:
            qord = np.argsort(kq)
            j = qord[np.clip(np.searchsorted(kq[qord], ks), 0, kq.size - 1)]
            a["every_support_row_has_its_query"] = bool(np.array_equal(kq[j], ks))
            a["support_strictly_before_query"] = bool(np.all(sa[:, 4] < qa[j, 4]))
            a["query_never_support"] = bool(np.all(sa[:, 3] != qa[j, 3]))
            a["support_count_matches"] = bool(np.array_equal(np.bincount(j, minlength=qa.shape[0]), qa[:, 8]))
        else:
            a["support_count_matches"] = bool(np.all(qa[:, 8] == 0)) if qa.size else True
        fixed_ok = True
        for p in ("FIXED_50", "FIXED_20"):
            m = qa[:, 0] == ptc[p]
            for u in np.unique(qa[m, 2]).tolist():
                mm = m & (qa[:, 2] == u)
                fixed_ok &= np.unique(qa[mm, 3]).size == 1 and np.array_equal(np.sort(qa[mm, 1]), np.array(H_FIXED[p]))
        a["fixed_query_identical_across_h"] = bool(fixed_ok)
        a["all_available_prior_ge_h"] = bool(np.all(qa[qa[:, 0] == ptc["ALL_AVAILABLE"], 5]
                                                    >= qa[qa[:, 0] == ptc["ALL_AVAILABLE"], 1]))
        a["one_query_per_user_per_subpanel_h"] = bool(np.unique(qa[:, :3], axis=0).shape[0] == qa.shape[0])
        a["n_permitted_equals_h"] = bool(np.array_equal(qa[:, 6], qa[:, 1]))
        a["h0_has_no_permitted_first"] = bool(np.all(qa[qa[:, 1] == 0, 7] == -1))
        a["h0_has_no_support"] = bool(np.all(qa[qa[:, 1] == 0, 8] == 0))
        a["query_decisions_are_e2e_validation"] = bool(np.all((qa[:, 4] >= TRAIN_END_US) & (qa[:, 4] < VALIDATION_END_US)))
        if label == "COLD_START":
            a["panel_is_first_16384_M03_by_key"] = bool(np.array_equal(np.sort(users), np.sort(self.cold_sorted[:N_PANEL])))
            a["every_user_cold"] = bool(np.all(strat != E.c("cold_stratum", "WARM")))
        elif label == "RECIPE_SELECTION":
            a["panel_is_M03_key_ranks_16384_32767"] = bool(np.array_equal(
                np.sort(users), np.sort(self.cold_sorted[N_PANEL:2 * N_PANEL])))
            a["disjoint_from_M04"] = not bool(np.intersect1d(users, self.m04).size)
        else:
            a["every_user_M01_and_M02_e2e"] = bool(np.all(((self.role[users] & COHORT) != 0)
                                                          & (self.u_neval[users] >= 1)))
            a["cold_stratum_warm"] = True
        elig = {}
        for p in ("FIXED_50", "FIXED_20", "ALL_AVAILABLE"):
            m = qa[:, 0] == ptc[p]
            for h in sorted(set(qa[m, 1].tolist())):
                mm = m & (qa[:, 1] == h)
                us = qa[mm, 2]
                elig[f"{p}_h{h}"] = {"users": int(us.size), "rankable_queries": int(qa[mm, 9].sum()),
                                     "support_labels": int(qa[mm, 8].sum()),
                                     "users_with_zero_support": int((qa[mm, 8] == 0).sum()),
                                     "by_stratum": {s: int(np.isin(us, np.flatnonzero(strat == E.c("cold_stratum", s))).sum())
                                                    for s in ("WARM", "COLD_NO_TRAIN_EVENTS",
                                                              "COLD_TRAIN_EVENTS_NO_LABEL")}}
        counts = {"panel_label": label, "n_users": int(n_u),
                  "stratum_mix": {s: int((strat == E.c("cold_stratum", s)).sum())
                                  for s in ("WARM", "COLD_NO_TRAIN_EVENTS", "COLD_TRAIN_EVENTS_NO_LABEL")},
                  "fixed50_eligible_users": int(pu["fixed50_eligible"].sum()),
                  "fixed20_eligible_users": int(pu["fixed20_eligible"].sum()),
                  "eligibility_by_subpanel_h": elig, "n_query_rows": int(qa.shape[0]), "n_support_rows": int(sa.shape[0]),
                  "note": "reported, never used to select (EVALUATION_CONTRACT_V4 E11.2)"}
        exp = [("n_users", int(n_u), 16384)]
        if label == "COLD_START":
            exp.append(("n_warm", int((strat == E.c("cold_stratum", "WARM")).sum()), 0))
        user_sets = {}
        for p in ("FIXED_50", "FIXED_20"):
            m = qa[:, 0] == ptc[p]
            u = np.unique(qa[m, 2])
            user_sets[p] = np.sort(self.u_dig[by_key][u]) if u.size else np.zeros(0, dtype="S32")
        for h in H_ALL:
            m = (qa[:, 0] == ptc["ALL_AVAILABLE"]) & (qa[:, 1] == h)
            u = np.unique(qa[m, 2])
            user_sets[f"ALL_AVAILABLE_h{h}"] = np.sort(self.u_dig[by_key][u]) if u.size else np.zeros(0, dtype="S32")
        return {"manifest_id": mid, "tables": {"panel_users": pu, "panel_queries": pq_t, "panel_support": ps_t},
                "assertions": a, "expected": exp, "parent": parent,
                "population_sha256": population_sha256(np.sort(self.u_key[users])), "n_users": int(n_u),
                "n_decisions": int(qa.shape[0]), "n_eval_rows": int(qa.shape[0]), "n_rankable_rows": int(qa[:, 9].sum()),
                "counts_by_cause": counts, "decision_table": None, "panel_label": label, "boot_user_sets": user_sets}

    def _cross_manifest_checks(self, man: dict) -> None:
        m1 = set(man["M01_COHORT_TRAIN"]["tables"]["clients"]["client_key_hex"].tolist())
        m2 = set(man["M02_VALIDATION_WARM"]["tables"]["users"]["client_key_hex"].tolist())
        m3 = set(man["M03_COLD_PANEL_VAL"]["tables"]["users"]["client_key_hex"].tolist())
        allk = set()
        n_rows = 0
        for mid, m in man.items():
            for tname in ("clients", "users", "panel_users"):
                if tname in m["tables"]:
                    ks = m["tables"][tname]["client_key_hex"].tolist()
                    if len(set(ks)) != len(ks):
                        raise BuildBlocked(f"{mid}.{tname}: duplicate client_key_hex")
                    allk |= set(ks)
                    n_rows += 1
        users_union = np.flatnonzero((self.role & (COHORT | COLD)) != 0)
        self.cross = {"M02_users_subset_of_M01": m2 <= m1, "M03_disjoint_from_M01": not (m3 & m1),
                      "distinct_client_keys_union": len(allk),
                      "distinct_users_union": int(users_union.size),
                      "all_client_digests_unique_in_ledger": bool(np.unique(self.u_dig).size == self.u_dig.size)}
        if not (self.cross["M02_users_subset_of_M01"] and self.cross["M03_disjoint_from_M01"]
                and self.cross["distinct_client_keys_union"] == self.cross["distinct_users_union"]):
            raise BuildBlocked(f"cross-manifest identity check failed: {self.cross}")

    # --------------------------------------------------------------------------------------- write
    def _columns(self, tbl: dict) -> list:
        cols = []
        for name, vals in tbl.items():
            dtype, enum = FIELD_TYPES[name]
            cols.append((name, dtype, vals))
        return cols

    def _check_no_raw_ids(self, mid: str, tname: str, tbl: dict) -> None:
        forb = set(self.spec["global_rules"]["no_raw_ids"]["forbidden_columns"])
        for name, vals in tbl.items():
            if name in forb:
                raise BuildBlocked(f"{mid}.{tname}: forbidden identifier column {name}")
            dtype, enum = FIELD_TYPES.get(name, (None, None))
            if dtype is None:
                raise BuildBlocked(f"{mid}.{tname}: unregistered column {name}")
            if dtype == "string":
                if enum == "HEX":
                    if not all(HEX_RE.match(x) for x in vals.tolist()):
                        raise BuildBlocked(f"{mid}.{tname}.{name}: a value is not a 64-char lowercase hex key")
                elif not isinstance(vals, EnumValues) or tuple(vals.vocab) != tuple(self.E.v[enum]):
                    raise BuildBlocked(f"{mid}.{tname}.{name}: string column is neither enum-typed nor client_key_hex")

    def _arrow(self, tbl: dict) -> pa.Table:
        arrays, names = [], []
        for name, vals in tbl.items():
            dtype, enum = FIELD_TYPES[name]
            if isinstance(vals, EnumValues):
                arr = pc.take(pa.array(list(vals.vocab), type=pa.string()), pa.array(vals.codes, type=pa.int64()))
            elif dtype == "string":
                arr = pa.array(list(vals), type=pa.string())
            else:
                arr = pa.array(np.asarray(vals), type={"int64": pa.int64(), "uint64": pa.uint64(), "int32": pa.int32(),
                                                       "int8": pa.int8(), "bool": pa.bool_()}[dtype])
            arrays.append(arr)
            names.append(name)
        return pa.Table.from_arrays(arrays, names=names)

    @staticmethod
    def _reread_columns(t: pa.Table) -> list:
        return reread_columns(t)

    def write(self, man: dict) -> dict:
        t0 = time.time()
        out_dir = "manifests"
        hashes = {}
        for mid in PRE_LOCK_IDS:
            m = man[mid]
            self._assert_manifest(m)
            tables_meta, files = [], {}
            for tname, tbl in m["tables"].items():
                self._check_no_raw_ids(mid, tname, tbl)
                content = table_content_sha256(self._columns(tbl))
                buf = io.BytesIO()
                pq.write_table(self._arrow(tbl), buf, compression="zstd", compression_level=9)
                rel = f"{out_dir}/{mid}.{tname}.parquet"
                p = atomic_write_bytes(self.out, rel, buf.getvalue())
                back = pq.ParquetFile(p).read()
                if table_content_sha256(self._reread_columns(back)) != content:
                    raise BuildBlocked(f"{mid}.{tname}: content hash changed after the Parquet round trip")
                fsha = sha256_file(p)
                n_rows = len(next(iter(tbl.values())))
                tables_meta.append({"table": tname, "row_count": int(n_rows), "table_content_sha256": content,
                                    "file_sha256": fsha})
                files[rel] = fsha
            header = self._header(m, tables_meta)
            hrel = f"{out_dir}/{mid}.header.json"
            hp = atomic_write_json(self.out, hrel, header)
            files[hrel] = sha256_file(hp)
            for rel, fsha in files.items():
                if sha256_file(self.out / rel) != fsha:
                    raise BuildBlocked(f"{rel}: file changed before the marker was written")
            write_marker(self.out, f"{out_dir}/{mid}.OK", files)
            hashes[mid] = {"tables": {x["table"]: {"row_count": x["row_count"],
                                                   "table_content_sha256": x["table_content_sha256"],
                                                   "file_sha256": x["file_sha256"]} for x in tables_meta},
                           "header_file_sha256": files[hrel],
                           "header_content_sha256": header["header_content_sha256"],
                           **{k: header[k] for k in ("ordered_decision_id_sha256", "eval_ordered_decision_id_sha256",
                                                     "rankable_ordered_decision_id_sha256", "population_sha256",
                                                     "n_users", "n_decisions", "n_eval_rows", "n_rankable_rows")}}
            self._say(f"wrote {mid}: " + ", ".join(f"{x['table']}={x['row_count']}" for x in tables_meta))
        self.timings["write_s"] = round(time.time() - t0, 1)
        return hashes

    def _assert_manifest(self, m: dict) -> None:
        bad = [k for k, v in m["assertions"].items() if v is not True]
        if bad:
            raise BuildBlocked(f"{m['manifest_id']}: assertions failed {bad}")
        if self.expect:
            miss = [(f, o, e) for f, o, e in m["expected"] if o != e]
            if miss:
                raise BuildBlocked(f"{m['manifest_id']}: expected counts mismatch {miss}")

    def _header(self, m: dict, tables_meta: list) -> dict:
        mid = m["manifest_id"]
        spec_m = next(x for x in self.spec["manifests"] if x["manifest_id"] == mid)
        src = {e["field"]: e["source"] for e in (spec_m.get("expected_counts") or []) if isinstance(e, dict)}
        dt = m["decision_table"]
        if dt:
            dec = m["tables"][dt]
            ids = dec["decision_id"]
            ev = dec["eval_mask"] if "eval_mask" in dec else np.zeros(ids.size, dtype=bool)
            rk = dec["rankable"] if "rankable" in dec else np.zeros(ids.size, dtype=bool)
            if mid in ("M01_COHORT_TRAIN", "M05_CALIB_TRAIN"):
                ordered = ordered_ids_sha256(ids)
            else:
                ordered = ordered_ids_sha256(ids)
            eval_h, rank_h = ordered_ids_sha256(ids[ev]), ordered_ids_sha256(ids[rk])
        else:
            q = m["tables"]["panel_queries"]
            ordered = ordered_ids_sha256(q["query_decision_id"])
            eval_h = ordered
            rank_h = ordered_ids_sha256(q["query_decision_id"][q["query_rankable"]])
        observed_enums = {}
        for tname, tbl in m["tables"].items():
            for name, vals in tbl.items():
                if isinstance(vals, EnumValues):
                    observed_enums[f"{tname}.{name}"] = sorted({vals.vocab[c] for c in np.unique(vals.codes).tolist()})
        h = {
            "manifest_id": mid, "spec_id": "MANIFEST_SPEC_V4", "spec_version": self.spec["spec_version"],
            "status": "BUILT_PRE_LOCK" if self.full else "PARTIAL_TEST_BUILD", "build_utc": utc_now(),
            "builder_code_sha256": self.code_sha, "dataset_version_hash": VERSION_HASH,
            "dataset_code_hash": DATASET_CODE_HASH, "feature_schema_version": SCHEMA_VERSION,
            "feature_schema_self_sha256": SCHEMA_SELF_SHA256,
            "data_protocol_lock_sha256": "PENDING until DATA_PROTOCOL_LOCK; the header is re-issued with it before training",
            "population_lock_sha256": "PENDING until POPULATION_PLAN",
            "salt": SALT_STR, "catalogue_K": int(self.K), "candidate_order_hash": self.candidate_order_hash,
            "parent_manifests": [m["parent"]] if m["parent"] else [], "tables": tables_meta,
            "ordered_decision_id_sha256": ordered, "eval_ordered_decision_id_sha256": eval_h,
            "rankable_ordered_decision_id_sha256": rank_h, "population_sha256": m["population_sha256"],
            "n_users": int(m["n_users"]), "n_decisions": int(m["n_decisions"]), "n_eval_rows": int(m["n_eval_rows"]),
            "n_rankable_rows": int(m["n_rankable_rows"]), "counts_by_cause": m["counts_by_cause"],
            "expected_counts_check": [{"field": f, "expected": e, "expected_source": src.get(f, "MANIFEST_SPEC_V4"),
                                       "observed": o, "match": o == e} for f, o, e in m["expected"]],
            "assertions": m["assertions"], "observed_enum_values": observed_enums,
            "previously_seen_evaluation_period": True, "all_methods_lock_sha256": None,
            "test_exposure_register_entry_id": None,
        }
        if "panel_label" in m:
            h["panel_label"] = m["panel_label"]
        if mid == "M01_COHORT_TRAIN":
            h["derived"] = {"central_canonical_order": "table central_order: row index into decisions, ascending "
                                                       "decision_id", "central_order_ordered_decision_id_sha256":
                            ordered_ids_sha256(np.sort(m["tables"]["decisions"]["decision_id"]))}
        if mid == "M06_CALIB_VALIDATION_WARM":
            h["spec_conflict_note"] = ("MANIFEST_SPEC M06 asserts every field equals M02 except eval_user_key, but its "
                                       "field_dictionary counts target_class_cohort_train_positives over M05 for "
                                       "calibration manifests; this build follows the field definition, so that field "
                                       "(and eval_user_key) differ from M02 by design")
        stable = {k: v for k, v in h.items() if k not in ("build_utc", "tables")}
        stable["tables"] = [{k: x[k] for k in ("table", "row_count", "table_content_sha256")} for x in tables_meta]
        h["header_content_sha256"] = sha256_bytes(canonical_json_bytes(stable))
        return h

    # ------------------------------------------------------------------------------------ bootstrap
    def bootstrap(self, man: dict) -> dict:
        t0 = time.time()
        sets = {"BOOT_M02_VALIDATION_WARM": ("M02_VALIDATION_WARM", man["M02_VALIDATION_WARM"]["e2e_user_digests"]),
                "BOOT_M03_COLD_PANEL_VAL": ("M03_COLD_PANEL_VAL", man["M03_COLD_PANEL_VAL"]["e2e_user_digests"]),
                "BOOT_M06_CALIB_VALIDATION_WARM": ("M06_CALIB_VALIDATION_WARM",
                                                   man["M06_CALIB_VALIDATION_WARM"]["e2e_user_digests"])}
        for mid in ("M04_COLD_H_PANEL", "M04W_WARM_TRUNCATION_PANEL"):
            for sub, dg in man[mid]["boot_user_sets"].items():
                sets[f"BOOT_{mid}_{sub}"] = (mid, dg)
        w_cache: dict = {}
        out = {}
        for bid, (mid, dg) in sets.items():
            n = int(dg.size)
            if n not in w_cache:
                w_cache[n] = w_sha256(n)
            rec = {"boot_id": bid, "manifest_id": mid, "user_set_sha256": user_set_sha256(dg), "n_users": n,
                   "n_resamples": BOOT_R, "seed": BOOT_SEED, "numpy_version": np.__version__,
                   "W_sha256": w_cache[n], "algorithm": self.spec["bootstrap"]["algorithm"],
                   "stored": False, "regeneration_rule": self.spec["bootstrap"]["regeneration_rule"]}
            atomic_write_json(self.out, f"manifests/bootstrap/{bid}.json", rec)
            out[bid] = {k: rec[k] for k in ("manifest_id", "user_set_sha256", "n_users", "W_sha256")}
        self.timings["bootstrap_s"] = round(time.time() - t0, 1)
        return out

    # ------------------------------------------------------------------------------------ receipts
    def receipts(self, man: dict, hashes: dict, boots: dict) -> dict:
        rec = {}
        for k, (val, src) in REF.FULL_POPULATION.items():
            got = self.recon.get(k)
            rec[k] = {"expected": val, "observed": got, "match": got == val, "source": src}
        for k, (val, src) in REF.POPULATION.items():
            got = self.pop_recon.get(k)
            if k.startswith("cohort262144.per_client"):
                cnt = man["M01_COHORT_TRAIN"]["tables"]["clients"]["row_count"]
                got = {"min": int(cnt.min()), "p50": float(np.percentile(cnt, 50, method="linear")),
                       "max": int(cnt.max())}[k.rsplit("_", 1)[1]]
            elif k.startswith("cohort262144.") and k.split(".")[1] in ("rankable", "oov"):
                got = man["M02_VALIDATION_WARM"]["counts_by_cause"]["n_rankable_rows" if k.endswith("rankable")
                                                                    else "n_oov_rows"]
            elif k.startswith("cohort65536.") and k.split(".")[1] in ("rankable", "oov"):
                got = man["M06_CALIB_VALIDATION_WARM"]["counts_by_cause"]["n_rankable_rows" if k.endswith("rankable")
                                                                          else "n_oov_rows"]
            elif k.endswith("classes_ge1_positive") or k.endswith("classes_ge5_positive") or \
                    k.endswith("val_classes_zero_positive"):
                mm = "M02_VALIDATION_WARM" if k.startswith("cohort262144") else "M06_CALIB_VALIDATION_WARM"
                c = man[mm]["counts_by_cause"]
                got = c[{"classes_ge1_positive": "n_classes_with_ge1_cohort_train_positive",
                         "classes_ge5_positive": "n_classes_with_ge5_cohort_train_positive",
                         "val_classes_zero_positive": "n_validation_target_classes_with_zero_cohort_train_positives"}[
                    k.split(".", 1)[1]]]
            rec[k] = {"expected": val, "observed": got, "match": got == val, "source": src}
        checks = {k: v for k, v in self.recon.items() if k.startswith("check.")}
        all_ok = all(v["match"] for v in rec.values()) and all(v == 0 for v in checks.values())
        recon = {"receipt": "CENSUS_RECONCILIATION", "full_build": self.full, "all_match": all_ok,
                 "reconciliation": rec, "integrity_checks_zero_expected": checks,
                 "clients_parquet_rows": self.recon.get("clients_parquet_rows"),
                 "support_check": self.support_check, "target_metadata_join": self.meta_checks,
                 "cross_manifest": self.cross, "no_test_rows_read": True,
                 "test_numbers_reported": "none (EVALUATION_CONTRACT_V4 section 10.4)"}
        atomic_write_json(self.out, "receipts/CENSUS_RECONCILIATION.json", recon)
        for mid in PRE_LOCK_IDS:
            m = man[mid]
            atomic_write_json(self.out, f"receipts/{mid}.counts.json", {
                "manifest_id": mid, "n_users": m["n_users"], "n_decisions": m["n_decisions"],
                "n_eval_rows": m["n_eval_rows"], "n_rankable_rows": m["n_rankable_rows"],
                "population_sha256": m["population_sha256"], "counts_by_cause": m["counts_by_cause"],
                "expected_counts_check": [{"field": f, "expected": e, "observed": o, "match": o == e}
                                          for f, o, e in m["expected"]], "assertions": m["assertions"]})
        mh = {"receipt": "MANIFEST_HASHES", "spec_version": self.spec["spec_version"],
              "builder_code_sha256": self.code_sha, "candidate_order_hash": self.candidate_order_hash,
              "feature_schema_self_sha256": SCHEMA_SELF_SHA256, "dataset_version_hash": VERSION_HASH,
              "full_build": self.full, "manifests": hashes, "bootstrap": boots,
              "cohort_key_hashes": {k: self.pop_hash[k] for k in self.pop_hash if k != "universe_size"}}
        atomic_write_json(self.out, "receipts/MANIFEST_HASHES.json", mh)
        if self.full and not all_ok:
            raise BuildBlocked("census reconciliation failed; see receipts/CENSUS_RECONCILIATION.json")
        return {"reconciliation_all_match": all_ok}

    # ------------------------------------------------------------------------------------ driver
    def run(self) -> dict:
        disk0 = shutil.disk_usage(self.out.parent if self.out.parent.exists() else Path.cwd())
        self.pass_support()
        self.pass_aggregate()
        self.select_populations()
        self.pass_extract()
        man = self.assemble()
        hashes = self.write(man)
        boots = self.bootstrap(man)
        res = self.receipts(man, hashes, boots)
        disk1 = shutil.disk_usage(self.out.parent if self.out.parent.exists() else Path.cwd())
        size = sum(p.stat().st_size for p in (self.out / "manifests").rglob("*") if p.is_file())
        build = {"receipt": "BUILD_RECEIPT", "utc": utc_now(), "full_build": self.full, "buckets": len(self.buckets),
                 "python": sys.version.split()[0], "numpy": np.__version__, "pyarrow": pa.__version__,
                 "platform": platform.platform(), "builder_code_sha256": self.code_sha, "timings_s": self.timings,
                 "wall_s": round(time.time() - self.t0, 1), "manifest_bytes_total": int(size),
                 "disk_free_bytes_before": int(disk0.free), "disk_free_bytes_after": int(disk1.free),
                 "test_gate": "no TEST manifest built; scans filtered to non-TEST splits", **res}
        atomic_write_json(self.out, "receipts/BUILD_RECEIPT.json", build)
        self._say(f"done; manifests {size / 1e6:.1f} MB; reconciliation all_match={res['reconciliation_all_match']}")
        return build


def reread_columns(t: pa.Table) -> list:
    """(name, dtype, values) of a re-read manifest table for table_content_sha256. String columns go through Arrow
    dictionary encoding (no per-row Python strings), which yields the identical per-row payload."""
    cols = []
    for name in t.column_names:
        dtype = FIELD_TYPES[name][0]
        col = t[name]
        if dtype == "string":
            d = col.combine_chunks().dictionary_encode() if col.num_chunks else pa.array([], pa.string()).dictionary_encode()
            vals = EnumValues(d.indices.to_numpy(zero_copy_only=False).astype(np.int64), d.dictionary.to_pylist())
        else:
            vals = col.to_numpy(zero_copy_only=False)
        cols.append((name, dtype, vals))
    return cols


def w_sha256(n_users: int, seed: int = BOOT_SEED, n_resamples: int = BOOT_R) -> str:
    """sha256 of W (int32 LE, row-major [n_resamples, n_users]) generated row by row (MANIFEST_SPEC bootstrap)."""
    import hashlib
    rng = np.random.default_rng(seed)
    h = hashlib.sha256()
    for _ in range(n_resamples):
        row = rng.integers(0, n_users, size=n_users) if n_users else np.zeros(0, dtype=np.int64)
        h.update(np.ascontiguousarray(row.astype("<i4")).tobytes())
    return h.hexdigest()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build the V4 pre-TEST manifests (DATA-04)")
    ap.add_argument("--buckets", type=str, default=None, help="comma list for a partial test build")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--manifest", type=str, default=None, help="refused for any TEST manifest id")
    a = ap.parse_args(argv)
    if a.manifest:
        refuse_test_manifest(a.manifest)
    b = [int(x) for x in a.buckets.split(",")] if a.buckets else None
    ManifestBuilder(out_root=Path(a.out) if a.out else None, buckets=b).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
