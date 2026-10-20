"""v4data09.d9_build (owner: Data engineer, DATA-09) - the RETRO_TEST manifests of the single TEST look, built ONLY under
the TEST gate (d9_gate.check_test_gate: ALL_METHODS_LOCK.json + its MANIFEST_BUILD TEST_EXPOSURE_REGISTER entry).

  MAIN_RETRO_TEST_5_100_131072    M07: every RETRO_TEST decision (ledger old_split TEST) of the MAIN_TRAIN client set,
                                  censored rows included                                   role MAIN_TEST
  UNEW_HOLDOUT_RETRO_TEST_32768   M07 over the UNEW_HOLDOUT_SUPPORT_32768 users           role UNSEEN_HOLDOUT_TEST
  SPARSE_PANEL_16384_RETRO_TEST   M07 over the 16,384 registered SPARSE panel users       role SPARSE_PANEL_TEST
  HEAVY_PANEL_16384_RETRO_TEST    M07 over the 16,384 registered HEAVY panel users        role HEAVY_PANEL_TEST
  TRUE_COLD_RETRO_TEST            M08: every RETRO_TEST decision of every user with >= 1 RETRO_TEST eval row and zero
                                  TRAIN-eligible examples (schema of TRUE_COLD_VALIDATION)  role TRUE_COLD_TEST
The support manifests are the registered DATA-06 ones, unchanged (not rebuilt, not rewritten).

How (D9Builder = v4data06.D6Builder = the LOCKED v4data.manifests.ManifestBuilder, subclassed, nothing edited):
  1. prepare(): the locked pass_support + pass_aggregate (non-TEST reads, exactly as DATA-06), then
     v4data06.d6_build.compute_bands (unedited) - the populations are therefore DATA-06's; a full build also checks every
     band's population_sha256_selection_key_v2 against REGISTRY_V2's entries;
  2. pass_test_extract(): ONE gated scan per bucket through d9_read.read_with_test (split IN TRAIN/VS/VC/TEST), rows
     typed by the locked _row_arrays; keeps the TEST rows of the band users and of every zero-eligible user, the TEST
     censor classes (d9_common.CENSOR_NOTE), the target metadata states (the locked _target_meta logic over a gated
     feature scan; grace-matured targets NOT_APPLICABLE), the class positives of each band's own TRAIN rows (the M01
     eligibility of _row_arrays) and the full-population RETRO_TEST structural counters (compared with the G0 census
     AFTER the build by d9_finalize, counts only);
  3. each manifest = the locked _eval_manifest (M02 / M03 shape; eval_user_key, rankable, oov_cause, class positives,
     users table exactly MAIN_VALIDATION's code) followed by the M07 column changes (v4_split RETRO_TEST, subperiod NONE,
     + grace_matured) and the RETRO_TEST assertions / counts;
  4. gated encode + header (M07 header_extra from the GateRecord) + gated write, .OK marker last; receipts.
TEST-only users (no TRAIN / VALIDATION row, so absent from the locked aggregate) are appended to the builder's user
arrays with zero counts; they can only enter TRUE_COLD_RETRO_TEST.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import ppsi_env  # noqa: E402  (import isolation first, T43)

import json  # noqa: E402

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402

from census_v2 import core as C2  # noqa: E402
from v4data.common import (GRACE_END_US, TEST_END_US, VALIDATION_END_US, EnumValues, builder_code_sha256,  # noqa: E402
                           digest_rows, digests_of, population_sha256, selection_keys_of, utc_now, v4_split_of)
from v4data.manifests import (COHORT as BIT_COHORT, FEATURES_B, HEX_RE, LEDGER_B, BuildBlocked, _np)  # noqa: E402
from v4data06.d6_build import D6Builder, band_integrity_checks, compute_bands  # noqa: E402
from v4data09 import d9_common as D9  # noqa: E402
from v4data09.d9_gate import GateRecord, TestGateRefused, check_test_gate, revalidate  # noqa: E402
from v4data09.d9_read import read_with_test  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

V4 = D9.V4
REGISTRY_V2 = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2.json"
BAND_SUPPORT_ID = {"MAIN": D9.DC.MAIN_TRAIN_ID, "U_NEW_TEST": D9.DC.UNEW_TEST_TRAIN_ID,
                   "SPARSE": D9.DC.SPARSE_TRAIN_ID, "HEAVY": D9.DC.HEAVY_TRAIN_ID}
STATUSES = ("OBSERVED_IN_PARTITION", "NO_OBSERVED_NEXT", "AMBIGUOUS_BOUNDARY", "OBSERVED_MATURED_IN_GRACE",
            "BOUNDARY_UNRESOLVED")
KEEP_LOCKED_ASSERTIONS = ("censor_none_iff_eval", "rankable_definition", "oov_implies_eval_and_unranked",
                          "oov_cause_consistent", "decision_id_unique", "every_user_in_M01", "no_user_in_M01",
                          "every_user_zero_eligible_examples", "cold_stratum_never_warm")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


class D9Builder(D6Builder):
    """D6Builder with the TEST gate checked before anything is opened, the extended field dictionary for the three
    locked table helpers (subclass overrides only), and the gated RETRO_TEST extraction / assembly below."""

    def __init__(self, *a, gate: GateRecord, **kw):
        revalidate(gate)                                  # first: before the dataset's VERSION.json is opened
        self.gate = gate
        super().__init__(*a, **kw)

    # ---------------------------------------------------------------- the locked helpers over FIELD_TYPES_D9
    def _columns(self, tbl: dict) -> list:
        return [(name, D9.FIELD_TYPES_D9[name][0], vals) for name, vals in tbl.items()]

    def _check_no_raw_ids(self, mid: str, tname: str, tbl: dict) -> None:
        forb = set(self.spec["global_rules"]["no_raw_ids"]["forbidden_columns"])
        for name, vals in tbl.items():
            if name in forb:
                raise BuildBlocked(f"{mid}.{tname}: forbidden identifier column {name}")
            dtype, enum = D9.FIELD_TYPES_D9.get(name, (None, None))
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
            dtype, enum = D9.FIELD_TYPES_D9[name]
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

    # ---------------------------------------------------------------- TEST censoring (EVALUATION_CONTRACT §2.2, M07)
    def _censor_test(self, r: dict, test: np.ndarray) -> np.ndarray:
        E = self.E
        st, ns = r["status"], r["nsplit"]
        out = np.full(st.shape, -1, dtype=np.int64)
        bu = st == E.c("target_status", "BOUNDARY_UNRESOLVED")
        ab = st == E.c("target_status", "AMBIGUOUS_BOUNDARY")
        nn = st == E.c("target_status", "NO_OBSERVED_NEXT")
        obs = (st == E.c("target_status", "OBSERVED_IN_PARTITION")) | \
              (st == E.c("target_status", "OBSERVED_MATURED_IN_GRACE"))
        legacy = bu & r["vs"] & (ns == E.c("next_split", "VALIDATION_CONFIRM"))
        right = bu & ((ns == E.c("next_split", "TEST")) | (ns == E.c("next_split", "LABEL_GRACE")))
        sel = [(r["eval"], "NONE"), (legacy, "CEN_LEGACY_BARRIER"), (right, "CEN_RIGHT_EDGE"),
               (ab, "CEN_AMBIGUOUS_NEXT_BARRIER"), (nn, "CEN_NO_NEXT")]
        for m, name in sel:                                             # precedence = list order (E2.1)
            out[test & m & (out < 0)] = E.c("censor_class", name)
        if np.any(test & (out < 0)):
            raise BuildBlocked("a RETRO_TEST row matches no censor rule: the build stops (E2.1)")
        if np.any(test & (r["eval"] != obs)):
            raise BuildBlocked("eval_mask != target_status in {OBSERVED_IN_PARTITION, OBSERVED_MATURED_IN_GRACE} on a "
                               "RETRO_TEST row")
        if np.any(test & (r["eval"] != r["has_obs"])):
            raise BuildBlocked("eval_mask != has_observed_target on a RETRO_TEST row")
        if np.any(~test & (st == E.c("target_status", "OBSERVED_MATURED_IN_GRACE"))):
            raise BuildBlocked("OBSERVED_MATURED_IN_GRACE outside the TEST split (LABEL_GRACE matures TEST only, E1.9)")
        return out

    # ---------------------------------------------------------------- target metadata (locked _target_meta, gated)
    def _target_meta_test(self, b: int, t: pa.Table, r: dict, ui: np.ndarray, uniq, rows: np.ndarray,
                          grace: np.ndarray) -> dict:
        """v4data.manifests.ManifestBuilder._target_meta over a GATED feature scan (TEST feature rows included), with
        one change: a grace-matured target has no model_ready row (LABEL_GRACE is excluded from the features by
        construction), so it stays NOT_APPLICABLE and is not joined (counted in meta_checks)."""
        E = self.E
        na = E.c("meta_token_state", "NOT_APPLICABLE")
        out = {k: np.full(rows.size, na, dtype=np.int64) for k in ("cat_state", "brand_state", "main_state")}
        self.meta_checks["grace_targets_not_joined"] += int((r["has_obs"][rows] & grace[rows]).sum())
        need = rows[r["has_obs"][rows] & ~grace[rows]]
        if need.size == 0:
            return out
        F = read_with_test(self._features(b), FEATURES_B, self.gate)
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
            raise BuildBlocked("an observed in-partition RETRO_TEST target has no next-representative feature row")
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

    # ---------------------------------------------------------------- the gated extraction pass
    def pass_test_extract(self, band_sets: dict) -> None:
        revalidate(self.gate)
        t0 = time.time()
        E = self.E
        n0 = int(self.u_dig.size)
        c_test = E.c("old_split", "TEST")
        bits = {name: 1 << i for i, name in enumerate(band_sets)}
        member = np.zeros(n0, dtype=np.int64)
        for name, idx in band_sets.items():
            member[np.asarray(idx, dtype=np.int64)] |= bits[name]
        pos = {name: np.zeros(self.K, dtype=np.int64) for name in band_sets}
        agg_keys = ("decisions", "has_observed_target", "loss_mask_true", "eval_mask_true", "target_oov_true",
                    "target_class_ge0", "distinct_users")
        cnt = {"per_old_split_TEST": dict.fromkeys(agg_keys, 0), "v4_retro_test": dict.fromkeys(agg_keys, 0)}
        for k in cnt:
            cnt[k]["target_status_counts"] = dict.fromkeys(STATUSES, 0)
        cnt["test_users"] = {"total_with_ge1_eval_row": 0, "warm_ge1_eligible_train_example": 0,
                             "cold_zero_eligible_train_examples": 0}
        self.meta_checks = {"target_rows_joined": 0, "target_pid_mismatch": 0, "target_ts_mismatch": 0,
                            "grace_targets_not_joined": 0}
        extra_dig, extra_bkt, parts = [], [], []
        n_extra = 0
        for b in self.buckets:
            t = read_with_test(self._ledger(b), LEDGER_B, self.gate)
            if t["user_id"].null_count:
                raise BuildBlocked("NULL user_id in the ledger")
            r = self._row_arrays(t)                       # locked typing / checks (VALIDATION censor rules untouched)
            uniq = pc.unique(t["user_id"])
            ui = _np(pc.index_in(t["user_id"], value_set=uniq), np.int64)
            dig_u = digests_of(uniq.to_pylist())
            bidx = np.flatnonzero(self.u_bucket == b).astype(np.int64)
            o = np.argsort(self.u_dig[bidx], kind="stable")
            sd, si = self.u_dig[bidx][o], bidx[o]
            if sd.size:
                p = np.clip(np.searchsorted(sd, dig_u), 0, sd.size - 1)
                hit = sd[p] == dig_u
                g_u = np.where(hit, si[p], -1).astype(np.int64)
            else:
                hit = np.zeros(dig_u.size, dtype=bool)
                g_u = np.full(dig_u.size, -1, dtype=np.int64)
            test = r["split"] == c_test
            new = np.flatnonzero(~hit)
            if new.size:
                non_test_users = np.unique(ui[~test])
                if np.intersect1d(new, non_test_users).size:
                    raise BuildBlocked(f"bucket {b}: a user absent from the non-TEST aggregate has a non-TEST row")
                g_u[new] = n0 + n_extra + np.arange(new.size, dtype=np.int64)
                n_extra += int(new.size)
                extra_dig.append(dig_u[new])
                extra_bkt.append(np.full(new.size, b, dtype=np.int16))
            g = g_u[ui]
            known = g < n0
            gm = np.zeros(g.size, dtype=np.int64)
            gm[known] = member[g[known]]
            nelig = np.zeros(g.size, dtype=np.int64)
            nelig[known] = self.u_nelig[g[known]]
            for name, bit in bits.items():                # class positives over each band's own M01-eligible rows
                sel = r["elig"] & ((gm & bit) != 0)
                pos[name] += np.bincount(r["tc"][sel], minlength=self.K).astype(np.int64)
            # full-population RETRO_TEST structural counters (the G0 census definitions, pass2_census.py)
            for key, m in (("per_old_split_TEST", test), ("v4_retro_test", v4_split_of(r["dts"]) == 2)):
                c = cnt[key]
                c["decisions"] += int(m.sum())
                c["has_observed_target"] += int((m & r["has_obs"]).sum())
                c["loss_mask_true"] += int((m & r["loss"]).sum())
                c["eval_mask_true"] += int((m & r["eval"]).sum())
                c["target_oov_true"] += int((m & r["oov"]).sum())
                c["target_class_ge0"] += int((m & (r["tc"] >= 0)).sum())
                c["distinct_users"] += int(np.unique(ui[m]).size)
                for s in STATUSES:
                    c["target_status_counts"][s] += int((m & (r["status"] == E.c("target_status", s))).sum())
            te_u = np.bincount(ui[test & r["eval"]], minlength=len(uniq))
            nelig_u = np.zeros(len(uniq), dtype=np.int64)
            nelig_u[hit] = self.u_nelig[g_u[hit]]
            cnt["test_users"]["total_with_ge1_eval_row"] += int((te_u > 0).sum())
            cnt["test_users"]["warm_ge1_eligible_train_example"] += int(((te_u > 0) & (nelig_u > 0)).sum())
            cnt["test_users"]["cold_zero_eligible_train_examples"] += int(((te_u > 0) & (nelig_u == 0)).sum())
            # the TEST rows of band users and of every zero-eligible user (the M08 candidates)
            keep = test & ((gm != 0) | (nelig == 0))
            rows = np.flatnonzero(keep)
            sup_row = np.full(r["tc"].size, -1, dtype=np.int64)
            nr = rows[r["has_obs"][rows]]
            if nr.size:
                pids = t["target_product_id"].take(pa.array(nr)).to_pylist()
                sup_row[nr] = np.fromiter((self.support.get(x, 0) for x in pids), dtype=np.int64, count=len(pids))
            censor = self._censor_test(r, test)
            grace = r["status"] == E.c("target_status", "OBSERVED_MATURED_IN_GRACE")
            meta = self._target_meta_test(b, t, r, ui, uniq, rows, grace)
            did = _np(t["decision_id"], np.int64)
            sup = np.where(r["has_obs"][rows], sup_row[rows], -1).astype(np.int64)
            parts.append({"g": g[rows], "decision_id": did[rows], "dts": r["dts"][rows], "tts": r["tts"][rows],
                          "split": r["split"][rows], "status": r["status"][rows], "nsplit": r["nsplit"][rows],
                          "nblock": r["nblock"][rows], "eval": r["eval"][rows], "tc": r["tc"][rows],
                          "oov": r["oov"][rows], "has_obs": r["has_obs"][rows], "support": sup,
                          "censor": censor[rows], "grace": grace[rows], **meta})
            del t, r
        if n_extra:
            ed = np.concatenate(extra_dig)
            z = np.zeros(n_extra, dtype=np.int64)
            self.u_dig = np.concatenate([self.u_dig, ed])
            self.u_key = np.concatenate([self.u_key, selection_keys_of(ed)])
            self.u_ntrain, self.u_nelig, self.u_nval = (np.concatenate([a, z]) for a in
                                                        (self.u_ntrain, self.u_nelig, self.u_nval))
            self.u_neval, self.u_nrank = (np.concatenate([a, z]) for a in (self.u_neval, self.u_nrank))
            self.u_bucket = np.concatenate([self.u_bucket, np.concatenate(extra_bkt)])
            if np.unique(self.u_dig).size != self.u_dig.size:
                raise BuildBlocked("client_key_hex collision after appending the TEST-only users (fatal, C1.2)")
        cat = lambda ps: {k: np.concatenate([p[k] for p in ps]) for k in ps[0]}  # noqa: E731
        self.rows_test = cat(parts)
        self.class_pos_test = pos
        self.test_counters = cnt
        self.n_test_only_users = n_extra
        self.timings["passT_test_extract_s"] = round(time.time() - t0, 1)
        self._say(f"PT gated extract: kept RETRO_TEST rows {self.rows_test['g'].size}; TEST-only users {n_extra}; "
                  f"target meta joins {self.meta_checks}")

    # ---------------------------------------------------------------- assembly (locked _eval_manifest + M07 changes)
    def set_role_band(self, band_users) -> None:
        self.role = np.zeros(self.u_dig.size, dtype=np.int64)
        self.role[np.asarray(band_users, dtype=np.int64)] |= BIT_COHORT

    def test_eval_manifest(self, definition: str, v: dict, class_pos: np.ndarray, kind: str, parent) -> dict:
        E = self.E
        locked_id = "M02_VALIDATION_WARM" if kind == "warm" else "M03_COLD_PANEL_VAL"
        m = self._eval_manifest(locked_id, {k: a for k, a in v.items() if k != "grace"}, class_pos, kind, parent)
        o = np.argsort(v["decision_id"], kind="stable")                # the order _eval_manifest applied
        vs = {k: a[o] for k, a in v.items()}
        dec = m["tables"]["decisions"]
        if not np.array_equal(dec["decision_id"], vs["decision_id"]):
            raise BuildBlocked(f"{definition}: row order differs from the locked _eval_manifest order")
        n = int(vs["g"].size)
        dec["v4_split"] = EnumValues(np.full(n, E.c("v4_split", "RETRO_TEST"), dtype=np.int64), E.v["v4_split"])
        dec["subperiod"] = EnumValues(np.full(n, E.c("subperiod", "NONE"), dtype=np.int64), E.v["subperiod"])
        grace = vs["grace"].astype(bool)
        dec["grace_matured"] = grace
        ev, tts, dts, rk = vs["eval"], vs["tts"], vs["dts"], dec["rankable"]
        matured = vs["status"] == E.c("target_status", "OBSERVED_MATURED_IN_GRACE")
        obs = (vs["status"] == E.c("target_status", "OBSERVED_IN_PARTITION")) | matured
        a = {k: m["assertions"][k] for k in KEEP_LOCKED_ASSERTIONS if k in m["assertions"]}
        a["decision_ts_in_retro_test"] = bool(np.all((dts >= VALIDATION_END_US) & (dts < TEST_END_US)))
        a["old_split_all_TEST"] = bool(np.all(vs["split"] == E.c("old_split", "TEST")))
        a["v4_split_all_RETRO_TEST"] = True
        a["subperiod_all_NONE"] = True
        a["eval_mask_equals_observed_or_matured_status"] = bool(np.array_equal(ev, obs))
        a["grace_matured_iff_status_matured"] = bool(np.array_equal(grace, matured))
        a["grace_targets_in_label_grace"] = bool(np.all((tts[grace] >= TEST_END_US) & (tts[grace] < GRACE_END_US)))
        a["in_partition_targets_before_test_end"] = bool(np.all(tts[ev & ~grace] < TEST_END_US))
        n_late = int((ev & (tts >= GRACE_END_US)).sum())
        a["no_eval_target_at_or_after_label_grace_end"] = n_late == 0
        a["grace_target_metadata_not_applicable"] = bool(all(
            np.all(vs[k][grace] == E.c("meta_token_state", "NOT_APPLICABLE"))
            for k in ("cat_state", "brand_state", "main_state")))
        c = m["counts_by_cause"]
        c["n_label_time_excluded"] = n_late
        c.pop("by_subperiod", None)
        c["by_subperiod"] = {"NONE": {"decisions": n, "eval_rows": int(ev.sum()),
                                      "users": int(np.unique(dec["eval_user_key"]).size)}}
        c["n_grace_matured_rows"] = int(grace.sum())
        c["n_grace_matured_rankable_rows"] = int((grace & rk).sum())
        c["n_grace_matured_oov_rows"] = int((grace & vs["oov"]).sum())
        c["n_grace_matured_target_metadata_not_applicable"] = int(grace.sum())
        c["notes"] = {"n_split_reassigned": "old split TEST relabelled v4_split RETRO_TEST (every row)",
                      "n_label_time_excluded": "eval rows with target_ts >= the end of LABEL_GRACE (must be 0)",
                      "users.n_validation_decisions": "the M02 column name is kept (same schema); it counts this "
                                                      "manifest's RETRO_TEST decisions",
                      "target_metadata_state_counts": "grace-matured targets are NOT_APPLICABLE (d9_common.GRACE_NOTE)"}
        m.update(manifest_id=definition, assertions=a, expected=[])
        return m


# ------------------------------------------------------------------------------------------------ checks
def registry_band_checks(bands: dict, kv2: np.ndarray, registry_path: Path) -> list:
    """Every rebuilt band = the registered support manifest's population_sha256_selection_key_v2 (REGISTRY_V2)."""
    reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    out = []
    for band, mid in BAND_SUPPORT_ID.items():
        want = ((reg.get("manifests") or {}).get(mid) or {}).get("population_sha256_selection_key_v2")
        got = population_sha256(np.sort(kv2[bands[band]]))
        out.append(D9.chk(f"{band}.population_sha256_selection_key_v2_eq_registry_{mid}", got, want,
                          f"REGISTRY_V2 manifests.{mid}"))
    return out


def validation_class_checks(man_counts: dict, registry_path: Path, v4: Path) -> list:
    """The class positives used here equal the VALIDATION counterpart's (same band, same TRAIN rows): the counterpart
    header's n_classes_with_ge1 / ge5 cohort-TRAIN positive counts (read from the registered, non-TEST header)."""
    reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    out = []
    for tid, counts in man_counts.items():
        vid = D9.VALIDATION_COUNTERPART[tid]
        hp = Path(v4) / ((reg.get("manifests") or {}).get(vid) or {}).get("path", "")
        if not hp.is_file():
            out.append(D9.chk(f"{tid}.class_positives_eq_{vid}", "header absent", "header present",
                              "registered VALIDATION header"))
            continue
        vc = json.loads(hp.read_text(encoding="utf-8"))["counts_by_cause"]
        for k in ("n_classes_with_ge1_cohort_train_positive", "n_classes_with_ge5_cohort_train_positive"):
            out.append(D9.chk(f"{tid}.{k}_eq_{vid}", counts[k], vc.get(k), f"{vid} header counts_by_cause"))
    return out


# ------------------------------------------------------------------------------------------------ driver
def run_build(b: D9Builder, out_root: Path, *, write: bool = True, registry_path: Path = REGISTRY_V2) -> dict:
    gate = revalidate(b.gate)
    t0 = time.time()
    lock = FC.lock_info()
    code = D9.d9_code_hashes()
    full = b.full
    checks = [D9.chk("code.v4data_builder_code_sha256_is_DATA04", b.code_sha, FC.DATA04_BUILDER_CODE_SHA256,
                     "DATA-04 handoff", hard=full)]
    b.prepare()
    t_prep = time.time()
    if full:
        checks += b.universe_checks()
    bands = compute_bands(b)                              # v4data06, unedited: DATA-06's populations
    checks += [c for c in band_integrity_checks(b, bands) if full or c["hard"]]   # population constants: full only
    kv2_base = bands["keys_v2"]
    if full:
        checks += registry_band_checks(bands, kv2_base, registry_path)
    hard = [c for c in checks if c["hard"] and not c["pass"]]
    if hard:
        raise BuildBlocked(f"DATA-09 band checks failed: {[c['name'] for c in hard]}")
    band_sets = {k: bands[k] for k in ("MAIN", "U_NEW_TEST", "SPARSE", "HEAVY")}
    b.pass_test_extract(band_sets)
    n_extra = b.n_test_only_users
    kv2 = np.concatenate([kv2_base, C2.selection_keys_v2(digest_rows(b.u_dig[kv2_base.size:]))]) if n_extra else kv2_base
    if np.unique(kv2).size != kv2.size:
        raise BuildBlocked("two users share a selection_key_v2 after appending the TEST-only users")
    R = b.rows_test
    n_all = int(b.u_dig.size)
    nelig_row = b.u_nelig[R["g"]]
    man, written, extras = {}, {}, {}

    def sub(users: np.ndarray) -> dict:
        um = np.zeros(n_all, dtype=bool)
        um[users] = True
        keep = um[R["g"]]
        return {k: a[keep] for k, a in R.items()}

    def n_e2e(m: dict) -> int:
        return int((m["tables"]["users"]["n_eval_rows"] >= 1).sum())

    specs = []
    for tid in (D9.MAIN_TEST_ID, D9.UNEW_TEST_ID, D9.SPARSE_TEST_ID, D9.HEAVY_TEST_ID):
        specs.append((tid, np.asarray(bands[D9.BAND_OF_ID[tid]], dtype=np.int64), "warm"))
    cold_users = np.unique(R["g"][R["eval"] & (nelig_row == 0)])
    specs.append((D9.TRUE_COLD_TEST_ID, cold_users, "cold"))
    class_counts = {}
    for tid, users, kind in specs:
        role = D9.ROLE_OF_ID[tid]
        v = sub(users)
        if kind == "warm":
            b.set_role_band(users)
        else:
            b.set_role_band(bands["MAIN"])                # no_user_in_M01 = disjoint from the MAIN band
        cp = b.class_pos_test[D9.CLASS_POS_BAND[tid]]
        m = b.test_eval_manifest(D9.DEFINITION[tid], v, cp, kind, D9.PARENT[tid])
        class_counts[tid] = {k: m["counts_by_cause"][k] for k in ("n_classes_with_ge1_cohort_train_positive",
                                                                  "n_classes_with_ge5_cohort_train_positive")}
        users_in_manifest = m["users_g"]
        bad = [k for k, x in m["assertions"].items() if x is not True]
        checks.append(D9.chk(f"{tid}.assertions_all_true", bad, [], "MANIFEST_SPEC M07 / M08 + DATA-09 RETRO_TEST rules"))
        checks.append(D9.chk(f"{tid}.users_subset_of_population", bool(np.all(np.isin(users_in_manifest, users))),
                             True, "population definition"))
        if kind == "warm":
            n_no_test = int(users.size - np.unique(v["g"]).size)
            extras[tid] = {"n_population_users": int(users.size), "n_population_users_without_retro_test_row": n_no_test,
                           "n_users_with_ge1_eval_row": n_e2e(m)}
        else:
            ok_cold = bool(np.all(b.u_nelig[users_in_manifest] == 0))
            checks.append(D9.chk(f"{tid}.every_user_zero_eligible", ok_cold, True, "M08 population"))
            checks.append(D9.chk(f"{tid}.every_user_has_ge1_eval_row", bool(np.all(m["tables"]["users"]["n_eval_rows"]
                                                                                    >= 1)), True, "M08 population"))
            checks.append(D9.chk(f"{tid}.n_users_eq_full_population_cold_test_users", int(users.size),
                                 b.test_counters["test_users"]["cold_zero_eligible_train_examples"],
                                 "structural: M08 = every zero-eligible user with >= 1 RETRO_TEST eval row"))
            for band in band_sets:
                checks.append(D9.chk(f"{tid}.disjoint_from_{band}", int(np.intersect1d(users, bands[band]).size), 0,
                                     "warm bands vs zero-eligible users"))
            extras[tid] = {"n_users_with_ge1_eval_row": n_e2e(m), "n_test_only_users": int(
                (users_in_manifest >= kv2_base.size).sum()),
                "selection_rule": "every user with >= 1 RETRO_TEST eval row and zero TRAIN-eligible examples "
                                  "(MANIFEST_SPEC M08 literally; NOT restricted to TRUE_COLD_VALIDATION users, "
                                  "TEST-only users included)"}
        man[tid] = (m, users, kind)
    if full:
        checks += validation_class_checks(class_counts, registry_path, V4)
    hard = [c for c in checks if c["hard"] and not c["pass"]]
    if hard:
        raise BuildBlocked(f"DATA-09 manifest checks failed: {[c['name'] for c in hard]}")

    hashes = {}
    for tid, (m, users, kind) in man.items():
        role = D9.ROLE_OF_ID[tid]
        keys_sorted = np.sort(kv2[users])
        if kind == "warm":
            band = D9.BAND_OF_ID[tid]
            population = {"band": band, "n_users": int(users.size),
                          "population_sha256_selection_key_v2": population_sha256(keys_sorted),
                          "support_manifest": D9.PARENT[tid], "validation_counterpart": D9.VALIDATION_COUNTERPART[tid],
                          "rule": "the registered support manifest's users (v4data06.d6_build.compute_bands, unedited); "
                                  "eval rows = their RETRO_TEST decisions; users without a RETRO_TEST row are absent"}
        else:
            population = {"band": "TRUE_COLD_RETRO_TEST", "n_users": int(users.size),
                          "population_sha256_selection_key_v2": population_sha256(keys_sorted),
                          "rule": extras[tid]["selection_rule"], "class_positive_reference": D9.DC.MAIN_TRAIN_ID}
        enc = D9.encode_tables_d9(b, tid, m, gate)
        meta = FC.tables_meta_of(enc)
        h = D9.d9_header(b, m, meta, out_id=tid, role=role, parent=D9.PARENT[tid], lock=lock, code=code,
                         population=population, expected=[("n_users_with_ge1_eval_row", extras[tid][
                             "n_users_with_ge1_eval_row"], extras[tid]["n_users_with_ge1_eval_row"],
                             "measured (MANIFEST_SPEC M07/M08: none pre-registered)")],
                         users_keys_v2=kv2[users], gate=gate, extra={"n_users_with_ge1_eval_row": extras[tid][
                             "n_users_with_ge1_eval_row"], "population_extra": extras[tid]})
        w = D9.write_manifest_d9(out_root, tid, enc, h, gate) if write else {}
        written[tid] = w
        hashes[tid] = FC.hashes_entry(h, meta, (w or {}).get("header_file_sha256"))
        if write:
            log(f"wrote {tid}: " + ", ".join(f"{t['table']}={t['row_count']}" for t in meta))
        man[tid] = (m, users, kind, h, meta)
        del enc
    if write:
        FC.atomic_write_bytes(out_root, "population/TRUE_COLD_RETRO_TEST_SELECTION_KEY_V2.u64be",
                              np.ascontiguousarray(np.sort(kv2[cold_users]).astype(">u8")).tobytes())
    rec = {"receipt": "DATA09_BUILD_RECEIPT", "task": D9.TASK, "status": "PASS", "full_build": full,
           "buckets": len(b.buckets), "python_utc_end": utc_now(),
           "isolation_modules_checked": len(ppsi_env.assert_isolated()), "lock": lock, "code": code,
           "test_gate": {**gate.binding(), "register_file_sha256_at_build": gate.register_file_sha256,
                         "lock_path": gate.lock_path, "register_path": gate.register_path},
           "checks": checks, "n_checks": len(checks), "n_failed": len([c for c in checks if not c["pass"]]),
           "manifests": hashes, "roles": D9.TEST_ROLES, "population_extra": extras,
           "full_population_retro_test_counters": b.test_counters,
           "full_population_counters_note": "counts only; compared with the G0 census (SPLIT_COUNTS per_old_split.TEST "
                                            "/ v4_retro_test, CLIENT_CENSUS s5.test_users) by d9_finalize AFTER the "
                                            "build (TEST_EXPOSURE_REGISTER entry 1)",
           "target_metadata_join": b.meta_checks, "n_test_only_users_appended": int(n_extra),
           "timings_s": {"prepare": round(t_prep - t0, 1), "total": round(time.time() - t0, 1), **b.timings},
           "test_split_read": True, "test_split_read_gate": "d9_read.read_with_test under the GateRecord above",
           "no_raw_ids_written": True}
    if write:
        FC.write_json_self_hashed(out_root, "receipts/DATA09_MANIFEST_HASHES.json",
                                  {"receipt": "DATA09_MANIFEST_HASHES", "data_version": D9.DATA_VERSION,
                                   "builder_code_sha256": b.code_sha, "d9_code_sha256": code["d9_code_sha256"],
                                   "candidate_order_hash": b.candidate_order_hash, "test_gate": gate.binding(),
                                   "manifests": hashes})
        FC.write_json_self_hashed(out_root, "receipts/DATA09_BUILD_RECEIPT.json", rec)
    rec["_man"] = man
    rec["_bands"] = bands
    rec["_kv2"] = kv2
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DATA-09 RETRO_TEST manifests (gated: ALL_METHODS_LOCK + MANIFEST_BUILD)")
    ap.add_argument("--lock", type=Path, default=None)
    ap.add_argument("--register", type=Path, default=None)
    ap.add_argument("--out", default=str(V4 / D9.OUT_REL))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--buckets", default="all", help="'all' or a comma list (partial smoke; disables hard checks)")
    a = ap.parse_args(argv)
    try:
        gate = check_test_gate(a.lock, a.register)       # FIRST: before any dataset file is opened
    except TestGateRefused as e:
        log(f"DATA-09 REFUSED (TEST gate closed): {e}")
        return 2
    log(f"TEST gate open: lock {gate.lock_sha256[:12]}... register entry {gate.entry_id}")
    buckets = None if a.buckets == "all" else [int(x) for x in a.buckets.split(",")]
    b = D9Builder(ds_root=Path(a.dataset) if a.dataset else None, out_root=Path(a.out), buckets=buckets, log=log,
                  gate=gate)
    if builder_code_sha256() != FC.DATA04_BUILDER_CODE_SHA256:
        log("WARNING: v4data builder_code_sha256 differs from DATA-04 (hard failure in a full build)")
    rec = run_build(b, Path(a.out))
    log(f"DATA-09 done: n_failed={rec['n_failed']} manifests={list(rec['manifests'])}")
    return 0 if rec["n_failed"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
