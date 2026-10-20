"""DATA-09 test helpers: an INDEPENDENT reference for the synthetic ledger (it does not call the builder) and an
independent re-implementation of the RETRO_TEST (M07 / M08) manifest assertions over re-read Parquet tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from census_v2 import core as C2
from v4data.common import (GRACE_END_US, TEST_END_US, TRAIN_END_US, VALIDATION_END_US, client_digest, digest_rows,
                           table_content_sha256)
from v4data09.d9_common import reread_columns_d9

CENSOR_OF_STATUS = {"NO_OBSERVED_NEXT": "CEN_NO_NEXT", "AMBIGUOUS_BOUNDARY": "CEN_AMBIGUOUS_NEXT_BARRIER",
                    "BOUNDARY_UNRESOLVED": "CEN_RIGHT_EDGE", "OBSERVED_IN_PARTITION": "NONE",
                    "OBSERVED_MATURED_IN_GRACE": "NONE"}


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def read_tables_d9(mdir: Path, mid: str) -> dict:
    h = load_json(Path(mdir) / f"{mid}.header.json")
    out = {"header": h}
    for t in h["tables"]:
        tbl = pq.ParquetFile(Path(mdir) / f"{mid}.{t['table']}.parquet").read()
        out[t["table"]] = {c: tbl[c].to_numpy(zero_copy_only=False) for c in tbl.column_names}
        out[f"{t['table']}.__content"] = table_content_sha256(reread_columns_d9(tbl))
        out[f"{t['table']}.__schema"] = tbl.schema.remove_metadata()
    return out


class Ledger:
    """The synthetic ledger read directly (pyarrow), with per-user eligibility and keys computed independently."""

    def __init__(self, ds: Path):
        tbl = pq.read_table(Path(ds) / "decisions" / "user_bucket=0" / "ledger.parquet")
        for col in ("decision_ts", "target_ts"):
            tbl = tbl.set_column(tbl.column_names.index(col), col, pc.cast(tbl[col], pa.int64()))
        t = tbl.to_pydict()
        self.rows = [dict(zip(t, v)) for v in zip(*t.values())]
        self.users = sorted({r["user_id"] for r in self.rows})
        self.hex = {u: client_digest(u).hex() for u in self.users}
        self.key_v2 = {u: int(C2.selection_keys_v2(digest_rows(np.array([client_digest(u)], dtype="S32")))[0])
                       for u in self.users}
        self.nelig = {u: 0 for u in self.users}
        self.ntrain = {u: 0 for u in self.users}
        self.nval_eval = {u: 0 for u in self.users}
        for r in self.rows:
            u = r["user_id"]
            if r["split"] == "TRAIN":
                self.ntrain[u] += 1
                tts = r["target_ts"]
                if r["loss_mask"] and r["target_class"] >= 0 and tts is not None and tts < TRAIN_END_US:
                    self.nelig[u] += 1
            if r["split"] in ("VALIDATION_SELECT", "VALIDATION_CONFIRM") and r["eval_mask"]:
                self.nval_eval[u] += 1
        self.by_did = {r["decision_id"]: r for r in self.rows}

    def test_rows(self, users) -> list:
        us = set(users)
        return [r for r in self.rows if r["split"] == "TEST" and r["user_id"] in us]

    def smallest_keys(self, users, n) -> list:
        return sorted(users, key=lambda u: self.key_v2[u])[:n]

    def pool(self) -> list:
        return self.smallest_keys([u for u in self.users if C2.C_LO <= self.nelig[u] <= C2.C_HI], 10 ** 9)

    def band(self, lo: int, hi: int) -> list:
        return self.pool()[lo:hi]

    def stratum_with_eval(self, lo: int, hi, n: int) -> list:
        return self.smallest_keys([u for u in self.users if lo <= self.nelig[u] and (hi is None or self.nelig[u] <= hi)
                                   and self.nval_eval[u] >= 1], n)

    def m08_users(self) -> list:
        return sorted({r["user_id"] for r in self.rows if r["split"] == "TEST" and r["eval_mask"]
                       and self.nelig[r["user_id"]] == 0})

    def class_positives(self, users, K: int) -> np.ndarray:
        us = set(users)
        pos = np.zeros(K, dtype=np.int64)
        for r in self.rows:
            if r["user_id"] in us and r["split"] == "TRAIN" and r["loss_mask"] and r["target_class"] >= 0 \
                    and r["target_ts"] is not None and r["target_ts"] < TRAIN_END_US:
                pos[r["target_class"]] += 1
        return pos


def m07_assertions(usr: dict, dec: dict) -> dict:
    """Independent RETRO_TEST (M07 / M08) assertions over re-read tables."""
    did = dec["decision_id"].astype(np.int64)
    dts, tts = dec["decision_ts_us"].astype(np.int64), dec["target_ts_us"].astype(np.int64)
    ev, tc = dec["eval_mask"].astype(bool), dec["target_class"].astype(np.int64)
    rk, oov, gr = dec["rankable"].astype(bool), dec["target_oov"].astype(bool), dec["grace_matured"].astype(bool)
    st, cen = dec["target_status"].astype(str), dec["censor_class"].astype(str)
    euk = dec["eval_user_key"].astype(np.int64)
    hexs = usr["client_key_hex"].astype(str)
    nu = usr["eval_user_key"].size
    return {
        "decision_ts_in_retro_test": bool(np.all((dts >= VALIDATION_END_US) & (dts < TEST_END_US))),
        "old_split_TEST": bool(np.all(dec["old_split"].astype(str) == "TEST")),
        "v4_split_RETRO_TEST": bool(np.all(dec["v4_split"].astype(str) == "RETRO_TEST")),
        "subperiod_NONE": bool(np.all(dec["subperiod"].astype(str) == "NONE")),
        "eval_mask_eq_observed_or_matured": bool(np.array_equal(ev, np.isin(st, ["OBSERVED_IN_PARTITION",
                                                                                "OBSERVED_MATURED_IN_GRACE"]))),
        "grace_matured_eq_status": bool(np.array_equal(gr, st == "OBSERVED_MATURED_IN_GRACE")),
        "censor_by_status": bool(all(CENSOR_OF_STATUS[s] == c for s, c in zip(st.tolist(), cen.tolist()))),
        "censor_none_iff_eval": bool(np.array_equal(cen == "NONE", ev)),
        "rankable_definition": bool(np.array_equal(rk, ev & (tc >= 0))),
        "oov_implies_eval_unranked": bool(np.all(~oov | (ev & (tc == -1)))),
        "grace_targets_in_label_grace": bool(np.all((tts[gr] >= TEST_END_US) & (tts[gr] < GRACE_END_US))),
        "in_partition_targets_before_test_end": bool(np.all(tts[ev & ~gr] < TEST_END_US)),
        "no_eval_target_after_grace": bool(np.all(tts[ev] < GRACE_END_US)),
        "grace_meta_not_applicable": bool(all(np.all(dec[k].astype(str)[gr] == "NOT_APPLICABLE") for k in (
            "target_category_state", "target_brand_state", "target_main_category_state"))),
        "decision_id_strictly_ascending": bool(did.size <= 1 or np.all(did[1:] > did[:-1])),
        "eval_user_key_dense_hex_ascending": bool(np.array_equal(usr["eval_user_key"], np.arange(nu))
                                                  and np.all(hexs[1:] > hexs[:-1])),
        "user_counts_match_rows": bool(np.array_equal(usr["n_validation_decisions"], np.bincount(euk, minlength=nu))
                                       and np.array_equal(usr["n_eval_rows"], np.bincount(euk[ev], minlength=nu))
                                       and np.array_equal(usr["n_rankable_rows"], np.bincount(euk[rk], minlength=nu))),
        "every_user_has_a_decision": bool(np.all(usr["n_validation_decisions"] >= 1)),
    }
