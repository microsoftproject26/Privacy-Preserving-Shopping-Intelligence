"""FS-01 test helpers: an INDEPENDENT re-implementation of the MANIFEST_SPEC M01 / M02 assertions over re-read
Parquet tables (it does not call the builder), and small readers. No dataset access."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from v4data.common import TRAIN_END_US, TRAIN_START_US, VALIDATION_END_US, VS_END_US, table_content_sha256
from v4data.manifests import reread_columns


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def read_tables(mdir: Path, mid: str) -> dict:
    h = load_json(Path(mdir) / f"{mid}.header.json")
    out = {"header": h}
    for t in h["tables"]:
        tbl = pq.ParquetFile(Path(mdir) / f"{mid}.{t['table']}.parquet").read()
        out[t["table"]] = {c: tbl[c].to_numpy(zero_copy_only=False) for c in tbl.column_names}
        out[f"{t['table']}.__content"] = table_content_sha256(reread_columns(tbl))
        out[f"{t['table']}.__rows"] = tbl.num_rows
    return out


def m01_assertions(cli: dict, dec: dict, central: dict, K: int) -> dict:
    n = cli["client_ord"].size
    hexs = cli["client_key_hex"].astype(str)
    rc = cli["row_count"].astype(np.int64)
    start = np.concatenate([[0], np.cumsum(rc)[:-1]])
    did = dec["decision_id"].astype(np.int64)
    co = dec["client_ord"].astype(np.int64)
    sk = cli["selection_key"].astype(np.uint64)
    order_ok = bool(np.all((co[1:] > co[:-1]) | ((co[1:] == co[:-1]) & (did[1:] > did[:-1])))) if did.size > 1 else True
    perm = central["row"].astype(np.int64)
    return {
        "target_ts_lt_train_end_not_sentinel": bool(np.all((dec["target_ts_us"] >= 0) & (dec["target_ts_us"] < TRAIN_END_US))),
        "target_class_in_range": bool(np.all((dec["target_class"] >= 0) & (dec["target_class"] < K))),
        "status_observed": bool(np.all(dec["target_status"].astype(str) == "OBSERVED_IN_PARTITION")),
        "old_split_train": bool(np.all(dec["old_split"].astype(str) == "TRAIN")),
        "decision_ts_in_train": bool(np.all((dec["decision_ts_us"] >= TRAIN_START_US) & (dec["decision_ts_us"] < TRAIN_END_US))),
        "client_ord_dense": bool(np.array_equal(cli["client_ord"], np.arange(n))),
        "row_count_ge1": bool(np.all(rc >= 1)),
        "row_count_eq_n_eligible": bool(np.array_equal(rc, cli["n_eligible_train_examples"])),
        "row_start_prefix_sum": bool(np.array_equal(cli["row_start"], start)),
        "ranges_contiguous_cover": bool(np.array_equal(np.repeat(np.arange(n), rc), co) and rc.sum() == did.size),
        "decisions_sorted_client_ord_decision_id": order_ok,
        "decision_id_unique": bool(np.unique(did).size == did.size),
        "client_key_hex_valid_unique_ascending": bool(all(len(h) == 64 and all(c in "0123456789abcdef" for c in h)
                                                          for h in hexs) and np.all(hexs[1:] > hexs[:-1])),
        "selection_key_is_first_16_hex": bool(np.array_equal(sk, np.array([int(h[:16], 16) for h in hexs], dtype=np.uint64))),
        "selection_key_strictly_ascending": bool(np.all(sk[1:] > sk[:-1])),
        "central_order_sorts_decision_id": bool(np.array_equal(np.sort(perm), np.arange(did.size))
                                                and np.all(np.diff(did[perm]) > 0)),
    }


def m02_assertions(usr: dict, dec: dict, train_cli: dict, train_dec: dict, K: int) -> dict:
    did = dec["decision_id"].astype(np.int64)
    dts = dec["decision_ts_us"]
    ev = dec["eval_mask"].astype(bool)
    tc = dec["target_class"].astype(np.int64)
    rk = dec["rankable"].astype(bool)
    oov = dec["target_oov"].astype(bool)
    st = dec["target_status"].astype(str)
    cen = dec["censor_class"].astype(str)
    osp = dec["old_split"].astype(str)
    sub = dec["subperiod"].astype(str)
    euk = dec["eval_user_key"].astype(np.int64)
    nu = usr["eval_user_key"].size
    hexs = usr["client_key_hex"].astype(str)
    pos = np.bincount(train_dec["target_class"].astype(np.int64), minlength=K)
    tpos = dec["target_class_cohort_train_positives"].astype(np.int64)
    return {
        "decision_ts_in_validation": bool(np.all((dts >= TRAIN_END_US) & (dts < VALIDATION_END_US))),
        "old_split_is_vs_or_vc": bool(np.all(np.isin(osp, ["VALIDATION_SELECT", "VALIDATION_CONFIRM"]))),
        "v4_split_validation": bool(np.all(dec["v4_split"].astype(str) == "VALIDATION")),
        "eval_mask_eq_observed": bool(np.array_equal(ev, st == "OBSERVED_IN_PARTITION")),
        "censor_none_iff_eval": bool(np.array_equal(cen == "NONE", ev)),
        "censored_rows_have_a_class": bool(np.all(np.isin(cen[~ev], ["CEN_NO_NEXT", "CEN_LEGACY_BARRIER", "CEN_RIGHT_EDGE",
                                                                     "CEN_AMBIGUOUS_NEXT_BARRIER"]))),
        "rankable_definition": bool(np.array_equal(rk, ev & (tc >= 0))),
        "oov_implies_eval_unranked": bool(np.all(~oov | (ev & (tc == -1)))),
        "no_eval_target_at_or_after_validation_end": bool(np.all(dec["target_ts_us"][ev] < VALIDATION_END_US)),
        "subperiod_matches_old_split_and_time": bool(np.array_equal(sub == "VAL_A_20191116_20191120", osp == "VALIDATION_SELECT")
                                                     and np.array_equal(sub == "VAL_A_20191116_20191120", dts < VS_END_US)),
        "decision_id_strictly_ascending": bool(did.size <= 1 or np.all(did[1:] > did[:-1])),
        "users_subset_of_train_clients": bool(set(hexs.tolist()) <= set(train_cli["client_key_hex"].astype(str).tolist())),
        "eval_user_key_dense_hex_ascending": bool(np.array_equal(usr["eval_user_key"], np.arange(nu))
                                                  and np.all(hexs[1:] > hexs[:-1])),
        "user_counts_match_rows": bool(np.array_equal(usr["n_validation_decisions"], np.bincount(euk, minlength=nu))
                                       and np.array_equal(usr["n_eval_rows"], np.bincount(euk[ev], minlength=nu))
                                       and np.array_equal(usr["n_rankable_rows"], np.bincount(euk[rk], minlength=nu))),
        "every_user_has_a_validation_decision": bool(np.all(usr["n_validation_decisions"] >= 1)),
        "train_positives_are_within_population": bool(np.array_equal(tpos[rk], pos[tc[rk]]) and np.all(tpos[~rk] == -1)),
    }
