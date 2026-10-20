"""census_v2.test_census02 (owner: Data Lead, CENSUS-02) - unit tests on a synthetic fixture (no real data).

Run:  python -B -m census_v2.test_census02 --receipt <file.json>      (from ppsi_v4/code)
Covers: strata boundaries 0/1/4/5/100/101 (A0/A1/B/B/C/C/D), selection_key_v2 known answer, key nesting and the
population hash, shares summing to 1, the FL step arithmetic, Gini / top shares, calibration sizes, the head-size
arithmetic against the MODEL-01 inventories, and an end-to-end census over a synthetic frozen-layout ledger that
contains TEST rows (which must stay invisible), OOV targets of both causes, censored VALIDATION rows and a catalogue-B
circularity case (one C user falls to B, one D user enters C).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from v4data.common import SALT, VERSION_HASH, population_sha256, sha256_file  # noqa: E402

from census_v2 import core  # noqa: E402
from census_v2 import census02  # noqa: E402

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def digest(uid: str) -> bytes:
    return hashlib.sha256(SALT + b"|" + uid.encode("utf-8")).digest()


# ------------------------------------------------------------------------------------------------ pure tests
@test
def test_strata_boundaries():
    n = np.array([0, 0, 1, 4, 5, 100, 101])
    ntr = np.array([0, 3, 1, 4, 5, 100, 101])
    got = [core.STRATA[c] for c in core.stratum_codes(n, ntr)]
    assert got == ["A0", "A1", "B", "B", "C", "C", "D"], got
    for bad_n, bad_t in (([1], [0]), ([5], [4]), ([-1], [2])):
        try:
            core.stratum_codes(np.array(bad_n), np.array(bad_t))
        except ValueError:
            pass
        else:
            raise AssertionError(f"inconsistent counts accepted: {bad_n} {bad_t}")


@test
def test_selection_key_v2_known_answer():
    d = digest("synthetic-user-1")
    want = int.from_bytes(hashlib.sha256(b"PPSI_V4_POP_5_100_SEL_20260926|" + d).digest()[:8], "big")
    assert core.selection_key_v2(d) == want
    assert int(core.selection_keys_v2([d])[0]) == want
    assert want != int.from_bytes(d[:8], "big"), "v2 key must differ from the historical key"
    # a digest with trailing zero bytes keeps its full width
    z = b"\x01" * 31 + b"\x00"
    assert core.selection_key_v2(z) == int.from_bytes(hashlib.sha256(core.SEL_PREFIX_V2 + z).digest()[:8], "big")


@test
def test_key_nesting_and_population_hash():
    ds = [hashlib.sha256(f"u{i}".encode()).digest() for i in range(5000)]
    k = np.sort(core.selection_keys_v2(ds))
    assert np.unique(k).size == k.size
    sizes = [64, 128, 1000, 2048, 5000]
    assert core.assert_nested(k, sizes)
    for s in sizes:
        assert population_sha256(k[:s]) == hashlib.sha256(k[:s].astype(">u8").tobytes()).hexdigest()
    bad = k.copy()
    bad[[10, 11]] = bad[[11, 10]]
    assert not core.assert_nested(bad, sizes)


@test
def test_calib_sizes():
    got = core.calib_sizes(32768)
    assert got == [("N/16", 2048), ("N/8", 4096), ("N/4", 8192), ("fixed_8192", 8192), ("fixed_16384", 16384)], got
    got = dict((lab, s) for lab, s in core.calib_sizes(1446986))
    assert got["N/16"] == 90436 and got["N/8"] == 180873 and got["N/4"] == 361746 and got["fixed_65536"] == 65536


@test
def test_fl_local_work_arithmetic():
    n = np.array([1, 16, 17, 32, 100])
    w = core.fl_local_work(n, 16, 2, 64)
    assert w["total_client_steps_per_sweep"] == 2 * (1 + 1 + 2 + 2 + 7)
    assert w["total_client_steps_per_EFE"] == 13
    assert w["fraction_clients_data_fits_one_batch"] == 2 / 5
    assert w["rounds_per_sweep"] == 1 and w["rounds_per_EFE_at_64_clients_per_round"] == 0.5
    assert w["per_client_steps_per_visit_quantiles"]["max"] == 14.0
    one = core.fl_local_work(np.array([7] * 10), 8, 1, 64)
    assert one["expected_max_client_steps_in_a_round"] == 1.0


@test
def test_gini_and_top_share():
    assert abs(core.gini([3, 3, 3, 3])) < 1e-15
    assert abs(core.gini([0, 0, 0, 10]) - 0.75) < 1e-15
    t = core.top_share(list(range(1, 101)), 0.10)
    assert t["top_k_clients"] == 10 and t["examples_held"] == sum(range(91, 101))


@test
def test_head_sizes_reproduce_inventory():
    for fn in ("GRU_CLEAN_V2_MANIFEST_INVENTORY.json", "SASREC_CLEAN_V2_MANIFEST_INVENTORY.json"):
        inv = json.loads((CODE / "v4models" / "inventory" / fn).read_text(encoding="utf-8"))
        h = core.head_sizes(int(inv["K"]), inv)
        assert h["total_shared_params_head_and_input_vocab_change"]["params"] == inv["parameter_inventory"]["total"]
        assert h["total_shared_params_head_only_change"]["params"] == inv["parameter_inventory"]["total"]
        small = core.head_sizes(1000, inv)
        assert small["W_out"]["rows"] == 1000


@test
def test_shares_sum_to_one_pure():
    n = np.array([0, 0, 1, 4, 5, 100, 101, 7, 2])
    ntr = np.array([0, 2, 1, 4, 5, 100, 101, 7, 2])
    st = core.stratum_codes(n, ntr)
    us = sum(core.share(int((st == c).sum()), st.size) for c in range(5))
    es = sum(core.share(int(n[st == c].sum()), int(n.sum())) for c in range(5))
    assert abs(us - 1.0) < 1e-12 and abs(es - 1.0) < 1e-12


# ------------------------------------------------------------------------------------------ synthetic dataset
T0 = dt.datetime(2019, 10, 5)
VS_T = dt.datetime(2019, 11, 17)
VC_T = dt.datetime(2019, 11, 21)
TE_T = dt.datetime(2019, 11, 25)


class Fixture:
    """Frozen-layout ledger fixture. Catalogue: p0..p9 (class = index). x1 is TRAIN-seen but uncatalogued
    (support 2 -> BELOW_SUPPORT); zz never appears in TRAIN (ABSENT_FROM_TRAIN)."""

    def __init__(self):
        self.rows = []
        self.did = 0
        self.ctx_i = 0

    def add(self, uid, split, ts, status, *, ctx, tgt=None, tc=-1, loss=False, ev=False, oov=False, has=False,
            nsplit=None, nblock=None, tts=None):
        self.did += 1
        self.rows.append({"user_id": uid, "decision_id": self.did, "split": split, "decision_ts": ts,
                          "target_ts": tts, "target_status": status, "next_split": nsplit,
                          "next_block_type": nblock, "has_observed_target": has, "loss_mask": loss, "eval_mask": ev,
                          "target_class": tc, "target_oov": oov, "target_product_id": tgt, "product_id": ctx})

    def train_sup(self, uid, k, ctxs, tclass=lambda i: i % 9):
        for i in range(k):
            ts = T0 + dt.timedelta(minutes=self.did)
            c = tclass(i)
            self.add(uid, "TRAIN", ts, "OBSERVED_IN_PARTITION", ctx=ctxs(i), tgt=f"p{c}", tc=c, loss=True, has=True,
                     tts=ts + dt.timedelta(seconds=30))

    def build(self, root: Path):
        f = self
        cyc9 = lambda i: f"p{i % 9}"  # noqa: E731
        # A0: VALIDATION only - one eval row, one censored NO_NEXT row
        f.add("a0", "VALIDATION_SELECT", VS_T, "OBSERVED_IN_PARTITION", ctx="p0", tgt="p0", tc=0, ev=True, has=True,
              tts=VS_T + dt.timedelta(minutes=1))
        f.add("a0", "VALIDATION_SELECT", VS_T + dt.timedelta(minutes=2), "NO_OBSERVED_NEXT", ctx="p1")
        # A1: TRAIN events without labels; VC eval row with an ABSENT OOV target
        for i in range(3):
            f.add("a1", "TRAIN", T0 + dt.timedelta(hours=i), "NO_OBSERVED_NEXT", ctx="p9", nsplit="TRAIN",
                  nblock="SINGLE_ITEM")
        f.add("a1", "VALIDATION_CONFIRM", VC_T, "OBSERVED_IN_PARTITION", ctx="p2", tgt="zz", oov=True, ev=True,
              has=True, tts=VC_T + dt.timedelta(minutes=1))
        # B: n=1 (+ one TRAIN OOV target x1 with support 2) and n=4 (two x1 context rows)
        f.train_sup("b1", 1, lambda i: "p9")
        f.add("b1", "TRAIN", T0 + dt.timedelta(days=1), "OBSERVED_IN_PARTITION", ctx="p9", tgt="x1", oov=True,
              has=True, tts=T0 + dt.timedelta(days=1, seconds=5))
        f.add("b1", "VALIDATION_SELECT", VS_T, "OBSERVED_IN_PARTITION", ctx="p0", tgt="p3", tc=3, ev=True, has=True,
              tts=VS_T + dt.timedelta(minutes=1))
        f.train_sup("b4", 4, lambda i: "x1" if i < 2 else "p9")
        # C: n=5 (one target p9 -> falls to B under catalogue B) and n=100
        f.train_sup("c5", 5, cyc9, tclass=lambda i: 9 if i == 0 else i % 9)
        f.add("c5", "VALIDATION_SELECT", VS_T, "OBSERVED_IN_PARTITION", ctx="p0", tgt="p1", tc=1, ev=True, has=True,
              tts=VS_T + dt.timedelta(minutes=1))
        f.add("c5", "VALIDATION_SELECT", VS_T + dt.timedelta(minutes=5), "OBSERVED_IN_PARTITION", ctx="p0",
              tgt="x1", oov=True, ev=True, has=True, tts=VS_T + dt.timedelta(minutes=6))
        f.add("c5", "VALIDATION_SELECT", VS_T + dt.timedelta(hours=70), "BOUNDARY_UNRESOLVED", ctx="p0",
              nsplit="VALIDATION_CONFIRM", nblock="SINGLE_ITEM")
        f.add("c5", "TEST", TE_T, "OBSERVED_IN_PARTITION", ctx="p0", tgt="p0", tc=0, ev=True, has=True,
              tts=TE_T + dt.timedelta(minutes=1))
        f.train_sup("c100", 100, cyc9)
        for i in range(3):
            f.add("c100", "VALIDATION_CONFIRM", VC_T + dt.timedelta(hours=i), "OBSERVED_IN_PARTITION", ctx="p0",
                  tgt="p2", tc=2, ev=True, has=True, tts=VC_T + dt.timedelta(hours=i, minutes=1))
        # D: n=101 with one p9 target (enters C under catalogue B)
        f.train_sup("d101", 101, lambda i: "p9", tclass=lambda i: 9 if i == 0 else i % 9)
        # TEST-only user: must be invisible
        f.add("t_only", "TEST", TE_T, "OBSERVED_IN_PARTITION", ctx="p0", tgt="p0", tc=0, ev=True, has=True,
              tts=TE_T + dt.timedelta(minutes=1))
        schema = pa.schema([("user_id", pa.string()), ("decision_id", pa.int64()), ("split", pa.string()),
                            ("decision_ts", pa.timestamp("us")), ("target_ts", pa.timestamp("us")),
                            ("target_status", pa.string()), ("next_split", pa.string()),
                            ("next_block_type", pa.string()), ("has_observed_target", pa.bool_()),
                            ("loss_mask", pa.bool_()), ("eval_mask", pa.bool_()), ("target_class", pa.int64()),
                            ("target_oov", pa.bool_()), ("target_product_id", pa.string()), ("product_id", pa.string())])
        tbl = pa.Table.from_pylist(f.rows, schema=schema)
        d = root / "decisions" / "user_bucket=0"
        d.mkdir(parents=True)
        pq.write_table(tbl, d / "ledger.parquet")
        tr = [r for r in f.rows if r["split"] == "TRAIN"]
        sup = {}
        for r in tr:
            sup[r["product_id"]] = sup.get(r["product_id"], 0) + 1
        self.support = sup
        cat = pa.table({"product_id": [f"p{i}" for i in range(10)], "output_class": pa.array(range(10), pa.int64()),
                        "product_idx": pa.array(range(3, 13), pa.int64()),
                        "representative_views": pa.array([sup.get(f"p{i}", 0) for i in range(10)], pa.int64())})
        (root / "mappings").mkdir()
        pq.write_table(cat, root / "mappings" / "catalogue.parquet")
        (root / "VERSION.json").write_text(json.dumps({"version_hash": VERSION_HASH}), encoding="utf-8")
        files = [{"path": "decisions/user_bucket=0/ledger.parquet",
                  "sha256": sha256_file(d / "ledger.parquet")},
                 {"path": "mappings/catalogue.parquet", "sha256": sha256_file(root / "mappings" / "catalogue.parquet")}]
        (root / "FILES.json").write_text(json.dumps({"files": files}), encoding="utf-8")
        return self


@test
def test_synthetic_census_end_to_end():
    tmp = Path(tempfile.mkdtemp(prefix="census02_fx_"))
    try:
        ds, out = tmp / "ds", tmp / "out"
        ds.mkdir()
        fx = Fixture().build(ds)
        assert fx.support["x1"] == 2 and "zz" not in fx.support
        c = census02.Census(ds, out, [0], prior_root=None, inventory_dir=CODE / "v4models" / "inventory")
        rc = c.run({"tests_passed": "fixture"})
        rec = json.loads((out / "CENSUS02_RECEIPT.json").read_text(encoding="utf-8"))
        assert rc == 0 and rec["status"] == "PASS", [x for x in rec["reproduction_checks"] if not x["pass"]]
        S = json.loads((out / "STRATA_CENSUS.json").read_text(encoding="utf-8"))["strata"]
        users = {k: S[k]["users"] for k in ("A0", "A1", "A", "B", "C", "D", "TOTAL")}
        assert users == {"A0": 1, "A1": 1, "A": 2, "B": 2, "C": 2, "D": 1, "TOTAL": 7}, users   # t_only invisible
        ex = {k: S[k]["supervised_examples"] for k in ("A", "B", "C", "D", "TOTAL")}
        assert ex == {"A": 0, "B": 5, "C": 105, "D": 101, "TOTAL": 211}, ex
        assert S["C"]["n_quantiles_linear"]["min"] == 5.0 and S["C"]["n_quantiles_linear"]["max"] == 100.0
        assert S["B"]["n_frequency_table"] == [[1, 1], [4, 1]]
        assert abs(sum(S[k]["share_of_users"] for k in ("A0", "A1", "B", "C", "D")) - 1) < 1e-12
        assert abs(sum(S[k]["share_of_supervised_examples"] for k in ("A0", "A1", "B", "C", "D")) - 1) < 1e-12
        v = {k: S[k]["validation"] for k in S}
        assert v["A0"]["eval_rows"] == 1 and v["A0"]["censored_rows_by_class"]["CEN_NO_NEXT"] == 1
        assert v["A1"]["oov_by_cause"] == {"ABSENT_FROM_TRAIN": 1, "BELOW_SUPPORT": 0}
        assert v["C"]["eval_rows"] == 5 and v["C"]["rankable_targets"] == 4
        assert v["C"]["oov_by_cause"] == {"ABSENT_FROM_TRAIN": 0, "BELOW_SUPPORT": 1}
        assert v["C"]["censored_rows_by_class"]["CEN_LEGACY_BARRIER"] == 1
        assert v["C"]["users_with_ge1_eval_row"] == 2 and v["D"]["users_with_ge1_eval_row"] == 0
        assert S["B"]["train_rows_by_cause"]["oov_below_support"] == 1
        assert S["A1"]["train_rows_by_cause"]["status_NO_OBSERVED_NEXT"] == 3
        assert S["TOTAL"]["validation"]["eval_rows"] == 8           # the TEST eval rows are not counted
        assert rec["rows_materialised_by_split"]["TEST"] == 0
        assert rec["rows_materialised_by_split"]["TRAIN"] == sum(1 for r in fx.rows if r["split"] == "TRAIN")
        P = json.loads((out / "CANDIDATE_N_PROJECTIONS.json").read_text(encoding="utf-8"))
        keys = np.sort(core.selection_keys_v2([digest("c5"), digest("c100")]))
        assert P["pool_size"] == 2 and P["cohorts"]["ALL_C"]["population_sha256"] == population_sha256(keys)
        assert P["cohorts"]["ALL_C"]["catalogue_classes_with_ge1_train_positive_in_cohort"] == 10
        B = json.loads((out / "CATALOGUE_OPTION_B.json").read_text(encoding="utf-8"))
        assert B["K_A"] == 10 and B["K_B"] == 9 and B["A_minus_B"] == 1 and B["B_minus_A"] == 0
        circ = B["circularity"]
        assert circ["C_users_to_B_sparse"] == 1 and circ["D_users_entering_C_under_B"] == 1, circ
        F = json.loads((out / "FL_LOCAL_WORK.json").read_text(encoding="utf-8"))
        w = F["cohorts"]["ALL_C"]["primary_B16_passes2"]
        assert w["total_client_steps_per_sweep"] == 2 * (1 + 7)
        pr = json.loads((out / "PRECISION_PRIOR.json").read_text(encoding="utf-8"))
        assert pr["label"] == "PRIOR_EVIDENCE_FROM_V1_MODELS" and pr["status"].startswith("NOT_COMPUTED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@test
def test_prior_refuses_test_paths():
    s = "/x/sasrec_v1/retro_eval/TEST/A1.ranks.npz"
    assert "TEST" in s.upper().replace("VALIDATION_CONFIRM", "")
    s2 = "/x/sasrec_v1/retro_eval/VALIDATION_CONFIRM/A1.ranks.npz"
    assert "TEST" not in s2.upper().replace("VALIDATION_CONFIRM", "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipt", default=None)
    a = ap.parse_args(argv)
    res = []
    for fn in TESTS:
        try:
            fn()
            res.append({"name": fn.__name__, "pass": True})
            print(f"PASS {fn.__name__}", flush=True)
        except Exception:
            res.append({"name": fn.__name__, "pass": False, "error": traceback.format_exc()[-3000:]})
            print(f"FAIL {fn.__name__}\n{traceback.format_exc()}", flush=True)
    n_ok = sum(r["pass"] for r in res)
    rec = {"tests_passed": n_ok, "tests_failed": len(res) - n_ok, "tests_total": len(res), "results": res,
           "module": "census_v2/test_census02.py", "module_sha256": sha256_file(Path(__file__)),
           "utc_date_u": census02.date_u()}
    if a.receipt:
        Path(a.receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(a.receipt).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: rec[k] for k in ("tests_passed", "tests_failed", "tests_total")}))
    return 0 if n_ok == len(res) else 1


if __name__ == "__main__":
    sys.exit(main())
