"""P2-14a: v4int.audit_e1_mia (the E1 membership-inference audit, PREG2 §7.1) on small synthetic models / data.

Metric math (AUC vs brute force, Clopper-Pearson vs exact binomial tails and closed forms, eps_LB, the A -> B threshold
transfer), the validity gate, the criteria mapping, the TRAIN-window refusals (columns, store, manifest names; a bad
manifest is refused before any store is built), target admission, and the full audit end to end on a SASREC_D64_B2
model with a small K: statuses, no eps_LB when invalid, value-free stdout and byte-identical outputs for the same seed.
No server, no real checkpoint, no frozen data is touched (synthetic batches; tmp directories only).
"""
from __future__ import annotations

import json
import math
import re
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from v4int import audit_e1_mia as E

K = 30
N_USERS, PER_USER = 60, 3


# ------------------------------------------------------------------------------------------------ helpers
def binom_tail_ge(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def binom_cdf(k, n, p):
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, k + 1))


def good_columns(n=6):
    from v4data.common import TRAIN_START_US
    return {"decision_id": np.arange(n, dtype=np.int64), "client_ord": np.repeat(np.arange(n // 2), 2),
            "decision_ts_us": TRAIN_START_US + np.arange(n, dtype=np.int64),
            "target_ts_us": TRAIN_START_US + 1 + np.arange(n, dtype=np.int64),
            "target_class": np.zeros(n, dtype=np.int64),
            "target_status": np.full(n, "OBSERVED_IN_PARTITION", dtype=object),
            "old_split": np.full(n, "TRAIN", dtype=object)}


def all_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from all_keys(v)


def small_bm(seed=1):
    from v4models import build, load_schema
    return build("SASREC", load_schema(), seed, "cpu", K=K, variant="SASREC_D64_B2")


def population(name, member, lo, hi, seed):
    """N_USERS users x PER_USER synthetic decisions, targets uniform in [lo, hi)."""
    from v4models import load_schema
    from v4models.synthetic import synthetic_examples_v2
    n = N_USERS * PER_USER
    g = torch.Generator().manual_seed(seed)
    b = synthetic_examples_v2(n, K, load_schema().widths, g)
    b["target_class"] = torch.randint(lo, hi, (n,), generator=g)
    b["loss_mask"] = torch.ones(n, dtype=torch.bool)
    dec_user = np.repeat(np.arange(N_USERS), PER_USER)
    cols = good_columns(n)
    wc = E.window_check(f"SYNTH_{name}_TRAIN", "TRAIN", {}, cols)

    def batches(bs):
        for a in range(0, n, bs):
            pos = np.arange(a, min(n, a + bs))
            yield pos, {k: (v[a:a + bs] if torch.is_tensor(v) and v.ndim >= 1 else v) for k, v in b.items()}

    return E.Population(name=name, member=member, manifest_id=f"SYNTH_{name}_TRAIN",
                        user_keys=np.arange(N_USERS) + (0 if member else 1000), dec_user=dec_user, batch_fn=batches,
                        n_users_population=N_USERS, record={"window_check": wc},
                        client_idx=np.arange(N_USERS) + (0 if member else 1000),
                        decision_ids=np.arange(n) + (0 if member else 10_000))


def state(bias_lo=None, bias_hi=None):
    """theta: output table and bias zero (every logit 0 -> CE = ln K for every row), optionally +3 on a class range."""
    bm = small_bm()
    sd = {k: v.detach().clone() for k, v in bm.module.state_dict().items()}
    sd["output_embed"].zero_()
    sd["output_bias"].zero_()
    if bias_lo is not None:
        sd["output_bias"][bias_lo:bias_hi] = 3.0
    return sd


def save_central(path: Path, sd, kind_file="endpoint_6.0.pt"):
    path.mkdir(parents=True, exist_ok=True)
    kind ={"endpoint_6.0.pt": "endpoint_6.0", "best.pt": "best"}[kind_file]
    meta = {"model_variant": "SASREC_D64_B2", "mark": 6.0 if kind_file == "endpoint_6.0.pt" else 2.5}
    torch.save({"kind": kind, "meta": meta, "model": sd}, path / kind_file)
    return path / kind_file


def save_fl(path: Path, sd, kind_file="best.pt"):
    from fl.checkpoint import atomic_save
    bm = small_bm()
    bm.module.load_state_dict(sd, strict=True)
    payload = {"kind": "weights", "run_id": "SYNTH", "config_digest": "synthetic", "round": 1, "efe": 6.0,
               "theta": dict(bm.adapter.broadcast_state(clone=True))}
    atomic_save(payload, path / kind_file)
    return path / kind_file


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    d = tmp_path_factory.mktemp("e1_ckpts")
    leak_m, leak_n = state(0, 15), state(15, 30)          # favour the members' / the non-members' classes
    return {"PRE": save_central(d / "pre", state()), "PRE_LEAKY": save_central(d / "pre_leaky", leak_m),
            "FA_WARM_FROZEN": save_fl(d / "fa", state()), "P2_WSTAR": save_fl(d / "w", leak_m),
            "P2_WSTAR_DP8": save_fl(d / "dp8", leak_m, "endpoint_6.0.pt"),
            "P2_WSTAR_DP1": save_fl(d / "dp1", leak_m, "endpoint_6.0.pt"),
            "P2_C1_FT_CENTRAL_FULL": save_central(d / "c1", leak_n, "best.pt")}


@pytest.fixture(scope="module")
def pops():
    return population("MEMBERS", True, 0, 15, 11), population("NONMEMBERS", False, 15, 30, 12)


def run(tmp_path, ckpts, pops, names, pre="PRE", tag="out"):
    lines = []
    tg = E.parse_targets([f"PRE={ckpts[pre]}"] + [f"{n}={ckpts[n]}" for n in names])
    audit = E.run_audit(tmp_path / tag, targets=tg, members=pops[0], nonmembers=pops[1], adapter=small_bm().adapter,
                        device=torch.device("cpu"), read_target=E.read_target_plain, n_boot=200, say=lines.append)
    return audit, lines


MAND = ["FA_WARM_FROZEN", "P2_WSTAR", "P2_C1_FT_CENTRAL_FULL"]


# ------------------------------------------------------------------------------------------------ metric math
def test_auc_edges_and_bruteforce():
    assert E.auc([3, 4], [1, 2]) == 1.0 and E.auc([1, 2], [3, 4]) == 0.0
    assert E.auc([1, 1], [1, 1, 1]) == 0.5
    rng = np.random.default_rng(0)
    pos, neg = rng.integers(0, 5, 37).astype(float), rng.integers(0, 5, 41).astype(float)   # many ties
    brute = np.mean([(p > q) + 0.5 * (p == q) for p in pos for q in neg])
    assert abs(E.auc(pos, neg) - brute) < 1e-12


def test_bootstrap_same_seed_same_ci_and_paired_indices():
    rng = np.random.default_rng(1)
    pos, neg = rng.normal(0.2, 1, 50), rng.normal(0, 1, 50)
    a = E.bootstrap_auc(pos, neg, n_boot=100, seed=7)
    b = E.bootstrap_auc(pos, neg, n_boot=100, seed=7)
    assert a == b and a["ci95"][0] <= a["auc"] <= a["ci95"][1]
    assert E.bootstrap_auc(np.zeros(50), np.zeros(50), n_boot=50, seed=7)["ci95"] == [0.5, 0.5]


def test_betainc_identities():
    assert abs(E.betainc(3.0, 3.0, 0.5) - 0.5) < 1e-12
    for a, b, x in ((2.0, 5.0, 0.3), (0.7, 1.9, 0.8), (40.0, 3.0, 0.95)):
        assert abs(E.betainc(a, b, x) + E.betainc(b, a, 1 - x) - 1.0) < 1e-12
    # I_p(k, n-k+1) = P(X >= k), X ~ Bin(n, p)
    assert abs(E.betainc(3, 8, 0.2) - binom_tail_ge(3, 10, 0.2)) < 1e-12


@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (1, 10), (5, 20), (3, 5000), (50, 5000), (4990, 5000)])
def test_clopper_pearson(k, n):
    lo, hi = E.cp_bounds(k, n)
    a = 0.05
    if k == 0:
        assert lo == 0.0 and abs(hi - (1 - (a / 2) ** (1 / n))) < 1e-10
    elif k == n:
        assert hi == 1.0 and abs(lo - (a / 2) ** (1 / n)) < 1e-10
    elif n <= 20:
        assert abs(binom_tail_ge(k, n, lo) - a / 2) < 1e-9 and abs(binom_cdf(k, n, hi) - a / 2) < 1e-9
    else:                                   # large n: the beta identity (exact tails would underflow pure Python)
        assert abs(E.betainc(k, n - k + 1, lo) - a / 2) < 1e-9
        assert abs(1 - E.betainc(k + 1, n - k, hi) - a / 2) < 1e-9
    assert 0.0 <= lo <= k / n <= hi <= 1.0
    with pytest.raises(E.E1Refused):
        E.cp_bounds(3, 2)


def test_eps_lower_bound():
    assert abs(E.eps_lower_bound(0.5, 0.01)["eps_lb"] - math.log(50)) < 1e-12
    z = E.eps_lower_bound(0.0, 0.01)
    assert z == {"eps_lb_raw": None, "eps_lb": 0.0}
    neg = E.eps_lower_bound(0.001, 0.01)
    assert neg["eps_lb"] == 0.0 and neg["eps_lb_raw"] < 0


def test_allowed_false_positives_exact():
    assert E.allowed_false_positives(5000, Fraction(1, 100)) == 50
    assert E.allowed_false_positives(5000, Fraction(1, 1000)) == 5
    assert E.allowed_false_positives(99, Fraction(1, 100)) == 0
    assert E.allowed_false_positives(100, Fraction(1, 100)) == 1


def test_threshold_transfer_a_to_b():
    neg_a = np.arange(100, dtype=float)                       # 0 .. 99
    assert E.threshold_at_fpr(neg_a, Fraction(1, 100)) == 98.0   # k = 1: only 99 exceeds it
    assert E.threshold_at_fpr(neg_a, Fraction(1, 1000)) == 99.0  # k = 0: none exceeds it
    assert E.threshold_at_fpr(np.full(10, 5.0), Fraction(1, 100)) == 5.0
    pos_b, neg_b = np.array([97.0, 98.0, 98.5, 100.0]), np.array([10.0, 98.0, 99.0, 120.0, 1.0])
    t = E.transfer(np.zeros(3), neg_a, pos_b, neg_b, Fraction(1, 100))
    assert (t["threshold"], t["tp_b"], t["fp_b"], t["fp_a"], t["allowed_fp_a"]) == (98.0, 2, 2, 1, 1)
    assert t["tpr_b"] == 0.5 and t["fpr_b"] == 0.4
    assert t["tpr_cp95"] == E.cp_bounds(2, 4) and t["fpr_cp95"] == E.cp_bounds(2, 5)
    assert t["eps_lb"] == max(0.0, math.log(t["tpr_cp95"][0] / t["fpr_cp95"][1]))


def test_score_metrics_uses_threshold_fpr_one_percent():
    rng = np.random.default_rng(3)
    tm, tn = rng.normal(1, 1, 400), rng.normal(0, 1, 400)
    half = np.zeros(400, dtype=bool)
    half[:200] = True
    m = E.score_metrics(tm, tn, half, half, n_boot=50, seed=1)
    assert m["eps_lb"] == m["tpr_at_fpr"]["1/100"]["eps_lb"] and m["eps_lb_threshold_fpr"] == "1/100"
    assert set(m["tpr_at_fpr"]) == {"1/1000", "1/100"}
    s = E.strip_eps(m)
    assert not [k for k in all_keys(s) if str(k).startswith("eps_lb") and k != "eps_lb_reported"]
    with pytest.raises(E.E1Refused):
        E.score_metrics(np.array([np.nan, 1.0]), np.array([0.0, 1.0]), [True, False], [True, False], n_boot=5, seed=1)


# ------------------------------------------------------------------------------------------------ criteria
def _m(auc_lo, tpr_lo, eps):
    return {"auc": {"ci95": [auc_lo, auc_lo + 0.1]}, "tpr_at_fpr": {"1/100": {"tpr_cp95": [tpr_lo, 1.0]}},
            "eps_lb": eps}


def test_validity_gate():
    assert E.validity({"auc": {"ci95": [0.47, 0.53]}})["status"] == "VALID"
    assert E.validity({"auc": {"ci95": [0.5, 0.6]}})["status"] == "VALID"
    assert E.validity({"auc": {"ci95": [0.501, 0.6]}})["status"] == "AUDIT_INVALID"
    assert E.validity({"auc": {"ci95": [0.4, 0.499]}})["status"] == "AUDIT_INVALID"


def test_criteria_mapping():
    assert E.classify("P2_WSTAR_DP8", _m(0.5, 0.0, 7.99))["status"] == "PASS"
    assert E.classify("P2_WSTAR_DP8", _m(0.5, 0.0, 8.0))["status"] == "PASS"
    assert E.classify("P2_WSTAR_DP8", _m(0.5, 0.0, 8.01))["status"] == "FAIL_INVESTIGATE"
    assert E.classify("P2_WSTAR_DP1", _m(0.5, 0.0, 1.2), 1.0)["status"] == "FAIL_INVESTIGATE"
    with pytest.raises(E.E1Refused):
        E.classify("P2_WSTAR_DP1", _m(0.5, 0.0, 0.2), 8.0)              # accountant epsilon must be the registered
    for lab in ("FA_WARM_FROZEN", "P2_WSTAR", "P2_C1_FT_CENTRAL_FULL"):
        assert E.classify(lab, _m(0.551, 0.0, 9.0))["status"] == "MATERIAL_LEAKAGE"
        assert E.classify(lab, _m(0.5, 0.031, 9.0))["status"] == "MATERIAL_LEAKAGE"
        assert E.classify(lab, _m(0.55, 0.03, 9.0))["status"] == "NO_MATERIAL_LEAKAGE"   # strict '>' both
    with pytest.raises(E.E1Refused):
        E.classify("PRE", _m(0.5, 0.0, 0.0))


# ------------------------------------------------------------------------------------------------ TRAIN window
def test_window_check_passes_and_records():
    rec = E.window_check("MAIN_TRAIN_5_100_131072", "TRAIN", {}, good_columns())
    assert rec["all_ok"] and rec["n_rows_checked"] == 6 and all(rec["checks"].values())


@pytest.mark.parametrize("mutate", ["decision_late", "target_late", "decision_early", "old_split", "v4_split",
                                    "status", "kind", "header", "missing"])
def test_window_check_refuses(mutate):
    from v4data.common import TRAIN_END_US, TRAIN_START_US
    cols, kind, header = good_columns(), "TRAIN", {}
    if mutate == "decision_late":
        cols["decision_ts_us"][-1] = TRAIN_END_US                    # first VALIDATION microsecond
    elif mutate == "target_late":
        cols["target_ts_us"][-1] = TRAIN_END_US + 5                  # a label inside the VALIDATION window
    elif mutate == "decision_early":
        cols["decision_ts_us"][0] = TRAIN_START_US - 1
    elif mutate == "old_split":
        cols["old_split"][2] = "VALIDATION_SELECT"
    elif mutate == "v4_split":
        cols["v4_split"] = np.array(["TRAIN"] * 5 + ["RETRO_TEST"], dtype=object)
    elif mutate == "status":
        cols["target_status"][1] = "NO_OBSERVED_NEXT"
    elif mutate == "kind":
        kind = "EVAL"
    elif mutate == "header":
        header = {"v4_split": "VALIDATION"}
    elif mutate == "missing":
        cols.pop("target_ts_us")
    with pytest.raises(E.E1Refused):
        E.window_check("MAIN_TRAIN_5_100_131072", kind, header, cols)


@pytest.mark.parametrize("mid", ["MAIN_VALIDATION_5_100_131072", "UNEW_HOLDOUT_VALIDATION_32768", "MAIN_TEST_X",
                                 "M07_TEST", "COLD_H_SUPPORT_QUERY"])
def test_validation_and_test_manifests_refused_by_name(mid):
    with pytest.raises(E.E1Refused):
        E.refuse_window_manifest_id(mid)


def test_store_window_check():
    from types import SimpleNamespace as NS
    ids, tgt = np.arange(4), np.zeros(4, dtype=np.int64)
    ds = NS(split="TRAIN", decision_id=ids, target_class=tgt, loss_mask=np.ones(4, dtype=bool))
    assert E.store_window_check("M", NS(splits=("TRAIN",)), ds, ids, tgt, ("TRAIN",))["all_ok"]
    for info, parts, d in ((NS(splits=("TRAIN", "VALIDATION_SELECT")), ("TRAIN",), ds),
                           (NS(splits=("TRAIN",)), ("TRAIN", "VALIDATION_SELECT"), ds),
                           (NS(splits=("TRAIN",)), ("TRAIN",), NS(**dict(vars(ds), split="VALIDATION"))),
                           (NS(splits=("TRAIN",)), ("TRAIN",), NS(**dict(vars(ds), decision_id=ids + 1)))):
        with pytest.raises(E.E1Refused):
            E.store_window_check("M", info, d, ids, tgt, parts)


def test_population_build_refuses_a_validation_row_before_any_store(monkeypatch):
    """A verified manifest holding one row outside the TRAIN window is refused before v4int.cache builds anything."""
    from types import SimpleNamespace as NS
    import v4int.cache as C
    from v4data.common import TRAIN_END_US

    def boom(*a, **k):
        raise AssertionError("build_subset_store must not be reached")

    monkeypatch.setattr(C, "build_subset_store", boom)
    cols = good_columns()
    cols["target_ts_us"][3] = TRAIN_END_US + 1
    vm = NS(header={"status": "MAIN"}, kind="TRAIN", columns=cols, transform_state_self_sha256="x")
    with pytest.raises(E.E1Refused, match="TRAIN-window"):
        E.build_train_population("MEMBERS", 2, np.random.default_rng(0), registry_path=Path("unused"),
                                 verify=lambda mid: vm)
    vm2 = NS(header={"status": "UNSEEN_HOLDOUT"}, kind="TRAIN", columns=good_columns(), transform_state_self_sha256="x")
    with pytest.raises(E.E1Refused, match="header status"):
        E.build_train_population("MEMBERS", 2, np.random.default_rng(0), registry_path=Path("unused"),
                                 verify=lambda mid: vm2)


def test_population_build_on_real_train_rows(real_env, monkeypatch):
    """The real builder on the INT-01 locked test environment (real TRAIN rows of bucket 0, M01 layout): verified
    manifest -> window check -> seeded users -> subset store of their TRAIN decisions (USER_HISTORY_L50, partitions TRAIN
    only) -> batches equal to the manifest. The registry status is the test header's; the store reads bucket 0 only."""
    import v4int.cache as C
    from intkit import REAL_DS
    from v4int.verify import verify_manifest
    orig = C.build_subset_store
    monkeypatch.setattr(C, "build_subset_store", lambda ids, **kw: orig(ids, **dict(kw, buckets=[0])))
    monkeypatch.setitem(E.POPULATIONS, "MEMBERS", dict(E.POPULATIONS["MEMBERS"], status="BUILT_PRE_LOCK"))
    vm_of = lambda mid: verify_manifest("M01_COHORT_TRAIN", real_env["paths"], expect_kind="TRAIN")  # noqa: E731
    pop = E.build_train_population("MEMBERS", 20, np.random.default_rng(E.SEED), registry_path=Path("unused"),
                                   ds_root=REAL_DS, verify=vm_of)
    r = pop.record
    assert r["window_check"]["all_ok"] and r["store_check"]["all_ok"] and all(r["store_check"]["checks"].values())
    assert r["store_partitions"] == ["TRAIN"] and r["n_users_selected"] == 20 and pop.user_keys.size == 20
    tab = real_env["tables"]["M01_COHORT_TRAIN"]["decisions"]
    want = np.sort(tab["decision_id"][np.isin(tab["client_ord"], pop.user_keys)])
    assert np.array_equal(pop.decision_ids, want)
    seen = []
    for pos, b in pop.batches(64):
        assert np.array_equal(b["decision_id"].numpy(), want[pos])
        seen.append(pos)
    assert np.array_equal(np.concatenate(seen), np.arange(want.size))
    again = E.build_train_population("MEMBERS", 20, np.random.default_rng(E.SEED), registry_path=Path("unused"),
                                     ds_root=REAL_DS, verify=vm_of)
    assert np.array_equal(again.user_keys, pop.user_keys)


def test_overlapping_populations_refused(pops):
    m = pops[0]
    with pytest.raises(E.E1Refused):
        E.check_disjoint(m, m)


# ------------------------------------------------------------------------------------------------ users / targets
def test_selection_and_split_deterministic():
    keys = np.repeat(np.arange(500), 3)
    a = E.select_users(keys, 100, np.random.default_rng(E.SEED))
    b = E.select_users(keys, 100, np.random.default_rng(E.SEED))
    assert np.array_equal(a, b) and a.size == 100 and np.all(np.diff(a) > 0)
    h = E.split_half(100, np.random.default_rng(E.SPLIT_STREAM))
    assert h.sum() == 50 and np.array_equal(h, E.split_half(100, np.random.default_rng(E.SPLIT_STREAM)))
    with pytest.raises(E.E1Refused):
        E.select_users(np.arange(5), 6, np.random.default_rng(0))


def test_parse_targets(ckpts, tmp_path):
    base = [f"{n}={ckpts[n]}" for n in ["PRE"] + MAND]
    tg = E.parse_targets(base)
    assert list(tg) == ["PRE", "FA_WARM_FROZEN", "P2_WSTAR", "P2_C1_FT_CENTRAL_FULL"]
    bad_cases = [base[1:],                                            # PRE missing
                 base[:-1],                                           # a mandatory non-DP target missing
                 base + [f"PRE={ckpts['PRE']}"],                      # repeated
                 base + [f"P2_FOO={ckpts['PRE']}"],                   # unknown label
                 base + [f"P2_WSTAR_DP8={ckpts['P2_WSTAR']}"],        # DP arm on best.pt (outside the accountant)
                 [f"PRE={ckpts['P2_C1_FT_CENTRAL_FULL']}"] + base[1:],  # PRE must be endpoint_6.0.pt
                 base + [f"P2_WSTAR_DP1={tmp_path}"],                 # a directory
                 base + ["P2_WSTAR_DP1"]]                             # not LABEL=PATH
    for c in bad_cases:
        with pytest.raises(E.E1Refused):
            E.parse_targets(c)


def test_apply_target_formats(ckpts):
    bm = small_bm(seed=5)
    d1 = E.apply_target(bm.adapter, E.read_target_plain("PRE", ckpts["PRE"]))
    d2 = E.apply_target(bm.adapter, E.read_target_plain("FA_WARM_FROZEN", ckpts["FA_WARM_FROZEN"]))
    assert d1 == d2                                                   # the same theta in both formats
    item = E.read_target_plain("P2_WSTAR", ckpts["P2_WSTAR"])
    item["obj"] = dict(item["obj"], theta={k: v for k, v in item["obj"]["theta"].items() if k != "output_bias"})
    with pytest.raises(E.E1Refused):
        E.apply_target(bm.adapter, item)
    item = E.read_target_plain("PRE", ckpts["PRE"])
    item["expect_state_digest"] = "0" * 64
    with pytest.raises(E.E1Refused):
        E.apply_target(bm.adapter, item)


def test_pre_sha_pin_refused_before_any_source_read(ckpts):
    with pytest.raises(E.E1Refused, match="registered theta_pre"):
        E.verify_source("PRE", ckpts["PRE"])


def test_dp_source_checks(monkeypatch):
    import v4int.launch_fl as LF
    monkeypatch.setattr(LF, "check_p2_dp_spec", lambda spec: None)       # the spec builder itself is launch_fl's
    pins = {"checkpoint_sha256": E.P2.PRE_CHECKPOINT_SHA256, "state_digest": E.P2.PRE_STATE_DIGEST}

    def src(arm="P2_WSTAR_DP8", eps=8.0, rid="P2_WSTAR_DP8_SMALL_s2026", init=pins):
        spec = {"arm": arm, "init_from": dict(init), "cfg": {"seed": 2026}}
        return {"arm": arm, "launch": {"arm_spec": spec, "arm_spec_sha256": LF.spec_digest(spec), "run_id": rid,
                                       "dp_z": {"epsilon": eps, "z": 0.5642}}}

    assert E.check_dp_source("P2_WSTAR_DP8", src())["accountant_epsilon"] == 8.0
    for bad in (src(eps=1.0), src(rid="P2_WSTAR_DP8_SMALL_s2027"), src(arm="P2_WSTAR_FA1024"),
                src(init=dict(pins, state_digest="0" * 64))):
        with pytest.raises(E.E1Refused):
            E.check_dp_source("P2_WSTAR_DP8", bad)
    with pytest.raises(E.E1Refused):
        E.check_dp_source("P2_WSTAR_DP1", src())                          # DP8 run offered as DP1


def test_cli_refusal_exit_code(ckpts, tmp_path, capsys):
    rc = E.main(["--targets", f"PRE={ckpts['PRE']}", "--out", str(tmp_path / "o"), "--device", "cpu", "--dry-run"])
    err = capsys.readouterr()
    assert rc == 1 and "E1_AUDIT REFUSED" in err.err and not (tmp_path / "o").exists()


def test_spec_text_is_the_registered_block():
    s = E.spec_text()
    assert s["file_sha256"] == E.PREG2_SHA256 and s["text"].startswith("## 7. ") and "### 7.1 E1" in s["text"]


# ------------------------------------------------------------------------------------------------ end to end
def test_audit_valid_statuses_and_files(tmp_path, ckpts, pops):
    audit, lines = run(tmp_path, ckpts, pops, MAND + ["P2_WSTAR_DP8"])
    t = audit["targets"]
    assert audit["audit_status"] == "VALID" and audit["eps_lb_reported"] is True
    assert audit["validity"]["auc_ci95"] == [0.5, 0.5]               # PRE: every logit 0 -> every raw CE ties
    assert t["PRE"]["status"] == "VALID_NULL"
    assert t["FA_WARM_FROZEN"]["status"] == "NO_MATERIAL_LEAKAGE"    # = theta_pre: calibrated score 0
    assert t["P2_WSTAR"]["status"] == "MATERIAL_LEAKAGE"             # favours the members' classes
    assert t["P2_WSTAR"]["calibrated"]["auc"]["auc"] == 1.0
    assert t["P2_C1_FT_CENTRAL_FULL"]["status"] == "NO_MATERIAL_LEAKAGE"   # favours the non-members
    dp8 = t["P2_WSTAR_DP8"]
    assert dp8["status"] == "PASS" and 0.0 < dp8["eps_lb"] <= 8.0     # perfect attack, 30 / 30 on half B
    tb = dp8["calibrated"]["tpr_at_fpr"]["1/100"]
    assert (tb["tp_b"], tb["fp_b"], tb["n_members_b"], tb["n_nonmembers_b"]) == (30, 0, 30, 30)
    assert t["P2_WSTAR_DP1"]["status"] == "NOT_AVAILABLE" and audit["targets_not_available"] == ["P2_WSTAR_DP1"]
    out = tmp_path / "out"
    a = json.loads((out / E.AUDIT_FILE).read_text(encoding="utf-8"))
    assert a == audit and E.with_self_sha(a)["self_sha256"] == a["self_sha256"]
    sc = json.loads((out / E.SCORES_FILE).read_text(encoding="utf-8"))
    assert E.with_self_sha(sc)["self_sha256"] == sc["self_sha256"]
    import hashlib
    assert hashlib.sha256((out / E.SCORES_FILE).read_bytes()).hexdigest() == a["per_user_scores"]["sha256"]
    assert len(sc["users"]["client_ord"]) == 2 * N_USERS and set(sc["targets"]) == set(a["inputs"])
    assert all(v == 0.0 for v in sc["targets"]["FA_WARM_FROZEN"]["score_calibrated"])
    assert a["window_check"]["members"]["all_ok"] and a["window_check"]["disjoint"]["no_shared_decision"]
    assert a["protocol"]["registered_scale"] is False                # test scale, recorded as such
    for ln in lines:                                                  # S8: no metric value on stdout
        assert not re.search(r"\d\.\d", ln), ln
        assert ln.startswith(("E1 STEP", "E1 TARGET", "E1 AUDIT", "E1 OUTPUT"))


def test_audit_dp1_fail_investigate(tmp_path, ckpts, pops):
    audit, _ = run(tmp_path, ckpts, pops, MAND + ["P2_WSTAR_DP1"])
    d = audit["targets"]["P2_WSTAR_DP1"]
    assert d["status"] == "FAIL_INVESTIGATE" and d["eps_lb"] > 1.0 and d["accountant_epsilon"] == 1.0


def test_audit_invalid_reports_no_eps_and_stops(tmp_path, ckpts, pops):
    audit, lines = run(tmp_path, ckpts, pops, MAND + ["P2_WSTAR_DP8"], pre="PRE_LEAKY")
    assert audit["audit_status"] == "AUDIT_INVALID" and audit["validity"]["includes_0_5"] is False
    assert audit["targets"]["PRE"]["status"] == "AUDIT_INVALID"
    for lab in MAND + ["P2_WSTAR_DP8"]:
        assert audit["targets"][lab]["status"] == "NOT_SCORED_AUDIT_INVALID"
        assert audit["inputs"][lab]["scored"] is False
    assert not [k for k in all_keys(audit["targets"]) if str(k).startswith("eps_lb") and k != "eps_lb_reported"]
    sc = json.loads((tmp_path / "out" / E.SCORES_FILE).read_text(encoding="utf-8"))
    assert list(sc["targets"]) == ["PRE"] and sc["eps_lb_reported"] is False
    assert "E1 STEP validity: AUDIT_INVALID" in lines


def test_audit_same_seed_same_bytes(tmp_path, ckpts, pops):
    run(tmp_path, ckpts, pops, MAND + ["P2_WSTAR_DP8"], tag="a")
    run(tmp_path, ckpts, pops, MAND + ["P2_WSTAR_DP8"], tag="b")
    for f in (E.AUDIT_FILE, E.SCORES_FILE):
        assert (tmp_path / "a" / f).read_bytes() == (tmp_path / "b" / f).read_bytes()


def test_audit_refuses_existing_out_dir(tmp_path, ckpts, pops):
    (tmp_path / "out").mkdir()
    with pytest.raises(E.E1Refused):
        run(tmp_path, ckpts, pops, MAND)
