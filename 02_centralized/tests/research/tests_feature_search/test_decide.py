"""v4fs.decide: the drop rule, the relative drop (seed mean, ratio bootstrapped directly) and the acceptance search.

Rule (binding, Fable R2 D3 amendment 5 + R4): ACCEPT iff for the WORSE family the seed-mean relative drop versus L0
has point <= 1.0 % AND 95 % UB <= 2.0 %; fallback: restore the most important dropped group once; then L3 if L3
passes, else L0. Synthetic per-user statistics only.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from fskit import assert_raises, nc
from v4eval.bootstrap import UserStats
from v4fs import decide as D
from v4fs import importance as I

N_USERS = 400


def stats(m, seed_label=""):
    m = np.asarray(m, dtype=np.float64)
    n = np.ones_like(m)
    return UserStats(np.arange(m.size, dtype=np.int64), m.copy(), n, m.copy(), n.copy(), seed_label)


def base_m(seed=0):
    return np.random.default_rng(seed).uniform(0.05, 0.4, N_USERS)


def test_relative_drop_exact_scaling_and_seed_mean():
    c = {2026: stats(base_m(1)), 2027: stats(base_m(2))}
    m = {s: stats(c[s].S * (1 - 0.005)) for s in c}
    d = D.relative_drop(c, m)
    assert abs(d["point"] - 0.5) < 1e-9 and abs(d["ub"] - 0.5) < 1e-9 and abs(d["ci95"][0] - 0.5) < 1e-9
    assert all(abs(d["per_seed"][s]["point"] - 0.5) < 1e-9 for s in ("2026", "2027"))
    assert d["n_users"] == N_USERS and d["seeds"] == [2026, 2027]
    same = base_m(3)                                                   # equal C per seed: the seed mean is the mean
    c2 = {2026: stats(same), 2027: stats(same)}
    m2 = {2026: stats(same), 2027: stats(same * 0.98)}
    d2 = D.relative_drop(c2, m2)
    assert abs(d2["point"] - 1.0) < 1e-9 and abs(d2["per_seed"]["2027"]["point"] - 2.0) < 1e-9
    assert abs(d2["mean_of_per_seed_points"] - 1.0) < 1e-9
    assert_raises(D.DecisionError, D.relative_drop, c, {2026: m[2026]})
    other = {s: UserStats(np.arange(N_USERS) + 1, v.S, v.N, v.SR, v.NR) for s, v in m.items()}
    assert_raises(D.DecisionError, D.relative_drop, c, other)


def test_the_ratio_ci_is_bootstrapped_directly():
    rng = np.random.default_rng(5)
    c = {2026: stats(base_m(1)), 2027: stats(base_m(2))}
    m = {s: stats(np.clip(c[s].S - 0.002 + rng.normal(0, 0.02, N_USERS), 0, 1)) for s in c}
    d = D.relative_drop(c, m)
    lo, hi = d["ci95"]
    assert lo < d["point"] < hi and d["ub"] == hi
    assert d["W_sha256"] and d["n_resamples_zero_denominator"] == 0


@pytest.mark.parametrize("point,ub,ok", [(1.0, 2.0, True), (0.3, 1.99, True), (1.0000001, 1.5, False),
                                         (0.5, 2.0000001, False), (-0.4, 0.8, True), (0.5, None, False)])
def test_pass_boundaries(point, ub, ok):
    assert D.passes({"point": point, "ub": ub}) is ok


def test_accept_uses_the_worse_family():
    a = D.accept({"GRU": {"point": 0.9, "ub": 1.9}, "SASREC": {"point": 0.4, "ub": 2.5}})
    assert a["worse_family"] == "GRU" and a["accept"] and not a["both_pass"]
    b = D.accept({"GRU": {"point": 0.2, "ub": 0.9}, "SASREC": {"point": 1.2, "ub": 1.8}})
    assert b["worse_family"] == "SASREC" and not b["accept"]
    t = D.accept({"GRU": {"point": 0.5, "ub": 2.1}, "SASREC": {"point": 0.5, "ub": 1.0}})
    assert t["worse_family"] == "GRU" and not t["accept"]                     # tie on the point: the larger UB
    assert_raises(D.DecisionError, D.accept, {"GRU": {"point": 0.1, "ub": 0.2}})


def _imp(values):
    return {g: {"importance": v, "ci95": [-1, 1], "within_noise": True} for g, v in values.items()}


def test_most_important_dropped_group():
    imp = {"GRU": _imp({"G2": 0.001, "G8": 0.004, "G12c": 0.004}), "SASREC": _imp({"G2": 0.0, "G8": 0.001,
                                                                                    "G12c": 0.002})}
    order = list(I.FS_GROUPS)
    assert D.most_important_dropped(["G2", "G8", "G12c"], imp, "GRU", order) == "G12c"     # tie: the other family
    assert D.most_important_dropped(["G2", "G8", "G12c"], imp, "SASREC", order) == "G12c"
    imp["SASREC"]["G12c"]["importance"] = 0.001
    assert D.most_important_dropped(["G2", "G8", "G12c"], imp, "GRU", order) == "G8"        # full tie: order
    assert_raises(D.DecisionError, D.most_important_dropped, [], imp, "GRU", order)


def _res(ok, worse="GRU"):
    return {"accept": ok, "worse_family": worse}


def test_search_sequence():
    order = list(I.FS_GROUPS)
    imp = {"GRU": _imp({"G13": 0.001, "G8": 0.003, "G12c": 0.002}),
           "SASREC": _imp({"G13": 0.0, "G8": 0.0, "G12c": 0.0})}
    L3, L0 = D.L3_ID, D.L0_ID
    assert D.search(dropped=[], results={})["schema_id"] == L3
    assert D.search(dropped=[], results={L3: _res(True)}) == {"action": "LOCK", "schema_id": L3,
                                                             "reason": "L3 passes the rule", "trace": [
                                                                 "no group is within noise for both families: S* = L3"]}
    assert D.search(dropped=[], results={L3: _res(False)})["schema_id"] == L0
    assert (L0, L3) == ("L0S_CLEAN_V2_SESSION", "L3S_NEAR_DUPLICATE_SESSION")          # SCHEMA-01R1 reference / base
    assert_raises(D.DecisionError, D.search, dropped=["G1", "G8"], importance_by_family=imp,
                  results={L3: _res(True)}, order=order)                              # G1 is REQUIRED (DL 47)
    drop = ["G12c", "G13", "G8"]
    s1 = D.sstar_id(drop)
    assert s1 == "L3S_NEAR_DUPLICATE_SESSION-minus-G12c+G13+G8"
    st = D.search(dropped=drop, importance_by_family=imp, results={L3: _res(True)}, order=order)
    assert st["action"] == "TRAIN" and st["schema_id"] == s1 and st["dropped_groups"] == ["G8", "G12c", "G13"]
    st = D.search(dropped=drop, importance_by_family=imp, results={L3: _res(False), s1: _res(True)}, order=order)
    assert st == dict(st, action="LOCK", schema_id=s1)
    st = D.search(dropped=drop, importance_by_family=imp, results={L3: _res(True), s1: _res(False, "GRU")},
                  order=order)
    s2 = D.sstar_id(["G13", "G12c"])
    assert st["action"] == "TRAIN" and st["schema_id"] == s2 and st["restored_group"] == "G8"
    for s2_ok, l3_ok, want in ((True, False, s2), (False, True, L3), (False, False, L0)):
        st = D.search(dropped=drop, importance_by_family=imp, order=order,
                      results={L3: _res(l3_ok), s1: _res(False, "GRU"), s2: _res(s2_ok)})
        assert st["action"] == "LOCK" and st["schema_id"] == want, (s2_ok, l3_ok)
    one = D.sstar_id(["G8"])                                           # one dropped group: restoring it is L3
    for l3_ok, want in ((True, L3), (False, L0)):
        st = D.search(dropped=["G8"], importance_by_family=imp, order=order,
                      results={L3: _res(l3_ok), one: _res(False, "GRU")})
        assert st["action"] == "LOCK" and st["schema_id"] == want and st["restored_group"] == "G8"


def _write_run(root, sid, fam, seed, endpoint, perms=None):
    d = root / f"{sid}_{fam}_{seed}"
    d.mkdir(parents=True)
    st = {"endpoint": stats(endpoint)}
    if perms is not None:
        st["base"] = stats(endpoint)
        for g in I.FS_GROUPS:
            for p in I.PERM_SEEDS:
                st[f"{g}|{p}"] = stats(perms.get(g, endpoint))
    I.save_stats(d / "FS_USERSTATS.npz", st, {"seed": seed})
    (d / "FS_SUMMARY.json").write_text(json.dumps({"run_label": D.RUN_LABEL, "schema": {"schema_id": sid},
                                                   "family": fam, "seed": seed,
                                                   "userstats_file": "FS_USERSTATS.npz"}), encoding="utf-8")
    return d


def test_decide_runs_end_to_end(tmp_path):
    dirs = []
    for fam in D.FAMILIES:
        for seed in (2026, 2027):
            m = base_m(seed + (0 if fam == "GRU" else 10))
            dirs.append(_write_run(tmp_path, D.L0_ID, fam, seed, m))
            l3 = m * 0.999
            perms = {g: l3 * 0.9 for g in I.FS_GROUPS if g not in ("G1", "G8", "G12c")}  # G1/G8/G12c: no effect
            dirs.append(_write_run(tmp_path, D.L3_ID, fam, seed, l3, perms))
    res = D.decide_runs(D.load_runs(dirs))
    assert res["dropped_groups"] == ["G8", "G12c"]                # G1 has no effect but is REQUIRED
    assert res["acceptance"][D.L3_ID]["accept"] and res["next"]["action"] == "TRAIN"
    assert res["next"]["schema_id"] == D.sstar_id(["G8", "G12c"])
    assert all(abs(res["relative_drops"][D.L3_ID][f]["point"] - 0.1) < 1e-9 for f in D.FAMILIES)
    assert "value" not in json.dumps(res["relative_drops"]), "no absolute MRR in the decision record"
    s1 = D.sstar_id(["G8", "G12c"])
    for fam in D.FAMILIES:
        for seed in (2026, 2027):
            m = base_m(seed + (0 if fam == "GRU" else 10))
            dirs.append(_write_run(tmp_path, s1, fam, seed, m * (0.995 if fam == "GRU" else 0.97)))
    res = D.decide_runs(D.load_runs(dirs))
    a = res["acceptance"][s1]
    assert a["worse_family"] == "SASREC" and not a["accept"]
    assert res["next"]["action"] == "TRAIN" and res["next"]["restored_group"] in ("G8", "G12c")


@nc("a rule that ignores the upper bound accepts a drop whose CI reaches 2.5 %")
def test_nc_rule_without_the_upper_bound():
    def naive(d):
        return d["point"] <= D.POINT_MAX_PERCENT
    assert not naive({"point": 0.9, "ub": 2.5}), "the naive rule accepted a drop with UB 2.5 %"
