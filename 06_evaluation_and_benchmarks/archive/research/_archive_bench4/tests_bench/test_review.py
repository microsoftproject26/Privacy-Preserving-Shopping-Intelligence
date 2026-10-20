"""REV-BM-02 fixes: B1 (FT TEST context), M1 (preflight, late atomic TEST_LOOK), M2 (own max_len), M3 (LOCK rules), M4
(validation population), M6 (run-time mirrors), M7 (no overwrite), B10 Q2 (PF), Q5 (Wave-1 cold count), PRE seed."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from conftest import make_release
from v4bench import central, common, data as D, device, final, fl_engine, recipe, selection
from v4bench.common import BenchRefused, read_json

DS = "s3_beauty"


@pytest.fixture()
def small(monkeypatch):
    monkeypatch.setattr(central, "EFFECTIVE_BATCH", 8)


def cfull(d, run_dir, run_id="t", **kw):
    return central.run_central(data=d, arm="C_FULL", variant="SASREC_D64_B2", peak_lr=4e-4, dropout=(0.1,), seed=2026,
                               rows=np.arange(d.N), select_users=None, run_dir=run_dir, run_id=run_id, **kw)


def gate(d):
    final._TOKENS["g" * 32] = d.dataset
    return "g" * 32


# ------------------------------------------------------------------------------------------------ B1
def test_ft_test_context_is_the_full_history_and_last_user_scores(tmp_path, small, monkeypatch):
    rel = make_release(tmp_path / "p", n_users=40, n_items=30, drop_holdout=0)          # the LAST users keep a holdout row
    d = D.load(DS, rel["root"])
    tv = D._load_holdout_view(d, rel["root"], _gate=gate(d))
    assert int(tv.users[-1]) == d.U - 1 and tv.source == "full"
    # the worker's view equals the original one
    wv = device.make_view(d, tv.split, tv.users, tv.ends, tv.targets, tv.source)
    sel = np.arange(len(tv.users))
    assert torch.equal(wv.batch(sel)["item_tokens"], tv.batch(sel)["item_tokens"])
    assert all(np.array_equal(a, b) for a, b in zip(wv.seen_pairs(sel), tv.seen_pairs(sel)))
    # the real FT path: every context the hook scores = the full-view context, the validation item included
    from v4bench.model import build_model
    ad = build_model(common.FL_VARIANT, d.K, 2026)["adapter"]
    th = tmp_path / "theta.pt"
    central.save_weights(ad, th, run_id="x", mark=0.0)
    seen = {}
    real = device.rank_chunk

    def spy(adapter, view, sel_, P=None):
        i = int(sel_[0])
        seen[i] = (view.batch(np.asarray([i]))["item_tokens"].clone(), view.seen_pairs(np.asarray([i]))[1].copy(), view.source)
        return real(adapter, view, sel_, P)
    monkeypatch.setattr(device, "rank_chunk", spy)
    ranks, _n, _w = device.ft_ranks(dataset=DS, data_root=rel["root"], theta_path=str(th), seed=2026, view=tv, workers=1,
                                    threads=1, say=lambda *_: None)
    assert (ranks >= 0).all() and len(seen) == len(tv.users)
    for i in range(len(tv.users)):
        u = int(tv.users[i])
        full = d.full_items[d.full_offsets[u]:d.full_offsets[u + 1]]
        want = full[-d.max_len:] + D.ITEM_OFFSET
        got = seen[i][0][0][:len(want)].numpy()
        assert seen[i][2] == "full" and np.array_equal(got, want)               # not the next user's first item
        assert sorted(seen[i][1].tolist()) == sorted(full.tolist()) and d.val_item[u] in seen[i][1]


# ------------------------------------------------------------------------------------------------ M1 / M2 / Q5 / M7
def _locked(tmp_path, small, **rk):
    rel = make_release(tmp_path / "p", n_users=40, n_items=30, **rk)
    d = D.load(DS, rel["root"])
    runs = tmp_path / "runs"
    cfull(d, runs / DS / "C_FULL_s2026")
    d8 = D.load(DS, rel["root"], max_len=8)
    central.run_central(data=d8, arm="PRE", variant="SASREC_D64_B2", peak_lr=4e-4, dropout=(0.1,), seed=2026,
                        rows=d8.band_rows(), select_users=np.flatnonzero(d8.band), run_dir=runs / DS / "PRE__B6_L8_D0.1_s2026",
                        run_id="pre8", max_len=8)
    final.make_lock(runs, DS, ["C_FULL_s2026", "PRE__B6_L8_D0.1_s2026"], data_fingerprint=d.fingerprint(), strict=False)
    return rel, d, runs


def test_a_crash_never_burns_the_look(tmp_path, small, monkeypatch):
    rel, d, runs = _locked(tmp_path, small)
    real = final.score_run
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if a[6].split == "TEST":
            raise RuntimeError("boom during the TEST pass")
        return real(*a, **k)
    monkeypatch.setattr(final, "score_run", flaky)
    with pytest.raises(RuntimeError, match="boom"):
        final.final_eval(runs, DS, d, rel["root"], workers=1, threads=1)
    assert not final.look_path(runs, DS).exists() and not (runs / DS / "TEST_RESULTS.json").exists()
    ev = [json.loads(x)["event"] for x in final.attempts_path(runs, DS).read_text().splitlines()]
    assert ev == ["start", "error"]                                     # the attempt log is append-only
    monkeypatch.setattr(final, "score_run", real)                       # the look is still available
    out = final.final_eval(runs, DS, d, rel["root"], workers=1, threads=1)
    ev = [json.loads(x)["event"] for x in final.attempts_path(runs, DS).read_text().splitlines()]
    assert ev == ["start", "error", "start", "end"]
    assert final.look_path(runs, DS).exists() and set(out["runs"]) == {"C_FULL_s2026", "PRE__B6_L8_D0.1_s2026"}
    assert not list((runs / DS).glob("TEST_LOOK.json.tmp*"))
    with pytest.raises(BenchRefused, match="already been taken"):
        final.final_eval(runs, DS, d, rel["root"], workers=1, threads=1)


def test_preflight_failure_happens_before_any_test_read(tmp_path, small, monkeypatch):
    rel, d, runs = _locked(tmp_path, small)
    real = final.score_run

    def bad(*a, **k):
        raise RuntimeError("cannot load")
    monkeypatch.setattr(final, "score_run", bad)
    opened = []
    orig = D._load_holdout_view
    monkeypatch.setattr(D, "_load_holdout_view", lambda *a, **k: opened.append(1) or orig(*a, **k))
    with pytest.raises(RuntimeError):
        final.final_eval(runs, DS, d, rel["root"], workers=1, threads=1)
    assert not opened and not final.look_path(runs, DS).exists()
    # the holdout is opened only AFTER every preflight score succeeded
    monkeypatch.setattr(final, "score_run", real)
    order = []
    monkeypatch.setattr(final, "score_run", lambda *a, **k: order.append(("score", a[6].split)) or real(*a, **k))
    monkeypatch.setattr(D, "_load_holdout_view", lambda *a, **k: order.append(("holdout", None)) or orig(*a, **k))
    final.final_eval(runs, DS, d, rel["root"], workers=1, threads=1)
    first_hold = [i for i, o in enumerate(order) if o[0] == "holdout"][0]
    assert all(o == ("score", "VALIDATION") for o in order[:first_hold]) and first_hold > 0


def test_each_run_is_scored_with_its_own_max_len(tmp_path, small, monkeypatch):
    rel, d, runs = _locked(tmp_path, small)
    seen = []
    real = final.score_run

    def spy(rec, res, rule, c, run_dir, data, view, **k):
        seen.append((rec["run"], view.split, data.max_len, int(view.batch(np.arange(len(view.users)))["item_tokens"].shape[1])))
        return real(rec, res, rule, c, run_dir, data, view, **k)
    monkeypatch.setattr(final, "score_run", spy)
    final.final_eval(runs, DS, d, rel["root"], workers=1, threads=1)
    by = {(r, s): (ml, w) for r, s, ml, w in seen}
    assert by[("C_FULL_s2026", "TEST")][0] == 50 and by[("PRE__B6_L8_D0.1_s2026", "TEST")][0] == 8
    assert by[("PRE__B6_L8_D0.1_s2026", "TEST")][1] <= 8 and by[("PRE__B6_L8_D0.1_s2026", "VALIDATION")][0] == 8


def test_wave1_final_refuses_cold_holdout_items(tmp_path, small):
    rel, d, runs = _locked(tmp_path, small, cold_holdout=2)
    with pytest.raises(BenchRefused, match="cold holdout items"):
        final.final_eval(runs, DS, d, rel["root"], workers=1, threads=1)
    assert not final.look_path(runs, DS).exists()


def test_finished_runs_are_never_overwritten(tmp_path, small):
    rel, d, runs = _locked(tmp_path, small)
    with pytest.raises(BenchRefused, match="never overwritten"):
        cfull(d, runs / DS / "C_FULL_s2026")
    with pytest.raises(BenchRefused, match="never overwritten"):
        fl_engine.run_fl(data=d, arm="FA", seed=2026, run_dir=runs / DS / "C_FULL_s2026", run_id="x")
    with pytest.raises(BenchRefused, match="never overwritten"):
        device.run_ft(data=d, data_root=rel["root"], arm="FT_FA", seed=2026, base_run_dir=runs / DS / "C_FULL_s2026",
                      run_dir=runs / DS / "C_FULL_s2026", run_id="x", workers=1, threads=1)


# ------------------------------------------------------------------------------------------------ M3
def fake(runs, name, arm, kind="fl", **extra):
    d = runs / DS / name
    d.mkdir(parents=True)
    _, sd = final._label_seed(name)
    res = {"status": "DONE_S_CAL" if kind == "scal" else "DONE_MAIN", "arm": arm, "kind": kind,
           "spec": {"seed": sd, "cfg": {"seed": sd}}, "checkpoints": {}, **extra}
    (d / "RESULT.json").write_text(json.dumps(res))


def test_lock_rules_for_grid_cells_s_cal_and_pins(tmp_path, monkeypatch):
    monkeypatch.setattr(recipe, "check_pins", lambda *a, **k: {})
    runs = tmp_path / "runs"
    base = runs / DS
    (base / "dp").mkdir(parents=True)
    (base / "dp" / "S_RECORD_FA_WARM_s2026.json").write_text('{"S": 1.0}')
    (base / "dp" / "S_RECORD_FA_WARM_s2027.json").write_text('{"S": 1.5}')
    for n, a in (("FA_s2026", "FA"), ("FA_s2027", "FA"), ("FA__B6_L50_D0.5_s2026", "FA"), ("FA__B6_L200_D0.1_s2026", "FA"),
                 ("FA__B6_L200_D0.1_s2027", "FA")):
        fake(runs, n, a)
    fake(runs, "S_CAL_s2026", "S_CAL", kind="scal", s_record=str(base / "dp" / "S_RECORD_FA_WARM_s2026.json"), S=1.0)
    fake(runs, "S_CAL_s2027", "S_CAL", kind="scal", s_record=str(base / "dp" / "S_RECORD_FA_WARM_s2027.json"), S=1.5)
    (base / "trackb").mkdir()
    (base / "trackb" / "WINNER_FA.json").write_text(json.dumps({"winner_run": "FA__B6_L200_D0.1_s2026"}))
    (base / "FA_WARM_DECISION.json").write_text(json.dumps({"FA_WARM_STAR": "FA_WARM"}))
    (base / "SEED3_DECISION.json").write_text(json.dumps({"third_seed_2028_needed_for": []}))
    names = ["FA_s2026", "FA_s2027", "FA__B6_L50_D0.5_s2026", "FA__B6_L200_D0.1_s2026", "FA__B6_L200_D0.1_s2027",
             "S_CAL_s2026", "S_CAL_s2027"]
    lock = final.make_lock(runs, DS, names, data_fingerprint="x")          # grid cell seed-2026-only + S_CAL + winner 2027: ok
    assert {"dp/S_RECORD_FA_WARM_s2026.json", "dp/S_RECORD_FA_WARM_s2027.json"} <= set(lock["pinned_files"]) and "trackb/WINNER_FA.json" in lock["pinned_files"]
    assert any(r["kind"] == "scal" and r["s_record_sha256"] for r in lock["runs"])
    final.verify_lock(runs, DS)
    (base / "dp" / "S_RECORD_FA_WARM_s2027.json").write_text('{"S": 2.0}')               # a pinned S record changed after the LOCK
    with pytest.raises(BenchRefused, match="pinned file"):
        final.verify_lock(runs, DS)
    (base / "dp" / "S_RECORD_FA_WARM_s2027.json").write_text('{"S": 1.5}')
    final.verify_lock(runs, DS)
    # NCs on a fresh lock path
    runs2 = tmp_path / "runs2"
    import shutil
    shutil.copytree(runs, runs2)
    import os
    os.chmod(runs2 / DS / "LOCK.json", 0o666)
    (runs2 / DS / "LOCK.json").unlink()
    shutil.rmtree(runs2 / DS / "FA__B6_L200_D0.1_s2027")
    with pytest.raises(BenchRefused, match="lacks seeds"):                 # the winner needs seed 2027
        final.make_lock(runs2, DS, [n for n in names if n != "FA__B6_L200_D0.1_s2027"], data_fingerprint="x")
    shutil.rmtree(runs2 / DS / "FA_s2027")
    with pytest.raises(BenchRefused, match="lacks seeds"):                 # a Track-A label needs both seeds
        final.make_lock(runs2, DS, [n for n in names if n not in ("FA_s2027", "FA__B6_L200_D0.1_s2027")],
                        data_fingerprint="x")


def test_incomplete_runs_cannot_be_locked(tmp_path):
    d = tmp_path / DS / "X_s2026"
    d.mkdir(parents=True)
    (d / "RESULT.json").write_text(json.dumps({"status": "INCOMPLETE_WALL_CAP"}))
    with pytest.raises(BenchRefused, match="INCOMPLETE"):
        final._run_record(d)


# ------------------------------------------------------------------------------------------------ M4
def test_validation_population_follows_the_rectools_filter(release):
    d = D.load(release["name"], release["root"])
    users = d.validation_users()
    inner_ok = [u for u in range(d.U) if d.inner_lengths[u] >= 2]
    present = set()
    for u in inner_ok:
        present.update(d.seq_items[d.seq_offsets[u]:d.seq_offsets[u + 1]].tolist())
    want = [u for u in inner_ok if int(d.val_item[u]) in present]
    assert users.tolist() == want and len(users) <= d.U
    assert len(d.validation_view().users) == len(users)
    # the trainer keeps the last session_max_len + 1 inner events: a short tail removes validation items that only occur early
    for tail in (2, 3, 5):
        pres = set()
        for u in inner_ok:
            pres.update(d.seq_items[d.seq_offsets[u]:d.seq_offsets[u + 1]][-tail:].tolist())
        assert d.validation_users(tail=tail).tolist() == [u for u in inner_ok if int(d.val_item[u]) in pres]
    assert len(d.validation_users(tail=2)) < len(users)


# ------------------------------------------------------------------------------------------------ M6
def test_mirrors_are_proven_against_the_pinned_files(tmp_path, monkeypatch):
    monkeypatch.setattr(recipe, "check_pins", lambda *a, **k: {"x": "y"})
    c = tmp_path / "contracts"
    (c / "fl").mkdir(parents=True)
    (c / "device").mkdir()
    for st, w in (("S7", recipe.WARM_WINNER["FA_WARM"]), ("S8", recipe.WARM_WINNER["FA_WARM_FROZEN"])):
        (c / "fl" / f"FL_WARM_SELECTION_{st}.json").write_text(json.dumps({"winner": dict(w)}))
    tun = {k: v for k, v in recipe.TUNING_WINNER.items()}
    (c / "fl" / "FL_TUNING_SELECTION.json").write_text(json.dumps({"winner": tun}))
    (c / "device" / "SELECTION_FT_C.json").write_text(json.dumps({"cell": "FT_C_f0.3_P4", "cells": {"lr_frac": 0.3, "passes": 4}}))
    rec = recipe.check_mirrors(c)
    assert rec["ft_cell"] == "FT_C_f0.3_P4" and set(rec["warm_winners"]) == {"FA_WARM", "FA_WARM_FROZEN"}
    (c / "fl" / "FL_TUNING_SELECTION.json").write_text(json.dumps({"winner": dict(tun, client_lr=1.0)}))
    with pytest.raises(BenchRefused, match="client_lr"):
        recipe.check_mirrors(c)
    (c / "fl" / "FL_TUNING_SELECTION.json").write_text(json.dumps({"winner": tun}))
    (c / "device" / "SELECTION_FT_C.json").write_text(json.dumps({"cell": "FT_C_f0.1_P1", "cells": {"lr_frac": 0.1, "passes": 1}}))
    with pytest.raises(BenchRefused, match="FT_C"):
        recipe.check_mirrors(c)
    (c / "device" / "SELECTION_FT_C.json").write_text(json.dumps({"cell": "FT_C_f0.3_P4", "cells": {"lr_frac": 0.3, "passes": 4}}))
    (c / "fl" / "FL_WARM_SELECTION_S7.json").write_text(json.dumps({"winner": {"warm_lr_factor": 1.0, "warm_server_lr_factor": 1.0,
                                                                              "label": "x"}}))
    with pytest.raises(BenchRefused):
        recipe.check_mirrors(c)


def test_real_contract_files_equal_the_mirrors_when_present():
    c = common.CONTRACTS_DEFAULT
    if not (c / "fl" / "FL_WARM_SELECTION_S7.json").exists():
        pytest.skip("S7 / S8 selection files are on the server only")
    assert recipe.check_mirrors(c)["tuning_winner_label"] == recipe.TUNING_WINNER["label"]


# ------------------------------------------------------------------------------------------------ PF, warm PRE seed
def test_pf_arm_end_to_end_and_pre_seed(tmp_path, small):
    rel = make_release(tmp_path / "p", n_users=60, n_items=30, drop_holdout=0)
    d = D.load(DS, rel["root"])
    runs = tmp_path / "runs" / DS
    out = fl_engine.run_fl(data=d, arm="PF", seed=2026, run_dir=runs / "PF_s2026", run_id="pf")
    assert out["status"] == "DONE_MAIN"
    res = read_json(runs / "PF_s2026" / "RESULT.json")
    assert res["spec"]["cfg"]["method"] == "PF" and res["spec"]["cfg"]["pf"]["lam"] == 1e-4
    for rule, c in res["checkpoints"].items():
        assert c["pf_store"]["sha256"]
    rec = final._run_record(runs / "PF_s2026")
    assert rec["checkpoints"]["PRACTICAL_BEST"]["pf_store"]["file"].endswith("pf_store_best.pt")
    # scoring uses shared + p_u: zeroing the store changes the ranks (the personal component is really evaluated)
    c = rec["checkpoints"]["PRACTICAL_BEST"]
    vv = d.validation_view()
    r_pf = final.score_run(rec, res, "PRACTICAL_BEST", c, runs / "PF_s2026", d, vv, dataset=DS, data_root=rel["root"],
                           workers=1, threads=1)
    assert (r_pf >= 0).all()
    P = fl_engine.load_pf_matrix(runs / "PF_s2026" / c["pf_store"]["file"], d)
    assert P.shape[0] == d.U and float(P.abs().sum()) > 0                    # a real personal store was trained and saved
    from v4bench.central import load_weights
    from v4bench.evaluate import evaluate_view
    from v4bench.model import build_model
    ad = build_model(common.FL_VARIANT, d.K, 2026)["adapter"]
    ad.load_state_(load_weights(runs / "PF_s2026" / c["file"])["theta"])
    assert not np.array_equal(evaluate_view(ad, vv), evaluate_view(ad, vv, P=P * 1e4))   # P enters the score path
    pw = recipe.build_fl_spec("PF_WARM_FROZEN", seed=2026, n_clients=100, init_from={"kind": "k", "run_id": "r", "run_label": "l",
                                                                                      "fl_init_kind": "e", "train_manifest": "t",
                                                                                      "family": "SASREC", "checkpoint_sha256": "a" * 64,
                                                                                      "state_digest": "b" * 64})
    assert pw["cfg"]["method"] == "PF" and pw["cfg"]["pf"]["lr"] == recipe.SMALL_PEAK_LR and pw["frozen_item_tables"]
    # the warm start needs the PRE of the SAME run seed
    central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7e-4, dropout=(0.1,), seed=2026,
                        rows=d.band_rows(), select_users=np.flatnonzero(d.band), run_dir=runs / "PRE_s2026", run_id="p")
    with pytest.raises(BenchRefused, match="seed-2027 PRE"):
        fl_engine.run_fl(data=d, arm="FA_WARM", seed=2027, run_dir=runs / "x", run_id="x", pre_run_dir=runs / "PRE_s2026")


def test_decisions_assert_track_a(tmp_path):
    from test_selection import fake_run
    fake_run(tmp_path, "FA_WARM_s2026", "FA_WARM", 2026, 0.1, dropout=0.5)
    fake_run(tmp_path, "FA_WARM_s2027", "FA_WARM", 2027, 0.1)
    fake_run(tmp_path, "FA_WARM_FROZEN_s2026", "FA_WARM_FROZEN", 2026, 0.1)
    fake_run(tmp_path, "FA_WARM_FROZEN_s2027", "FA_WARM_FROZEN", 2027, 0.1)
    with pytest.raises(BenchRefused, match="Track-A cell"):
        selection.decide_warm(tmp_path, DS, ["FA_WARM_s2026", "FA_WARM_s2027"],
                              ["FA_WARM_FROZEN_s2026", "FA_WARM_FROZEN_s2027"])


def test_b10_q4_ml1m_budget_96_refused(tmp_path):
    from v4bench import run as R
    rc = R.main(["run", "--dataset", "ml_1m", "--arm", "C_FULL", "--seed", "2026", "--budget-efe", "96",
                 "--data-root", str(tmp_path), "--runs-root", str(tmp_path), "--threads", "16"])
    assert rc == 2
