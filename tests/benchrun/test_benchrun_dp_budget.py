"""The DP chain (S_CAL -> S record -> FA_1024 -> DP8), the exposure-budget option, the early stop and the wall cap,
on a tiny synthetic release."""
from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import make_release

from ppsi.benchrun import central, fl_engine, recipe
from ppsi.benchrun import data as D
from ppsi.benchrun.common import BenchRefused, read_json

DS = "s3_beauty"


def _recipe(tmp_path, **dp):
    p = tmp_path / "recipe.json"
    p.write_text(json.dumps({"dp": dp}), encoding="utf-8")
    return recipe.load_recipe(p)


def test_dp_chain_and_budget_option(tmp_path, monkeypatch):
    monkeypatch.setattr(central, "EFFECTIVE_BATCH", 8)
    rcp = _recipe(tmp_path, T=24)                                # a shorter chain for the test (S_CAL: 20 rounds)
    rel = make_release(tmp_path / "p", n_users=300, n_items=40)
    d = D.load(DS, rel["root"])
    runs = tmp_path / "runs" / DS
    assert recipe.dp_cohort(300, rcp) == 2
    pre = central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=1e-3, dropout=(0.1,), seed=2026,
                              rows=d.band_rows(), select_users=np.flatnonzero(d.band), run_dir=runs / "PRE_s2026",
                              run_id="p")
    assert pre["status"] == "DONE_MAIN"
    out = fl_engine.run_fl(data=d, arm="S_CAL", seed=2026, run_dir=runs / "S_CAL_s2026", run_id="sc", rcp=rcp,
                           dp_dir=runs / "dp")
    assert out["status"] == "DONE_S_CAL" and out["S"] > 0
    rec = runs / "dp" / "S_RECORD_COLD_s2026.json"
    assert rec.exists()
    with pytest.raises(BenchRefused):                              # DP8 without an S record is refused
        fl_engine.run_fl(data=d, arm="DP8", seed=2026, run_dir=runs / "x", run_id="x", rcp=rcp)
    for arm in ("FA_1024", "DP8"):
        o = fl_engine.run_fl(data=d, arm=arm, seed=2026, run_dir=runs / f"{arm}_s2026", run_id=arm, rcp=rcp,
                             s_record=rec)
        assert o["status"] == "DONE_MAIN"
        sp = read_json(runs / f"{arm}_s2026" / "ARM_SPEC.json")["arm_spec"]
        assert sp["s_record"]["S"] == out["S"] and sp["cfg"]["dp"]["clip_norm"] == out["S"]
        assert read_json(runs / f"{arm}_s2026" / "RESULT.json")["spec"]["plan"]["sampling"] == "poisson"
    o = fl_engine.run_fl(data=d, arm="S_CAL_WARM_FROZEN", seed=2026, run_dir=runs / "S_CAL_WARM_FROZEN_s2026",
                         run_id="scw", rcp=rcp, pre_run_dir=runs / "PRE_s2026", dp_dir=runs / "dp")
    assert o["status"] == "DONE_S_CAL" and (runs / "dp" / "S_RECORD_FA_WARM_FROZEN_s2026.json").exists()
    with pytest.raises(BenchRefused):                              # the cold S record cannot feed a warm DP run
        fl_engine.run_fl(data=d, arm="DP8_WARM_FROZEN", seed=2026, run_dir=runs / "y", run_id="y", rcp=rcp,
                         pre_run_dir=runs / "PRE_s2026", s_record=rec)
    with pytest.raises(BenchRefused, match="seed"):                # a run binds its own seed's record
        fl_engine.run_fl(data=d, arm="DP8", seed=2027, run_dir=runs / "z", run_id="z", rcp=rcp, s_record=rec)

    # budget 24 with the stretched schedule (no early stop): CONTROLLED_BUDGET = the 24-EFE endpoint
    full = central.run_central(data=d, arm="C_FULL", variant="SASREC_D64_B2", peak_lr=5e-4, dropout=(0.1,), seed=2026,
                               rows=np.arange(d.N), select_users=None, run_dir=tmp_path / "cf24", run_id="c24",
                               budget_efe=24)
    res = full["result"]
    assert res["spec"]["budget_efe"] == 24
    assert res["checkpoints"]["CONTROLLED_BUDGET"]["file"].endswith("endpoint_24.0.pt")
    assert res["tracker"]["history"][0]["mark"] == 2.0 and len(res["tracker"]["history"]) == 12   # k * 24 / 12 EFE
    with pytest.raises(BenchRefused):
        central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=1e-3, dropout=(0.1,), seed=2026,
                            rows=d.band_rows(), select_users=None, run_dir=tmp_path / "bad", run_id="b", budget_efe=12)
    with pytest.raises(BenchRefused, match="never overwritten"):
        central.run_central(data=d, arm="C_FULL", variant="SASREC_D64_B2", peak_lr=5e-4, dropout=(0.1,), seed=2026,
                            rows=np.arange(d.N), select_users=None, run_dir=tmp_path / "cf24", run_id="c24",
                            budget_efe=24)


def _flat_summary(value):
    m = {"ndcg@10": value, "hr@10": value, "ndcg@20": value, "hr@20": value, "mrr@20": value, "n_users": 5,
         "n_unreachable": 0}
    return {g: dict(m) for g in ("ALL", "BAND", "OTHERS", "BAND_COVERED_TARGET", "BAND_UNCOVERED_TARGET")}


@pytest.mark.parametrize("k", [3, 6])
def test_early_stop_after_exactly_k_flat_marks(tmp_path, monkeypatch, k):
    monkeypatch.setattr(central, "EFFECTIVE_BATCH", 8)
    monkeypatch.setattr(central, "summarize", lambda ranks, view, band: _flat_summary(0.1))   # mark 1 improves only
    rel = make_release(tmp_path / "p", n_users=60, n_items=30)
    d = D.load(DS, rel["root"])
    out = central.run_central(data=d, arm="C_FULL", variant="SASREC_D64_B2", peak_lr=5e-4, dropout=(0.1,), seed=2026,
                              rows=np.arange(d.N), select_users=None, run_dir=tmp_path / "es", run_id="es",
                              budget_efe=24, early_stop_marks=k)
    res = out["result"]
    hist = res["tracker"]["history"]
    assert len(hist) == 1 + k and res["early_stopped_at"] == hist[-1]["mark"]
    assert [h["improved"] for h in hist] == [True] + [False] * k
    assert set(res["checkpoints"]) == {"PRACTICAL_BEST"} and res["controlled_budget"] == "N/A (early stopped)"
    assert not (tmp_path / "es" / "ckpt" / "endpoint_24.0.pt").exists()


def test_wall_cap_makes_the_run_incomplete_and_ineligible(tmp_path, monkeypatch):
    monkeypatch.setattr(central, "EFFECTIVE_BATCH", 8)
    rel = make_release(tmp_path / "p", n_users=60, n_items=30)
    d = D.load(DS, rel["root"])
    out = central.run_central(data=d, arm="C_FULL", variant="SASREC_D64_B2", peak_lr=5e-4, dropout=(0.1,), seed=2026,
                              rows=np.arange(d.N), select_users=None, run_dir=tmp_path / "cap", run_id="cap",
                              wall_cap_hours=1e-12)
    assert out["status"] == "INCOMPLETE_WALL_CAP"
    assert read_json(tmp_path / "cap" / "RESULT.json")["status"] == "INCOMPLETE_WALL_CAP"
    with pytest.raises(BenchRefused):                                  # never rerun
        central.run_central(data=d, arm="C_FULL", variant="SASREC_D64_B2", peak_lr=5e-4, dropout=(0.1,), seed=2026,
                            rows=np.arange(d.N), select_users=None, run_dir=tmp_path / "cap", run_id="cap")


def test_poisson_inclusion_is_exactly_q():
    rcp = recipe.load_recipe()
    cls = fl_engine._exact_q_plan_cls(rcp["dp"]["q_nominal"])
    keys = [f"u{i:07d}" for i in range(6040)]
    m = recipe.dp_cohort(6040, rcp)
    plan = cls.build(keys, seed=2026, manifest_hash="x", group_size=m, sampling="poisson")
    assert plan.inclusion_probability == 2.0 ** -7 and plan.group_size == m == 47
    sizes = np.asarray([len(plan.round_positions(r)) for r in range(4000)])
    exp = 6040 / 128
    assert abs(sizes.mean() - exp) < 4 * np.sqrt(6040 * (1 / 128) * (127 / 128) / 4000)     # 4 sigma around N q
    assert np.array_equal(plan.round_positions(7), plan.round_positions(7))                  # a function of (seed, r)


def test_stretched_schedule_ends_at_the_floor():
    from ppsi.central import schedule
    t = central.stretched_lr_table(1.0, 70, 8, 24)
    assert t["final_exposures"] == 24 * 70 and t["min_lr"] > 0 and abs(t["last_lr"] - 0.01) < 1e-9
    t6 = central.stretched_lr_table(1.0, 70, 8, 6)
    ref = schedule.planned_lr_table(1.0, 70, 8, "MAIN", "end", schedule="S0")
    assert np.allclose(t6["lr"], ref["lr"], rtol=0, atol=1e-12)
