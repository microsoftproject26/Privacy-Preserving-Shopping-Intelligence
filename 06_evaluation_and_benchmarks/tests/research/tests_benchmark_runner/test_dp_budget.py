"""R4 / O2 (S_CAL -> S_RECORD -> FA_1024 -> DP8) and O1 (exposure budget axis, early stop) on a tiny synthetic release."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import make_release
from v4bench import central, data as D, fl_engine, recipe
from v4bench.common import BenchRefused, read_json

DS = "s3_beauty"


def test_dp_cohort_matches_the_registered_m():
    assert [recipe.dp_cohort(n) for n in (6040, 22363, 19412, 35598)] == [47, 175, 152, 278]   # ML-1M Beauty Toys Sports


def test_dp_specs_follow_r4():
    s = recipe.build_fl_spec("S_CAL", seed=2026, n_clients=22363)
    assert s["cfg"]["dp"] == {"clip_norm": None, "noise_multiplier": 0.0, "denominator": 175}
    assert s["plan"] == {"group_size": 175, "sampling": "poisson"} and s["max_rounds"] == 20
    d8 = recipe.build_fl_spec("DP8", seed=2027, n_clients=22363, clip_norm=1.5)
    assert d8["cfg"]["dp"]["noise_multiplier"] == 0.5642 and d8["cfg"]["endpoint_rounds"] == 384
    assert d8["cfg"]["dropout_p"] == 0.0 and d8["cfg"]["solver"]["passes"] == 2          # DL 219: P = 2, B = 16
    assert d8["dp_accounting"]["T"] == 384 and d8["dp_accounting"]["delta"] == 5e-06
    fa = recipe.build_fl_spec("FA_1024", seed=2026, n_clients=22363, clip_norm=1.5)
    assert fa["cfg"]["dp"]["noise_multiplier"] == 0.0
    with pytest.raises(BenchRefused):
        recipe.build_fl_spec("DP8", seed=2026, n_clients=22363)                            # needs S
    with pytest.raises(BenchRefused):
        recipe.build_fl_spec("S_CAL", seed=2028, n_clients=22363)                          # S_CAL: one per PRE seed (2026, 2027)


def test_dp_chain_and_budget_axis(tmp_path, monkeypatch):
    monkeypatch.setattr(central, "EFFECTIVE_BATCH", 8)
    monkeypatch.setitem(recipe.DP, "T", 24)                      # shorter chain for the test only (S_CAL still 20 rounds)
    rel = make_release(tmp_path / "p", n_users=300, n_items=40)
    d = D.load(DS, rel["root"])
    runs = tmp_path / "runs" / DS
    assert recipe.dp_cohort(300) == 2
    pre = central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7e-4, dropout=(0.1,), seed=2026,
                              rows=d.band_rows(), select_users=np.flatnonzero(d.band), run_dir=runs / "PRE_s2026",
                              run_id="p")
    assert pre["status"] == "DONE_MAIN"
    out = fl_engine.run_fl(data=d, arm="S_CAL", seed=2026, run_dir=runs / "S_CAL_s2026", run_id="sc", dp_dir=runs / "dp")
    assert out["status"] == "DONE_S_CAL" and out["S"] > 0
    rec = runs / "dp" / "S_RECORD_COLD_s2026.json"
    assert rec.exists()
    with pytest.raises(BenchRefused):                              # DP8 without an S record is refused
        fl_engine.run_fl(data=d, arm="DP8", seed=2026, run_dir=runs / "x", run_id="x")
    for arm in ("FA_1024", "DP8"):
        o = fl_engine.run_fl(data=d, arm=arm, seed=2026, run_dir=runs / f"{arm}_s2026", run_id=arm, s_record=rec)
        assert o["status"] == "DONE_MAIN"
        sp = read_json(runs / f"{arm}_s2026" / "ARM_SPEC.json")["arm_spec"]
        assert sp["s_record"]["S"] == out["S"] and sp["cfg"]["dp"]["clip_norm"] == out["S"]
        assert read_json(runs / f"{arm}_s2026" / "RESULT.json")["spec"]["plan"]["sampling"] == "poisson"
    o = fl_engine.run_fl(data=d, arm="S_CAL_WARM_FROZEN", seed=2026, run_dir=runs / "S_CAL_WARM_FROZEN_s2026",
                         run_id="scw", pre_run_dir=runs / "PRE_s2026", dp_dir=runs / "dp")
    assert o["status"] == "DONE_S_CAL" and (runs / "dp" / "S_RECORD_FA_WARM_FROZEN_s2026.json").exists()
    with pytest.raises(BenchRefused):                              # the cold S record cannot feed a warm DP run
        fl_engine.run_fl(data=d, arm="DP8_WARM_FROZEN", seed=2026, run_dir=runs / "y", run_id="y",
                         pre_run_dir=runs / "PRE_s2026", s_record=rec)
    o = fl_engine.run_fl(data=d, arm="DP8_WARM_FROZEN", seed=2026, run_dir=runs / "DP8_WARM_FROZEN_s2026", run_id="dw",
                         pre_run_dir=runs / "PRE_s2026", s_record=runs / "dp" / "S_RECORD_FA_WARM_FROZEN_s2026.json")
    assert o["status"] == "DONE_MAIN"

    # B12: the s2027 chain: S_CAL per PRE seed, each run binds its own seed's record, the old s2026 file name is still accepted
    central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7e-4, dropout=(0.1,), seed=2027,
                        rows=d.band_rows(), select_users=np.flatnonzero(d.band), run_dir=runs / "PRE_s2027", run_id="p27")
    s26 = runs / "dp" / "S_RECORD_FA_WARM_FROZEN_s2026.json"
    s27 = runs / "dp" / "S_RECORD_FA_WARM_FROZEN_s2027.json"
    with pytest.raises(BenchRefused, match="seed"):                # s2027 cannot use the s2026 record (and vice versa)
        fl_engine.run_fl(data=d, arm="DP8_WARM_FROZEN", seed=2027, run_dir=runs / "bad27", run_id="b27",
                         pre_run_dir=runs / "PRE_s2027", s_record=s26)
    o = fl_engine.run_fl(data=d, arm="S_CAL_WARM_FROZEN", seed=2027, run_dir=runs / "S_CAL_WARM_FROZEN_s2027", run_id="sc27",
                         pre_run_dir=runs / "PRE_s2027", dp_dir=runs / "dp")
    assert o["status"] == "DONE_S_CAL" and s27.exists()
    assert read_json(s27)["seed"] == 2027 and read_json(s26)["seed"] == 2026
    assert read_json(s27)["init_digest"] != read_json(s26)["init_digest"]          # theta_pre differs per PRE seed
    for arm in ("FA_1024_WARM_FROZEN", "DP8_WARM_FROZEN"):
        o = fl_engine.run_fl(data=d, arm=arm, seed=2027, run_dir=runs / f"{arm}_s2027", run_id=f"{arm}27",
                             pre_run_dir=runs / "PRE_s2027", s_record=s27)
        assert o["status"] == "DONE_MAIN"
        assert read_json(runs / f"{arm}_s2027" / "ARM_SPEC.json")["arm_spec"]["s_record"]["S"] == read_json(s27)["S"]
    with pytest.raises(BenchRefused, match="seed"):
        fl_engine.run_fl(data=d, arm="DP8_WARM_FROZEN", seed=2026, run_dir=runs / "bad26", run_id="b26",
                         pre_run_dir=runs / "PRE_s2026", s_record=s27)
    old = runs / "dp" / "S_RECORD_FA_WARM_FROZEN.json"                               # the pre-B12 name of the s2026 record
    old.write_bytes(s26.read_bytes())
    o = fl_engine.run_fl(data=d, arm="FA_1024_WARM_FROZEN", seed=2026, run_dir=runs / "FA_1024_WARM_FROZEN_s2026", run_id="old",
                         pre_run_dir=runs / "PRE_s2026", s_record=old)
    assert o["status"] == "DONE_MAIN"

    # O1: budget 24 with the stretched schedule (no early stop): CONTROLLED_BUDGET = the 24-EFE endpoint
    full = central.run_central(data=d, arm="C_FULL", variant="SASREC_D64_B2", peak_lr=4e-4, dropout=(0.1,), seed=2026,
                               rows=np.arange(d.N), select_users=None, run_dir=tmp_path / "cf24", run_id="c24",
                               budget_efe=24)
    res = full["result"]
    assert res["spec"]["budget_efe"] == 24 and res["checkpoints"]["CONTROLLED_BUDGET"]["file"].endswith("endpoint_24.0.pt")
    assert res["tracker"]["history"][0]["mark"] == 2.0 and len(res["tracker"]["history"]) == 12   # k * 24 / 12 EFE
    pre24 = central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7e-4, dropout=(0.1,), seed=2026,
                                rows=d.band_rows(), select_users=np.flatnonzero(d.band), run_dir=tmp_path / "pre24",
                                run_id="p24", budget_efe=24)
    assert len(pre24["result"]["tracker"]["history"]) == 12 and pre24["result"]["early_stopped_at"] is None
    with pytest.raises(BenchRefused):
        central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7e-4, dropout=(0.1,), seed=2026,
                            rows=d.band_rows(), select_users=None, run_dir=tmp_path / "bad", run_id="b", budget_efe=12)
    with pytest.raises(BenchRefused, match="never overwritten"):      # M7
        central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7e-4, dropout=(0.1,), seed=2026,
                            rows=d.band_rows(), select_users=None, run_dir=tmp_path / "pre24", run_id="b", budget_efe=24)


def _flat_summary(value):
    m = {"ndcg@10": value, "hr@10": value, "ndcg@20": value, "hr@20": value, "mrr@20": value, "n_users": 5,
         "n_unreachable": 0}
    return {g: dict(m) for g in ("ALL", "BAND", "OTHERS", "BAND_COVERED_TARGET", "BAND_UNCOVERED_TARGET")}


@pytest.mark.parametrize("k", [3, 6])
def test_early_stop_after_exactly_k_flat_marks(tmp_path, monkeypatch, k):
    monkeypatch.setattr(central, "EFFECTIVE_BATCH", 8)
    monkeypatch.setattr(central, "summarize", lambda ranks, view, band: _flat_summary(0.1))   # mark 1 improves, rest flat
    rel = make_release(tmp_path / "p", n_users=60, n_items=30)
    d = D.load(DS, rel["root"])
    out = central.run_central(data=d, arm="C_FULL", variant="SASREC_D64_B2", peak_lr=4e-4, dropout=(0.1,), seed=2026,
                              rows=np.arange(d.N), select_users=None, run_dir=tmp_path / "es", run_id="es",
                              budget_efe=24, early_stop_marks=k)
    res = out["result"]
    hist = res["tracker"]["history"]
    assert len(hist) == 1 + k and res["early_stopped_at"] == hist[-1]["mark"]       # mark 1 + exactly k flat marks
    assert [h["improved"] for h in hist] == [True] + [False] * k
    assert set(res["checkpoints"]) == {"PRACTICAL_BEST"} and res["controlled_budget"] == "N/A (early stopped)"
    assert set(res["validation"]) == {"PRACTICAL_BEST"} and res["checkpoints"]["PRACTICAL_BEST"]["mark"] == 2.0
    assert not (tmp_path / "es" / "ckpt" / "endpoint_24.0.pt").exists()


def test_wall_cap_makes_the_run_incomplete_and_ineligible(tmp_path, monkeypatch):
    monkeypatch.setattr(central, "EFFECTIVE_BATCH", 8)
    rel = make_release(tmp_path / "p", n_users=60, n_items=30)
    d = D.load(DS, rel["root"])
    out = central.run_central(data=d, arm="C_FULL", variant="SASREC_D64_B2", peak_lr=4e-4, dropout=(0.1,), seed=2026,
                              rows=np.arange(d.N), select_users=None, run_dir=tmp_path / "cap", run_id="cap",
                              wall_cap_hours=1e-12)
    assert out["status"] == "INCOMPLETE_WALL_CAP"
    from v4bench import final, selection
    assert read_json(tmp_path / "cap" / "RESULT.json")["status"] == "INCOMPLETE_WALL_CAP"
    with pytest.raises(BenchRefused, match="INCOMPLETE|DONE_MAIN"):
        final._run_record(tmp_path / "cap")
    with pytest.raises(BenchRefused):
        central.run_central(data=d, arm="C_FULL", variant="SASREC_D64_B2", peak_lr=4e-4, dropout=(0.1,), seed=2026,
                            rows=np.arange(d.N), select_users=None, run_dir=tmp_path / "cap", run_id="cap")   # not rerun


def test_poisson_inclusion_is_exactly_q(tmp_path):
    from v4bench import fl_engine
    from fl.participation import ParticipationPlan  # noqa: F401
    cls = fl_engine._exact_q_plan_cls()
    keys = [f"u{i:07d}" for i in range(6040)]
    m = recipe.dp_cohort(6040)
    plan = cls.build(keys, seed=2026, manifest_hash="x", group_size=m, sampling="poisson")
    assert plan.inclusion_probability == 2.0 ** -7 and plan.group_size == m == 47
    sizes = np.asarray([len(plan.round_positions(r)) for r in range(4000)])
    exp = 6040 / 128
    assert abs(sizes.mean() - exp) < 4 * np.sqrt(6040 * (1 / 128) * (127 / 128) / 4000)     # 4 sigma around N q (not m)
    assert abs(sizes.mean() - m) > 0.0 and abs(sizes.mean() - exp) < abs(sizes.mean() - m) + 0.2
    assert np.array_equal(plan.round_positions(7), plan.round_positions(7))                  # a function of (seed, r) only
    spec = recipe.build_fl_spec("DP8", seed=2026, n_clients=6040, clip_norm=1.0)
    assert spec["dp_accounting"]["q_actual"] == 2.0 ** -7 and spec["plan"]["group_size"] == 47


def test_stretched_schedule_ends_at_the_registered_floor():
    t = central.stretched_lr_table(1.0, 70, 8, 24)
    assert t["final_exposures"] == 24 * 70 and t["min_lr"] > 0 and abs(t["last_lr"] - 0.01) < 1e-9
    t6 = central.stretched_lr_table(1.0, 70, 8, 6)
    from v4train import schedule
    ref = schedule.planned_lr_table(1.0, 70, 8, "MAIN", "end", schedule="S0")
    assert np.allclose(t6["lr"], ref["lr"], rtol=0, atol=1e-12)         # same shape as the registered 6-EFE schedule
