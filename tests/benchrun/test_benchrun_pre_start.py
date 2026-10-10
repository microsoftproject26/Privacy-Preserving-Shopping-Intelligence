"""Warm federated arms started from PRE's best inner-validation checkpoint, and from a longer-budget PRE run, on a
tiny synthetic release (recipe pre_start)."""
from __future__ import annotations

import json

import numpy as np
import pytest
from benchrun_testkit import make_release

from ppsi.benchrun import central, fl_engine, recipe
from ppsi.benchrun import data as D
from ppsi.benchrun.common import BenchRefused, read_json

DS = "s3_beauty"
SEED = 2026
FL = {"server_opt": "fedadam", "server_lr": 0.02, "client_lr": 0.08, "server_tau": 1e-6}


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(central, "EFFECTIVE_BATCH", 8)
    base = tmp_path_factory.mktemp("bench_pre_start")
    rel = make_release(base / "processed", n_users=60, n_items=30)
    d = D.load(DS, rel["root"])
    runs = base / "runs" / DS
    for name, budget, drop in (("PRE_s2026", 6, (0.1,)), ("PRE__B24_L50_D0.5_s2026", 24, (0.5,))):
        out = central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=1e-3, dropout=drop, seed=SEED,
                                  rows=d.band_rows(), select_users=np.flatnonzero(d.band), run_dir=runs / name,
                                  run_id=f"t_{name}", budget_efe=budget)
        assert out["status"] == "DONE_MAIN"

    def rcp(**pre_start):
        p = base / f"r_{len(list(base.glob('r_*.json')))}.json"
        p.write_text(json.dumps({"fl": FL, "pre_start": pre_start}), encoding="utf-8")
        return recipe.load_recipe(p)
    yield {"d": d, "runs": runs, "rcp": rcp}
    mp.undo()


def test_warm_start_from_the_best_checkpoint(world):
    runs = world["runs"]
    out = fl_engine.run_fl(data=world["d"], arm="FA_WARM", seed=SEED, run_dir=runs / "FA_WARM_best_s2026",
                           run_id="t_best", rcp=world["rcp"](checkpoint="best"), pre_run_dir=runs / "PRE_s2026")
    assert out["status"] == "DONE_MAIN"
    init = read_json(runs / "FA_WARM_best_s2026" / "RESULT.json")["init_from"]
    pre = read_json(runs / "PRE_s2026" / "RESULT.json")["checkpoints"]
    assert init["fl_init_kind"] == "practical_best"
    assert init["checkpoint_sha256"] == pre["PRACTICAL_BEST"]["sha256"]


def test_a_longer_budget_pre_needs_match_cell_off(world):
    runs, pre24 = world["runs"], world["runs"] / "PRE__B24_L50_D0.5_s2026"
    with pytest.raises(BenchRefused, match="6-EFE"):
        fl_engine.run_fl(data=world["d"], arm="FA_WARM", seed=SEED, run_dir=runs / "FA_WARM_refused_s2026",
                         run_id="t_refused", rcp=world["rcp"](checkpoint="best"), pre_run_dir=pre24)
    out = fl_engine.run_fl(data=world["d"], arm="FA_WARM", seed=SEED, run_dir=runs / "FA_WARM_pre24_s2026",
                           run_id="t_pre24", rcp=world["rcp"](checkpoint="best", match_cell=False),
                           pre_run_dir=pre24)
    assert out["status"] == "DONE_MAIN"
    init = read_json(runs / "FA_WARM_pre24_s2026" / "RESULT.json")["init_from"]
    assert init["checkpoint_sha256"] == read_json(pre24 / "RESULT.json")["checkpoints"]["PRACTICAL_BEST"]["sha256"]
