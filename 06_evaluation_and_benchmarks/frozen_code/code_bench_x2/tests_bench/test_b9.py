"""Amendment 07 B9: (2) Track-B warm cells start from the 6-EFE PRE of the same cell, (3) Poisson arms P = 2 / sweep arms
P = 1, (4) the DP row follows FA_WARM*, (5) Wave 2 = Amazon 2023 only."""
from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import make_release
from v4bench import central, common, data as D, fl_engine, recipe, selection
from v4bench.common import BenchRefused

DS = "s3_beauty"
IDENT = {"kind": "PRETRAIN_FOR_FL", "run_label": "PRETRAIN_FOR_FL", "fl_init_kind": "endpoint_6.0",
         "train_manifest": "x", "family": "SASREC", "run_id": "PRETRAIN_FOR_FL_x", "checkpoint_sha256": "a" * 64,
         "state_digest": "b" * 64, "checkpoint_bytes": 1, "model_variant": "SASREC_D64_B2", "seed": 2026}


def test_local_passes_poisson_2_sweep_1():
    for arm in ("FA", "FA_WARM", "FA_WARM_FROZEN", "FA_Q8", "FP"):
        init = IDENT if recipe.warm_variant(arm) else None
        s = recipe.build_fl_spec(arm, seed=2026, n_clients=22363, init_from=init)
        assert s["cfg"]["solver"]["passes"] == 1 and s["cfg"]["solver"]["batch_size"] == 16, arm
        assert s["cfg"]["endpoint_rounds"] == recipe.fa_rounds(22363, 1) and s["plan"]["sampling"] == "sweep"
    for arm in ("S_CAL", "FA_1024", "DP8", "S_CAL_WARM", "DP8_WARM_FROZEN"):
        init = IDENT if recipe.warm_variant(arm) else None
        s = recipe.build_fl_spec(arm, seed=2026, n_clients=22363, init_from=init, clip_norm=None if arm.startswith("S_CAL") else 1.0)
        assert s["cfg"]["solver"]["passes"] == 2 and s["cfg"]["solver"]["batch_size"] == 16, arm
        assert s["cfg"]["endpoint_rounds"] == 384 and s["plan"]["sampling"] == "poisson" and s["cfg"]["dropout_p"] == 0.0


def test_dp_row_follows_the_warm_decision(tmp_path):
    (tmp_path / DS).mkdir()
    (tmp_path / DS / "FA_WARM_DECISION.json").write_text(json.dumps({"FA_WARM_STAR": "FA_WARM"}))
    selection.check_dp_variant(tmp_path, DS, "DP8_WARM")             # S7 -> DP_WARM, S_CAL_WARM: allowed
    selection.check_dp_variant(tmp_path, DS, "S_CAL_WARM")
    selection.check_dp_variant(tmp_path, DS, "DP8")                  # cold twin: not bound
    selection.check_dp_variant(tmp_path, DS, "FA")                   # not a DP-chain arm
    for arm in ("DP8_WARM_FROZEN", "S_CAL_WARM_FROZEN", "FA_1024_WARM_FROZEN"):
        with pytest.raises(BenchRefused, match="FA_WARM"):           # S8 chain while FA_WARM* = S7
            selection.check_dp_variant(tmp_path, DS, arm)
    (tmp_path / DS / "FA_WARM_DECISION.json").unlink()
    (tmp_path / DS / "FA_WARM_DECISION.json").write_text(json.dumps({"FA_WARM_STAR": "FA_WARM_FROZEN"}))
    selection.check_dp_variant(tmp_path, DS, "DP8_WARM_FROZEN")
    with pytest.raises(BenchRefused):
        selection.check_dp_variant(tmp_path, DS, "DP8_WARM")
    (tmp_path / DS / "FA_WARM_DECISION.json").unlink()
    with pytest.raises(BenchRefused, match="missing"):
        selection.check_dp_variant(tmp_path, DS, "DP8_WARM_FROZEN")


def test_amendment_names_map_to_the_runner_arms():
    assert recipe.ALIASES["DP_WARM"] == "DP8_WARM" and recipe.ALIASES["DP8_FROZEN"] == "DP8_WARM_FROZEN"
    for v in recipe.ALIASES.values():
        assert v in recipe.FL_ARMS


def test_wave2_is_amazon_only():
    assert common.WAVE2_DATASETS == ("amazon2023_video_games", "amazon2023_baby_products",
                                     "amazon2023_beauty_and_personal_care")
    for n in ("mbht_taobao", "mbht_tmall"):
        with pytest.raises(BenchRefused):
            common.check_dataset(n)


def test_warm_cell_needs_the_6_efe_pre_of_the_same_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(central, "EFFECTIVE_BATCH", 8)
    rel = make_release(tmp_path / "p", n_users=60, n_items=30)
    d = D.load(DS, rel["root"])
    runs = tmp_path / "runs"
    kw = dict(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7e-4, seed=2026, rows=d.band_rows(),
              select_users=np.flatnonzero(d.band))
    central.run_central(dropout=(0.1,), run_dir=runs / "pre6", run_id="p6", **kw)
    central.run_central(dropout=(0.1,), run_dir=runs / "pre24", run_id="p24", budget_efe=24, **kw)
    central.run_central(dropout=(0.5,), run_dir=runs / "pre6_d05", run_id="p6d", **kw)
    with pytest.raises(BenchRefused, match="6-EFE PRE"):             # a 24-EFE PRE is a PRE-alone row only
        fl_engine.run_fl(data=d, arm="FA_WARM", seed=2026, run_dir=runs / "x1", run_id="x1", pre_run_dir=runs / "pre24")
    with pytest.raises(BenchRefused, match="SAME"):                  # the PRE must come from the same (max_len, dropout)
        fl_engine.run_fl(data=d, arm="FA_WARM", seed=2026, run_dir=runs / "x2", run_id="x2", pre_run_dir=runs / "pre6_d05")
    out = fl_engine.run_fl(data=d, arm="FA_WARM", seed=2026, run_dir=runs / "ok", run_id="ok", pre_run_dir=runs / "pre6_d05",
                           dropout=(0.5,))
    assert out["status"] == "DONE_MAIN"
