"""The recipe: defaults, merging a JSON file, validation, the rounds rule and the federated arm specs."""
from __future__ import annotations

import json

import pytest

from ppsi.benchrun import recipe
from ppsi.benchrun.common import BenchRefused


def test_default_recipe_loads_and_is_hashed():
    r = recipe.load_recipe()
    assert len(r["recipe_sha256"]) == 64 and r["fl"]["model_variant"] == "SASREC_D64_B2"
    assert recipe.load_recipe()["recipe_sha256"] == r["recipe_sha256"]
    assert set(recipe.ALL_ARMS) >= {"C_FULL", "PRE", "FA", "FA_WARM", "FA_WARM_FROZEN", "DP8_WARM_FROZEN", "FT_FA"}
    assert recipe.WARM_LINE["FP_WARM"] == ("FP", "FA_WARM") and recipe.WARM_LINE["DP8_WARM_FROZEN"] == (
        "DP8", "FA_WARM_FROZEN")


def test_a_partial_file_is_merged_and_unknown_keys_refused(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"fl": {"client_lr": 0.2}, "central": {"PRE": {"peak_lr": 5e-4}}}), encoding="utf-8")
    r = recipe.load_recipe(p)
    assert r["fl"]["client_lr"] == 0.2 and r["fl"]["local_batch"] == 16 and r["central"]["PRE"]["peak_lr"] == 5e-4
    assert r["recipe_sha256"] != recipe.load_recipe()["recipe_sha256"]
    for bad in ({"fl": {"clientlr": 0.2}}, {"nope": {}}, {"fl": {"server_opt": "sgd"}}, {"fl": {"client_lr": -1}},
                {"knobs": {"dropout_p": 1.0}}, {"warm": {"FA_WARM": {"warm_lr_factor": 0}}}):
        p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(BenchRefused):
            recipe.load_recipe(p)


def test_rounds_rule():
    r = recipe.load_recipe()
    assert [recipe.fa_rounds(n, 1, r) for n in (60, 6040, 22363)] == [-(-60 * n // 576) for n in (60, 6040, 22363)]
    assert recipe.fa_rounds(6040, 2, r) == -(-60 * 6040 // (576 * 2))
    assert [recipe.dp_cohort(n, r) for n in (6040, 22363, 300)] == [47, 175, 2]


def test_sweep_spec_from_the_recipe():
    r = recipe.load_recipe()
    s = recipe.build_fl_spec("FA", seed=2026, n_clients=600, recipe=r)
    assert s["plan"] == {"group_size": 64, "sampling": "sweep"} and s["max_rounds"] == -(-60 * 600 // 576)
    assert s["cfg"]["server_opt"] is None and s["cfg"]["solver"]["lr"] == 0.05 and s["cfg"]["dropout_p"] == 0.1
    assert s["cfg"]["exposure_point"] == "end" and s["upload_codec"] is None
    assert recipe.build_fl_spec("FA_Q8", seed=2026, n_clients=600, recipe=r)["upload_codec"]["bits"] == 8
    pf = recipe.build_fl_spec("PF", seed=2026, n_clients=600, recipe=r)
    assert pf["cfg"]["method"] == "PF" and pf["cfg"]["pf"]["lr"] == r["fl"]["lr_peak_base"]


def test_warm_needs_init_and_cold_refuses_it():
    r = recipe.load_recipe()
    with pytest.raises(BenchRefused):
        recipe.build_fl_spec("FA_WARM", seed=2026, n_clients=100, recipe=r)
    with pytest.raises(BenchRefused):
        recipe.build_fl_spec("FA", seed=2026, n_clients=100, recipe=r, init_from={"kind": "PRE"})
    s = recipe.build_fl_spec("FP_WARM_FROZEN", seed=2026, n_clients=100, recipe=r, init_from={"kind": "PRE"})
    assert s["frozen_item_tables"] == list(recipe.FROZEN_ITEM_KEYS) and s["cfg"]["method"] == "FP"


def test_server_optimisers_and_warm_server_factor(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"fl": {"server_opt": "fedadam", "server_lr": 0.02},
                             "warm": {"FA_WARM": {"warm_server_lr_factor": 0.2, "warm_lr_factor": 0.5}}}),
                 encoding="utf-8")
    r = recipe.load_recipe(p)
    cold = recipe.build_fl_spec("FA", seed=1, n_clients=100, recipe=r)["cfg"]["server_opt"]
    assert cold == {"name": "fedadam", "lr": 0.02, "beta1": 0.9, "beta2": 0.99, "v_init": "tau_squared", "tau": 1e-3}
    warm = recipe.build_fl_spec("FA_WARM", seed=1, n_clients=100, recipe=r, init_from={"k": 1})["cfg"]
    assert warm["server_opt"]["lr"] == 0.02 / 5 and warm["solver"]["lr"] == pytest.approx(0.5 * 0.05)
    p.write_text(json.dumps({"fl": {"server_opt": "fedavgm", "server_lr": 1.0}}), encoding="utf-8")
    m = recipe.build_fl_spec("FA", seed=1, n_clients=100, recipe=recipe.load_recipe(p))["cfg"]["server_opt"]
    assert m == {"name": "fedavgm", "lr": 1.0, "beta1": 0.9}


def test_dp_specs():
    r = recipe.load_recipe()
    s = recipe.build_fl_spec("S_CAL", seed=2026, n_clients=22363, recipe=r)
    assert s["cfg"]["dp"] == {"clip_norm": None, "noise_multiplier": 0.0, "denominator": 175}
    assert s["plan"] == {"group_size": 175, "sampling": "poisson"} and s["max_rounds"] == 20
    d8 = recipe.build_fl_spec("DP8", seed=2027, n_clients=22363, recipe=r, clip_norm=1.5)
    assert d8["cfg"]["dp"]["noise_multiplier"] == r["dp"]["z"] and d8["cfg"]["endpoint_rounds"] == 384
    assert d8["cfg"]["dropout_p"] == 0.0 and d8["cfg"]["solver"]["passes"] == 2
    assert d8["dp_accounting"]["T"] == 384 and d8["dp_accounting"]["m"] == 175
    assert recipe.build_fl_spec("FA_1024", seed=2026, n_clients=22363, recipe=r,
                                clip_norm=1.5)["cfg"]["dp"]["noise_multiplier"] == 0.0
    with pytest.raises(BenchRefused):
        recipe.build_fl_spec("DP8", seed=2026, n_clients=22363, recipe=r)                  # needs S


def test_central_and_device_recipes():
    r = recipe.load_recipe()
    assert recipe.central_recipe("PRE", r)["model_variant"] == "SASREC_D64_B2"
    k = recipe.ft_recipe_kwargs(r)
    assert k == {"arm": "FT_C", "peak_lr": r["fl"]["lr_peak_base"], "lr_frac": 0.1, "passes": (1,), "batch_size": 16}
