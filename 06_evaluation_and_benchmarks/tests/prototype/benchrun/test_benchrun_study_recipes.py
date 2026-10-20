"""The study's benchmark recipes in config/benchrun load, and the fixed recipes give the intended effective rates."""
from __future__ import annotations

import json

import pytest

from ppsi.benchrun import recipe
from ppsi_paths import REPO_ROOT

CONFIG = REPO_ROOT / "config" / "benchrun"
WARM = {"s3_beauty": (0.3, 0.001), "s3_toys": (0.3, 0.001), "s3_sports": (0.025, 0.0003), "ml_1m": (0.1, 0.003),
        "amazon2023_video_games": (0.025, 0.0003), "amazon2023_baby_products": (0.025, 0.0003)}


@pytest.mark.parametrize("path", sorted(CONFIG.glob("*.json")), ids=lambda p: p.stem)
def test_every_study_recipe_loads(path):
    r = recipe.load_recipe(path)
    assert r["fl"]["model_variant"] == "SASREC_D64_B2" and len(r["recipe_sha256"]) == 64
    assert set(json.loads(path.read_text(encoding="utf-8"))) <= set(recipe.DEFAULT_RECIPE)


def test_registered_recipe():
    r = recipe.load_recipe(CONFIG / "registered_recipe.json")
    cold = recipe.build_fl_spec("FA", seed=2026, n_clients=6040, recipe=r)["cfg"]
    assert cold["server_opt"]["lr"] == 0.01 and cold["server_opt"]["tau"] == 1e-3 and cold["solver"]["lr"] == 0.1
    assert cold["endpoint_rounds"] == recipe.fa_rounds(6040, 1, r) == 630
    assert r["pre_start"] == recipe.DEFAULT_PRE_START


@pytest.mark.parametrize("ds", sorted(WARM))
def test_fixed_recipe_effective_rates(ds):
    r = recipe.load_recipe(CONFIG / f"fixed_recipe_{ds}.json")
    cold = recipe.build_fl_spec("FA", seed=2026, n_clients=1000, recipe=r)["cfg"]
    assert cold["server_opt"]["lr"] == 0.001 and cold["server_opt"]["tau"] == 1e-6 and cold["solver"]["lr"] == 0.1
    assert cold["endpoint_rounds"] == recipe.fa_rounds(1000, 1, r) and r["knobs"]["endpoint_efe"] == 24.0
    warm = recipe.build_fl_spec("FA_WARM", seed=2026, n_clients=1000, recipe=r, init_from={"k": 1})["cfg"]
    client_lr, server_lr = WARM[ds]
    assert warm["solver"]["lr"] == pytest.approx(client_lr) and warm["server_opt"]["lr"] == pytest.approx(server_lr)
    assert r["pre_start"] == {"checkpoint": "best", "match_cell": False}
