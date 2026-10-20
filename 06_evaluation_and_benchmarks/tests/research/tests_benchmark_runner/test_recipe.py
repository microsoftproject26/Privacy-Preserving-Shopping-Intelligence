"""Recipe tests: the mirrored REES46 arm specs equal v4int.launch_fl.arm_spec (when the r6 tree is importable)."""
from __future__ import annotations

import pytest

from v4bench import recipe
from v4bench.common import BenchRefused

IDENT = {"kind": "PRETRAIN_FOR_FL", "run_label": "PRETRAIN_FOR_FL", "fl_init_kind": "endpoint_6.0",
         "train_manifest": "PRETRAIN_TRAIN_5_100_NONMAIN", "family": "SASREC", "run_id": "PRETRAIN_FOR_FL_x",
         "checkpoint_sha256": "a" * 64, "state_digest": "b" * 64, "checkpoint_bytes": 1, "model_variant": "SASREC_D64_B2",
         "seed": 2026}
PEAK = 0.0007114121116984968


def test_rounds_rule_matches_the_registered_numbers():
    assert recipe.fa_rounds(32768, 2) == 1707 and recipe.fa_rounds(131072, 2) == 6827     # 06J §3 numbers
    assert recipe.fa_rounds(22363, 1) == 2330
    assert recipe.fa_rounds(6040, 1) == -(-60 * 6040 // 576)


def test_tuning_point_values():
    s = recipe.build_fl_spec("FA", seed=2026, n_clients=22363)
    c = s["cfg"]
    assert c["server_opt"] == {"name": "fedadam", "lr": 0.01, "beta1": 0.9, "beta2": 0.99, "tau": 0.001,
                               "v_init": "tau_squared"}
    assert c["solver"] == {"lr": 0.1, "batch_size": 16, "passes": 1, "clip": 1.0, "weight_decay": 0.0, "fused": None,
                           "optimizer": "sgd_m"}
    assert (c["lr_shape"], c["dropout_p"], c["endpoint_rounds"], c["endpoint_efe"]) == ("fl_const", 0.1, 2330, 6.0)
    assert s["plan"] == {"group_size": 64, "sampling": "sweep"}


def test_phase2_and_unimplemented_arms():
    fp = recipe.build_fl_spec("FP_WARM", seed=2026, n_clients=100, init_from=IDENT)
    assert fp["cfg"]["method"] == "FP" and fp["cfg"]["mu"] == 0.01 and fp["cfg"]["solver"]["lr"] == 0.025
    q8 = recipe.build_fl_spec("FA_Q8", seed=2026, n_clients=100)
    assert q8["upload_codec"]["name"] == "Q8_SR_PER_TENSOR"
    for a in ("DP8", "FA_1024_WARM"):                              # DP arms need the S record / theta_pre
        with pytest.raises(BenchRefused):
            recipe.build_fl_spec(a, seed=2026, n_clients=100)


def test_warm_needs_init_and_cold_refuses_it():
    with pytest.raises(BenchRefused):
        recipe.build_fl_spec("FA_WARM", seed=2026, n_clients=100)
    with pytest.raises(BenchRefused):
        recipe.build_fl_spec("FA", seed=2026, n_clients=100, init_from=IDENT)


def test_central_recipes_and_ft_cell():
    assert recipe.central_recipe("C_FULL")["model_variant"] == "SASREC_D256_B3"
    assert recipe.central_recipe("PRE")["peak_lr"] == PEAK
    k = recipe.ft_recipe_kwargs()
    assert k["lr_frac"] == 0.3 and k["passes"] == (4,) and k["peak_lr"] == PEAK


@pytest.mark.parametrize("arm", ["FA", "FA_WARM", "FA_WARM_FROZEN", "FP_WARM", "FA_Q8"])
def test_equals_launch_fl_arm_spec(arm):
    pytest.importorskip("torch")
    try:
        import v4train  # noqa: F401
        from v4int import launch_fl as L
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"v4int.launch_fl is not importable here: {e}")
    t = recipe.TUNING_WINNER
    n = 22363
    warm = recipe.warm_variant(arm)
    w = recipe.WARM_WINNER.get(warm) if warm else None
    ref = L.arm_spec(arm, seed=2026, peak_lr=PEAK, fused=None, lr_mult=1.0, server_opt="fedadam", server_lr=0.01,
                     server_tau=1e-3, lr_shape="fl_const", client_opt="sgd_m", client_lr=0.1, local_passes=1,
                     local_batch=16, init_from=dict(IDENT) if warm else None,
                     warm_lr_factor=w["warm_lr_factor"] if w else None,
                     warm_server_lr_factor=w["warm_server_lr_factor"] if w else None)
    mine = recipe.build_fl_spec(arm, seed=2026, n_clients=n, init_from=dict(IDENT) if warm else None)
    # the registered MAIN round count (N = 131072) is replaced by the benchmark's own rule, everything else is equal
    assert ref["cfg"]["endpoint_rounds"] == L.fa_rounds(131072, 1)
    rc, mc = dict(ref["cfg"]), dict(mine["cfg"])
    assert mc.pop("endpoint_rounds") == recipe.fa_rounds(n, 1) and rc.pop("endpoint_rounds")
    assert rc == mc
    for k in ("plan", "lr_peak_base", "upload_codec", "warm_lr_factor", "warm_server_lr_factor", "frozen_item_tables",
              "init_from", "server_opt", "server_lr", "lr_shape", "client_opt", "client_lr", "local_passes"):
        assert ref.get(k) == mine.get(k), k
    assert L.fa_rounds(n, 1) == recipe.fa_rounds(n, 1)


@pytest.mark.parametrize("mine_arm,ref_arm,z", [("S_CAL", "S_CAL", None), ("FA_1024", "FA_1024", None), ("DP8", "DP", 0.5642),
                                                ("DP8_WARM_FROZEN", "DP_WARM_FROZEN", 0.5642), ("PF", "PF", None),
                                                ("PF_WARM", "PF_WARM", None)])
def test_poisson_and_pf_cfg_equal_launch_fl(mine_arm, ref_arm, z):
    try:
        import v4train  # noqa: F401
        from v4int import launch_fl as L
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"v4int.launch_fl is not importable here: {e}")
    warm = recipe.warm_variant(mine_arm)
    w = recipe.WARM_WINNER.get(warm) if warm else None
    clip = None if ref_arm in ("S_CAL", "PF", "PF_WARM") else 1.5
    ref = L.arm_spec(ref_arm, seed=2026, peak_lr=PEAK, fused=None, clip_norm=clip, z=z, server_opt="fedadam", server_lr=0.01,
                     server_tau=1e-3, lr_shape="fl_const", client_opt="sgd_m", client_lr=0.1, init_from=dict(IDENT) if warm else None,
                     local_passes=1 if ref_arm.startswith("PF") else None,
                     warm_lr_factor=w["warm_lr_factor"] if w else None,
                     warm_server_lr_factor=w["warm_server_lr_factor"] if w else None)
    mine = recipe.build_fl_spec(mine_arm, seed=2026, n_clients=22363, init_from=dict(IDENT) if warm else None, clip_norm=clip)
    rc, mc = dict(ref["cfg"]), dict(mine["cfg"])
    for c in (rc, mc):
        if c.get("dp"):
            c["dp"] = {k: v for k, v in c["dp"].items() if k != "denominator"}     # m = 1024 vs round(2^-7 N)
    if ref_arm in ("PF", "PF_WARM"):                                               # sweep: the benchmark rounds rule differs
        rc.pop("endpoint_rounds"); mc.pop("endpoint_rounds")
    assert rc == mc
