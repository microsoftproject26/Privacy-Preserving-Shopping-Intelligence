"""End-to-end tests on a tiny synthetic release: central C_FULL / PRE, FL FA / FA_WARM / FA_WARM_FROZEN, FT, LOCK + TEST gate.

The real FLRun / Server / client code runs (60 users -> 7 rounds); only the central effective batch is lowered to 8 so that
the 6-pass schedule has more than one update per mark on 70-400 decisions.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import make_release
from v4bench import central, data as D, device, final, fl_engine
from v4bench.common import BenchRefused, read_json, sha256_file

DS = "s3_beauty"
SEED = 2026


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setattr(central, "EFFECTIVE_BATCH", 8)
    base = tmp_path_factory.mktemp("bm")
    rel = make_release(base / "processed", n_users=60, n_items=30)
    d = D.load(DS, rel["root"])
    runs = base / "runs"
    w = {"d": d, "root": rel["root"], "runs": runs, "base": base}

    def rd(name):
        return runs / DS / name

    out = central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7.114e-4, dropout=(0.1,), seed=SEED,
                              rows=d.band_rows(), select_users=np.flatnonzero(d.band), run_dir=rd("PRE_s2026"),
                              run_id="t_pre")
    assert out["status"] == "DONE_MAIN"
    central.run_central(data=d, arm="C_FULL", variant="SASREC_D256_B3", peak_lr=4.0e-4, dropout=(0.1,), seed=SEED,
                        rows=np.arange(d.N), select_users=None, run_dir=rd("C_FULL_s2026"), run_id="t_cfull")
    central.run_central(data=d, arm="PRE", variant="SASREC_D64_B2", peak_lr=7.114e-4, dropout=(0.1,), seed=SEED,
                        rows=d.band_rows(), select_users=np.flatnonzero(d.band), run_dir=rd("PRE_again"),
                        run_id="t_pre")
    for arm in ("FA", "FA_WARM", "FA_WARM_FROZEN"):
        fl_engine.run_fl(data=d, arm=arm, seed=SEED, run_dir=rd(f"{arm}_s2026"), run_id=f"t_{arm}",
                         pre_run_dir=rd("PRE_s2026") if arm != "FA" else None)
    w["rd"] = rd
    yield w
    mp.undo()


def _theta_of(world, name, rule="CONTROLLED_BUDGET"):
    rd = world["rd"](name)
    return central.load_weights(rd / read_json(rd / "RESULT.json")["checkpoints"][rule]["file"])["theta"]


def test_phase2_arms_run_end_to_end(world):
    d, rd = world["d"], world["rd"]
    for arm in ("FA_Q8", "FP_WARM"):
        out = fl_engine.run_fl(data=d, arm=arm, seed=SEED, run_dir=rd(f"{arm}_s2026"), run_id=f"t_{arm}",
                               pre_run_dir=rd("PRE_s2026") if arm.endswith("WARM") else None)
        assert out["status"] == "DONE_MAIN"
    q8 = read_json(rd("FA_Q8_s2026") / "ARM_SPEC.json")["arm_spec"]
    assert q8["upload_codec"]["bits"] == 8
    fp = read_json(rd("FP_WARM_s2026") / "ARM_SPEC.json")["arm_spec"]
    assert fp["cfg"]["method"] == "FP" and fp["cfg"]["mu"] == 0.01


def test_central_runs_finish_with_pb_and_cb(world):
    for name in ("PRE_s2026", "C_FULL_s2026"):
        res = read_json(world["rd"](name) / "RESULT.json")
        assert res["status"] == "DONE_MAIN" and set(res["checkpoints"]) == {"PRACTICAL_BEST", "CONTROLLED_BUDGET"}
        assert res["checkpoints"]["CONTROLLED_BUDGET"]["mark"] == 6.0
        for c in res["checkpoints"].values():
            assert sha256_file(world["rd"](name) / c["file"]) == c["sha256"]
        assert set(res["validation"]["PRACTICAL_BEST"]) >= {"ALL", "BAND", "OTHERS", "BAND_COVERED_TARGET", "BAND_UNCOVERED_TARGET"}


def test_central_is_deterministic(world):
    a = read_json(world["rd"]("PRE_s2026") / "RESULT.json")["checkpoints"]
    b = read_json(world["rd"]("PRE_again") / "RESULT.json")["checkpoints"]
    assert a["CONTROLLED_BUDGET"]["sha256"] == b["CONTROLLED_BUDGET"]["sha256"]
    assert a["PRACTICAL_BEST"]["sha256"] == b["PRACTICAL_BEST"]["sha256"]


def test_pre_trains_on_the_band_only(world):
    d = world["d"]
    spec = read_json(world["rd"]("PRE_s2026") / "ARM_SPEC.json")
    assert spec["N_decisions"] == int(d.band_rows().size) < d.N
    assert spec["select_population"] == "BAND"
    assert read_json(world["rd"]("C_FULL_s2026") / "ARM_SPEC.json")["N_decisions"] == d.N


def test_fl_rounds_follow_the_exposure_rule(world):
    d = world["d"]
    spec = read_json(world["rd"]("FA_s2026") / "ARM_SPEC.json")
    assert spec["rounds_full_run"] == -(-60 * 60 // 576) == 7
    s = spec["arm_spec"]
    assert s["cfg"]["solver"]["batch_size"] == 16 and s["cfg"]["solver"]["passes"] == 1
    assert s["cfg"]["server_opt"]["name"] == "fedadam" and s["cfg"]["lr_shape"] == "fl_const"
    assert read_json(world["rd"]("FA_s2026") / "RESULT.json")["status"] == "DONE_MAIN"


def test_frozen_tables_stay_frozen(world):
    pre = _theta_of(world, "PRE_s2026")
    frozen = _theta_of(world, "FA_WARM_FROZEN_s2026")
    free = _theta_of(world, "FA_WARM_s2026")
    for k in ("item_embed.weight", "output_embed", "output_bias"):
        assert torch.equal(pre[k], frozen[k]), k                       # bit-identical to theta_pre
        assert not torch.equal(pre[k], free[k]), k                     # NC: the unfrozen warm arm moved them
    assert any(not torch.equal(pre[k], frozen[k]) for k in pre if k not in
               ("item_embed.weight", "output_embed", "output_bias"))   # the rest of the model trained
    res = read_json(world["rd"]("FA_WARM_FROZEN_s2026") / "RESULT.json")
    assert res["frozen"]["frozen_keys"] == ["item_embed.weight", "output_embed", "output_bias"]
    assert read_json(world["rd"]("FA_WARM_s2026") / "RESULT.json")["frozen"] is None


def test_warm_arms_use_the_registered_factors(world):
    s7 = read_json(world["rd"]("FA_WARM_s2026") / "ARM_SPEC.json")["arm_spec"]
    s8 = read_json(world["rd"]("FA_WARM_FROZEN_s2026") / "ARM_SPEC.json")["arm_spec"]
    assert s7["cfg"]["solver"]["lr"] == pytest.approx(0.025) and s7["cfg"]["server_opt"]["lr"] == pytest.approx(0.001)
    assert s8["cfg"]["solver"]["lr"] == pytest.approx(0.1) and s8["cfg"]["server_opt"]["lr"] == pytest.approx(0.001)
    assert "frozen_item_tables" not in s7 and s8["frozen_item_tables"]
    init = read_json(world["rd"]("FA_WARM_s2026") / "RESULT.json")["init_from"]
    assert init["checkpoint_sha256"] == read_json(world["rd"]("PRE_s2026") / "RESULT.json")[
        "checkpoints"]["CONTROLLED_BUDGET"]["sha256"]


def test_warm_arm_needs_theta_pre(world):
    with pytest.raises(BenchRefused):
        fl_engine.run_fl(data=world["d"], arm="FA_WARM", seed=SEED, run_dir=world["base"] / "x", run_id="x")


def test_nc_unfinished_pre_refused(world, tmp_path):
    (tmp_path / "RESULT.json").write_text('{"status": "RUNNING", "arm": "PRE"}')
    with pytest.raises(BenchRefused):
        fl_engine.load_theta_pre(tmp_path, DS)


def test_ft_run_and_gate(world):
    d, rd = world["d"], world["rd"]
    out = device.run_ft(data=d, data_root=world["root"], arm="FT_FA_WARM", seed=SEED,
                        base_run_dir=rd("FA_WARM_s2026"), run_dir=rd("FT_FA_WARM_s2026"), run_id="t_ft", workers=1,
                        threads=1)
    assert out["status"] == "DONE_MAIN" and set(out["validation"]) == {"PRACTICAL_BEST", "CONTROLLED_BUDGET"}
    with pytest.raises(BenchRefused):                                   # a wrong-arm base is refused
        device.run_ft(data=d, data_root=world["root"], arm="FT_FA", seed=SEED, base_run_dir=rd("FA_WARM_s2026"),
                      run_dir=rd("FT_bad"), run_id="x", workers=1, threads=1)

    runs, names = world["runs"], ["C_FULL_s2026", "PRE_s2026", "FA_s2026", "FA_WARM_s2026", "FT_FA_WARM_s2026"]
    with pytest.raises(BenchRefused, match="LOCK|does not exist"):      # no LOCK -> no TEST
        final.open_test_gate(runs, DS)
    with pytest.raises(BenchRefused):
        final.final_eval(runs, DS, d, world["root"])
    assert not (runs / DS / "TEST_LOOK.json").exists() and not (runs / DS / "TEST_RESULTS.json").exists()

    with pytest.raises(BenchRefused):                                   # an unfinished / absent run cannot be locked
        final.make_lock(runs, DS, names + ["NOPE_s1"], data_fingerprint=d.fingerprint(), strict=False)
    with pytest.raises(BenchRefused, match="missing|absent"):                  # the production LOCK needs the decisions first
        final.make_lock(runs, DS, names, data_fingerprint=d.fingerprint())
    assert not final.lock_path(runs, DS).exists()
    lock = final.make_lock(runs, DS, names, data_fingerprint=d.fingerprint(), strict=False)
    assert [r["run"] for r in lock["runs"]] == sorted(names)
    with pytest.raises(BenchRefused):                                   # the LOCK is write-once
        final.make_lock(runs, DS, names, data_fingerprint=d.fingerprint(), strict=False)
    with pytest.raises(BenchRefused, match="not listed"):               # a run outside the LOCK is refused
        final.verify_lock(runs, DS, ["FA_WARM_FROZEN_s2026"])

    ck = rd("PRE_s2026") / "ckpt" / "best.pt"                           # NC: a checkpoint changed after the LOCK
    raw = ck.read_bytes()
    ck.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(BenchRefused):
        final.verify_lock(runs, DS)
    ck.write_bytes(raw)

    with pytest.raises(BenchRefused):                                   # the holdout is unreadable without the token
        D._load_holdout_view(d, world["root"], _gate="x" * 32)

    res = final.final_eval(runs, DS, d, world["root"], workers=1, threads=1)
    assert set(res["runs"]) == set(names)
    for per in res["runs"].values():
        assert set(per) == {"PRACTICAL_BEST", "CONTROLLED_BUDGET"}
        assert per["PRACTICAL_BEST"]["ALL"]["n_users"] == res["n_test_users"] > 0
        assert set(per["PRACTICAL_BEST"]) >= {"ALL", "BAND", "OTHERS", "BAND_COVERED_TARGET", "BAND_UNCOVERED_TARGET"}
    with pytest.raises(BenchRefused, match="already been taken"):       # ONE look per dataset
        final.final_eval(runs, DS, d, world["root"])
