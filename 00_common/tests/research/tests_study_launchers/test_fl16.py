"""FL-16 (PREG1_AMENDMENT_06N N3.2, 06K §3 DP8_FROZEN, 06L_E E1) — the warm method set in launch_fl: FP / PF / FA_Q8 /
S_CAL / FA_1024 / DP started from theta_pre, bound to the S7 variant (FA_WARM, `*_WARM`) or the S8 variant
(FA_WARM_FROZEN, `*_WARM_FROZEN`, frozen item tables), with the warm DP chain's own S record (CPU, tiny synthetic worlds
of test_fl05 / test_fl12; no real data, no GPU).

Proves: every existing arm spec / RunConfig digest (the FL-10 224-pair pin, the FL-15 32-pair warm pin, a 145-pair
table MEASURED on the pre-FL-16 launch_fl 3c0d91b5) and every pinned trajectory (the 7 cold arms, FA_WARM /
FA_WARM_FROZEN at two tuning points, MEASURED pre-FL-16) are unchanged; a warm-line arm's spec is its cold arm's cfg +
the FA_WARM keys (fl RunConfig digest = the cold arm's at factor 1) and its run from theta_pre is bitwise the cold arm's
engine on the theta_pre world (PF store and Q8 bytes included); the PF personal LR is never multiplied by the warm
factor; the *_WARM_FROZEN arms keep the tables bitwise, count the uploaded bytes only (Q8 too) and clip / noise the
uploaded keys only (DP accounting unchanged); S_CAL_WARM[_FROZEN] writes its own record and never S_RECORD.json;
FA_1024_WARM / DP_WARM[_FROZEN] refuse the cold record, the other variant's record, a record of another theta_pre,
another configuration, a smoke record for a real run, a tampered record and another plan; a MAIN warm-line run takes
both factors only from its variant's registered selection (S7 vs S8), refuses a missing selection, CLI factors and a
missing theta_pre; 06L_E E1: z > 0 warm runs write no CLIP_LOG and no TRAIN_LOSS.
NCs (strict xfail on AssertionError): each new admission / refusal removed or mutated in the real wiring.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

import fl.dp as DPM
import v4int.launch_fl as F
import v4int.lr_screen as L
from fl.synthetic import make_tiny_adapter
from intkit import assert_raises, nc
from test_fl05_launch import run_tiny, tiny_spec
from test_fl10_fedadam_launch import (CALIB_EVAL, PRE_FL10_SHA, PRE_FL10_TRAJ, _tiny, all_grid_digests,  # noqa: F401
                                      ladder)
from test_fl12_launch import pre, s4, s4_spec, untied_world, warm_runs, warm_sel, warm_world  # noqa: F401
from test_fl15 import IDENT, PRE_FL15_WARM_SHA, WIN_025_01, _bind_main, resign_2d, warm_digest_table

T = F.tuning_config
PEAK = 0.000711
S7_LINE = ("FP_WARM", "PF_WARM", "FA_Q8_WARM", "S_CAL_WARM", "FA_1024_WARM", "DP_WARM")
S8_LINE = tuple(a + "_FROZEN" for a in S7_LINE)
# MEASURED on the pre-FL-16 launch_fl (sha256 3c0d91b5…) with scratchpad fl16_pins.py (the same table as pre_table()).
PRE_FL16_TABLE_SHA = "2f4d8990e47ef49593e5b36fabb9f058c41e7214fc4af6a88907eb8b3d47e507"      # 145 (spec, RunConfig) pairs
WINDOWS_PRE_FL16_WARM_TRAJ = {   # original platform's (server digest, TRAIN_LOSS sha256)
    "FA_WARM_FROZEN|default": ("c5fd8a79bf215295d0fef922255be26cb63c6cc49745303de1692cb288c70574",
                               "c9c54463b77c0f9da507dd073af13e56ec620c68b8b1c35d0bc9945cb0c4a722"),
    "FA_WARM_FROZEN|s4": ("c704c156acd0c5773b4a5394861dcbb0c429b9e8de75f2d8aa65bca407a80a0b",
                          "8cff3b23666845b477153c250643ea0bb5be81bb04a720272cf71db0d9d7fbd9"),
    "FA_WARM|default": ("d2fe489d596d3ac4e5c125ca35264e43de3ff89f8ae45ad46023a062390e5f8a",
                        "31d48c05d0bd7257efba36b94ca2de14063f531427bf3df14df4ed3b8c17649e"),
    "FA_WARM|s4": ("5f6b27ea264ab4af1bf48f55f6f0cd7a2952f82d6acf4ce7f148a3c63752bc5c",
                   "739d455566ca68dd130d7dc378391ab7edcf746a6c653d77fbc4910fd13bcd10")}
# Independently reproduced from frozen code_p2r4 and this work tree on Linux.
# Both server and TRAIN_LOSS digests match for all four trajectories. Evidence:
# DELIVERY_REVIEW/evidence/LINUX_WARM_TRAJECTORIES_COMPARISON.json.
LINUX_PRE_FL16_WARM_TRAJ = {
    "FA_WARM|default": ("f334b781d4aa54a14dd5da48450d4a41ba75511a155db167c6d839ac1ab1abd6",
                        "f326873d2dbd34cd0cf3c760721ecdc871634e6aff568e13b4572d3f26e4b278"),
    "FA_WARM|s4": ("49275b55c6cd3d556ede634e5e29fba50e121c2ce9665fdfb5dde9f754373369",
                   "56a54276dbf74b3dc1152ea94f0e6ba73b9e663de42a10a8f13e1d09c68d6b94"),
    "FA_WARM_FROZEN|default": ("22c18ead074f6a55241e9d7b895d2dc8883e7a4e45e99b263891c1e52ca5dc00",
                               "06270ae382a36fa81a24059068cc7afc5e65c059142b718e8ecdf42bf51d5174"),
    "FA_WARM_FROZEN|s4": ("b56cadd38a9b8ce5f8f6680a7d68957a3035fa16a3466d83bc7a1ad917d1061f",
                          "e7fbc95462b7c5d693005ad0249b4b0af5e1c05794d171e4e5a485864f707572")}
PRE_FL16_WARM_TRAJ = LINUX_PRE_FL16_WARM_TRAJ if sys.platform.startswith("linux") else WINDOWS_PRE_FL16_WARM_TRAJ
S4_POINT = dict(server_opt="fedadam", server_lr=0.01, lr_shape="fl_const", client_opt="sgd_m", client_lr=0.1)
TK = ({}, dict(optimizer="fedadam", server_lr=0.01, lr_mult=1.0, lr_shape="fl_const"),
      dict(optimizer="fedavgm", lr_mult=2.0),
      dict(optimizer="fedadam", server_lr=0.01, server_tau=1e-3, lr_shape="fl_const", client_opt="sgd_m",
           client_lr=0.1, local_passes=1),
      dict(optimizer="fedadam", server_lr=0.001, client_opt="sgd_m", client_lr=0.1, lr_shape="fl_const",
           local_passes=4))
# the FL-16 arms' digest table (new_arm_table), MEASURED on the FL-16 launch_fl: pins them against later edits
FL16_NEW_ARM_TABLE_SHA = "5fd7f6d33c62b79ec1d8ff5655f671271f4c10acad3a33b350f0dc00391c636f"   # 440 pairs


@pytest.fixture(scope="module", autouse=True)
def _strict_numerics_module():
    from fl.numerics import enforce_strict_fp32
    torch.set_num_threads(1)
    enforce_strict_fp32()
    yield


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _dp_kw(arm):
    ca = F.base_arm(arm)
    return ({"clip_norm": 0.2843} if ca in ("FA_1024", "DP") else {}) | ({"z": 0.5642} if ca == "DP" else {})


def pre_table() -> dict:
    """The 145-pair table of the existing arms (7 cold + FA_WARM / FA_WARM_FROZEN) x TK x seeds x warm factors."""
    out = {}
    for arm in F.ARMS + ("FA_WARM", "FA_WARM_FROZEN"):
        for seed in (2026, 2027):
            if arm == "S_CAL" and seed != 2026:
                continue
            for i, tk in enumerate(TK):
                eff, prule, _ = F.poisson_tuning(arm, T(**tk))
                if arm in F.WARM_ARMS:
                    wfs = (None, 0.25) if arm == "FA_WARM" else (None, 4.0)
                    sfs = (None,) + ((0.1,) if eff["optimizer"] == "fedadam" else (0.25,))
                else:
                    wfs, sfs = (None,), (None,)
                for wf in wfs:
                    for sf in sfs:
                        kw = dict(_dp_kw(arm))
                        if arm in F.WARM_ARMS:
                            kw.update(init_from=IDENT, warm_lr_factor=wf, warm_server_lr_factor=sf)
                        s = F.arm_spec(arm, seed=seed, peak_lr=PEAK, fused=True, poisson_local_rule=prule,
                                       **kw, **F.tuning_kwargs(eff))
                        out[f"{arm}|{seed}|{i}|{wf}|{sf}"] = [F.spec_digest(s), F.fl_run_config(s, "x", 64).digest()]
    return out


def new_arm_table() -> dict:
    out = {}
    for arm in F.WARM_LINE_ARMS:
        v = F.warm_variant(arm)
        for seed in (2026, 2027):
            if F.base_arm(arm) == "S_CAL" and seed != 2026:
                continue
            for i, tk in enumerate(TK):
                eff, prule, _ = F.poisson_tuning(arm, T(**tk))
                for wf in (None, F.WARM_FACTORS[v][1]):
                    for sf in (None,) + ((0.1,) if eff["optimizer"] == "fedadam" else (0.25,)):
                        s = F.arm_spec(arm, seed=seed, peak_lr=PEAK, fused=True, poisson_local_rule=prule,
                                       init_from=IDENT, warm_lr_factor=wf, warm_server_lr_factor=sf, **_dp_kw(arm),
                                       **F.tuning_kwargs(eff))
                        out[f"{arm}|{seed}|{i}|{wf}|{sf}"] = [F.spec_digest(s), F.fl_run_config(s, "x", 64).digest()]
    return out


def _tsha(d) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def wtiny(arm, ident, **kw):
    """A tiny spec of any arm (warm arms get init_from); Poisson arms: group = the DP denominator = 4 (q = 0.4)."""
    if F.is_warm_arm(arm):
        kw["init_from"] = ident
    if F.is_poisson_arm(arm):
        s = tiny_spec(arm, rounds=24, **{**_dp_kw(arm), **kw})
        s["plan"]["group_size"] = 4
        s["cfg"]["dp"]["denominator"] = 4
        return s
    return tiny_spec(arm, rounds=12, **kw)


def tables(adapter) -> dict:
    return {k: v.detach().clone() for k, v in adapter.broadcast_state().items() if k in F.FROZEN_ITEM_KEYS}


# ================================================================================================ invariance
def test_existing_arm_digests_and_trajectories_unchanged(pre, tmp_path):
    g = all_grid_digests()
    assert len(g) == 224 and _tsha(g) == PRE_FL10_SHA
    w = warm_digest_table()
    assert len(w) == 32 and _tsha(w) == PRE_FL15_WARM_SHA
    p = pre_table()
    assert len(p) == 145 and _tsha(p) == PRE_FL16_TABLE_SHA
    assert F.ARMS == ("FA", "FP", "PF", "FA_1024", "DP", "S_CAL", "FA_Q8") and F.WARM_ARMS == ("FA_WARM", "FA_WARM_FROZEN")
    assert F.POISSON_ARMS == ("FA_1024", "DP", "S_CAL") and F.SWEEP_ARMS == ("FA", "FP", "PF", "FA_Q8", "FA_WARM",
                                                                            "FA_WARM_FROZEN")
    assert F.WARM_STAGE == {"FA_WARM": "S7", "FA_WARM_FROZEN": "S8"}               # lr_screen reads these unchanged
    assert set(F.ALL_ARMS) == set(F.ARMS) | set(F.WARM_ARMS) | set(S7_LINE) | set(S8_LINE)
    for arm in F.ARMS:
        assert F.base_arm(arm) == arm and F.warm_variant(arm) is None and not F.is_warm_arm(arm)
        assert "init_from" not in F.arm_spec(arm, seed=2026, peak_lr=PEAK, **_dp_kw(arm))
    for arm in F.ARMS:                                                              # the 7 pinned cold trajectories
        assert run_tiny(tmp_path, _tiny(arm), arm)["summary"]["server_digest"] == PRE_FL10_TRAJ[arm], arm
    for arm in F.WARM_ARMS:
        for name, kw in (("default", {}), ("s4", S4_POINT)):
            out = run_tiny(tmp_path, tiny_spec(arm, rounds=12, init_from=pre["ident"], **kw), f"{arm}_{name}",
                           world=warm_world(pre))
            got = (out["summary"]["server_digest"], _sha(tmp_path / f"{arm}_{name}" / F.TRAIN_LOSS_FILE))
            assert got == PRE_FL16_WARM_TRAJ[f"{arm}|{name}"], (arm, name)
            assert "dp_noise_dimension" not in out["summary"]


def test_new_arm_digest_table_pinned():
    t = new_arm_table()
    assert len(t) == 440 and _tsha(t) == FL16_NEW_ARM_TABLE_SHA


# ================================================================================================ specs
def test_warm_line_spec_is_the_cold_arm_plus_the_warm_keys():
    assert tuple(F.WARM_LINE) == S7_LINE + S8_LINE
    for arm in S7_LINE + S8_LINE:
        ca, v = F.base_arm(arm), F.warm_variant(arm)
        assert arm in (ca + "_WARM", ca + "_WARM_FROZEN") and v == ("FA_WARM_FROZEN" if arm.endswith("_FROZEN")
                                                                    else "FA_WARM")
        assert F.METHOD_CODE[arm] == F.METHOD_CODE[ca] and F.is_frozen_arm(arm) == arm.endswith("_FROZEN")
        seed = 2026
        for tk in TK:
            eff, prule, _ = F.poisson_tuning(arm, T(**tk))
            ceff, cprule, _ = F.poisson_tuning(ca, T(**tk))
            assert (eff, prule) == (ceff, cprule)                                  # the DL 219 rule as the cold arm
            cold = F.arm_spec(ca, seed=seed, peak_lr=PEAK, poisson_local_rule=prule, **_dp_kw(arm),
                              **F.tuning_kwargs(eff))
            warm = F.arm_spec(arm, seed=seed, peak_lr=PEAK, init_from=IDENT, poisson_local_rule=prule,
                              **_dp_kw(arm), **F.tuning_kwargs(eff))
            extra = {"init_from"} | ({"frozen_item_tables"} if F.is_frozen_arm(arm) else set())
            assert set(warm) == set(cold) | extra and warm["arm"] == arm
            assert {k: v2 for k, v2 in warm.items() if k not in extra | {"arm"}} == \
                   {k: v2 for k, v2 in cold.items() if k != "arm"}
            assert F.fl_run_config(warm, "x", 64).digest() == F.fl_run_config(cold, "x", 64).digest()
            assert F.spec_digest(warm) != F.spec_digest(cold)
            assert F.spec_tuning(warm) == F.spec_tuning(cold)
            F.check_warm_spec(warm)
        lab = F.warm_row_labels(F.arm_spec(arm, seed=seed, peak_lr=PEAK, init_from=IDENT, **_dp_kw(arm)))
        assert lab["fl_init"] == "PRETRAIN_FOR_FL" and ("fl_frozen_item_tables" in lab) == F.is_frozen_arm(arm)


def test_warm_factors_pf_personal_lr_q8_codec_and_refusals():
    pf = F.arm_spec("PF_WARM", seed=2026, peak_lr=PEAK, init_from=IDENT, warm_lr_factor=0.25, lr_mult=2.0)
    assert pf["cfg"]["lr_peak_local"] == 0.25 * (2.0 * PEAK) == pf["cfg"]["solver"]["lr"]
    assert pf["cfg"]["pf"] == F.arm_spec("PF", seed=2026, peak_lr=PEAK)["cfg"]["pf"]      # personal LR = SMALL peak
    pfs = F.arm_spec("PF_WARM_FROZEN", seed=2026, peak_lr=PEAK, init_from=IDENT, warm_lr_factor=4.0, **S4_POINT)
    assert pfs["cfg"]["pf"]["lr"] == PEAK and pfs["cfg"]["lr_peak_local"] == 4.0 * 0.1
    q8 = F.arm_spec("FA_Q8_WARM_FROZEN", seed=2026, peak_lr=PEAK, init_from=IDENT)
    assert q8["upload_codec"] == F.Q8_CODEC and q8["cfg"]["method"] == "FA"
    fp = F.arm_spec("FP_WARM", seed=2026, peak_lr=PEAK, init_from=IDENT)
    assert fp["cfg"]["mu"] == F.KNOBS["mu"]
    dp = F.arm_spec("DP_WARM_FROZEN", seed=2027, peak_lr=PEAK, init_from=IDENT, clip_norm=0.3, z=0.5642,
                    server_opt="fedadam", server_lr=0.01, warm_server_lr_factor=0.1)
    assert dp["cfg"]["dp"] == {"clip_norm": 0.3, "noise_multiplier": 0.5642, "denominator": 1024}
    assert dp["cfg"]["server_opt"]["lr"] == 0.001 and dp["plan"] == {"group_size": 1024, "sampling": "poisson"}
    assert dp["cfg"]["endpoint_rounds"] == 384 and dp["frozen_item_tables"] == list(F.FROZEN_ITEM_KEYS)
    sc = F.arm_spec("S_CAL_WARM", seed=2026, peak_lr=PEAK, init_from=IDENT)
    assert sc["max_rounds"] == 20 and sc["cfg"]["dp"]["clip_norm"] is None
    bad_init = dict(IDENT, train_manifest="MAIN_TRAIN_5_100_131072")
    for arm, kw in (("FP_WARM", {}),                                                 # no theta_pre
                    ("FP", {"init_from": IDENT}),                                     # theta_pre on a cold arm
                    ("FP_WARM", {"init_from": IDENT, "warm_lr_factor": 4.0}),         # the S8 grid on an S7 arm
                    ("PF_WARM_FROZEN", {"init_from": IDENT, "warm_lr_factor": 0.25}),  # the S7 grid on an S8 arm
                    ("FA_Q8_WARM", {"init_from": IDENT, "warm_server_lr_factor": 0.1}),   # FedAvg axis is {1, 0.25}
                    ("DP_WARM", {"init_from": bad_init, "clip_norm": 0.3, "z": 0.5642}),   # MAIN-trained init
                    ("S_CAL_WARM", {"init_from": IDENT, "seed": 2027}),               # S_CAL: seed 2026 only
                    ("FA_1024_WARM", {"init_from": IDENT}),                           # no clip record
                    ("DP_WARM", {"init_from": IDENT, "clip_norm": 0.3}),              # no z
                    ("FA_1024_WARM", {"init_from": IDENT, "clip_norm": 0.3, "local_passes": 1}),   # DL 219
                    ("FP_WARM", {"init_from": IDENT, "role": "CALIBRATION"})):        # never a CALIB screen arm
        kw = dict(kw)
        assert_raises(F.LaunchRefused, F.arm_spec, arm, seed=kw.pop("seed", 2026), peak_lr=PEAK, **kw)
    s = F.arm_spec("FP_WARM_FROZEN", seed=2026, peak_lr=PEAK, init_from=IDENT)
    for bad in ({k: v for k, v in s.items() if k != "frozen_item_tables"}, dict(s, arm="FP_WARM"),
                dict(s, arm="FP"), dict(s, warm_lr_factor=0.25)):
        assert_raises(F.LaunchRefused, F.check_warm_spec, bad)


def _args(tmp_path, pre, arm, *extra, init=True, smoke=True, seed="2026"):
    a = ["--arm", arm, "--seed", seed, "--gpu", "0", "--run-dir", str(tmp_path / "r"), "--history-policy",
         "USER_HISTORY_L50"] + (["--init-from", str(pre["path"])] if init else []) + list(extra)
    return F.parse_args(a + (["--smoke"] if smoke else []))


def _flags(arm):
    ca = F.base_arm(arm)
    out = ["--freeze-item-tables"] if F.is_frozen_arm(arm) else []
    if ca in ("FA_1024", "DP"):
        out += ["--clip-record", "s.json"]
    if ca == "DP":
        out += ["--epsilon", "8"]
    return out


def test_warm_line_preflight_admissions_refusals_and_run_ids(pre, tmp_path):
    for arm in S7_LINE + S8_LINE:
        ok = _args(tmp_path, pre, arm, *_flags(arm))
        F.fl05_preflight(ok)                                                      # smoke: admitted
        F.preflight(_args(tmp_path, pre, arm, *_flags(arm), "--dry-run"))
        F.fl05_preflight(_args(tmp_path, pre, arm, *_flags(arm), "--warm-lr-factor",
                               str(F.WARM_FACTORS[F.warm_variant(arm)][1])))
        sel = str(tmp_path / "W.json")
        bad = [(_flags(arm), dict(init=False)),                                    # no theta_pre
               ([f for f in _flags(arm) if f != "--freeze-item-tables"] + ([] if F.is_frozen_arm(arm)
                                                                          else ["--freeze-item-tables"]), {}),
               (_flags(arm) + ["--warm-lr-factor", "0.5"], {}),                   # unregistered factor
               (_flags(arm) + ["--role", "CALIBRATION"], {}),                     # never a CALIB screen arm
               (_flags(arm), dict(smoke=False)),                                  # MAIN without --warm-selection
               (_flags(arm) + ["--warm-selection", sel, "--warm-lr-factor", "0.25"], dict(smoke=False)),
               (_flags(arm) + ["--warm-selection", sel, "--warm-server-lr-factor", "0.25"], dict(smoke=False)),
               (_flags(arm) + ["--warm-selection", sel], {})]                     # selection on a smoke run
        for extra, kw in bad:
            assert_raises(F.LaunchRefused, F.fl05_preflight, _args(tmp_path, pre, arm, *extra, **kw))
    F.fl05_preflight(_args(tmp_path, pre, "FP_WARM", "--warm-selection", str(tmp_path / "W.json"), smoke=False))
    pf = lambda a, *x, **k: F.preflight(_args(tmp_path, pre, a, "--dry-run", *x, **k))      # noqa: E731
    assert_raises(F.LaunchRefused, pf, "DP_WARM", "--clip-record", "s.json")               # no --epsilon
    assert_raises(F.LaunchRefused, pf, "FA_1024_WARM", "--clip-record", "s.json", "--epsilon", "8")
    assert_raises(F.LaunchRefused, pf, "DP_WARM_FROZEN", "--freeze-item-tables", "--epsilon", "8")   # no record
    assert_raises(F.LaunchRefused, pf, "S_CAL_WARM", seed="2027")
    rid = lambda a, *x: F.default_run_id(_args(tmp_path, pre, a, *x, smoke=False))        # noqa: E731
    assert rid("DP_WARM_FROZEN", "--epsilon", "8") == "FL_DP_WARM_FROZEN_eps8_SMALL_s2026"
    assert rid("PF_WARM") == "FL_PF_WARM_SMALL_s2026" and rid("S_CAL_WARM") == "FL_S_CAL_WARM_SMALL_s2026"
    assert F.default_run_id(_args(tmp_path, pre, "DP", "--epsilon", "1", init=False, smoke=False)) == \
        "FL_DP_eps1_SMALL_s2026"                                                   # the cold id unchanged


# ================================================================================================ engine
def test_warm_line_runs_are_the_cold_engine_from_theta_pre(pre, tmp_path):
    ident = pre["ident"]
    for arm in S7_LINE:
        ca = F.base_arm(arm)
        c = run_tiny(tmp_path, wtiny(ca, ident), f"c_{ca}", world=warm_world(pre))       # the cold arm on theta_pre
        w = run_tiny(tmp_path, wtiny(arm, ident), f"w_{arm}", world=warm_world(pre))
        done = "DONE_S_CAL" if ca == "S_CAL" else "DONE_MAIN"
        assert c["status"] == w["status"] == done, arm
        assert w["summary"]["server_digest"] == c["summary"]["server_digest"], arm
        assert w["summary"]["counters"] == c["summary"]["counters"]
        assert w["summary"]["bytes"] == c["summary"]["bytes"], arm                        # Q8 bytes included
        if ca == "PF":                                                                    # the personal component
            assert w["summary"]["pf_store"]["digest"] == c["summary"]["pf_store"]["digest"]
            assert (tmp_path / f"w_{arm}" / F.PF_STORE_BEST_FILE).is_file()
        if ca == "FA_Q8":
            assert w["summary"]["bytes"]["per_visit"]["upload_codec"] == "Q8_SR_PER_TENSOR"
        if ca == "S_CAL":
            assert w["summary"]["s_record"]["S"] == c["summary"]["s_record"]["S"]
            continue
        rows = _json(tmp_path / f"w_{arm}" / "RESULT_ROWS.json")["rows"]
        assert rows and all(r["method"] == arm and r["method_code"] == F.METHOD_CODE[ca] and r["fl_init"] ==
                            "PRETRAIN_FOR_FL" and r["fl_init_checkpoint_sha256"] == ident["checkpoint_sha256"]
                            for r in rows)
        if ca in ("FA_1024", "DP"):
            assert w["summary"]["dp_noise_dimension"]["uploaded_numel"] == \
                make_tiny_adapter(seed=1, tied=False).manifest.shared_numel
    cold = run_tiny(tmp_path, wtiny("FP", ident), "cold_theta0", world=untied_world())
    assert cold["summary"]["server_digest"] != run_tiny(tmp_path, wtiny("FP_WARM", ident), "fpw",
                                                        world=warm_world(pre))["summary"]["server_digest"]
    q = run_tiny(tmp_path, wtiny("FA_Q8_WARM", ident, warm_lr_factor=0.25), "q8f", world=warm_world(pre))
    assert _json(tmp_path / "q8f" / "RESULT_ROWS.json")["rows"][0]["fl_warm_lr_factor"] == 0.25
    assert q["summary"]["server_digest"] != run_tiny(tmp_path, wtiny("FA_Q8_WARM", ident), "q81",
                                                     world=warm_world(pre))["summary"]["server_digest"]


def test_frozen_variants_tables_bytes_and_resume(pre, tmp_path):
    ident = pre["ident"]
    full = make_tiny_adapter(seed=1, tied=False).manifest
    fb = sum(full.entries[k].nbytes for k in F.FROZEN_ITEM_KEYS)
    t0 = tables(pre["adapter"])
    for arm in S8_LINE:
        w = warm_world(pre)
        out = run_tiny(tmp_path, wtiny(arm, ident), arm, world=w)
        assert out["status"] in ("DONE_MAIN", "DONE_S_CAL"), arm
        st = tables(w["server_adapter"])
        assert all(torch.equal(t0[k], st[k]) for k in F.FROZEN_ITEM_KEYS), arm              # bitwise at theta_pre
        by = out["summary"]["bytes"]["per_visit"]
        assert by["download_bytes"] == full.shared_bytes - fb and by["frozen_bytes_once"] == fb, arm
        assert by["uploaded_numel"] == w["server_adapter"].manifest.shared_numel
        if F.base_arm(arm) == "FA_Q8":                   # Q8 on the uploaded keys: sum(numel) + 4 x n_tensors (+ 8)
            man = w["server_adapter"].manifest
            assert by["codec_payload_bytes"] == man.shared_numel + 4 * len(man.shared_keys)
            assert by["upload_bytes"] == by["codec_payload_bytes"] + 8 and len(man.shared_keys) == \
                len(full.shared_keys) - len(F.FROZEN_ITEM_KEYS)
            assert "per_client_incl_frozen_once" in out["summary"]["bytes"]
        else:
            assert by["upload_bytes"] == full.shared_bytes - fb + 8
        assert out["summary"]["frozen_item_tables"]["frozen_keys"] == list(F.FROZEN_ITEM_KEYS)
        nonfrozen = run_tiny(tmp_path, wtiny(arm.replace("_FROZEN", ""), ident), arm + "_nf", world=warm_world(pre))
        assert nonfrozen["summary"]["server_digest"] != out["summary"]["server_digest"], arm
    spec = wtiny("DP_WARM_FROZEN", ident)
    straight = run_tiny(tmp_path, spec, "dpf_s", world=warm_world(pre))
    assert run_tiny(tmp_path, spec, "dpf_p", world=warm_world(pre), interrupt_at=11)["status"] == "INTERRUPTED"
    res = run_tiny(tmp_path, spec, "dpf_p", world=warm_world(pre), resume=True)
    assert res["summary"]["server_digest"] == straight["summary"]["server_digest"]


def _spy_dp(monkeypatch) -> dict:
    seen = {"clip": set(), "noise": set(), "noise_numel": set()}
    real_norm, real_noise = DPM.flat_delta_norm, DPM.noise_like

    def norm(upload, theta_r, keys):
        seen["clip"].add(tuple(keys))
        return real_norm(upload, theta_r, keys)

    def noise(keys, template, **kw):
        seen["noise"].add(tuple(keys))
        seen["noise_numel"].add(sum(int(template[k].numel()) for k in keys))
        return real_noise(keys, template, **kw)
    monkeypatch.setattr(DPM, "flat_delta_norm", norm)
    monkeypatch.setattr(DPM, "noise_like", noise)
    return seen


def test_frozen_dp_clips_and_noises_the_uploaded_dims_only(pre, tmp_path, monkeypatch):
    seen = _spy_dp(monkeypatch)
    w = warm_world(pre)
    out = run_tiny(tmp_path, wtiny("DP_WARM_FROZEN", pre["ident"]), "dpf", world=w)
    up = tuple(w["server_adapter"].manifest.shared_keys)
    assert out["status"] == "DONE_MAIN" and seen["clip"] == {up} and seen["noise"] == {up}
    assert not set(F.FROZEN_ITEM_KEYS) & set(up)
    dim = out["summary"]["dp_noise_dimension"]
    assert dim["uploaded_numel"] == w["server_adapter"].manifest.shared_numel == next(iter(seen["noise_numel"]))
    assert dim["frozen_keys_excluded"] == list(F.FROZEN_ITEM_KEYS)
    full = make_tiny_adapter(seed=1, tied=False).manifest
    assert dim["uploaded_numel"] == full.shared_numel - sum(full.entries[k].numel for k in F.FROZEN_ITEM_KEYS)
    assert out["summary"]["dp"]["noise_multiplier"] == 0.5642 and out["summary"]["dp"]["denominator"] == 4
    seen2 = _spy_dp(monkeypatch)                                           # S7: the full shared dimension
    w2 = warm_world(pre)
    run_tiny(tmp_path, wtiny("DP_WARM", pre["ident"]), "dp7", world=w2)
    assert seen2["clip"] == {tuple(full.shared_keys)} and set(F.FROZEN_ITEM_KEYS) <= set(full.shared_keys)


def test_06l_e_z_positive_warm_runs_record_no_clipping_and_no_train_loss(pre, tmp_path):
    for arm in ("DP_WARM", "DP_WARM_FROZEN", "FA_1024_WARM", "FA_1024_WARM_FROZEN", "S_CAL_WARM", "S_CAL_WARM_FROZEN"):
        out = run_tiny(tmp_path, wtiny(arm, pre["ident"]), arm, world=warm_world(pre))
        d = tmp_path / arm
        z_pos = F.base_arm(arm) == "DP"
        clipped_z0 = F.base_arm(arm) == "FA_1024"
        assert (d / F.CLIP_LOG_FILE).exists() == clipped_z0 and ("clip_log" in out["summary"]) == clipped_z0, arm
        assert (d / F.TRAIN_LOSS_FILE).exists() == (not z_pos), arm
        assert ("train_loss_log" in out["summary"]) == (not z_pos), arm
        if z_pos:
            assert not any("clip" in k for k in out["summary"] if k != "clip_norm")
            assert not F.train_loss_applies(out["summary"]["arm_spec"])
            assert not F.clip_log_applies(out["summary"]["arm_spec"])


# ================================================================================================ the warm S record
@pytest.fixture(scope="module")
def s_chain(pre, tmp_path_factory):
    root = tmp_path_factory.mktemp("schain")
    ident = pre["ident"]
    out = {"root": root}
    for arm in ("S_CAL_WARM", "S_CAL_WARM_FROZEN", "S_CAL"):
        spec = wtiny(arm, ident)
        o = run_tiny(root, spec, arm, world=warm_world(pre) if arm != "S_CAL" else untied_world())
        assert o["status"] == "DONE_S_CAL"
        out[arm] = {"spec": spec, "summary": o["summary"], "dir": root / arm}
    return out


def test_s_cal_warm_writes_its_own_record(s_chain, pre):
    for arm, fname in (("S_CAL_WARM", "S_RECORD_WARM.json"), ("S_CAL_WARM_FROZEN", "S_RECORD_WARM_FROZEN.json")):
        d = s_chain[arm]["dir"]
        assert not (d / "S_RECORD.json").exists() and (d / fname).is_file()
        rec = _json(d / fname)
        assert rec["format"] == F.S_FORMAT_WARM and rec["arm"] == arm and rec["self_sha256"] == F._canon(rec)
        assert rec["init_from"] == pre["ident"] and rec["init_sha256"] == pre["ident"]["state_digest"]
        assert rec["stage"] == ("S8" if arm.endswith("FROZEN") else "S7") and rec["label"] == "S_CALIBRATION"
        assert rec["frozen_item_tables"] == (list(F.FROZEN_ITEM_KEYS) if arm.endswith("FROZEN") else None)
        assert rec["arm_spec_sha256"] == F.spec_digest(s_chain[arm]["spec"])
        assert rec["config_digest"] == F.fl_run_config(s_chain[arm]["spec"], arm, 64).digest()
        assert s_chain[arm]["summary"]["s_record"] == {"S": rec["S"], "self_sha256": rec["self_sha256"],
                                                      "n_norms": rec["n_norms"], "file": fname,
                                                      "format": F.S_FORMAT_WARM}
        assert not list((d / "ckpt").glob("*.pt"))                                     # S_CAL keeps no weights
        assert_raises(F.LaunchRefused, F.load_s_record, d / fname, smoke=False)         # the cold loader refuses it
    cold = s_chain["S_CAL"]["dir"]
    assert (cold / "S_RECORD.json").is_file() and not list(cold.glob("S_RECORD_WARM*.json"))
    assert set(s_chain["S_CAL"]["summary"]["s_record"]) == {"S", "self_sha256", "n_norms"}     # cold summary as before
    s7, s8 = (_json(s_chain[a]["dir"] / f) for a, f in (("S_CAL_WARM", "S_RECORD_WARM.json"),
                                                       ("S_CAL_WARM_FROZEN", "S_RECORD_WARM_FROZEN.json")))
    assert s7["S"] != s8["S"]                                              # S8: the encoder-only delta norms


def _resign(obj: dict, path: Path, **changes) -> Path:
    o = dict(obj, **changes)
    o["self_sha256"] = F._canon(o)
    path.write_text(json.dumps(o), encoding="utf-8")
    return path


def test_warm_dp_takes_only_its_warm_record(s_chain, pre, tmp_path):
    ident = pre["ident"]
    r7 = s_chain["S_CAL_WARM"]["dir"] / "S_RECORD_WARM.json"
    r8 = s_chain["S_CAL_WARM_FROZEN"]["dir"] / "S_RECORD_WARM_FROZEN.json"
    rc = s_chain["S_CAL"]["dir"] / "S_RECORD.json"
    sc7, sc8 = s_chain["S_CAL_WARM"]["spec"], s_chain["S_CAL_WARM_FROZEN"]["spec"]
    keys, mh = warm_world(pre)["keys"], "tiny-manifest"
    for arm, rec, sc in (("FA_1024_WARM", r7, sc7), ("DP_WARM", r7, sc7), ("FA_1024_WARM_FROZEN", r8, sc8),
                         ("DP_WARM_FROZEN", r8, sc8)):
        got = F.load_warm_s_record(rec, arm=arm, scal_spec=sc, init_identity=ident, smoke=False)
        assert got["S"] == _json(rec)["S"]
        assert F.check_warm_s_plan(got, keys, mh, wtiny(arm, ident)) == got["plan_digest"]
        with pytest.raises(F.LaunchRefused, match="cold-start"):                   # the cold record: refused
            F.load_warm_s_record(rc, arm=arm, scal_spec=sc, init_identity=ident, smoke=False)
    obj = _json(r7)
    other = dict(ident, checkpoint_sha256="1" * 64)
    cases = {"s8_record_for_s7": (r8, "FA_1024_WARM", sc7, ident, False),
             "s7_record_for_s8": (r7, "DP_WARM_FROZEN", sc8, ident, False),
             "other_init": (r7, "DP_WARM", sc7, other, False),
             "no_init": (r7, "DP_WARM", sc7, None, False),
             "other_config": (r7, "DP_WARM", dict(sc7, cfg=dict(sc7["cfg"], lr_peak_local=1.0)), ident, False),
             "tampered": (tmp_path / "t.json", "DP_WARM", sc7, ident, False),
             "smoke_record": (_resign(obj, tmp_path / "sm.json", label="SMOKE_NOT_EVIDENCE"), "DP_WARM", sc7, ident,
                              False),
             "other_seed": (_resign(obj, tmp_path / "sd.json", seed=2027), "DP_WARM", sc7, ident, False),
             "other_arm_field": (_resign(obj, tmp_path / "af.json", arm="S_CAL_WARM_FROZEN"), "DP_WARM", sc7, ident,
                                 False),
             "other_init_sha": (_resign(obj, tmp_path / "is.json", init_sha256="0" * 64), "DP_WARM", sc7, ident,
                                False),
             "other_spec_sha": (_resign(obj, tmp_path / "ss.json", arm_spec_sha256="0" * 64), "DP_WARM", sc7, ident,
                                False),
             "cold_format_resigned": (_resign(obj, tmp_path / "cf.json", format=F.S_FORMAT), "DP_WARM", sc7, ident,
                                      False),
             "cold_arm": (r7, "FA_1024", sc7, ident, False)}
    (tmp_path / "t.json").write_text(json.dumps(dict(obj, S=obj["S"] * 2)), encoding="utf-8")
    for name, (path, arm, sc, init, smoke) in cases.items():
        with pytest.raises(F.LaunchRefused):
            F.load_warm_s_record(path, arm=arm, scal_spec=sc, init_identity=init, smoke=smoke)
    F.load_warm_s_record(tmp_path / "sm.json", arm="DP_WARM", scal_spec=sc7, init_identity=ident, smoke=True)
    other_keys = keys[:-1]
    with pytest.raises(F.LaunchRefused, match="plan digest"):
        F.check_warm_s_plan(obj, other_keys, mh, wtiny("DP_WARM", ident))
    with pytest.raises(F.LaunchRefused, match="plan digest"):
        F.check_warm_s_plan(obj, keys, "another-manifest", wtiny("DP_WARM", ident))
    s2027 = wtiny("DP_WARM", ident, seed=2027)                                  # a seed-2027 run: S of seed 2026
    assert F.check_warm_s_plan(obj, keys, mh, s2027) == obj["plan_digest"]


def test_warm_chain_end_to_end(s_chain, pre, tmp_path):
    ident = pre["ident"]
    for var, sarm, fname in (("", "S_CAL_WARM", "S_RECORD_WARM.json"),
                             ("_FROZEN", "S_CAL_WARM_FROZEN", "S_RECORD_WARM_FROZEN.json")):
        rec = F.load_warm_s_record(s_chain[sarm]["dir"] / fname, arm="DP_WARM" + var, scal_spec=s_chain[sarm]["spec"],
                                   init_identity=ident, smoke=False)
        logs = {}
        for arm in ("FA_1024_WARM" + var, "DP_WARM" + var):
            o = run_tiny(tmp_path, wtiny(arm, ident, clip_norm=rec["S"]), arm, world=warm_world(pre))
            s = o["summary"]
            assert o["status"] == "DONE_MAIN" and s["dp"]["clip_norm"] == rec["S"] and s["evals"][-1][0] == 12
            logs[arm] = ({k: s["participation"][k] for k in ("sampled", "cohort_size_min", "cohort_size_max")},
                         s["lr_table"]["digest"])
        assert logs["FA_1024_WARM" + var] == logs["DP_WARM" + var]              # same cohorts / LR table: noise only


# ================================================================================================ MAIN / smoke wiring
def _warm_record(path: Path, scal: dict, ident, *, S=0.37, run_id="FL_S_CAL_WARM_SMALL_s2026", smoke=False) -> Path:
    norm = {"S": S, "n_norms": 20, "rounds_0based": list(range(20)), "rule": "median"}
    obj = F.s_record_obj_warm(norm, spec=scal, run_id=run_id, seed=2026,
                              config_digest=F.fl_run_config(scal, run_id, 64).digest(), plan_digest="p" * 64,
                              init_sha256=ident["state_digest"], smoke=smoke)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _cold_record(path: Path) -> Path:
    obj = F.s_record_obj({"S": 0.5, "n_norms": 20, "rounds_0based": list(range(20)), "rule": "median"},
                         run_id="FL_S_CAL_SMALL_s2026", seed=2026, config_digest="c" * 64, plan_digest="p" * 64,
                         init_sha256="i" * 64, smoke=False)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _scal_for(arm, tun, ident, wf=None, wsf=None):
    eff, prule, _ = F.poisson_tuning(arm, tun)
    return F.warm_s_cal_spec(arm, peak_lr=PEAK, fused=True, init_from=ident, tun=eff, warm_lr_factor=wf,
                             warm_server_lr_factor=wsf, poisson_local_rule=prule)


def test_resolve_spec_binds_the_warm_s_record(warm_sel, pre, tmp_path):
    ident, s4f, w4 = pre["ident"], warm_sel["s4_file"], warm_sel["w4"]
    for arm in ("FA_1024_WARM", "DP_WARM", "FA_1024_WARM_FROZEN", "DP_WARM_FROZEN"):
        rec = _warm_record(tmp_path / arm / "rec.json", _scal_for(arm, w4, ident), ident)
        eps = ["--epsilon", "8"] if F.base_arm(arm) == "DP" else []
        frz = ["--freeze-item-tables"] if F.is_frozen_arm(arm) else []
        args = _args(tmp_path, pre, arm, "--fl-tuning", str(s4f), "--clip-record", str(rec), *eps, *frz, seed="2027")
        spec, extra = F.resolve_spec(args, PEAK, fused=True, init_identity=ident)
        assert spec["cfg"]["dp"]["clip_norm"] == 0.37 and spec["arm"] == arm
        assert spec["cfg"]["dp"]["noise_multiplier"] == (0.5642 if F.base_arm(arm) == "DP" else 0.0)
        assert extra["s_record"]["format"] == F.S_FORMAT_WARM and extra["s_record"]["file_sha256"] == _sha(rec)
        assert (spec["cfg"]["solver"]["passes"], spec["cfg"]["solver"]["batch_size"]) == (2, 16)     # DL 219
        if F.base_arm(arm) == "DP":
            assert extra["dp_z"]["z"] == 0.5642 and extra["dp_z"]["T"] == 384 and extra["dp_z"]["N"] == 131072
        cold = _cold_record(tmp_path / arm / "cold.json")
        args.clip_record = cold
        with pytest.raises(F.LaunchRefused, match="cold-start"):
            F.resolve_spec(args, PEAK, fused=True, init_identity=ident)
        args.clip_record = rec                                                   # the run at another warm factor
        args.warm_lr_factor = F.WARM_FACTORS[F.warm_variant(arm)][1]
        with pytest.raises(F.LaunchRefused, match="config digest"):
            F.resolve_spec(args, PEAK, fused=True, init_identity=ident)
        args.warm_lr_factor = None
        with pytest.raises(F.LaunchRefused, match="theta_pre"):                  # another theta_pre
            F.resolve_spec(args, PEAK, fused=True, init_identity=dict(ident, checkpoint_sha256="1" * 64))
    args = _args(tmp_path, pre, "DP", "--fl-tuning", str(s4f), "--clip-record", str(tmp_path / "DP_WARM" / "rec.json"),
                 "--epsilon", "8", init=False)
    assert_raises(F.LaunchRefused, F.resolve_spec, args, PEAK, fused=True)       # the cold DP refuses a warm record
    args.clip_record = tmp_path / "DP_WARM" / "cold.json"
    assert F.resolve_spec(args, PEAK, fused=True)[1]["s_record"] == {            # the cold path as before
        "path": str(args.clip_record), "S": 0.5, "self_sha256": _json(args.clip_record)["self_sha256"]}


def _reg(tmp_path, monkeypatch, warm_sel, stage, values=None) -> Path:
    arm = "FA_WARM" if stage == "S7" else "FA_WARM_FROZEN"
    src = tmp_path / f"src_{stage}.json"
    L.write_selection_warm(stage, warm_runs(warm_sel, arm), src, warm_sel["s4_file"])
    reg = tmp_path / "contracts_fl" / f"FL_WARM_SELECTION_{stage}.json"
    if values is None:
        reg.parent.mkdir(parents=True, exist_ok=True)
        reg.write_bytes(src.read_bytes())
    else:
        resign_2d(_json(src), reg, values=values)
    monkeypatch.setitem(F.WARM_SELECTION_PATH, stage, reg)
    return reg


def _main(tmp_path, pre, warm_sel, arm, reg, *extra, dry=False):
    ca = F.base_arm(arm)
    fl = (["--freeze-item-tables"] if F.is_frozen_arm(arm) else []) + (["--epsilon", "8"] if ca == "DP" else [])
    return _args(tmp_path, pre, arm, "--warm-selection", str(reg), "--fl-tuning", str(warm_sel["s4_file"]), *fl,
                 *extra, *(["--dry-run"] if dry else []), smoke=False, seed="2027" if ca != "S_CAL" else "2026")


WIN8 = {(1.0, 1.0): 0.100, (1.0, 0.1): 0.100, (4.0, 1.0): 0.100, (4.0, 0.1): 0.105}


def test_main_warm_line_takes_both_factors_from_its_variant(warm_sel, pre, tmp_path, monkeypatch):
    if warm_sel["w4"]["optimizer"] != "fedadam":
        pytest.skip("the tiny S4 winner is not FedAdam")                        # (it is: the fixture's ladder)
    ident = pre["ident"]
    r7 = _reg(tmp_path, monkeypatch, warm_sel, "S7", WIN_025_01)
    r8 = _reg(tmp_path, monkeypatch, warm_sel, "S8", WIN8)
    mb = _bind_main(monkeypatch, warm_sel)
    eta = warm_sel["w4"]["server_lr"]
    for arm in S7_LINE + S8_LINE:
        frozen = F.is_frozen_arm(arm)
        reg, want = (r8, (4.0, 0.1)) if frozen else (r7, (0.25, 0.1))
        extra = []
        if F.base_arm(arm) in ("FA_1024", "DP"):
            rec = _warm_record(tmp_path / "recs" / f"{arm}.json", _scal_for(arm, warm_sel["w4"], ident, *want), ident)
            extra = ["--clip-record", str(rec)]
        args = _main(tmp_path, pre, warm_sel, arm, reg, *extra)
        F.fl05_preflight(args)
        spec, ex = F.resolve_spec(args, PEAK, fused=True, init_identity=ident)
        assert (spec["warm_lr_factor"], spec["warm_server_lr_factor"]) == want, arm
        assert spec["cfg"]["server_opt"]["lr"] == eta / 10 and ex["warm_selection"]["sha256"] == _sha(reg)
        assert ex["warm_selection"]["stage"] == ("S8" if frozen else "S7") and ex["main_selection"] == mb
        assert spec["init_from"] == ident and (spec.get("frozen_item_tables") is not None) == frozen
        dspec, dex = F.resolve_spec(_main(tmp_path, pre, warm_sel, arm, reg, *extra, dry=True), PEAK, fused=True,
                                    init_identity=ident)
        assert F.spec_digest(dspec) == F.spec_digest(spec) and dex["warm_selection"]["dry_run"]
        other = r7 if frozen else r8                                            # the other variant's file
        assert_raises(F.LaunchRefused, F.resolve_spec, _main(tmp_path, pre, warm_sel, arm, other, *extra), PEAK,
                      fused=True, init_identity=ident)
        assert_raises(F.LaunchRefused, F.resolve_spec, args, PEAK, fused=True,   # another theta_pre
                      init_identity=dict(ident, checkpoint_sha256="1" * 64))
        assert_raises(F.LaunchRefused, F.resolve_spec, args, PEAK, fused=True, init_identity=None)
    rec1 = _warm_record(tmp_path / "recs" / "f1.json", _scal_for("DP_WARM", warm_sel["w4"], ident), ident)
    with pytest.raises(F.LaunchRefused, match="config digest"):                 # S calibrated at factor 1: refused
        F.resolve_spec(_main(tmp_path, pre, warm_sel, "DP_WARM", r7, "--clip-record", str(rec1)), PEAK, fused=True,
                       init_identity=ident)


# ================================================================================================ negative controls
@nc("the pre-FL-16 S loader (format-only cold check) lets DP_WARM run with the cold contracts/dp/S_RECORD.json")
def test_nc_cold_s_record_for_warm_dp(warm_sel, pre, tmp_path, monkeypatch):
    monkeypatch.setattr(F, "load_warm_s_record", lambda path, **kw: F.load_s_record(path, smoke=kw["smoke"]))  # MUTANT
    cold = _cold_record(tmp_path / "cold.json")
    args = _args(tmp_path, pre, "DP_WARM", "--fl-tuning", str(warm_sel["s4_file"]), "--clip-record", str(cold),
                 "--epsilon", "8")
    assert_raises(F.LaunchRefused, F.resolve_spec, args, PEAK, fused=True, init_identity=pre["ident"])


@nc("a warm preflight whose --warm-selection gate covers FA_WARM / FA_WARM_FROZEN only admits a MAIN FP_WARM run "
    "without the registered selection")
def test_nc_warm_line_main_without_selection(pre, tmp_path, monkeypatch):
    real = F.warm_preflight

    def mutant(args):                                                              # MUTANT (pre-FL-16 gate)
        if args.arm in F.WARM_ARMS:
            return real(args)
        if args.init_from is None or bool(args.freeze_item_tables) != F.is_frozen_arm(args.arm):
            raise F.LaunchRefused("init / freeze")
    monkeypatch.setattr(F, "warm_preflight", mutant)
    assert_raises(F.LaunchRefused, F.fl05_preflight, _args(tmp_path, pre, "FP_WARM", smoke=False))


@nc("a launcher that knows only FA_WARM / FA_WARM_FROZEN as warm arms configures PF_WARM without theta_pre (a cold "
    "start under a warm label)")
def test_nc_warm_line_without_theta_pre(pre, tmp_path, monkeypatch):
    monkeypatch.setattr(F, "is_warm_arm", lambda arm: arm in F.WARM_ARMS)           # MUTANT
    assert_raises(F.LaunchRefused, F.resolve_spec, _args(tmp_path, pre, "PF_WARM", init=False), PEAK, fused=True,
                  init_identity=None)


@nc("a warm S record of another theta_pre binds when its init check is removed")
def test_nc_s_record_other_theta_pre(s_chain, pre, monkeypatch):
    other = dict(pre["ident"], checkpoint_sha256="1" * 64)
    monkeypatch.setattr(F, "check_s_record_init", lambda obj, init_identity, path: None)   # MUTANT
    assert_raises(F.LaunchRefused, F.load_warm_s_record, s_chain["S_CAL_WARM"]["dir"] / "S_RECORD_WARM.json",
                  arm="DP_WARM", scal_spec=s_chain["S_CAL_WARM"]["spec"], init_identity=other, smoke=False)


@nc("a warm S record calibrated at another configuration (warm factor 1) binds a factor-0.25 run when the config "
    "binding is removed")
def test_nc_s_record_other_config(warm_sel, pre, tmp_path, monkeypatch):
    ident = pre["ident"]
    rec = _warm_record(tmp_path / "rec.json", _scal_for("DP_WARM", warm_sel["w4"], ident), ident)
    monkeypatch.setattr(F, "check_s_record_config", lambda obj, scal_spec, path: None)    # MUTANT
    args = _args(tmp_path, pre, "DP_WARM", "--fl-tuning", str(warm_sel["s4_file"]), "--clip-record", str(rec),
                 "--epsilon", "8", "--warm-lr-factor", "0.25")
    assert_raises(F.LaunchRefused, F.resolve_spec, args, PEAK, fused=True, init_identity=ident)


@nc("a *_WARM_FROZEN arm whose freeze binding is the pre-FL-16 FA_WARM_FROZEN-only rule clips / noises the item "
    "tables too")
def test_nc_frozen_dp_clips_all_dims(pre, tmp_path, monkeypatch):
    monkeypatch.setattr(F, "is_frozen_arm", lambda arm: arm == "FA_WARM_FROZEN")      # MUTANT
    seen = _spy_dp(monkeypatch)
    w = warm_world(pre)
    run_tiny(tmp_path, wtiny("DP_WARM_FROZEN", pre["ident"]), "dpf", world=w)
    assert all(not set(F.FROZEN_ITEM_KEYS) & set(k) for k in seen["clip"] | seen["noise"])


@nc("a z > 0 warm run (DP_WARM_FROZEN) writes TRAIN_LOSS.jsonl when the DP gate is removed")
def test_nc_dp_warm_train_loss(pre, tmp_path, monkeypatch):
    monkeypatch.setattr(F, "train_loss_applies", lambda spec: True)                  # MUTANT
    run_tiny(tmp_path, wtiny("DP_WARM_FROZEN", pre["ident"]), "dpf", world=warm_world(pre))
    assert not (tmp_path / "dpf" / F.TRAIN_LOSS_FILE).exists()
