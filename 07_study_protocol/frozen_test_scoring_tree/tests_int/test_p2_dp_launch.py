"""P2-08a (PREG2 §5.5; Fable gate FABLE_GATE_ENDGAME_SCRIPTS_20261003 §9.2 E-O4d; PHASE2_IMPLEMENTATION_PLAN P2-08a, Part A)
— the W* DP-FedAvg chain in v4int.launch_fl: P2_WSTAR_S_CAL -> S_RECORD_P2.json -> P2_WSTAR_FA1024 / _DP8 / _DP1 (CPU;
tiny TinyRec warm worlds of test_fl12_launch and a tiny SASREC_D64_B2 world for a LoRA W*; no data, no GPU).

Proves: SPECS — the registered machinery (Poisson m 1,024, T 384, the registered z 0.5642 / 1.3094, FA1024 z 0, S_CAL
20 rounds at seed 2026, passes 2 / batch 16 for FA1024 and DP alike, W*'s server / client recipe and structure); FA1024
and DP8 specs differ only in z (+ the arm label / eps record); at the incumbent recipe P2_WSTAR_DP8's engine config IS
Phase-1 DP_WARM_FROZEN's at the S8 cell (same fl RunConfig digest: K12w comparability); REFUSALS — a z or T other than
the registered one, passes != 2 / batch != 16, another m, S_CAL at another seed, a bad S, no S-record sha, a WB3 / WC5 W*
(O1), every Phase-1 option / --epsilon / --freeze-item-tables / CALIBRATION / a non-smoke run without the registered
selection; S_RECORD_P2 — written by P2_WSTAR_S_CAL only (format PPSI_V4_DP_S_RECORD_P2_1, self-sha, the S_CAL arm-spec
sha / RunConfig digest, theta_pre, W*, selection; no weights), checked like check_s_content, refused by the Phase-1
loaders; the binding refuses the cold / warm records, a smoke record for a real run, tampered content, another theta_pre,
another W*, another selection, another config; the chain END TO END on tiny worlds (FA1024: CLIP_LOG + TRAIN_LOSS +
n_clients; DP8: none of them, UPDATE_NORM_LOG post-noise only; rows labelled; the S plan = the run's seed-2026 Poisson
plan); resolve_spec from a recipe file (smoke) and from the bound FINAL selection (by sha; real load_z); a LoRA W*
(DP over the LoRA keys; the whole base excluded from the noise dimension).
NCs (strict xfail): a DP8 spec at a non-registered z; the Phase-1 cold S record admitted for the chain.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

import v4int.launch_fl as F
from intkit import assert_raises, nc
from test_p2_launch import FAKE_IDENT, PEAK, _args, _write_recipe, pre  # noqa: F401  (pre = module fixture)
from test_p2_wire import sasrec_world

INC = F.check_p2_recipe(dict(server_opt="fedadam", server_lr=1e-3, client_lr=0.1))     # W* = the incumbent (frozen)
SHA = "a" * 64
S4 = dict(optimizer="fedadam", server_lr=0.01, lr_shape="fl_const", client_opt="sgd_m", client_lr=0.1, local_passes=1)


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = json.dumps(v, sort_keys=True)
    return out


def chain(rec=INC, ident=FAKE_IDENT, S=0.31, sel=None, seed=2026):
    kw = dict(recipe=rec, seed=seed, peak_lr=PEAK, init_from=ident, selection=sel, clip_norm=S, s_record_sha256=SHA)
    return {"S_CAL": F.p2_s_cal_spec(rec, peak_lr=PEAK, init_from=ident, selection=sel),
            "FA1024": F.p2_dp_arm_spec("P2_WSTAR_FA1024", **kw),
            "DP8": F.p2_dp_arm_spec("P2_WSTAR_DP8", z=0.5642, **kw),
            "DP1": F.p2_dp_arm_spec("P2_WSTAR_DP1", z=1.3094, **kw)}


# ================================================================================================ specs
def test_dp_chain_specs_are_the_registered_machinery():
    assert (F.DP_ACCOUNTING["m"], F.DP_ACCOUNTING["T"]) == (F.KNOBS["m"], F.KNOBS["rounds_1024"]) == (1024, 384)
    assert F.P2_DP_ARMS == ("P2_WSTAR_S_CAL", "P2_WSTAR_FA1024", "P2_WSTAR_DP8", "P2_WSTAR_DP1")
    assert not set(F.P2_DP_ARMS) & set(F.ALL_ARMS + F.P2_ARMS) and F.P2_ARMS == ("P2_W",)
    c = chain()
    for name, s in c.items():
        F.check_p2_dp_spec(s)
        cfg, sol = s["cfg"], s["cfg"]["solver"]
        assert s["plan"] == {"group_size": 1024, "sampling": "poisson"} and cfg["endpoint_rounds"] == 384
        assert s["max_rounds"] == (20 if name == "S_CAL" else 384) and cfg["dropout_p"] == 0.0 and s["role"] == "MAIN"
        assert sol == {"lr": 0.1, "batch_size": 16, "passes": 2, "clip": 1.0, "weight_decay": 0.0, "fused": None,
                       "optimizer": "sgd_m"} and cfg["lr_shape"] == "fl_const" and cfg["method"] == "FA"
        assert cfg["server_opt"] == {"name": "fedadam", "lr": 0.001, "beta1": 0.9, "beta2": 0.99, "tau": 1e-3,
                                     "v_init": "tau_squared"}
        assert s["frozen_item_tables"] == list(F.FROZEN_ITEM_KEYS) and s["p2_label"] == F.p2_label(INC)
        assert F.update_norm_applies(s) and F.METHOD_CODE[s["arm"]] == "FA"
    assert c["S_CAL"]["cfg"]["dp"] == {"clip_norm": None, "noise_multiplier": 0.0, "denominator": 1024}
    assert c["S_CAL"]["cfg"]["seed"] == 2026 and "p2_s_record_sha256" not in c["S_CAL"]
    assert c["FA1024"]["cfg"]["dp"] == {"clip_norm": 0.31, "noise_multiplier": 0.0, "denominator": 1024}
    assert c["DP8"]["cfg"]["dp"]["noise_multiplier"] == 0.5642 and c["DP1"]["cfg"]["dp"]["noise_multiplier"] == 1.3094
    fa, d8, d1 = (_flat(c[k]) for k in ("FA1024", "DP8", "DP1"))
    diff = sorted(k for k in set(fa) | set(d8) if fa.get(k) != d8.get(k))
    assert diff == ["arm", "cfg.dp.noise_multiplier", "p2_dp.epsilon", "p2_dp.noise_multiplier", "p2_dp.semantics"]
    assert sorted(k for k in set(d8) | set(d1) if d8.get(k) != d1.get(k)) == [
        "arm", "cfg.dp.noise_multiplier", "p2_dp.epsilon", "p2_dp.noise_multiplier"]
    assert F.fl_run_config(c["FA1024"], "x", 64).digest() != F.fl_run_config(c["DP8"], "x", 64).digest()
    # K12w comparability: at the incumbent recipe the DP8 engine config IS Phase-1 DP_WARM_FROZEN's (S8 cell, DL 219)
    eff, prule, _ = F.poisson_tuning("DP_WARM_FROZEN", F.tuning_config(**S4))
    p1 = F.arm_spec("DP_WARM_FROZEN", seed=2026, peak_lr=PEAK, fused=True, init_from=FAKE_IDENT, clip_norm=0.31,
                    z=0.5642, warm_server_lr_factor=0.1, poisson_local_rule=prule, **F.tuning_kwargs(eff))
    assert p1["cfg"] == c["DP8"]["cfg"] and p1["plan"] == c["DP8"]["plan"] and p1["max_rounds"] == 384
    assert F.fl_run_config(p1, "x", 64).digest() == F.fl_run_config(c["DP8"], "x", 64).digest()
    lab = F.p2_dp_run_labels(c["DP8"])
    assert lab["fl_local_passes"] == 2 and lab["fl_local_batch"] == 16 and lab["p2_stage"] == "WD"
    assert lab["fl_dp_epsilon"] == 8.0 and lab["fl_dp_noise_multiplier"] == 0.5642 and lab["fl_dp_T"] == 384
    assert lab["study"] == "PREG2" and lab["row_status"] == "P2_ARM" and lab["p2_recipe_label"] == F.p2_label(INC)
    wb1 = F.check_p2_recipe(dict(INC, items="unfrozen", item_lr_mult=0.01))
    s = F.p2_dp_arm_spec("P2_WSTAR_DP8", recipe=wb1, seed=2027, peak_lr=PEAK, init_from=FAKE_IDENT, clip_norm=0.2,
                         z=0.5642, s_record_sha256=SHA)
    assert s["cfg"]["solver"]["item_lr_mult"] == 0.01 and "frozen_item_tables" not in s
    wc = F.check_p2_recipe(dict(INC, lora_rank=4, lora_lr_factor=10.0))
    s = F.p2_dp_arm_spec("P2_WSTAR_FA1024", recipe=wc, seed=2027, peak_lr=PEAK, init_from=FAKE_IDENT, clip_norm=0.2,
                         s_record_sha256=SHA)
    assert s["lora"]["seed"] == 2027 and s["cfg"]["solver"]["lr"] == 1.0 and "frozen_item_tables" not in s


def test_dp_chain_spec_refusals():
    c = chain()
    kw = dict(recipe=INC, seed=2026, peak_lr=PEAK, init_from=FAKE_IDENT, clip_norm=0.31, s_record_sha256=SHA)
    for arm, z in (("P2_WSTAR_DP8", 0.6), ("P2_WSTAR_DP8", 1.3094), ("P2_WSTAR_DP8", None), ("P2_WSTAR_DP1", 0.5642),
                   ("P2_WSTAR_FA1024", 0.5), ("P2_WSTAR_DP8", True)):
        assert_raises(F.LaunchRefused, F.p2_dp_arm_spec, arm, z=z, **kw)            # only the registered z
    for S in (None, 0.0, -1.0, float("nan"), float("inf"), True):
        assert_raises(F.LaunchRefused, F.p2_dp_arm_spec, "P2_WSTAR_DP8", **dict(kw, clip_norm=S, z=0.5642))
    for bad in (None, "x", "A" * 64):
        assert_raises(F.LaunchRefused, F.p2_dp_arm_spec, "P2_WSTAR_FA1024", **dict(kw, s_record_sha256=bad))
    assert_raises(F.LaunchRefused, F.p2_dp_arm_spec, "P2_WSTAR_S_CAL", recipe=INC, seed=2027, peak_lr=PEAK,
                  init_from=FAKE_IDENT)                                             # S_CAL at 2026 only
    assert_raises(F.LaunchRefused, F.p2_dp_arm_spec, "P2_WSTAR_S_CAL", recipe=INC, seed=2026, peak_lr=PEAK,
                  init_from=FAKE_IDENT, clip_norm=0.3)
    assert_raises(F.LaunchRefused, F.p2_dp_arm_spec, "P2_W", **kw)
    assert_raises(F.LaunchRefused, F.p2_dp_arm_spec, "P2_WSTAR_FA1024", **dict(kw, init_from=None))
    assert_raises(F.LaunchRefused, F.p2_dp_arm_spec, "P2_WSTAR_FA1024",
                  **dict(kw, selection={"sha256": "1" * 64, "self_sha256": "2" * 64, "label": "other"}))
    for o1 in (dict(INC, items="unfrozen", item_lr_mult=0.1, upload="delta_rows"), dict(INC, personal_head=True)):
        r = F.check_p2_recipe(o1)
        assert_raises(F.LaunchRefused, F.p2_dp_arm_spec, "P2_WSTAR_FA1024", **dict(kw, recipe=r))   # O1 (WB3 / WC5)
        assert_raises(F.LaunchRefused, F.p2_s_cal_spec, r, peak_lr=PEAK, init_from=FAKE_IDENT)
    assert_raises(F.LaunchRefused, F.arm_spec, "P2_WSTAR_DP8", seed=2026, peak_lr=PEAK)
    s = c["DP8"]
    tampered = [dict(s, max_rounds=400), dict(s, cfg=dict(s["cfg"], endpoint_rounds=400)),            # T
                dict(s, cfg=dict(s["cfg"], solver=dict(s["cfg"]["solver"], passes=1))),               # passes
                dict(s, cfg=dict(s["cfg"], solver=dict(s["cfg"]["solver"], batch_size=8))),           # batch
                dict(s, plan={"group_size": 512, "sampling": "poisson"}),                             # m
                dict(s, cfg=dict(s["cfg"], dp=dict(s["cfg"]["dp"], denominator=512))),
                dict(s, cfg=dict(s["cfg"], dp=dict(s["cfg"]["dp"], noise_multiplier=0.7))),          # z
                dict(s, cfg=dict(s["cfg"], dropout_p=0.1)), dict(s, p2_dp=dict(s["p2_dp"], epsilon=1.0)),
                dict(s, p2_label="x"), dict(s, frozen_item_tables=None), dict(s, arm="P2_WSTAR_DP1"),
                {k: v for k, v in s.items() if k != "p2_s_record_sha256"}, dict(s, cfg=dict(s["cfg"], seed="x"))]
    for t in tampered:
        assert_raises(F.LaunchRefused, F.check_p2_dp_spec, t)
    assert_raises(F.LaunchRefused, F.check_p2_dp_spec, dict(c["S_CAL"], max_rounds=21))


# ================================================================================================ preflight / run ids
def _dp_args(tmp_path, pre, arm, *extra, smoke=True, seed="2026", init=True):
    return _args(tmp_path, pre, arm, *extra, smoke=smoke, seed=seed, init=init)


def test_dp_chain_preflight_admissions_refusals_and_run_ids(monkeypatch, tmp_path, pre):
    f = _write_recipe(tmp_path / "W.json", INC)
    rec = ["--p2-recipe", str(f)]
    sfile = ["--clip-record", str(tmp_path / "S_RECORD_P2.json")]
    F.fl05_preflight(_dp_args(tmp_path, pre, "P2_WSTAR_S_CAL", *rec))
    F.preflight(_dp_args(tmp_path, pre, "P2_WSTAR_S_CAL", *rec, "--dry-run", "--smoke-rounds", "50"))
    assert_raises(F.LaunchRefused, F.preflight, _dp_args(tmp_path, pre, "P2_WSTAR_S_CAL", *rec, "--dry-run",
                                                         "--smoke-rounds", "51"))   # the Phase-1 smoke limit (50)
    for arm in ("P2_WSTAR_FA1024", "P2_WSTAR_DP8", "P2_WSTAR_DP1"):
        F.fl05_preflight(_dp_args(tmp_path, pre, arm, *rec, *sfile))
        F.preflight(_dp_args(tmp_path, pre, arm, *rec, *sfile, "--dry-run"))
        F.fl05_preflight(_dp_args(tmp_path, pre, arm, "--p2-selection", str(F.P2_SELECTION_PATH), *sfile,
                                  smoke=False, seed="2027"))
        assert_raises(F.LaunchRefused, F.fl05_preflight, _dp_args(tmp_path, pre, arm, *rec))      # no --clip-record
        assert_raises(F.LaunchRefused, F.preflight, _dp_args(tmp_path, pre, arm, *rec, *sfile, "--epsilon", "8",
                                                             "--dry-run"))       # the arm fixes eps
        assert_raises(F.LaunchRefused, F.fl05_preflight, _dp_args(tmp_path, pre, arm, *rec, *sfile, smoke=False))
    p1 = {"lr_mult": ["--lr-mult", "2"], "lr_selection": ["--lr-selection", "x.json"], "fl_tuning": ["--fl-tuning", "x"],
          "server_opt": ["--server-opt", "fedadam"], "server_lr": ["--server-lr", "0.01"],
          "server_tau": ["--server-tau", "1e-05"], "lr_shape": ["--lr-shape", "fl_const"],
          "client_opt": ["--client-opt", "sgd_m"], "client_lr": ["--client-lr", "0.1"],
          "local_passes": ["--local-passes", "2"], "local_batch": ["--local-batch", "16"],
          "warm_lr_factor": ["--warm-lr-factor", "4"], "warm_selection": ["--warm-selection", "w.json"],
          "warm_server_lr_factor": ["--warm-server-lr-factor", "0.1"], "epsilon": ["--epsilon", "8"],
          "theta0_variants": ["--theta0-variants", "t.json"]}
    assert sorted(p1) == sorted(F.P2_DP_REFUSED_FLAGS)
    for flag in p1.values():
        assert_raises(F.LaunchRefused, F.fl05_preflight, _dp_args(tmp_path, pre, "P2_WSTAR_DP8", *rec, *sfile, *flag))
    sel = str(F.P2_SELECTION_PATH)
    wb3 = _write_recipe(tmp_path / "WB3.json", F.check_p2_recipe(dict(INC, items="unfrozen", item_lr_mult=0.1,
                                                                      upload="delta_rows")))
    bad = [("P2_WSTAR_S_CAL", rec + sfile, {}), ("P2_WSTAR_S_CAL", rec, {"seed": "2027"}),
           ("P2_WSTAR_DP8", rec + sfile + ["--freeze-item-tables"], {}), ("P2_WSTAR_DP8", rec + sfile, {"init": False}),
           ("P2_WSTAR_DP8", rec + sfile + ["--role", "CALIBRATION"], {}), ("P2_WSTAR_DP8", sfile, {}),
           ("P2_WSTAR_DP8", rec + sfile + ["--p2-selection", sel], {}),
           ("P2_WSTAR_DP8", ["--p2-recipe", str(wb3)] + sfile, {})]                  # WB3 W*: O1
    for arm, extra, kw in bad:
        assert_raises(F.LaunchRefused, F.fl05_preflight, _dp_args(tmp_path, pre, arm, *extra, **kw))
    assert F.default_run_id(_dp_args(tmp_path, pre, "P2_WSTAR_DP8", *rec, *sfile)) == \
        "SMOKE_NOT_EVIDENCE__P2_WSTAR_DP8_FEDADAM_eta0.001_c0.1_SMALL_s2026"
    assert F.default_run_id(_dp_args(tmp_path, pre, "P2_WSTAR_DP1", "--p2-selection", sel, *sfile, smoke=False,
                                     seed="2027")) == "P2_WSTAR_DP1_SMALL_s2027"
    for a in ("FA", "DP", "DP_WARM_FROZEN", "P2_W"):                                  # the new arms name no old one
        assert not F.is_p2_dp_arm(a) and F.semantic_arm(a) == F.base_arm(a)


# ================================================================================================ the P2 S record
def _p2_record(path: Path, scal: dict, ident, *, S=0.31, run_id="P2_WSTAR_S_CAL_SMALL_s2026", smoke=False, **over):
    norm = {"S": S, "n_norms": 20, "rounds_0based": list(range(20)), "rule": "median"}
    obj = F.s_record_obj_p2(norm, spec=scal, run_id=run_id, seed=2026,
                            config_digest=F.fl_run_config(scal, run_id, 64).digest(), plan_digest="p" * 64,
                            init_sha256=ident["state_digest"], smoke=smoke)
    if over:
        obj.update(over)
        obj["self_sha256"] = F._canon(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _patch_tiny(monkeypatch):
    monkeypatch.setitem(F.DP_ACCOUNTING, "m", 4)
    monkeypatch.setitem(F.DP_ACCOUNTING, "T", 24)      # S_CAL keeps its registered 20 rounds (fl.dp rule)


def _exec(tmp_path, spec, name, world, **kw):
    return F.execute(spec=spec, run_id=name, run_dir=tmp_path / name, say=lambda *_: None, **world, **kw)


def test_s_record_p2_chain_end_to_end(monkeypatch, tmp_path, pre):
    from test_fl12_launch import warm_world
    _patch_tiny(monkeypatch)
    ident = pre["ident"]
    scal = F.p2_s_cal_spec(INC, peak_lr=PEAK, init_from=ident)
    o = _exec(tmp_path, scal, "scal", warm_world(pre))
    assert o["status"] == "DONE_S_CAL"
    d = tmp_path / "scal"
    rp = d / F.S_RECORD_P2_FILE
    assert rp.is_file() and not (d / "S_RECORD.json").exists() and not list(d.glob("S_RECORD_WARM*.json"))
    assert not list((d / "ckpt").rglob("*.pt"))                                       # S_CAL keeps no weights
    rec = _json(rp)
    assert rec["format"] == F.S_FORMAT_P2 and rec["self_sha256"] == F._canon(rec) and rec["arm"] == "P2_WSTAR_S_CAL"
    assert rec["label"] == "S_CALIBRATION" and rec["seed"] == 2026 and rec["rounds_0based"] == list(range(20))
    assert rec["init_from"] == ident and rec["theta_pre_state_digest"] == rec["init_sha256"] == ident["state_digest"]
    assert rec["arm_spec_sha256"] == F.spec_digest(scal) and rec["config_digest"] == F.fl_run_config(scal, "scal", 64).digest()
    assert rec["p2_label"] == F.p2_label(INC) and rec["p2_selection"] is None and rec["lora"] is None
    assert rec["frozen_item_tables"] == list(F.FROZEN_ITEM_KEYS)
    assert o["summary"]["s_record"] == {"S": rec["S"], "self_sha256": rec["self_sha256"], "n_norms": rec["n_norms"],
                                        "file": F.S_RECORD_P2_FILE, "format": F.S_FORMAT_P2}
    F.check_s_content(rec, rp)
    assert F.load_p2_s_record(rp, scal_spec=scal, init_identity=ident, smoke=False)["S"] == rec["S"]
    assert_raises(F.LaunchRefused, F.load_s_record, rp, smoke=False)                  # the Phase-1 loaders refuse it
    assert_raises(F.LaunchRefused, F.load_warm_s_record, rp, arm="DP_WARM_FROZEN", scal_spec=scal, init_identity=ident,
                  smoke=False)
    # the binding's refusals
    other = F.check_p2_recipe(dict(INC, server_lr=1e-4))
    bads = [(_p2_record(tmp_path / "b" / "smoke.json", scal, ident, smoke=True), {}),           # smoke for a real run
            (_p2_record(tmp_path / "b" / "neg.json", scal, ident, S=-1.0), {}),                 # content (S rule)
            (_p2_record(tmp_path / "b" / "rounds.json", scal, ident, rounds_0based=list(range(19))), {}),
            (_p2_record(tmp_path / "b" / "rule.json", scal, ident, rule="mean"), {}),
            (_p2_record(tmp_path / "b" / "arm.json", scal, ident, arm="S_CAL_WARM_FROZEN"), {}),
            (_p2_record(tmp_path / "b" / "init.json", scal, ident, init_from=dict(ident, checkpoint_sha256="1" * 64)), {}),
            (_p2_record(tmp_path / "b" / "tdig.json", scal, ident, theta_pre_state_digest="e" * 64), {}),
            (_p2_record(tmp_path / "b" / "isha.json", scal, ident, init_sha256="e" * 64), {}),
            (_p2_record(tmp_path / "b" / "cfg.json", scal, ident, config_digest="0" * 64), {}),
            (_p2_record(tmp_path / "b" / "frz.json", scal, ident, frozen_item_tables=None), {}),
            (_p2_record(tmp_path / "b" / "ok.json", scal, ident),
             {"scal_spec": F.p2_s_cal_spec(other, peak_lr=PEAK, init_from=ident)}),             # another W*
            (_p2_record(tmp_path / "b" / "ok2.json", scal, ident),
             {"scal_spec": F.p2_s_cal_spec(INC, peak_lr=PEAK, init_from=ident, selection={
                 "sha256": "1" * 64, "self_sha256": "2" * 64, "label": F.p2_label(INC)})})]     # another selection
    for p, kw in bads:
        assert_raises(F.LaunchRefused, F.load_p2_s_record, p, **dict(dict(scal_spec=scal, init_identity=ident,
                                                                          smoke=False), **kw))
    assert F.load_p2_s_record(bads[0][0], scal_spec=scal, init_identity=ident, smoke=True)["label"] == "SMOKE_NOT_EVIDENCE"
    tam = _json(rp)
    tam["S"] = 9.0                                                                    # not re-signed
    (tmp_path / "b" / "tam.json").write_text(json.dumps(tam), encoding="utf-8")
    cold = F.s_record_obj({"S": 0.3, "n_norms": 20, "rounds_0based": list(range(6)), "rule": "median"}, run_id="x",
                          seed=2026, config_digest="c", plan_digest="p", init_sha256=ident["state_digest"], smoke=False)
    (tmp_path / "b" / "cold.json").write_text(json.dumps(cold), encoding="utf-8")
    for name in ("tam.json", "cold.json"):
        assert_raises(F.LaunchRefused, F.load_p2_s_record, tmp_path / "b" / name, scal_spec=scal, init_identity=ident,
                      smoke=True)
    # FA1024 and DP8 from the record
    w = warm_world(pre)
    assert F.check_warm_s_plan(rec, w["keys"], w["manifest_hash"], scal) == rec["plan_digest"]
    S, ssha = rec["S"], rec["self_sha256"]
    fa = F.p2_dp_arm_spec("P2_WSTAR_FA1024", recipe=INC, seed=2026, peak_lr=PEAK, init_from=ident, clip_norm=S,
                          s_record_sha256=ssha)
    dp8 = F.p2_dp_arm_spec("P2_WSTAR_DP8", recipe=INC, seed=2027, peak_lr=PEAK, init_from=ident, clip_norm=S,
                           z=0.5642, s_record_sha256=ssha)
    assert F.check_warm_s_plan(rec, w["keys"], w["manifest_hash"], dp8) == rec["plan_digest"]   # = run()'s plan check
    of, od = _exec(tmp_path, fa, "fa", warm_world(pre)), _exec(tmp_path, dp8, "dp8", warm_world(pre))
    assert of["status"] == od["status"] == "DONE_MAIN"
    df, dd = tmp_path / "fa", tmp_path / "dp8"
    assert (df / F.CLIP_LOG_FILE).is_file() and (df / F.TRAIN_LOSS_FILE).is_file()
    assert not (dd / F.CLIP_LOG_FILE).exists() and not (dd / F.TRAIN_LOSS_FILE).exists()      # 06L_E E1 (z > 0)
    lf = [json.loads(x) for x in (df / F.UPDATE_NORM_FILE).read_text(encoding="utf-8").splitlines()]
    ld = [json.loads(x) for x in (dd / F.UPDATE_NORM_FILE).read_text(encoding="utf-8").splitlines()]
    assert [x["round"] for x in ld] == list(range(24)) and od["summary"]["update_norm_log"]["complete"]
    assert any("n_clients" in x for x in lf) and not any("n_clients" in x for x in ld)       # post-noise only
    for o, s in ((of, fa), (od, dp8)):
        sm = o["summary"]
        assert sm["p2"]["dp_chain"] == s["p2_dp"] and sm["p2"]["label"] == F.p2_label(INC)
        assert sm["dp_noise_dimension"]["frozen_keys_excluded"] == list(F.FROZEN_ITEM_KEYS)
        assert not set(F.FROZEN_ITEM_KEYS) & set(sm["dp_noise_dimension"]["uploaded_keys"])
    rows = _json(dd / "RESULT_ROWS.json")["rows"]
    assert rows and all(r["method"] == "P2_WSTAR_DP8" and r["method_code"] == "FA" and r["study"] == "PREG2" and
                        r["row_status"] == "P2_ARM" and r["p2_stage"] == "WD" and r["fl_dp_epsilon"] == 8.0 and
                        r["fl_local_passes"] == 2 and r["fl_local_batch"] == 16 and
                        r["p2_recipe_label"] == F.p2_label(INC) and r["fl_frozen_item_tables"] for r in rows)
    assert of["summary"]["server_digest"] != od["summary"]["server_digest"]
    # resolve_spec from a recipe file (smoke) binds this S record (FA1024: no z record needed in the tiny world)
    f = _write_recipe(tmp_path / "W.json", INC)
    args = _dp_args(tmp_path, pre, "P2_WSTAR_FA1024", "--p2-recipe", str(f), "--clip-record", str(rp))
    spec, extra = F.resolve_spec(args, PEAK, True, ident)
    assert spec == fa and extra["s_record"]["format"] == F.S_FORMAT_P2 and extra["s_record"]["S"] == S
    assert extra["s_record"]["plan_digest"] == rec["plan_digest"] and extra["p2_recipe"]["label"] == F.p2_label(INC)


# ================================================================================================ MAIN binding
def _selection_obj(rec, ident):
    obj = {"format": F.P2_SELECTION_FORMAT, "stage": "FINAL", "winner": F.p2_label(rec), "winner_recipe": dict(rec),
           "common": {"init_from": dict(ident)}, "flags": []}
    obj["self_sha256"] = F._canon(obj)
    return obj


def test_main_chain_binds_the_registered_selection_and_real_z(monkeypatch, tmp_path, pre):
    reg = tmp_path / "contracts" / "fl" / "P2_WARM_SELECTION_FINAL.json"
    reg.parent.mkdir(parents=True)
    monkeypatch.setattr(F, "P2_SELECTION_PATH", reg)
    stub = types.ModuleType("v4int.p2_select")
    stub.load_p2_selection = lambda path, stage: {}
    monkeypatch.setitem(sys.modules, "v4int.p2_select", stub)
    ident = pre["ident"]
    reg.write_text(json.dumps(_selection_obj(INC, ident)), encoding="utf-8")
    fsha = hashlib.sha256(reg.read_bytes()).hexdigest()
    a = _dp_args(tmp_path, pre, "P2_WSTAR_S_CAL", "--p2-selection", str(reg), smoke=False)
    scal, extra = F.resolve_spec(a, PEAK, True, ident)
    want_sel = {"sha256": fsha, "self_sha256": _json(reg)["self_sha256"], "label": F.p2_label(INC)}
    assert scal["p2_selection"] == want_sel and extra["p2_selection"]["sha256"] == fsha
    assert F.default_run_id(a) == "P2_WSTAR_S_CAL_SMALL_s2026"
    rp = _p2_record(tmp_path / "S" / "S_RECORD_P2.json", scal, ident)
    a8 = _dp_args(tmp_path, pre, "P2_WSTAR_DP8", "--p2-selection", str(reg), "--clip-record", str(rp), smoke=False,
                  seed="2027")
    s8, e8 = F.resolve_spec(a8, PEAK, True, ident)                                    # the REAL z record (load_z)
    assert s8["cfg"]["dp"] == {"clip_norm": 0.31, "noise_multiplier": 0.5642, "denominator": 1024}
    assert e8["dp_z"]["z"] == 0.5642 and e8["dp_z"]["T"] == 384 and e8["dp_z"]["m"] == 1024
    assert s8["p2_selection"] == want_sel and s8["p2_s_record_sha256"] == _json(rp)["self_sha256"]
    F.check_p2_dp_spec(s8)
    a1 = _dp_args(tmp_path, pre, "P2_WSTAR_DP1", "--p2-selection", str(reg), "--clip-record", str(rp), smoke=False)
    assert F.resolve_spec(a1, PEAK, True, ident)[0]["cfg"]["dp"]["noise_multiplier"] == 1.3094
    smoke_scal = F.p2_s_cal_spec(INC, peak_lr=PEAK, init_from=ident)                  # selection None (recipe file)
    for p in (_p2_record(tmp_path / "S" / "unbound.json", smoke_scal, ident),
              _p2_record(tmp_path / "S" / "smoke.json", scal, ident, smoke=True)):
        bad = _dp_args(tmp_path, pre, "P2_WSTAR_DP8", "--p2-selection", str(reg), "--clip-record", str(p), smoke=False)
        assert_raises(F.LaunchRefused, F.resolve_spec, bad, PEAK, True, ident)
    for o1 in (dict(INC, items="unfrozen", item_lr_mult=0.1, upload="delta_rows"), dict(INC, personal_head=True)):
        reg.write_text(json.dumps(_selection_obj(F.check_p2_recipe(o1), ident)), encoding="utf-8")
        assert_raises(F.LaunchRefused, F.resolve_spec, a, PEAK, True, ident)          # O1: WB3 / WC5 W* refused


def test_dp_chain_dry_runs_through_the_cli_path(monkeypatch, tmp_path, pre):
    from test_p2_launch import FAKE_RECIPE
    monkeypatch.setattr(F, "_small_recipe", lambda _a: dict(FAKE_RECIPE))
    f = _write_recipe(tmp_path / "W.json", INC)
    out = []
    a = _dp_args(tmp_path, pre, "P2_WSTAR_S_CAL", "--p2-recipe", str(f), "--dry-run", smoke=False)
    F.preflight(a)
    assert F.dry_run(a, say=out.append) == 0
    d = json.loads(out[-1])
    scal = F.p2_s_cal_spec(INC, peak_lr=PEAK, init_from=pre["ident"])
    assert d["arm_spec"] == json.loads(json.dumps(scal)) and d["arm_spec_sha256"] == F.spec_digest(scal)
    rid = F.default_run_id(a)
    assert rid == "P2_WSTAR_S_CAL_FEDADAM_eta0.001_c0.1_SMALL_s2026"
    assert d["fl_config_digest"] == F.fl_run_config(scal, rid, 64).digest()
    rp = _p2_record(tmp_path / "S_RECORD_P2.json", scal, pre["ident"], run_id=rid)
    a8 = _dp_args(tmp_path, pre, "P2_WSTAR_DP8", "--p2-recipe", str(f), "--clip-record", str(rp), "--dry-run",
                  smoke=False, seed="2027")
    F.preflight(a8)
    assert F.dry_run(a8, say=out.append) == 0
    d8 = json.loads(out[-1])
    assert d8["arm_spec"]["cfg"]["dp"] == {"clip_norm": 0.31, "noise_multiplier": 0.5642, "denominator": 1024}
    assert d8["dp_z"]["z"] == 0.5642 and d8["s_record"]["format"] == F.S_FORMAT_P2 and d8["arm_spec"]["max_rounds"] == 384
    assert F.main(["--arm", "P2_WSTAR_DP8", "--seed", "2026", "--gpu", "0", "--run-dir", str(tmp_path / "r"),
                   "--history-policy", "USER_HISTORY_L50", "--init-from", str(pre["path"]), "--p2-recipe", str(f),
                   "--clip-record", str(rp), "--epsilon", "8", "--dry-run"]) == 2          # --epsilon: refused (rc 2)


# ================================================================================================ LoRA W*
def test_lora_wstar_dp_chain_noises_the_lora_keys_only(monkeypatch, tmp_path):
    _patch_tiny(monkeypatch)
    w0 = sasrec_world(n=10)
    ident = dict(FAKE_IDENT, state_digest=w0["init_sha256"])
    wc = F.check_p2_recipe(dict(INC, lora_rank=4, lora_lr_factor=10.0))
    scal = F.p2_s_cal_spec(wc, peak_lr=PEAK, init_from=ident)
    o = _exec(tmp_path, scal, "scal", sasrec_world(n=10))
    assert o["status"] == "DONE_S_CAL"
    rp = tmp_path / "scal" / F.S_RECORD_P2_FILE
    rec = F.load_p2_s_record(rp, scal_spec=scal, init_identity=ident, smoke=False)
    assert rec["lora"] == scal["lora"] and rec["init_sha256"] != ident["state_digest"]       # the LoRA-wrapped init
    s8 = F.p2_dp_arm_spec("P2_WSTAR_DP8", recipe=wc, seed=2026, peak_lr=PEAK, init_from=ident, clip_norm=rec["S"],
                          z=0.5642, s_record_sha256=rec["self_sha256"])
    o8 = _exec(tmp_path, s8, "dp8", sasrec_world(n=10))
    assert o8["status"] == "DONE_MAIN"
    nd = o8["summary"]["dp_noise_dimension"]
    base = w0["server_adapter"].manifest.shared_keys
    fz = nd["frozen_keys_excluded"]                                   # the whole base (LoRALinear: base_weight / bias)
    assert nd["uploaded_numel"] == 2048 * 4 and not set(fz) & set(nd["uploaded_keys"])
    assert sorted(k.replace(".base_weight", ".weight").replace(".base_bias", ".bias") for k in fz) == sorted(base)
    assert o8["summary"]["bytes"]["per_visit"]["shared_numel"] == 2048 * 4
    assert sorted(o8["summary"]["update_norm_log"]["trained_keys"]) == sorted(nd["uploaded_keys"])


# ================================================================================================ negative controls
@nc("a DP8 spec at a non-registered z (no new accountant) must be refused")
def test_nc_dp8_at_unregistered_z():
    try:
        F.p2_dp_arm_spec("P2_WSTAR_DP8", recipe=INC, seed=2026, peak_lr=PEAK, init_from=FAKE_IDENT, clip_norm=0.3,
                         z=0.6, s_record_sha256=SHA)
    except F.LaunchRefused:
        raise AssertionError("refused (only the registered z)")


@nc("the Phase-1 cold S record (theta0 trajectory) admitted for the W* chain")
def test_nc_cold_record_for_the_chain(tmp_path):
    scal = F.p2_s_cal_spec(INC, peak_lr=PEAK, init_from=FAKE_IDENT)
    cold = F.s_record_obj({"S": 0.3, "n_norms": 20, "rounds_0based": list(range(20)), "rule": "median"}, run_id="x",
                          seed=2026, config_digest="c", plan_digest="p", init_sha256="d" * 64, smoke=False)
    p = tmp_path / "S_RECORD.json"
    p.write_text(json.dumps(cold), encoding="utf-8")
    try:
        F.load_p2_s_record(p, scal_spec=scal, init_identity=FAKE_IDENT, smoke=False)
    except F.LaunchRefused:
        raise AssertionError("refused (the chain takes its own record only)")
