"""P2-01 / P2-02 (PREG2 §3.3 / §5; PHASE2_IMPLEMENTATION_PLAN; Fable gate §9) — the P2_W arm family in v4int.launch_fl
(CPU, tiny warm worlds of test_fl12_launch; no data, no GPU).

Proves: FIREWALL — a 56-pair (arm spec, RunConfig) table of the Phase-1 arms FA / FP / PF / FA_Q8 / FA_1024 / DP8 /
S_CAL / FA_WARM / FA_WARM_FROZEN / DP_WARM_FROZEN (default and S4 tuning, warm factors, MAIN + CALIBRATION) MEASURED on
the pre-P2 launch_fl (c9fdc8c0) is unchanged, the 21 production CALIB screen arm-spec digests recorded in
contracts/fl/FL_TUNING_S2|S3|SELECTION.json still reproduce, the Phase-1 grids / lists / method codes / smoke limit are
unchanged; DRY RUNS of the 21 WA cells (18 core + 2 ext + incumbent) through launch_fl.dry_run (3,414 CALIB rounds,
FL_CONST, client SGD-m, frozen tables, the server-step mapping, labels = v4int.p2_select's), a WB and a WC5 recipe;
REFUSALS — every P2 flag for every Phase-1 arm, every Phase-1 option for P2_W, MAIN without the registered selection,
bad recipes, a tampered P2 spec; MAIN binding of the registered P2_WARM_SELECTION_FINAL.json (path, sha, format,
self-sha, stage, label, verifier, theta_pre); P2-T3 LOCAL — the incumbent recipe's cfg is S8Tc's cfg (FA_WARM_FROZEN
CALIB at the S4 point, server-LR factor 0.1; same fl RunConfig digest) and a tiny run through the P2 path is bit-identical
to the Phase-1 path (theta digest, TRAIN_LOSS bytes, evaluations, ledger); D1-ext — UPDATE_NORM_LOG.jsonl only for P2,
complete, trimmed on resume, the trained keys in RUN_SUMMARY, a resumed smoke run bitwise equal to a straight one; rows
labelled study PREG2; WB (ItemLRSolver) and WC5 (PF) tiny runs complete.
NCs (strict xfail): the incumbent at the Phase-1 client LR grid would be refused for P2 (P2 grid admitted only for P2);
a Phase-1 arm that accepted --p2-recipe.
"""
from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

import v4int.launch_fl as F
from fl.client import ItemLRSolver
from intkit import assert_raises, nc
from test_fl12_launch import make_pre, warm_world

PEAK = 0.000711
V4 = F.V4
IDENT_KEYS = F.INIT_IDENTITY_KEYS
S4 = dict(optimizer="fedadam", server_lr=0.01, lr_shape="fl_const", client_opt="sgd_m", client_lr=0.1, local_passes=1)
PRE_P2_TABLE_SHA = "0c89ec7759eca30a1354ddd37f6c47842cf8ca003766901c7c127b6216ebcd56"   # 56 pairs, pre-P2 launch_fl
FAKE_RECIPE = {"recipe_id": "SMALL_TEST", "recipe_sha256": "0" * 64, "recipe_registry": "test",
               "recipe_registry_self_sha256": "0" * 64, "production_pin": None, "peak_lr": PEAK, "dropout": [0.1],
               "schedule": "test", "model_variant": "SASREC_D64_B2"}


@pytest.fixture(scope="module")
def pre(tmp_path_factory):
    root = tmp_path_factory.mktemp("p2pre")
    p, adapter = make_pre(root)
    state, ident, launch = F.load_pretrain_init(p, model_variant="SASREC_D64_B2")
    return {"path": p, "adapter": adapter, "state": state, "ident": ident, "launch": launch}


def phase1_table(ident) -> dict:
    """The same 56-pair table as the scratchpad p2_pins.py that MEASURED PRE_P2_TABLE_SHA on the pre-P2 code."""
    arms = {"FA": "FA", "FP": "FP", "PF": "PF", "FA_Q8": "FA_Q8", "FA_1024": "FA_1024", "DP8": "DP", "S_CAL": "S_CAL",
            "FA_WARM": "FA_WARM", "FA_WARM_FROZEN": "FA_WARM_FROZEN", "DP_WARM_FROZEN": "DP_WARM_FROZEN"}
    out = {}
    for label, arm in arms.items():
        for tname, tk in (("default", {}), ("s4", S4)):
            eff, prule, _ = F.poisson_tuning(arm, F.tuning_config(**tk))
            kw = {}
            ca = F.base_arm(arm)
            if ca in ("FA_1024", "DP"):
                kw["clip_norm"] = 0.2843
            if ca == "DP":
                kw["z"] = 0.5642
            cells = [(None, None)]
            if F.is_warm_arm(arm):
                v = F.warm_variant(arm)
                sfs = (None, 0.1) if eff["optimizer"] == "fedadam" else (None, 0.25)
                cells = [(wf, sf) for wf in (None, F.WARM_FACTORS[v][1]) for sf in sfs]
            for wf, sf in cells:
                k2 = dict(kw)
                if F.is_warm_arm(arm):
                    k2.update(init_from=ident, warm_lr_factor=wf, warm_server_lr_factor=sf)
                for role in (("MAIN", "CALIBRATION") if arm in F.CALIB["arms"] else ("MAIN",)):
                    s = F.arm_spec(arm, seed=2026, peak_lr=PEAK, fused=True, poisson_local_rule=prule, role=role,
                                   **k2, **F.tuning_kwargs(eff))
                    out[f"{label}|{tname}|wf={wf}|sf={sf}|{role}"] = [
                        F.spec_digest(s), F.fl_run_config(s, f"FL_{label}_SMALL_s2026", 64).digest()]
    return out


FAKE_IDENT = {"kind": "PRETRAIN_FOR_FL", "run_id": "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026",
              "run_label": "PRETRAIN_FOR_FL", "train_manifest": "PRETRAIN_TRAIN_5_100_NONMAIN",
              "fl_init_kind": "endpoint_6.0", "checkpoint_sha256": "c" * 64, "checkpoint_bytes": 123,
              "state_digest": "d" * 64, "model_variant": "SASREC_D64_B2", "family": "SASREC", "seed": 2026}


def _recipe(**kw):
    return F.check_p2_recipe(dict({"server_opt": "fedadam", "server_lr": 1e-3, "client_lr": 0.1}, **kw))


def _write_recipe(path: Path, rec) -> Path:
    path.write_text(json.dumps(F.p2_recipe_file_obj(rec)), encoding="utf-8")
    return path


def _args(tmp_path, pre, arm, *extra, init=True, smoke=True, seed="2026"):
    a = ["--arm", arm, "--seed", seed, "--gpu", "0", "--run-dir", str(tmp_path / "r"), "--history-policy",
         "USER_HISTORY_L50"] + (["--init-from", str(pre["path"])] if init else []) + list(extra)
    return F.parse_args(a + (["--smoke"] if smoke else []))


# ================================================================================================ firewall
def test_phase1_digests_grids_and_lists_unchanged():
    t = phase1_table(FAKE_IDENT)
    assert len(t) == 56 and hashlib.sha256(json.dumps(t, sort_keys=True).encode()).hexdigest() == PRE_P2_TABLE_SHA
    assert t["FA|default|wf=None|sf=None|MAIN"] == ["fc715be7b951108a55e7f55968d0e9bf2851cb8878d6542b174d90e2b6ae3f88",
                                                    "66770288acf78c816e98bf65c78896acb4d16b41767ef6c7809af1fe13c5c74c"]
    assert F.WARM_FACTORS == {"FA_WARM": (1.0, 0.25), "FA_WARM_FROZEN": (1.0, 4.0)}
    assert F.WARM_SERVER_FACTORS == {"fedadam": (1.0, 0.1), "fedavgm": (1.0, 0.25), "fedavg": (1.0, 0.25)}
    assert F.CLIENT_LRS == (0.01, 0.1, 1.0) and F.CLIENT_EDGE_LRS == (3.0, 10.0, 0.003)
    assert F.LR_MULTS == (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0) and F.FEDADAM_LRS == (0.01, 0.1)
    assert F.FEDADAM_EDGE_LRS == (0.003, 0.001) and F.LOCAL_PASSES == (1, 2, 4) and F.LOCAL_BATCHES == (8, 16)
    assert F.MAX_SMOKE_ROUNDS == 50 and F.ARMS == ("FA", "FP", "PF", "FA_1024", "DP", "S_CAL", "FA_Q8")
    assert not set(F.P2_ARMS) & set(F.ALL_ARMS) and len(F.ALL_ARMS) == 21
    assert {a: F.METHOD_CODE[a] for a in F.ALL_ARMS} == {a: F.METHOD_CODE[F.base_arm(a)] if a in F.WARM_LINE else
                                                         F.METHOD_CODE[a] for a in F.ALL_ARMS}
    assert F.METHOD_CODE["FA"] == "FA" and F.METHOD_CODE["PF"] == "PF" and F.METHOD_CODE["DP"] == "FA"
    assert all(F.update_norm_applies({"arm": a}) is False for a in F.ALL_ARMS)


def test_production_calib_arm_spec_digests_reproduce():
    files = [V4 / "contracts" / "fl" / f for f in ("FL_TUNING_S2.json", "FL_TUNING_S3.json", "FL_TUNING_SELECTION.json")]
    if not all(p.is_file() for p in files):
        pytest.skip("the registered FL tuning files are not present on this host")
    n = 0
    for p in files:
        obj = json.loads(p.read_text(encoding="utf-8"))
        for c in obj["configs"]:
            t = F.tuning_config(**{k: c.get(k) for k in F.TUNING_KEYS})
            s = F.arm_spec("FA", seed=2026, peak_lr=float(obj["lr_peak_base"]), fused=True, role="CALIBRATION",
                           **F.tuning_kwargs(t))
            assert F.spec_digest(s) == c["arm_spec_sha256"], (p.name, c["label"])
            n += 1
    assert n == 21


# ================================================================================================ dry runs
def _dry(monkeypatch, tmp_path, pre, rec, *extra, smoke=False):
    monkeypatch.setattr(F, "_small_recipe", lambda _a: dict(FAKE_RECIPE))
    f = _write_recipe(tmp_path / f"R_{F.p2_run_tag(rec)}.json", rec)
    args = _args(tmp_path, pre, "P2_W", "--p2-recipe", str(f), "--dry-run", *extra, smoke=smoke)
    F.preflight(args)
    out = []
    assert F.dry_run(args, say=out.append) == 0
    return json.loads(out[-1]), args


def test_wa_cells_dry_runs(monkeypatch, tmp_path, pre):
    cells = F.p2_wa_recipes()
    assert len(cells) == 21 and len({F.p2_label(c) for c in cells}) == 21
    ids = set()
    for rec in cells:
        d, args = _dry(monkeypatch, tmp_path, pre, rec, "--role", "CALIBRATION")
        s = d["arm_spec"]
        assert d["arm_spec_sha256"] == F.spec_digest(s) and d["p2_recipe"]["label"] == F.p2_label(rec)
        assert s["arm"] == "P2_W" and s["role"] == "CALIBRATION" and s["p2_recipe"] == rec
        assert s["cfg"]["endpoint_rounds"] == s["max_rounds"] == 3414 and s["cfg"]["lr_shape"] == "fl_const"
        sol = s["cfg"]["solver"]
        assert sol == {"lr": rec["client_lr"], "batch_size": 16, "passes": 1, "clip": 1.0, "weight_decay": 0.0,
                       "fused": None, "optimizer": "sgd_m"} and s["cfg"]["lr_peak_local"] == rec["client_lr"]
        assert s["frozen_item_tables"] == list(F.FROZEN_ITEM_KEYS) and s["init_from"] == pre["ident"]
        so = s["cfg"].get("server_opt")
        if rec["server_opt"] == "fedavg":
            assert so == (None if rec["server_lr"] == 1.0 else {"name": "fedavgm", "lr": rec["server_lr"], "beta1": 0.0})
        elif rec["server_opt"] == "fedavgm":
            assert so == {"name": "fedavgm", "lr": rec["server_lr"], "beta1": 0.9}
        else:
            assert so == {"name": "fedadam", "lr": rec["server_lr"], "beta1": 0.9, "beta2": 0.99, "tau": 1e-3,
                          "v_init": "tau_squared"}
        rid = F.default_run_id(args)
        assert rid.startswith("P2_W_") and rid.endswith("_CALIB_s2026")
        assert d["fl_config_digest"] == F.fl_run_config(s, rid, 64).digest()
        ids.add(rid)
    assert len(ids) == 21
    try:
        import v4int.p2_select as PS                       # P2-06: the selection tool's cells and labels
    except ImportError:
        return
    assert set(PS.wa_cells()) == {F.p2_label(c) for c in cells}
    for lab, (_set, r) in PS.wa_cells().items():
        assert F.p2_label(F.check_p2_recipe(r)) == lab and F.check_p2_recipe(r) == r


def test_main_wb_and_wc5_dry_runs(monkeypatch, tmp_path, pre):
    d, args = _dry(monkeypatch, tmp_path, pre, _recipe(server_opt="fedavgm", server_lr=0.1, client_lr=0.01))
    s = d["arm_spec"]
    assert s["role"] == "MAIN" and s["max_rounds"] == 13654 and F.default_run_id(args).endswith("_SMALL_s2026")
    wb = _recipe(items="unfrozen", item_lr_mult=0.01)
    d, _ = _dry(monkeypatch, tmp_path, pre, wb, "--role", "CALIBRATION")
    s = d["arm_spec"]
    assert "frozen_item_tables" not in s and s["cfg"]["solver"]["item_lr_mult"] == 0.01
    assert type(F.fl_run_config(s, "x", 64).solver) is ItemLRSolver and F.p2_label(wb).endswith("|items=unfrozen|ilr=0.01")
    pf = _recipe(personal_head=True)
    d, args = _dry(monkeypatch, tmp_path, pre, pf, "--role", "CALIBRATION")
    s = d["arm_spec"]
    assert s["cfg"]["method"] == "PF" and s["cfg"]["pf"] == {"lr": PEAK, "lam": 1e-4, "clip": 1.0}
    assert F.default_run_id(args).endswith("_PF_CALIB_s2026") and F.p2_label(pf).endswith("|pf")


# ================================================================================================ refusals
P1_ARMS_FOR_REFUSAL = ("FA", "FP", "PF", "FA_Q8", "S_CAL", "FA_WARM", "FA_WARM_FROZEN", "DP_WARM_FROZEN", "PF_WARM")


def _p1_flags(arm):
    out = ["--freeze-item-tables"] if F.is_frozen_arm(arm) else []
    if F.base_arm(arm) in ("FA_1024", "DP"):
        out += ["--clip-record", "s.json"]
    if F.base_arm(arm) == "DP":
        out += ["--epsilon", "8"]
    return out


def test_phase1_arms_refuse_every_p2_flag(tmp_path, pre):
    f = _write_recipe(tmp_path / "R.json", _recipe())
    for arm in P1_ARMS_FOR_REFUSAL:
        init = F.is_warm_arm(arm)
        F.fl05_preflight(_args(tmp_path, pre, arm, *_p1_flags(arm), init=init))        # the plain arm: admitted
        for flag in (["--p2-recipe", str(f)], ["--p2-selection", str(F.P2_SELECTION_PATH)]):
            for kw in ({}, {"smoke": False}):
                a = _args(tmp_path, pre, arm, *_p1_flags(arm), *flag, init=init, **kw)
                assert_raises(F.LaunchRefused, F.fl05_preflight, a)
                assert_raises(F.LaunchRefused, F.preflight, _args(tmp_path, pre, arm, *_p1_flags(arm), *flag,
                                                                  "--dry-run", init=init, **kw))
        assert_raises(F.LaunchRefused, F.arm_spec, arm, seed=2026, peak_lr=PEAK, p2_recipe=_recipe(),
                      init_from=FAKE_IDENT if init else None)
    assert_raises(F.LaunchRefused, F.preflight, _args(tmp_path, pre, "FA", "--smoke-rounds", "51", init=False))


def test_p2_arm_refuses_phase1_options_and_bad_inputs(tmp_path, pre):
    f = _write_recipe(tmp_path / "R.json", _recipe())
    ok = _args(tmp_path, pre, "P2_W", "--p2-recipe", str(f))
    F.fl05_preflight(ok)
    F.preflight(_args(tmp_path, pre, "P2_W", "--p2-recipe", str(f), "--dry-run", "--smoke-rounds", "65"))
    assert F.P2_MAX_SMOKE_ROUNDS == 65                     # P2-T3: 64 updates + the loss line of theta_64 (m2)
    assert_raises(F.LaunchRefused, F.preflight, _args(tmp_path, pre, "P2_W", "--p2-recipe", str(f), "--dry-run",
                                                      "--smoke-rounds", "66"))
    p1 = {"lr_mult": ["--lr-mult", "2"], "lr_selection": ["--lr-selection", "x.json"], "fl_tuning": ["--fl-tuning", "x"],
          "server_opt": ["--server-opt", "fedadam"], "server_lr": ["--server-lr", "0.01"],
          "server_tau": ["--server-tau", "1e-05"], "lr_shape": ["--lr-shape", "fl_const"],
          "client_opt": ["--client-opt", "sgd_m"], "client_lr": ["--client-lr", "0.1"],
          "local_passes": ["--local-passes", "1"], "local_batch": ["--local-batch", "8"],
          "warm_lr_factor": ["--warm-lr-factor", "4"], "warm_selection": ["--warm-selection", "w.json"],
          "warm_server_lr_factor": ["--warm-server-lr-factor", "0.1"], "clip_record": ["--clip-record", "s.json"],
          "epsilon": ["--epsilon", "8"], "theta0_variants": ["--theta0-variants", "t.json"]}
    assert sorted(p1) == sorted(F.P2_REFUSED_FLAGS)
    for name, flag in p1.items():
        assert_raises(F.LaunchRefused, F.fl05_preflight, _args(tmp_path, pre, "P2_W", "--p2-recipe", str(f), *flag))
    sel = str(F.P2_SELECTION_PATH)
    bad = [(["--p2-recipe", str(f), "--freeze-item-tables"], {}),
           (["--p2-recipe", str(f)], {"init": False}),                                      # no theta_pre
           ([], {}),                                                                         # neither input
           (["--p2-recipe", str(f), "--p2-selection", sel], {}),                             # both
           (["--p2-recipe", str(f)], {"smoke": False}),                                      # MAIN without selection
           (["--p2-selection", sel, "--role", "CALIBRATION"], {}),                           # selection is MAIN-only
           (["--p2-recipe", str(f), "--role", "CALIBRATION"], {"seed": "2027"})]             # CALIB at 2026 only
    for extra, kw in bad:
        assert_raises(F.LaunchRefused, F.fl05_preflight, _args(tmp_path, pre, "P2_W", *extra, **kw))
    F.fl05_preflight(_args(tmp_path, pre, "P2_W", "--p2-selection", sel, smoke=False))   # MAIN: selection admitted
    recipes = [dict(server_opt="fedadam", server_lr=0.01, client_lr=0.1),                 # not a P2 server LR
               dict(server_opt="fedavg", server_lr=1e-4, client_lr=0.01),                 # fedadam LR on fedavg
               dict(server_opt="fedadam", server_lr=1e-4, client_lr=1.0),                 # Phase-1 client LR, not P2
               dict(server_opt="sgd", server_lr=1e-4, client_lr=0.01),
               dict(server_opt="fedadam", server_lr=1e-4, client_lr=0.01, item_lr_mult=0.1),          # frozen + mult
               dict(server_opt="fedadam", server_lr=1e-4, client_lr=0.01, items="unfrozen"),          # no mult
               dict(server_opt="fedadam", server_lr=1e-4, client_lr=0.01, items="unfrozen", item_lr_mult=0.5),
               dict(server_opt="fedadam", server_lr=1e-4, client_lr=0.01, items="unfrozen", item_lr_mult=0.1,
                    upload="delta_rows"),                                                # P2-03 not installed
               dict(server_opt="fedadam", server_lr=1e-4, client_lr=0.01, lora_rank=4, lora_lr_factor=10.0),
               dict(server_opt="fedadam", server_lr=1e-4, client_lr=0.01, items="unfrozen", item_lr_mult=0.1,
                    personal_head=True),
               dict(server_opt="fedadam", server_lr=1e-4, client_lr=0.01, extra=1),
               dict(server_opt="fedadam", server_lr=True, client_lr=0.01)]
    for r in recipes:
        assert_raises(F.LaunchRefused, F.check_p2_recipe, r)
    good = F.p2_recipe_file_obj(_recipe())
    for i, obj in enumerate((dict(good, label="x"), dict(good, format="PPSI_V4_P2_RECIPE_0"), dict(good, extra=1))):
        p = tmp_path / f"bad{i}.json"
        p.write_text(json.dumps(obj), encoding="utf-8")
        assert_raises(F.LaunchRefused, F.fl05_preflight, _args(tmp_path, pre, "P2_W", "--p2-recipe", str(p)))
    s = F.p2_arm_spec("P2_W", recipe=_recipe(), seed=2026, peak_lr=PEAK, role="CALIBRATION", init_from=FAKE_IDENT)
    F.check_p2_spec(s)
    for t in (dict(s, max_rounds=10), dict(s, cfg=dict(s["cfg"], seed=2027)), dict(s, frozen_item_tables=None),
              dict(s, p2_label="x"), {k: v for k, v in s.items() if k != "frozen_item_tables"}):
        assert_raises(F.LaunchRefused, F.check_p2_spec, t)
    assert_raises(F.LaunchRefused, F.arm_spec, "P2_W", seed=2026, peak_lr=PEAK, p2_recipe=_recipe(),
                  init_from=FAKE_IDENT, lr_mult=2.0)
    assert_raises(F.LaunchRefused, F.arm_spec, "P2_W", seed=2026, peak_lr=PEAK, p2_recipe=_recipe(),
                  init_from=FAKE_IDENT, server_opt="fedadam", server_lr=0.01)
    assert_raises(F.LaunchRefused, F.p2_arm_spec, "P2_W", recipe=_recipe(), seed=2026, peak_lr=PEAK, init_from=None)
    assert_raises(F.LaunchRefused, F.p2_arm_spec, "P2_W", recipe=_recipe(), seed=2026, peak_lr=PEAK,
                  init_from=dict(FAKE_IDENT, train_manifest="MAIN_TRAIN_5_100_131072"))     # identity rule unchanged


# ================================================================================================ MAIN binding
def _selection_obj(rec, ident):
    obj = {"format": F.P2_SELECTION_FORMAT, "stage": "FINAL", "winner": F.p2_label(rec), "winner_recipe": dict(rec),
           "common": {"init_from": dict(ident)}, "flags": []}
    obj["self_sha256"] = F._canon(obj)
    return obj


def test_main_binds_the_registered_selection(monkeypatch, tmp_path, pre):
    reg = tmp_path / "contracts" / "fl" / "P2_WARM_SELECTION_FINAL.json"
    reg.parent.mkdir(parents=True)
    monkeypatch.setattr(F, "P2_SELECTION_PATH", reg)
    calls = []
    stub = types.ModuleType("v4int.p2_select")
    stub.load_p2_selection = lambda path, stage: calls.append((str(path), stage)) or {}
    monkeypatch.setitem(sys.modules, "v4int.p2_select", stub)
    rec = _recipe(server_opt="fedavg", server_lr=0.3, client_lr=0.01)
    reg.write_text(json.dumps(_selection_obj(rec, pre["ident"])), encoding="utf-8")
    args = _args(tmp_path, pre, "P2_W", "--p2-selection", str(reg), smoke=False)
    F.preflight(_args(tmp_path, pre, "P2_W", "--p2-selection", str(reg), "--dry-run", smoke=False))
    spec, extra = F.resolve_spec(args, PEAK, fused=True, init_identity=pre["ident"])
    b = extra["p2_selection"]
    assert b["sha256"] == hashlib.sha256(reg.read_bytes()).hexdigest() and b["recipe"] == rec and b["init_from_checked"]
    assert calls == [(str(reg), "FINAL")] and spec["max_rounds"] == 13654 and spec["role"] == "MAIN"
    assert spec["cfg"]["server_opt"] == {"name": "fedavgm", "lr": 0.3, "beta1": 0.0}
    assert F.default_run_id(args) == "P2_WSTAR_SMALL_s2026"
    copy = tmp_path / "copy.json"
    copy.write_bytes(reg.read_bytes())
    assert_raises(F.LaunchRefused, F.resolve_spec, _args(tmp_path, pre, "P2_W", "--p2-selection", str(copy),
                                                         smoke=False), PEAK, True, pre["ident"])
    good = json.loads(reg.read_text(encoding="utf-8"))
    tampered = [dict(good, winner="fedavg|eta=1|c=0.01"), dict(_selection_obj(rec, pre["ident"]), stage="WA"),
                dict(_selection_obj(rec, pre["ident"]), format="PPSI_V4_P2_WARM_SELECTION_A_1"),
                _selection_obj(rec, dict(pre["ident"], checkpoint_sha256="1" * 64))]
    for i, obj in enumerate(tampered):
        if i in (1, 2):
            obj["self_sha256"] = F._canon(obj)
        reg.write_text(json.dumps(obj), encoding="utf-8")
        assert_raises(F.LaunchRefused, F.resolve_spec, args, PEAK, True, pre["ident"])
    reg.write_text(json.dumps(good), encoding="utf-8")

    def refuse(path, stage):
        raise ValueError("decision does not re-derive")
    stub.load_p2_selection = refuse
    assert_raises(F.LaunchRefused, F.resolve_spec, args, PEAK, True, pre["ident"])
    monkeypatch.setitem(sys.modules, "v4int.p2_select", None)                      # verifier absent -> refused
    assert_raises(F.LaunchRefused, F.resolve_spec, args, PEAK, True, pre["ident"])


# ================================================================================================ P2-T3 (local) + D1-ext
def test_incumbent_recipe_is_s8tc_cfg():
    inc = F.p2_wa_recipes()[-1]
    assert F.p2_label(inc) == "fedadam|eta=0.001|c=0.1"
    p = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="CALIBRATION", init_from=FAKE_IDENT, p2_recipe=inc)
    s = F.arm_spec("FA_WARM_FROZEN", seed=2026, peak_lr=PEAK, fused=True, role="CALIBRATION", init_from=FAKE_IDENT,
                   warm_server_lr_factor=0.1, **F.tuning_kwargs(F.tuning_config(**S4)))
    assert p["cfg"] == s["cfg"] and p["plan"] == s["plan"] and p["max_rounds"] == s["max_rounds"] == 3414
    assert p["frozen_item_tables"] == s["frozen_item_tables"] and p["init_from"] == s["init_from"]
    assert F.fl_run_config(p, "S8Tc", 64).digest() == F.fl_run_config(s, "S8Tc", 64).digest()


def _tiny(monkeypatch, n=12, group=4):
    """P2 tiny worlds: the registered builders under a small client count / group (rounds = ceil(60 n / (36 x 1)))."""
    monkeypatch.setitem(F.CALIB, "n_clients", n)
    monkeypatch.setattr(F, "MAIN_N_CLIENTS", n)
    monkeypatch.setitem(F.KNOBS, "group", group)


def _run(tmp_path, pre, spec, name, **kw):
    return F.execute(spec=spec, run_id=name, run_dir=tmp_path / name, say=lambda *_: None, **warm_world(pre), **kw)


def _read(p):
    return Path(p).read_bytes() if Path(p).is_file() else None


def test_p2t3_tiny_run_bitwise_equals_the_phase1_path(monkeypatch, tmp_path, pre):
    _tiny(monkeypatch)
    inc = F.p2_wa_recipes()[-1]
    p = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="CALIBRATION", init_from=pre["ident"], p2_recipe=inc)
    s = F.arm_spec("FA_WARM_FROZEN", seed=2026, peak_lr=PEAK, fused=True, role="CALIBRATION", init_from=pre["ident"],
                   warm_server_lr_factor=0.1, **F.tuning_kwargs(F.tuning_config(**S4)))
    assert p["cfg"] == s["cfg"] and p["max_rounds"] == 20
    a = _run(tmp_path, pre, p, "p2")
    b = _run(tmp_path, pre, s, "p1")
    assert a["status"] == b["status"] == "DONE_MAIN"
    sa, sb = a["summary"], b["summary"]
    assert sa["server_digest"] == sb["server_digest"] and sa["ledger"] == sb["ledger"] and sa["evals"] == sb["evals"]
    assert _read(tmp_path / "p2" / F.TRAIN_LOSS_FILE) == _read(tmp_path / "p1" / F.TRAIN_LOSS_FILE) is not None
    assert _read(tmp_path / "p1" / F.UPDATE_NORM_FILE) is None and "update_norm_log" not in sb
    un = sa["update_norm_log"]
    assert un["complete"] and un["rounds_logged"] == 20 and un["server_steps"] >= 1
    assert un["trained_keys"] and not set(F.FROZEN_ITEM_KEYS) & set(un["trained_keys"])
    lines = [json.loads(x) for x in (tmp_path / "p2" / F.UPDATE_NORM_FILE).read_text(encoding="utf-8").splitlines()]
    assert [x["round"] for x in lines] == list(range(20)) and all("n_clients" in x for x in lines)
    rows = json.loads((tmp_path / "p2" / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert rows and all(r["method"] == "P2_W" and r["study"] == "PREG2" and r["p2_recipe_label"] == F.p2_label(inc)
                        and r["row_status"] == "FL_LR_SCREEN" and r["method_code"] == "FA" for r in rows)
    assert sa["p2"]["label"] == F.p2_label(inc) and sa["p2"]["pre_server"] is None and sa["p2"]["post_run"] is None


def test_p2_resume_main_rows_wb_and_pf(monkeypatch, tmp_path, pre):
    _tiny(monkeypatch, n=60)                               # 100 rounds: the 8 smoke rounds stay below the first mark
    rec = _recipe(server_opt="fedavgm", server_lr=0.1, client_lr=0.03)
    spec = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="MAIN", init_from=pre["ident"], p2_recipe=rec)
    straight = _run(tmp_path, pre, spec, "s", smoke_rounds=8)
    part = _run(tmp_path, pre, spec, "p", smoke_rounds=8, interrupt_at=5)
    assert part["status"] == "INTERRUPTED"
    res = _run(tmp_path, pre, spec, "p", smoke_rounds=8, resume=True)
    assert res["status"] == straight["status"] == "SMOKE_DONE_NOT_EVIDENCE"
    assert res["summary"]["server_digest"] == straight["summary"]["server_digest"]
    lines = [json.loads(x) for x in (tmp_path / "p" / F.UPDATE_NORM_FILE).read_text(encoding="utf-8").splitlines()]
    assert [x["round"] for x in lines] == list(range(8))
    stepped = [x for x in lines if x["server_step"]]
    assert stepped[0]["cos_prev_available"] is False
    first_after = next(x for x in lines if x["round"] >= 5 and x["server_step"])
    assert first_after["cos_prev_available"] is False                       # in-memory Delta_{r-1} restarts on resume
    sl = [json.loads(x) for x in (tmp_path / "s" / F.UPDATE_NORM_FILE).read_text(encoding="utf-8").splitlines()]
    strip = (lambda x: {k: v for k, v in x.items() if k not in ("cos_prev", "cos_prev_available")})  # noqa: E731
    assert [strip(x) for x in lines] == [strip(x) for x in sl]
    full = _run(tmp_path, pre, spec, "m")
    rows = json.loads((tmp_path / "m" / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert rows and all(r["row_status"] == "P2_ARM" and r["comparison_status"] == "P2_FOLLOWUP" and
                        r["study"] == "PREG2" and r["fl_client_lr"] == 0.03 for r in rows)
    wb = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="CALIBRATION", init_from=pre["ident"],
                    p2_recipe=_recipe(items="unfrozen", item_lr_mult=0.1))
    o = _run(tmp_path, pre, wb, "wb")
    assert o["status"] == "DONE_MAIN" and "frozen_item_tables" not in o["summary"]
    assert set(F.FROZEN_ITEM_KEYS) <= set(o["summary"]["update_norm_log"]["trained_keys"])
    pf = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="CALIBRATION", init_from=pre["ident"],
                    p2_recipe=_recipe(personal_head=True))
    o = _run(tmp_path, pre, pf, "pf")
    assert o["status"] == "DONE_MAIN" and o["summary"]["pf_store"]["clients"] > 0
    prow = json.loads((tmp_path / "pf" / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert all(r["method_code"] == "PF" for r in prow)
    assert full["summary"]["server_digest"] != o["summary"]["server_digest"]


def test_update_norm_log_never_blocks_completion(tmp_path):
    """REV-P2-01_02 m5: a torn last line or a duplicated round is flagged in the summary, never raised; trim drops a
    torn line (the log is descriptive)."""
    good = [{"round": r, "server_step": True, "cos_prev_available": r > 0, "total_T": {"agg_delta_l2": 1.0}}
            for r in range(4)]
    p = tmp_path / F.UPDATE_NORM_FILE
    p.write_text("".join(json.dumps(r) + "\n" for r in good) + '{"round": 4, "server_st', encoding="utf-8")
    s = F.update_norm_summary(tmp_path, 5, trained_keys=["a"], trained_numel=3)
    assert s["unparsable_lines"] == 1 and s["complete"] is False and s["missing_rounds"] == [4]
    assert s["strictly_increasing"] is True and s["rounds_logged"] == 4
    F.trim_update_norm_log(tmp_path, 3)
    assert [r["round"] for r in F.read_update_norm_log(tmp_path)] == [0, 1, 2]
    p.write_text("".join(json.dumps(r) + "\n" for r in good + [good[2]]), encoding="utf-8")
    s = F.update_norm_summary(tmp_path, 4)
    assert s["strictly_increasing"] is False and s["complete"] is False and s["unparsable_lines"] == 0
    p.write_text("".join(json.dumps(r) + "\n" for r in good), encoding="utf-8")
    s = F.update_norm_summary(tmp_path, 4)
    assert s["complete"] is True and s["strictly_increasing"] is True and s["cos_prev_unavailable_rounds"] == [0]


# ================================================================================================ negative controls
@nc("the P2 client-LR grid is admitted for P2 arms only: a Phase-1 tuning point at c = 0.03 must be refused")
def test_nc_phase1_admits_p2_client_lr():
    try:
        F.tuning_config(client_opt="sgd_m", client_lr=0.03)
    except F.LaunchRefused:
        raise AssertionError("refused (the Phase-1 grid is unchanged)")


@nc("a Phase-1 arm that accepted --p2-recipe would break the firewall")
def test_nc_phase1_arm_accepts_p2_recipe(tmp_path, pre):
    f = _write_recipe(tmp_path / "R.json", _recipe())
    a = _args(tmp_path, pre, "FA", "--p2-recipe", str(f), init=False)
    try:
        F.fl05_preflight(a)
    except F.LaunchRefused:
        raise AssertionError("refused (the firewall holds)")
