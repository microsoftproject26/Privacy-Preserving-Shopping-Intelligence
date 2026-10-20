"""INT-03 item 1 / 4 — v4int.launch_fl (CPU, tiny synthetic model and clients; the real-data wiring is covered by
test_int03_real_clients.py).

  * the registered arm specs (FA / FP / PF: 64 per round, sweep, 10% drop-out, 6,827 rounds, B16 x 2; FA_1024 / DP:
    m 1,024, Poisson (FL-07), T 384, no drop-out, clip S, z from the registered record; S_CAL: seed 2026, 20 rounds);
  * refusals: family / arm / epsilon / clip record / S_CAL seed / smoke bounds / G1 gate / GPU (+ --allow-shared-gpu);
  * the engine end to end: 12 normalized evaluations, RESULT_ROWS (PRACTICAL_BEST + CONTROLLED_BUDGET, relabelled
    rows), bytes per visit / client / fleet, latest removed, PF store; determinism (two runs bitwise); resume exact;
    S_CAL -> S_RECORD -> FA_1024 / DP; smoke deletes every weight file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import v4int.launch_fl as F
from intkit import assert_raises, nc
from test_launcher import fake_runner, gate

from fl.numerics import state_digest
from fl.synthetic import make_clients, make_tiny_adapter

PEAK = 1e-2


def _tiny_spec(arm, rounds=6, group=4, **kw):
    s = F.arm_spec(arm, seed=kw.pop("seed", 2026), peak_lr=PEAK, **kw)
    s["cfg"]["endpoint_rounds"] = rounds
    s["max_rounds"] = min(rounds, s["max_rounds"])
    if s["plan"]["sampling"] == "sweep":
        s["plan"]["group_size"] = group
    return s


def _world(n=10, seed=5, impure=False):
    server = make_tiny_adapter(seed=1)
    worker = make_tiny_adapter(seed=1)
    clients = {c.key: c for c in make_clients(n, seed=seed)}
    calls = {"n": 0}

    def get_client(k):
        calls["n"] += 1
        c = clients[k]
        if impure and calls["n"] > 2:                    # an impure source: rows depend on a call counter
            ex = dict(c.examples)
            ex["target_class"] = torch.where(ex["target_class"] >= 0, (ex["target_class"] + 1) % 24,
                                             ex["target_class"])
            return type(c)(c.key, ex)
        return c
    from fl.client import valid_rows
    nv = {k: int(valid_rows(c.examples).numel()) for k, c in clients.items()}

    def evaluate(mark):
        v = float(sum(p.double().sum() for p in server.shared_parameters()))
        rows = [{"metric": "mrr", "k": 20, "value": v, "split": "VALIDATION", "method": "C",
                 "training_paradigm": "CENTRAL", "row_status": "CENTRAL_CONTEST"}]
        return abs(v) % 1.0 + mark, rows, np.arange(5), {"E2E": {"n": 5}}
    return dict(server_adapter=server, init_sha256=state_digest(server.broadcast_state()), workers=[worker],
                keys=sorted(clients), n_valid_by_key=nv, get_client=get_client,
                n_decisions=sum(nv.values()), manifest_hash="tiny-manifest", evaluate=evaluate)


def _run(tmp_path, spec, name="r", **kw):
    w = _world(impure=kw.pop("impure", False))
    out = F.execute(spec=spec, run_id=name, run_dir=tmp_path / name, say=lambda *_: None, **w, **kw)
    return out, w


# ------------------------------------------------------------------------------------------------ specs
def test_registered_arm_specs():
    fa = F.arm_spec("FA", seed=2026, peak_lr=PEAK)
    assert fa["plan"] == {"group_size": 64, "sampling": "sweep"}
    assert (fa["cfg"]["dropout_p"], fa["cfg"]["endpoint_rounds"], fa["cfg"]["solver"]["batch_size"],
            fa["cfg"]["solver"]["passes"], fa["cfg"]["mu"], fa["cfg"]["pf"], fa["cfg"]["dp"]) == \
        (0.1, 6827, 16, 2, 0.0, None, None)
    assert F.arm_spec("FP", seed=2026, peak_lr=PEAK)["cfg"]["mu"] == 0.01
    assert F.arm_spec("PF", seed=2026, peak_lr=PEAK)["cfg"]["pf"] == {"lr": PEAK, "lam": 1e-4, "clip": 1.0}
    f1024 = F.arm_spec("FA_1024", seed=2027, peak_lr=PEAK, clip_norm=2.5)
    assert f1024["plan"] == {"group_size": 1024, "sampling": "poisson"}                 # FL-07 / amendment 06C
    assert (f1024["cfg"]["dropout_p"], f1024["cfg"]["endpoint_rounds"], f1024["cfg"]["dp"]) == \
        (0.0, 384, {"clip_norm": 2.5, "noise_multiplier": 0.0, "denominator": 1024})
    dp = F.arm_spec("DP", seed=2026, peak_lr=PEAK, clip_norm=2.5, z=0.5642)
    assert dp["cfg"]["dp"]["noise_multiplier"] == 0.5642
    sc = F.arm_spec("S_CAL", seed=2026, peak_lr=PEAK)
    assert sc["max_rounds"] == 20 and sc["cfg"]["dp"]["clip_norm"] is None
    for bad in (lambda: F.arm_spec("S_CAL", seed=2027, peak_lr=PEAK), lambda: F.arm_spec("FA_1024", seed=1,
                peak_lr=PEAK), lambda: F.arm_spec("DP", seed=1, peak_lr=PEAK, clip_norm=1.0),
                lambda: F.arm_spec("C", seed=1, peak_lr=PEAK)):
        assert_raises(F.LaunchRefused, bad)
    for arm, extra in (("FA", {}), ("FP", {}), ("PF", {}), ("FA_1024", {"clip_norm": 1.0}),
                       ("DP", {"clip_norm": 1.0, "z": 0.5642}), ("S_CAL", {})):
        F.fl_run_config(F.arm_spec(arm, seed=2026, peak_lr=PEAK, **extra), "x", 64)       # fl.runtime accepts it


def test_registered_z_and_its_refusals(tmp_path):
    assert F.load_z(8)["z"] == 0.5642 and F.load_z(1)["z"] == 1.3094
    assert_raises(F.LaunchRefused, F.load_z, 2)
    rec = json.loads(F.DP_Z_RECORD.read_text(encoding="utf-8"))
    rec["records"][0]["z"] = 0.5
    bad = tmp_path / "z.json"
    bad.write_text(json.dumps(rec), encoding="utf-8")
    assert_raises(F.LaunchRefused, F.load_z, 8, bad)


def test_architecture_lock_winner_only():
    assert F.assert_arm_allowed("FA", "SASREC")["method_code"] == "FA"
    assert F.assert_arm_allowed("DP", "SASREC")["method_code"] == "FA"
    assert_raises(F.LaunchRefused, F.assert_arm_allowed, "FA", "GRU")


def _args(tmp_path, *extra, smoke=True):
    a = ["--arm", "FA", "--seed", "2026", "--gpu", "0", "--run-dir", str(tmp_path / "r"),
         "--history-policy", "USER_HISTORY_L50"] + list(extra)
    if smoke:
        a.append("--smoke")
    return F.parse_args(a)


def test_preflight_refusals_and_shared_gpu(tmp_path):
    g = gate(tmp_path)
    ok = F.preflight(_args(tmp_path, smoke=False), runner=fake_runner(), environ={}, gate_path=g)
    assert ok["g1_gate"]["required"] and "shared_gpu" not in ok["gpu_check"]
    for extra in (["--family", "GRU"], ["--arm", "DP"], ["--epsilon", "8"], ["--arm", "FA_1024"],
                  ["--arm", "S_CAL", "--seed", "2027"], ["--smoke-rounds", "999"], ["--checkpoint-every", "0"]):
        assert_raises(F.LaunchRefused, F.preflight, _args(tmp_path, *extra), runner=fake_runner(), environ={},
                      gate_path=g)
    assert_raises(F.LaunchRefused, F.preflight, _args(tmp_path, smoke=False), runner=fake_runner(), environ={},
                  gate_path=tmp_path / "absent.json")
    busy = fake_runner(apps=[("00000000:46:00.0", "4242")], mem={"0": 3000.0, "1": 0.0})
    assert_raises(F.LaunchRefused, F.preflight, _args(tmp_path), runner=busy, environ={}, gate_path=g)
    rec = F.preflight(_args(tmp_path, "--allow-shared-gpu"), runner=busy, environ={}, gate_path=g)
    assert rec["allow_shared_gpu"] and rec["gpu_check"]["shared_gpu"]["memory_used_mib_at_launch"] == 3000.0
    assert_raises(F.LaunchRefused, F.preflight, _args(tmp_path, "--allow-shared-gpu"), runner=fake_runner(fail=True),
                  environ={}, gate_path=g)
    assert F.preflight(_args(tmp_path, "--dry-run"), runner=fake_runner(fail=True), environ={},
                       gate_path=g)["gpu_check"] == {"skipped": "dry-run"}


# ------------------------------------------------------------------------------------------------ the engine
def test_fa_end_to_end_rows_bytes_and_retention_files(tmp_path):
    out, w = _run(tmp_path, _tiny_spec("FA"))
    assert out["status"] == "DONE_MAIN"
    rd = tmp_path / "r"
    rows = json.loads((rd / "RESULT_ROWS.json").read_text(encoding="utf-8"))["rows"]
    assert [r["selection_rule"] for r in rows] == ["PRACTICAL_BEST", "CONTROLLED_BUDGET"]
    assert all(r["method"] == "FA" and r["training_paradigm"] == "FEDERATED" and r["c_matched_eligible"] is False
               and r["row_status"] == "DOWNSTREAM_ARM" for r in rows)
    s = json.loads((rd / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
    assert s["status"] == "DONE_MAIN" and s["evals"][-1][0] == 12 and s["participation"]["rounds"] == 6
    b = s["bytes"]
    vb = b["per_visit"]["send_plus_receive_bytes"]
    assert b["fleet"]["total"] == s["participation"]["survived"] * vb
    assert b["per_client"]["max"] == s["counters"]["max_visits"] * vb
    assert (rd / "evals" / "eval_6.0.ranks_int32.npy").exists()
    assert s["checkpoints"] == ["best.pt", "endpoint_6.0.pt"]                   # latest removed at the end


def test_determinism_two_runs_bitwise(tmp_path):
    a, _ = _run(tmp_path, _tiny_spec("FP"), "a")
    b, _ = _run(tmp_path, _tiny_spec("FP"), "b")
    assert a["summary"]["server_digest"] == b["summary"]["server_digest"]
    assert a["summary"]["evals"] == b["summary"]["evals"] and a["summary"]["ledger"] == b["summary"]["ledger"]


def _resume_equal(tmp_path, impure=False):
    straight, _ = _run(tmp_path, _tiny_spec("PF"), "s")
    part, _ = _run(tmp_path, _tiny_spec("PF"), "p", interrupt_at=3, impure=impure)
    assert part["status"] == "INTERRUPTED"
    resumed, _ = _run(tmp_path, _tiny_spec("PF"), "p", resume=True, impure=impure)
    for k in ("server_digest", "store_digest", "evals", "ledger", "counters"):
        assert straight["summary"][k] == resumed["summary"][k], k


def test_resume_is_exact_pf_store_written(tmp_path):
    _resume_equal(tmp_path)
    assert (tmp_path / "p" / "PF_STORE.pt").exists()


def test_s_cal_then_fa_1024_and_dp(tmp_path):
    out, _ = _run(tmp_path, _tiny_spec("S_CAL", rounds=384), "cal")
    assert out["status"] == "DONE_S_CAL"
    rec = F.load_s_record(tmp_path / "cal" / "S_RECORD.json", smoke=False)
    assert rec["n_norms"] > 0 and rec["rounds_0based"] == list(range(20)) and rec["S"] > 0
    assert list((tmp_path / "cal" / "ckpt" / "cal").glob("*.pt")) == []           # no weights kept
    for arm, z in (("FA_1024", None), ("DP", 0.5642)):
        o, _ = _run(tmp_path, _tiny_spec(arm, rounds=4, clip_norm=rec["S"], z=z), arm)
        assert o["status"] == "DONE_MAIN" and o["summary"]["dp"]["clip_norm"] == rec["S"]
    tam = json.loads((tmp_path / "cal" / "S_RECORD.json").read_text(encoding="utf-8"))
    tam["S"] = 1.0
    (tmp_path / "t.json").write_text(json.dumps(tam), encoding="utf-8")
    assert_raises(F.LaunchRefused, F.load_s_record, tmp_path / "t.json", smoke=False)


def test_smoke_deletes_weights_and_relabels(tmp_path):
    spec = _tiny_spec("FA", rounds=6827, group=4)
    out, _ = _run(tmp_path, spec, "sm", smoke_rounds=3, ckpt_every=1)
    assert out["status"] == "SMOKE_DONE_NOT_EVIDENCE"
    assert list((tmp_path / "sm" / "ckpt" / "sm").glob("*.pt")) == []
    rows = json.loads((tmp_path / "sm" / "SMOKE_EVAL.json").read_text(encoding="utf-8"))["rows"]
    assert rows and all(r["split"] == "SMOKE_NOT_EVIDENCE" and r["method"] == "FA" for r in rows)
    assert_raises(F.LaunchRefused, _run, tmp_path, _tiny_spec("FA", rounds=6), "sm2", smoke_rounds=6)


def test_fresh_launch_refuses_an_existing_latest(tmp_path):
    _run(tmp_path, _tiny_spec("FA"), "x", interrupt_at=2)
    assert_raises(F.LaunchRefused, _run, tmp_path, _tiny_spec("FA"), "x")


# ------------------------------------------------------------------------------------------------ negative controls
@nc("an impure client source (rows depend on a call counter) makes the resumed run differ")
def test_nc_impure_client_source_breaks_resume(tmp_path):
    _resume_equal(tmp_path, impure=True)


@nc("the unrelabelled harness rows (method C, CENTRAL) are not downstream-arm rows")
def test_nc_unrelabelled_rows(tmp_path):
    rows = _world()["evaluate"](6.0)[1]
    assert all(r["method"] == "FA" and r["training_paradigm"] == "FEDERATED" for r in rows)


@nc("a DP arm that used the receipt's unrounded bisection z would not be the registered z")
def test_nc_unrounded_z(tmp_path):
    rec = json.loads(F.DP_Z_RECORD.read_text(encoding="utf-8"))
    assert rec["records"][0]["z_bisection"] == F.DP_REGISTERED_Z[8.0]


@nc("a GPU check without the ownership refusal would accept a busy GPU without --allow-shared-gpu")
def test_nc_busy_gpu_accepted_without_flag(tmp_path, monkeypatch):
    import v4sup.gpu as G
    monkeypatch.setattr(G, "gpu_ready", lambda *a, **k: (True, "ok"))           # MUTANT: no ownership check
    busy = fake_runner(apps=[("00000000:46:00.0", "4242")], mem={"0": 100.0, "1": 0.0})
    try:
        F.preflight(_args(tmp_path), runner=busy, environ={}, gate_path=gate(tmp_path))
        refused = False
    except F.LaunchRefused:
        refused = True
    assert refused, "without the ownership check a busy GPU is accepted"


@nc("an FA spec without the registered drop-out drops no client")
def test_nc_fa_without_dropout(tmp_path):
    spec = _tiny_spec("FA")
    spec["cfg"]["dropout_p"] = 0.0                                               # MUTANT
    out, _ = _run(tmp_path, spec)
    assert out["summary"]["participation"]["dropped"] > 0
