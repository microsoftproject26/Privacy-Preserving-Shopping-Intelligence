"""P2-T3 receipt tool (tools/p2_t3_check.py; REV-P2-01_02 R1 / m2) on synthetic tiny runs (CPU; no data, no value).

A tiny "S8Tc" (FA_WARM_FROZEN, CALIBRATION, S4 point, server-LR factor 0.1, run to DONE_MAIN) and the P2_W incumbent
recipe run as a 65-round smoke through execute (the P2-T3 server command, scaled down: 800 rounds so 65 stay below the
first mark) give verdict BIT_IDENTICAL, and the receipt is accepted by v4int.p2_select.check_p2t3_receipt (the closed
P2-06 checker); a different recipe, a 64-round smoke and a tampered loss line give FAIL (still written, refused by
p2_select); a missing input refuses with exit 2 and writes nothing; the output is O_EXCL; find_s8tc picks the unique
f1/sf0.1 cell.
NC (strict xfail): a 64-line comparison would not cover the 64th update (the m2 reason for 65 lines).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

import v4int.launch_fl as F
from intkit import nc
from test_fl12_launch import make_pre, warm_world
import ppsi_paths

V4 = ppsi_paths.PROJECT_ROOT.resolve()       # the ppsi_v4 workspace (contracts, company records)
PEAK = 0.000711
S4 = dict(optimizer="fedadam", server_lr=0.01, lr_shape="fl_const", client_opt="sgd_m", client_lr=0.1, local_passes=1)


def _tool():
    spec = importlib.util.spec_from_file_location("p2_t3_check", ppsi_paths.locate("tools/p2_t3_check.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module", autouse=True)
def _strict_numerics_module():
    """The module-scoped tiny runs execute before the per-test strict-FP32 fixture (as in test_fl16)."""
    import torch
    from fl.numerics import enforce_strict_fp32
    torch.set_num_threads(1)
    enforce_strict_fp32()
    yield


@pytest.fixture(scope="module")
def pre(tmp_path_factory):
    root = tmp_path_factory.mktemp("p2t3pre")
    p, adapter = make_pre(root)
    state, ident, launch = F.load_pretrain_init(p, model_variant="SASREC_D64_B2")
    return {"path": p, "state": state, "ident": ident}


def _launch(run_dir: Path, *, policy="USER_HISTORY_L50", gpu="NVIDIA A40") -> None:
    obj = {"clients": {"n": 10, "n_decisions": 123, "manifest_hash": "tiny-manifest"}, "history_policy": policy,
           "device": {"physical_gpu": 0, "name": gpu},
           "fl05": {"role_record": {"manifests": {"train": "CALIB_TRAIN_5_100_32768",
                                                  "eval": "CALIB_VALIDATION_5_100_32768"}}}}
    (Path(run_dir) / "LAUNCH_000.json").write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture(scope="module")
def runs(tmp_path_factory, pre):
    """Module-scoped tiny worlds under the patched registered builders (CALIB 480 clients -> 800 rounds, group 4)."""
    mp = pytest.MonkeyPatch()
    mp.setitem(F.CALIB, "n_clients", 480)
    mp.setitem(F.KNOBS, "group", 4)
    root = tmp_path_factory.mktemp("p2t3")
    inc = F.p2_wa_recipes()[-1]

    def go(name, spec, **kw):
        out = F.execute(spec=spec, run_id=name, run_dir=root / name, say=lambda *_: None, **warm_world(pre), **kw)
        _launch(root / name)
        return out

    s8 = F.arm_spec("FA_WARM_FROZEN", seed=2026, peak_lr=PEAK, fused=True, role="CALIBRATION", init_from=pre["ident"],
                    warm_server_lr_factor=0.1, **F.tuning_kwargs(F.tuning_config(**S4)))
    p2 = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="CALIBRATION", init_from=pre["ident"], p2_recipe=inc)
    other = F.arm_spec("P2_W", seed=2026, peak_lr=PEAK, role="CALIBRATION", init_from=pre["ident"],
                       p2_recipe=F.check_p2_recipe({"server_opt": "fedadam", "server_lr": 1e-3, "client_lr": 0.03}))
    assert s8["max_rounds"] == p2["max_rounds"] == 800
    out = {"root": root, "s8": go("S8Tc_tiny", s8), "p2": go("p2_65", p2, smoke_rounds=65),
           "p2_64": go("p2_64", p2, smoke_rounds=64), "other": go("p2_other", other, smoke_rounds=65)}
    (root / "decoy").mkdir()
    (root / "decoy" / "ARM_SPEC.json").write_text(json.dumps({"arm_spec": dict(s8, warm_server_lr_factor=None)}),
                                                  encoding="utf-8")
    yield out
    mp.undo()


def _p2select_rec(run_dir: Path) -> dict:
    rr = json.loads((run_dir / "RESULT_ROWS.json").read_text(encoding="utf-8"))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    return {"run_id": rr["run_id"], "run_dir": str(run_dir), "result_rows_sha256": sha(run_dir / "RESULT_ROWS.json"),
            "run_summary_sha256": sha(run_dir / "RUN_SUMMARY.json")}


def test_p2t3_receipt_pass_is_accepted_by_p2_select(runs, tmp_path):
    T = _tool()
    root = runs["root"]
    assert runs["s8"]["status"] == "DONE_MAIN" and runs["p2"]["status"] == "SMOKE_DONE_NOT_EVIDENCE"
    assert T.find_s8tc(root) == root / "S8Tc_tiny"
    out = tmp_path / "P2T3_RECEIPT.json"
    assert T.main(["--p2-run-dir", str(root / "p2_65"), "--s8tc-search-root", str(root), "--out", str(out)]) == 0
    rec = json.loads(out.read_text(encoding="utf-8"))
    assert rec["verdict"] == "BIT_IDENTICAL" and all(rec["checks"].values()) and rec["rounds"] == 64
    assert rec["test"] == "P2-T3" and rec["s8tc_run_id"] == "S8Tc_tiny" and rec["train_loss_lines_compared"] == 65
    assert rec["self_sha256"] == T._canon(rec) and rec["p2_smoke_server_digest"]
    try:
        import v4int.p2_select as PS
    except ImportError:
        pytest.skip("v4int.p2_select is not installed")
    got = PS.check_p2t3_receipt(out, _p2select_rec(root / "S8Tc_tiny"))
    assert got["verdict"] == "BIT_IDENTICAL" and got["rounds"] == 64 and got["s8tc_run_id"] == "S8Tc_tiny"
    assert T.main(["--p2-run-dir", str(root / "p2_65"), "--s8tc-run-dir", str(root / "S8Tc_tiny"),
                   "--out", str(out)]) == 2                                   # O_EXCL: never overwritten


def test_p2t3_fail_cases_and_refusals(runs, tmp_path):
    T = _tool()
    root, s8 = runs["root"], runs["root"] / "S8Tc_tiny"
    other = T.check(s8, root / "p2_other")
    assert other["verdict"] == "FAIL" and not other["checks"]["p2_is_incumbent"] and not other["checks"]["spec_cfg"]
    assert not other["checks"]["train_loss_65_lines_equal"]
    short = T.check(s8, root / "p2_64")
    assert short["verdict"] == "FAIL" and not short["checks"]["p2_smoke_65_rounds"]
    assert not short["checks"]["train_loss_65_lines_equal"] and short["checks"]["spec_cfg"]
    # a tampered loss line (one byte) in a copy of the P2 run -> FAIL
    import shutil
    bad = tmp_path / "p2_bad"
    shutil.copytree(root / "p2_65", bad)
    lines = (bad / "TRAIN_LOSS.jsonl").read_text(encoding="utf-8").splitlines()
    r = json.loads(lines[40])
    r["loss_sum"] = float(r["loss_sum"]) * (1 + 2 ** -40)
    lines[40] = json.dumps(r, sort_keys=True)
    (bad / "TRAIN_LOSS.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    t = T.check(s8, bad)
    assert t["verdict"] == "FAIL" and not t["checks"]["train_loss_65_lines_equal"] and t["checks"]["spec_cfg"]
    # another history policy / GPU class -> FAIL
    _launch(bad, policy="SESSION_L50", gpu="other")
    t = T.check(s8, bad)
    assert not t["checks"]["history_policy"] and not t["checks"]["gpu_name"]
    out = tmp_path / "R_fail.json"
    assert T.main(["--p2-run-dir", str(root / "p2_other"), "--s8tc-run-dir", str(s8), "--out", str(out)]) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "FAIL"
    try:
        import v4int.p2_select as PS
        ok = False
        try:
            PS.check_p2t3_receipt(out, _p2select_rec(s8))
        except PS.P2SelectionRefused:
            ok = True
        assert ok, "a FAIL receipt must be refused by p2_select"
    except ImportError:
        pass
    # missing input -> refused, nothing written
    gone = tmp_path / "s8_no_rows"
    shutil.copytree(s8, gone)
    (gone / "RESULT_ROWS.json").unlink()
    out2 = tmp_path / "R_none.json"
    assert T.main(["--p2-run-dir", str(root / "p2_65"), "--s8tc-run-dir", str(gone), "--out", str(out2)]) == 2
    assert not out2.exists()


@nc("comparing only 64 TRAIN_LOSS lines leaves the 64th update (theta_64) unchecked: the 65-line rule must matter")
def test_nc_64_lines_cover_all_updates(runs):
    root = runs["root"]
    a = (root / "S8Tc_tiny" / "TRAIN_LOSS.jsonl").read_text(encoding="utf-8").splitlines()
    b = (root / "p2_64" / "TRAIN_LOSS.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(b) == 64 and a[:64] == b[:64]                                   # 64 lines agree ...
    assert len(b) >= 65                                                        # ... but theta_64 has no line
