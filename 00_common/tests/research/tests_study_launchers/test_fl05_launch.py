"""FL-05 items 1, 2, 4 — v4int.launch_fl: --lr-mult / --lr-selection, --role CALIBRATION, the FA_Q8 arm (CPU, tiny
synthetic model and clients, as test_int03_launch_fl).

  * digest invariance: with every FL-05 option at its default, the arm-spec digest and the fl RunConfig digest of every
    pre-FL-05 arm (FA / FP / PF / FA_1024 / DP eps 8 and 1 / S_CAL; seeds 2026 / 2027; fused True / None) are the
    values computed from the PRE-FL-05 code (BASELINE below, computed on 2026-09-28 from launch_fl.py sha256
    288f57ad940ff7b184a779a0dc1df6c1927c649887b01d5a2009afd5afcb18f8, peak 0.000711, clip 1.2345, run id
    FL_<ARM>_SMALL_s<seed>, ckpt_every 64);
  * --lr-mult m multiplies lr_peak_local / the solver LR only (PF personal LR, mu, DP knobs unchanged); other values
    refused; --lr-selection exclusive with --lr-mult;
  * --role CALIBRATION: FA only, seed 2026, 1,707 rounds (= ceil(3 x 32,768 / 64 / 0.9)); rows role CALIBRATION /
    run_label FL_LR_SCREEN;
  * FA_Q8: method code FA, rows FA_Q8, Q8 bytes in the summary and the ledger, deterministic, resume exact, differs from
    FA; ARM_SPEC.json refuses a resume under another arm spec.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import v4int.launch_fl as F
from intkit import assert_raises, nc
from test_launcher import fake_runner, gate

from fl.numerics import state_digest
from fl.synthetic import make_clients, make_tiny_adapter

PEAK_REG = 0.000711
PEAK = 1e-2
BASELINE = {   # (arm spec sha256, fl RunConfig digest) from the pre-FL-05 code
    # FL-07 (amendment 06C): the FA_1024 / DP / S_CAL arm-spec digests are re-pinned to the Poisson plan (their old
    # values are kept in tests_int/test_fl07_launch.OLD_SPEC); their fl RunConfig digests are unchanged.
    "DP|2026|None|0.5642": ("b9463e189ac9718d47cc4ef9b7182338d8326a9b1acb768f50a59e15d052fef2",
        "82def72233f5578bf5e536f3bd161d67e4c83095d5879d1ab1a086bd6f5e64e3"),
    "DP|2026|None|1.3094": ("9e378bb12006eaca777ec5ca21c23f6af9251b440bade8da3586684ab78a78e7",
        "5cc89352c0d244dc68072d3fbd9ca08f6988e2c18efbdf9fe06b93b55f63d009"),
    "DP|2026|True|0.5642": ("90893fcf2fc5591132cba1af6a5526b1a9584b196564fc122c740b262b9d6f19",
        "1fd2616ead2f9e9f82dc103c4f68d74344a7789c50bcc464843d5e4e5016e172"),
    "DP|2026|True|1.3094": ("2823a424d6bf1b379a0b32994b378713025463ea57a871dbb87c2b54806a1a04",
        "a04034fb01d8685d32891cf6b7663d99ce1e9d818836e8c34f60785bc5454035"),
    "DP|2027|None|0.5642": ("c061b9b1b4f8a292b838adfc959384b7e05df83cbf240e3b5018c84f3f1d648a",
        "79005978f7d177e96fbaf8025fe572b775943bfe7de9ba8cf67938e51090c16c"),
    "DP|2027|None|1.3094": ("b1ea808f00dca9bc3fc35905d33949f062a842886c9868be448b29c313823e40",
        "98a217e668f7cd1e957f28b2819250ad4141a3d0ea5861d1c662705047f0c207"),
    "DP|2027|True|0.5642": ("bea8ef83c8c4aae26124adc6734fa961ecfeb0a80c0dd094f47aec3eac66f5df",
        "a7f387d369fbac51a887a8e5625a8fcb23ef903e6d5e9a51ec705d1850b5f472"),
    "DP|2027|True|1.3094": ("638a7d59dbc7da01894823b54ccec93d58736b1a3e8e07516be6e397018b12e8",
        "cb1656fa8366769359c08c14d5b8a6906c888f89a389d95a7fd2431c1762088c"),
    "FA_1024|2026|None|None": ("a1ac4b6e425164684060aa9b1ccdc4de4563e40a4a8af7218d590d989d5a0915",
        "10d828d10d6d4a1208194c0ce8ca5109ce1e137ddbdee175de4a5f25688459be"),
    "FA_1024|2026|True|None": ("0b0bca4fd18bb217bf8a340bc41503872e2639c0448460efc3c148a4bae7b0d5",
        "28226a1de2fb6311304dc112cb1e10f0178a99d982d1d8f50313ae19e6dc79c1"),
    "FA_1024|2027|None|None": ("e3770d460ac7095e140016ae3af84ca291faf4034e147f185d6dd62371be0c7d",
        "3ec9dad8ed3cb57c53f456da3e49a70f110a6b72d9986557105bf7c22b22088e"),
    "FA_1024|2027|True|None": ("ef127fcf7ab3de72b2d21571685f95a7cb528d176d7bf8448778e58c9002ef3d",
        "53ae6905fe61780c1868ed782691ba5a475d6f8be39503b89116b8b60c3ef5c1"),
    "FA|2026|None|None": ("c2951ecb8342ff8982e6a12186c6a4e56f9be8283675692551a909559c803744",
        "6cf22edf1582ce0b827a81b64ab96e30755e337c09a6f0946b94c1dcde802996"),
    "FA|2026|True|None": ("fc715be7b951108a55e7f55968d0e9bf2851cb8878d6542b174d90e2b6ae3f88",
        "66770288acf78c816e98bf65c78896acb4d16b41767ef6c7809af1fe13c5c74c"),
    "FA|2027|None|None": ("a1faeb9c1a631a80fa91e7792e2b70144f1bc6b5c363e431278508dd1f8aa88a",
        "c6fbdca2583aa25ce850976a2f9b1e2754b31368d7c6e9175cf44cb4b05209a6"),
    "FA|2027|True|None": ("baa83e178d2e47a5aa1a6eef415e16df515a940ad23b6df00ae25b67c96c6870",
        "e93fb19531d95b2d6267073020c10f44f7c3cc8e63afc79923a9f5ca86a94c99"),
    "FP|2026|None|None": ("a07722e3cf33b0683bb3da6a2a08af4c5a5e247cd64d7751940b24b68da2f288",
        "dac992feb1ba6adb4305efb7738fdf359c771622cb0388bc2404ec45aaf64d35"),
    "FP|2026|True|None": ("0fe231f1962b04f266800d6d66152d5382e42832d6c6606be7ffa9992e626aeb",
        "bc8689f3119d40e64b4bfd8c707c01473aa5043a2498f026a8e7c131165cf880"),
    "FP|2027|None|None": ("9a4137ca31fc9d95c223d221c34a3859f123db6fa4717a8830030f7d175fc3ad",
        "64752dbbf263b1ababe3621f547709a0a6ac28fe814621213ffc8117e5c85382"),
    "FP|2027|True|None": ("b0f41ecfde3b5de8b529fa7b26a60f1e90abb7b8044093cb6f5b8e0e502883a3",
        "b39fef982023eb15eeec0318e096e97605bf97efb9eb264b677bded8102a3ffb"),
    "PF|2026|None|None": ("2b289317a06bb5293298ed171c72cb381e1a98df536400e06d00c9ebb3281b2d",
        "85adf928b7a6fab4d5ca1355b5a7ebcf577c7873218fff8a97303ca45138af0f"),
    "PF|2026|True|None": ("cb06f821e4a190523af89b45416f09fdd8057404006cfcf1b2024c44e4b168b2",
        "11ca4fbc8dd6f072c18dfda3a14623eed09aee2721c9fc7aae384169fdcb1363"),
    "PF|2027|None|None": ("b331cd84552af85d29d1789e1346f0640fefc48c312a63fdcf0c668823b4e9df",
        "dc337dd6438ae48299023c2ca7ee780739c34878c2375e32f2f64410c4cefc11"),
    "PF|2027|True|None": ("73c4ad734c2be4065179f4ea792f13cd8b8fa15a6d248b53a1dc4c1a746de89e",
        "cfa2f5a0957cd3c2bf5a025c4d64b1e7675683f2bdbf0b807b1b6ff8aad062fc"),
    "S_CAL|2026|None|None": ("41f1a8f02095d2d5bbbb43828c16102be567b8851d3cd804ec4e7ff0c5f0a1a1",
        "997fc3076147d14717d09f12d97f147ef84bcf5fc5df07b5743d8077d038b421"),
    "S_CAL|2026|True|None": ("757505d8c28957fefd95c87f64b3d5adee236c36eb06a410173809fe53e4efd2",
        "8fb83823caeb687137c9b221d50228366a3ad45c7564b44b2a5f403106e727b3"),
}


def _combos():
    for arm in ("DP", "FA", "FA_1024", "FP", "PF", "S_CAL"):
        for seed in (2026, 2027):
            if arm == "S_CAL" and seed != 2026:
                continue
            for fused in (True, None):
                for z in ((0.5642, 1.3094) if arm == "DP" else (None,)):
                    kw = {}
                    if arm in ("FA_1024", "DP"):
                        kw["clip_norm"] = 1.2345
                    if arm == "DP":
                        kw["z"] = z
                    yield f"{arm}|{seed}|{fused}|{z}", arm, seed, fused, kw


# ------------------------------------------------------------------------------------------------ item 1: digests
def test_digests_identical_to_pre_fl05_code_at_defaults():
    seen = 0
    for tag, arm, seed, fused, kw in _combos():
        for extra in ({}, {"lr_mult": 1.0, "role": "MAIN"}, {"lr_mult": 1}):
            s = F.arm_spec(arm, seed=seed, peak_lr=PEAK_REG, fused=fused, **kw, **extra)
            got = (F.spec_digest(s), F.fl_run_config(s, f"FL_{arm}_SMALL_s{seed}", 64).digest())
            assert got == BASELINE[tag], (tag, extra)
        seen += 1
    assert seen == len(BASELINE) == 26


def test_lr_mult_scales_the_local_lr_only():
    for m in (0.5, 2.0):
        for arm, kw in (("FA", {}), ("FP", {}), ("PF", {}), ("FA_1024", {"clip_norm": 1.2345}),
                        ("DP", {"clip_norm": 1.2345, "z": 0.5642}), ("S_CAL", {}), ("FA_Q8", {})):
            base = F.arm_spec(arm, seed=2026, peak_lr=PEAK_REG, fused=True, **kw)
            s = F.arm_spec(arm, seed=2026, peak_lr=PEAK_REG, fused=True, lr_mult=m, **kw)
            assert s["cfg"]["lr_peak_local"] == m * PEAK_REG == s["cfg"]["solver"]["lr"]
            assert s["lr_mult"] == m and s["lr_peak_base"] == PEAK_REG
            assert s["cfg"]["pf"] == base["cfg"]["pf"] and s["cfg"]["mu"] == base["cfg"]["mu"]
            assert s["cfg"]["dp"] == base["cfg"]["dp"] and s["plan"] == base["plan"]
            assert s["max_rounds"] == base["max_rounds"] and s["knobs"] == base["knobs"]
            if arm == "PF":
                assert s["cfg"]["pf"]["lr"] == PEAK_REG                          # personal LR never multiplied
            assert F.spec_digest(s) != F.spec_digest(base)
            assert F.fl_run_config(s, "x", 64).digest() != F.fl_run_config(base, "x", 64).digest()
    for bad in (1.5, 0.0, 3, -1, "two"):
        assert_raises(F.LaunchRefused, F.arm_spec, "FA", seed=2026, peak_lr=PEAK_REG, lr_mult=bad)


def test_fa_q8_spec_is_fa_plus_the_codec():
    fa = F.arm_spec("FA", seed=2026, peak_lr=PEAK_REG, fused=True)
    q8 = F.arm_spec("FA_Q8", seed=2026, peak_lr=PEAK_REG, fused=True)
    assert q8["cfg"] == fa["cfg"] and q8["plan"] == fa["plan"] and q8["max_rounds"] == fa["max_rounds"]
    assert q8["upload_codec"]["name"] == "Q8_SR_PER_TENSOR" and fa["upload_codec"] is None
    assert F.spec_digest(q8) != F.spec_digest(fa)
    assert F.METHOD_CODE["FA_Q8"] == "FA" and F.assert_arm_allowed("FA_Q8", "SASREC")["method_code"] == "FA"
    # disclosed: with the same run id the fl RunConfig is FA's (the codec lives in the arm spec / ARM_SPEC.json)
    assert F.fl_run_config(q8, "x", 64).digest() == F.fl_run_config(fa, "x", 64).digest()


# ------------------------------------------------------------------------------------------------ item 2: CALIBRATION
def test_calibration_role_spec_and_refusals():
    assert F.fa_rounds(32768) == 1707 == F.CALIB["rounds"] and F.fa_rounds(131072) == 6827 == F.KNOBS["rounds"]
    c = F.arm_spec("FA", seed=2026, peak_lr=PEAK_REG, fused=True, role="CALIBRATION", lr_mult=0.5)
    fa = F.arm_spec("FA", seed=2026, peak_lr=PEAK_REG, fused=True)
    assert c["cfg"]["endpoint_rounds"] == c["max_rounds"] == 1707 and c["role"] == "CALIBRATION"
    drop = ("endpoint_rounds", "lr_peak_local", "solver")
    assert {k: v for k, v in c["cfg"].items() if k not in drop} == {k: v for k, v in fa["cfg"].items() if k not in drop}
    c1 = F.arm_spec("FA", seed=2026, peak_lr=PEAK_REG, fused=True, role="CALIBRATION")
    assert F.spec_digest(c1) != F.spec_digest(fa)
    for arm, kw in (("FP", {}), ("PF", {}), ("FA_1024", {"clip_norm": 1.0}), ("DP", {"clip_norm": 1.0, "z": 0.5642}),
                    ("S_CAL", {}), ("FA_Q8", {})):
        assert_raises(F.LaunchRefused, F.arm_spec, arm, seed=2026, peak_lr=PEAK_REG, role="CALIBRATION", **kw)
    assert_raises(F.LaunchRefused, F.arm_spec, "FA", seed=2027, peak_lr=PEAK_REG, role="CALIBRATION")
    assert_raises(F.LaunchRefused, F.arm_spec, "FA", seed=2026, peak_lr=PEAK_REG, role="TEST")
    F.fl_run_config(c, "x", 64)                                                   # fl.runtime accepts it


def _args(tmp_path, *extra, smoke=True, arm="FA", seed="2026"):
    a = ["--arm", arm, "--seed", seed, "--gpu", "0", "--run-dir", str(tmp_path / "r"),
         "--history-policy", "USER_HISTORY_L50"] + list(extra)
    if smoke:
        a.append("--smoke")
    return F.parse_args(a)


def test_preflight_fl05_refusals_and_run_ids(tmp_path):
    g = gate(tmp_path)

    def pf(a):
        return F.preflight(a, runner=fake_runner(), environ={}, gate_path=g)
    assert pf(_args(tmp_path))["fl05"] == {"role": "MAIN", "lr_mult": None, "lr_selection": None}
    assert pf(_args(tmp_path, "--lr-mult", "2", "--role", "calibration"))["fl05"]["role"] == "CALIBRATION"
    sel = tmp_path / "LR_SELECTION.json"
    for a in (_args(tmp_path, "--lr-mult", "1.5"), _args(tmp_path, "--lr-mult", "1", "--lr-selection", str(sel)),
              _args(tmp_path, "--role", "CALIBRATION", arm="FP"),
              _args(tmp_path, "--role", "CALIBRATION", seed="2027"),
              _args(tmp_path, "--role", "CALIBRATION", "--lr-selection", str(sel)), _args(tmp_path, "--role", "X"),
              _args(tmp_path, "--role", "CALIBRATION", arm="FA_Q8")):
        assert_raises(F.LaunchRefused, pf, a)

    def rid(*e, **k):
        return F.default_run_id(_args(tmp_path, *e, **k))
    assert rid(smoke=False) == "FL_FA_SMALL_s2026" and rid("--lr-mult", "2", smoke=False) == "FL_FA_SMALL_s2026"
    assert rid("--epsilon", "8", arm="DP", smoke=False) == "FL_DP_eps8_SMALL_s2026"
    assert rid(arm="FA_Q8", seed="2027", smoke=False) == "FL_FA_Q8_SMALL_s2027"
    assert rid("--role", "CALIBRATION", "--lr-mult", "0.5", smoke=False) == "FL_FA_LRSCREEN_x0.5_CALIB_s2026"
    assert rid("--role", "CALIBRATION", smoke=False) == "FL_FA_LRSCREEN_x1_CALIB_s2026"
    assert rid().startswith("SMOKE_NOT_EVIDENCE__FL_FA_SMALL")


# ------------------------------------------------------------------------------------------------ engine (tiny)
def fl05_world(n=10, seed=5, eval_id="M02_VALIDATION_WARM"):
    """test_int03's tiny world, with rows shaped like rows_from_result (aggregation / denominator / metric / k) whose
    E2E macro mrr@20 value equals the returned metric."""
    server = make_tiny_adapter(seed=1)
    worker = make_tiny_adapter(seed=1)
    clients = {c.key: c for c in make_clients(n, seed=seed)}
    from fl.client import valid_rows
    nv = {k: int(valid_rows(c.examples).numel()) for k, c in clients.items()}

    def evaluate(mark):
        v = float(abs(sum(p.double().sum() for p in server.shared_parameters())) % 1.0)
        rows = []
        for agg in ("micro", "macro"):
            for den in ("E2E", "RANKABLE"):
                for met, k in (("mrr", 20), ("hr", 10)):
                    main = (agg, den, met, k) == ("macro", "E2E", "mrr", 20)
                    rows.append({"aggregation": agg, "target_denominator": den, "metric": met, "k": k,
                                 "value": v if main else v / 3, "value_status": "OK", "split": "VALIDATION",
                                 "eval_manifest_id": eval_id, "method": "C", "training_paradigm": "CENTRAL",
                                 "row_status": "CENTRAL_CONTEST", "run_label": "PRODUCTION"})
        return v, rows, np.arange(5), {"E2E": {"n": 5}}
    return dict(server_adapter=server, init_sha256=state_digest(server.broadcast_state()), workers=[worker],
                keys=sorted(clients), n_valid_by_key=nv, get_client=clients.__getitem__,
                n_decisions=sum(nv.values()), manifest_hash="tiny-manifest", evaluate=evaluate)


def tiny_spec(arm, rounds=60, group=4, **kw):
    s = F.arm_spec(arm, seed=kw.pop("seed", 2026), peak_lr=PEAK, **kw)
    s["cfg"]["endpoint_rounds"] = rounds
    s["max_rounds"] = min(rounds, s["max_rounds"])
    if s["plan"]["sampling"] == "sweep":
        s["plan"]["group_size"] = group
    return s


def run_tiny(tmp_path, spec, name="r", world=None, **kw):
    w = world or fl05_world()
    return F.execute(spec=spec, run_id=name, run_dir=tmp_path / name, say=lambda *_: None, **w, **kw)


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def test_calibration_rows_labelled_and_lr_mult_recorded(tmp_path):
    out = run_tiny(tmp_path, tiny_spec("FA", role="CALIBRATION", lr_mult=2.0), "c",
                   world=fl05_world(eval_id="M06_CALIB_VALIDATION_WARM"))
    assert out["status"] == "DONE_MAIN"
    rows = _json(tmp_path / "c" / "RESULT_ROWS.json")["rows"]
    assert rows and all(r["role"] == r["manifest_role"] == "CALIBRATION" and r["run_label"] == "FL_LR_SCREEN"
                        and r["row_status"] == "FL_LR_SCREEN" and r["fl_lr_mult"] == 2.0 and r["method"] == "FA"
                        for r in rows)
    s = _json(tmp_path / "c" / "RUN_SUMMARY.json")
    assert [e[0] for e in s["evals"]] == list(range(1, 13))
    assert s["arm_spec"]["lr_mult"] == 2.0 and s["arm_spec_sha256"] == F.spec_digest(s["arm_spec"])
    side = _json(tmp_path / "c" / "ARM_SPEC.json")
    assert side["arm_spec_sha256"] == s["arm_spec_sha256"] and side["arm"] == "FA"
    main = run_tiny(tmp_path, tiny_spec("FA"), "m")
    mrows = _json(tmp_path / "m" / "RESULT_ROWS.json")["rows"]
    assert all(r["row_status"] == "DOWNSTREAM_ARM" and r["fl_lr_mult"] == 1.0 and "role" not in r for r in mrows)
    assert main["summary"]["server_digest"] != out["summary"]["server_digest"]


def test_fa_q8_engine_bytes_determinism_and_differs_from_fa(tmp_path):
    from fl.quant import q8_upload_bytes
    a = run_tiny(tmp_path, tiny_spec("FA_Q8", rounds=12), "a")
    b = run_tiny(tmp_path, tiny_spec("FA_Q8", rounds=12), "b")
    fa = run_tiny(tmp_path, tiny_spec("FA", rounds=12), "fa")
    assert a["status"] == "DONE_MAIN"
    assert a["summary"]["server_digest"] == b["summary"]["server_digest"] != fa["summary"]["server_digest"]
    rows = _json(tmp_path / "a" / "RESULT_ROWS.json")["rows"]
    assert rows and all(r["method"] == "FA_Q8" and r["method_code"] == "FA" for r in rows)
    s = _json(tmp_path / "a" / "RUN_SUMMARY.json")
    man = make_tiny_adapter(seed=1).manifest
    up, down = q8_upload_bytes(man), man.shared_bytes
    bv = s["bytes"]["per_visit"]
    assert bv["upload_bytes"] == up == man.shared_numel + 4 * len(man.shared_keys) + 8 and bv["download_bytes"] == down
    surv = s["participation"]["survived"]
    assert s["bytes"]["fleet"]["bytes_up"] == surv * up and s["bytes"]["fleet"]["bytes_down"] == surv * down
    assert s["bytes"]["fleet"]["total"] == surv * (up + down)
    assert _json(tmp_path / "fa" / "RUN_SUMMARY.json")["bytes"]["fleet"]["bytes_up"] == surv * (down + 8)


def test_fa_q8_resume_exact_and_arm_spec_guard(tmp_path):
    straight = run_tiny(tmp_path, tiny_spec("FA_Q8", rounds=12), "s")
    part = run_tiny(tmp_path, tiny_spec("FA_Q8", rounds=12), "p", interrupt_at=5)
    assert part["status"] == "INTERRUPTED"
    assert_raises(F.LaunchRefused, run_tiny, tmp_path, tiny_spec("FA", rounds=12), "p", resume=True)
    assert_raises(F.LaunchRefused, run_tiny, tmp_path, tiny_spec("FA_Q8", rounds=12, lr_mult=2.0), "p", resume=True)
    res = run_tiny(tmp_path, tiny_spec("FA_Q8", rounds=12), "p", resume=True)
    for k in ("server_digest", "evals", "ledger", "counters"):
        assert straight["summary"][k] == res["summary"][k], k
    run_tiny(tmp_path, tiny_spec("FA_Q8", rounds=12), "q", interrupt_at=5)
    (tmp_path / "q" / "ARM_SPEC.json").unlink()
    assert_raises(F.LaunchRefused, run_tiny, tmp_path, tiny_spec("FA_Q8", rounds=12), "q", resume=True)
    fresh = run_tiny(tmp_path, tiny_spec("FA_Q8", rounds=12), "e", resume=True)        # empty dir: starts fresh
    assert fresh["summary"]["server_digest"] == straight["summary"]["server_digest"]


def test_lr_mult_changes_the_trajectory_only_through_the_lr(tmp_path):
    a = run_tiny(tmp_path, tiny_spec("FP", rounds=12), "a")
    b = run_tiny(tmp_path, tiny_spec("FP", rounds=12, lr_mult=1.0), "b")
    c = run_tiny(tmp_path, tiny_spec("FP", rounds=12, lr_mult=0.5), "c")
    assert a["summary"]["server_digest"] == b["summary"]["server_digest"] != c["summary"]["server_digest"]
    assert a["summary"]["arm_spec_sha256"] == b["summary"]["arm_spec_sha256"]
    ta, tc = a["summary"]["lr_table"], c["summary"]["lr_table"]
    assert tc["final_lr"] == pytest.approx(0.5 * ta["final_lr"], rel=1e-15) and ta["E_total"] == tc["E_total"]


# ------------------------------------------------------------------------------------------------ negative controls
@nc("hashing the spec WITH the FL-05 keys at their defaults changes every registered digest")
def test_nc_naive_digest_changes_registered_digests():
    s = F.arm_spec("FA", seed=2026, peak_lr=PEAK_REG, fused=True)
    naive = hashlib.sha256(json.dumps(s, sort_keys=True, default=str).encode()).hexdigest()   # MUTANT: no drop rule
    assert naive == BASELINE["FA|2026|True|None"][0]


@nc("fl.runtime alone (config digest, same run id) accepts an FA_Q8 checkpoint resumed as FA: ARM_SPEC.json is needed")
def test_nc_runtime_alone_refuses_cross_arm_resume(tmp_path):
    from fl.checkpoint import CheckpointError, CheckpointManager
    from fl.participation import ParticipationPlan
    from fl.runtime import FLRun
    from fl.server import Server
    run_tiny(tmp_path, tiny_spec("FA_Q8", rounds=12), "p", interrupt_at=5)
    w = fl05_world()
    spec = tiny_spec("FA", rounds=12)
    plan = ParticipationPlan.build(list(w["keys"]), seed=2026, manifest_hash=w["manifest_hash"],
                                   group_size=spec["plan"]["group_size"], sampling=spec["plan"]["sampling"])

    def resume():
        FLRun.resume(F.fl_run_config(spec, "p", 64), plan, Server(w["server_adapter"], init_sha256=w["init_sha256"]),
                     w["get_client"], w["n_decisions"], workers=w["workers"],
                     ckpt=CheckpointManager(tmp_path / "p" / "ckpt", "p"), n_valid_by_key=w["n_valid_by_key"])
    assert_raises((CheckpointError, F.LaunchRefused), resume)
