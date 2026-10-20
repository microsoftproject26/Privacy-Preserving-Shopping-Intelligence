"""The launcher refuses fail-closed: without the G1 approval file (non-smoke), without an explicit GPU, with a foreign
GPU process (mocked nvidia-smi), on FULL_TRAIN, on out-of-range smoke options; it exports the GPU and the T3 CUBLAS
value before any heavy import, and it never writes company/gates/G1_APPROVED.json."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import v4int.launch_central as L
from intkit import assert_raises, nc
from v4int.smoke import smoke_cap_before_first_mark

CODE = Path(__file__).resolve().parents[1]
BUS = {"0": "00000000:46:00.0", "1": "00000000:E8:00.0"}


def fake_runner(apps=(), mem=None, fail=False, rc=0):
    mem = mem or {"0": 0.0, "1": 0.0}

    def run(argv, **kw):
        if fail:
            raise FileNotFoundError("nvidia-smi")
        q = argv[1]
        if q.startswith("--query-gpu=index,pci.bus_id"):
            out = "\n".join(f"{i}, {b}" for i, b in BUS.items())
        elif q.startswith("--query-compute-apps"):
            out = "\n".join(f"{b}, {pid}" for b, pid in apps)
        elif q.startswith("--query-gpu=index,uuid"):
            out = "\n".join(f"{i}, GPU-{i}, NVIDIA A16, {mem[i]}, 15356, 0" for i in BUS)
        else:
            raise AssertionError(f"unexpected nvidia-smi query {argv}")
        return subprocess.CompletedProcess(argv, rc, out + "\n", "")
    return run


def args(tmp_path, *extra, smoke=True, phase="CALIBRATION", gpu="0"):
    a = ["--family", "GRU", "--seed", "2026", "--phase", phase, "--peak-lr", "1e-3", "--run-dir", str(tmp_path / "r"),
         "--history-policy", "SESSION_L50"]
    if gpu is not None:
        a += ["--gpu", gpu]
    if smoke:
        a += ["--smoke", "--checkpoint-every", "300"]
    return L.parse_args(a + list(extra))


def gate(tmp_path) -> Path:
    p = tmp_path / "G1_APPROVED.json"
    p.write_text(json.dumps({"gate": "G1", "note": "test fixture only"}), encoding="utf-8")
    return p


def test_the_real_g1_file_does_not_exist():
    """Pre-G1 this asserted the file's absence. TRAIN-03: Ops wrote company/gates/G1_APPROVED.json at G1 (16:09Z,
    2026-09-27); the invariant that stays is that NO test / launcher writes it: when present it is Ops' strict-JSON
    G1 record (gate G1), and the suite's run_all checks it is byte-unchanged by the run."""
    if L.G1_GATE.exists():
        import json as _json
        obj = _json.loads(L.G1_GATE.read_text(encoding="utf-8"))
        assert obj.get("gate") == "G1" and obj.get("document") == "G1_APPROVED"


def test_non_smoke_refused_without_g1_file(tmp_path):
    a = args(tmp_path, smoke=False)
    assert_raises(L.LaunchRefused, L.preflight, a, runner=fake_runner(), environ={}, gate_path=tmp_path / "none.json")
    if not L.G1_GATE.exists():                                    # the real path, while G1 is not approved
        assert_raises(L.LaunchRefused, L.preflight, a, runner=fake_runner(), environ={})
    bad = tmp_path / "bad.json"
    bad.write_text("{NaN}", encoding="utf-8")
    assert_raises(L.LaunchRefused, L.preflight, a, runner=fake_runner(), environ={}, gate_path=bad)


def test_non_smoke_passes_preflight_with_g1_file_and_a_free_gpu(tmp_path):
    rec = L.preflight(args(tmp_path, smoke=False), runner=fake_runner(), environ={}, gate_path=gate(tmp_path))
    assert rec["g1_gate"]["required"] and len(rec["g1_gate"]["sha256"]) == 64
    assert rec["gpu_check"]["gpu_ready"] == "ok"
    main = args(tmp_path, smoke=False, phase="MAIN")
    assert_raises(L.LaunchRefused, L.preflight, main, runner=fake_runner(), environ={}, gate_path=gate(tmp_path))
    ok = args(tmp_path, "--calibration-record", str(tmp_path / "CAL.json"), smoke=False, phase="MAIN")
    L.preflight(ok, runner=fake_runner(), environ={}, gate_path=gate(tmp_path))


def test_smoke_needs_no_g1_file(tmp_path):
    rec = L.preflight(args(tmp_path), runner=fake_runner(), environ={}, gate_path=tmp_path / "none.json")
    assert rec["g1_gate"] == {"required": False, "reason": "smoke (SMOKE_NOT_EVIDENCE)"}


def test_explicit_gpu_required(tmp_path):
    with pytest.raises(SystemExit):
        args(tmp_path, gpu=None)
    assert_raises(L.LaunchRefused, L.preflight, args(tmp_path, gpu="-1"), runner=fake_runner(), environ={})


def test_foreign_gpu_process_refused(tmp_path):
    a = args(tmp_path)
    assert_raises(L.LaunchRefused, L.preflight, a, runner=fake_runner(apps=[(BUS["0"], "4242")]), environ={})
    assert_raises(L.LaunchRefused, L.preflight, a, runner=fake_runner(apps=[("00000000:99:00.0", "7")]), environ={})
    assert_raises(L.LaunchRefused, L.preflight, a, runner=fake_runner(fail=True), environ={})
    assert_raises(L.LaunchRefused, L.preflight, a, runner=fake_runner(rc=9), environ={})
    assert_raises(L.LaunchRefused, L.preflight, a, runner=fake_runner(mem={"0": 2400.0, "1": 0.0}), environ={})
    L.preflight(a, runner=fake_runner(apps=[(BUS["1"], "4242")], mem={"0": 0.0, "1": 5000.0}), environ={})


def test_other_refusals(tmp_path):
    fr = fake_runner()
    for extra in (("--population", "FULL_TRAIN"), ("--smoke-updates", "601"), ("--smoke-updates", "0"),
                  ("--smoke-interrupt-at", "250"), ("--smoke-interrupt-at", "600"), ("--continue-run",),
                  ("--resume", "--continue-run")):
        assert_raises(L.LaunchRefused, L.preflight, args(tmp_path, *extra), runner=fr, environ={})
    L.preflight(args(tmp_path, "--smoke-interrupt-at", "300"), runner=fr, environ={})
    assert_raises(L.LaunchRefused, L.preflight, args(tmp_path), runner=fr, environ={"CUDA_VISIBLE_DEVICES": "1"})
    L.preflight(args(tmp_path), runner=fr, environ={"CUDA_VISIBLE_DEVICES": "0"})
    assert_raises(L.LaunchRefused, L.preflight, args(tmp_path, "--smoke-interrupt-at", "300", smoke=False),
                  runner=fr, environ={}, gate_path=gate(tmp_path))


def test_export_environment_before_imports():
    env = {}
    rec = L.export_environment(1, env)
    lock = json.loads(L.PREG1_LOCK.read_text(encoding="utf-8"))
    assert env["CUDA_VISIBLE_DEVICES"] == "1" and env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["CUBLAS_WORKSPACE_CONFIG"] == lock["T3_eval_numerics"]["flags"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert rec["CUDA_VISIBLE_DEVICES"] == "1"
    assert_raises(L.LaunchRefused, L.export_environment, 0, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"})


def test_data_paths_and_manifest_ids_are_not_hard_coded(tmp_path):
    from v4int.verify import DataPaths
    a = args(tmp_path)
    assert L.data_paths(a) == DataPaths.default() and a.train_manifest is None and a.eval_manifest is None
    b = args(tmp_path, "--manifest-dir", str(tmp_path / "m"), "--lock-path", str(tmp_path / "l.json"),
             "--train-manifest", "MX_TRAIN", "--eval-manifest", "MX_EVAL", "--population-label", "HISTORICAL_POPULATION")
    p = L.data_paths(b)
    assert p.manifest_dir == tmp_path / "m" and p.lock_path == tmp_path / "l.json"
    assert p.manifest_hashes_path == DataPaths.default().manifest_hashes_path
    assert (b.train_manifest, b.eval_manifest, b.population_label) == ("MX_TRAIN", "MX_EVAL", "HISTORICAL_POPULATION")
    src = (CODE / "v4int" / "launch_central.py").read_text(encoding="utf-8")
    for const in ("M01_COHORT_TRAIN", "M05_CALIB_TRAIN", "M02_VALIDATION_WARM", "M06_CALIB_VALIDATION_WARM",
                  "3906929", "3_906_929", "990362", "990_362"):
        assert const not in src, const
    for mod in ("providers.py", "verify.py", "cache.py"):
        src = (CODE / "v4int" / mod).read_text(encoding="utf-8").split('"""', 2)[-1]       # code, not the docstring
        for const in ("3906929", "3_906_929", "990362", "990_362", '"M01_COHORT_TRAIN"', '"M05_CALIB_TRAIN"'):
            assert const not in src, (mod, const)


def test_determinism_constant_is_the_proposed_lock_value():
    assert L.DETERMINISM_MODE == "STRICT"


def test_smoke_cap_is_below_the_first_mark():
    assert smoke_cap_before_first_mark(600, 256, 990_362)                   # M05: 153,600 < 495,181
    assert smoke_cap_before_first_mark(600, 256, 3_906_929)
    assert not smoke_cap_before_first_mark(600, 256, 300_000)
    assert not smoke_cap_before_first_mark(0, 256, 990_362)


def _cli(tmp_path, *a):
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-B", str(CODE / "v4int" / "launch_central.py"), *a], cwd=str(CODE),
                          env=env, capture_output=True, text=True, timeout=300)


def test_cli_refuses_without_g1_and_without_gpu(tmp_path):
    base = ["--family", "GRU", "--seed", "2026", "--phase", "CALIBRATION", "--peak-lr", "1e-3",
            "--run-dir", str(tmp_path / "run"), "--history-policy", "SESSION_L50"]
    g1_before = L.G1_GATE.read_bytes() if L.G1_GATE.exists() else None
    p = _cli(tmp_path, *base, "--gpu", "0")
    # before G1: refused for the missing G1 file; after G1 (TRAIN-03 note): refused fail-closed further on (the GPU
    # ownership / nvidia-smi check on this host) — never launched either way
    assert p.returncode == 2 and "LAUNCH REFUSED" in p.stderr, p.stderr[-800:]
    if g1_before is None:
        assert "G1" in p.stderr, p.stderr[-800:]
    p = _cli(tmp_path, *base)
    assert p.returncode == 2 and "--gpu" in p.stderr, p.stderr[-800:]
    assert not (tmp_path / "run").exists()
    assert (L.G1_GATE.read_bytes() if L.G1_GATE.exists() else None) == g1_before, "the launcher never writes G1"


@nc("a launcher that skipped the GPU ownership check would accept a GPU with a foreign compute process")
def test_nc_foreign_process_would_be_missed_without_the_check(tmp_path, monkeypatch):
    import v4sup.gpu as G
    monkeypatch.setattr(G, "gpu_ready", lambda needed, runner=None: (True, "ok (check disabled)"))
    assert_raises(L.LaunchRefused, L.preflight, args(tmp_path), runner=fake_runner(apps=[(BUS["0"], "4242")]),
                  environ={})


def _two_host_roots(tmp_path, ctx):
    """Two byte-identical copies of REGISTRY_V2.json under two DIFFERENT absolute roots (simulating this workspace
    read locally, e.g. F:\\...\\ppsi_v4, vs. on the server, /home/jovyan/.../ppsi_v4): same relative subpath, same
    bytes, different absolute prefix — exactly the host/root difference F2 must be independent of."""
    rel = Path(ctx["manifest_registry_rel_path"])
    roots = (tmp_path / "host_a_windows_style_root", tmp_path / "an" / "entirely" / "different" / "host_b_server_root")
    for root in roots:
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(ctx["manifest_registry_path"].read_bytes())
    return roots, rel


def test_f2_trainer_view_hash_is_root_independent(tmp_path):
    """INT-02R1 F2: build_trainer_registry_view's own self_sha256 must not encode an absolute path (derived_from
    carries the LOCK's relative registry path, never the resolved absolute one), so the SAME REGISTRY_V2 content
    produces the SAME view hash whether read from a local workspace root or the server's."""
    from v4int.verify import DataPaths
    paths = DataPaths.default()
    ctx = L.bind_data_protocol_lock_v2(paths, smoke=True)
    (root_a, root_b), rel = _two_host_roots(tmp_path, ctx)
    v1 = L.build_trainer_registry_view(root_a / rel, tmp_path / "run1",
                                       registry_rel_path=ctx["manifest_registry_rel_path"],
                                       registry_file_sha256=ctx["manifest_registry_file_sha256"])
    v2 = L.build_trainer_registry_view(root_b / rel, tmp_path / "run2",
                                       registry_rel_path=ctx["manifest_registry_rel_path"],
                                       registry_file_sha256=ctx["manifest_registry_file_sha256"])
    assert v1["self_sha256"] == v2["self_sha256"]
    assert v1["path"] != v2["path"]                          # the (host-specific) file location still differs
    view = json.loads(v1["path"].read_text(encoding="utf-8"))
    assert view["derived_from"]["path"] == ctx["manifest_registry_rel_path"] and "/" in view["derived_from"]["path"]
    assert not Path(view["derived_from"]["path"]).is_absolute()
    assert view["derived_from"]["file_sha256"] == ctx["manifest_registry_file_sha256"]
    assert view["derived_from"]["self_sha256"] == ctx["manifest_registry_self_sha256"]


@nc("without F2, the view's self_sha256 would vary with the absolute workspace root")
def test_nc_without_f2_the_view_hash_would_vary_with_the_root(tmp_path):
    from v4int.verify import DataPaths
    paths = DataPaths.default()
    ctx = L.bind_data_protocol_lock_v2(paths, smoke=True)
    (root_a, root_b), rel = _two_host_roots(tmp_path, ctx)

    def buggy_view(manifest_registry_path):                # the pre-F2 behaviour: the ABSOLUTE path, no file_sha256
        from v4train.registry import with_self_sha256
        reg = json.loads(Path(manifest_registry_path).read_text(encoding="utf-8"))
        roles_view = {r: dict(reg["roles"][r]) for r in ("MAIN", "CALIBRATION") if r in reg["roles"]}
        needed = {mid for pair in roles_view.values() for mid in pair.values()}
        obj = with_self_sha256({"data_version": reg["data_version"], "roles": roles_view,
                                "manifests": {mid: dict(reg["manifests"][mid]) for mid in needed},
                                "derived_from": {"path": str(manifest_registry_path), "self_sha256": reg.get("self_sha256")}})
        return obj["self_sha256"]
    a = buggy_view(root_a / rel)
    b = buggy_view(root_b / rel)
    assert a == b          # FAILS: the buggy view's hash differs because the absolute path differs
