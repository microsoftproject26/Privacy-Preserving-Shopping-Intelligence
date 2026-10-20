"""TRAIN-03 — the launcher side of the P3 queue: the p3_driver command line (MAIN, --recipe-id, from a --code-root)
parses and passes the preflight of a non-smoke launch; a code root without the launcher is refused; a MAIN launch with
--continuation-budget and a tuned recipe is refused by the RunConfig rule (automatic C3 continuation, no budget file)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import v4int.launch_central as L
import v4train.p3_driver as P3
from intkit import assert_raises
from test_launcher import fake_runner, gate


def _launch_fixture(run_id="fixture", config_sha="cfg"):
    args = SimpleNamespace(smoke=False, population_label="COHORT", history_policy="USER_HISTORY_L50",
                           train_manifest=None, eval_manifest=None, gpu=0, resume=False, continue_run=False)
    lock = {"lock_path": "lock.json", "lock_self_sha256": "ls", "lock_file_sha256": "lf",
            "data_version": "dv", "manifest_registry_file_sha256": "mf",
            "manifest_registry_self_sha256": "ms", "schema_id": "sid", "schema_self_sha256": "ss",
            "schema_registry_path": "schema.json", "schema_registry_file_sha256": "sf"}
    kw = dict(args=args, key={"k": 1}, train_id="train", eval_id="eval", want_train="train", want_eval="eval",
              data_paths={"root": "data"}, lockctx=lock, manifest_registry_path="manifest.json",
              trainer_view={"path": "view.json", "self_sha256": "vs", "roles": ["MAIN"]},
              pre={"utc": "2026-09-28T00:00:00Z"}, env_rec={"cuda": "0"}, verified_files=3,
              isolated_modules=4, device_name="cpu-fixture", device_uuid="", device_total_memory=123,
              torch_version="torch-test", numpy_version="numpy-test", python_version="3.13", run_id=run_id,
              config_sha256=config_sha, init_sha256="init", trainer_code_sha="trainer", t_prov=1.234,
              train_provider={"n": 1}, eval_provider={"n": 2}, det_passed="STRICT",
              determinism_at_init={"mode": "STRICT"}, disk={"free": 100}, tuned=None, tuning=None,
              theta0_init_sha256=None, run_label="PRODUCTION", stop_at=0, interrupt_at=0, ce=0)
    return kw


def test_launch_dict_equals_old_construction_fixture():
    kw = _launch_fixture()
    old = {"task": "INT-01 launch_central", "label": "PRODUCTION", "population_label": "COHORT",
           "history_policy": "USER_HISTORY_L50", "comparison_key": {"k": 1},
           "manifests": {"train": "train", "eval": "eval", "trainer_expects": ["train", "eval"],
                         "source": "v4train.provider.expected_manifests (REGISTRY_V2 role)"},
           "data_paths": {"root": "data"},
           "data_protocol_lock_v2": {"path": "lock.json", "lock_self_sha256": "ls", "lock_file_sha256": "lf",
                                     "data_version": "dv", "manifest_registry": {"path": "manifest.json",
                                     "file_sha256": "mf", "self_sha256": "ms"},
                                     "trainer_registry_view": {"path": "view.json", "self_sha256": "vs",
                                                               "roles": ["MAIN"]},
                                     "schema": {"schema_id": "sid", "schema_self_sha256": "ss",
                                                "registry_path": "schema.json", "registry_file_sha256": "sf"}},
           "utc_start": "2026-09-28T00:00:00Z", "preflight": {"utc": "2026-09-28T00:00:00Z"},
           "environment": {"cuda": "0"}, "snapshot": {"verified_files": 3, "isolated_modules": 4},
           "device": {"physical_gpu": 0, "in_process": "cuda:0", "name": "cpu-fixture", "uuid": "",
                      "total_memory_bytes": 123},
           "torch": "torch-test", "numpy": "numpy-test", "python": "3.13", "run_id": "fixture",
           "config_sha256": "cfg", "init_sha256": "init", "trainer_code_sha256": "trainer",
           "providers_build_s": 1.23, "train_provider": {"n": 1}, "eval_provider": {"n": 2},
           "mode": "fresh", "determinism": {"constant": L.DETERMINISM_MODE,
                                           "passed_to_runconfig": "STRICT", "at_trainer_init": {"mode": "STRICT"}},
           "disk_at_start": {"free": 100}, "tuned_recipe": None, "tuning": None, "smoke": None}
    assert json.dumps(L.build_launch_dict(**kw), sort_keys=True, separators=(",", ":")) == json.dumps(
        old, sort_keys=True, separators=(",", ":"))


def test_p3_command_line_parses_and_passes_preflight(tmp_path):
    cmd = P3.launch_cmd(Path(L.CODE), "GRU", 2027, 0, tmp_path / "r", "02B_GRU", history_policy="USER_HISTORY_L50")
    assert Path(cmd[2]) == Path(L.CODE) / "v4int" / "launch_central.py"
    a = L.parse_args(cmd[3:])
    assert (a.phase, a.recipe_id, a.seed, a.gpu, a.calibration_record) == ("MAIN", "02B_GRU", 2027, 0, None)
    rec = L.preflight(a, runner=fake_runner(), environ={}, gate_path=gate(tmp_path))
    assert rec["tuned_recipe"] == {"recipe_id": "02B_GRU"}
    assert "--resume" in P3.launch_cmd(Path(L.CODE), "GRU", 2026, 0, tmp_path / "r", "02B_GRU", resume=True)


def test_p3_driver_refuses_a_code_root_without_the_launcher(tmp_path):
    assert P3.main(["--family", "GRU", "--gpus", "0", "--out-dir", str(tmp_path / "o"), "--code-root",
                    str(tmp_path / "empty")]) == 2


def test_budget_file_with_an_automatic_contest_run_is_refused():
    from intkit import assert_raises as ar  # noqa: F401
    from v4train.recipe import RecipeError, RunConfig
    from v4train.registry import HISTORICAL_REGISTRY_PATH
    c = RunConfig(run_id="x", family="GRU", seed=2026, exposure_point="end", cuda_visible_devices="0",
                  determinism_mode="STRICT", manifest_registry=str(HISTORICAL_REGISTRY_PATH),
                  data_version="CLEAN_V2_HISTORICAL_262144", continuation_budget="b.json")
    assert_raises(RecipeError, c.validate)


def test_sealed_cpu_launcher_boundary_scans_launch_and_streams(tmp_path, monkeypatch, capsys):
    """Exercise launcher dispatch with a CPU trainer adapter; scan the launch boundary as well as run files."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tests_train"))
    from trainkit import art, cfg, scripted, trainer
    from v4train.checkpoint import write_json_atomic
    from v4train.schedule import eval_marks

    rd = tmp_path / "contest"
    vals = {m: round(0.1 + 0.021 * i + 0.000123457, 9)
            for i, m in enumerate(eval_marks("MAIN") + eval_marks("CONTINUATION"))}

    def cpu_run(args, pre, env_rec):
        c = cfg("launcher_cpu_contest", "GRU", auto_continuation=True)
        tr = trainer(rd, c, eval_override=scripted(vals))
        inputs = _launch_fixture(c.run_id, tr.config_sha256)
        inputs.update(pre=dict(pre, utc="2026-09-28T00:00:00Z"), env_rec=env_rec)
        launch = L.build_launch_dict(**inputs)
        lp = L._next_launch_path(rd)
        write_json_atomic(lp, L._json_safe(launch))
        print(f"[launch] {c.run_id}: {lp.name} written")
        assert tr.run()["status"] == "DONE_CONTINUATION"
        return 0

    monkeypatch.setattr(L, "preflight", lambda args: {"status": "CPU_TEST"})
    monkeypatch.setattr(L, "export_environment", lambda gpu: {"device": "cpu"})
    monkeypatch.setattr(L, "run", cpu_run)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    rc = L.main(["--family", "GRU", "--seed", "2026", "--phase", "MAIN", "--recipe-id", "02B_GRU",
                 "--gpu", "0", "--run-dir", str(rd), "--history-policy", "USER_HISTORY_L50"])
    streams = capsys.readouterr()
    assert rc == 0 and (rd / "sealed").is_dir()
    launches = sorted(rd.glob("LAUNCH_*.json"))
    assert launches and json.loads(launches[0].read_text(encoding="utf-8"))["mode"] == "fresh"

    sealed = art(rd)
    values = set()
    for line in (sealed / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for key in ("value", "loss_mean", "grad_norm_mean_preclip", "grad_norm_max_preclip",
                    "head_common_mode_norm"):
            if isinstance(row.get(key), float):
                values.add(row[key])
        values.update(v for v in (row.get("secondary") or {}).values() if isinstance(v, float))
    for p in list((sealed / "evals").glob("*.rows.json")) + [p for p in
            (sealed / "RESULT_ROWS.json", sealed / "RESULT_ROWS_EXTENDED.json") if p.exists()]:
        values.update(row["value"] for row in json.loads(p.read_text(encoding="utf-8"))["rows"]
                      if isinstance(row.get("value"), float))
    values = {v for v in values if len(repr(v)) >= 8}
    assert len(values) > 10
    forbidden = ('"value"', "last_value", "loss_mean", "grad_norm", "improved", "practical_best",
                 "controlled_budget", "secondary", "ndcg", "mrr")
    surfaces = [(p.name, p.read_bytes().decode("latin-1")) for p in launches]
    surfaces += [("stdout", streams.out), ("stderr", streams.err)]
    leaks = []
    for name, content in surfaces:
        for value in values:
            for form in {repr(value), json.dumps(value), f"{value:.6f}"}:
                if re.search(r"(?<![\d.])" + re.escape(form) + r"(?!\d)", content):
                    leaks.append((name, form))
        if name.endswith(".json"):
            leaks += [(name, key) for key in forbidden if key in content]
    assert not leaks, f"metric values / keys at launcher boundary: {sorted(set(leaks))[:10]}"
