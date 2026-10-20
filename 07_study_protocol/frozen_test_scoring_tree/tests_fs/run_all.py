"""SCHEMA-01 runner (tests_fs): the v4fs test suite (pytest, CPU, strict FP32), the negative-control intent check, one
CLI dry run in a fresh interpreter, and the integrity checks, with every exit code recorded.

  python -B code/tests_fs/run_all.py

Steps (fresh interpreters, -B and PYTHONDONTWRITEBYTECODE=1, CUBLAS_WORKSPACE_CONFIG=:4096:8, pytest cache off):
  1. before: snapshot hashes, no __pycache__ under snap_v1;
  2. pytest over code/tests_fs (junit + full output);
  3. pytest --runxfail on the negative controls only: every test_nc_* must FAIL with an AssertionError (the expected
     non-zero exit code);
  4. `python -B code/v4fs/run_fs.py ... --dry-run` for L3 / GRU / seed 2026 in a fresh process (export before import,
     the whole pipeline), its FS_SUMMARY checked, the directory removed;
  5. after: the same integrity checks; no __pycache__ in v4fs / tests_fs / v4models; no run directory left behind.
Writes only into code/tests_fs/receipts/.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
REC = HERE / "receipts"
GENERATED = ("pytest_output.txt", "junit.xml", "nc_runxfail_output.txt", "nc_runxfail_junit.xml", "dry_run_cli.txt",
             "TEST_RUN.json")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")
OWNED = ("v4fs", "tests_fs", "v4models", "tests_models")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run(args, env=None):
    return subprocess.run([sys.executable, "-B", *args], cwd=str(CODE), env=env or ENV, capture_output=True, text=True)


def integrity() -> dict:
    p = _run(["-c", "import ppsi_env; ppsi_env.verify_snapshot(); print('OK')"])
    return {"verify_snapshot_exit_code": p.returncode, "stderr_tail": p.stderr.strip()[-400:],
            "pycache_dirs_under_snap_v1": [str(c) for c in (CODE / "snap_v1").rglob("__pycache__")],
            "pycache_dirs_in_owned": [str(c) for d in OWNED for c in (CODE / d).rglob("__pycache__")],
            "code_sha256": {str(p.relative_to(CODE)).replace("\\", "/"): _sha(p)
                            for d in ("v4fs", "v4models") for p in sorted((CODE / d).glob("*.py"))}}


def parse_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    cases = []
    for tc in root.iter("testcase"):
        status, msg, exc = "passed", None, []
        for child in tc:
            if child.tag in ("failure", "error"):
                status, msg = child.tag, (child.get("message") or "")[:300]
                exc = sorted(set(re.findall(r"^E\s+(\w+(?:Error|Exception))\b", child.text or "", flags=re.M)))
            elif child.tag == "skipped":
                status = "xfailed" if child.get("type") == "pytest.xfail" else "skipped"
                msg = (child.get("message") or "")[:300]
        cases.append({"name": f"{tc.get('classname')}::{tc.get('name')}", "status": status, "message": msg,
                      "exception_types": exc})
    counts = {}
    for c in cases:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {"counts": counts, "cases": cases}


def main() -> int:
    REC.mkdir(exist_ok=True)
    for f in GENERATED:
        (REC / f).unlink(missing_ok=True)
    res = {"task": "SCHEMA-01", "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
           "cpu_only": True, "before": integrity()}
    t0 = time.time()
    p = _run(["-m", "pytest", str(HERE), "-q", "-rA", "-p", "no:cacheprovider", "--junitxml", str(REC / "junit.xml")])
    (REC / "pytest_output.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    ju = parse_junit(REC / "junit.xml")
    ncs = {c["name"]: ("FAILED_AS_INTENDED" if c["status"] == "xfailed" else f"UNEXPECTED_{c['status'].upper()}")
           for c in ju["cases"] if c["name"].split("::")[-1].split("[")[0].startswith("test_nc_")}
    summary = [ln for ln in p.stdout.strip().splitlines() if " in " in ln and ("passed" in ln or "failed" in ln)]
    res.update({"pytest_exit_code": p.returncode, "pytest_summary": summary[-1] if summary else None,
                "pytest_wall_s": round(time.time() - t0, 1), "counts": ju["counts"],
                "skipped": [c for c in ju["cases"] if c["status"] == "skipped"],
                "negative_controls": ncs,
                "negative_controls_all_failed_as_intended": bool(ncs) and all(v == "FAILED_AS_INTENDED"
                                                                              for v in ncs.values())})
    q = _run(["-m", "pytest", str(HERE), "-q", "-p", "no:cacheprovider", "--runxfail", "-k", "test_nc_",
              "--junitxml", str(REC / "nc_runxfail_junit.xml")])
    (REC / "nc_runxfail_output.txt").write_text(q.stdout + "\n--- stderr ---\n" + q.stderr, encoding="utf-8")
    rx = parse_junit(REC / "nc_runxfail_junit.xml")
    intent = {c["name"]: {"status": c["status"], "exception_types": c["exception_types"],
                          "assertion": (c["message"] or "").splitlines()[0][:200] if c["message"] else None}
              for c in rx["cases"]}
    res["nc_runxfail"] = {"exit_code_expected_nonzero": q.returncode, "counts": rx["counts"],
                          "all_fail_with_assertion": bool(intent) and all(
                              v["status"] == "failure" and set(v["exception_types"]) <= {"AssertionError"}
                              for v in intent.values()),
                          "same_set_as_suite": sorted(intent) == sorted(ncs), "per_control": intent}
    work = HERE / "_work" / "cli_dry_run"
    shutil.rmtree(work, ignore_errors=True)
    env = {k: v for k, v in ENV.items() if k != "CUDA_VISIBLE_DEVICES"}
    d = _run([str(CODE / "v4fs" / "run_fs.py"), "--family", "GRU", "--schema", "L3S_NEAR_DUPLICATE_SESSION", "--seed", "2026",
              "--out", str(work), "--dry-run", "--fs-registry", str(work.parent / "no_registry.json"),
              "--log-every", "5"], env=env)
    (REC / "dry_run_cli.txt").write_text(d.stdout + "\n--- stderr ---\n" + d.stderr, encoding="utf-8")
    dry = {"exit_code": d.returncode}
    if d.returncode == 0:
        s = json.loads((work / "FS_SUMMARY.json").read_text(encoding="utf-8"))
        launch = json.loads((work / "FS_LAUNCH.json").read_text(encoding="utf-8"))
        dry.update({"run_label": s["run_label"], "weight_files_left": s["weight_files_left"],
                    "pt_files": [str(x) for x in work.rglob("*.pt")], "n_importance_groups": len(s["importance"]),
                    "importance_baseline_equals_controlled_budget": s["importance_baseline_equals_controlled_budget"],
                    "environment": launch["environment"], "trainer_support": launch["trainer_support"],
                    "deleted": sorted({r["kind"] for r in s["retention"] if "sha256" in r})})
    shutil.rmtree(work, ignore_errors=True)
    for x in (work.parent, HERE / "_work"):
        try:
            x.rmdir()
        except OSError:
            pass
    res["dry_run_cli"] = dry
    res["after"] = integrity()
    res["work_left"] = (HERE / "_work").exists()
    import torch  # noqa: E402  (version string only)
    res["torch"] = torch.__version__
    res["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["receipt_sha256"] = {f: _sha(REC / f) for f in GENERATED if f != "TEST_RUN.json" and (REC / f).exists()}
    (REC / "TEST_RUN.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    ok = (p.returncode == 0 and res["negative_controls_all_failed_as_intended"]
          and res["nc_runxfail"]["all_fail_with_assertion"] and res["nc_runxfail"]["same_set_as_suite"]
          and dry.get("exit_code") == 0 and dry.get("weight_files_left") == [] and not dry.get("pt_files")
          and res["before"]["verify_snapshot_exit_code"] == 0 and res["after"]["verify_snapshot_exit_code"] == 0
          and not res["after"]["pycache_dirs_under_snap_v1"] and not res["after"]["pycache_dirs_in_owned"]
          and not res["work_left"])
    print(json.dumps({k: res[k] for k in ("pytest_exit_code", "pytest_summary", "counts",
                                          "negative_controls_all_failed_as_intended")}))
    print("nc_runxfail:", json.dumps({k: res["nc_runxfail"][k] for k in ("counts", "all_fail_with_assertion",
                                                                          "same_set_as_suite")}))
    print("dry_run_cli:", json.dumps(dry))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
