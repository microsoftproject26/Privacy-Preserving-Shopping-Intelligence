"""TRAIN-01 runner: the trainer test suite (pytest, CPU, strict FP32), the negative-control intent check and the
integrity checks, with every exit code recorded.

  python -B code/tests_train/run_all.py

Steps (fresh interpreters, -B and PYTHONDONTWRITEBYTECODE=1, pytest cache provider off):
  1. before: snapshot hashes, no __pycache__ under snap_v1, sha256 of the imported-not-edited packages
     (v4models, v4eval, v4sup, fl, ppsi_env.py);
  2. pytest over code/tests_train (junit + full output);
  3. pytest --runxfail on the negative controls only: every test_nc_* must FAIL with an AssertionError (expected
     non-zero exit code);
  4. after: the same integrity checks; the imported packages must be byte-identical; no run directory is left behind.
Writes only into code/tests_train/receipts/.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import ppsi_paths

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
REC = HERE / "receipts"
GENERATED = ("pytest_output.txt", "junit.xml", "nc_runxfail_output.txt", "nc_runxfail_junit.xml", "TEST_RUN.json")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUBLAS_WORKSPACE_CONFIG=":4096:8")   # the launcher's T3 export
IMPORTED = ("v4models", "v4eval", "v4sup", "fl", "v4fs")
# live trees of concurrent tasks (never written by v4train / tests_train): FL-02 fl/, EVAL-03 v4eval/, and SCHEMA-01's
# v4models/ and v4fs/ (TRAIN-01R2 ran while SCHEMA-01 was landing its derived-schema support); changes there are
# reported, not failed. Any change in v4sup/ or ppsi_env.py fails the run.
LIVE_TREES = ("fl/", "v4eval/", "v4models/", "v4fs/")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run(args):
    return subprocess.run([sys.executable, "-B", *args], cwd=str(CODE), env=ENV, capture_output=True, text=True)


def imported_hashes() -> dict:
    out = {}
    for pkg in IMPORTED:
        if not ppsi_paths.code_dir(pkg).is_dir():
            continue
        for p in sorted(ppsi_paths.code_dir(pkg).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                out[ppsi_paths.legacy_rel(p)] = _sha(p)
    out["ppsi_env.py"] = _sha(ppsi_paths.code_dir("ppsi_env.py"))
    return out


def integrity() -> dict:
    p = _run(["-c", "import ppsi_env; ppsi_env.verify_snapshot(); print('OK')"])
    return {"verify_snapshot_exit_code": p.returncode, "stderr_tail": p.stderr.strip()[-400:],
            "pycache_dirs_under_snap_v1": [str(c) for c in (ppsi_paths.code_dir("snap_v1")).rglob("__pycache__")],
            "pycache_dirs_in_owned": [str(c) for d in ("v4train", "tests_train") for c in ppsi_paths.code_dir(d).rglob("__pycache__")],
            "imported_sha256": imported_hashes()}


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
                      "exception_types": exc, "time_s": float(tc.get("time", 0))})
    counts = {}
    for c in cases:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {"counts": counts, "cases": cases}


def main() -> int:
    REC.mkdir(exist_ok=True)
    for f in GENERATED:
        (REC / f).unlink(missing_ok=True)
    res = {"task": "TRAIN-02", "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0], "cpu_only": True}
    before = integrity()
    t0 = time.time()
    p = _run(["-m", "pytest", str(HERE), "-q", "-rA", "-p", "no:cacheprovider", "--junitxml", str(REC / "junit.xml")])
    (REC / "pytest_output.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    junit = parse_junit(REC / "junit.xml")
    ncs = {c["name"]: ("FAILED_AS_INTENDED" if c["status"] == "xfailed" else f"UNEXPECTED_{c['status'].upper()}")
           for c in junit["cases"] if c["name"].split("::")[-1].split("[")[0].startswith("test_nc_")}
    summary = [ln for ln in p.stdout.strip().splitlines() if " in " in ln and ("passed" in ln or "failed" in ln)]
    res.update({"pytest_exit_code": p.returncode, "pytest_summary": summary[-1] if summary else None,
                "pytest_wall_s": round(time.time() - t0, 1), "counts": junit["counts"], "negative_controls": ncs,
                "negative_controls_all_failed_as_intended": bool(ncs) and all(v == "FAILED_AS_INTENDED"
                                                                              for v in ncs.values())})
    q = _run(["-m", "pytest", str(HERE), "-q", "-p", "no:cacheprovider", "--runxfail", "-k", "test_nc_",
              "--junitxml", str(REC / "nc_runxfail_junit.xml")])
    (REC / "nc_runxfail_output.txt").write_text(q.stdout + "\n--- stderr ---\n" + q.stderr, encoding="utf-8")
    rx = parse_junit(REC / "nc_runxfail_junit.xml")
    per = {c["name"]: {"status": c["status"], "exception_types": c["exception_types"],
                       "assertion": (c["message"] or "").splitlines()[0][:200] if c["message"] else None}
           for c in rx["cases"]}
    res["nc_runxfail"] = {"exit_code_expected_nonzero": q.returncode, "counts": rx["counts"],
                          "all_fail_with_assertion": bool(per) and all(
                              v["status"] == "failure" and set(v["exception_types"]) <= {"AssertionError"}
                              for v in per.values()),
                          "same_set_as_suite": sorted(per) == sorted(ncs), "per_control": per}
    after = integrity()
    res["integrity_before"] = {k: v for k, v in before.items() if k != "imported_sha256"}
    res["integrity_after"] = {k: v for k, v in after.items() if k != "imported_sha256"}
    b_, a_ = before["imported_sha256"], after["imported_sha256"]
    changed = sorted(k for k in set(b_) | set(a_) if b_.get(k) != a_.get(k))
    res["imported_changed_files"] = changed
    live = LIVE_TREES                                    # see LIVE_TREES above; v4train never writes there
    res["imported_packages_unchanged"] = not [c for c in changed if not c.startswith(live)]
    res["concurrent_changes_in_live_trees"] = [c for c in changed if c.startswith(live)]
    res["imported_packages_n_files"] = len(a_)
    res["imported_packages_sha256"] = hashlib.sha256(json.dumps(after["imported_sha256"], sort_keys=True)
                                                     .encode()).hexdigest()
    res["work_dir_left_behind"] = (HERE / "_work").exists()
    import tempfile  # noqa: E402  (trainkit's per-process registry dirs live under the system temp dir)
    res["registry_temp_dirs_left_behind"] = sorted(str(p) for p in Path(tempfile.gettempdir()).glob("v4train_r2_*"))
    h = _run(["-c", "import v4train; from v4train.trainer import trainer_code_sha256; print(trainer_code_sha256())"])
    res["trainer_code_sha256"] = h.stdout.strip() if h.returncode == 0 else f"ERROR {h.stderr.strip()[-300:]}"
    res["snapshot_unchanged"] = (before["verify_snapshot_exit_code"] == 0 and after["verify_snapshot_exit_code"] == 0
                                 and not after["pycache_dirs_under_snap_v1"])
    import torch  # noqa: E402  (version string only)
    res["torch"] = torch.__version__
    res["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["receipt_sha256"] = {f: _sha(REC / f) for f in GENERATED if f != "TEST_RUN.json" and (REC / f).exists()}
    (REC / "TEST_RUN.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    ok = (p.returncode == 0 and res["negative_controls_all_failed_as_intended"]
          and res["nc_runxfail"]["all_fail_with_assertion"] and res["nc_runxfail"]["same_set_as_suite"]
          and res["snapshot_unchanged"] and res["imported_packages_unchanged"] and not res["work_dir_left_behind"]
          and not res["registry_temp_dirs_left_behind"] and not after["pycache_dirs_in_owned"]
          and not str(res["trainer_code_sha256"]).startswith("ERROR"))
    print(json.dumps({k: res[k] for k in ("pytest_exit_code", "pytest_summary", "counts",
                                          "negative_controls_all_failed_as_intended", "snapshot_unchanged",
                                          "imported_packages_unchanged", "work_dir_left_behind",
                                          "registry_temp_dirs_left_behind", "concurrent_changes_in_live_trees",
                                          "trainer_code_sha256")}))
    print("nc_runxfail:", json.dumps({k: res["nc_runxfail"][k] for k in ("counts", "all_fail_with_assertion",
                                                                          "same_set_as_suite")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
