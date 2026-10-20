"""BASE-01 runner: the tests_base suite (pytest, CPU), the negative-control intent check and the scope-integrity
check, with every exit code recorded in receipts/TEST_RUN.json.

  python -B code/tests_base/run_all.py

Steps (fresh interpreters, -B and PYTHONDONTWRITEBYTECODE=1, pytest cache provider off):
  1. before: sha256 of every OTHER top-level Python package under code/ (everything with an __init__.py except
     v4base itself — the read-only dependencies BASE-01 must never edit: v4data, v4data06, v4eval, v4fs, v4fs_data,
     v4hist, v4int, v4models, v4sup, v4train, fl, plus census_v2/featstat_v1), discovered dynamically (not a fixed
     list) so a missed package can never hide a scope violation; no __pycache__ under v4base/tests_base;
  2. pytest over code/tests_base (junit + full output);
  3. pytest --runxfail on the negative controls only: every test_nc_* must FAIL with an AssertionError;
  4. after: the same package hashes (must be byte-identical) and no leftover __pycache__.
Writes only into code/tests_base/receipts/.
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

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
REC = HERE / "receipts"
GENERATED = ("pytest_output.txt", "junit.xml", "nc_runxfail_output.txt", "nc_runxfail_junit.xml", "TEST_RUN.json")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
OWNED = ("v4base", "tests_base")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run(args):
    return subprocess.run([sys.executable, "-B", *args], cwd=str(CODE), env=ENV, capture_output=True, text=True)


def imported_packages() -> tuple:
    """Every top-level Python package under code/ except v4base itself — discovered dynamically (an __init__.py
    marks a package), so this check can never silently miss a newly-added locked package."""
    return tuple(sorted(p.name for p in CODE.iterdir() if p.is_dir() and (p / "__init__.py").is_file()
                        and p.name not in OWNED))


def imported_hashes() -> dict:
    out = {}
    for pkg in imported_packages():
        for p in sorted((CODE / pkg).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                out[str(p.relative_to(CODE)).replace("\\", "/")] = _sha(p)
    return out


def integrity() -> dict:
    return {"pycache_dirs_in_owned": [str(c) for d in OWNED for c in (CODE / d).rglob("__pycache__")],
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
    res = {"task": "BASE-01", "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0], "cpu_only": True,
           "imported_packages": list(imported_packages())}
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
    res["locked_changed_files"] = sorted(k for k in set(b_) | set(a_) if b_.get(k) != a_.get(k))
    res["locked_packages_unchanged"] = not res["locked_changed_files"]
    res["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (REC / "TEST_RUN.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("pytest_exit_code", "counts", "negative_controls_all_failed_as_intended",
                                          "locked_packages_unchanged")}, indent=2))
    return 0 if (res["pytest_exit_code"] == 0 and res["locked_packages_unchanged"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
