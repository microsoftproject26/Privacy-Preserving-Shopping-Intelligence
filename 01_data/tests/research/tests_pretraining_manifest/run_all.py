"""Run the DATA-10 test suite and write a strict-JSON receipt with the exit code and per-outcome counts.

Usage: python -B run_all.py <receipt.json>   (from code/tests_data10; CUDA_VISIBLE_DEVICES="" on the server)
Environment: D10_TESTS_SYNTHETIC_ONLY=1 skips the real-data tests; D10_PYTEST_BASETEMP sets the pytest temporary
root (removed after the run; default: a fresh system temporary directory).
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    xml = out.with_suffix(".junit.xml")
    base = Path(os.environ.get("D10_PYTEST_BASETEMP") or tempfile.mkdtemp(prefix="data10_pytest_"))
    t0 = time.time()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "--no-header", f"--junitxml={xml}",
                        f"--basetemp={base}"], cwd=HERE, capture_output=True, text=True, env=env)
    shutil.rmtree(base, ignore_errors=True)
    root = ET.parse(xml).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    cases = []
    for tc in suite.iter("testcase"):
        kind = "passed"
        for child in tc:
            if child.tag in ("failure", "error"):
                kind = child.tag
            elif child.tag == "skipped":
                kind = "xfailed" if (child.get("type") == "pytest.xfail"
                                     or "NEGATIVE CONTROL" in (child.get("message") or "")) else "skipped"
        cases.append({"test": f"{tc.get('classname')}::{tc.get('name')}", "outcome": kind})
    counts = {}
    for c in cases:
        counts[c["outcome"]] = counts.get(c["outcome"], 0) + 1
    nc_cases = [c for c in cases if c["test"].split("::")[-1].startswith("test_nc_")]
    nc_run = [c for c in nc_cases if c["outcome"] != "skipped"]
    rec = {"receipt": "DATA-10 test run", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "exit_code": r.returncode, "wall_s": round(time.time() - t0, 1), "python": sys.version.split()[0],
           "platform": platform.platform(), "synthetic_only": os.environ.get("D10_TESTS_SYNTHETIC_ONLY") == "1",
           "counts": counts,
           "negative_controls": {"n": len(nc_cases), "n_run": len(nc_run), "n_skipped": len(nc_cases) - len(nc_run),
                                 "all_failed_as_intended": bool(nc_run) and all(c["outcome"] == "xfailed" for c in nc_run)},
           "summary_line": (r.stdout.strip().splitlines() or [""])[-1], "cases": cases,
           "stdout_tail": r.stdout.strip().splitlines()[-40:] if r.returncode else []}
    out.write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(rec["summary_line"], "exit", r.returncode)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
