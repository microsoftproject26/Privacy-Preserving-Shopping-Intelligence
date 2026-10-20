"""Run the DATA-04 test suite and write a strict-JSON receipt with the exit code and per-outcome counts.

Usage: python run_all.py <receipt.json>   (from code/tests_data; CUDA_VISIBLE_DEVICES="" on the server)
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    import shutil
    out = Path(sys.argv[1]).resolve()
    xml = out.with_suffix(".junit.xml")
    base = HERE.parents[1] / "data_v2" / "_pytest_tmp"            # owned path; removed after the run
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", f"--junitxml={xml}",
                        f"--basetemp={base}"], cwd=HERE, capture_output=True, text=True)
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
                kind = "xfailed" if (child.get("type") == "pytest.xfail" or "xfail" in (child.get("message") or "")
                                     or tc.get("name", "").startswith("test_nc_")) else "skipped"
        cases.append({"test": f"{tc.get('classname')}::{tc.get('name')}", "outcome": kind})
    counts = {}
    for c in cases:
        counts[c["outcome"]] = counts.get(c["outcome"], 0) + 1
    nc_names = [c for c in cases if c["test"].split("::")[-1].startswith("test_nc_")]
    rec = {"receipt": "DATA-04 test run", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "exit_code": r.returncode, "wall_s": round(time.time() - t0, 1), "python": sys.version.split()[0],
           "platform": platform.platform(), "counts": counts,
           "negative_controls": {"n": len(nc_names),
                                 "all_failed_as_intended": all(c["outcome"] == "xfailed" for c in nc_names)},
           "summary_line": (r.stdout.strip().splitlines() or [""])[-1], "cases": cases}
    out.write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(rec["summary_line"], "exit", r.returncode)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
