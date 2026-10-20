"""FL-01 runner: the full test suite (pytest, CPU) + the per-visit micro-benchmark, with exit codes recorded.

  python -B code/tests_fl/run_all.py

Writes only into code/tests_fl/receipts/: pytest_output.txt, junit.xml, BENCH_VISIT_CPU.json, TEST_RUN.json (plus the
receipts the tests themselves write: LO_DETERMINISM_1024*.json, T34_POOL_RUNS.json, DP_Z_RECORD_FL04.json). Byte-code
writing is disabled (-B / PYTHONDONTWRITEBYTECODE) and the pytest cache provider is off, so nothing is written
elsewhere (pytest tmp_path goes to %TEMP%: point TMP/TEMP outside the V4 tree).
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
REC = HERE / "receipts"
GENERATED = ("pytest_output.txt", "junit.xml", "BENCH_VISIT_CPU.json", "TEST_RUN.json", "LO_DETERMINISM_1024.json",
             "LO_DETERMINISM_1024_reverse_subprocess.json", "T34_POOL_RUNS.json", "DP_Z_RECORD_FL04.json")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    cases = []
    for tc in root.iter("testcase"):
        name = tc.get("name")
        status = "passed"
        for child in tc:
            if child.tag in ("failure", "error"):
                status = child.tag
            elif child.tag == "skipped":
                status = "xfailed" if (child.get("type") == "pytest.xfail") else "skipped"
        cases.append({"name": f"{tc.get('classname')}::{name}", "status": status, "time_s": float(tc.get("time", 0))})
    counts = {}
    for c in cases:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    ncs = [c for c in cases if c["name"].split("::")[-1].startswith("test_nc_")]
    return {"counts": counts, "n_cases": len(cases),
            "negative_controls": {c["name"]: ("FAILED_AS_INTENDED" if c["status"] == "xfailed" else
                                              f"UNEXPECTED_{c['status'].upper()}") for c in ncs},
            "cases": cases}


def main() -> int:
    REC.mkdir(exist_ok=True)
    for f in GENERATED:
        (REC / f).unlink(missing_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-B", "-m", "pytest", str(HERE), "-q", "-rA", "-p", "no:cacheprovider",
                        "--junitxml", str(REC / "junit.xml")], cwd=str(CODE), env=env, capture_output=True, text=True)
    (REC / "pytest_output.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    t_tests = time.time() - t0
    t1 = time.time()
    b = subprocess.run([sys.executable, "-B", "-m", "fl.bench_visit", "--out", str(REC / "BENCH_VISIT_CPU.json"),
                        "--reps", "50", "--threads", "1", "--families"], cwd=str(CODE), env=env, capture_output=True,
                       text=True)
    t_bench = time.time() - t1
    junit = parse_junit(REC / "junit.xml") if (REC / "junit.xml").exists() else {"counts": {}, "negative_controls": {}}
    nc_ok = bool(junit["negative_controls"]) and all(v == "FAILED_AS_INTENDED" for v in junit["negative_controls"].values())
    summary_line = [ln for ln in p.stdout.strip().splitlines() if " in " in ln and ("passed" in ln or "failed" in ln)]
    import torch  # noqa: E402  (only for the version string)
    res = {"task": "FL-01", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
           "torch": torch.__version__, "cpu_only": True,
           "pytest_exit_code": p.returncode, "pytest_summary": summary_line[-1] if summary_line else None,
           "pytest_wall_s": round(t_tests, 1), "counts": junit["counts"],
           "negative_controls_all_failed_as_intended": nc_ok, "negative_controls": junit["negative_controls"],
           "bench_exit_code": b.returncode, "bench_wall_s": round(t_bench, 1), "bench_stdout": b.stdout.strip()[-2000:],
           "receipt_sha256": {f: _sha(REC / f) for f in GENERATED if f != "TEST_RUN.json" and (REC / f).exists()}}
    (REC / "TEST_RUN.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("pytest_exit_code", "pytest_summary", "counts",
                                          "negative_controls_all_failed_as_intended", "bench_exit_code")}))
    return max(p.returncode, b.returncode, 0 if nc_ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
