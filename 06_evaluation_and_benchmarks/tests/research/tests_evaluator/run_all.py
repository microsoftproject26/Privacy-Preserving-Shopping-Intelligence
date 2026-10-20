"""EVAL-02 runner: the full evaluator test suite (pytest, CPU), with exit code, counts and negative-control statuses.

  python -B code/tests_eval/run_all.py

Writes only into code/tests_eval/receipts/: pytest_output.txt, junit.xml, TEST_RUN.json. Byte-code writing is disabled
(-B / PYTHONDONTWRITEBYTECODE) and the pytest cache provider is off, so nothing is written elsewhere (in particular no
__pycache__ in code/snap_v1).
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
GENERATED = ("pytest_output.txt", "junit.xml", "TEST_RUN.json", "T3_PLATFORM_BITWISE_NOTE.json")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def parse_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    cases = []
    for tc in root.iter("testcase"):
        status = "passed"
        for child in tc:
            if child.tag in ("failure", "error"):
                status = child.tag
            elif child.tag == "skipped":
                status = "xfailed" if child.get("type") == "pytest.xfail" else "skipped"
        cases.append({"name": f"{tc.get('classname')}::{tc.get('name')}", "status": status,
                      "time_s": float(tc.get("time", 0))})
    counts = {}
    for c in cases:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    ncs = {c["name"]: ("FAILED_AS_INTENDED" if c["status"] == "xfailed" else f"UNEXPECTED_{c['status'].upper()}")
           for c in cases if c["name"].split("::")[-1].startswith("test_nc_")}
    return {"counts": counts, "n_cases": len(cases), "negative_controls": ncs, "cases": cases}


def main() -> int:
    REC.mkdir(exist_ok=True)
    for f in GENERATED:
        (REC / f).unlink(missing_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-B", "-m", "pytest", str(HERE), "-q", "-rA", "-p", "no:cacheprovider",
                        f"--junitxml={REC / 'junit.xml'}"], cwd=str(HERE), env=env, capture_output=True, text=True)
    wall = time.time() - t0
    (REC / "pytest_output.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    junit = parse_junit(REC / "junit.xml")
    sys.path.insert(0, str(CODE))
    import torch  # noqa: E402
    from v4eval import EVALUATOR_ID, evaluator_code_sha256  # noqa: E402
    receipt = {
        "task": "EVAL-02", "runner": "code/tests_eval/run_all.py", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exit_code": p.returncode, "wall_s": round(wall, 2), "python": sys.version.split()[0],
        "platform": platform.platform(), "torch": torch.__version__, "numpy": __import__("numpy").__version__,
        "evaluator_id": EVALUATOR_ID, "evaluator_code_sha256": evaluator_code_sha256(),
        "test_files_sha256": {f.name: _sha(f) for f in sorted(HERE.glob("*.py"))},
        "counts": junit["counts"], "n_cases": junit["n_cases"],
        "negative_controls": junit["negative_controls"],
        "all_negative_controls_failed_as_intended": all(v == "FAILED_AS_INTENDED"
                                                        for v in junit["negative_controls"].values()),
        "cases": junit["cases"],
        "data": "synthetic only; no manifest, ledger or TEST row read",
        "platform_fp_notes": (json.loads((REC / "T3_PLATFORM_BITWISE_NOTE.json").read_text(encoding="utf-8"))
                              if (REC / "T3_PLATFORM_BITWISE_NOTE.json").is_file() else "MISSING_EVIDENCE"),
    }
    (REC / "TEST_RUN.json").write_text(json.dumps(receipt, indent=1, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("exit_code", "counts", "n_cases", "all_negative_controls_failed_as_intended",
                                               "evaluator_code_sha256")}, indent=1))
    return p.returncode


if __name__ == "__main__":
    raise SystemExit(main())
