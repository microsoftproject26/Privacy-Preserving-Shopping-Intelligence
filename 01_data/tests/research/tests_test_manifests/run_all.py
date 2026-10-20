"""DATA-09 runner (the DATA-07 / DATA-08 style): the tests_data09 suite, the negative-control intent check (--runxfail:
every test_nc_* that runs must FAIL with an AssertionError) and the integrity checks, recorded in one strict-JSON receipt.

  python -B run_all.py <receipt.json> [--basetemp <dir>]        (from code/tests_data09; CUDA_VISIBLE_DEVICES="" on the server)

Environment: D9_TESTS_SYNTHETIC_ONLY=1 skips the real-data tests; D9_PYTEST_BASETEMP is the default basetemp (else
<ppsi_v4>/../_pytest_tmp/DATA-09). The basetemp is removed after the run. Before and after, sha256 of the locked /
not-owned inputs this task must not change (v4data, v4data06, v4data08, v4fs_data, census_v2, v4train, the register tool,
contracts/*.json + contracts/manifest_registry/*.json, company/registers/TEST_EXPOSURE_REGISTER.json) must be identical,
ALL_METHODS_LOCK.json must be absent before and after on a local host (the synthetic gate lives in the basetemp), and no
__pycache__ may appear in the owned trees (v4data09, tests_data09). exit_code in the receipt is 0 only if all of this holds.
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
import ppsi_paths

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = ppsi_paths.PROJECT_ROOT.resolve()   # the ppsi_v4 workspace (contracts, company gates)
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
LOCKED_DIRS = ("v4data", "v4data06", "v4data08", "v4fs_data", "census_v2", "v4train")
OWNED_DIRS = ("v4data09", "tests_data09")
ALL_METHODS_LOCK = V4 / "company" / "gates" / "ALL_METHODS_LOCK.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def locked_hashes() -> dict:
    files = [p for d in LOCKED_DIRS for p in sorted(ppsi_paths.code_dir(d).rglob("*.py")) if "__pycache__" not in p.parts]
    files += sorted((V4 / "contracts").glob("*.json")) + sorted((V4 / "contracts" / "manifest_registry").glob("*.json"))
    files += [V4 / "tools" / "test_exposure_register.py", V4 / "company" / "registers" / "TEST_EXPOSURE_REGISTER.json"]

    def v4_rel(p):  # the old ppsi_v4-relative name: repository code as code/<name>, workspace files relative to V4
        try:
            return "code/" + ppsi_paths.legacy_rel(p)
        except ValueError:
            return str(p.relative_to(V4)).replace("\\", "/")
    return {v4_rel(p): _sha(p) for p in files if p.is_file()}


def parse_junit(path: Path) -> list:
    root = ET.parse(path).getroot()
    cases = []
    for tc in root.iter("testcase"):
        kind, msg, exc = "passed", None, []
        for child in tc:
            if child.tag in ("failure", "error"):
                kind, msg = child.tag, (child.get("message") or "")[:300]
                exc = sorted(set(re.findall(r"^E\s+(\w+(?:Error|Exception))\b", child.text or "", flags=re.M)))
            elif child.tag == "skipped":
                kind = "xfailed" if (child.get("type") == "pytest.xfail"
                                     or "NEGATIVE CONTROL" in (child.get("message") or "")) else "skipped"
        cases.append({"test": f"{tc.get('classname')}::{tc.get('name')}", "outcome": kind, "message": msg,
                      "exception_types": exc})
    return cases


def _is_nc(name: str) -> bool:
    return name.split("::")[-1].split("[")[0].startswith("test_nc_")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out = Path(argv[0]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    bt = Path(argv[argv.index("--basetemp") + 1]) if "--basetemp" in argv else Path(
        os.environ.get("D9_PYTEST_BASETEMP") or (V4.parent / "_pytest_tmp" / "DATA-09" / "run_all"))
    xml, xml_nc = out.with_suffix(".junit.xml"), out.with_suffix(".nc_runxfail.junit.xml")
    bt.mkdir(parents=True, exist_ok=True)                 # pytest creates the basetemp itself, not its parents
    lock_before = ALL_METHODS_LOCK.exists()
    before = locked_hashes()
    t0 = time.time()
    r = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
                        f"--junitxml={xml}", f"--basetemp={bt / 'suite'}"], cwd=HERE, capture_output=True, text=True,
                       env=ENV)
    cases = parse_junit(xml)
    counts: dict = {}
    for c in cases:
        counts[c["outcome"]] = counts.get(c["outcome"], 0) + 1
    nc_cases = [c for c in cases if _is_nc(c["test"])]
    nc_run = [c for c in nc_cases if c["outcome"] != "skipped"]
    q = subprocess.run([sys.executable, "-B", "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
                        "--runxfail", "-k", "test_nc_", f"--junitxml={xml_nc}", f"--basetemp={bt / 'nc'}"], cwd=HERE,
                       capture_output=True, text=True, env=ENV)
    rx = [c for c in parse_junit(xml_nc) if c["outcome"] != "skipped"]
    rx_ok = bool(rx) and all(c["outcome"] == "failure" and set(c["exception_types"]) <= {"AssertionError"} for c in rx)
    shutil.rmtree(bt, ignore_errors=True)
    after = locked_hashes()
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    pyc = [str(p) for d in OWNED_DIRS for p in ppsi_paths.code_dir(d).rglob("__pycache__")]
    nc_ok = bool(nc_run) and all(c["outcome"] == "xfailed" for c in nc_run)
    ok = (r.returncode == 0 and nc_ok and rx_ok and sorted(c["test"] for c in rx) == sorted(c["test"] for c in nc_run)
          and not changed and not pyc and ALL_METHODS_LOCK.exists() == lock_before)
    rec = {"receipt": "DATA-09 test run", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "exit_code": 0 if ok else 1, "pytest_exit_code": r.returncode, "wall_s": round(time.time() - t0, 1),
           "python": sys.version.split()[0], "platform": platform.platform(), "basetemp": str(bt),
           "synthetic_only": os.environ.get("D9_TESTS_SYNTHETIC_ONLY") == "1", "counts": counts,
           "negative_controls": {"n": len(nc_cases), "n_run": len(nc_run), "n_skipped": len(nc_cases) - len(nc_run),
                                 "all_failed_as_intended": nc_ok},
           "nc_runxfail": {"exit_code_expected_nonzero": q.returncode, "n": len(rx),
                           "all_fail_with_assertion_error": rx_ok,
                           "per_control": {c["test"]: {"outcome": c["outcome"], "exception_types": c["exception_types"]}
                                           for c in rx}},
           "locked_inputs": {"n_files": len(after), "changed": changed,
                             "sha256": hashlib.sha256(json.dumps(after, sort_keys=True).encode()).hexdigest()},
           "all_methods_lock_present_before_after": [lock_before, ALL_METHODS_LOCK.exists()],
           "pycache_in_owned": pyc, "summary_line": (r.stdout.strip().splitlines() or [""])[-1],
           "cases": [{k: c[k] for k in ("test", "outcome")} for c in cases],
           "stdout_tail": r.stdout.strip().splitlines()[-40:] if r.returncode else []}
    out.write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(rec["summary_line"], "| nc runxfail all AssertionError:", rx_ok, "| locked unchanged:", not changed,
          "| exit", rec["exit_code"])
    return rec["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
