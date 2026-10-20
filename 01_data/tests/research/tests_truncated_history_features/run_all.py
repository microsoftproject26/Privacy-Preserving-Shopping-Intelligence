"""DATA-07 runner: the tests_trunc suite (pytest, CPU, strict FP32), the negative-control intent check (--runxfail: every
test_nc_* must FAIL with an AssertionError) and the integrity checks, recorded in receipts/TEST_RUN.json.

  python -B code/tests_trunc/run_all.py [--basetemp <dir>]

Locked / not-owned trees are hashed before and after (v4data, v4hist, v4eval, v4train, v4models, fl, v4sup, every
v4int file except launch_trunc.py, ppsi_env.py) and must be byte-identical; the snapshot must verify; no __pycache__
may appear in the owned trees (v4trunc, tests_trunc).
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
V4 = ppsi_paths.PROJECT_ROOT.resolve()   # the ppsi_v4 workspace (contracts, company gates)
REC = HERE / "receipts"
GENERATED = ("pytest_output.txt", "junit.xml", "nc_runxfail_output.txt", "nc_runxfail_junit.xml", "TEST_RUN.json")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
LOCKED = ("v4data", "v4hist", "v4eval", "v4train", "v4models", "fl", "v4sup", "v4int")
OWNED_DIRS = ("v4trunc", "tests_trunc")
OWNED_FILES = ("v4int/launch_trunc.py",)
DEFAULT_BASETEMP = V4.parent / "_pytest_tmp" / "DATA-07"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run(args):
    return subprocess.run([sys.executable, "-B", *args], cwd=str(CODE), env=ENV, capture_output=True, text=True)


def locked_hashes() -> dict:
    out = {}
    for pkg in LOCKED:
        for p in sorted(ppsi_paths.code_dir(pkg).rglob("*.py")):
            rel = ppsi_paths.legacy_rel(p)
            if "__pycache__" not in p.parts and rel not in OWNED_FILES:
                out[rel] = _sha(p)
    out["ppsi_env.py"] = _sha(ppsi_paths.code_dir("ppsi_env.py"))
    return out


def integrity() -> dict:
    p = _run(["-c", "import ppsi_env; ppsi_env.verify_snapshot(); ppsi_env.assert_isolated(); print('OK')"])
    return {"verify_snapshot_exit_code": p.returncode, "stderr_tail": p.stderr.strip()[-400:],
            "pycache_dirs_under_snap_v1": [str(c) for c in (ppsi_paths.code_dir("snap_v1")).rglob("__pycache__")],
            "pycache_dirs_in_owned": [str(c) for d in OWNED_DIRS for c in ppsi_paths.code_dir(d).rglob("__pycache__")],
            "locked_sha256": locked_hashes()}


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
    counts: dict = {}
    for c in cases:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {"counts": counts, "cases": cases}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    bt = Path(argv[argv.index("--basetemp") + 1]) if "--basetemp" in argv else DEFAULT_BASETEMP
    (bt / "tmp").mkdir(parents=True, exist_ok=True)            # B-F6: create --basetemp (parents) + TMP/TEMP under it
    ENV.update(TMP=str(bt / "tmp"), TEMP=str(bt / "tmp"))
    REC.mkdir(exist_ok=True)
    for f in GENERATED:
        (REC / f).unlink(missing_ok=True)
    res = {"task": "DATA-07", "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
           "cpu_only": True, "basetemp": str(bt)}
    before = integrity()
    t0 = time.time()
    p = _run(["-m", "pytest", str(HERE), "-q", "-rA", "-p", "no:cacheprovider", "--basetemp", str(bt / "suite"),
              "--junitxml", str(REC / "junit.xml")])
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
              "--basetemp", str(bt / "nc"), "--junitxml", str(REC / "nc_runxfail_junit.xml")])
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
    res["integrity_before"] = {k: v for k, v in before.items() if k != "locked_sha256"}
    res["integrity_after"] = {k: v for k, v in after.items() if k != "locked_sha256"}
    b_, a_ = before["locked_sha256"], after["locked_sha256"]
    res["locked_changed_files"] = sorted(k for k in set(b_) | set(a_) if b_.get(k) != a_.get(k))
    res["locked_unchanged"] = not res["locked_changed_files"]
    res["locked_n_files"] = len(a_)
    res["locked_sha256"] = hashlib.sha256(json.dumps(a_, sort_keys=True).encode()).hexdigest()
    res["snapshot_unchanged"] = (before["verify_snapshot_exit_code"] == 0 and after["verify_snapshot_exit_code"] == 0
                                 and not after["pycache_dirs_under_snap_v1"])
    owned = [pp for d in OWNED_DIRS for pp in sorted(ppsi_paths.code_dir(d).rglob("*.py")) if "__pycache__" not in pp.parts]
    owned += [ppsi_paths.locate("code/" + f) for f in OWNED_FILES]
    res["owned_sha256"] = {ppsi_paths.legacy_rel(pp): _sha(pp) for pp in owned}
    import torch  # noqa: E402  (version string only)
    res["torch"] = torch.__version__
    res["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["receipt_sha256"] = {f: _sha(REC / f) for f in GENERATED if f != "TEST_RUN.json" and (REC / f).exists()}
    ok = (p.returncode == 0 and res["negative_controls_all_failed_as_intended"]
          and res["nc_runxfail"]["all_fail_with_assertion"] and res["nc_runxfail"]["same_set_as_suite"]
          and res["snapshot_unchanged"] and res["locked_unchanged"] and not after["pycache_dirs_in_owned"])
    res["overall_exit_code"] = 0 if ok else 1
    (REC / "TEST_RUN.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("pytest_exit_code", "pytest_summary", "counts",
                                          "negative_controls_all_failed_as_intended", "snapshot_unchanged",
                                          "locked_unchanged", "overall_exit_code")}))
    print("nc_runxfail:", json.dumps({k: res["nc_runxfail"][k] for k in ("counts", "all_fail_with_assertion",
                                                                          "same_set_as_suite")}))
    return res["overall_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
