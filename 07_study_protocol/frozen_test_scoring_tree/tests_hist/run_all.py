"""DATA-05 runner: the history-policy suite (tests_hist), its negative-control intent check, and the INT-01 regression
suite (tests_int, unchanged checks; only the now-required history_policy argument was added to its calls), with every
exit code recorded in <receipts>/TEST_RUN.json.

  python -B code/tests_hist/run_all.py [--receipts-dir DIR]        (default: code/tests_hist/receipts)

Steps (fresh interpreters, -B and PYTHONDONTWRITEBYTECODE=1, pytest cache provider off, CPU only):
  1. before: snapshot hashes (ppsi_env.verify_snapshot / assert_isolated), no __pycache__ under snap_v1, sha256 of the
     locked trees (v4train, v4data, v4eval, v4models, fl, v4sup, census_v2, contracts, ppsi_env.py), G1 file absent;
  2. pytest over code/tests_hist (junit + full output);
  3. pytest --runxfail on its negative controls: every test_nc_* must FAIL with an AssertionError (or an
     AssertionError subclass of v4hist: HistoryLeakError / HistoryOrderError / HistoryAlignmentError);
  4. pytest over code/tests_int (the INT-01 suite) and its negative-control intent check, outputs in the same
     receipts directory (the INT-01 receipts in code/tests_int/receipts are never touched);
  5. after: the same integrity checks; the locked trees byte-identical; G1 still absent; no __pycache__ in owned trees.
"""
from __future__ import annotations

import argparse
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
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1")    # explicit CPU (OPS-07 rule)
LOCKED = ("v4train", "v4data", "v4eval", "v4models", "fl", "v4sup", "census_v2")
OWNED = ("v4hist", "tests_hist", "v4int", "tests_int")
G1_GATE = V4 / "company" / "gates" / "G1_APPROVED.json"
ASSERTION_TYPES = {"AssertionError", "HistoryLeakError", "HistoryOrderError", "HistoryAlignmentError",
                   "ProviderIntegrityError"}
_EXC = re.compile(r"^E\s+((?:\w+\.)*\w+(?:Error|Exception))\b", flags=re.M)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LEAD = re.compile(r"^((?:\w+\.)*\w+(?:Error|Exception)):")
DEPENDENCY_SKIP = "is not on this host"                  # the importorskip reason of a declared host dependency


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run(args):
    return subprocess.run([sys.executable, "-B", *args], cwd=str(CODE), env=ENV, capture_output=True, text=True)


def _pytest(args, tag: str):
    """pytest with its own --basetemp, deleted right after the call (the server disk rule: no temporary state
    outlives the step that made it)."""
    import shutil
    import tempfile
    bt = Path(tempfile.gettempdir()) / f"data05_pytest_{tag}"
    try:
        return _run(["-m", "pytest", *args, "--basetemp", str(bt)])
    finally:
        shutil.rmtree(bt, ignore_errors=True)


def locked_hashes() -> dict:
    out = {}
    for pkg in LOCKED:
        for p in sorted((CODE / pkg).rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                out[str(p.relative_to(CODE)).replace("\\", "/")] = _sha(p)
    for p in sorted((V4 / "contracts").rglob("*")):
        if p.is_file():
            out["../contracts/" + str(p.relative_to(V4 / "contracts")).replace("\\", "/")] = _sha(p)
    out["ppsi_env.py"] = _sha(CODE / "ppsi_env.py")
    return out


def integrity() -> dict:
    p = _run(["-c", "import ppsi_env; ppsi_env.verify_snapshot(); ppsi_env.assert_isolated(); print('OK')"])
    return {"verify_snapshot_exit_code": p.returncode, "stderr_tail": p.stderr.strip()[-400:],
            "pycache_dirs_under_snap_v1": [str(c) for c in (CODE / "snap_v1").rglob("__pycache__")],
            "pycache_dirs_in_owned": [str(c) for d in OWNED for c in (CODE / d).rglob("__pycache__")],
            "g1_gate_file_exists": G1_GATE.exists(),
            "g1_gate_file_sha256": hashlib.sha256(G1_GATE.read_bytes()).hexdigest() if G1_GATE.exists() else None,
            "locked_sha256": locked_hashes()}


def parse_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    cases = []
    for tc in root.iter("testcase"):
        status, msg, exc = "passed", None, []
        for child in tc:
            if child.tag in ("failure", "error"):
                status, msg = child.tag, _ANSI.sub("", child.get("message") or "")[:300]
                exc = sorted({m.split(".")[-1] for m in _EXC.findall(_ANSI.sub("", child.text or ""))})
                lead = _LEAD.match(msg or "")
                if not exc and lead:                                       # the message names the exception
                    exc = [lead.group(1).split(".")[-1]]
                if not exc and (msg or "").startswith("assert "):       # a bare pytest-rewritten assert statement
                    exc = ["AssertionError"]
            elif child.tag == "skipped":
                status = "xfailed" if child.get("type") == "pytest.xfail" else "skipped"
                msg = (child.get("message") or "")[:300]
        cases.append({"name": f"{tc.get('classname')}::{tc.get('name')}", "status": status, "message": msg,
                      "exception_types": exc, "time_s": float(tc.get("time", 0))})
    counts = {}
    for c in cases:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {"counts": counts, "cases": cases}


def suite(rec_dir: Path, tag: str, path: Path) -> dict:
    ju, out = rec_dir / f"{tag}_junit.xml", rec_dir / f"{tag}_pytest_output.txt"
    t0 = time.time()
    p = _pytest([str(path), "-q", "-rA", "-p", "no:cacheprovider", "--color=no", "--junitxml", str(ju)], tag)
    out.write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    junit = parse_junit(ju)
    def _nc_state(c):
        if c["status"] == "xfailed":
            return "FAILED_AS_INTENDED"
        if c["status"] == "skipped" and DEPENDENCY_SKIP in (c["message"] or ""):
            return "SKIPPED_DEPENDENCY_ABSENT"          # a declared missing dependency on this host (never a pass)
        return f"UNEXPECTED_{c['status'].upper()}"
    ncs = {c["name"]: _nc_state(c) for c in junit["cases"]
           if c["name"].split("::")[-1].split("[")[0].startswith("test_nc_")}
    dep_skipped = sorted(k for k, v in ncs.items() if v == "SKIPPED_DEPENDENCY_ABSENT")
    # a module skipped at collection for a declared missing dependency (its --runxfail record reads "collection
    # skipped" under the module name)
    dep_modules = {c["name"].split("::")[-1] for c in junit["cases"]
                   if c["status"] == "skipped" and DEPENDENCY_SKIP in (c["message"] or "")}
    summary = [ln for ln in p.stdout.strip().splitlines() if " in " in ln and ("passed" in ln or "failed" in ln)]
    res = {"pytest_exit_code": p.returncode, "pytest_summary": summary[-1] if summary else None,
           "pytest_wall_s": round(time.time() - t0, 1), "counts": junit["counts"], "negative_controls": ncs,
           "negative_controls_all_failed_as_intended": any(v == "FAILED_AS_INTENDED" for v in ncs.values()) and all(
               v in ("FAILED_AS_INTENDED", "SKIPPED_DEPENDENCY_ABSENT") for v in ncs.values()),
           "negative_controls_skipped_dependency_absent": dep_skipped}
    jx, ox = rec_dir / f"{tag}_nc_runxfail_junit.xml", rec_dir / f"{tag}_nc_runxfail_output.txt"
    q = _pytest([str(path), "-q", "-p", "no:cacheprovider", "--color=no", "--runxfail", "-k", "test_nc_",
                 "--junitxml", str(jx)], tag + "_nc")
    ox.write_text(q.stdout + "\n--- stderr ---\n" + q.stderr, encoding="utf-8")
    rx = parse_junit(jx)
    per = {c["name"]: {"status": c["status"], "exception_types": c["exception_types"],
                       "assertion": (c["message"] or "").splitlines()[0][:200] if c["message"] else None}
           for c in rx["cases"] if c["name"] not in dep_skipped
           and not (c["status"] == "skipped" and (DEPENDENCY_SKIP in (c["message"] or "")
                                                  or c["name"].split("::")[-1] in dep_modules))}
    ncs = {k: v for k, v in ncs.items() if k not in dep_skipped}
    res["nc_runxfail"] = {"exit_code_expected_nonzero": q.returncode, "counts": rx["counts"],
                          "all_fail_with_assertion": bool(per) and all(
                              v["status"] == "failure" and v["exception_types"]
                              and set(v["exception_types"]) <= ASSERTION_TYPES for v in per.values()),
                          "same_set_as_suite": sorted(per) == sorted(ncs), "per_control": per}
    res["files"] = [ju.name, out.name, jx.name, ox.name]
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--receipts-dir", type=Path, default=HERE / "receipts")
    a = ap.parse_args(argv)
    rec_dir = a.receipts_dir
    rec_dir.mkdir(parents=True, exist_ok=True)
    for f in rec_dir.glob("*"):
        if f.is_file() and (f.suffix in (".xml", ".txt") or f.name == "TEST_RUN.json"):
            f.unlink()
    date_u = lambda: subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True,  # noqa: E731
                                    text=True).stdout.strip() or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res = {"task": "DATA-05", "utc_start": date_u(), "host": platform.node(), "platform": platform.platform(),
           "python": sys.version.split()[0], "cpu_only": True}
    before = integrity()
    res["tests_hist"] = suite(rec_dir, "tests_hist", HERE)
    res["tests_int_regression"] = suite(rec_dir, "tests_int", CODE / "tests_int")
    after = integrity()
    res["integrity_before"] = {k: v for k, v in before.items() if k != "locked_sha256"}
    res["integrity_after"] = {k: v for k, v in after.items() if k != "locked_sha256"}
    b_, a_ = before["locked_sha256"], after["locked_sha256"]
    res["locked_changed_files"] = sorted(k for k in set(b_) | set(a_) if b_.get(k) != a_.get(k))
    res["locked_unchanged"] = not res["locked_changed_files"]
    res["locked_n_files"] = len(a_)
    res["locked_sha256"] = hashlib.sha256(json.dumps(a_, sort_keys=True).encode()).hexdigest()
    # snapshot bytes verified before and after; no __pycache__ CREATED by this run (pre-existing ones, e.g. written on
    # the server at 13:47Z by an earlier foreign python 3.11 process, are recorded and left in place)
    res["pycache_under_snap_v1_preexisting"] = before["pycache_dirs_under_snap_v1"]
    res["snapshot_unchanged"] = (before["verify_snapshot_exit_code"] == 0 and after["verify_snapshot_exit_code"] == 0
                                 and set(after["pycache_dirs_under_snap_v1"]) <= set(before["pycache_dirs_under_snap_v1"]))
    # TRAIN-03 (after G1: Ops wrote company/gates/G1_APPROVED.json): the suite never creates / changes / deletes it
    res["g1_gate_file_never_created"] = (before["g1_gate_file_exists"] == after["g1_gate_file_exists"]
                                         and before["g1_gate_file_sha256"] == after["g1_gate_file_sha256"])
    res["owned_sha256"] = {str(p.relative_to(CODE)).replace("\\", "/"): _sha(p) for d in OWNED
                           for p in sorted((CODE / d).rglob("*")) if p.is_file() and p.suffix in (".py", ".sh", ".ini")
                           and "__pycache__" not in p.parts}
    import torch  # noqa: E402  (version string only)
    res["torch"] = torch.__version__
    res["utc_end"] = date_u()
    res["receipt_sha256"] = {f: _sha(rec_dir / f) for s in ("tests_hist", "tests_int_regression")
                             for f in res[s]["files"] if (rec_dir / f).exists()}

    def ok_suite(s):
        return (s["pytest_exit_code"] == 0 and s["negative_controls_all_failed_as_intended"]
                and s["nc_runxfail"]["all_fail_with_assertion"] and s["nc_runxfail"]["same_set_as_suite"])
    ok = (ok_suite(res["tests_hist"]) and ok_suite(res["tests_int_regression"]) and res["snapshot_unchanged"]
          and res["locked_unchanged"] and res["g1_gate_file_never_created"] and not after["pycache_dirs_in_owned"])
    res["overall_exit_code"] = 0 if ok else 1
    (rec_dir / "TEST_RUN.json").write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    for s in ("tests_hist", "tests_int_regression"):
        print(s, json.dumps({k: res[s][k] for k in ("pytest_exit_code", "pytest_summary", "counts",
                                                     "negative_controls_all_failed_as_intended")}),
              json.dumps({k: res[s]["nc_runxfail"][k] for k in ("counts", "all_fail_with_assertion",
                                                                 "same_set_as_suite")}))
    print(json.dumps({k: res[k] for k in ("snapshot_unchanged", "locked_unchanged", "g1_gate_file_never_created",
                                          "overall_exit_code")}))
    return res["overall_exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
