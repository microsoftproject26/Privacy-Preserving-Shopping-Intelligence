"""MODEL-01 runner: the full model test suite (pytest, CPU, strict FP32), the negative-control intent check, the
diff-document check and a few measured values, with every exit code recorded.

  python -B code/tests_models/run_all.py

Steps (each in a fresh interpreter with -B and PYTHONDONTWRITEBYTECODE=1, pytest cache provider off):
  1. snapshot hashes + no __pycache__ under snap_v1 (before);
  2. pytest over code/tests_models (junit + full output);
  3. pytest --runxfail on the negative controls only: every test_nc_* must now FAIL with AssertionError (proves each
     control fails at its check, not by crashing) — this step's non-zero exit code is the expected outcome;
  4. python -m v4models.diffdoc --check;
  5. measured values for the handoff (T19 drift / memorization numbers, recentering idempotence at K = 64);
  6. snapshot hashes + no __pycache__ under snap_v1 (after).
Writes only into code/tests_models/receipts/.
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
REC = HERE / "receipts"
GENERATED = ("pytest_output.txt", "junit.xml", "nc_runxfail_output.txt", "nc_runxfail_junit.xml",
             "diffdoc_check.txt", "MEASURED.json", "TEST_RUN.json")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run(args, **kw):
    return subprocess.run([sys.executable, "-B", *args], cwd=str(CODE), env=ENV, capture_output=True, text=True, **kw)


def snapshot_state() -> dict:
    p = _run(["-c", "import ppsi_env, json; ppsi_env.verify_snapshot(); print('OK')"])
    caches = [str(c) for c in (CODE / "snap_v1").rglob("__pycache__")]
    return {"verify_snapshot_exit_code": p.returncode, "verify_snapshot_stdout": p.stdout.strip()[-200:],
            "stderr_tail": p.stderr.strip()[-500:], "pycache_dirs_under_snap_v1": caches,
            "manifest_sha256": _sha(CODE / "snap_v1" / "SNAPSHOT_MANIFEST.json")}


def parse_junit(path: Path) -> dict:
    root = ET.parse(path).getroot()
    cases = []
    for tc in root.iter("testcase"):
        status, msg, exc_types = "passed", None, []
        for child in tc:
            if child.tag in ("failure", "error"):
                status, msg = child.tag, (child.get("message") or "")[:300]
                exc_types = sorted(set(re.findall(r"^E\s+(\w+(?:Error|Exception))\b", child.text or "", flags=re.M)))
            elif child.tag == "skipped":
                status = "xfailed" if child.get("type") == "pytest.xfail" else "skipped"
                msg = (child.get("message") or "")[:300]
        cases.append({"name": f"{tc.get('classname')}::{tc.get('name')}", "status": status, "message": msg,
                      "exception_types": exc_types, "time_s": float(tc.get("time", 0))})
    counts = {}
    for c in cases:
        counts[c["status"]] = counts.get(c["status"], 0) + 1
    return {"counts": counts, "n_cases": len(cases), "cases": cases}


MEASURE = r'''
import json, sys
sys.path.insert(0, r"{here}")
import ppsi_env, torch
from fl.numerics import enforce_strict_fp32
enforce_strict_fp32(); torch.set_num_threads(1)
import test_t19_memorization_monitors as t19
import test_build_adapter as tb
out = {{}}
for fam in ("GRU", "SASREC"):
    a, hist = t19._memorization_check(fam)
    out["memorization_" + fam] = {{"steps": t19.STEPS, "loss_first": hist["loss"][0], "loss_last": hist["loss"][-1],
                                   "grad_norm_max_preclip": max(hist["gn"]), "grad_norm_min_preclip": min(hist["gn"])}}
for rc in (True, False):
    d = t19._drift_run(recenter=rc)
    out["sasrec_head_common_mode_30_steps_recenter_" + str(rc)] = {{
        "first": d[0]["common_mode_norm"], "last": d[-1]["common_mode_norm"], "max_abs_b_mean": max(abs(h["b_mean"]) for h in d),
        "last_ratio_to_median_row_norm": d[-1]["ratio"]}}
bm = tb.build("SASREC", tb.SCHEMA, 2026, K=64)
s0 = bm.theta0(); bm.adapter.server_post_aggregate(); s1 = bm.adapter.broadcast_state()
from fl.numerics import state_digest
out["sasrec_K64_second_recentering"] = {{"max_abs_change": max(float((s0[k] - s1[k]).abs().max()) for k in ("output_embed", "output_bias")),
                                        "bitwise_equal": state_digest(s1) == bm.init_sha256}}
print(json.dumps(out))
'''


def main() -> int:
    REC.mkdir(exist_ok=True)
    for f in GENERATED:
        (REC / f).unlink(missing_ok=True)
    res = {"task": "MODEL-01 + SCHEMA-01", "utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": platform.node(), "platform": platform.platform(), "python": sys.version.split()[0],
           "cpu_only": True}
    res["snapshot_before"] = snapshot_state()

    t0 = time.time()
    p = _run(["-m", "pytest", str(HERE), "-q", "-rA", "-p", "no:cacheprovider", "--junitxml", str(REC / "junit.xml")])
    (REC / "pytest_output.txt").write_text(p.stdout + "\n--- stderr ---\n" + p.stderr, encoding="utf-8")
    junit = parse_junit(REC / "junit.xml")
    ncs = {c["name"]: ("FAILED_AS_INTENDED" if c["status"] == "xfailed" else f"UNEXPECTED_{c['status'].upper()}")
           for c in junit["cases"] if c["name"].split("::")[-1].split("[")[0].startswith("test_nc_")}
    summary = [ln for ln in p.stdout.strip().splitlines() if " in " in ln and ("passed" in ln or "failed" in ln)]
    res.update({"pytest_exit_code": p.returncode, "pytest_summary": summary[-1] if summary else None,
                "pytest_wall_s": round(time.time() - t0, 1), "counts": junit["counts"],
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
    # a bare `assert` shows no exception name in the report; any named exception must be AssertionError
    res["nc_runxfail"] = {"exit_code_expected_nonzero": q.returncode, "counts": rx["counts"],
                          "all_fail_with_assertion": bool(intent) and all(
                              v["status"] == "failure" and set(v["exception_types"]) <= {"AssertionError"}
                              for v in intent.values()),
                          "same_set_as_suite": sorted(intent) == sorted(ncs), "per_control": intent}

    d = _run(["-m", "v4models.diffdoc", "--check"])
    (REC / "diffdoc_check.txt").write_text(d.stdout + d.stderr, encoding="utf-8")
    res["diffdoc_check_exit_code"] = d.returncode

    m = _run(["-c", MEASURE.format(here=str(HERE))])
    res["measure_exit_code"] = m.returncode
    if m.returncode == 0:
        (REC / "MEASURED.json").write_text(json.dumps(json.loads(m.stdout.strip().splitlines()[-1]), indent=1),
                                           encoding="utf-8")
    else:
        res["measure_stderr_tail"] = m.stderr[-2000:]

    res["snapshot_after"] = snapshot_state()
    res["snapshot_unchanged"] = (res["snapshot_before"]["verify_snapshot_exit_code"] == 0
                                 and res["snapshot_after"]["verify_snapshot_exit_code"] == 0
                                 and not res["snapshot_after"]["pycache_dirs_under_snap_v1"])
    import torch  # noqa: E402  (version string only)
    res["torch"] = torch.__version__
    res["utc_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["receipt_sha256"] = {f: _sha(REC / f) for f in GENERATED if f != "TEST_RUN.json" and (REC / f).exists()}
    (REC / "TEST_RUN.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    ok = (p.returncode == 0 and res["negative_controls_all_failed_as_intended"] and res["nc_runxfail"]
          ["all_fail_with_assertion"] and res["nc_runxfail"]["same_set_as_suite"] and d.returncode == 0
          and m.returncode == 0 and res["snapshot_unchanged"])
    print(json.dumps({k: res[k] for k in ("pytest_exit_code", "pytest_summary", "counts",
                                          "negative_controls_all_failed_as_intended", "diffdoc_check_exit_code",
                                          "measure_exit_code", "snapshot_unchanged")}))
    print("nc_runxfail:", json.dumps({k: res["nc_runxfail"][k] for k in ("counts", "all_fail_with_assertion",
                                                                          "same_set_as_suite")}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
