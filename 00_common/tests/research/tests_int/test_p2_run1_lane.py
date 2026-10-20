"""P2-RUN1-LANE: company/plans/ops_scripts/p2/laneP2_RUN1.sh (plan mode, refusals, FAKE-runner launch). The real script is never
run on real data: PY is a fake interpreter that only writes status files. Proves: the plan lists P2-T3 first, the exact registered
WA cells (checked against launch_fl.p2_wa_recipes / p2_run_tag and read back by launch_fl.load_p2_recipe), the 8 C1 CALIB runs
(run ids = central_ft.c1_run_id) and 1 C2 run; every run goes to runs_p2 with the code tree code_p2r1 (never runs_ds3 for writing,
never runs_bench*); refusals (H_test.sh running, code tree missing / not read-only, bad root / slots / stage); recipe files
write-once and deterministic; incumbent reuse only on a BIT_IDENTICAL receipt; resume / skip / failure isolation; slots per GPU;
no outcome word in the driver output or the logs."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

import v4int.central_ft as CF
import v4int.launch_fl as F
import ppsi_paths

SCRIPT = ppsi_paths.PROJECT_ROOT.resolve() / "company" / "plans" / "ops_scripts" / "p2" / "laneP2_RUN1.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None or not SCRIPT.exists(), reason="bash / the script is not available here")
METRIC_WORDS = re.compile(r"\b(mrr|ndcg|recall|hit|hits|metric|metrics|value|values|score|scores|loss|auc|accuracy)\b", re.I)
NUMBER = re.compile(r"\b0\.\d{5,}")                  # a long decimal in the driver output / logs (hyper-parameters are short: 0.0001)

FAKE = r"""#!/usr/bin/env bash
shift
script=$1; shift
base=$(basename "$script")
rd=""; gpu="-"; out=""; dry=""; smoke=0; resume=0
while [ $# -gt 0 ]; do case $1 in --run-dir) rd=$2; shift;; --gpu) gpu=$2; shift;; --out) out=$2; shift;; --dry-run-out) dry=$2; shift;;
  --smoke) smoke=1;; --resume) resume=1;; esac; shift; done
echo "start $base $gpu ${rd:-$out} r$resume s$smoke d${dry:+1}" >> "$FAKELOG"
sleep 0.15
rc=0
case $base in
  p2_t3_check.py)
    v=${FAKE_T3:-BIT_IDENTICAL}; mkdir -p "$(dirname "$out")"
    echo "{\"test\": \"P2-T3\", \"verdict\": \"$v\", \"rounds\": 64}" > "$out"; [ "$v" = BIT_IDENTICAL ] || rc=1;;
  *)
    if [ -n "$dry" ]; then mkdir -p "$(dirname "$dry")"; echo '{}' > "$dry"
    else
      mkdir -p "$rd"; if [ "$base" = launch_fl.py ]; then echo '{}' > "$rd/ARM_SPEC.json"; else echo '{}' > "$rd/RUN_START.json"; fi
      if [ -n "${FAKE_FAIL:-}" ] && [ "$resume" = 0 ] && case $rd in *$FAKE_FAIL*) true;; *) false;; esac; then rc=1
      elif [ "$smoke" = 1 ]; then echo '{"status": "SMOKE_DONE_NOT_EVIDENCE"}' > "$rd/STATUS.json"
      else echo '{"status": "DONE_MAIN"}' > "$rd/RUN_SUMMARY.json"; echo '{"status": "DONE_MAIN"}' > "$rd/STATUS.json"; fi
    fi;;
esac
echo "end $base $gpu ${rd:-$out} rc$rc" >> "$FAKELOG"
exit $rc
"""


_TEMP_ROOTS = []


@pytest.fixture(autouse=True)
def _cleanup_fake_lanes():
    yield
    while _TEMP_ROOTS:
        root = _TEMP_ROOTS.pop()
        # The fake frozen tree is deliberately read-only, including its directories.
        # Unlock only directories created by this fixture before removing them.
        for p in root.rglob("*"):
            if p.is_dir():
                p.chmod(0o755)
        shutil.rmtree(root)


def _setup(code=True):
    sp = Path(tempfile.mkdtemp(prefix="p2run1"))
    _TEMP_ROOTS.append(sp)
    v = sp / "v"
    if code:
        for f in ("v4int/launch_fl.py", "v4int/central_ft.py", "v4int/c_full_small.py"):
            p = v / "code_p2r1" / f
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# stub\n")
            p.chmod(0o444)
        frozen = v / "code_p2r1"
        for p in frozen.rglob("*"):
            if p.is_dir():
                p.chmod(0o555)
        frozen.chmod(0o555)
    (v / "tools").mkdir(parents=True, exist_ok=True)
    (v / "tools" / "p2_t3_check.py").write_text("# stub\n")
    init = v / "runs_pretrain" / "PRETRAIN_FOR_FL_SMALL_SASREC_D64_B2_s2026" / "PRETRAIN_FOR_FL_INIT.json"
    init.parent.mkdir(parents=True)
    init.write_text("{}\n")
    fake = sp / "fakepy"
    fake.write_bytes(FAKE.replace("\r\n", "\n").encode())
    fake.chmod(0o755)
    log = sp / "fake.log"
    env = dict(os.environ, V=v.as_posix(), PY=fake.as_posix(), FAKELOG=log.as_posix(), POLL="0.1")
    env.pop("P2_RUNS", None)

    def go(*extra, **e):
        return subprocess.run([BASH, SCRIPT.as_posix(), *extra], capture_output=True, text=True, env=dict(env, **e), timeout=300)
    return sp, v, log, go


def parse(out):
    jobs, cur = [], None
    for ln in out.splitlines():
        m = re.match(r"^(\S+)\s+gpu=(\d+) soft-after=\[(.*)\] est=(.*)$", ln)
        if m:
            cur = {"name": m.group(1), "gpu": m.group(2), "soft": m.group(3), "est": m.group(4)}
            jobs.append(cur)
        elif cur is not None:
            for key in ("recipe", "dry", "cmd", "check", "out", "state", "note"):
                if ln.startswith(f"    {key}: "):
                    cur[key] = ln[len(f"    {key}: "):]
    return jobs


def test_plan_lists_t3_first_the_registered_wa_cells_8_c1_and_c2(tmp_path):
    sp, v, log, go = _setup(code=False)
    r = go()
    assert r.returncode == 0 and "# P2 RUN-1 lane plan: 31 jobs" in r.stdout and not (v / "runs_p2").exists()
    assert go("--plan").stdout == r.stdout                                           # plan is the default and deterministic
    jobs = parse(r.stdout)
    names = [j["name"] for j in jobs]
    assert names[0] == "p2t3" and len(names) == len(set(names)) == 31
    wa = [j for j in jobs if j["name"].startswith("wa.")]
    assert len(wa) == 21 and names[1:22] == [j["name"] for j in wa]
    assert {j["name"] for j in wa} == {"wa." + F.p2_run_tag(rec) for rec in F.p2_wa_recipes()}    # exactly the registered cells
    assert [j["name"] for j in wa] == ["wa." + F.p2_run_tag(rec) for rec in F.p2_wa_recipes()]    # in the registered order
    assert "CONDITIONAL" in wa[-1]["note"] and wa[-1]["name"] == "wa.FEDADAM_eta0.001_c0.1"
    c1 = [j for j in jobs if j["name"].startswith("c1.")]
    want = [CF.c1_run_id(var, "CALIBRATION", lr, 2026) for var in ("FULL", "FROZEN") for lr in (1e-5, 3e-5, 1e-4, 3e-4)]
    assert [Path(j["out"]).name for j in c1] == want and len(c1) == 8
    assert sum("--p2-freeze-item-tables" in j["cmd"] for j in c1) == 4
    c2 = [j for j in jobs if j["name"] == "c2.main"]
    assert len(c2) == 1 and Path(c2[0]["out"]).name == "P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026" and names[-1] == "c2.main"
    assert "--smoke" not in c2[0]["cmd"] and "--role" not in c2[0]["cmd"]
    assert "--smoke --smoke-rounds 65" in jobs[0]["cmd"] and "p2_t3_check.py" in jobs[0]["check"]
    assert all(j["soft"] == "p2t3" for j in jobs[1:]) and jobs[0]["soft"] == ""
    # --c2-early puts C2 right after P2-T3
    assert [j["name"] for j in parse(go("--c2-early").stdout)][:2] == ["p2t3", "c2.main"]
    # the stage filter
    assert len(parse(go("--stages", "c1").stdout)) == 8


def test_every_run_goes_to_runs_p2_and_code_p2r1(tmp_path):
    sp, v, log, go = _setup(code=False)
    jobs = parse(go("--gpus", "0 1").stdout)
    root = v.as_posix() + "/runs_p2/"
    for j in jobs:
        assert j["out"].startswith(root), j["name"]
        for part in (j.get("dry"), j["cmd"], j.get("check")):
            if part is None:
                continue
            assert not re.search(r"runs_bench", part), j["name"]
            for rd in re.findall(r"--run-dir (\S+)", part) + re.findall(r"--dry-run-out (\S+)", part) + re.findall(r"--out (\S+)", part):
                assert rd.startswith(root), (j["name"], rd)
        assert " -B " + v.as_posix() + "/code_p2r1/v4int/" in j["cmd"], j["name"]
        assert "--gpu " + j["gpu"] in j["cmd"]
        if "recipe" in j:
            assert j["recipe"].startswith(root + "recipes/") and f"--p2-recipe {j['recipe']}" in j["cmd"]
    # runs_ds3 appears only as the READ-ONLY search root of the P2-T3 check
    for j in jobs[1:]:
        assert "runs_ds3" not in j["cmd"] + j.get("dry", "")
    assert re.findall(r"--s8tc-search-root (\S+)", jobs[0]["check"]) == [v.as_posix() + "/runs_ds3/S7S8_T"]
    assert jobs[0]["check"].count("runs_ds3") == 1
    assert {j["gpu"] for j in jobs} == {"0", "1"}
    assert {j["gpu"] for j in parse(go("--gpus", "3").stdout)} == {"3"}


def test_refusals(tmp_path):
    sp, v, log, go = _setup()
    # DL 459 expanded the registered lane limit from 4 to 16 slots per GPU.
    for slots in ("0", "17", "x"):
        assert go("--launch", "--slots-per-gpu", slots).returncode == 2
    for slots in ("1", "16"):
        assert go("--plan", "--slots-per-gpu", slots).returncode == 0
    assert go("--launch", "--gpus", "x").returncode == 2
    assert go("--plan", "--stages", "wb").returncode == 2
    r = go("--launch", P2_RUNS=(v / "runs_ds3").as_posix())
    assert r.returncode == 2 and "runs_ds3" in r.stderr
    assert go("--launch", P2_RUNS=(v / "runs_bench_gpu").as_posix()).returncode == 2
    assert go("--launch", CODE=(v / "code_ds3r7").as_posix()).returncode == 2                     # only code_p2r1
    # an H step is running: refused before anything else, nothing written
    hdir = sp / "h"
    hdir.mkdir()
    (hdir / "H_test.sh").write_text("sleep 90\n")
    p = subprocess.Popen([BASH, (hdir / "H_test.sh").as_posix()])
    try:
        r = go("--launch")
        assert r.returncode == 2 and "H step" in r.stderr and not (v / "runs_p2").exists()
        assert "WARNING" in go("--plan").stdout                                                    # plan mode only warns
    finally:
        p.kill(); p.wait()
    # code tree missing / a writable file inside
    sp2, v2, log2, go2 = _setup(code=False)
    r = go2("--launch")
    assert r.returncode == 2 and "missing" in r.stderr and not (v2 / "runs_p2").exists()
    sp3, v3, log3, go3 = _setup()
    (v3 / "code_p2r1" / "v4int" / "launch_fl.py").chmod(0o644)
    r = go3("--launch")
    assert r.returncode == 2 and "not read-only" in r.stderr and not (v3 / "runs_p2").exists()
    (v3 / "code_p2r1" / "v4int" / "launch_fl.py").chmod(0o444)
    assert go3("--launch", FAKE_T3="BIT_IDENTICAL").returncode == 0


def _runs(log):
    return [x.split() for x in log.read_text().splitlines() if x]


def test_fake_launch_order_slots_recipes_reuse_resume_and_value_free(tmp_path):
    sp, v, log, go = _setup()
    r = go("--launch", "--gpus", "0 1")
    assert r.returncode == 0, r.stdout + r.stderr
    rows = _runs(log)
    starts = [x for x in rows if x[0] == "start"]
    # P2-T3 first: its smoke, then its check, before any other job starts
    assert starts[0][1] == "launch_fl.py" and starts[0][5] == "s1" and starts[1][1] == "p2_t3_check.py"
    end_check = next(i for i, x in enumerate(rows) if x[0] == "end" and x[1] == "p2_t3_check.py")
    first_other = next(i for i, x in enumerate(rows) if x[0] == "start" and x[1] != "p2_t3_check.py" and not (x[1] == "launch_fl.py" and x[5] == "s1"))
    assert end_check < first_other
    real = [x for x in starts if x[6] == "d" and x[1] != "p2_t3_check.py" and x[5] == "s0"]
    rds = sorted(Path(x[3]).name for x in real)
    expect = sorted([ "P2_W_" + F.p2_run_tag(rec) + "_CALIB_s2026" for rec in F.p2_wa_recipes()][:-1]            # incumbent REUSED
                    + [CF.c1_run_id(var, "CALIBRATION", lr, 2026) for var in ("FULL", "FROZEN") for lr in (1e-5, 3e-5, 1e-4, 3e-4)]
                    + ["P2_C2_CFULL_SMALL_SASREC_D64_B2_s2026"])
    assert rds == expect and len(rds) == 29
    assert "REUSED" in r.stdout and "wa.FEDADAM_eta0.001_c0.1" in r.stdout
    assert all(x[3].startswith((v / "runs_p2").as_posix() + "/") for x in starts)
    assert (v / "runs_p2" / "P2T3" / "P2T3_RECEIPT.json").exists() and not (v / "runs_ds3").exists()
    # dry runs: C1 + C2 only, before their launches
    dry = [x for x in starts if x[6] == "d1"]
    assert len(dry) == 9
    # slots: at most 2 jobs per GPU at any time
    running, peak = {}, {}
    for x in rows:
        if x[1] == "p2_t3_check.py" or "/dry/" in x[3]:
            continue
        g = x[2]
        running[g] = running.get(g, 0) + (1 if x[0] == "start" else -1)
        peak[g] = max(peak.get(g, 0), running[g])
    assert set(peak) == {"0", "1"} and max(peak.values()) <= 2
    # recipe files: deterministic, write-once, read by the REAL launcher loader, equal to the registered WA recipes
    rec_dir = v / "runs_p2" / "recipes"
    files = sorted(rec_dir.glob("*.json"))
    assert len(files) == 22                                                                         # 21 cells + the P2-T3 copy
    got = {F.load_p2_recipe(f)[1]["label"]: F.load_p2_recipe(f)[0] for f in files if not f.name.startswith("P2T3")}
    assert got == {F.p2_label(rec): rec for rec in F.p2_wa_recipes()}
    t3 = F.load_p2_recipe(rec_dir / "P2T3_INCUMBENT_RECIPE.json")[0]
    assert t3 == F.p2_wa_recipes()[-1]
    before = {f.name: f.read_bytes() for f in files}
    # markers, logs, lock
    st = v / "runs_p2" / "_lane_state"
    assert (st / "p2t3.ok").exists() and not list(st.glob("*.fail")) and not (st / "DRIVER.lock").exists()
    assert (v / "runs_p2" / "logs" / "lane_summary.tsv").exists()
    # value-free driver output, plan and logs
    texts = r.stdout + r.stderr + go().stdout + "".join(p.read_text() for p in (v / "runs_p2" / "logs").glob("*"))
    assert not METRIC_WORDS.search(texts), METRIC_WORDS.search(texts).group(0)
    assert not NUMBER.search(r.stdout + r.stderr + "".join(p.read_text() for p in (v / "runs_p2" / "logs").glob("*")))
    # relaunch: nothing runs again (finished runs skipped), recipe files untouched
    log.write_text("")
    r2 = go("--launch", "--gpus", "0 1")
    assert r2.returncode == 0 and log.read_text() == "", r2.stdout
    assert before == {f.name: f.read_bytes() for f in rec_dir.glob("*.json")}
    # a changed recipe file is a refusal, not an overwrite
    f0 = sorted(rec_dir.glob("P2_W_FEDAVG_eta0.3_c0.003.json"))[0]
    f0.chmod(0o644); f0.write_text("{}\n")
    assert go("--launch").returncode == 2


def test_incumbent_rerun_when_p2t3_fails_and_failure_isolation_then_resume(tmp_path):
    sp, v, log, go = _setup()
    r = go("--launch", "--gpus", "0 1", FAKE_T3="FAIL")
    assert r.returncode == 0 and "REUSED" not in r.stdout
    assert any("P2_W_FEDADAM_eta0.001_c0.1_CALIB_s2026" in x[3] and x[0] == "start" for x in _runs(log))
    assert len([x for x in _runs(log) if x[0] == "start" and x[5] == "s0" and x[6] == "d" and x[1] != "p2_t3_check.py"]) == 30
    # isolation: a failing C1 run does not stop the others; the lane exits 1 and reports it; a relaunch resumes ONLY it
    sp, v, log, go = _setup()
    r = go("--launch", "--gpus", "0 1", FAKE_FAIL="C1_FT_CENTRAL_FROZEN_CALIB_lr1e-4")
    assert r.returncode == 1 and "FAILED" in r.stdout, r.stdout + r.stderr
    st = v / "runs_p2" / "_lane_state"
    assert [p.name for p in st.glob("*.fail")] == ["c1.FROZEN.1e-4.fail"]
    assert (st / "c2.main.ok").exists() and (st / "c1.FROZEN.3e-4.ok").exists()
    log.write_text("")
    r2 = go("--launch", "--gpus", "0 1")
    assert r2.returncode == 0, r2.stdout + r2.stderr
    starts = [x for x in _runs(log) if x[0] == "start" and x[6] == "d" and x[1] != "p2_t3_check.py"]
    assert len(starts) == 1 and "FROZEN_CALIB_lr1e-4" in starts[0][3] and starts[0][4] == "r1"        # launcher's own --resume
    assert list(st.glob("c1.FROZEN.1e-4.fail.*")) and not list(st.glob("*.fail"))                    # old failure kept aside
    texts = r.stdout + r2.stdout + "".join(p.read_text() for p in (v / "runs_p2" / "logs").glob("*"))
    assert not METRIC_WORDS.search(texts)


def test_one_lane_at_a_time_and_stop_file(tmp_path):
    sp, v, log, go = _setup()
    st = v / "runs_p2" / "_lane_state"
    st.mkdir(parents=True)
    holder = subprocess.Popen([BASH, "-c", "echo $$; exec sleep 20"], stdout=subprocess.PIPE, text=True)
    try:
        pid = holder.stdout.readline().strip()
        (st / "DRIVER.lock").mkdir()
        (st / "DRIVER.lock" / "pid").write_text(pid)
        r = go("--launch")
        assert r.returncode == 2 and "lock held" in r.stderr
    finally:
        holder.kill(); holder.wait()
    (st / "DRIVER.lock" / "pid").write_text("999999")                                               # stale lock: taken over
    (st / "STOP").write_text("")
    r = go("--launch")
    assert r.returncode == 3 and not log.exists() or log.read_text() == ""                          # STOP: launches nothing new
    (st / "STOP").unlink()
    assert go("--launch").returncode == 0
