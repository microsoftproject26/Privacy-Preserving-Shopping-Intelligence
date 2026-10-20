"""run_wave1.sh: plan mode (job list, dependency order, resource caps, no lock / final) and a FAKE-runner launch that checks the
slot limit, the dependency order and the failure propagation. The real script is never launched against real data."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "company" / "plans" / "ops_scripts" / "bench" / "run_wave1.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None or not SCRIPT.exists(), reason="bash / the script is not available here")


def plan(tmp_path, *args):
    env = dict(os.environ, V=(tmp_path / "v").as_posix())
    r = subprocess.run([BASH, SCRIPT.as_posix(), *args], capture_output=True, text=True, env=env, timeout=60)
    return r


def parse(out):
    jobs, cur = [], None
    for ln in out.splitlines():
        m = re.match(r"^(\S+)\s+w=(\d+) deps=\[(.*)\]$", ln)
        if m:
            cur = {"name": m.group(1), "w": int(m.group(2)), "deps": m.group(3).split()}
            jobs.append(cur)
        elif ln.startswith("    ") and cur is not None:
            cur["cmd"] = ln.strip()
    return jobs


def test_plan_is_the_default_and_launches_nothing(tmp_path):
    r = plan(tmp_path)
    assert r.returncode == 0 and "# Wave-1 plan" in r.stdout and not (tmp_path / "v").exists()
    assert plan(tmp_path, "--plan").stdout == r.stdout


def test_plan_dependency_order_resources_and_scope(tmp_path):
    r = plan(tmp_path, "--plan")
    jobs = parse(r.stdout)
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names)) > 200
    pos = {n: i for i, n in enumerate(names)}
    for j in jobs:
        assert all(pos[d] < pos[j["name"]] for d in j["deps"]), j["name"]          # topological
        assert 1 <= j["w"] <= 6
        assert "nice -n 15" in j["cmd"] or j["cmd"].startswith("win2027")
        assert " lock " not in j["cmd"] and " final " not in j["cmd"] and "--final" not in j["cmd"]   # stops before lock / final
        m = re.search(r"--workers (\d+)", j["cmd"])
        assert j["w"] == (int(m.group(1)) if m else 1) and j["w"] * 16 <= 96
    for D in ("s3_beauty", "s3_sports", "s3_toys", "ml_1m"):
        for key in (f"manifest.{D}", f"decide.{D}", f"scal.{D}.2026", f"scal.{D}.2027", f"seed3.{D}", f"tbsel.star.{D}", f"win.pre.{D}",
                    f"cfull.{D}.2026", f"cfull.{D}.2027", f"ft.{D}.2027", f"dp8.{D}.2026", f"fa1024.{D}.2027",
                    f"ph2.PF.{D}.2026", f"ph2.FA_Q8.{D}.2027", f"ph2.FP.{D}.2026"):
            assert key in pos, key
        assert pos[f"decide.{D}"] > max(pos[f"fawarm.{D}.{s}"] for s in (2026, 2027))
        for sd in (2026, 2027):                                       # B12: S_CAL per PRE seed, each run binds its own
            assert pos[f"scal.{D}.{sd}"] > pos[f"decide.{D}"] and pos[f"scal.{D}.{sd}"] > pos[f"pre.{D}.{sd}"]
            assert pos[f"dp8.{D}.{sd}"] > pos[f"scal.{D}.{sd}"] and pos[f"fa1024.{D}.{sd}"] > pos[f"scal.{D}.{sd}"]
            assert f"scal.{D}.{sd}" in next(j for j in jobs if j["name"] == f"dp8.{D}.{sd}")["deps"]
        assert pos[f"seed3.{D}"] > pos[f"decide.{D}"] and pos[f"tbsel.star.{D}"] > pos[f"decide.{D}"]
        assert pos[f"win.cfull.{D}"] > pos[f"tbsel.cfull.{D}"]
    cmds = {j["name"]: j["cmd"] for j in jobs}
    assert not [n for n in names if n.startswith("tb.cfull.ml_1m.96")]                    # B10 Q4
    assert any(n.startswith("tb.cfull.s3_beauty.96") for n in names)
    assert any(n.startswith("tb.pre.ml_1m.96") for n in names)                            # PRE keeps {6, 24, 96}
    assert any(".200." in n for n in names if "ml_1m" in n) and not any(".200." in n for n in names if "beauty" in n)
    assert "--budget-efe" not in cmds["tb.fa.s3_beauty.6.50.0.2"] and "--budget-efe" not in cmds["tb.star.s3_beauty.6.50.0.2"]
    assert "PRE__B6_L50_D0.2_s2026" in cmds["tb.star.s3_beauty.6.50.0.2"]                 # warm cell: PRE of the SAME cell
    assert "PRE_s2026" in cmds["tb.star.s3_beauty.6.50.0.5"] or "PRE__B6_L50_D0.5_s2026" in cmds["tb.star.s3_beauty.6.50.0.5"]
    assert "--s-record" in cmds["dp8.s3_toys.2027"] and "$(srec s3_toys 2027)" in cmds["dp8.s3_toys.2027"]
    assert "--pre-run" in cmds["ph2.PF.s3_toys.2027"] and "PRE_s2027" in cmds["ph2.PF.s3_toys.2027"]


def test_slot_limit_is_capped_at_96_threads(tmp_path):
    r = plan(tmp_path, "--plan", "--max-slots", "7")
    assert r.returncode == 2 and "96 threads" in r.stderr


FAKE = r"""#!/usr/bin/env bash
if [ "$1" = "-c" ]; then exec "$REAL_PY" "$@"; fi
shift 3                                  # -B -m v4bench.run
sub=$1; shift
W=1; arm=""; seed=""; ds=""; b=6; ml=50; dr=""; RS=0
while [ $# -gt 0 ]; do case $1 in --workers) W=$2; shift;; --arm) arm=$2; shift;; --seed) seed=$2; shift;; --dataset) ds=$2; shift;;
  --budget-efe) b=$2; shift;; --max-len) ml=$2; shift;; --dropout) dr=$2; shift;; --resume) RS=1;; esac; shift; done
R=$V/runs_bench/$ds
echo "start $W $sub ${arm:--} ${seed:--} rs$RS" >> "$FAKELOG"
sleep 0.3
if [ -n "${FAKE_FAIL:-}" ] && [ "$sub" = run ] && [ "$arm" = "${FAKE_FAIL%%:*}" ] && [ "$seed" = "${FAKE_FAIL##*:}" ]; then
  echo "end $W $sub ${arm:--} ${seed:--} FAIL" >> "$FAKELOG"; exit 1; fi
mkdir -p "$R/dp" "$R/trackb"
case $sub in
  manifest) echo '{}' > "$R/MANIFEST.json";;
  decide-warm) echo '{"FA_WARM_STAR": "FA_WARM_FROZEN"}' > "$R/FA_WARM_DECISION.json";;
  seed3) echo '{"third_seed_2028_needed_for": []}' > "$R/SEED3_DECISION.json";;
  trackb-select) a=$arm; cell='[24, 50, 0.2]'; case $a in FA|FA_WARM*) cell='[6, 50, 0.2]';; esac
     echo "{\"winner_cell\": $cell, \"winner_run\": \"x\"}" > "$R/trackb/WINNER_$a.json";;
  run) label=$arm; if [ "$ml" != 50 ] || [ -n "$dr" ] || [ "$b" != 6 ]; then label="${arm}__B${b}_L${ml}_D${dr:-0.1}"; fi
     mkdir -p "$R/${label}_s$seed"; echo '{"status": "DONE_MAIN"}' > "$R/${label}_s$seed/RESULT.json"
     case $arm in S_CAL*) echo '{}' > "$R/dp/S_RECORD_FA_WARM_FROZEN_s$seed.json";; esac;;
esac
echo "end $W $sub ${arm:--} ${seed:--} ok" >> "$FAKELOG"
"""


def _setup():
    import tempfile
    sp = Path(tempfile.mkdtemp(prefix="bm02w1"))                      # the script assumes space-free paths (as on the server)
    v = sp / "v"
    (v / "code_bench4").mkdir(parents=True)
    log = sp / "fake.log"
    fake = sp / "fakepy"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    env = dict(os.environ, V=v.as_posix(), PY=fake.as_posix(), REAL_PY=Path(sys.executable).as_posix(),
               FAKELOG=log.as_posix(), POLL="0.2")

    def go(*extra, fail=None):
        e = dict(env, **({"FAKE_FAIL": fail} if fail else {}))
        return subprocess.run([BASH, SCRIPT.as_posix(), *extra, "--datasets", "s3_toys"], capture_output=True, text=True,
                              env=e, timeout=600)
    return v, log, go


def test_launch_with_a_fake_runner_respects_slots_order_and_failures(tmp_path):
    v, log, go = _setup()
    r = go("--launch")
    assert r.returncode == 0 and "failed or skipped: 0" in r.stdout, r.stdout + r.stderr
    lines = [x.split() for x in log.read_text().splitlines()]
    running = peak = 0
    for ev, w, *_ in lines:
        running += int(w) if ev == "start" else -int(w)
        peak = max(peak, running)
    assert 1 < peak <= 6
    starts = [x for x in lines if x[0] == "start"]
    order = [(x[2], x[3], x[4]) for x in starts]
    first = lambda sub, arm, seed: order.index((sub, arm, seed))                       # noqa: E731
    assert first("decide-warm", "-", "-") > max(first("run", "FA_WARM", "2027"), first("run", "FA_WARM_FROZEN", "2027"))
    for sd in ("2026", "2027"):                                                         # B12: S_CAL per seed before its DP runs
        assert first("run", "S_CAL_WARM_FROZEN", sd) > first("decide-warm", "-", "-")
        assert first("run", "DP8_WARM_FROZEN", sd) > first("run", "S_CAL_WARM_FROZEN", sd)
        assert first("run", "FA_1024_WARM_FROZEN", sd) > first("run", "S_CAL_WARM_FROZEN", sd)
    assert first("seed3", "-", "-") > first("decide-warm", "-", "-")
    names = [x[3] for x in starts if x[2] == "run"]
    assert "FT_FA_WARM_FROZEN" in names and "PF_WARM_FROZEN" in names and "FA_Q8_WARM_FROZEN" in names
    assert not [x for x in starts if x[2] in ("lock", "final")]
    log.write_text("")                                      # a second launch skips every finished run (write-once)
    r2 = go("--launch")
    assert r2.returncode == 0 and log.read_text() == ""
    # failure propagation (fresh state): a failing FA_WARM_FROZEN s2026 stops decide-warm and everything downstream
    shutil.rmtree(v / "runs_bench")
    log.write_text("")
    r3 = go("--launch", fail="FA_WARM_FROZEN:2026")
    assert r3.returncode == 0 and "failed or skipped" in r3.stdout and "failed or skipped: 0" not in r3.stdout
    arms = {(x[2], x[3], x[4]) for x in [y.split() for y in log.read_text().splitlines()] if x[0] == "start"}
    assert ("run", "C_FULL", "2027") in arms and ("run", "FA", "2027") in arms             # other branches continue
    assert ("decide-warm", "-", "-") not in arms and not any(a[1].startswith("DP8") for a in arms)
    assert not any(a[0] == "seed3" for a in arms) and not any(a[1].startswith("S_CAL") for a in arms)
    st = v / "runs_bench" / "_wave1_state"
    assert (st / "decide.s3_toys.fail").exists() and (st / "dp8.s3_toys.2026.fail").exists()
    assert (st / "cfull.s3_toys.2027.ok").exists()


def test_resume_reruns_only_failed_jobs_and_never_touches_finished_runs(tmp_path):
    v, log, go = _setup()
    r1 = go("--launch", fail="DP8_WARM_FROZEN:2027")                    # the s2027 DP8 job fails; nothing is downstream of it
    assert "failed or skipped: 1" in r1.stdout, r1.stdout
    before = {p: p.stat().st_mtime_ns for p in (v / "runs_bench" / "s3_toys").glob("*/RESULT.json")}
    assert before and not (v / "runs_bench" / "s3_toys" / "DP8_WARM_FROZEN_s2027" / "RESULT.json").exists()
    log.write_text("")
    r2 = go("--resume")
    assert r2.returncode == 0 and "failed or skipped: 0" in r2.stdout, r2.stdout
    starts = [x.split() for x in log.read_text().splitlines() if x.startswith("start")]
    assert [(x[3], x[4], x[5]) for x in starts] == [("DP8_WARM_FROZEN", "2027", "rs1")]   # only that run, with run --resume
    assert all(p.stat().st_mtime_ns == m for p, m in before.items())                      # finished runs untouched
    assert (v / "runs_bench" / "s3_toys" / "DP8_WARM_FROZEN_s2027" / "RESULT.json").exists()
    r3 = go("--resume")                                                                   # all done: nothing runs again
    assert r3.returncode == 0 and len([x for x in log.read_text().splitlines() if x.startswith("start")]) == 1
    # a failed upstream job: its dependents are re-run too (a seed-2027 S_CAL failure stops FA_1024 / DP8 s2027)
    shutil.rmtree(v / "runs_bench")
    log.write_text("")
    r4 = go("--launch", fail="S_CAL_WARM_FROZEN:2027")
    assert "failed or skipped" in r4.stdout and "failed or skipped: 0" not in r4.stdout
    log.write_text("")
    go("--resume")
    got = sorted((x.split()[3], x.split()[4]) for x in log.read_text().splitlines() if x.startswith("start"))
    assert got == [("DP8_WARM_FROZEN", "2027"), ("FA_1024_WARM_FROZEN", "2027"), ("S_CAL_WARM_FROZEN", "2027")], got
