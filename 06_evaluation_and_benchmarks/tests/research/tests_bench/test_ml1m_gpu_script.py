"""run_ml1m_gpu.sh (BM-04): plan mode, --device cuda only on `run` lines, the fresh runs_bench_gpu root, the refusals, and a FAKE-runner
launch (device / gpu / runs-root reach the runner, slot cap, nothing written to runs_bench). The real script is never run on real data."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import ppsi_paths

SCRIPT = ppsi_paths.PROJECT_ROOT.resolve() / "company" / "plans" / "ops_scripts" / "bench" / "run_ml1m_gpu.sh"
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None or not SCRIPT.exists(), reason="bash / the script is not available here")


def sh(tmp_path, *args, **env_extra):
    env = dict(os.environ, V=(tmp_path / "v").as_posix(), **env_extra)
    return subprocess.run([BASH, SCRIPT.as_posix(), *args], capture_output=True, text=True, env=env, timeout=60)


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


def test_plan_is_default_prints_the_dag_and_launches_nothing(tmp_path):
    r = sh(tmp_path)
    assert r.returncode == 0 and "# ML-1M GPU plan" in r.stdout and not (tmp_path / "v").exists()
    assert sh(tmp_path, "--plan").stdout == r.stdout
    jobs = parse(r.stdout)
    names = [j["name"] for j in jobs]
    assert len(names) == len(set(names)) == 73
    pos = {n: i for i, n in enumerate(names)}
    for j in jobs:
        assert all(pos[d] < pos[j["name"]] for d in j["deps"]), j["name"]
        assert " lock " not in j["cmd"] and " final " not in j["cmd"]
    for key in ("manifest.ml_1m", "decide.ml_1m", "scal.ml_1m.2027", "seed3.ml_1m", "tbsel.star.ml_1m", "win.pre.ml_1m",
                "ft.ml_1m.2026", "dp8.ml_1m.2026", "fa1024.ml_1m.2027", "ph2.PF.ml_1m.2026"):
        assert key in pos, key
    assert not any("beauty" in n or "sports" in n or "toys" in n for n in names)


def test_run_lines_carry_device_gpu_root_and_other_lines_do_not(tmp_path):
    jobs = parse(sh(tmp_path, "--plan", "--gpus", "0 1").stdout)
    root = (tmp_path / "v").as_posix() + "/runs_bench_gpu"
    runs = [j for j in jobs if " v4bench.run run " in j["cmd"] or j["cmd"].startswith("RUNFLAGS=")]
    others = [j for j in jobs if j not in runs]
    assert len(runs) > 50 and {j["name"].split(".")[0] for j in others} >= {"manifest", "decide", "seed3", "tbsel"}
    for j in runs:
        assert "--device cuda --gpu " in j["cmd"] and f"--runs-root {root}" in j["cmd"], j["name"]
    for j in others:
        assert "--device" not in j["cmd"] and "--gpu" not in j["cmd"] and f"--runs-root {root}" in j["cmd"], j["name"]
    for j in jobs:
        assert not re.search(r"runs_bench(/|\s|$)", j["cmd"]), j["name"]
        assert "code_bench4" not in j["cmd"]
    win = [j for j in jobs if j["name"].startswith("win.")]
    assert len(win) == 4 and all(j["cmd"].startswith('RUNFLAGS="--device cuda --gpu ') for j in win)
    gpus = {re.search(r"--gpu (\d+)", j["cmd"]).group(1) for j in runs}
    assert gpus == {"0", "1"}                                                  # alternation over two GPUs, by job index
    g1 = {re.search(r"--gpu (\d+)", j["cmd"]).group(1) for j in parse(sh(tmp_path, "--plan").stdout) if "--gpu" in j["cmd"]}
    assert g1 == {"0"}                                                         # default: single GPU 0
    assert sh(tmp_path, "--plan", "--gpus", "0 1").stdout == sh(tmp_path, "--plan", "--gpus", "0 1").stdout   # deterministic
    assert "--gpu 3" in sh(tmp_path, "--plan", "--gpu", "3").stdout


def test_refusals(tmp_path):
    r = sh(tmp_path, "--plan", "--datasets", "s3_toys")
    assert r.returncode == 2 and "ML-1M only" in r.stderr
    assert sh(tmp_path, "--plan", "--datasets", "ml_1m s3_toys").returncode == 2
    v = (tmp_path / "v").as_posix()
    r = sh(tmp_path, "--plan", GPU_RUNS=v + "/runs_bench")
    assert r.returncode == 2 and "runs_bench" in r.stderr
    r = sh(tmp_path, "--plan", CODE=v + "/code_bench4")
    assert r.returncode == 2 and "code_bench4" in r.stderr
    r = sh(tmp_path, "--plan", "--max-slots", "9")
    assert r.returncode == 2 and "at most 8" in r.stderr
    assert sh(tmp_path, "--plan", "--gpus", "0 x").returncode == 2
    assert sh(tmp_path, "--plan", "--max-slots", "8").returncode == 0
    assert "slots: 8" in sh(tmp_path, "--plan", "--gpus", "0 1").stdout and "slots: 4" in sh(tmp_path, "--plan").stdout


FAKE = r"""#!/usr/bin/env bash
if [ "$1" = "-c" ]; then exec "$REAL_PY" "$@"; fi
shift 3
sub=$1; shift
W=1; arm=""; seed=""; ds=""; b=6; ml=50; dr=""; dev=-; gpu=-; rr=-
while [ $# -gt 0 ]; do case $1 in --workers) W=$2; shift;; --arm) arm=$2; shift;; --seed) seed=$2; shift;; --dataset) ds=$2; shift;;
  --budget-efe) b=$2; shift;; --max-len) ml=$2; shift;; --dropout) dr=$2; shift;; --device) dev=$2; shift;; --gpu) gpu=$2; shift;;
  --runs-root) rr=$2; shift;; esac; shift; done
R=$rr/$ds
echo "start $W $sub ${arm:--} ${seed:--} $dev $gpu b$b L$ml D${dr:--}" >> "$FAKELOG"
[ -n "${FAKE_YIELD:-}" ] && [ "$sub" = run ] && touch "$rr/_gpu_state/YIELD"
sleep 0.2
mkdir -p "$R/dp" "$R/trackb"
case $sub in
  manifest) echo '{}' > "$R/MANIFEST.json";;
  decide-warm) echo '{"FA_WARM_STAR": "FA_WARM_FROZEN"}' > "$R/FA_WARM_DECISION.json";;
  seed3) echo '{}' > "$R/SEED3_DECISION.json";;
  trackb-select) a=$arm; cell='[24, 200, 0.2]'; case $a in FA|FA_WARM*) cell='[6, 200, 0.2]';; esac
     echo "{\"winner_cell\": $cell, \"winner_run\": \"x\"}" > "$R/trackb/WINNER_$a.json";;
  run) label=$arm; if [ "$ml" != 50 ] || [ -n "$dr" ] || [ "$b" != 6 ]; then label="${arm}__B${b}_L${ml}_D${dr:-0.1}"; fi
     mkdir -p "$R/${label}_s$seed"; echo '{"status": "DONE_MAIN"}' > "$R/${label}_s$seed/RESULT.json"
     case $arm in S_CAL*) echo '{}' > "$R/dp/S_RECORD_FA_WARM_FROZEN_s$seed.json";; esac;;
esac
[ -n "${FAKE_YIELD_ON:-}" ] && [ "$sub $arm $seed" = "$FAKE_YIELD_ON" ] && touch "$YIELD_FILE"
echo "end $W $sub ${arm:--} ${seed:--} ok $gpu" >> "$FAKELOG"
"""


def _setup():
    sp = Path(tempfile.mkdtemp(prefix="bm04"))
    v = sp / "v"
    (v / "code_bench5b").mkdir(parents=True)
    log = sp / "fake.log"
    fake = sp / "fakepy"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    env = dict(os.environ, V=v.as_posix(), PY=fake.as_posix(), REAL_PY=Path(sys.executable).as_posix(), FAKELOG=log.as_posix(), POLL="0.1")

    def go(*extra, **e):
        return subprocess.run([BASH, SCRIPT.as_posix(), *extra], capture_output=True, text=True, env=dict(env, **e), timeout=900)
    return sp, v, log, go


def _peaks(log):
    running, peak, per, pk = 0, 0, {}, {}
    for x in (y.split() for y in log.read_text().splitlines()):
        w, g = int(x[1]), x[6]
        d = w if x[0] == "start" else -w
        running += d
        per[g] = per.get(g, 0) + d   # g == "-" for the non-run jobs (no GPU): excluded below
        peak = max(peak, running)
        if g != "-": pk[g] = max(pk.get(g, 0), per[g])
    return peak, pk


def test_fake_launch_flags_slots_root_state_and_value_free_logs(tmp_path):
    sp, v, log, go = _setup()
    r = go("--launch", "--gpus", "0 1", "--max-slots", "8", "--slots-per-gpu", "2")
    assert r.returncode == 0 and "failed or skipped: 0" in r.stdout, r.stdout + r.stderr
    peak, pk = _peaks(log)
    assert 1 < peak <= 4 and set(pk) == {"0", "1"} and max(pk.values()) <= 2          # explicit slots per GPU
    starts = [x for x in (y.split() for y in log.read_text().splitlines()) if x[0] == "start"]
    for x in starts:
        if x[2] == "run":
            assert x[5] == "cuda" and x[6] in ("0", "1"), x
        else:
            assert x[5] == "-" and x[6] == "-", x
    assert not (v / "runs_bench").exists() and (v / "runs_bench_gpu" / "ml_1m" / "MANIFEST.json").exists()
    assert sorted(p.name for p in v.iterdir()) == ["code_bench5b", "runs_bench_gpu"]       # state only under runs_bench_gpu
    assert (v / "runs_bench_gpu" / "_gpu_state").is_dir() and (v / "runs_bench_gpu" / "logs").is_dir()
    assert not (v / "runs_bench_gpu" / "_gpu_state" / "DRIVER.lock").exists()             # lock released
    # value-free: the driver's own output and logs carry names, return codes and times only
    texts = r.stdout + r.stderr + "".join(p.read_text() for p in (v / "runs_bench_gpu" / "logs").glob("gpu_*.log"))
    assert not re.search(r"mrr|ndcg|recall|hit@|\b0\.\d{4,}", texts, re.I)
    # plan mode on the REAL (existing) state dir: works, reports the state, writes nothing
    before = sorted((p.as_posix(), p.stat().st_mtime_ns) for p in v.rglob("*"))
    rp = go("--plan")
    assert rp.returncode == 0 and "# state (read-only):" in rp.stdout and "0 fail markers" in rp.stdout
    assert before == sorted((p.as_posix(), p.stat().st_mtime_ns) for p in v.rglob("*"))
    log.write_text("")
    r2 = go("--launch", "--gpus", "0 1")
    assert r2.returncode == 0 and log.read_text() == ""                                   # write-once: nothing runs twice
    assert go("--launch", "--slots-per-gpu", "5").returncode == 2


def test_yield_file_at_start_and_between_jobs_then_resume(tmp_path):
    sp, v, log, go = _setup()
    st = v / "runs_bench_gpu" / "_gpu_state"
    st.mkdir(parents=True)
    (st / "YIELD").write_text("")
    r0 = go("--launch")
    assert r0.returncode == 0 and "YIELDED" in r0.stdout and not log.exists() or log.read_text() == ""
    (st / "YIELD").unlink()
    # a YIELD that appears while jobs run: running jobs finish, nothing new starts, exit 0, no .fail markers
    r1 = go("--launch", "--gpus", "0 1", FAKE_YIELD="1")
    assert r1.returncode == 0 and "YIELDED" in r1.stdout, r1.stdout + r1.stderr
    lines = [x.split() for x in log.read_text().splitlines()]
    ns = sum(1 for x in lines if x[0] == "start")
    assert 1 <= ns < 73 and ns == sum(1 for x in lines if x[0] == "end")                  # every started job finished
    assert not list(st.glob("*.fail")) and not (st / "DRIVER.lock").exists()
    (st / "YIELD").unlink()
    r2 = go("--resume", "--gpus", "0 1")
    assert r2.returncode == 0 and "failed or skipped: 0" in r2.stdout, r2.stdout + r2.stderr
    keys = [tuple(x[2:5] + x[7:]) for x in (y.split() for y in log.read_text().splitlines()) if x[0] == "start"]
    assert len(keys) == len(set(keys))   # no run repeated


def test_yield_to_a_running_h_or_lane_process(tmp_path):
    sp, v, log, go = _setup()
    p = subprocess.Popen([BASH, "-c", "sleep 6"])
    try:
        r = go("--launch", YIELD_RE="[s]leep 6")
        assert r.returncode == 0 and "YIELDED" in r.stdout and (not log.exists() or log.read_text() == "")
    finally:
        p.wait()


def test_exactly_one_ml1m_driver(tmp_path):
    sp, v, log, go = _setup()
    st = v / "runs_bench_gpu" / "_gpu_state"
    st.mkdir(parents=True)
    holder = subprocess.Popen([BASH, "-c", "echo $$; exec sleep 31"], stdout=subprocess.PIPE, text=True)
    try:
        pid = holder.stdout.readline().strip()
        (st / "DRIVER.lock").mkdir()
        (st / "DRIVER.lock" / "pid").write_text(pid)
        r = go("--launch")
        assert r.returncode == 2 and "lock held" in r.stderr
    finally:
        holder.kill(); holder.wait()
    (st / "DRIVER.lock" / "pid").write_text("999999")                                      # stale lock: taken over
    assert go("--launch", FAKE_YIELD="").returncode == 0 and not (st / "DRIVER.lock").exists()
    other = sp / "other"
    other.mkdir()
    (other / "run_ml1m_gpu.sh").write_text("sleep 6" + chr(10))
    (other / "run_wave1.sh").write_text("sleep 6" + chr(10))
    p2 = subprocess.Popen([BASH, (other / "run_wave1.sh").as_posix()])                      # default datasets include ml_1m
    try:
        r = go("--launch")
        assert r.returncode == 2 and "run_wave1.sh covering ml_1m" in r.stderr
    finally:
        p2.wait()
    assert go("--plan").returncode == 0                                                    # plan mode never refuses on running drivers
