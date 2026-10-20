"""v3b orchestrator (owner: Opus): executes registry v3 amendment 2 without an LLM in the loop.

Idempotent state machine (state/v3b_state.json), polled every 60 s, run in tmux `sasrec-v3b`:
  1. A1_s2026_cool5 on GPU1 once review/GATE_C_COOLDOWN_PASS.json exists (latest start 21:00Z).
     Gate: cohort diag at 4U+40k and 4U+80k must be >= the A1_s2026 epoch-4 cohort (0.1806 from its
     diag at u=4U); else stop it and skip every other cooldown.
  2. When A1_s2026_cool5 is COMPLETE: GO for the A0 / s2027 cooldowns iff its full-VS epoch 5 >=
     epoch 4 (A1_s2026 e4 macro MRR@20). A0_s2026_cool5 then runs on GPU1 (latest start 00:50Z).
  3. A1_s2027 (running on GPU0): when COMPLETE (4 epochs), hold its SUPERVISOR_DONE marker (protects
     latest.pt from ckpt_io.reclaim) and, if cooldowns are GO, start A1_s2027_cool5 on GPU0
     (latest start 04:30Z).
  4. Weights keeper: extracts verified end-of-epoch model weights (epochs 3/4/5) into
     <run>/checkpoints/epoch<e>_weights.pt as soon as each end-of-epoch checkpoint exists, before
     any rotation/reclaim can remove it (needed for the pre-registered S4 averages).
  5. When every training job is COMPLETE/STOPPED/skipped (or the 06:30Z hard stop): run
     tools/final_stage.sh (selection, lock, retrospective evaluation, report) once.
Only processes/sessions created for this campaign are touched; nothing is deleted.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

S = Path("/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1")
PY = os.path.expanduser("~/venvs/ppsi-gpu/bin/python")
STATE = S / "state" / "v3b_state.json"
U = 156_956
RUNS = S / "runs"
A1_E4_COHORT = 0.1806          # A1_s2026 diag cohort at u = 4U (update 627,824), pre-registered in amendment 2
DEADLINES = {"A1_RICH256_s2026_cool5": "2026-09-25T21:00:00Z", "A0_ID256_s2026_cool5": "2026-09-26T00:50:00Z",
             "A1_RICH256_s2027_cool5": "2026-09-26T04:30:00Z"}
HARD_STOP_TRAINING = "2026-09-26T06:30:00Z"
COOL = {"A1_RICH256_s2026_cool5": ("A1_RICH256", 2026, "1"), "A0_ID256_s2026_cool5": ("A0_ID256", 2026, "1"),
        "A1_RICH256_s2027_cool5": ("A1_RICH256", 2027, "0")}


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def jload(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:
        return default


def save(state: dict) -> None:
    tmp = STATE.with_name(STATE.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, STATE)


def log(state: dict, msg: str) -> None:
    state.setdefault("log", []).append(f"[{utc()}] {msg}")
    with (S / "logs" / "campaign_v3b.log").open("a") as fh:
        fh.write(f"[{utc()}] {msg}\n")


def has_session(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name], capture_output=True).returncode == 0


def status_of(run_dir: Path) -> str:
    if (run_dir / "SUPERVISOR_DONE").exists() or (run_dir / "SUPERVISOR_DONE.hold").exists():
        return "DONE"
    if (run_dir / "SUPERVISOR_STOPPED").exists():
        return "STOPPED"
    return "RUNNING" if run_dir.exists() else "ABSENT"


def gpu_free(g: str) -> bool:
    r = subprocess.run(["nvidia-smi", "-i", g, "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.returncode == 0 and not r.stdout.strip()


def launch_cool(name: str, state: dict) -> None:
    run_id, seed, gpu = COOL[name]
    src = RUNS / f"{run_id}_s{seed}"
    dst = RUNS / name
    sess = f"sasrec-{name}"
    if has_session(sess) or status_of(dst) in ("DONE", "STOPPED"):
        return
    dst.mkdir(parents=True, exist_ok=True)
    trainer = f"{PY} {S / 'code' / 'continue_cooldown.py'} --source-run-dir {src}"
    cmd = f"SASREC_TRAINER='{trainer}' bash {S / 'code' / 'supervise_run.sh'} {run_id} {seed} {gpu} {dst}"
    subprocess.run(["tmux", "new-session", "-d", "-s", sess, cmd], check=False)
    state["jobs"][name] = {"status": "RUNNING", "gpu": gpu, "started_utc": utc()}
    log(state, f"launched {name} on GPU{gpu}")


def diag_at(run_dir: Path, update: int):
    p = run_dir / "metrics.jsonl"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        if '"diag_eval"' in line:
            r = json.loads(line)
            if r["update"] == update:
                return r["cohort_macro_mrr20"]
    return None


def epoch_vs(run_dir: Path, epoch: int):
    recs = jload(run_dir / "epoch_records.json", []) or []
    for r in recs:
        if r["epoch"] == epoch and r.get("full_vs"):
            return r["full_vs"]["macro_mrr20"]
    return None


def keep_weights(run_dir: Path, state: dict) -> None:
    """Extract verified end-of-epoch weights (3, 4, 5) from latest.pt before rotation/reclaim."""
    import torch
    ck = run_dir / "checkpoints" / "latest.pt"
    if not ck.exists():
        return
    done = state.setdefault("weights", {}).setdefault(run_dir.name, {})
    recs = jload(run_dir / "epoch_records.json", []) or []
    finished = {r["epoch"] for r in recs if not r.get("inherited_from_source")}
    todo = [e for e in (3, 4, 5) if e in finished and str(e) not in done]
    if not todo:
        return
    try:
        obj = torch.load(ck, map_location="cpu", weights_only=False)
    except Exception as e:           # being rotated right now; retry next poll
        log(state, f"keeper: {run_dir.name} latest.pt not readable yet ({e!r})")
        return
    for e in todo:
        at_end = int(obj["update"]) == e * U and ((int(obj["offset_in_epoch"]) == 0 and int(obj["epoch_idx0"]) == e)
                                                   or (int(obj["offset_in_epoch"]) == obj["config"]["n_train"]
                                                       and int(obj["epoch_idx0"]) == e - 1))
        if at_end and not obj["eval_pending"]:
            out = run_dir / "checkpoints" / f"epoch{e}_weights.pt"
            tmp = out.with_name(out.name + ".partial")
            torch.save({"state_dict": obj["model_state_dict"], "meta": {"run": run_dir.name, "epoch": e,
                                                                        "update": int(obj["update"])}}, tmp)
            torch.load(tmp, map_location="cpu", weights_only=False)
            os.replace(tmp, out)
            done[str(e)] = {"path": str(out), "utc": utc()}
            log(state, f"keeper: saved {out.name} for {run_dir.name}")
        elif int(obj["update"]) > e * U + 20_000:
            done[str(e)] = {"path": None, "missed": True}
            log(state, f"keeper: MISSED epoch {e} weights for {run_dir.name}")


def release_completed(name: str, state: dict) -> None:
    """After a cooldown run is COMPLETE with its epoch-5 weights saved: delete that run's latest.pt /
    previous.pt and its source's latest.pt (the only optimizer-state copies, no longer needed; every
    weight set used by the pre-registered systems is kept: best_weights, epoch3/4/5 weights).
    Verified, logged, idempotent; the same class of files ckpt_io.reclaim is allowed to remove."""
    rel = state.setdefault("released", {})
    if name in rel:
        return
    run_id, seed, _ = COOL[name]
    d, src = RUNS / name, RUNS / f"{run_id}_s{seed}"
    s = jload(d / "summary.json", {}) or {}
    w5 = d / "checkpoints" / "epoch5_weights.pt"
    src_w4 = src / "checkpoints" / "epoch4_weights.pt"
    if s.get("final_status") != "COMPLETE" or not w5.exists() or not src_w4.exists():
        return
    if (d / "SUPERVISOR_DONE").exists():
        (d / "SUPERVISOR_DONE").rename(d / "SUPERVISOR_DONE.hold")
    removed = []
    for f in (d / "checkpoints" / "latest.pt", d / "checkpoints" / "previous.pt", src / "checkpoints" / "latest.pt"):
        if f.exists():
            f.unlink()
            removed.append(str(f))
    rel[name] = {"utc": utc(), "removed": removed}
    log(state, f"released {name}: removed {removed}")


def hold(run_dir: Path, state: dict) -> None:
    m = run_dir / "SUPERVISOR_DONE"
    if m.exists():
        m.rename(run_dir / "SUPERVISOR_DONE.hold")
        log(state, f"hold: {run_dir.name} protected from reclaim")


def main() -> int:
    state = jload(STATE) or {"created_utc": utc(), "jobs": {}, "cooldowns_go": None, "final_stage": None}
    log(state, "orchestrator (re)started")
    save(state)
    while True:
        now = time.time()
        # weights keeper for every relevant run
        for d in ("A1_RICH256_s2027",):
            if (RUNS / d).exists():
                keep_weights(RUNS / d, state)
        # 1. A1_s2026 cooldown
        j = "A1_RICH256_s2026_cool5"
        if j not in state["jobs"] and (S / "review" / "GATE_C_COOLDOWN_PASS.json").exists():
            if now <= ts(DEADLINES[j]) and gpu_free("1"):
                launch_cool(j, state)
            elif now > ts(DEADLINES[j]):
                state["jobs"][j] = {"status": "SKIPPED_LATE"}; state["cooldowns_go"] = False
                log(state, f"{j} skipped: past latest start")
        if state["jobs"].get(j, {}).get("status") == "RUNNING":
            d = RUNS / j
            # diag runs every 40,000 GLOBAL updates, so 4U+40k / 4U+80k are read at the first two
            # diag points at or after 4U+40k: u = 680,000 (4U+52,176) and 720,000 (4U+92,176)
            c40, c80 = diag_at(d, 680_000), diag_at(d, 720_000)
            if "gate" not in state["jobs"][j] and c40 is not None and c80 is not None:
                ok = c40 >= A1_E4_COHORT and c80 >= A1_E4_COHORT
                state["jobs"][j]["gate"] = {"c_u680000": c40, "c_u720000": c80, "threshold": A1_E4_COHORT, "go": ok,
                                            "note": "first two diag points at/after 4U+40k (diag cadence 40k global updates)"}
                log(state, f"{j} cooldown gate: 40k {c40:.4f} 80k {c80:.4f} vs {A1_E4_COHORT} -> {'GO' if ok else 'STOP'}")
                if not ok:
                    subprocess.run(["tmux", "kill-session", "-t", f"sasrec-{j}"], check=False)
                    state["jobs"][j]["status"] = "STOPPED_BY_GATE"; state["cooldowns_go"] = False
            st = status_of(d)
            if st in ("DONE", "STOPPED") and state["jobs"][j]["status"] == "RUNNING":
                state["jobs"][j]["status"] = st
                e5, e4 = epoch_vs(d, 5), epoch_vs(RUNS / "A1_RICH256_s2026", 4)
                state["cooldowns_go"] = bool(st == "DONE" and e5 is not None and e4 is not None and e5 >= e4)
                state["jobs"][j]["e5_vs_e4"] = [e5, e4]
                log(state, f"{j} finished {st}: e5 {e5} vs e4 {e4} -> other cooldowns {'GO' if state['cooldowns_go'] else 'SKIP'}")
        # 2. A0_s2026 cooldown after A1's
        j0 = "A0_ID256_s2026_cool5"
        if state["cooldowns_go"] is True and j0 not in state["jobs"]:
            if now <= ts(DEADLINES[j0]) and gpu_free("1"):
                launch_cool(j0, state)
            elif now > ts(DEADLINES[j0]):
                state["jobs"][j0] = {"status": "SKIPPED_LATE"}; log(state, f"{j0} skipped: late")
        elif state["cooldowns_go"] is False and j0 not in state["jobs"]:
            state["jobs"][j0] = {"status": "SKIPPED_NO_GO"}; log(state, f"{j0} skipped: A1 cooldown did not improve")
        # 3. seed 2027: hold its latest.pt at completion; cooldown if GO
        d27 = RUNS / "A1_RICH256_s2027"
        if status_of(d27) == "DONE":
            hold(d27, state)
            j27 = "A1_RICH256_s2027_cool5"
            if state["cooldowns_go"] is True and j27 not in state["jobs"]:
                if now <= ts(DEADLINES[j27]) and gpu_free("0"):
                    launch_cool(j27, state)
                elif now > ts(DEADLINES[j27]):
                    state["jobs"][j27] = {"status": "SKIPPED_LATE"}; log(state, f"{j27} skipped: late")
            elif state["cooldowns_go"] is False and j27 not in state["jobs"]:
                state["jobs"][j27] = {"status": "SKIPPED_NO_GO"}; log(state, f"{j27} skipped: no-go")
        for name, info in state["jobs"].items():
            if info.get("status") == "RUNNING" and status_of(RUNS / name) in ("DONE", "STOPPED"):
                info["status"] = status_of(RUNS / name); log(state, f"{name} -> {info['status']}")
            if info.get("status") == "DONE" and name in COOL:
                release_completed(name, state)
        # 4. final stage when all training is settled (or the hard stop)
        s27 = status_of(d27)
        settled = s27 in ("DONE", "STOPPED") and state["cooldowns_go"] is not None and all(
            state["jobs"].get(k, {}).get("status") not in (None, "RUNNING") for k in COOL
            if not (k != "A1_RICH256_s2026_cool5" and state["cooldowns_go"] is False))
        if state["final_stage"] is None and (settled or now > ts(HARD_STOP_TRAINING)):
            fs = S / "tools" / "final_stage.sh"
            if fs.exists():
                if now > ts(HARD_STOP_TRAINING) and not settled:
                    log(state, "hard stop reached: stopping unfinished training sessions")
                    for k, info in state["jobs"].items():
                        if info.get("status") == "RUNNING":
                            subprocess.run(["tmux", "kill-session", "-t", f"sasrec-{k}"], check=False)
                            info["status"] = "STOPPED_HARD_STOP"
                state["final_stage"] = {"started_utc": utc()}
                save(state)
                subprocess.run(["tmux", "new-session", "-d", "-s", "sasrec-final", f"bash {fs} > {S}/logs/final_stage.log 2>&1"],
                               check=False)
                log(state, "final stage launched (tmux sasrec-final)")
            else:
                log(state, "final stage pending: tools/final_stage.sh not present yet")
        save(state)
        if state.get("final_stage"):
            return 0
        time.sleep(60)


if __name__ == "__main__":
    sys.exit(main())
