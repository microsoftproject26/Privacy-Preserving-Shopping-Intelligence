"""queue_runner.py -- detached state machine executing config/run_registry.locked.json's
queue_plan with NO LLM in the loop (SASREC_INTERFACES.md sec 7, review/FABLE_GATE_P2_RESPONSE.md,
owner 38 h deadline). Owner: Sonnet-runtime.

v2 protocol (Gate P2): only A0_ID256 (GPU0) and A1_RICH256 (GPU1), seed 2026, launched
simultaneously. Fixed epochs, no early stopping, no pair barrier (train_sasrec.py no
longer takes --pair-dir/--pair-partner and exit code 7 no longer exists). No replicas
(`replication.additional_seeds == []`).

Cut rule: once BOTH run dirs' `epoch_records.json` (a JSON list written by the trainer,
distinct from summary.json) show epoch 2, this module calls Opus's pure
`sasrec_schedule.epoch_cap(now_s, deadline_s, arms, ...)` EXACTLY ONCE and writes
`state/epoch_cap.json` atomically; that file is never rewritten once it exists. Both
anchors were launched with `--epoch-cap-file state/epoch_cap.json` so the trainer can
read the cap itself once decided.

Once both anchors are terminal: `compare_pair_summaries` records whether they finished
with equal epoch counts and equal per-epoch permutation digests (pair marked INCOMPLETE
if either failed to COMPLETE -- selection still runs over whichever COMPLETE among
them). Then `sasrec_schedule.select_configuration`, `state/selection.json`, and finally
lock_retro.py lock/eval and render_report.py (commands overridable by env for tests).
The replica phase is a no-op whenever `replication.additional_seeds` is empty.

State is atomic JSON at `$S/state/queue_state.json` (or `$SASREC_QUEUE_ROOT/state/...`
for tests). A restart re-derives truth from live tmux sessions + on-disk markers before
trusting the persisted state, so it never launches a duplicate run.

Env overrides (tests only; production uses none of these):
  SASREC_QUEUE_ROOT        redirect the whole workspace root (registry/runs/state/locks/...)
  SASREC_TRAINER            forwarded to supervise_run.sh (fake trainer command)
  SASREC_QUEUE_POLL_SECONDS poll interval, default 60
  SASREC_EPOCH_CAP_IMPORT    "module:function" for the cut-rule pure function,
                             default "sasrec_schedule:epoch_cap" (tests may point this at
                             a stub module on PYTHONPATH if Opus's real one isn't landed yet)
  SASREC_LOCK_CMD            overrides "<py> lock_retro.py lock"
  SASREC_RETRO_EVAL_CMD       overrides "<py> lock_retro.py eval"
  SASREC_REPORT_CMD           overrides "<py> render_report.py"
  SASREC_QUEUE_NO_SAMPLER=1  skip launching the background resources.py sampler
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import paths  # noqa: E402  import-first convention
import resources  # noqa: E402

GIB = 2 ** 30
CODE_DIR = Path(__file__).resolve().parent
PYTHON = os.environ.get("LOBOT_PYTHON", str(Path.home() / "venvs" / "ppsi-gpu" / "bin" / "python"))


def _root() -> Path:
    override = os.environ.get("SASREC_QUEUE_ROOT")
    return Path(override) if override else paths.S


def _poll_seconds() -> float:
    return float(os.environ.get("SASREC_QUEUE_POLL_SECONDS", "60"))


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def _pid_alive(pid: int, needle: str = "") -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    if not needle:
        return True
    cmdline = Path(f"/proc/{pid}/cmdline")
    try:
        txt = cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        return needle in txt
    except OSError:
        return True  # can't verify; assume alive rather than risk a duplicate


def _tmux_has_session(session: str) -> bool:
    r = subprocess.run(["tmux", "has-session", "-t", session],
                       capture_output=True, text=True)
    return r.returncode == 0


# ---------------------------------------------------------------------------
# pure decision helper (unit-tested directly; no I/O)
# ---------------------------------------------------------------------------

def plan_replica_concurrency(selected_group: str, measured_rss_bytes: Optional[int],
                              mem_limit_bytes: Optional[int], mem_current_bytes: Optional[int],
                              headroom_floor_bytes: int = 12 * GIB) -> str:
    """Returns "concurrent", "sequential", or "unknown" (treated as sequential: the
    conservative default). Only item-only configs are launched concurrently
    unconditionally; a rich-channel winner requires >= 12 GiB projected headroom
    after BOTH replicas' measured RSS."""
    if selected_group != "rich_frozen_channels":
        return "concurrent"
    if measured_rss_bytes is None or mem_limit_bytes is None or mem_current_bytes is None:
        return "unknown"
    available = mem_limit_bytes - mem_current_bytes
    projected_after_second = available - measured_rss_bytes
    return "concurrent" if projected_after_second >= headroom_floor_bytes else "sequential"


def arm_from_epoch_records(records: list) -> dict:
    """Build one `arms[i]` entry (SASREC_INTERFACES v2 cut rule) from a run's
    `epoch_records.json` list. `eval_seconds` is None until floor-epoch evals start
    (epochs 1-2 carry no full-VS eval under schedule S')."""
    if not records:
        return {"epochs_done": 0, "mean_train_epoch_seconds": None, "eval_seconds": None}
    epochs_done = max(int(r.get("epoch", 0)) for r in records)
    train_secs = [float(r["epoch_seconds"]) for r in records if r.get("epoch_seconds") is not None]
    mean_train = sum(train_secs) / len(train_secs) if train_secs else None
    eval_secs = [float(r["eval_seconds"]) for r in records if r.get("eval_seconds") is not None]
    eval_seconds = sum(eval_secs) / len(eval_secs) if eval_secs else None
    return {"epochs_done": epochs_done, "mean_train_epoch_seconds": mean_train, "eval_seconds": eval_seconds}


def compare_pair_summaries(summary_a: Optional[dict], summary_b: Optional[dict]) -> dict:
    """Pure equal-epoch / equal-permutation-digest check (Gate P2 revised pair rule:
    no barrier, but the pair must never be COMPARED at unequal epochs). Either arm not
    COMPLETE -> INCOMPLETE (selection still runs over whichever IS complete; this dict
    is purely a reporting record, never a gate on select_configuration itself)."""
    if (not summary_a or not summary_b
            or summary_a.get("final_status") != "COMPLETE"
            or summary_b.get("final_status") != "COMPLETE"):
        return {"pair_status": "INCOMPLETE", "equal_epochs": None,
                "equal_perm_sha256": None, "mismatches": []}
    ea, eb = summary_a.get("epochs_completed"), summary_b.get("epochs_completed")
    equal_epochs = ea is not None and ea == eb
    recs_a = {r.get("epoch"): r.get("perm_sha256") for r in summary_a.get("epoch_records", [])}
    recs_b = {r.get("epoch"): r.get("perm_sha256") for r in summary_b.get("epoch_records", [])}
    common = sorted(set(recs_a) & set(recs_b))
    mismatches = [e for e in common if recs_a[e] != recs_b[e]]
    equal_perm = len(common) > 0 and len(mismatches) == 0
    return {"pair_status": "COMPLETE" if (equal_epochs and equal_perm) else "MISMATCH",
            "equal_epochs": equal_epochs, "equal_perm_sha256": equal_perm, "mismatches": mismatches}


class QueueRunner:
    def __init__(self, root: Optional[Path] = None):
        self.root = root or _root()
        self.registry_path = self.root / "config" / "run_registry.locked.json"
        self.state_path = self.root / "state" / "queue_state.json"
        self.lock_path = self.root / "state" / "queue_runner.pid"
        self.runs_root = self.root / "runs"
        self.registry = _read_json(self.registry_path)
        if self.registry is None:
            raise FileNotFoundError(f"registry not found: {self.registry_path}")
        self.models = {m["id"]: m for m in self.registry["models"]}
        self.primary_seed = int(self.registry["training"]["seed"])
        self.replica_seeds = list(self.registry["replication"]["additional_seeds"])
        qp = self.registry["queue_plan"]
        if "phase" in qp:  # v2 (Gate P2): a single phase, one run per GPU, no chaining
            self.gpu_plan = {"0": list(qp["phase"].get("gpu0", [])),
                             "1": list(qp["phase"].get("gpu1", []))}
        else:  # v1 fallback: phase2 (pair) then phase3 (challengers), chained per GPU
            self.gpu_plan = {
                "0": list(qp.get("phase2", {}).get("gpu0", [])) + list(qp.get("phase3", {}).get("gpu0", [])),
                "1": list(qp.get("phase2", {}).get("gpu1", [])) + list(qp.get("phase3", {}).get("gpu1", [])),
            }
        self.pair = self.registry.get("pair", {})
        self.pair_members = set(self.pair.get("members", []))
        self.epoch_cap_path = self.root / "state" / "epoch_cap.json"
        self.state = self._load_or_init_state()

    # -- state -------------------------------------------------------------
    def _init_state(self) -> dict:
        return {
            "phase": "PRIMARY",
            "gpu_queues": {"0": list(self.gpu_plan["0"]), "1": list(self.gpu_plan["1"])},
            "gpu_current": {"0": None, "1": None},
            "runs": {},
            "selection": None,
            "pair_parity": None,
            "epoch_cap": None,
            "replicas": {},
            "replica_concurrency": None,
            "lock_rc": None, "retro_rc": None, "report_rc": None,
            "blocked_reason": None,
            "sampler_pid": None,
            "updated_utc": None,
        }

    def _load_or_init_state(self) -> dict:
        st = _read_json(self.state_path)
        return st if st is not None else self._init_state()

    def save_state(self) -> None:
        self.state["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        _write_json_atomic(self.state_path, self.state)

    # -- run bookkeeping -----------------------------------------------------
    def run_key(self, run_id: str, seed: int) -> str:
        return f"{run_id}_s{seed}"

    def run_dir(self, run_id: str, seed: int) -> Path:
        return self.runs_root / self.run_key(run_id, seed)

    def session_name(self, run_id: str, seed: int) -> str:
        return f"sasrec-{run_id}-s{seed}"

    def _terminal_status(self, run_dir: Path) -> Optional[str]:
        if (run_dir / "SUPERVISOR_DONE").exists():
            return "DONE"
        if (run_dir / "SUPERVISOR_STOPPED").exists():
            return "STOPPED"
        return None

    def launch(self, run_id: str, seed: int, gpu: str, extra: list[str]) -> None:
        run_dir = self.run_dir(run_id, seed)
        key = self.run_key(run_id, seed)
        session = self.session_name(run_id, seed)
        if _tmux_has_session(session):
            self.state["runs"].setdefault(key, {})["status"] = "RUNNING"
            self.state["runs"][key]["session"] = session
            return
        terminal = self._terminal_status(run_dir)
        if terminal is not None:
            self.state["runs"].setdefault(key, {})["status"] = terminal
            return
        run_dir.mkdir(parents=True, exist_ok=True)
        supervise = CODE_DIR / "supervise_run.sh"
        trainer_env = os.environ.get("SASREC_TRAINER", "")
        parts = []
        if trainer_env:
            parts.append(f"SASREC_TRAINER={shlex.quote(trainer_env)}")
        parts += ["bash", shlex.quote(str(supervise)), shlex.quote(run_id), shlex.quote(str(seed)),
                  shlex.quote(str(gpu)), shlex.quote(str(run_dir))]
        parts += [shlex.quote(a) for a in extra]
        cmd_str = " ".join(parts)
        subprocess.run(["tmux", "new-session", "-d", "-s", session, cmd_str], check=False)
        self.state["runs"][key] = {"run_id": run_id, "seed": seed, "gpu": gpu,
                                   "session": session, "status": "RUNNING"}

    def poll_run(self, run_id: str, seed: int) -> Optional[str]:
        """Returns "DONE"/"STOPPED" once terminal, else None. Also relaunches a run
        whose tmux session died without leaving a terminal marker (rare: the
        supervisor itself was killed) -- supervise_run.sh's own --resume detection
        picks the checkpoint back up, so this never duplicates in-flight work."""
        run_dir = self.run_dir(run_id, seed)
        key = self.run_key(run_id, seed)
        terminal = self._terminal_status(run_dir)
        if terminal is not None:
            self.state["runs"].setdefault(key, {})["status"] = terminal
            summary = _read_json(run_dir / "summary.json")
            if summary is not None:
                self.state["runs"][key]["summary"] = summary
            return terminal
        session = self.state["runs"].get(key, {}).get("session") or self.session_name(run_id, seed)
        if not _tmux_has_session(session):
            # session vanished mid-run with no terminal marker: relaunch (resumes).
            gpu = self.state["runs"].get(key, {}).get("gpu", "0")
            extra = self._extra_args_for(run_id)
            self.launch(run_id, seed, gpu, extra)
        return None

    def _extra_args_for(self, run_id: str) -> list[str]:
        # v2 (Gate P2): no pair barrier; every anchor gets the shared cut-rule file
        # path so the trainer can read the decided epoch_cap once it exists.
        return ["--epoch-cap-file", str(self.epoch_cap_path)]

    # -- primary phase -------------------------------------------------------
    def tick_primary(self) -> None:
        for gpu in ("0", "1"):
            cur = self.state["gpu_current"][gpu]
            if cur is None:
                if self.state["gpu_queues"][gpu]:
                    run_id = self.state["gpu_queues"][gpu].pop(0)
                    extra = self._extra_args_for(run_id)
                    self.launch(run_id, self.primary_seed, gpu, extra)
                    self.state["gpu_current"][gpu] = run_id
                continue
            result = self.poll_run(cur, self.primary_seed)
            if result is not None:
                self.state["gpu_current"][gpu] = None

    def primary_complete(self) -> bool:
        if any(self.state["gpu_queues"][g] for g in ("0", "1")):
            return False
        if any(self.state["gpu_current"][g] is not None for g in ("0", "1")):
            return False
        return True

    def primary_summaries(self) -> list[dict]:
        out = []
        for run_id in self.models:
            key = self.run_key(run_id, self.primary_seed)
            rec = self.state["runs"].get(key, {})
            summary = rec.get("summary") or _read_json(self.run_dir(run_id, self.primary_seed) / "summary.json")
            if summary is None:
                summary = {"run_id": run_id, "final_status": rec.get("status", "MISSING")}
            out.append(summary)
        return out

    # -- cut rule (Gate P2) -----------------------------------------------------
    def _read_epoch_records(self, run_id: str, seed: int) -> list:
        data = _read_json(self.run_dir(run_id, seed) / "epoch_records.json", default=[])
        return data if isinstance(data, list) else []

    def _deadline_s(self) -> float:
        iso = self.registry["owner_deadline"]["deadline_utc"]
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()

    def _resolve_epoch_cap_fn(self):
        spec = os.environ.get("SASREC_EPOCH_CAP_IMPORT", "sasrec_schedule:epoch_cap")
        mod_name, func_name = spec.split(":", 1)
        mod = importlib.import_module(mod_name)
        return getattr(mod, func_name)

    def maybe_compute_epoch_cap(self) -> None:
        """Cut rule (SASREC_INTERFACES v2 / registry.cut_rule): once BOTH anchors'
        epoch_records.json show epoch 2, call Opus's pure epoch_cap() exactly once and
        persist it. The on-disk file (not in-memory state) is the single source of
        truth for "already decided", so this is safe across restarts."""
        if self.epoch_cap_path.exists():
            return
        records = {mid: self._read_epoch_records(mid, self.primary_seed) for mid in self.models}
        arms_by_id = {mid: arm_from_epoch_records(r) for mid, r in records.items()}
        if not all(a["epochs_done"] >= 2 for a in arms_by_id.values()):
            return
        order = [m["id"] for m in self.registry["models"]]
        arms = [arms_by_id[mid] for mid in order]
        sched = self.registry.get("schedule", {})
        now_s = time.time()
        deadline_s = self._deadline_s()
        fn = self._resolve_epoch_cap_fn()
        epoch_cap = int(fn(now_s, deadline_s, arms,
                           planned=int(sched.get("planned_epochs", 6)),
                           min_epochs=int(sched.get("min_epochs_under_cut", 4)),
                           post_hours=2.0, safety_hours=1.0,
                           floor_start=int(sched.get("floor_start_epoch", 3)),
                           eval_seconds_default=1600))
        result = {"epoch_cap": epoch_cap,
                  "decided_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                  "inputs": {"arms": arms, "now_s": now_s, "deadline_s": deadline_s,
                            "registry_order": order}}
        _write_json_atomic(self.epoch_cap_path, result)
        self.state["epoch_cap"] = result

    # -- pair parity (Gate P2: no barrier, but never compare at unequal epochs) ---
    def compute_pair_parity(self) -> dict:
        ids = [m["id"] for m in self.registry["models"]]
        if len(ids) != 2:
            return {"pair_status": "N/A", "equal_epochs": None, "equal_perm_sha256": None, "mismatches": []}
        summaries = []
        for mid in ids:
            key = self.run_key(mid, self.primary_seed)
            s = self.state["runs"].get(key, {}).get("summary")
            if s is None:
                s = _read_json(self.run_dir(mid, self.primary_seed) / "summary.json")
            summaries.append(s)
        result = compare_pair_summaries(summaries[0], summaries[1])
        result["ids"] = ids
        return result

    # -- select ---------------------------------------------------------------
    def do_select(self) -> None:
        import sasrec_schedule
        summaries = self.primary_summaries()
        order = [m["id"] for m in self.registry["models"]]
        result = sasrec_schedule.select_configuration(summaries, order)
        self.state["selection"] = result
        sel_path = self.root / "state" / "selection.json"
        _write_json_atomic(sel_path, result)

    # -- replicas ---------------------------------------------------------------
    def _measure_rss_for_run(self, run_id: str, seed: int) -> Optional[int]:
        pid_file = self.run_dir(run_id, seed) / "logs" / "trainer.pid"
        txt = pid_file.read_text().strip() if pid_file.exists() else ""
        if not txt.isdigit():
            return None
        return resources.rss_of_pids([int(txt)])["total_bytes"] or None

    def tick_replicate(self) -> None:
        sel = self.state["selection"]
        winner = sel.get("selected") if sel else None
        reps = self.state["replicas"]
        if winner is None:
            self.state["blocked_reason"] = "no eligible configuration selected"
            self.state["phase"] = "LOCK"
            return
        group = self.models[winner]["group"]
        seed0, seed1 = self.replica_seeds[0], self.replica_seeds[1]
        key0, key1 = self.run_key(winner, seed0), self.run_key(winner, seed1)

        if not reps:
            self.state["replicas"] = {"seed0": {"run_id": winner, "seed": seed0, "gpu": "0", "status": "PENDING"},
                                      "seed1": {"run_id": winner, "seed": seed1, "gpu": "1", "status": "PENDING"}}
            reps = self.state["replicas"]

        r0, r1 = reps["seed0"], reps["seed1"]
        concurrency = self.state.get("replica_concurrency")

        if r0["status"] == "PENDING":
            self.launch(winner, seed0, "0", [])
            r0["status"] = "RUNNING"
            if group != "rich_frozen_channels":
                concurrency = "concurrent"

        if concurrency in (None, "unknown") and r1["status"] == "PENDING":
            rss = self._measure_rss_for_run(winner, seed0)
            mem = resources.cgroup_memory()
            concurrency = plan_replica_concurrency(group, rss, mem.get("limit_bytes"),
                                                    mem.get("current_bytes"))
        self.state["replica_concurrency"] = concurrency

        if r1["status"] == "PENDING":
            if concurrency == "concurrent":
                self.launch(winner, seed1, "1", [])
                r1["status"] = "RUNNING"
            else:
                res0 = self.poll_run(winner, seed0)
                if res0 is not None:
                    r0["status"] = res0
                    self.launch(winner, seed1, "1", [])
                    r1["status"] = "RUNNING"
                return

        if r0["status"] == "RUNNING":
            res0 = self.poll_run(winner, seed0)
            if res0 is not None:
                r0["status"] = res0
        if r1["status"] == "RUNNING":
            res1 = self.poll_run(winner, seed1)
            if res1 is not None:
                r1["status"] = res1

        if r0["status"] in ("DONE", "STOPPED") and r1["status"] in ("DONE", "STOPPED"):
            self.state["phase"] = "LOCK"

    # -- post-training pipeline ------------------------------------------------
    def _run_cmd(self, cmd: str, timeout: float = 3600) -> int:
        try:
            r = subprocess.run(shlex.split(cmd), cwd=str(CODE_DIR), timeout=timeout)
            return r.returncode
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"queue_runner: command failed to start: {cmd!r}: {exc}", file=sys.stderr)
            return 255

    def tick_lock(self) -> None:
        cmd = os.environ.get("SASREC_LOCK_CMD", f"{PYTHON} {CODE_DIR / 'lock_retro.py'} lock")
        self.state["lock_rc"] = self._run_cmd(cmd, timeout=6 * 3600)   # re-hashes the frozen dataset (Gate C NB1)
        self.state["phase"] = "RETRO_EVAL"

    def tick_retro(self) -> None:
        if self.state["lock_rc"] == 0:
            cmd = os.environ.get("SASREC_RETRO_EVAL_CMD", f"{PYTHON} {CODE_DIR / 'lock_retro.py'} eval")
            self.state["retro_rc"] = self._run_cmd(cmd, timeout=8 * 3600)
            if self.state["retro_rc"] != 0:   # the same locked evaluation resumes its missing shards once
                self.state["retro_first_rc"] = self.state["retro_rc"]
                self.state["retro_rc"] = self._run_cmd(cmd, timeout=8 * 3600)
        else:
            self.state["retro_rc"] = None
            self.state["blocked_reason"] = (self.state.get("blocked_reason") or "") + "; lock failed"
        self.state["phase"] = "REPORT"

    def tick_report(self) -> None:
        cmd = os.environ.get("SASREC_REPORT_CMD", f"{PYTHON} {CODE_DIR / 'render_report.py'}")
        self.state["report_rc"] = self._run_cmd(cmd)
        self.state["phase"] = "DONE"

    # -- sampler / lock file ----------------------------------------------------
    def ensure_sampler(self) -> None:
        if os.environ.get("SASREC_QUEUE_NO_SAMPLER") == "1":
            return
        pid = self.state.get("sampler_pid")
        if pid and _pid_alive(int(pid), needle="resources.py"):
            return
        out_jsonl = self.root / "logs" / "resources_samples.jsonl"
        out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        log_path = self.root / "logs" / "resources_sampler.log"
        with open(log_path, "ab") as logf:
            proc = subprocess.Popen(
                ["setsid", PYTHON, str(CODE_DIR / "resources.py"), "--loop", str(out_jsonl),
                 "--interval", "60"],
                stdin=subprocess.DEVNULL, stdout=logf, stderr=logf, start_new_session=True,
            )
        self.state["sampler_pid"] = proc.pid

    def acquire_lock(self) -> bool:
        rec = _read_json(self.lock_path)
        if rec and _pid_alive(int(rec.get("pid", -1)), needle="queue_runner.py"):
            return False
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(self.lock_path, {"pid": os.getpid(),
                                            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        return True

    def release_lock(self) -> None:
        rec = _read_json(self.lock_path)
        if rec and int(rec.get("pid", -1)) == os.getpid():
            try:
                self.lock_path.unlink()
            except OSError:
                pass

    # -- main loop ------------------------------------------------------------
    def tick(self) -> None:
        phase = self.state["phase"]
        if phase == "PRIMARY":
            self.tick_primary()
            self.maybe_compute_epoch_cap()
            if self.primary_complete():
                if self.state["selection"] is None:
                    self.do_select()
                if self.state["pair_parity"] is None:
                    self.state["pair_parity"] = self.compute_pair_parity()
                self.state["phase"] = "REPLICATE"
        elif phase == "REPLICATE":
            if not self.replica_seeds:  # v2: no-op, nothing to replicate
                self.state["phase"] = "LOCK"
            else:
                self.tick_replicate()
        elif phase == "LOCK":
            self.tick_lock()
        elif phase == "RETRO_EVAL":
            self.tick_retro()
        elif phase == "REPORT":
            self.tick_report()
        self.save_state()

    def run_forever(self) -> None:
        if not self.acquire_lock():
            print("queue_runner: another instance is already running (pid lock held); exiting", file=sys.stderr)
            return
        try:
            self.ensure_sampler()
            self.save_state()
            while self.state["phase"] != "DONE":
                self.tick()
                if self.state["phase"] != "DONE":
                    time.sleep(_poll_seconds())
        finally:
            self.release_lock()

    def status(self) -> dict:
        return self.state

    def dry_run_plan(self) -> dict:
        return {"gpu_plan": self.gpu_plan, "pair": self.pair,
                "replica_seeds": self.replica_seeds, "primary_seed": self.primary_seed,
                "models": list(self.models.keys())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run-plan", action="store_true")
    args = ap.parse_args(argv)

    qr = QueueRunner()
    if args.status:
        print(json.dumps(qr.status(), indent=2, default=str))
        return 0
    if args.dry_run_plan:
        print(json.dumps(qr.dry_run_plan(), indent=2, default=str))
        return 0
    qr.run_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
