"""vs_systems.py (owner: Opus) — VS-DESCRIPTIVE evaluation of the pre-registered systems S2 and S4a on GPU1.

Authority: review/AUTO_DECISION_GPU1.md (Fable claude-fable-5-1, owner-delegated: GO_WITH_CHANGES).

Scope:
- P0 (CPU): A0-e5 vs A1-e5 complementarity (plus A1 e3/e4/e5 pairwise) from the SAVED campaign
  rank files, whose sha256 are verified against summary.json.
- P1 (GPU): S2 = LogProbEnsemble([A0_s2026 e5, A1_s2026 e5]), equal weights, the mean of per-model
  log-softmax (amendment 2 L46–53). Member ranks are captured in the same pass (forward hooks + the
  evaluator's own `rank_targets`) and compared with the saved rank files. This is the mechanics
  validation that replaces the rehearsal.
- P2 (GPU): S4a = uniform state_dict average of A1_s2026 epochs {3,4,5}.
- P3 (CPU): paired user-level bootstrap, 1,000 resamples, seed 2026, for S2 − A1e5 and S4a − A1e5,
  run regardless of sign.

Hard rules:
- VALIDATION_SELECT ONLY. No VC/TEST/TRAIN store.
- Never imports lock_retro/lock_retro_v3b (their claim path).
- Writes only under this workspace; the campaign's runs/code/review are read-only.
- Imports only the byte-identical snapshot in code/snap (see SNAPSHOT_MANIFEST.json).

Guards: a background thread aborts the process (exit 97) if any of these holds:
- GPU0 training throughput drops more than 10% versus a 3-minute baseline (re-baselined when GPU0
  changes phase);
- free disk < 10 GiB;
- cgroup memory > 56 GiB, or own RSS > 12 GiB;
- VRAM > 12 GB;
- the time is past 03:00Z.

No new stage starts after 02:00Z. The results are VS-descriptive (selection split, post-hoc look):
never a headline, never compared with GRU, never used to add, drop or re-weight any system.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent
SNAP = CODE / "snap"
sys.path.insert(0, str(SNAP))
import paths  # noqa: E402  (snapshot copy; puts snap/reuse + vendor on sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ensembles  # noqa: E402
import eval_full  # noqa: E402
import precision  # noqa: E402
from sasrec_model import SASRecConfig, build_sasrec  # noqa: E402

W = Path(os.environ.get("VSS_ROOT", str(CODE.parent)))
RUNS = Path(os.environ.get("VSS_CAMPAIGN_RUNS", "/home/jovyan/ahmedabdelhamed01/Privacy-Preserving-Shopping-Intelligence/.local_ai/sasrec_v1/runs"))
SPLIT = "VALIDATION_SELECT"
LIMITS = {"gpu0_drop": 0.10, "disk_gib": 10.0, "cgroup_gib": 56.0, "rss_gib": 12.0, "vram_gb": 12.0}
NO_NEW_STAGE_AFTER, HARD_STOP = "2026-09-26T02:00:00Z", "2026-09-26T03:00:00Z"
BOOT_N, BOOT_SEED = 1000, 2026


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ts(s: str) -> float:
    return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")))


def sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False))
    tmp.replace(p)


def status(**kw) -> None:
    p = W / "state" / "status.json"
    s = json.loads(p.read_text()) if p.exists() else {"created_utc": utc(), "stages": {}}
    s.update(kw); s["updated_utc"] = utc()
    write_json(p, s)


# ---------------------------------------------------------------- inputs (read-only campaign artifacts)
def summary(run: str) -> dict:
    return json.loads((RUNS / run / "summary.json").read_text())


def inputs() -> dict:
    a0c, a1c, a1b = summary("A0_ID256_s2026_cool5"), summary("A1_RICH256_s2026_cool5"), summary("A1_RICH256_s2026")
    for s in (a0c, a1c, a1b):
        assert s["final_status"] == "COMPLETE"
    assert a0c["best_epoch"] == 5 and a1c["best_epoch"] == 5 and a1b["best_epoch"] == 3   # pre-registered member rule
    # members are each run's best_weights (epoch recorded in summary) -> sha verified against summary.json
    best = {"A0_e5": (a0c, 5), "A1_e5": (a1c, 5), "A1_e3": (a1b, 3)}
    w = {}
    for key, (s, ep) in best.items():
        bw = s["checkpoints"]["best_weights"]
        if int(bw["epoch"]) != ep:
            raise RuntimeError(f"{key}: best_weights epoch {bw['epoch']} != {ep}")
        if sha(bw["path"]) != bw["sha256"]:
            raise RuntimeError(f"{key}: best_weights sha drift")
        w[key] = Path(bw["path"])
    w["A1_e4"] = RUNS / "A1_RICH256_s2026/checkpoints/epoch4_weights.pt"      # sha recorded (not in summary)
    # e3/e4 ranks come from the BASE run (the weights' own run); the cool5 copies must be the same files
    for ep in ("3", "4"):
        if a1c["ranks"][ep]["sha256"] != a1b["ranks"][ep]["sha256"]:
            raise RuntimeError(f"A1 e{ep} ranks differ between base and cool5 summaries")
    r = {"A0_e5": a0c["ranks"]["5"], "A1_e5": a1c["ranks"]["5"], "A1_e3": a1b["ranks"]["3"], "A1_e4": a1b["ranks"]["4"]}
    for k, rec in r.items():
        if sha(rec["path"]) != rec["sha256"]:
            raise RuntimeError(f"saved ranks sha drift for {k}")
    return {"weights": {k: {"path": str(p), "sha256": sha(p)} for k, p in w.items()},
            "ranks": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in r.items()},
            "registry": {"A0": a0c["config"]["registry_entry"], "A1": a1c["config"]["registry_entry"]}}


def load_saved_ranks(rec: dict, ds) -> np.ndarray:
    z = np.load(rec["path"])
    if bytes(z["decision_id_sha256"]).hex() != ds.sha256():
        raise RuntimeError(f"decision order mismatch for {rec['path']}")
    return z["ranks"].astype(np.int64)


# ---------------------------------------------------------------- guard thread
class Guard(threading.Thread):
    def __init__(self, device_index: int = 0):
        super().__init__(daemon=True)
        self.dev = device_index
        self.baseline, self.samples, self.phase, self.started = None, [], None, time.time()
        self.log = W / "logs" / "guard.jsonl"; self.log.parent.mkdir(parents=True, exist_ok=True)

    def gpu0_run(self):
        """The active GPU0 training run: the newest RUNNING s2027 base or cool5 status."""
        for name in ("A1_RICH256_s2027_cool5", "A1_RICH256_s2027"):
            p = RUNS / name / "status.json"
            if p.exists():
                s = json.loads(p.read_text())
                if s.get("state") == "RUNNING" and time.time() - p.stat().st_mtime < 900:
                    return name, float((s.get("eta") or {}).get("decisions_per_s") or 0.0)
        return None, None

    def abort(self, why: str) -> None:
        status(state="ABORTED", reason=why, aborted_utc=utc())
        with open(self.log, "a") as f:
            f.write(json.dumps({"utc": utc(), "abort": why}) + "\n")
        os._exit(97)

    def baseline_3min(self) -> None:
        vals = []
        for _ in range(6):
            self.check_resources()                         # hard stop / resources checked during baseline too
            run, dps = self.gpu0_run()
            if run and dps:
                vals.append(dps); self.phase = run
            time.sleep(30)
        self.baseline = float(np.mean(vals)) if vals else None
        self.samples = []
        with open(self.log, "a") as f:
            f.write(json.dumps({"utc": utc(), "baseline_dps": self.baseline, "phase": self.phase, "n": len(vals)}) + "\n")

    def check_resources(self) -> None:
        free = os.statvfs(str(W)); free_gib = free.f_bavail * free.f_frsize / 2**30
        if free_gib < LIMITS["disk_gib"]:
            self.abort(f"disk free {free_gib:.2f} GiB < {LIMITS['disk_gib']}")
        try:
            cg = int(Path("/sys/fs/cgroup/memory.current").read_text()) / 2**30
            if cg > LIMITS["cgroup_gib"]:
                self.abort(f"cgroup memory {cg:.1f} GiB > {LIMITS['cgroup_gib']}")
        except FileNotFoundError:
            pass
        rss = 0.0
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 2**20
        if rss > LIMITS["rss_gib"]:
            self.abort(f"own RSS {rss:.1f} GiB > {LIMITS['rss_gib']}")
        if torch.cuda.is_available():
            vram = torch.cuda.memory_reserved(self.dev) / 1e9
            if vram > LIMITS["vram_gb"]:
                self.abort(f"VRAM {vram:.1f} GB > {LIMITS['vram_gb']}")
        if time.time() > ts(HARD_STOP):
            self.abort("hard stop 03:00Z")

    def run(self) -> None:
        while True:
            self.check_resources()
            run, dps = self.gpu0_run()
            if run and self.baseline is None:              # no valid baseline yet (stale status at start): retry
                self.baseline_3min()
                continue
            if run and run != self.phase and self.phase is not None:
                with open(self.log, "a") as f:
                    f.write(json.dumps({"utc": utc(), "phase_change": [self.phase, run]}) + "\n")
                self.baseline_3min()                       # GPU0 changed phase: re-baseline
                continue
            if run and dps and self.baseline:
                self.samples.append(dps); self.samples = self.samples[-10:]
                if time.time() - self.started > 300 and len(self.samples) >= 10:
                    m = float(np.mean(self.samples))
                    if m < (1 - LIMITS["gpu0_drop"]) * self.baseline:
                        self.abort(f"GPU0 throughput {m:.0f} < 90% of baseline {self.baseline:.0f}")
            with open(self.log, "a") as f:
                f.write(json.dumps({"utc": utc(), "gpu0": run, "dps": dps, "baseline": self.baseline}) + "\n")
            time.sleep(30)


# ---------------------------------------------------------------- metrics helpers
def metrics(ranks: np.ndarray, ds) -> dict:
    out = {"n": int(len(ranks))}
    for k in (1, 5, 10, 20):
        a = eval_full.aggregate(ranks, ds, k=k)["end_to_end"]
        for agg in ("macro", "micro"):
            for m in ("mrr", "recall"):
                out[f"{agg}_{m}@{k}"] = a[agg][f"{m}@{k}"]
    out["users"] = int(eval_full.aggregate(ranks, ds, k=20)["end_to_end"]["macro"]["user_count"])
    return out


def hits20(ranks: np.ndarray) -> np.ndarray:
    c = eval_full._credits_from_ranks(torch.as_tensor(ranks, dtype=torch.long), 20)
    return (c["recall@20"].numpy() > 0)


def complementarity(ra: np.ndarray, rb: np.ndarray, ds, names=("A0_e5", "A1_e5")) -> dict:
    pop = np.asarray(ds.eval_mask, dtype=bool)
    ha, hb = hits20(ra)[pop], hits20(rb)[pop]
    n = int(pop.sum())
    both, a_only, b_only = int((ha & hb).sum()), int((ha & ~hb).sum()), int((~ha & hb).sum())
    union = np.maximum(ha, hb)
    uk = np.asarray(ds.eval_user_key)[pop]
    uniq, inv = np.unique(uk, return_inverse=True)
    per_user_union = np.bincount(inv, weights=union.astype(float)) / np.bincount(inv)
    ka, va = eval_full.per_user_metric(ra, ds, "mrr", 20)
    kb, vb = eval_full.per_user_metric(rb, ds, "mrr", 20)
    assert np.array_equal(ka, kb)
    d = vb - va
    return {"population": "end_to_end (eval_mask rows; OOV/unscored count as miss)", "n": n,
            f"both_hit@20": both, f"{names[0]}_only": a_only, f"{names[1]}_only": b_only, "neither": n - both - a_only - b_only,
            f"micro_R@20_{names[0]}": float(ha.mean()), f"micro_R@20_{names[1]}": float(hb.mean()),
            "oracle_union_micro_R@20": float(union.mean()), "oracle_union_macro_R@20": float(per_user_union.mean()),
            f"per_user_macro_mrr20_{names[1]}_minus_{names[0]}": {"mean": float(d.mean()), "q05": float(np.quantile(d, 0.05)),
                                                                   "q50": float(np.quantile(d, 0.5)), "q95": float(np.quantile(d, 0.95)),
                                                                   f"share_users_{names[0]}_better": float((d < 0).mean()),
                                                                   "share_users_equal": float((d == 0).mean())}}


def state_dict_sha(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode()); h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- model building
def build(entry: dict, sd: dict, device) -> torch.nn.Module:
    m = build_sasrec(SASRecConfig.from_registry(entry), seed=2026, device="cpu")
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


class TargetStashStore:
    """Transparent proxy over the real store: records each batch's target/eval_mask/start so that
    member forward hooks can rank with the evaluator's own `rank_targets` (use with prefetch=False)."""

    def __init__(self, inner):
        self.inner, self.stash = inner, {}

    def batch(self, ds, positions):
        b = self.inner.batch(ds, positions)
        self.stash = {"start": int(positions[0]), "target": b["target_class"], "eval_mask": b["eval_mask"]}
        return b

    def __getattr__(self, name):
        return getattr(self.inner, name)


def member_rank_hooks(ens, proxy: TargetStashStore, N: int, device):
    arrays = [torch.full((N,), -1, dtype=torch.long) for _ in ens.members]

    def make(i):
        def hook(_mod, _inp, logits):
            tgt = proxy.stash["target"].to(device); em = proxy.stash["eval_mask"].to(device)
            need = em & (tgt >= 0)
            if bool(need.any()):
                idx = need.nonzero(as_tuple=True)[0]
                r = eval_full.rank_targets(logits.index_select(0, idx), tgt.index_select(0, idx))
                arrays[i][idx.cpu() + proxy.stash["start"]] = r.cpu()
        return hook
    handles = [m.register_forward_hook(make(i)) for i, m in enumerate(ens.members)]
    return arrays, handles


# ---------------------------------------------------------------- stages
def run_S2(store, ds, inp, device, out: Path) -> dict:
    sd0 = ensembles.load_state_dict_file(inp["weights"]["A0_e5"]["path"])
    sd1 = ensembles.load_state_dict_file(inp["weights"]["A1_e5"]["path"])
    ens = ensembles.LogProbEnsemble([build(inp["registry"]["A0"], sd0, device), build(inp["registry"]["A1"], sd1, device)]).to(device).eval()
    proxy = TargetStashStore(store)
    arrays, handles = member_rank_hooks(ens, proxy, len(ds), device)
    t0 = time.time()
    with precision.strict_eval():
        r = eval_full.evaluate_model(ens, proxy, ds, device, eval_batch=1024, k=20, return_ranks=True, prefetch=False)
    for h in handles:
        h.remove()
    ranks = np.asarray(r["ranks"]).astype(np.int64)
    checks = {}
    for i, key in enumerate(("A0_e5", "A1_e5")):
        mine = arrays[i].numpy(); saved = load_saved_ranks(inp["ranks"][key], ds)
        checks[key] = {"identical_fraction": float((mine == saved).mean()), "n_diff": int((mine != saved).sum()),
                       "max_abs_diff": int(np.abs(mine - saved).max())}
    np.savez_compressed(out / "ranks.npz", ranks=ranks.astype(np.int32),
                        decision_id_sha256=np.frombuffer(bytes.fromhex(ds.sha256()), dtype=np.uint8))
    return {"system": "S2", "members": ["A0_ID256_s2026 e5 (cool5 best)", "A1_RICH256_s2026 e5 (cool5 best)"],
            "rule": "equal weights; mean of per-model log_softmax over K=158486; A0 item kwargs only",
            "metrics": metrics(ranks, ds), "member_rank_consistency_vs_saved": checks,
            "seconds": round(time.time() - t0, 1), "rows_per_s": r["_rows_per_s"], "peak_gpu_mem_bytes": r["_peak_gpu_mem_bytes"],
            "ranks_sha256": sha(out / "ranks.npz")}


def run_S4a(store, ds, inp, device, out: Path) -> dict:
    files = [inp["weights"][k]["path"] for k in ("A1_e3", "A1_e4", "A1_e5")]
    avg = ensembles.average_state_dicts(files)
    model = build(inp["registry"]["A1"], avg, device)
    t0 = time.time()
    with precision.strict_eval():
        r = eval_full.evaluate_model(model, store, ds, device, eval_batch=1024, k=20, return_ranks=True)
    ranks = np.asarray(r["ranks"]).astype(np.int64)
    np.savez_compressed(out / "ranks.npz", ranks=ranks.astype(np.int32),
                        decision_id_sha256=np.frombuffer(bytes.fromhex(ds.sha256()), dtype=np.uint8))
    return {"system": "S4a", "rule": "uniform state_dict average of A1_s2026 epochs {3,4,5}, strict load",
            "input_weights_sha256": {k: inp["weights"][k]["sha256"] for k in ("A1_e3", "A1_e4", "A1_e5")},
            "averaged_state_dict_sha256": state_dict_sha(avg), "metrics": metrics(ranks, ds),
            "seconds": round(time.time() - t0, 1), "rows_per_s": r["_rows_per_s"], "peak_gpu_mem_bytes": r["_peak_gpu_mem_bytes"],
            "ranks_sha256": sha(out / "ranks.npz")}


def boot(ranks_a, ranks_b, ds) -> dict:
    ka, va = eval_full.per_user_metric(ranks_a, ds, "mrr", 20)
    kb, vb = eval_full.per_user_metric(ranks_b, ds, "mrr", 20)
    return eval_full.paired_bootstrap(ka, va, kb, vb, n=BOOT_N, seed=BOOT_SEED)


def main() -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise SystemExit("must run with CUDA_VISIBLE_DEVICES=1 (GPU1)")
    if time.time() > ts(NO_NEW_STAGE_AFTER):
        raise SystemExit("past 02:00Z: no new stage")
    res = W / "results"; res.mkdir(parents=True, exist_ok=True)
    status(state="STARTING", pid=os.getpid(), manifest_sha256=sha(SNAP / "SNAPSHOT_MANIFEST.json"),
           label="VS-descriptive (selection split), post-hoc look; not a headline; never vs GRU")
    guard = Guard(0)
    guard.baseline_3min()
    guard.start()
    precision.configure_strict_fp32(2026)
    device = torch.device("cuda:0")
    inp = inputs(); write_json(res / "inputs.json", inp)
    from full_store import CompactStore
    status(state="BUILDING_VS_STORE")
    store = CompactStore.build(paths.DATASET_ROOT, splits=(SPLIT,), channels="F4")
    ds = store.decisions(SPLIT, "eval")
    if len(ds) != paths.N_VS:
        raise RuntimeError(f"VS decisions {len(ds)} != {paths.N_VS}")
    saved = {k: load_saved_ranks(v, ds) for k, v in inp["ranks"].items()}
    if not (res / "complementarity.json").exists():                       # P0 (CPU)
        comp = {"A0_e5_vs_A1_e5": complementarity(saved["A0_e5"], saved["A1_e5"], ds),
                "A1_e3_vs_A1_e5": complementarity(saved["A1_e3"], saved["A1_e5"], ds, ("A1_e3", "A1_e5")),
                "A1_e4_vs_A1_e5": complementarity(saved["A1_e4"], saved["A1_e5"], ds, ("A1_e4", "A1_e5")),
                "A1_e3_vs_A1_e4": complementarity(saved["A1_e3"], saved["A1_e4"], ds, ("A1_e3", "A1_e4"))}
        write_json(res / "complementarity.json", comp)
    for name, fn in (("S2", run_S2), ("S4a", run_S4a)):                  # P1, P2 (GPU1)
        out = res / name; out.mkdir(exist_ok=True)
        if (out / "result.json").exists():
            continue
        if time.time() > ts(NO_NEW_STAGE_AFTER):
            status(state="STOPPED_BY_TIME", skipped=name)
            break
        status(state=f"RUNNING_{name}", stage_start_utc=utc())
        write_json(out / "result.json", fn(store, ds, inp, device, out))
    bt = {}
    for name in ("S2", "S4a"):                                           # P3 (CPU), unconditional
        if not (res / name / "result.json").exists():
            continue
        rk = np.load(res / name / "ranks.npz")["ranks"].astype(np.int64)
        bt[f"{name}_minus_A1_e5"] = boot(rk, saved["A1_e5"], ds)
    write_json(res / "bootstrap.json", dict(bt, note="paired user-level, descriptive only on the selection split; computed regardless of sign"))
    status(state="DONE" if all((res / n / "result.json").exists() for n in ("S2", "S4a")) else "PARTIAL", done_utc=utc())
    return 0


if __name__ == "__main__":
    sys.exit(main())
