"""Registry v3 amendment 2: selection, lock and ONE retrospective VC/TEST evaluation (owner: Opus).

VC/TEST were inspected in the GRU project: every result here is labelled
"PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK". The lock is written before any VC/TEST row
loads, and fixes every model and pre-registered system that will ever be evaluated there.

  python lock_retro_v3b.py select                      # headline single model (full VS only)
  python lock_retro_v3b.py lock                        # locks/SASREC_LOCK.json (O_EXCL)
  python lock_retro_v3b.py eval --worker K --workers 2 # disjoint (split, entry) shards, resumable
  python lock_retro_v3b.py finalize                    # archived fingerprint gates, bootstraps, RESULTS
Reuses the audited helpers of lock_retro.py (headline, metrics_from_ranks, fingerprint,
verify_archived, integrity) unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import time
from pathlib import Path

import paths  # noqa: F401

import numpy as np
import torch

import lock_retro as base
from lock_retro import sha, atomic_json, exclusive_json, log, headline, metrics_from_ranks, fingerprint, verify_archived

S = paths.S
RUNS = S / "runs"
SEL = S / "state" / "selection_v3b.json"
AM2 = S / "config" / "run_registry.v3_amendment_2.json"
SINGLES = (("A0_ID256", 2026), ("A1_RICH256", 2026), ("A1_RICH256", 2027))
SPLITS = ("VALIDATION_CONFIRM", "TEST")
ROWS = {"VALIDATION_CONFIRM": 2_461_568, "TEST": 5_862_968}


def run_summary(run_id: str, seed: int):
    """The run's final summary: the cooldown continuation if it COMPLETED, else the base run;
    (None, None) if neither is COMPLETE (the run is then reported as unavailable, never silently)."""
    cool = RUNS / f"{run_id}_s{seed}_cool5"
    for d in (cool, RUNS / f"{run_id}_s{seed}"):
        s = base._summary(d) if (d / "summary.json").exists() else None
        if s and s.get("final_status") == "COMPLETE":
            return d, s
    return None, None


def base_summary(run_id: str, seed: int):
    d = RUNS / f"{run_id}_s{seed}"
    s = base._summary(d) if (d / "summary.json").exists() else None
    return s if s and s.get("final_status") == "COMPLETE" else None


def epoch_weight_file(run_id: str, seed: int, epoch: int):
    src, cool = RUNS / f"{run_id}_s{seed}", RUNS / f"{run_id}_s{seed}_cool5"
    cands = {3: [src / "checkpoints" / "epoch3_weights.pt"], 4: [src / "checkpoints" / "epoch4_weights.pt"],
             5: [cool / "checkpoints" / "epoch5_weights.pt"]}[epoch]
    if epoch == 3:   # the base run's best_weights holds epoch 3 when epoch 3 was its best
        s = base._summary(src)
        if s.get("best_epoch") == 3:
            cands.append(Path(s["checkpoints"]["best_weights"]["path"]))
    for p in cands:
        if p.exists():
            return p
    return None


MIN_S4_EPOCHS = 2


def cmd_select(_a) -> int:
    from sasrec_schedule import select_configuration
    rows, unavailable = [], []
    for rid, seed in SINGLES:
        d, s = run_summary(rid, seed)
        if s is None:
            unavailable.append(f"{rid}_s{seed}")
            continue
        rows.append({"run_id": f"{rid}_s{seed}", "seed": seed, "final_status": s["final_status"],
                     "best_metric": s["best_metric"], "best_epoch": s["best_epoch"],
                     "micro_recall20_at_best": s["micro_recall20_at_best"], "n_params": s["n_params"],
                     "summary_dir": str(d)})
    sel = select_configuration(rows, [f"{r}_s{s}" for r, s in SINGLES])
    sel["candidates"] = rows
    sel["unavailable_runs"] = unavailable
    sel["rule_source"] = "amendment 2: max full-VS macro MRR@20 over {A0_s2026, A1_s2026, A1_s2027} x epochs {3,4,5}"
    atomic_json(SEL, sel)
    log(f"selected {sel['selected']} ({sel.get('selected_best_metric')})")
    return 0


def _single_entry(rid: str, seed: int) -> dict:
    d, s = run_summary(rid, seed)
    bw = (s.get("checkpoints") or {}).get("best_weights") or {}
    p = Path(bw["path"])
    if bw.get("sha256") and bw["sha256"] != sha(p):
        raise SystemExit(f"best weights hash drift for {rid}_s{seed}: summary {bw['sha256']} vs file {sha(p)}")
    return {"kind": "single", "run_id": rid, "seed": seed, "group": s["group"], "summary_dir": str(d),
            "summary_sha256": sha(d / "summary.json"), "best_epoch": s["best_epoch"],
            "best_metric_full_vs": s["best_metric"], "weights": [str(p)], "weights_sha256": [sha(p)],
            "registry_entry": s["config"]["registry_entry"]}


def cmd_lock(_a) -> int:
    lock_path = base.LOCK_PATH
    if lock_path.exists():
        log("lock exists (immutable)")
        return 0
    import train_sasrec as ts
    reg, reg_sha = ts.load_registry()
    sel = json.loads(SEL.read_text(encoding="utf-8"))
    entries, dropped = {}, {}
    for r, sd in SINGLES:
        if run_summary(r, sd)[1] is not None:
            entries[f"{r}_s{sd}"] = _single_entry(r, sd)
        else:
            dropped[f"{r}_s{sd}"] = "no COMPLETE summary (base or cooldown)"
    systems = {"S1": ["A1_RICH256_s2026", "A1_RICH256_s2027"], "S2": ["A0_ID256_s2026", "A1_RICH256_s2026"],
               "S3": ["A0_ID256_s2026", "A1_RICH256_s2026", "A1_RICH256_s2027"]}
    for name, members in systems.items():
        miss = [m for m in members if m not in entries]
        if miss:
            dropped[name] = f"member(s) unavailable: {miss}"
            continue
        entries[name] = {"kind": "ensemble_logprob_mean", "members": members,
                         "rule": "equal weights fixed a priori; mean of per-member log-softmax; members at their best epoch"}
    for name, (rid, seed) in {"S4a": ("A1_RICH256", 2026), "S4b": ("A1_RICH256", 2027)}.items():
        if f"{rid}_s{seed}" not in entries:
            dropped[name] = f"run {rid}_s{seed} unavailable"
            continue
        files = [(e, epoch_weight_file(rid, seed, e)) for e in (3, 4, 5)]
        avail = [(e, p) for e, p in files if p is not None]
        if len(avail) < MIN_S4_EPOCHS:
            dropped[name] = f"only epochs {[e for e, _ in avail]} available (< {MIN_S4_EPOCHS}); an average would equal a single checkpoint"
            continue
        entries[name] = {"kind": "weight_average", "run": f"{rid}_s{seed}", "group": "rich_frozen_channels",
                         "epochs_available": [e for e, _ in avail], "epochs_missing": [e for e, p in files if p is None],
                         "weights": [str(p) for _, p in avail], "weights_sha256": [sha(p) for _, p in avail],
                         "registry_entry": entries[f"{rid}_s{seed}"]["registry_entry"], "seed": seed,
                         "rule": "uniform state_dict average of the run's epochs {3,4,5} (whatever exists, disclosed)"}
    pair = {"comparable": False, "reason": "an anchor is unavailable"}
    if "A0_ID256_s2026" in entries and "A1_RICH256_s2026" in entries:
        e0, e1 = entries["A0_ID256_s2026"], entries["A1_RICH256_s2026"]
        both_cool = e0["summary_dir"].endswith("_cool5") and e1["summary_dir"].endswith("_cool5")
        if both_cool:
            a0, a1, basis = base._summary(Path(e0["summary_dir"])), base._summary(Path(e1["summary_dir"])), "both cooldowns (5 epochs)"
        elif e0["best_epoch"] <= 4 and e1["best_epoch"] <= 4:
            a0, a1, basis = base_summary("A0_ID256", 2026), base_summary("A1_RICH256", 2026), "base runs (4 epochs); both bests <= epoch 4"
        else:
            a0 = a1 = None
            basis = "one best comes from a cooldown epoch the other arm did not receive"
        if a0 is not None and a1 is not None:
            ok = (a0["epochs_completed"] == a1["epochs_completed"] and a0.get("perm_sha256_by_epoch") == a1.get("perm_sha256_by_epoch")
                  and a0.get("eval_epochs") == a1.get("eval_epochs"))
            pair = {"comparable": bool(ok), "basis": basis, "epochs": [a0["epochs_completed"], a1["epochs_completed"]],
                    "reason": None if ok else "epochs / permutation digests / eval epochs differ"}
        else:
            pair = {"comparable": False, "basis": basis, "reason": basis}
    integrity = base.verify_frozen_now(S / "state" / "verify_frozen_before_retro.json")
    if not integrity.get("all_ok"):
        raise SystemExit("frozen dataset integrity failed; refusing to lock")
    archived = {n: {sp: {"path": str(base.GRU_FINAL / f), "sha256": sha(base.GRU_FINAL / f)}
                    for sp, f in zip(SPLITS, files[:2])} for n, files in base.ARCHIVED.items()}
    gate_c = S / "review" / "GATE_C_PASS.json"
    pinned = json.loads(gate_c.read_text()).get("pinned_runtime_sha256", {})
    lock = {"created_utc": base.utc(), "host": socket.gethostname(), "label": base.LABEL,
            "test_exposure": "previously_seen_in_GRU_project", "vc_exposure": "previously_seen_in_GRU_project",
            "selected_single_model": sel["selected"], "selection": sel, "entries": entries,
            "headline": sel["selected"], "pair_check": pair, "dropped_entries": dropped, "archived_baselines": archived,
            "gru_lock": {"path": str(base.GRU_LOCK), "sha256": sha(base.GRU_LOCK)},
            "gru_results": {k: {"path": str(v), "sha256": sha(v)} for k, v in base.GRU_RESULTS.items()},
            "registry_sha256": reg_sha,
            "amendments": {p.name: sha(p) for p in sorted((S / "config").glob("run_registry.v3_amendment_*.json"))},
            "gate_files": {p.name: sha(p) for p in sorted((S / "review").glob("GATE_C*_PASS.json"))},
            "runtime_code_drift_since_gate_c": {f: {"pinned": h, "now": sha(paths.CODE / f)}
                                                for f, h in pinned.items() if sha(paths.CODE / f) != h},
            "code_sha256": ts.code_sha256_map(), "dataset_version_hash": paths.VERSION_HASH, "integrity": integrity}
    exclusive_json(lock_path, lock)
    log(f"wrote {lock_path} sha256={sha(lock_path)}")
    return 0


def tasks(lock: dict) -> list:
    cost = {n: (len(e["members"]) if e["kind"].startswith("ensemble") else 1) for n, e in lock["entries"].items()}
    ts_ = sorted(((sp, n) for sp in SPLITS for n in lock["entries"]), key=lambda t: -cost[t[1]] * ROWS[t[0]])
    load = [0.0, 0.0]
    assign = {}
    for sp, n in ts_:
        w = 0 if load[0] <= load[1] else 1
        assign[(sp, n)] = w
        load[w] += cost[n] * ROWS[sp]
    return [(sp, n, w) for (sp, n), w in sorted(assign.items())]


def build_model(entry: dict, lock: dict, device):
    import train as reuse_train
    from sasrec_model import SASRecConfig, SASRecCE
    import ensembles

    def one(e, sd=None):
        cfg = SASRecConfig.from_registry(e["registry_entry"])
        m = SASRecCE(cfg, vocab_sizes=dict(reuse_train.VOCAB_SIZES),
                     product_idx_of_class=reuse_train.load_product_idx_of_class(), seed=int(e["seed"]))
        if sd is None:
            for p, h in zip(e["weights"], e["weights_sha256"]):
                if sha(p) != h:
                    raise SystemExit(f"weights changed since lock: {p}")
            sd = ensembles.load_state_dict_file(e["weights"][0])
        m.load_state_dict(sd)
        return m.to(device).eval()

    if entry["kind"] == "single":
        return one(entry)
    if entry["kind"] == "weight_average":
        for p, h in zip(entry["weights"], entry["weights_sha256"]):
            if sha(p) != h:
                raise SystemExit(f"weights changed since lock: {p}")
        return one(entry, ensembles.average_state_dicts(entry["weights"]))
    return ensembles.LogProbEnsemble([one(lock["entries"][m]) for m in entry["members"]]).to(device).eval()


def claim(lock_sha: str) -> None:
    if base.CLAIM_PATH.exists():
        c = json.loads(base.CLAIM_PATH.read_text())
        if c.get("lock_sha256") != lock_sha:
            raise SystemExit("an evaluation was claimed under a different lock: refusing a second look")
        return
    try:
        exclusive_json(base.CLAIM_PATH, {"lock_sha256": lock_sha, "claimed_utc": base.utc(), "host": socket.gethostname()})
    except FileExistsError:
        claim(lock_sha)


def cmd_eval(a) -> int:
    import precision
    import sasrec_data
    from eval_full import evaluate_model
    if base.DONE_PATH.exists():
        log("retrospective evaluation already complete")
        return 0
    lock = json.loads(base.LOCK_PATH.read_text())
    lock_sha = sha(base.LOCK_PATH)
    claim(lock_sha)
    precision.configure_strict_fp32(2026)
    device = torch.device(os.environ.get("SASREC_EVAL_DEVICE", "cuda:0"))   # tests use cpu
    mine = [(sp, n) for sp, n, w in tasks(lock) if w == a.worker]
    man_p = base.OUT / f"manifest_w{a.worker}.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    for sp in SPLITS:
        todo = [n for s_, n in mine if s_ == sp]
        if not todo:
            continue
        sdir = base.OUT / sp
        sdir.mkdir(parents=True, exist_ok=True)
        stores = {}

        def store_for(entry):
            ch = "F0" if (entry["kind"] == "single" and entry["group"] == "item_only") else "F4"
            if ch not in stores:
                stores[ch] = sasrec_data.build_eval_store(ch, sp, lock_path=base.LOCK_PATH)
            return stores[ch]

        for n in todo:
            rp, jp = sdir / f"{n}.ranks.npz", sdir / f"{n}.json"
            key = f"{sp}/{n}"
            if key in man and sha(rp) == man[key]["ranks_sha256"] and jp.exists():
                continue
            store = store_for(lock["entries"][n])
            ds = store.decisions(sp, "eval")
            if lock["entries"][n]["kind"].startswith("ensemble") and "A0_ID256_s2026" in lock["entries"][n]["members"]:
                f0 = store_for({"kind": "single", "group": "item_only"})
                pos = np.random.default_rng(7).choice(len(ds), min(20_000, len(ds)), replace=False)
                b0, b4 = f0.batch(f0.decisions(sp, "eval"), pos), store.batch(ds, pos)
                for k in ("item_tokens", "lengths", "attention_mask", "position_ids"):
                    if not torch.equal(b0[k], b4[k]):
                        raise SystemExit(f"{sp}: F0/F4 stores disagree on {k}; mixed ensemble refused")
            model = build_model(lock["entries"][n], lock, device)
            t0 = time.time()
            with precision.strict_eval():
                res = evaluate_model(model, store, ds, device, eval_batch=1024, k=20, return_ranks=True)
            ranks = np.asarray(res.pop("ranks"), dtype=np.int64)
            tmp = sdir / f"{n}.ranks.tmp{os.getpid()}.npz"
            np.savez_compressed(tmp, ranks=ranks.astype(np.int32))
            os.replace(tmp, rp)
            b, extra = metrics_from_ranks(ranks, ds)
            hl = headline(b, extra)
            hl["eval_seconds"] = time.time() - t0
            atomic_json(jp, hl)
            man[key] = {"ranks_sha256": sha(rp), "json_sha256": sha(jp), "utc": base.utc(), "fingerprint": fingerprint(ds)}
            atomic_json(man_p, man)
            log(f"worker {a.worker}: {key} macro MRR@20 {hl['end_to_end']['macro']['mrr@20']:.5f} ({hl['eval_seconds']:.0f}s)")
            del model
            torch.cuda.empty_cache()
        stores.clear()
    return 0


def cmd_finalize(_a) -> int:
    from eval_full import per_user_metric, paired_bootstrap
    import sasrec_data
    lock = json.loads(base.LOCK_PATH.read_text())
    man = {}
    for p in sorted(base.OUT.glob("manifest_w*.json")):
        man.update(json.loads(p.read_text()))
    if base.DONE_PATH.exists():
        log("already finalized")
        return 0
    missing = [f"{sp}/{n}" for sp, n, _ in tasks(lock) if f"{sp}/{n}" not in man]
    if missing:
        raise SystemExit(f"missing evaluations: {missing}")
    res = {"label": base.LABEL, "lock_sha256": sha(base.LOCK_PATH), "headline": lock["headline"], "splits": {},
           "models": {}, "systems": {}, "bootstraps": {}, "archived_fingerprint_gate": {}, "seed_variation": {}}
    ranks_by = {}
    for sp in SPLITS:
        store = sasrec_data.build_eval_store("F0", sp, lock_path=base.LOCK_PATH)
        ds = store.decisions(sp, "eval")
        fp = fingerprint(ds)
        res["splits"][sp] = {"fingerprint": fp}
        for n, e in lock["entries"].items():
            key = f"{sp}/{n}"
            if man[key]["fingerprint"] != fp:
                raise SystemExit(f"{key}: population fingerprint differs from the F0 rebuild")
            hl = json.loads((base.OUT / sp / f"{n}.json").read_text())
            (res["models"] if e["kind"] == "single" else res["systems"]).setdefault(n, {"kind": e["kind"]})[sp] = hl
            ranks_by[(sp, n)] = np.load(base.OUT / sp / f"{n}.ranks.npz")["ranks"].astype(np.int64)
        gru_json = json.loads(base.GRU_RESULTS[sp].read_text())
        for name, files in base.ARCHIVED.items():
            rpath = base.GRU_FINAL / (files[0] if sp == "VALIDATION_CONFIRM" else files[1])
            if sha(rpath) != lock["archived_baselines"][name][sp]["sha256"]:
                raise SystemExit(f"archived ranks changed since lock: {rpath}")
            r = np.load(rpath).astype(np.int64)
            v = verify_archived(name, sp, r, ds, gru_json)
            res["archived_fingerprint_gate"][f"{sp}/{name}"] = {"exact_match_fields": v["exact_match_fields"]}
            res["models"].setdefault(name, {"source": "archived_ranks"})[sp] = headline(v["base"], v["extra"])
            ranks_by[(sp, name)] = r
        if sp == "TEST":
            head = lock["headline"]
            pairs = [(head, "GRU_F0_locked"), (head, "TRANS_full")]
            if lock["pair_check"]["comparable"]:
                pairs.append(("A1_RICH256_s2026", "A0_ID256_s2026"))
            else:
                res["bootstraps_skipped"] = {"A1_RICH256_s2026_minus_A0_ID256_s2026": lock["pair_check"]}
            for x, y in pairs:
                if (sp, x) not in ranks_by or (sp, y) not in ranks_by:
                    res.setdefault("bootstraps_skipped", {})[f"{x}_minus_{y}"] = "ranks unavailable"
                    continue
                kx, vx = per_user_metric(ranks_by[(sp, x)], ds, metric="mrr", k=20, denominator="end_to_end")
                ky, vy = per_user_metric(ranks_by[(sp, y)], ds, metric="mrr", k=20, denominator="end_to_end")
                bt = paired_bootstrap(kx, vx, ky, vy, n=base.BOOT_N, seed=base.BOOT_SEED)
                bt.update({"n_resamples": base.BOOT_N, "seed": base.BOOT_SEED, "metric": "user-macro MRR@20 end_to_end",
                           "note": "conditional on these fitted models and a previously inspected period"})
                res["bootstraps"].setdefault(sp, {})[f"{x}_minus_{y}"] = bt
        del store
    a1 = {s: res["models"][f"A1_RICH256_s{s}"]["TEST"]["end_to_end"]["macro"]["mrr@20"]
          for s in (2026, 2027) if f"A1_RICH256_s{s}" in res["models"]}
    vals = np.array(list(a1.values()))
    res["seed_variation"] = {"config": "A1_RICH256", "split": "TEST", "macro_mrr20_by_seed": {str(k): v for k, v in a1.items()},
                             "mean": float(vals.mean()) if vals.size else None,
                             "std_ddof1": float(vals.std(ddof=1)) if vals.size > 1 else None,
                             "note": "two seeds at most; user bootstrap CIs do not capture training-seed variance"}
    res["dropped_entries"] = lock.get("dropped_entries", {})
    out = base.OUT / "RETRO_RESULTS.json"
    atomic_json(out, res)
    exclusive_json(base.DONE_PATH, {"completed_utc": base.utc(), "results": str(out), "results_sha256": sha(out)})
    log(f"wrote {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select"); sub.add_parser("lock"); sub.add_parser("finalize")
    e = sub.add_parser("eval"); e.add_argument("--worker", type=int, required=True)
    e.add_argument("--workers", type=int, default=2, choices=[2], help="the partition is fixed at 2 GPU shards")
    a = ap.parse_args(argv)
    return {"select": cmd_select, "lock": cmd_lock, "eval": cmd_eval, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
