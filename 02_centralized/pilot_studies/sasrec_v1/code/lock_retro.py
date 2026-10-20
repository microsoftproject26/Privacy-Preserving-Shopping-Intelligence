"""SASRec lock + ONE retrospective VC/TEST evaluation (owner: Opus). 04 'Lock and retrospective evaluation'.

VALIDATION_CONFIRM and TEST were already inspected in the GRU project. This module
labels every result "PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK"; nothing here
feeds a model/config/seed choice (the lock is written before any VC/TEST row loads).

  python lock_retro.py lock                 # new SASREC_LOCK.json (O_EXCL); never touches gru_v1/locks
  python lock_retro.py eval                 # resumable: completed (split, model) results are immutable
  python lock_retro.py eval --mechanics-split VALIDATION_SELECT --out-dir DIR   # mechanics test, no claim
  python lock_retro.py status

Models evaluated: the matched anchors' best weights (A0/A1, seed 2026), the selected
configuration's seeds (registry v2: seed 2026 only, no replicas), and the archived locked GRU (F0, plus F4 descriptive),
TRANS-full and POP-full through their saved rank vectors, but only after the rebuilt
population reproduces the archived headline metrics exactly (fingerprint gate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import paths  # noqa: F401

import numpy as np
import torch

LOCKS = paths.S / "locks"
LOCK_PATH = LOCKS / "SASREC_LOCK.json"
CLAIM_PATH = LOCKS / "RETRO_CLAIM.json"
DONE_PATH = LOCKS / "RETRO_DONE.json"
OUT = paths.S / "retro_eval"
GRU_FINAL = paths.GRU_WORKSPACE / "final_eval"
GRU_LOCK = paths.GRU_WORKSPACE / "locks" / "MODEL_LOCK.json"
ARCHIVED = {  # name -> (VC ranks file, TEST ranks file, key in GRU results json)
    "GRU_F0_locked": ("vc_ranks_F0_item_only_h256.npy", "ranks_F0_item_only_h256.npy", "F0_item_only_h256"),
    "GRU_F4_descriptive": ("vc_ranks_F4_rich_side_information_h256.npy", "ranks_F4_rich_side_information_h256.npy",
                           "F4_rich_side_information_h256"),
    "TRANS_full": ("vc_ranks_TRANS_full.npy", "ranks_TRANS_full.npy", "TRANS_full"),
    "POP_full": ("vc_ranks_POP_full.npy", "ranks_POP_full.npy", "POP_full"),
}
GRU_RESULTS = {"VALIDATION_CONFIRM": GRU_FINAL / "VC_RESULTS.json", "TEST": GRU_FINAL / "TEST_RESULTS.json"}
LABEL = "PREVIOUSLY INSPECTED / RETROSPECTIVE BENCHMARK"
KS = (1, 2, 5, 10, 20)
BOOT_N, BOOT_SEED = 1000, 2026


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(p) -> Optional[str]:
    p = Path(p)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, allow_nan=False, default=str)
        fh.flush(); os.fsync(fh.fileno())
    os.replace(tmp, path)


def exclusive_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, allow_nan=False, default=str)
        fh.flush(); os.fsync(fh.fileno())


def log(msg: str) -> None:
    print(f"[{utc()}] {msg}", flush=True)


# ---------------------------------------------------------------------------- lock
def verify_frozen_now(out: Path) -> dict:
    """Re-hash all frozen dataset files (reused audited verify_frozen) before the lock."""
    import verify_frozen
    res = verify_frozen.verify(paths.DATASET_ROOT)
    atomic_json(out, res)
    return {k: v for k, v in res.items() if k not in ("missing", "mismatched")} | {
        "n_missing": len(res["missing"]), "n_mismatched": len(res["mismatched"]), "receipt": str(out)}


def _summary(run_dir: Path) -> dict:
    return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))


def _run_dir(run_id: str, seed: int) -> Path:
    return paths.S / "runs" / f"{run_id}_s{seed}"


def cmd_lock(args) -> int:
    if LOCK_PATH.exists():
        log(f"lock already exists: {LOCK_PATH} (immutable)")
        return 0
    import train_sasrec as ts
    reg, reg_sha = ts.load_registry()
    sel = json.loads((paths.S / "state" / "selection.json").read_text(encoding="utf-8"))
    chosen = sel.get("selected")
    if not chosen:
        raise SystemExit("no selected configuration in state/selection.json")
    primary_seed = int(reg["replication"]["primary_seed"])
    seeds = [primary_seed] + [int(s) for s in reg["replication"]["additional_seeds"]]
    models = {}
    wanted = [("A0_ID256", primary_seed, "anchor_item_only"), ("A1_RICH256", primary_seed, "anchor_rich")]
    wanted += [(chosen, s, "selected_seed") for s in seeds]
    for rid, seed, role in wanted:
        name = f"{rid}_s{seed}"
        if name in models:
            models[name]["roles"].append(role)
            continue
        d = _run_dir(rid, seed)
        if not (d / "summary.json").exists():
            models[name] = {"run_id": rid, "seed": seed, "roles": [role], "status": "MISSING"}
            continue
        s = _summary(d)
        bw = (s.get("checkpoints") or {}).get("best_weights") or {}
        bpath = Path(bw.get("path") or (d / "checkpoints" / "best_weights.pt"))
        actual = sha(bpath)
        if s.get("final_status") != "COMPLETE" or actual is None:
            models[name] = {"run_id": rid, "seed": seed, "roles": [role], "status": s.get("final_status", "MISSING")}
            continue
        if bw.get("sha256") and bw["sha256"] != actual:
            raise SystemExit(f"best weights hash drift for {name}")
        models[name] = {"run_id": rid, "seed": seed, "roles": [role], "status": "COMPLETE", "run_dir": str(d),
                        "group": s["group"], "best_epoch": s["best_epoch"], "best_metric_full_vs": s["best_metric"],
                        "best_weights": str(bpath), "best_weights_sha256": actual,
                        "summary_sha256": sha(d / "summary.json"), "registry_entry": s["config"]["registry_entry"]}
    pair_check = {"comparable": False, "reason": "an anchor is not COMPLETE"}
    a0, a1 = models.get(f"A0_ID256_s{primary_seed}", {}), models.get(f"A1_RICH256_s{primary_seed}", {})
    if a0.get("status") == "COMPLETE" and a1.get("status") == "COMPLETE":
        s0, s1 = _summary(Path(a0["run_dir"])), _summary(Path(a1["run_dir"]))
        same_epochs = s0["epochs_completed"] == s1["epochs_completed"]
        same_perm = s0.get("perm_sha256_by_epoch") == s1.get("perm_sha256_by_epoch")
        same_eval = s0.get("eval_epochs") == s1.get("eval_epochs")
        pair_check = {"comparable": bool(same_epochs and same_perm and same_eval),
                      "epochs": [s0["epochs_completed"], s1["epochs_completed"]],
                      "equal_permutation_digests": bool(same_perm), "equal_eval_epochs": bool(same_eval),
                      "precision_training": [s0.get("precision_training"), s1.get("precision_training")]}
    integrity = verify_frozen_now(paths.S / "state" / "verify_frozen_before_retro.json")
    if not integrity.get("all_ok"):
        raise SystemExit("frozen dataset integrity failed; refusing to lock")
    archived = {}
    for name, (vc_f, te_f, _) in ARCHIVED.items():
        archived[name] = {"VALIDATION_CONFIRM": {"path": str(GRU_FINAL / vc_f), "sha256": sha(GRU_FINAL / vc_f)},
                          "TEST": {"path": str(GRU_FINAL / te_f), "sha256": sha(GRU_FINAL / te_f)}}
    gate_c = paths.S / "review" / "GATE_C_PASS.json"
    pinned = json.loads(gate_c.read_text(encoding="utf-8")).get("pinned_runtime_sha256", {}) if gate_c.exists() else {}
    runtime_drift = {f: {"pinned": h, "now": sha(paths.CODE / f)} for f, h in pinned.items() if sha(paths.CODE / f) != h}
    lock = {"created_utc": utc(), "host": socket.gethostname(), "label": LABEL,
            "gate_c": {"path": str(gate_c), "sha256": sha(gate_c)}, "runtime_code_drift_since_gate_c": runtime_drift,
            "test_exposure": "previously_seen_in_GRU_project", "vc_exposure": "previously_seen_in_GRU_project",
            "selected_configuration": chosen, "selection": sel, "primary_seed": primary_seed, "seeds": seeds,
            "models": models, "pair_check": pair_check, "archived_baselines": archived,
            "gru_lock": {"path": str(GRU_LOCK), "sha256": sha(GRU_LOCK)},
            "gru_results": {k: {"path": str(v), "sha256": sha(v)} for k, v in GRU_RESULTS.items()},
            "registry_sha256": reg_sha, "code_sha256": ts.code_sha256_map(),
            "dataset_version_hash": paths.VERSION_HASH, "integrity": integrity}
    exclusive_json(LOCK_PATH, lock)
    log(f"wrote {LOCK_PATH} sha256={sha(LOCK_PATH)}")
    return 0


# ---------------------------------------------------------------------------- eval
def headline(res: dict, extra_k: dict) -> dict:
    def blk(b):
        return {"n": b["n"], "users": b["macro"].get("user_count"),
                "micro": {k: float(v) for k, v in b["micro"].items()},
                "macro": {k: float(v) for k, v in b["macro"].items() if k != "user_count"}}
    out = {"end_to_end": blk(res["end_to_end"]), "closed": blk(res["closed"])}
    for k, a in extra_k.items():
        for den in ("end_to_end", "closed"):
            out[den]["micro"].update({kk: float(vv) for kk, vv in a[den]["micro"].items()})
            out[den]["macro"].update({kk: float(vv) for kk, vv in a[den]["macro"].items() if kk != "user_count"})
    out["cohorts"] = {c: {"n": res["cohorts"][c]["end_to_end"]["n"],
                          "macro_mrr@20": float(res["cohorts"][c]["end_to_end"]["macro"]["mrr@20"]),
                          "micro_recall@20": float(res["cohorts"][c]["end_to_end"]["micro"]["recall@20"])}
                      for c in ("warm", "unseen")}
    out["slices"] = res.get("slices")
    return out


def metrics_from_ranks(ranks: np.ndarray, ds) -> tuple[dict, dict]:
    from eval_full import aggregate
    base = aggregate(ranks, ds, k=20)
    extra = {k: aggregate(ranks, ds, k=k) for k in KS if k != 20}
    return base, extra


def fingerprint(ds) -> dict:
    def h(a):
        return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()
    return {"n": int(len(ds)), "decision_id_sha256": ds.sha256(),
            "eval_mask_sha256": h(np.asarray(ds.eval_mask, dtype=bool)),
            "target_class_sha256": h(np.asarray(ds.target_class, dtype=np.int64)),
            "eval_user_key_sha256": h(np.asarray(ds.eval_user_key, dtype=np.int64)),
            "n_eval": int(np.asarray(ds.eval_mask).sum()),
            "n_oov": int((np.asarray(ds.eval_mask) & (np.asarray(ds.target_class) < 0)).sum()),
            "users": int(np.unique(np.asarray(ds.eval_user_key)[np.asarray(ds.eval_mask, dtype=bool)]).shape[0])}


def verify_archived(name: str, split: str, ranks: np.ndarray, ds, gru_json: dict) -> dict:
    """Fingerprint gate: the rebuilt population must reproduce the archived headline exactly."""
    key = ARCHIVED[name][2]
    ref = gru_json[key]
    rankable = np.asarray(ds.eval_mask, dtype=bool) & (np.asarray(ds.target_class) >= 0)
    if ranks.shape[0] != len(ds):
        raise SystemExit(f"{name}/{split}: archived ranks length {ranks.shape[0]} != population {len(ds)}")
    if not np.array_equal(ranks >= 1, rankable) or bool((ranks[~rankable] != -1).any()):
        raise SystemExit(f"{name}/{split}: archived rank support does not match the rebuilt population")
    base, extra = metrics_from_ranks(ranks, ds)
    checks = {}
    for den in ("end_to_end", "closed"):
        for kind in ("micro", "macro"):
            for m in ("mrr@20", "recall@20", "ndcg@20"):
                a, b = float(base[den][kind][m]), float(ref[den][kind][m])
                checks[f"{den}.{kind}.{m}"] = (a, b)
                if a != b:
                    raise SystemExit(f"{name}/{split}: fingerprint gate failed on {den}.{kind}.{m}: {a} != {b}")
    if int(base["end_to_end"]["n"]) != int(ref["end_to_end"]["n"]):
        raise SystemExit(f"{name}/{split}: population size differs")
    return {"exact_match_fields": len(checks), "base": base, "extra": extra}


def load_sasrec(info: dict, device, store=None) -> torch.nn.Module:
    from sasrec_model import SASRecConfig, SASRecCE
    cfg = SASRecConfig.from_registry(info["registry_entry"])
    if store is not None and getattr(store, "version_hash", None) == "SYNTHETIC":
        model = SASRecCE(cfg, vocab_sizes=store.vocab_sizes, product_idx_of_class=store.product_idx_of_class,
                         seed=int(info["seed"]))
    else:
        import train as reuse_train
        model = SASRecCE(cfg, vocab_sizes=dict(reuse_train.VOCAB_SIZES),
                         product_idx_of_class=reuse_train.load_product_idx_of_class(), seed=int(info["seed"]))
    if sha(info["best_weights"]) != info["best_weights_sha256"]:
        raise SystemExit(f"best weights changed since lock: {info['best_weights']}")
    obj = torch.load(info["best_weights"], map_location="cpu", weights_only=False)
    sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
    model.load_state_dict(sd)
    return model.to(device).eval()


def cmd_eval(args) -> int:
    import precision
    from eval_full import evaluate_model, per_user_metric, paired_bootstrap
    import sasrec_data
    mech = args.mechanics_split
    if mech:
        if mech != "VALIDATION_SELECT":
            raise SystemExit("mechanics mode only on VALIDATION_SELECT")
        out_dir = Path(args.out_dir)
        if not str(out_dir).startswith(str(paths.S / "tests")):
            raise SystemExit("mechanics output must live under $S/tests")
        lock = json.loads(Path(args.lock).read_text(encoding="utf-8"))
        splits = ("VALIDATION_SELECT",)
    else:
        if not LOCK_PATH.exists():
            raise SystemExit("no SASREC_LOCK.json: run `lock` first")
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        lock_sha = sha(LOCK_PATH)
        if CLAIM_PATH.exists():
            claim = json.loads(CLAIM_PATH.read_text(encoding="utf-8"))
            if claim.get("lock_sha256") != lock_sha:
                raise SystemExit("an evaluation was claimed under a different lock: refusing a second look")
            log("resuming the SAME locked retrospective evaluation (missing shards only)")
        else:
            exclusive_json(CLAIM_PATH, {"lock_sha256": lock_sha, "claimed_utc": utc(), "host": socket.gethostname()})
        if DONE_PATH.exists():
            log("retrospective evaluation already complete")
            return 0
        out_dir = OUT
        splits = ("VALIDATION_CONFIRM", "TEST")
    precision.configure_strict_fp32(2026)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    results = {"label": LABEL, "lock_sha256": None if mech else sha(LOCK_PATH), "splits": {}, "models": {},
               "bootstraps": {}, "seed_variation": {}, "archived_fingerprint_gate": {}}
    manifest_p = out_dir / "manifest.json"
    manifest = json.loads(manifest_p.read_text(encoding="utf-8")) if manifest_p.exists() else {}
    ranks_by = {}
    for split in splits:
        sdir = out_dir / split
        sdir.mkdir(parents=True, exist_ok=True)
        stores = {}
        for ch in ("F0", "F4"):
            need = [n for n, m in lock["models"].items() if m.get("status") == "COMPLETE"
                    and (m["group"] == "rich_frozen_channels") == (ch == "F4")]
            if not need and not (ch == "F0"):
                continue
            if mech and lock.get("synthetic"):
                from synthetic_store import SyntheticStore
                stores[ch] = SyntheticStore.build(channels=ch, **lock["synthetic"])
            elif mech:
                stores[ch] = sasrec_data.build_training_store(ch)
            else:
                stores[ch] = sasrec_data.build_eval_store(ch, split, lock_path=LOCK_PATH)
            ds = stores[ch].decisions(split, "eval")
            fp = fingerprint(ds)
            if "fingerprint" in results["splits"].get(split, {}) and results["splits"][split]["fingerprint"] != fp:
                raise SystemExit(f"{split}: F0 and F4 stores disagree on the population")
            results["splits"][split] = {"fingerprint": fp}
            for name, info in lock["models"].items():
                if info.get("status") != "COMPLETE" or (info["group"] == "rich_frozen_channels") != (ch == "F4"):
                    continue
                rp = sdir / f"{name}.ranks.npz"
                jp = sdir / f"{name}.json"
                key = f"{split}/{name}"
                if key in manifest and sha(rp) == manifest[key]["ranks_sha256"] and jp.exists():
                    ranks = np.load(rp)["ranks"].astype(np.int64)
                else:
                    model = load_sasrec(info, device, stores[ch])
                    with precision.strict_eval():
                        res = evaluate_model(model, stores[ch], ds, device, eval_batch=1024, k=20, return_ranks=True)
                    ranks = np.asarray(res.pop("ranks"), dtype=np.int64)
                    tmp = sdir / f"{name}.ranks.tmp{os.getpid()}.npz"
                    np.savez_compressed(tmp, ranks=ranks.astype(np.int32))
                    os.replace(tmp, rp)
                    del model
                    torch.cuda.empty_cache() if device.type == "cuda" else None
                base, extra = metrics_from_ranks(ranks, ds)
                hl = headline(base, extra)
                atomic_json(jp, hl)
                manifest[key] = {"ranks_sha256": sha(rp), "json_sha256": sha(jp), "utc": utc()}
                atomic_json(manifest_p, manifest)
                results["models"].setdefault(name, {"source": "evaluated", "roles": info["roles"],
                                                    "run_id": info["run_id"], "seed": info["seed"]})[split] = hl
                ranks_by[(split, name)] = (ranks, ds)
            if ch == "F0" and not mech:
                gru_json = json.loads(GRU_RESULTS[split].read_text(encoding="utf-8"))
                for name, files in ARCHIVED.items():
                    rpath = GRU_FINAL / (files[0] if split == "VALIDATION_CONFIRM" else files[1])
                    rsha = sha(rpath)
                    if rsha != lock["archived_baselines"][name][split]["sha256"]:
                        raise SystemExit(f"archived ranks changed since lock: {rpath}")
                    ranks = np.load(rpath).astype(np.int64)
                    v = verify_archived(name, split, ranks, ds, gru_json)
                    results["archived_fingerprint_gate"][f"{split}/{name}"] = {
                        "exact_match_fields": v["exact_match_fields"], "ranks_sha256": rsha}
                    results["models"].setdefault(name, {"source": "archived_ranks"})[split] = headline(v["base"], v["extra"])
                    ranks_by[(split, name)] = (ranks, ds)
        del stores
    # bootstraps (TEST, or the mechanics split)
    boot_split = "TEST" if not mech else "VALIDATION_SELECT"
    primary = f"{lock['selected_configuration']}_s{lock['primary_seed']}"
    pairs = [(primary, "GRU_F0_locked"), (primary, "TRANS_full")]
    if lock.get("pair_check", {}).get("comparable"):
        pairs.append(("A1_RICH256_s2026", "A0_ID256_s2026"))
    else:
        results["bootstraps_skipped"] = {"A1_RICH256_s2026_minus_A0_ID256_s2026": lock.get("pair_check")}
    for a, b in pairs:
        if (boot_split, a) in ranks_by and (boot_split, b) in ranks_by:
            ra, dsa = ranks_by[(boot_split, a)]
            rb, dsb = ranks_by[(boot_split, b)]
            ka, va = per_user_metric(ra, dsa, metric="mrr", k=20, denominator="end_to_end")
            kb, vb = per_user_metric(rb, dsb, metric="mrr", k=20, denominator="end_to_end")
            bt = paired_bootstrap(ka, va, kb, vb, n=BOOT_N, seed=BOOT_SEED)
            bt.update({"n_resamples": BOOT_N, "seed": BOOT_SEED, "metric": "user-macro MRR@20 end_to_end",
                       "note": "conditional on these fitted models and a previously inspected period"})
            results["bootstraps"].setdefault(boot_split, {})[f"{a}_minus_{b}"] = bt
    seeds = {}
    for s in lock["seeds"]:
        nm = f"{lock['selected_configuration']}_s{s}"
        m = results["models"].get(nm, {}).get(boot_split)
        if m:
            seeds[str(s)] = m["end_to_end"]["macro"]["mrr@20"]
    if seeds:
        vals = np.array(list(seeds.values()), dtype=np.float64)
        results["seed_variation"] = {"config": lock["selected_configuration"], "split": boot_split,
                                     "macro_mrr20_by_seed": seeds, "mean": float(vals.mean()),
                                     "std_ddof1": float(vals.std(ddof=1)) if vals.size > 1 else None,
                                     "note": "user bootstrap CIs do not capture training-seed variance"}
    out_path = out_dir / ("RETRO_RESULTS.json" if not mech else "MECHANICS_RESULTS.json")
    atomic_json(out_path, results)
    if not mech:
        exclusive_json(DONE_PATH, {"completed_utc": utc(), "results": str(out_path), "results_sha256": sha(out_path)})
    log(f"wrote {out_path}")
    return 0


def cmd_status(_args) -> int:
    for p in (LOCK_PATH, CLAIM_PATH, DONE_PATH, OUT / "manifest.json"):
        print(p, "exists" if p.exists() else "missing", sha(p) or "")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lock")
    e = sub.add_parser("eval")
    e.add_argument("--mechanics-split", default=None)
    e.add_argument("--out-dir", default=None)
    e.add_argument("--lock", default=None, help="mechanics mode: a test lock JSON")
    sub.add_parser("status")
    a = ap.parse_args(argv)
    return {"lock": cmd_lock, "eval": cmd_eval, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
