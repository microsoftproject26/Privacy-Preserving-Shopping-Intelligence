"""SYS-04c -- synthetic STRICT bench of the new model variants on the A16 (TRAIN-02 item 0c; PREG1_AMENDMENT_02R1 §4).

Drives the REAL v4train.CentralTrainer (STRICT determinism, fused AdamW C8 groups, clip 1.0, full-K CE, dropout on,
exposure_point "end", B 256) on synthetic providers at the full catalogue K = 158,486 with USER_HISTORY_L50-length
windows (max_len 50), built from the production schema S* (DATA_PROTOCOL_LOCK_V2) through each variant's HP trial
(role TUNING, test_overrides: SMOKE_NOT_EVIDENCE; the first trial of the variant in HP_REGISTRY_P2 gives its peak LR,
dropout and schedule). Each rep stops after --updates updates (default 300, well before the first 2.0-EFE mark: no
evaluation, no weights kept) and records updates/s (after a 20-update warmup, from a per-update clock) and the peak GPU
memory (torch.cuda.max_memory_allocated, reset before the model is built), plus the final parameter-state digest.

  CUDA_VISIBLE_DEVICES=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B code/tests_train_gpu/sys04c_variant_bench.py all \
      --out-dir <dir> [--variants GRU_D256 SASREC_D256_B3 SASREC_D384_B2] [--reps 2] [--updates 300]
  (subcommands: `run` = one rep in this process; `all` = every variant x rep in FRESH processes, then the summary)

Summary (SYS04C_VARIANT_BENCH.json): per variant the two reps' updates/s and peak GiB, whether the reps are bitwise equal
(parameter digest and loss sequence: the "one short STRICT bitwise run per variant" check), and the 02R1 §4 triggers:
a variant below 0.5 x its assumed rate (GRU_D256 12 upd/s, SASREC_D384_B2 6.5 upd/s; SASREC_D256_B3 has no registered
rate: reported only) or peaking above 8 GiB. `--device cpu --K 500 --N 4000 --updates 12` is the local plumbing check.
Synthetic data only; no dataset row, no manifest, no MAIN_VALIDATION.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for _p in (str(CODE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ASSUMED_UPD_S = {"GRU_D256": 12.0, "SASREC_D384_B2": 6.5}      # 02R1 §4
TRIGGER_FRACTION = 0.5
PEAK_GIB_MAX = 8.0
NEW_VARIANTS = ("GRU_D256", "SASREC_D256_B3", "SASREC_D384_B2")
KEY = {"data_protocol_hash": "SYS04C-SYN", "feature_contract_hash": "SYS04C-SYN", "split": "VALIDATION",
       "label_policy_hash": "SYS04C-SYN", "population_hash": "SYS04C-SYN", "candidate_order_hash": "SYNTHETIC-CANDIDATES"}


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8", newline="\n")


def _add_session_keys(p, n: int, K: int, widths, seed: int) -> None:
    """The SCHEMA-01R1 / DL 48 session keys the S* projection needs: synthetic_examples_v2 draws them AFTER every other
    tensor, so regenerating from the provider's seed with them on leaves the provider's own tensors unchanged."""
    import torch
    from v4models.synthetic import synthetic_examples_v2
    ex = synthetic_examples_v2(int(n), int(K), widths, torch.Generator().manual_seed(int(seed)), max_len=50, window=50,
                               with_session_offset=True, with_session_intent=True)
    if not torch.equal(ex["item_tokens"], p._ex["item_tokens"]):
        raise RuntimeError("session-key regeneration changed the provider's tensors")
    for k, v in ex.items():
        if k not in p._ex:
            p._ex[k] = v


def cmd_run(args) -> int:
    import v4train  # noqa: F401  (captures CUDA_VISIBLE_DEVICES before ppsi_env)
    import torch
    from fl.numerics import state_digest
    from v4models import variants as V
    from v4train.hp_registry import load_hp_registry
    from v4train.provider import SyntheticEvalProvider, SyntheticTrainProvider
    from v4train.recipe import RunConfig
    from v4train.registry import with_self_sha256
    from v4train.trainer import CentralTrainer, InjectedInterruption, trainer_code_sha256
    from v4eval.frozen_config import eval_device_class

    name, K, N, U = args.variant, int(args.K), int(args.N), int(args.updates)
    fam = V.VARIANTS[name]["family"]
    if name == "SASREC_D64_B2":
        from v4int import small_recipe
        small_recipe.install()
        registry = small_recipe.load_ladder(args.ladder)
    else:
        registry = load_hp_registry(args.hp_registry) if args.hp_registry else load_hp_registry()
    t = [x for x in registry["families"][fam] if x["model_variant"] == name][0]
    if args.schema == "BASE":
        from v4models.schema import load_schema
        schema, widths = load_schema(), None
    else:
        from v4fs.schemas import load_registered
        from v4models.schema import load_schema
        schema = load_registered(args.schema)
        widths = load_schema().widths                   # providers serve the full Clean V2 layout; the adapter projects
    base_w = widths or schema.widths
    run_dir = Path(args.work)
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.zeros(1, device=dev)                      # create the CUDA context first (R5 / OPS-11: 'Invalid device argument')
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    tp = SyntheticTrainProvider(N, K, base_w, seed=2026, max_len=50, manifest_id="SYS04C_TRAIN")
    ep = SyntheticEvalProvider(min(2000, N), K, base_w, seed=2027, max_len=50, manifest_id="SYS04C_EVAL")
    _add_session_keys(tp, N, K, base_w, 2026)                  # S* keeps the SCHEMA-01R1 session channels
    _add_session_keys(ep, min(2000, N), K, base_w, 2027)
    ent = lambda p, n: {"n_decisions": int(n), **{h: p.manifest_binding[h] for h in ("content_sha256",  # noqa: E731
                                                                                      "header_sha256")}}
    reg = with_self_sha256({"data_version": "SYS04C_SYNTHETIC", "roles": {"CALIBRATION": {
        "train": "SYS04C_TRAIN", "eval": "SYS04C_EVAL"}}, "manifests": {"SYS04C_TRAIN": ent(tp, tp.n_decisions),
                                                                       "SYS04C_EVAL": ent(ep, ep.manifest.n)}})
    rp = run_dir.parent / f"{run_dir.name}_REGISTRY.json"
    rp.write_text(json.dumps(reg), encoding="utf-8")
    cfg = RunConfig(run_id=f"SYS04C_{name}_r{args.rep}", family=fam, seed=2026, exposure_point="end",
                    cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), determinism_mode="STRICT",
                    manifest_registry=str(rp), data_version="SYS04C_SYNTHETIC", role="TUNING", phase="CALIBRATION",
                    peak_lr=t["peak_lr"], hp_id=t["hp_id"], hp_registry=str(args.ladder) if name == "SASREC_D64_B2" else args.hp_registry,
                    model_variant=name, dropout=tuple(t["dropout"]),
                    schedule=t["schedule"], effective_batch=256, log_every=10, key=dict(KEY), test_overrides=True)
    clock = {}

    def fault(u):
        clock[u] = time.perf_counter()
        if u >= U:
            raise InjectedInterruption("SYS-04c bench: stop")

    res = {"variant": name, "family": fam, "rep": int(args.rep), "hp_id": t["hp_id"], "K": K, "N": N, "updates": U,
           "device": str(dev), "torch": torch.__version__, "trainer_code_sha256": trainer_code_sha256(),
           "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,   # INT-06: the STRICT
           "eval_device_class": eval_device_class(dev), "cuda": torch.version.cuda,        # receipt names its GPU
           "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"), "schema": schema.schema_id,
           "utc_start": _utc()}
    tr = CentralTrainer(cfg, schema, tp, ep, run_dir, device=str(dev), K=K, fault=fault)
    try:
        tr.run()
        raise RuntimeError("the bench reached a mark: N too small for --updates")
    except InjectedInterruption:
        pass
    us = sorted(clock)
    lo = [u for u in us if u >= min(20, max(1, U // 4))]
    ups = (lo[-1] - lo[0]) / (clock[lo[-1]] - clock[lo[0]]) if len(lo) > 1 else None
    recs = [json.loads(x) for x in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    res.update(updates_done=int(tr.state.updates), updates_per_s=ups, shared_numel=tr.adapter.manifest.shared_numel,
               init_sha256=tr.built.init_sha256, param_state_digest=state_digest(tr.adapter.broadcast_state()),
               loss_sequence=[r["loss_mean"] for r in recs if r["kind"] == "monitor"],
               peak_gib=(torch.cuda.max_memory_allocated(dev) / 2 ** 30) if dev.type == "cuda" else None,
               utc_end=_utc())
    for k in list(tr.ckpt.index):
        tr.ckpt.delete(k, "SYS-04c bench: no weights kept")
    shutil.rmtree(run_dir, ignore_errors=True)
    rp.unlink(missing_ok=True)
    _write(Path(args.out), res)
    print(json.dumps({k: res[k] for k in ("variant", "rep", "updates_per_s", "peak_gib", "param_state_digest")}))
    return 0


def summarize(out_dir: Path, variants) -> dict:
    table = []
    for v in variants:
        reps = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob(f"run_{v}_r*.json"))]
        ups = [r.get("updates_per_s") for r in reps]
        peaks = [r.get("peak_gib") for r in reps]
        bitwise = (len(reps) >= 2 and len({r["param_state_digest"] for r in reps}) == 1
                   and all(r["loss_sequence"] == reps[0]["loss_sequence"] for r in reps))
        trig = []
        if v in ASSUMED_UPD_S and any(u is not None and u < TRIGGER_FRACTION * ASSUMED_UPD_S[v] for u in ups):
            trig.append(f"below {TRIGGER_FRACTION} x the assumed {ASSUMED_UPD_S[v]} upd/s")
        if any(p is not None and p > PEAK_GIB_MAX for p in peaks):
            trig.append(f"peak above {PEAK_GIB_MAX} GiB")
        table.append({"variant": v, "n_reps": len(reps), "updates_per_s": ups, "peak_gib": peaks,
                      "assumed_upd_s": ASSUMED_UPD_S.get(v), "bitwise_rep_to_rep": bitwise,
                      "shared_numel": reps[0]["shared_numel"] if reps else None, "triggers": trig,
                      "eval_device_classes": sorted({str(r.get("eval_device_class")) for r in reps}),
                      "device_names": sorted({str(r.get("device_name")) for r in reps})})
    return {"what": "SYS-04c synthetic STRICT bench of the TRAIN-02 model variants (02R1 §4)", "utc": _utc(),
            "table": table, "any_trigger": any(r["triggers"] for r in table),
            "all_bitwise": all(r["bitwise_rep_to_rep"] for r in table),
            "eval_device_classes": sorted({c for r in table for c in r["eval_device_classes"]}),
            "device_names": sorted({c for r in table for c in r["device_names"]})}


def cmd_all(args) -> int:
    out = Path(args.out_dir).resolve()                 # children run with cwd=code/ (R5 / OPS-11: runs landed elsewhere)
    out.mkdir(parents=True, exist_ok=True)
    codes = {}
    for v in args.variants:
        for rep in range(1, int(args.reps) + 1):
            cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "run", "--variant", v, "--rep", str(rep),
                   "--device", args.device, "--K", str(args.K), "--N", str(args.N), "--updates", str(args.updates),
                   "--schema", args.schema, "--out", str(out / f"run_{v}_r{rep}.json"),
                   "--work", str(out / f"_work_{v}_r{rep}")]
            if args.ladder:
                cmd.extend(["--ladder", str(args.ladder)])
            if args.hp_registry:
                cmd.extend(["--hp-registry", str(args.hp_registry)])
            codes[f"{v}/r{rep}"] = subprocess.run(cmd, cwd=str(CODE)).returncode
    s = summarize(out, args.variants)
    s["exit_codes"] = codes
    s["all_checks_pass"] = bool(s["all_bitwise"] and all(c == 0 for c in codes.values())
                               and len(s["eval_device_classes"]) == 1)          # INT-06: one device class
    _write(out / "SYS04C_VARIANT_BENCH.json", s)
    print(json.dumps(s, indent=1, default=str)[:4000])
    return 0 if all(c == 0 for c in codes.values()) else 1


def _schema_default() -> str:
    lock = json.loads((CODE.parent / "contracts" / "DATA_PROTOCOL_LOCK_V2.json").read_text(encoding="utf-8"))
    return lock["features_and_transforms"]["production_schema"]["schema_id"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "all"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--K", type=int, default=158_486)
        p.add_argument("--N", type=int, default=100_000)
        p.add_argument("--updates", type=int, default=300)
        p.add_argument("--schema", default=None, help="default: DATA_PROTOCOL_LOCK_V2's S*; BASE = Clean V2")
        p.add_argument("--hp-registry", type=Path, default=None)
        p.add_argument("--ladder", type=Path, default=None)
    r = sub.choices["run"]
    r.add_argument("--variant", required=True)
    r.add_argument("--rep", type=int, required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--work", required=True)
    a = sub.choices["all"]
    a.add_argument("--out-dir", required=True)
    a.add_argument("--variants", nargs="+", default=list(NEW_VARIANTS))
    a.add_argument("--reps", type=int, default=2)
    args = ap.parse_args(argv)
    args.schema = args.schema or _schema_default()
    if "SASREC_D64_B2" in ([args.variant] if args.cmd == "run" else args.variants):
        from v4int import small_recipe
        if args.ladder is None:
            if args.cmd == "run":
                raise ValueError("--ladder is required for a SMALL single run")
            args.ladder = Path(args.out_dir).resolve() / "REGISTERED_SMALL_LADDER.json"
            args.ladder.parent.mkdir(parents=True, exist_ok=True)
            if not args.ladder.exists():
                args.ladder.write_text(json.dumps(small_recipe.ladder_obj(), sort_keys=True) + "\n", encoding="utf-8")
    return cmd_run(args) if args.cmd == "run" else cmd_all(args)


if __name__ == "__main__":
    raise SystemExit(main())
