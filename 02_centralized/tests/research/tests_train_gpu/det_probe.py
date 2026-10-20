"""SYS-04 -- training determinism probe on the A16 (evidence for PREG1_AMENDMENT_01 item 4).

Drives the REAL v4train.CentralTrainer (not a stand-in) on synthetic providers, small N and K, for ~300 updates per
run: fused AdamW with the C8 groups, clip 1.0, full-K CE, dropout on, exposure_point "end". Two modes:

  strict   RunConfig(determinism_mode="STRICT"): deterministic algorithms, warn_only off. The trainer refuses to
           construct unless the launcher exported CUBLAS_WORKSPACE_CONFIG ":4096:8" or ":16:8" (checked before
           anything else -- CentralTrainer._check_cublas). Records whether any op raises, and which, with the full
           error text; also records a construction-time refusal (e.g. the CUBLAS negative check) the same way.
  warn     RunConfig(determinism_mode="WARN_ONLY", test_overrides=True; WARN_ONLY needs an override since STRICT is
           the locked value). Captures every nondeterminism warning, deduplicated by message.

SYS-04c (TRAIN-01R1): `determinism_mode` is now a REQUIRED RunConfig key the trainer applies, verifies against the
live torch state, and records (RUN_START, every checkpoint, RUN_SUMMARY, every result row, the mode-derived
`precision_id`); a resume whose mode or CUBLAS setting drifted is refused. This closes the gap the original SYS-04
probe wrapper stood in for: **the wrapper is gone**. `run` passes `determinism_mode` straight through `RunConfig`;
the real trainer applies and re-verifies it (at construction, at every run/resume start, and after every evaluation)
with no help from this script. `trainer_code_sha256()` (from `v4train.trainer`, computed over the live v4train/*.py
files) is recorded in every result so a receipt is traceable to the exact trainer code that produced it; the SYS-04 /
SYS-04b receipts were produced under trainer hash `5f217943...` (pre-TRAIN-01R1) and are left untouched as historical
evidence -- they are not reproducible against the current trainer (which now refuses a config with no
`determinism_mode` at all).

Replication: each mode that runs is repeated twice in FRESH processes (`run` subcommand invoked twice) with the same
seed; `compare` loads both runs' saved parameter state (.pt) and result JSON and reports bitwise equality (parameters,
optimizer state digest, per-update loss sequence) and the maximum absolute parameter difference. `compare` also
accepts two DIFFERENT modes (e.g. strict rep 1 vs warn rep 1): it then records `kind: "cross_mode"` instead of
`"replication"`, so `summarize` keeps the two kinds of evidence apart -- the family x mode table (run-to-run
replication within one mode) plus a separate `cross_mode` list (does warn-only already agree with strict, bit for
bit). `summarize` merges every run_*.json / compare_*.json in a directory into that report.

SYS-04b (production shapes): `--K` / `--N` on the `run` subcommand override the small-catalogue defaults below so the
same script can be re-run at the real catalogue size (K=158,486) without duplicating it; `--updates` already existed.
Peak GPU memory (`torch.cuda.max_memory_allocated`, reset before schema/provider/model construction so the figure
covers the whole run, not just the loop) is recorded in every `run` result alongside updates/s.

`--allow-unset-cublas` (SYS-04c): skips this script's OWN CUBLAS preflight guard (below) so a `--mode strict` run
with `CUBLAS_WORKSPACE_CONFIG` genuinely unset reaches the real trainer and is refused BY IT
(`CentralTrainer._check_cublas`, the first thing `__init__` does, before any provider/model/CUDA touch) -- used only
for the negative-refusal check, never for a real determinism run.

Subcommands (mirrors code/fl/gpu_receipt.py's pattern: one OS process per check, then a summarize pass):
    det_probe.py run     --family GRU|SASREC --mode strict|warn --rep 1|2 --device cuda:0 --updates 300
                          [--K 2000] [--N 200000] [--allow-unset-cublas] --out run.json --state state.pt
                          --work <run_dir>
    det_probe.py compare --a run_a.json --state-a state_a.pt --b run_b.json --state-b state_b.pt --out compare.json
    det_probe.py summarize --dir <results_dir> --out SYS04_DET_RECEIPT.json

Synthetic data only; nothing here reads a dataset row or a real manifest. `--device cpu` gives the local syntax /
plumbing check used before the server run (no strict-mode determinism error is expected on CPU; it only proves the
code path runs end to end).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for _p in (str(CODE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import v4train  # noqa: E402,F401  (entry point: captures CUDA_VISIBLE_DEVICES before ppsi_env's own import, which
                 # v4train/__init__.py performs internally -- the same order code/tests_train/test_cuda_visibility.py
                 # 's PROBE uses; nothing else may be imported before this)

import torch  # noqa: E402

from fl.numerics import max_abs_diff, state_digest, states_equal  # noqa: E402
from v4models.schema import load_schema  # noqa: E402
from v4train.provider import SyntheticEvalProvider, SyntheticTrainProvider  # noqa: E402
from v4train.recipe import RunConfig  # noqa: E402
from v4train.trainer import CentralTrainer, InjectedInterruption, TrainerError, trainer_code_sha256  # noqa: E402

DEFAULT_K = 2_000                   # small catalogue (SYS-04: "small N and K"); still exercises full-K CE
DEFAULT_N = 200_000                 # chosen so DEFAULT_UPDATES * EFFECTIVE_BATCH stays under the first 0.5-EFE mark
EFFECTIVE_BATCH = 256                # C8 recipe value
DEFAULT_UPDATES = 300
SEED = 2026                        # a registered main-run seed (recipe.RECIPE.seeds); same seed across repeats
CUBLAS_WORKSPACE_CONFIG = ":4096:8"  # the frozen T3 value; v4train.trainer requires this for ANY cuda run (T3, not
                                     # only strict mode) -- see CentralTrainer.__init__'s CUBLAS_WORKSPACE_CONFIG check
KEY = {"data_protocol_hash": "SYS04-SYN-DATA-PROTOCOL", "feature_contract_hash": "SYS04-SYN-FEATURES",
       "split": "VALIDATION", "label_policy_hash": "SYS04-SYN-LABEL-POLICY",
       "population_hash": "SYS04-SYN-POPULATION", "candidate_order_hash": "SYNTHETIC-CANDIDATES"}


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _config(run_id: str, family: str, mode: str) -> RunConfig:
    determinism_mode = "STRICT" if mode == "strict" else "WARN_ONLY"
    return RunConfig(run_id=run_id, family=family, seed=SEED, exposure_point="end",
                     cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), determinism_mode=determinism_mode,
                     phase="MAIN", peak_lr=1e-3, effective_batch=EFFECTIVE_BATCH, log_every=1, checkpoint_every=0,
                     key=dict(KEY), test_overrides=True)


def _providers(schema, K: int, N: int):
    widths = schema.widths
    tp = SyntheticTrainProvider(N, K, widths, seed=SEED, manifest_id="M01_COHORT_TRAIN")
    ep = SyntheticEvalProvider(min(2_000, N), K, widths, seed=SEED + 1, manifest_id="M02_VALIDATION_WARM")
    return tp, ep


def _stop_after(n_updates: int):
    """Raises InjectedInterruption right after update n_updates (trainkit.py's proven pattern): a clean, intentional
    stop of the REAL training loop well before any evaluation mark, not a trainer change."""
    def f(u: int) -> None:
        if u == n_updates:
            raise InjectedInterruption(f"SYS-04 probe: intentional stop after update {u}")
    return f


def _opt_state_bytes(opt: "torch.optim.Optimizer") -> bytes:
    """Canonical bytes of AdamW's state_dict (CPU tensors, keys sorted by repr) for a bitwise sha256 digest."""
    def enc(o) -> bytes:
        if torch.is_tensor(o):
            t = o.detach().to("cpu").contiguous()
            return b"T|" + str(t.dtype).encode() + b"|" + str(tuple(t.shape)).encode() + b"|" + t.numpy().tobytes()
        if isinstance(o, dict):
            return b"{" + b"".join(enc(k) + b":" + enc(o[k]) for k in sorted(o, key=repr)) + b"}"
        if isinstance(o, (list, tuple)):
            return b"[" + b",".join(enc(x) for x in o) + b"]"
        return repr(o).encode()
    return enc(opt.state_dict())


# --------------------------------------------------------------------------------------------------------- run
def cmd_run(args) -> int:
    family = args.family.upper()
    mode = args.mode
    device = args.device
    n_updates = int(args.updates)
    K, N = int(args.K), int(args.N)
    out = Path(args.out)
    state_out = Path(args.state) if args.state else None
    run_dir = Path(args.work)

    if device.startswith("cuda"):
        if not args.allow_unset_cublas and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
            raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must be {CUBLAS_WORKSPACE_CONFIG!r} before CUDA init "
                              f"(export it in the shell before python starts); got "
                              f"{os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}")
        if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
            raise RuntimeError("CUDA_VISIBLE_DEVICES is unset: export it in the shell before python starts (SYS-04: "
                              "GPU 1 only)")
    device_obj = torch.device(device) if device.startswith("cuda") else None

    result = {"family": family, "mode": mode, "rep": int(args.rep), "seed": SEED, "N": N, "K": K,
             "effective_batch": EFFECTIVE_BATCH, "updates_target": n_updates, "device": device,
             "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
             "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "torch": torch.__version__, "cuda_build": torch.version.cuda,
             "trainer_code_sha256": trainer_code_sha256(),
             "gpu_name": (torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available()
                          else None),
             "utc_start": _utc()}
    if run_dir.exists():
        shutil.rmtree(run_dir)
    if device_obj is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device_obj)     # reset before any allocation: covers model+optimizer too
    caught_msgs: list = []
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            schema = load_schema()
            tp, ep = _providers(schema, K, N)
            cfg = _config(f"sys04-{family.lower()}-{mode}-r{args.rep}", family, mode)
            tr = CentralTrainer(cfg, schema, tp, ep, run_dir, device=device, K=K, fault=_stop_after(n_updates))
            t0 = time.perf_counter()
            try:
                tr.run()
                raise RuntimeError(f"SYS-04 probe misconfigured: the run reached phase completion before update "
                                  f"{n_updates} (N={N} too small for this budget); no determinism verdict recorded")
            except InjectedInterruption:
                pass
            wall = time.perf_counter() - t0
            caught_msgs = [f"{x.category.__name__}: {x.message}" for x in w]
        recs = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        losses = [r["loss_mean"] for r in recs if r["kind"] == "monitor"]
        state = {k: v.detach().to("cpu").clone() for k, v in tr.adapter.broadcast_state(clone=True).items()}
        nondet_warnings = sorted({m for m in caught_msgs if "determinist" in m.lower()})
        run_start = _read_json(run_dir / "RUN_START.json")
        result.update(status="RAN", updates_done=int(tr.state.updates), wall_s=wall,
                     updates_per_s=(tr.state.updates / wall if wall > 0 else None),
                     param_state_digest=state_digest(state),
                     optimizer_state_digest=hashlib.sha256(_opt_state_bytes(tr.opt)).hexdigest(),
                     n_loss_records=len(losses), loss_sequence=[float(x) for x in losses],
                     deterministic_algorithms_effective=bool(torch.are_deterministic_algorithms_enabled()),
                     warn_only_effective=bool(torch.is_deterministic_algorithms_warn_only_enabled()),
                     n_warnings_total=len(caught_msgs), nondeterminism_warnings=nondet_warnings,
                     run_start_determinism=run_start.get("determinism"),
                     run_start_precision_id=run_start.get("precision_id"),
                     run_start_trainer_code_sha256=run_start.get("trainer_code_sha256"))
        if state_out is not None:
            torch.save(state, state_out)
    except (RuntimeError, TrainerError) as e:
        nondet_warnings = sorted({m for m in caught_msgs if "determinist" in m.lower()})
        result.update(status="RAISES", error_type=type(e).__name__, error_text=str(e),
                     error_traceback=traceback.format_exc(limit=40), n_warnings_total=len(caught_msgs),
                     nondeterminism_warnings=nondet_warnings)
    if device_obj is not None and torch.cuda.is_available():
        peak = int(torch.cuda.max_memory_allocated(device_obj))
        result.update(peak_memory_allocated_bytes=peak, peak_memory_allocated_gib=peak / 2 ** 30)
    else:
        result.update(peak_memory_allocated_bytes=None, peak_memory_allocated_gib=None)
    result["utc_end"] = _utc()
    _write_json(out, result)
    print(json.dumps({k: result[k] for k in ("family", "mode", "rep", "status", "updates_done", "updates_per_s",
                                             "peak_memory_allocated_gib", "error_type", "trainer_code_sha256")
                      if k in result}, default=str))
    return 0


# ----------------------------------------------------------------------------------------------------- compare
def cmd_compare(args) -> int:
    a, b = _read_json(args.a), _read_json(args.b)
    same_mode = (a["mode"] == b["mode"])
    out = {"family": a["family"], "mode_a": a["mode"], "mode_b": b["mode"],
          "mode": (a["mode"] if same_mode else f"{a['mode']}_vs_{b['mode']}"),
          "kind": ("replication" if same_mode else "cross_mode"), "rep_a": a["rep"], "rep_b": b["rep"]}
    if a["status"] != "RAN" or b["status"] != "RAN":
        out.update(comparable=False, reason=f"a.status={a['status']!r} b.status={b['status']!r}")
        _write_json(Path(args.out), out)
        print(json.dumps(out, default=str))
        return 0
    sa = torch.load(args.state_a, map_location="cpu", weights_only=True)
    sb = torch.load(args.state_b, map_location="cpu", weights_only=True)
    bitwise_params = states_equal(sa, sb)
    param_diff = max_abs_diff(sa, sb)
    opt_equal = a["optimizer_state_digest"] == b["optimizer_state_digest"]
    la, lb = a["loss_sequence"], b["loss_sequence"]
    loss_equal = (la == lb)
    loss_diff = max((abs(x - y) for x, y in zip(la, lb)), default=0.0) if len(la) == len(lb) else float("inf")
    out.update(comparable=True, bitwise_params=bitwise_params, param_max_abs_diff=param_diff,
              optimizer_state_digest_equal=opt_equal, loss_sequence_equal=loss_equal,
              loss_sequence_max_abs_diff=loss_diff, n_loss_records_a=len(la), n_loss_records_b=len(lb),
              bitwise_overall=bool(bitwise_params and opt_equal and loss_equal))
    _write_json(Path(args.out), out)
    print(json.dumps(out, default=str))
    return 0


# --------------------------------------------------------------------------------------------------- summarize
def cmd_summarize(args) -> int:
    d = Path(args.dir)
    runs: dict = {}
    all_runs: list = []
    for p in sorted(d.glob("run_*.json")):
        r = _read_json(p)
        runs.setdefault((r["family"], r["mode"]), []).append(r)
        all_runs.append(r)
    compares: dict = {}
    cross_mode: list = []
    for p in sorted(d.glob("compare_*.json")):
        c = _read_json(p)
        if c.get("kind", "replication") == "cross_mode":
            cross_mode.append(c)
        else:
            compares[(c["family"], c.get("mode", c.get("mode_a")))] = c
    table = []
    for (family, mode), reps in sorted(runs.items()):
        reps = sorted(reps, key=lambda r: r["rep"])
        runs_ok = all(r["status"] == "RAN" for r in reps) and len(reps) > 0
        cmp_ = compares.get((family, mode))
        row = {"family": family, "mode": mode, "n_reps": len(reps), "runs": runs_ok,
              "updates_per_s": [r.get("updates_per_s") for r in reps],
              "peak_memory_allocated_gib": [r.get("peak_memory_allocated_gib") for r in reps],
              "offending_ops": sorted({f"{r['error_type']}: {r['error_text']}" for r in reps
                                      if r["status"] == "RAISES"}),
              "nondeterminism_warnings": sorted({m for r in reps for m in r.get("nondeterminism_warnings", [])})}
        if cmp_ and cmp_.get("comparable"):
            row.update(bitwise_run_to_run=cmp_["bitwise_overall"], param_max_abs_diff=cmp_["param_max_abs_diff"],
                      loss_sequence_equal=cmp_["loss_sequence_equal"])
        else:
            row.update(bitwise_run_to_run=None, param_max_abs_diff=None, loss_sequence_equal=None)
        table.append(row)
    cross_table = [{"family": c["family"], "mode_a": c["mode_a"], "mode_b": c["mode_b"],
                    "comparable": c.get("comparable"), "bitwise_overall": c.get("bitwise_overall"),
                    "param_max_abs_diff": c.get("param_max_abs_diff"),
                    "loss_sequence_equal": c.get("loss_sequence_equal")}
                   for c in sorted(cross_mode, key=lambda c: (c["family"], c["mode_a"], c["mode_b"]))]
    negative_checks = []
    for p in sorted(d.glob("negcheck_*.json")):
        nc = _read_json(p)
        negative_checks.append({"name": p.stem, "family": nc.get("family"), "mode": nc.get("mode"),
                                "device": nc.get("device"), "cublas_workspace_config": nc.get("cublas_workspace_config"),
                                "status": nc.get("status"), "refused": nc.get("status") == "RAISES",
                                "error_type": nc.get("error_type"), "error_text": nc.get("error_text")})
    cfg_src = all_runs[0] if all_runs else {}
    receipt = {"what": "SYS-04 training determinism probe (v4train.CentralTrainer, synthetic providers, A16 GPU 1; "
                      "evidence for PREG1_AMENDMENT_01 item 4)",
              "utc": _utc(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
              "config": {"N": cfg_src.get("N", DEFAULT_N), "K": cfg_src.get("K", DEFAULT_K),
                         "effective_batch": cfg_src.get("effective_batch", EFFECTIVE_BATCH),
                         "seed": cfg_src.get("seed", SEED), "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
                         "trainer_code_sha256": cfg_src.get("trainer_code_sha256"),
                         "determinism_path": "RunConfig.determinism_mode (SYS-04c / TRAIN-01R1): the real trainer "
                                             "applies and verifies the mode; no probe wrapper is used"},
              "table": table, "cross_mode": cross_table, "negative_checks": negative_checks}
    _write_json(Path(args.out), receipt)
    print(json.dumps({"table": table, "cross_mode": cross_table, "negative_checks": negative_checks},
                     default=str)[:4000])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--family", required=True, choices=["GRU", "SASREC", "gru", "sasrec"])
    r.add_argument("--mode", required=True, choices=["strict", "warn"])
    r.add_argument("--rep", required=True, type=int)
    r.add_argument("--device", default="cuda:0")
    r.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    r.add_argument("--K", type=int, default=DEFAULT_K)
    r.add_argument("--N", type=int, default=DEFAULT_N)
    r.add_argument("--allow-unset-cublas", action="store_true",
                   help="skip this script's own CUBLAS preflight guard so a STRICT run with the env var genuinely "
                        "unset reaches the real trainer's own refusal (negative check only)")
    r.add_argument("--out", required=True)
    r.add_argument("--state", default=None)
    r.add_argument("--work", required=True)

    c = sub.add_parser("compare")
    c.add_argument("--a", required=True)
    c.add_argument("--state-a", required=True)
    c.add_argument("--b", required=True)
    c.add_argument("--state-b", required=True)
    c.add_argument("--out", required=True)

    s = sub.add_parser("summarize")
    s.add_argument("--dir", required=True)
    s.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "compare":
        return cmd_compare(args)
    return cmd_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
