"""v4fs.stage_b — the FS Stage B driver (FS-02).

Stage A (runs_g1/ops07_sync3/fs_stageA.sh) trains L0S_CLEAN_V2_SESSION and L3S_NEAR_DUPLICATE_SESSION x {GRU,
SASREC} x {2026, 2027} into runs_fs/stageA/<schema>__<FAMILY>__<seed>/. Stage B continues the v4fs.decide search
from there, TRAINING at most one candidate schema S* (and, on the single fallback, one more) before it LOCKs:

  python -B code/v4fs/stage_b.py --stage-a-dir runs_fs/stageA --out runs_fs/stageB --final runs_fs/FS_FINAL.json \\
         --v1-root <v1> --theta0-gpu 1
  python -B code/v4fs/stage_b.py --dry-run --stage-a-dir <8 synthetic dry run_fs dirs> --out <dir> --final <path>

Per call (idempotent: every step is skipped when its output already exists; a run directory is never deleted):
  1. **Decide** (v4fs.decide.decide_runs over the known run dirs; DECISION_<letter>.json, "A" first).
     - LOCK: write runs_fs/FS_FINAL.json (action, schema_id, dropped/restored groups, the full decision trace) and
       stop -- this call is now finished for good (a further call is a no-op: FS_FINAL.json already exists).
     - TRAIN schema_id S*: continue.
  2. **Register S*** (schemas.derive(CANDIDATE_BASE_ID, dropped_groups); contracts/derived_schemas/<S*>.json +
     REGISTRY.json, self-hash recomputed the v4fs.schemas way over EVERY registered schema, old and new, so an
     earlier round's derived schema is never dropped from the registry by a later round's write).
  3. **theta0**: registers S* for both families x {2026, 2027} on one physical GPU (server_theta0.sh's own pattern:
     GPU ownership + a memory guard fail-closed, CUDA_VISIBLE_DEVICES, CUBLAS_WORKSPACE_CONFIG=:4096:8), merged into
     THETA0_REGISTRY.json with the lock-guarded `v4fs.schemas.merge_theta0` (FS-02's extension: safe on the server).
  4. **Train**: if the 4 runs_fs/stageB/<S*>__<FAMILY>__<seed>/FS_SUMMARY.json are missing, this call STOPS here and
     reports the exact `fs_stageB.sh` launch commands (one detached queue per physical GPU, exactly like
     runs_g1/ops07_sync3/fs_stageA.sh) -- training is hours long and is launched by Ops, never by this script.
  5. Once every FS_SUMMARY.json exists, the next call folds the new runs in and **decides again**. TRAIN (the
     single fallback, v4fs.decide.search): repeat 2-4 for that schema. LOCK: write FS_FINAL.json.

No GPU is ever touched without the same fail-closed ownership + memory check run_fs.py itself uses; no run
directory is ever reused or deleted; `--dry-run` skips every GPU / theta0 step (decide + schema registration only,
for a fast local test with a small synthetic registry).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True
import ppsi_env  # noqa: E402,F401  (isolation first, same as v4fs.theta0 / v4fs.run_fs)

FAMILIES = ("GRU", "SASREC")
SEEDS = (2026, 2027)
GPU_SEED = {0: 2026, 1: 2027}                        # the Stage A convention (runs_g1/ops07_sync3/fs_stageA.sh)
STAGE_A_SCHEMAS = ("L0S_CLEAN_V2_SESSION", "L3S_NEAR_DUPLICATE_SESSION")
STAGE_A_DIR = V4 / "runs_fs" / "stageA"
STAGE_B_DIR = V4 / "runs_fs" / "stageB"
FINAL_PATH = V4 / "runs_fs" / "FS_FINAL.json"
FS_STAGE_B_SH = HERE / "fs_stageB.sh"
CUBLAS = ":4096:8"
GPU_MEM_FOREIGN_MIB = 256
MAX_ROUNDS = 4                                       # v4fs.decide.search reaches LOCK within 3 (belt and braces)
DECISION_LETTERS = "ABCDEFGH"
FINAL_FORMAT = "PPSI_V4_FS_FINAL_1"


class StageBError(RuntimeError):
    pass


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def say(log_fn: Callable, msg: str) -> None:
    log_fn(f"{utc()} [stage_b] {msg}")


# --------------------------------------------------------------------------------------------------------- run dirs
def run_dir(base: Path, schema_id: str, family: str, seed: int) -> Path:
    return Path(base) / f"{schema_id}__{family}__{seed}"


def stage_a_dirs(base: Path = STAGE_A_DIR) -> list:
    return [run_dir(base, sid, fam, seed) for sid in STAGE_A_SCHEMAS for fam in FAMILIES for seed in SEEDS]


def round_dirs(schema_id: str, base: Path = STAGE_B_DIR) -> list:
    return [run_dir(base, schema_id, fam, seed) for fam in FAMILIES for seed in SEEDS]


def missing_summaries(dirs: Sequence[Path]) -> list:
    return [d for d in dirs if not (Path(d) / "FS_SUMMARY.json").exists()]


# ------------------------------------------------------------------------------------------------------------ decide
def decide_from_dirs(dirs: Sequence[Path]) -> dict:
    from v4fs import decide as D
    return D.decide_runs(D.load_runs(dirs))


def write_decision(decision: dict, out_dir: Path, letter: str) -> Path:
    from v4train.checkpoint import write_json_atomic
    p = Path(out_dir) / f"DECISION_{letter}.json"
    if not p.exists():
        write_json_atomic(p, decision)
    return p


# --------------------------------------------------------------------------------------------------- schema register
def _existing_extra(registry_path: Path, out_dir: Path) -> dict:
    """Every already-registered schema that write_registry does not itself recompute (i.e. every derived S* / S**
    of an earlier Stage B round), read back from its own file so a later round's write_registry call keeps it."""
    from v4fs import schemas as S
    if not Path(registry_path).is_file():
        return {}
    reg = S.read_registry(registry_path)
    extra = {}
    for sid, ent in reg["schemas"].items():
        if sid in S.WRITTEN_SCHEMAS:
            continue
        extra[sid] = json.loads((Path(out_dir) / ent["path"]).read_text(encoding="utf-8"))
    return extra


def register_schema(schema_id: str, dropped_groups: Sequence[str], *, registry_path: Optional[Path] = None,
                    out_dir: Optional[Path] = None, candidate_base: Optional[str] = None,
                    log_fn: Callable = print) -> dict:
    """Idempotent: contracts/derived_schemas/<schema_id>.json + REGISTRY.json (v4fs.schemas.derive / write_registry,
    the same self-hash rule); a no-op when schema_id is already registered with the expected self-hash. Every
    earlier round's derived schema stays registered (`_existing_extra`): write_registry never drops one."""
    from v4fs import schemas as S
    registry_path = Path(registry_path) if registry_path else S.REGISTRY_PATH
    out_dir = Path(out_dir) if out_dir else S.DERIVED_DIR
    base_id = candidate_base or S.CANDIDATE_BASE_ID
    ds = S.derive(base_id, dropped_groups, registry_path=registry_path)
    if ds.schema_id != schema_id:
        raise StageBError(f"derive({base_id!r}, {sorted(dropped_groups)}) gives {ds.schema_id!r}, not {schema_id!r}")
    if registry_path.is_file():
        reg = S.read_registry(registry_path)
        ent = reg["schemas"].get(schema_id)
        if ent is not None and ent["self_sha256"] == ds.self_sha256:
            say(log_fn, f"register_schema: {schema_id} already registered (self {ds.self_sha256[:12]}...): skip")
            return reg
    extra = _existing_extra(registry_path, out_dir)
    extra[schema_id] = S.derived_obj(ds)
    say(log_fn, f"register_schema: writing {schema_id} (self {ds.self_sha256[:12]}...) into {registry_path}")
    reg = S.write_registry(out_dir=out_dir, extra=extra)
    verify = S.load_registered(schema_id, registry_path, base=ds.base)
    if verify.self_sha256 != ds.self_sha256:
        raise StageBError(f"{schema_id}: round-trip self-hash mismatch after write_registry")
    return reg


# ---------------------------------------------------------------------------------------------------------- theta0
def theta0_missing(schema_id: str, self_sha256: str, torch_build: str, *, seeds: Sequence[int] = SEEDS,
                   theta0_path: Optional[Path] = None) -> list:
    from v4fs import schemas as S
    theta0_path = Path(theta0_path) if theta0_path else S.THETA0_PATH
    out = []
    for fam in FAMILIES:
        for seed in seeds:
            if S.registered_theta0(schema_id, self_sha256, fam, seed, torch_build, theta0_path) is None:
                out.append((fam, seed))
    return out


def gpu_precheck(gpu: int, *, runner: Optional[Callable] = None) -> dict:
    """The server_theta0.sh / run_fs preflight GPU check: fail-closed ownership + a memory guard."""
    from v4sup.gpu import default_runner, gpu_ready
    runner = runner or default_runner
    ok, why = gpu_ready([int(gpu)], runner=runner)
    if not ok:
        raise StageBError(f"GPU {gpu} ownership check refused: {why}")
    from v4fs.run_fs import _gpu_rows
    rows = _gpu_rows(runner)
    if str(int(gpu)) not in rows:
        raise StageBError(f"GPU {gpu} is not enumerated by nvidia-smi ({sorted(rows)})")
    used = rows[str(int(gpu))]["memory_used_mib"]
    if used > GPU_MEM_FOREIGN_MIB:
        raise StageBError(f"GPU {gpu} has {used} MiB in use with no visible compute process: refused")
    return {"gpu_ready": why, "nvidia_smi": rows}


def register_theta0(schema_id: str, *, gpu: int, v1_root: Path, out_dir: Path, theta0_path: Optional[Path] = None,
                    schema_registry_path: Optional[Path] = None, seeds: Sequence[int] = SEEDS,
                    runner: Optional[Callable] = None, environ=None, log_fn: Callable = print) -> dict:
    """Idempotent: registers schema_id's theta0 for both families x `seeds` on the physical GPU `gpu` (the
    server_theta0.sh GPU check, CUDA_VISIBLE_DEVICES, CUBLAS_WORKSPACE_CONFIG=:4096:8), merged into theta0_path with
    the lock-guarded v4fs.schemas.merge_theta0 (FS-02 extension: safe to run on the server). `environ` defaults to
    the real process environment (run_fs.export_environment's own pattern); a test may inject a plain dict instead.
    NOTE: `v4fs.theta0.measure` itself always loads the schema from the DEFAULT registry (v4fs.schemas.REGISTRY_PATH)
    -- `schema_registry_path` here only gates the idempotency pre-check (theta0_missing); in production the two are
    always the same file (schema_id was just registered there by register_schema), so this never matters in
    practice, but a non-default `schema_registry_path` is for that pre-check only."""
    import torch
    from v4fs import schemas as S
    from v4fs import theta0 as T
    env = os.environ if environ is None else environ
    theta0_path = Path(theta0_path) if theta0_path else S.THETA0_PATH
    schema_registry_path = Path(schema_registry_path) if schema_registry_path else S.REGISTRY_PATH
    ds = S.load_registered(schema_id, schema_registry_path)
    missing = theta0_missing(schema_id, ds.self_sha256, torch.__version__, seeds=seeds, theta0_path=theta0_path)
    if not missing:
        say(log_fn, f"register_theta0: {schema_id} already fully registered for torch {torch.__version__}: skip")
        return {"skipped": True}
    say(log_fn, f"register_theta0: {schema_id} missing {missing}: checking GPU {gpu}")
    check = gpu_precheck(gpu, runner=runner)
    prev = env.get("CUDA_VISIBLE_DEVICES")
    if prev is not None and prev != str(int(gpu)):
        raise StageBError(f"inherited CUDA_VISIBLE_DEVICES={prev!r} differs from --theta0-gpu {gpu}")
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    prev_cublas = env.get("CUBLAS_WORKSPACE_CONFIG")
    if prev_cublas is not None and prev_cublas != CUBLAS:
        raise StageBError(f"inherited CUBLAS_WORKSPACE_CONFIG={prev_cublas!r} differs from {CUBLAS!r}")
    env["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise StageBError("expected exactly one visible CUDA device after CUDA_VISIBLE_DEVICES is set")
    res = T.measure([schema_id], list(seeds), Path(v1_root), device="cuda:0", log=log_fn)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = out_dir / f"THETA0_{schema_id}.json"
    S.write_file(receipt, json.dumps(dict(res, gpu_check=check), indent=1, sort_keys=True, allow_nan=False) + "\n")
    if not res["ok"]:
        raise StageBError(f"theta0 measurement for {schema_id} failed the L0/GPU guards: see {receipt}")
    S.merge_theta0(theta0_path, res["records"])
    say(log_fn, f"register_theta0: {schema_id} merged into {theta0_path}")
    return {"skipped": False, "receipt": str(receipt)}


# --------------------------------------------------------------------------------------------------------- training
def launch_commands(schema_id: str, *, out_dir: Path = STAGE_B_DIR, sh: Path = FS_STAGE_B_SH,
                    gpu_seed: Mapping[int, int] = None) -> list:
    gpu_seed = dict(gpu_seed or GPU_SEED)
    out_dir = Path(out_dir)
    cmds = []
    for gpu, seed in sorted(gpu_seed.items()):
        nohup_log = out_dir / "logs" / f"nohup_gpu{gpu}.log"
        cmds.append(f"mkdir -p {out_dir}/logs && nohup bash {sh} {gpu} {seed} {schema_id} "
                    f"> {nohup_log} 2>&1 & disown")
    return cmds


# ------------------------------------------------------------------------------------------------------------ drive
def drive(*, stage_a_dir: Path = STAGE_A_DIR, out_dir: Path = STAGE_B_DIR, final_path: Path = FINAL_PATH,
         gpu_seed: Optional[Mapping[int, int]] = None, theta0_gpu: int = 1, v1_root: Optional[Path] = None,
         schema_registry_path: Optional[Path] = None, theta0_registry_path: Optional[Path] = None,
         derived_out_dir: Optional[Path] = None, receipts_dir: Optional[Path] = None, dry_run: bool = False,
         runner: Optional[Callable] = None, log_fn: Callable = print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = Path(final_path)
    receipts_dir = Path(receipts_dir) if receipts_dir else out_dir / "theta0_receipts"
    if final_path.exists():
        say(log_fn, f"drive: {final_path} already written: nothing to do")
        return json.loads(final_path.read_text(encoding="utf-8"))
    known = list(stage_a_dirs(stage_a_dir))
    trace_rounds = []
    for i in range(MAX_ROUNDS):
        letter = DECISION_LETTERS[i]
        decision = decide_from_dirs(known)
        write_decision(decision, out_dir, letter)
        nxt = decision["next"]
        trace_rounds.append({"letter": letter, "next": nxt, "n_runs": len(known)})
        say(log_fn, f"drive: round {letter}: {nxt['action']} {nxt.get('schema_id')}")
        if nxt["action"] == "LOCK":
            final = {"format": FINAL_FORMAT, "action": "LOCK", "schema_id": nxt["schema_id"],
                     "dropped_groups": nxt.get("dropped_groups"), "restored_group": nxt.get("restored_group"),
                     "reason": nxt.get("reason"), "rounds": trace_rounds, "final_decision": decision, "utc": utc()}
            from v4train.checkpoint import write_json_atomic
            write_json_atomic(final_path, final)
            say(log_fn, f"drive: LOCKED {nxt['schema_id']}: wrote {final_path}")
            return final
        if nxt["action"] != "TRAIN":
            raise StageBError(f"unexpected next action {nxt['action']!r}")
        sid, dropped = nxt["schema_id"], nxt.get("dropped_groups", [])
        register_schema(sid, dropped, registry_path=schema_registry_path, out_dir=derived_out_dir, log_fn=log_fn)
        if not dry_run:
            if v1_root is None:
                raise StageBError("--v1-root is required once TRAIN is reached (theta0 registration)")
            register_theta0(sid, gpu=theta0_gpu, v1_root=v1_root, out_dir=receipts_dir,
                            theta0_path=theta0_registry_path, schema_registry_path=schema_registry_path,
                            runner=runner, log_fn=log_fn)
        dirs = round_dirs(sid, out_dir)
        missing = missing_summaries(dirs)
        if missing:
            cmds = launch_commands(sid, out_dir=out_dir, gpu_seed=gpu_seed)
            log_fn("drive: WAITING_FOR_OPS -- run the training queues, then re-run stage_b:")
            for c in cmds:
                say(log_fn, f"  {c}")
            return {"status": "WAITING_FOR_OPS_TRAINING", "schema_id": sid, "dropped_groups": dropped,
                    "missing_run_dirs": [str(d) for d in missing], "launch_commands": cmds, "rounds": trace_rounds}
        known += dirs
    raise StageBError(f"the FS search did not reach LOCK within {MAX_ROUNDS} rounds (trace: {trace_rounds})")


# --------------------------------------------------------------------------------------------------------------- cli
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage-a-dir", default=str(STAGE_A_DIR))
    ap.add_argument("--out", default=str(STAGE_B_DIR))
    ap.add_argument("--final", default=str(FINAL_PATH))
    ap.add_argument("--v1-root", default=None, help="required once a TRAIN step needs theta0 registration")
    ap.add_argument("--theta0-gpu", type=int, default=1)
    ap.add_argument("--schema-registry", default=None)
    ap.add_argument("--theta0-registry", default=None)
    ap.add_argument("--derived-out-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="decide + schema registration only; no GPU / theta0")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        res = drive(stage_a_dir=Path(args.stage_a_dir), out_dir=Path(args.out), final_path=Path(args.final),
                   theta0_gpu=args.theta0_gpu, v1_root=Path(args.v1_root) if args.v1_root else None,
                   schema_registry_path=Path(args.schema_registry) if args.schema_registry else None,
                   theta0_registry_path=Path(args.theta0_registry) if args.theta0_registry else None,
                   derived_out_dir=Path(args.derived_out_dir) if args.derived_out_dir else None,
                   dry_run=args.dry_run)
    except StageBError as e:
        print(f"[stage_b] REFUSED: {e}", file=sys.stderr)
        return 2
    if res.get("action") == "LOCK":
        print(json.dumps({k: res[k] for k in ("action", "schema_id", "dropped_groups", "restored_group")}, indent=1))
        return 0
    print(json.dumps(res, indent=1, default=str))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
