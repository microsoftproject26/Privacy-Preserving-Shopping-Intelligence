"""v4fs.run_fs — the FEATURE_SEARCH_SCREEN central run of one (family, derived schema, seed) (SCHEMA-01; DL 39;
binding: Fable R2 + R4, POPULATION_5_100_SCIENTIFIC_SYNTHESIS §D3).

  python -B code/v4fs/run_fs.py --family GRU --schema L3_NEAR_DUPLICATE --seed 2026 --gpu 1 --out <run_dir>
         [--fs-registry contracts/manifest_registry/REGISTRY_FS.json] [--manifest-dir ... data path overrides]
  python -B code/v4fs/run_fs.py --family GRU --schema L3_NEAR_DUPLICATE --seed 2026 --out <dir> --dry-run

Order (fail-closed; nothing heavy is imported before step 3):
  1. preflight (no torch): family GRU | SASREC; seed 2026 | 2027; the derived schema is in
     contracts/derived_schemas/REGISTRY.json (self-hash verified); the FS manifest registry (FS-01; self-hash verified)
     names the FEATURE_SEARCH_SCREEN train / eval manifests, and every TEST manifest id (M07-M09, "TEST" in the id)
     is refused before anything is opened; the output directory holds no earlier FS result (evidence is never
     overwritten); not --dry-run: an explicit physical GPU (--gpu >= 0), an inherited CUDA_VISIBLE_DEVICES equal to
     it, GPU ownership fail-closed (v4sup.gpu.gpu_ready: any compute process or any nvidia-smi failure refuses) and a
     memory guard (memory.used > 256 MiB with no visible process refuses); the disk guard;
  2. export CUDA_VISIBLE_DEVICES=<gpu>, CUBLAS_WORKSPACE_CONFIG=:4096:8 (determinism STRICT) and
     PYTHONDONTWRITEBYTECODE=1 before torch / v4train / ppsi_env are imported;
  3. import v4train; the process must see exactly one CUDA device and uses cuda:0 (--dry-run: CPU);
  4. providers: v4int StoreTrainProvider / StoreEvalProvider(manifest_id, history_policy="USER_HISTORY_L50",
     paths=...) with the Clean V2 DEFAULT transform state (the FS-fitted state is never used by the screen: there is
     no override); --dry-run: v4train's synthetic providers (full Clean V2 batches). The comparison key comes from
     v4int.providers.comparison_key_base plus the schema id / self-hash and the run label;
  5. the v4train CentralTrainer on the derived schema (v4models.build projects the full batch in the adapter): seed
     2026 | 2027, the executed peak 1e-3, effective batch 256, 6 EFE (the MAIN schedule), exposure "end",
     determinism STRICT, expected theta0 = the registered hash for (schema, family, seed, torch build)
     (contracts/derived_schemas/THETA0_REGISTRY.json; fail-closed when not registered), plus the TRAIN-01R2 fields
     when the RunConfig declares them (manifest_registry, data_version, role / run_label, schema registry);
  SCHEMA-01R1 (DECISION_LOG 47 / 48): the reference is L0S_CLEAN_V2_SESSION and the candidate base
     L3S_NEAR_DUPLICATE_SESSION (importance "auto" = L3S: 15 groups, G1 descriptive, 14 droppable); an S schema needs
     DATA-05's keys (session_offset_tokens, the G13 keys), checked on the first train and eval batch (check_session_offsets);
     --dry-run adds deterministic synthetic keys (SessionOffsetTrain / SessionOffsetEval) until DATA-05 emits them;
  6. the statistic is the CONTROLLED_BUDGET 6.0 endpoint client-macro MRR@20: its per-user statistics are rebuilt
     from the trainer's 6.0 ranks (evals/eval_6.0.ranks_int32.npy + the eval manifest) and must reproduce the trainer's
     value; they go to FS_USERSTATS.npz (key "endpoint") for v4fs.decide;
  7. retention: `latest`, `best` (and `best_extended`) are deleted as soon as the run is done; the `endpoint_6.0`
     weights are kept ONLY while the grouped permutation importance runs (L3: the 14 groups x 3 permutation seeds;
     --importance auto = L3 only), then deleted; every deletion is logged by the checkpoint store with the removed
     file's sha256 and every digest is in FS_SUMMARY.json; no weight file may remain (an importance failure keeps
     `endpoint_6.0` for --importance-only, recorded);
  8. every output row is labelled run_label FEATURE_SEARCH_SCREEN (row_status / comparison_status too,
     c_matched_eligible and downstream_seed_allowed False, selection_rule None; the trainer's rule is kept as
     fs_selection_rule); `write_rows` refuses any unlabelled row, and a production launch refuses a trainer that
     cannot label its own rows (the TRAIN-01R2 run label).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Optional

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

RUN_LABEL = "FEATURE_SEARCH_SCREEN"
ROLE = "FEATURE_SEARCH_SCREEN"
HISTORY_POLICY = "USER_HISTORY_L50"
FAMILIES = ("GRU", "SASREC")
SEEDS = (2026, 2027)
PEAK_LR = 1e-3
EFFECTIVE_BATCH = 256
END_EFE = 6.0
EXPOSURE_POINT = "end"
DETERMINISM_MODE = "STRICT"
CUBLAS = ":4096:8"
GPU_MEM_FOREIGN_MIB = 256
DISK_FLOOR_GIB = 8.0
IMPORTANCE_SCHEMAS = ("L3S_NEAR_DUPLICATE_SESSION",)   # SCHEMA-01R1: the candidate base L3S (DECISION_LOG 47)
REFERENCE_SCHEMA = "L0S_CLEAN_V2_SESSION"
SESSION_KEY = "session_offset_tokens"
SESSION_VOCAB = 9
INTENT_TOKEN_KEY = "session_mode_main_category_tokens"          # G13 (DECISION_LOG 48)
INTENT_FLOAT_KEYS = ("current_matches_session_mode", "current_matches_session_mode_available", "session_mode_share")
STATISTIC_MARK = 6.0
FS_REGISTRY = V4 / "contracts" / "manifest_registry" / "REGISTRY_FS.json"
SCHEMA_REGISTRY = V4 / "contracts" / "derived_schemas" / "REGISTRY.json"
THETA0_REGISTRY = V4 / "contracts" / "derived_schemas" / "THETA0_REGISTRY.json"
RESULT_FILE = "FS_RESULT.json"
SUMMARY_FILE = "FS_SUMMARY.json"
USERSTATS_FILE = "FS_USERSTATS.npz"
TEST_ID = re.compile(r"(^M0[789])|TEST", re.IGNORECASE)
DRY = {"N": 200, "K": 64, "B": 32, "n_eval": 90, "max_len": 8}
LABEL_FIELDS = {"run_label": RUN_LABEL, "row_status": RUN_LABEL, "comparison_status": RUN_LABEL,
                "c_matched_eligible": False, "downstream_seed_allowed": False}


class FSRefused(RuntimeError):
    pass


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _strict_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"),
                      parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))


def verify_self_hash(obj: dict, what: str) -> str:
    """The V4 canonical self-hash (v4data.common.canonical_json_bytes) under the object's own *self_sha256 key."""
    keys = [k for k in obj if k.endswith("self_sha256") and isinstance(obj[k], str) and len(obj[k]) == 64]
    if len(keys) != 1:
        raise FSRefused(f"{what}: expected exactly one top-level *self_sha256 field, found {keys}")
    body = {k: v for k, v in obj.items() if k != keys[0]}
    got = hashlib.sha256(_canon(body)).hexdigest()
    if got != obj[keys[0]]:
        raise FSRefused(f"{what}: self-hash mismatch (stated {obj[keys[0]]}, recomputed {got})")
    return got


def refuse_test(manifest_id: str) -> None:
    if TEST_ID.search(str(manifest_id or "")):
        raise FSRefused(f"manifest {manifest_id!r} refused: TEST manifests are never used by the feature search")


def fs_manifests(reg: dict) -> tuple:
    role = (reg.get("roles") or {}).get(ROLE)
    if not role or not role.get("train") or not role.get("eval"):
        raise FSRefused(f"the manifest registry has no {ROLE} role with train and eval ids")
    for m in (role["train"], role["eval"]):
        refuse_test(m)
    for m in (reg.get("manifests") or {}):
        refuse_test(m)
    return str(role["train"]), str(role["eval"])


def schema_entry(schema_id: str, path: Path = SCHEMA_REGISTRY) -> dict:
    reg = _strict_json(path)
    verify_self_hash(reg, str(path))
    ent = reg["schemas"].get(schema_id)
    if ent is None:
        raise FSRefused(f"schema {schema_id!r} is not registered in {path} ({sorted(reg['schemas'])})")
    return {"registry_sha256": _sha(path), "registry_self_sha256": reg["self_sha256"], **ent}


# ------------------------------------------------------------------------------------------------ preflight
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True, choices=FAMILIES)
    ap.add_argument("--schema", required=True, help="schema_id in contracts/derived_schemas/REGISTRY.json")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--out", required=True, help="the run directory (created; must not hold an FS result)")
    ap.add_argument("--gpu", type=int, default=None, help="PHYSICAL GPU index (required unless --dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="synthetic providers on CPU (tests; FEATURE_SEARCH_SCREEN "
                                                            "labels, test_overrides)")
    ap.add_argument("--fs-registry", default=str(FS_REGISTRY))
    ap.add_argument("--schema-registry", default=str(SCHEMA_REGISTRY))
    ap.add_argument("--theta0-registry", default=str(THETA0_REGISTRY))
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--manifest-dir", default=None)
    ap.add_argument("--lock-path", default=None)
    ap.add_argument("--manifest-hashes", default=None)
    ap.add_argument("--header-list", default=None)
    ap.add_argument("--importance", choices=("auto", "on", "off"), default="auto",
                    help="auto: the 14-group importance for L3 only (the binding D3 design)")
    ap.add_argument("--alignment", choices=("WHOLE_COLUMN", "LENGTH_MATCHED"), default="LENGTH_MATCHED",
                    help="LENGTH_MATCHED (binding: Fable D3 alignment ruling, DECISION_LOG 45); WHOLE_COLUMN is a diagnostic only")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--importance-only", action="store_true",
                    help="a finished run whose importance failed: load its kept endpoint_6.0, compute, delete")
    return ap.parse_args(argv)


def _gpu_rows(runner: Callable) -> dict:
    try:
        p = runner(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"])
    except Exception as e:
        raise FSRefused(f"GPU check fail-closed: nvidia-smi unavailable ({e})") from e
    if p.returncode != 0:
        raise FSRefused(f"GPU check fail-closed: nvidia-smi exit {p.returncode}")
    rows = {}
    for line in p.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 6:
            raise FSRefused(f"GPU check fail-closed: unparsable nvidia-smi line {line!r}")
        rows[parts[0]] = {"uuid": parts[1], "name": parts[2], "memory_used_mib": float(parts[3]),
                          "memory_total_mib": float(parts[4]), "utilization_pct": float(parts[5])}
    return rows


def importance_wanted(args) -> bool:
    return args.importance == "on" or (args.importance == "auto" and args.schema in IMPORTANCE_SCHEMAS)


def preflight(args: argparse.Namespace, *, runner: Optional[Callable] = None, environ=None,
              disk_usage_fn: Callable = shutil.disk_usage) -> dict:
    """Every refusal that needs no torch import. Returns the record written into FS_LAUNCH.json."""
    env = os.environ if environ is None else environ
    rec = {"utc": utc(), "argv": list(sys.argv), "host": platform.node(), "dry_run": bool(args.dry_run),
           "run_label": RUN_LABEL, "history_policy": HISTORY_POLICY, "importance": importance_wanted(args),
           "alignment": args.alignment}
    if args.family not in FAMILIES:
        raise FSRefused(f"family must be one of {FAMILIES}")
    if int(args.seed) not in SEEDS:
        raise FSRefused(f"the feature search uses seeds {SEEDS} (Fable R4), not {args.seed}")
    rec["schema"] = schema_entry(args.schema, Path(args.schema_registry))
    fsr = Path(args.fs_registry)
    if args.dry_run and not fsr.is_file():
        rec["fs_registry"] = {"path": None, "note": "dry run: synthetic manifests"}
        rec["manifests"] = {"train": "DRYRUN_FS_TRAIN", "eval": "DRYRUN_FS_VALIDATION"}
    else:
        if not fsr.is_file():
            raise FSRefused(f"FS manifest registry {fsr} does not exist (FS-01)")
        reg = _strict_json(fsr)
        rec["fs_registry"] = {"path": str(fsr), "sha256": _sha(fsr), "self_sha256": verify_self_hash(reg, str(fsr)),
                              "data_version": reg.get("data_version")}
        tr, ev = fs_manifests(reg)
        rec["manifests"] = {"train": tr, "eval": ev}
    out = Path(args.out)
    for f in (RESULT_FILE, SUMMARY_FILE) + (() if args.importance_only else ("RUN_START.json",)):
        if (out / f).exists():
            raise FSRefused(f"{out / f} exists: an FS run directory is never reused (evidence is never overwritten)")
    if args.importance_only and not (out / "RUN_SUMMARY.json").exists():
        raise FSRefused(f"--importance-only needs a finished run in {out} (RUN_SUMMARY.json)")
    if not args.dry_run:
        from v4sup.gpu import default_runner, gpu_ready              # no torch
        runner = runner or default_runner
        if args.gpu is None or int(args.gpu) < 0:
            raise FSRefused("an explicit physical GPU id (--gpu >= 0) is required")
        inherited = env.get("CUDA_VISIBLE_DEVICES")
        if inherited is not None and inherited != str(int(args.gpu)):
            raise FSRefused(f"inherited CUDA_VISIBLE_DEVICES={inherited!r} differs from --gpu {args.gpu}")
        ok, why = gpu_ready([int(args.gpu)], runner=runner)
        if not ok:
            raise FSRefused(f"GPU ownership check refused physical GPU {args.gpu}: {why}")
        rows = _gpu_rows(runner)
        if str(int(args.gpu)) not in rows:
            raise FSRefused(f"GPU check fail-closed: physical GPU {args.gpu} is not enumerated ({sorted(rows)})")
        used = rows[str(int(args.gpu))]["memory_used_mib"]
        if used > GPU_MEM_FOREIGN_MIB:
            raise FSRefused(f"GPU {args.gpu} has {used} MiB in use with no visible compute process: refused")
        rec["gpu_check"] = {"gpu_ready": why, "nvidia_smi": rows, "inherited_cuda_visible_devices": inherited}
    probe = out if out.exists() else next((p for p in out.parents if p.exists()), Path("."))
    free = disk_usage_fn(str(probe)).free / 2 ** 30
    rec["disk"] = {"path": str(probe), "free_gib": round(free, 3)}
    if free < DISK_FLOOR_GIB + (0.0 if args.dry_run else 2.0):
        raise FSRefused(f"disk guard: free {free:.2f} GiB below the floor ({DISK_FLOOR_GIB} GiB + planned peak)")
    return rec


def export_environment(gpu: Optional[int], dry_run: bool, environ=None) -> dict:
    env = os.environ if environ is None else environ
    prev = env.get("CUBLAS_WORKSPACE_CONFIG")
    if prev is not None and prev != CUBLAS:
        raise FSRefused(f"inherited CUBLAS_WORKSPACE_CONFIG={prev!r} differs from {CUBLAS!r} (STRICT)")
    env["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if dry_run:
        if not env.get("CUDA_VISIBLE_DEVICES"):
            env["CUDA_VISIBLE_DEVICES"] = "-1"                 # CPU dry run: no GPU visible, explicit value
    else:
        env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    return {"CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"), "CUBLAS_WORKSPACE_CONFIG": CUBLAS,
            "PYTHONDONTWRITEBYTECODE": "1"}


# ------------------------------------------------------------------------------------------------ labels
def label_row(row: dict) -> dict:
    r = dict(row)
    r["fs_selection_rule"] = r.get("selection_rule")
    r["selection_rule"] = None
    r.update(LABEL_FIELDS)
    return r


def check_labelled(rows) -> None:
    bad = [i for i, r in enumerate(rows) if not isinstance(r, dict) or r.get("run_label") != RUN_LABEL]
    if bad:
        raise FSRefused(f"refusing to write {len(bad)} unlabelled row(s) (run_label must be {RUN_LABEL}); first {bad[:5]}")
    bad = [i for i, r in enumerate(rows) if r.get("selection_rule") is not None or r.get("c_matched_eligible")
           or r.get("downstream_seed_allowed")]
    if bad:
        raise FSRefused(f"rows {bad[:5]} carry a selection rule or eligibility flag: screening rows are never "
                        "selection evidence")


def write_rows(path: Path, payload: dict, rows_keys=("rows",)) -> str:
    """Strict JSON, atomic; every row list under `rows_keys` must be labelled FEATURE_SEARCH_SCREEN."""
    from v4train.checkpoint import write_json_atomic
    if payload.get("run_label") != RUN_LABEL:
        raise FSRefused(f"refusing to write {path}: the payload is not labelled {RUN_LABEL}")
    for k in rows_keys:
        check_labelled(payload.get(k) or [])
    return write_json_atomic(Path(path), payload)


# ------------------------------------------------------------------------------------------------ trainer interface
def runconfig_fields(RunConfig) -> set:
    return {f.name for f in dataclasses.fields(RunConfig)}


def trainer_label_support(RunConfig) -> dict:
    f = runconfig_fields(RunConfig)
    return {"manifest_registry": "manifest_registry" in f, "data_version": "data_version" in f,
            "role": "role" in f, "run_label_property": isinstance(getattr(RunConfig, "run_label", None), property),
            "schema_registry": "schema_registry" in f,
            "labels_rows": "role" in f and isinstance(getattr(RunConfig, "run_label", None), property)}


def make_config(RunConfig, *, run_id, family, seed, cuda_visible, key, expected_init, dry_run: bool,
                manifest_registry: Optional[str], data_version: Optional[str], schema_id: str,
                schema_registry: str, log_every: int):
    """The FS RunConfig; the TRAIN-01R2 fields are passed when the RunConfig declares them (planned interface)."""
    f = runconfig_fields(RunConfig)
    kw = dict(run_id=run_id, family=family, seed=int(seed), exposure_point=EXPOSURE_POINT,
              cuda_visible_devices=cuda_visible, determinism_mode=DETERMINISM_MODE, phase="MAIN", population="COHORT",
              peak_lr=PEAK_LR, effective_batch=DRY["B"] if dry_run else EFFECTIVE_BATCH, log_every=int(log_every),
              key=dict(key), expected_init_sha256=expected_init, test_overrides=bool(dry_run))
    extra = {"manifest_registry": manifest_registry, "data_version": data_version, "role": ROLE,
             "schema_registry": schema_registry}             # TRAIN-01R2: the run label is derived from the role
    for k, v in extra.items():
        if k in f and v is not None:
            kw[k] = v
    return RunConfig(**{k: v for k, v in kw.items() if k in f})


def _dry_registry(run_dir: Path, train_id: str, eval_id: str, tp, ep) -> Path:
    """A synthetic FS-01-shaped manifest registry for the dry run (TRAIN-01R2 reads it when it declares the field)."""
    obj = {"data_version": "DRYRUN_FS_SYNTHETIC", "roles": {ROLE: {"train": train_id, "eval": eval_id}},
           "manifests": {train_id: {"path": None, "n_decisions": tp.n_decisions, "fingerprint": tp.fingerprint},
                         eval_id: {"path": None, "n_decisions": ep.manifest.n,
                                   "n_eval_rows": int(ep.manifest.eval_mask.sum()), "fingerprint": ep.fingerprint}}}
    obj["self_sha256"] = hashlib.sha256(_canon(obj)).hexdigest()
    p = run_dir / "DRYRUN_REGISTRY_FS.json"
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return p


# ------------------------------------------------------------------------------------------------ session offsets
def check_session_offsets(batch, what: str, ep=None, schema=None) -> None:
    """Refuse a provider without DATA-05's keys when the schema keeps them: `session_offset_tokens` (int64 [B, L], 0 on
    PAD, 1..8 on valid positions, 1 at the current event) and the G13 keys (session_mode_main_category_tokens int64
    [B, L], 0 on PAD, 1..vocab-1 on valid positions; the three [B] floats, the mask in {0, 1})."""
    import torch
    from v4models.derived import EXTENSION_CHANNELS
    if batch is None:
        it = iter(ep.batches(int(getattr(ep, "eval_batch", 1024))))
        try:
            batch = next(it)
        finally:
            close = getattr(it, "close", None)
            if close is not None:
                close()
    kept = set(schema.channel_names) if schema is not None else {SESSION_KEY}
    am = torch.as_tensor(batch["attention_mask"]).bool()
    if INTENT_TOKEN_KEY in kept:
        v = int(EXTENSION_CHANNELS[INTENT_TOKEN_KEY]["vocab"])
        m = batch.get(INTENT_TOKEN_KEY)
        if m is None or m.dtype != torch.int64 or m.shape != am.shape or bool((m[~am] != 0).any()) or \
                bool(((m[am] < 1) | (m[am] >= v)).any()):
            raise FSRefused(f"the {what} provider's {INTENT_TOKEN_KEY} is missing or breaks the contract (int64 [B, L]; "
                            f"0 on PAD; 1..{v - 1} on valid positions)")
        for k in INTENT_FLOAT_KEYS:
            x = batch.get(k)
            if x is None or not torch.is_floating_point(x) or int(x.numel()) != int(am.shape[0]) or \
                    not bool(torch.isfinite(x).all()):
                raise FSRefused(f"the {what} provider's {k} is missing or not a finite float [B]")
        mk = batch["current_matches_session_mode_available"]
        if not bool(((mk == 0) | (mk == 1)).all()):
            raise FSRefused(f"the {what} provider's current_matches_session_mode_available is not a 0/1 mask")
    if SESSION_KEY not in kept:
        return
    t = batch.get(SESSION_KEY)
    if t is None:
        raise FSRefused(f"the {what} provider emits no {SESSION_KEY!r} (DATA-05), but the schema keeps it")
    last = torch.as_tensor(batch["lengths"]).long() - 1
    cur = t.gather(1, last.view(-1, 1)).squeeze(1)
    if t.dtype != torch.int64 or t.shape != am.shape or bool((t[~am] != 0).any()) or \
            bool(((t[am] < 1) | (t[am] >= SESSION_VOCAB)).any()) or bool((cur != 1).any()):
        raise FSRefused(f"the {what} provider's {SESSION_KEY} breaks the contract (int64 [B, L]; 0 on PAD; 1..8 on "
                        "valid positions; 1 at the current event)")


def _synthetic_offsets(batch: dict) -> dict:
    """Deterministic synthetic DATA-05 keys for the dry run (session_offset_tokens and the G13 keys; a pure function
    of decision_id and lengths)."""
    import torch
    from v4models.synthetic import synthetic_session_intent, synthetic_session_offsets
    b = dict(batch)
    seg = (torch.as_tensor(b["decision_id"]).long() % 5) + 1
    lengths = torch.as_tensor(b["lengths"]).long()
    b[SESSION_KEY] = synthetic_session_offsets(lengths, int(b["attention_mask"].shape[1]), seg=seg)
    b.update(synthetic_session_intent(b[SESSION_KEY], lengths))
    return b


class SessionOffsetTrain:
    """Dry run only: a BatchProvider that adds a synthetic session_offset_tokens key (DATA-05 not yet emitting it)."""
    split = "TRAIN"

    def __init__(self, inner):
        self.inner = inner
        self.manifest_id = inner.manifest_id
        self.fingerprint = hashlib.sha256((inner.fingerprint + "|SYNTHETIC_SESSION_OFFSET_1").encode()).hexdigest()

    @property
    def n_decisions(self) -> int:
        return self.inner.n_decisions

    def decision_ids(self):
        return self.inner.decision_ids()

    def batch(self, positions):
        return _synthetic_offsets(self.inner.batch(positions))


class SessionOffsetEval:
    """Dry run only: an EvalProvider that adds a synthetic session_offset_tokens key."""

    def __init__(self, inner):
        self.inner = inner
        self.manifest = inner.manifest
        self.fingerprint = hashlib.sha256((inner.fingerprint + "|SYNTHETIC_SESSION_OFFSET_1").encode()).hexdigest()

    def batches(self, batch_size: int):
        return (_synthetic_offsets(b) for b in self.inner.batches(batch_size))


# ------------------------------------------------------------------------------------------------ run
def run(args: argparse.Namespace, pre: dict, env_rec: dict, *, say: Callable = print,
        disk_usage_fn: Callable = shutil.disk_usage) -> dict:
    import v4train                                           # captures CUDA_VISIBLE_DEVICES, then ppsi_env
    import ppsi_env
    snap = ppsi_env.verify_snapshot()
    ppsi_env.assert_isolated()
    import numpy as np
    import torch
    from fl.numerics import state_digest
    from v4eval.bootstrap import point
    from v4fs import schemas as S
    from v4fs.importance import grouped_permutation_importance, save_stats, stats_from_ranks
    from v4train.checkpoint import read_json, sha256_file, write_json_atomic
    from v4train.recipe import RunConfig
    from v4train.trainer import CentralTrainer, trainer_code_sha256

    support = trainer_label_support(RunConfig)
    if not args.dry_run and not support["labels_rows"]:
        raise FSRefused("the installed v4train RunConfig cannot carry a run label / role (TRAIN-01R2 R3): a "
                        "production FS run would write unlabelled trainer rows; refused")
    if not args.dry_run:
        if v4train.LAUNCH_CUDA_VISIBLE_DEVICES != str(int(args.gpu)):
            raise FSRefused(f"v4train captured CUDA_VISIBLE_DEVICES={v4train.LAUNCH_CUDA_VISIBLE_DEVICES!r}")
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise FSRefused("expected exactly one visible CUDA device")
        dev = torch.device("cuda:0")
        torch.cuda.set_device(dev)
    else:
        dev = torch.device("cpu")
    torch.set_num_threads(int(args.n_threads))
    schema = S.load_registered(args.schema, Path(args.schema_registry))
    do_imp = importance_wanted(args)
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_id, eval_id = pre["manifests"]["train"], pre["manifests"]["eval"]
    run_id = args.run_id or f"FS__{args.schema}__{args.family}_s{args.seed}"
    t0 = time.time()
    if args.dry_run:
        from v4train.provider import SyntheticEvalProvider, SyntheticTrainProvider, expected_manifests
        if not support["manifest_registry"]:              # the pre-R2 trainer enforces its historical ids
            train_id, eval_id = expected_manifests("COHORT", "MAIN")
        K = DRY["K"]
        tp = SyntheticTrainProvider(DRY["N"], K, schema.base.widths, seed=3, manifest_id=train_id,
                                    max_len=DRY["max_len"])
        ep = SyntheticEvalProvider(DRY["n_eval"], K, schema.base.widths, seed=4, manifest_id=eval_id,
                                   max_len=DRY["max_len"])
        tp, ep = SessionOffsetTrain(tp), SessionOffsetEval(ep)   # until DATA-05's key exists (SCHEMA-01R1)
        key = {"data_protocol_hash": "DRYRUN-DATA", "feature_contract_hash": "DRYRUN-FEATURES", "split": "VALIDATION",
               "label_policy_hash": "DRYRUN-LABELS", "population_hash": "DRYRUN-POPULATION",
               "candidate_order_hash": "SYNTHETIC-CANDIDATES"}
        manifest_registry = str(_dry_registry(run_dir, train_id, eval_id, tp, ep))
        data_version = "DRYRUN_FS_SYNTHETIC"
        build_kw = {"K": K}
        expected = None
        paths_rec = None
    else:
        from dataclasses import replace
        from v4data.common import dataset_root
        from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
        from v4int.verify import DataPaths
        paths = DataPaths.default()                        # the Clean V2 transform state: never overridden here
        over = {k: getattr(args, a) for k, a in (("manifest_dir", "manifest_dir"), ("lock_path", "lock_path"),
                                                 ("manifest_hashes_path", "manifest_hashes"),
                                                 ("header_list_path", "header_list"))
                if getattr(args, a, None) is not None}
        paths = replace(paths, **{k: Path(v) for k, v in over.items()}) if over else paths
        if Path(paths.transform_state_path) != DataPaths.default().transform_state_path:
            raise FSRefused("the screen uses the Clean V2 default transform state only")
        paths_rec = paths.as_json()
        # FS manifests are not in DATA_PROTOCOL_LOCK: verify them by the manifest registry (DATA-05 path; Ops fix)
        reg_path = Path(pre["fs_registry"]["path"])
        tp = StoreTrainProvider(train_id, history_policy=HISTORY_POLICY, paths=paths, log=say,
                                manifest_registry=reg_path, registry_root=V4)
        ep = StoreEvalProvider(eval_id, history_policy=HISTORY_POLICY, paths=paths, log=say,
                               manifest_registry=reg_path, registry_root=V4)
        key = comparison_key_base(paths, tp, ep)
        manifest_registry = pre["fs_registry"]["path"]
        data_version = pre["fs_registry"].get("data_version")
        build_kw = {"v1_root": dataset_root()}
        reg = S.registered_theta0(schema.schema_id, schema.self_sha256, args.family, args.seed, torch.__version__,
                                  Path(args.theta0_registry))
        if reg is None:
            raise FSRefused(f"no registered theta0 for {schema.schema_id}/{args.family}/seed {args.seed} on torch "
                            f"{torch.__version__} ({args.theta0_registry}): register it first (v4fs.theta0)")
        expected = reg["init_sha256"]
    for m in (tp.manifest_id, ep.manifest.manifest_id):
        refuse_test(m)
    if SESSION_KEY in schema.channel_names or INTENT_TOKEN_KEY in schema.channel_names:   # SCHEMA-01R1: DATA-05 keys
        check_session_offsets(tp.batch(np.arange(min(8, tp.n_decisions))), "train", schema=schema)
        check_session_offsets(None, "eval", ep=ep, schema=schema)
    key = dict(key, schema_id=schema.schema_id, schema_self_sha256=schema.self_sha256, history_policy=HISTORY_POLICY)
    t_prov = time.time() - t0
    cfg = make_config(RunConfig, run_id=run_id, family=args.family, seed=args.seed,
                      cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), key=key, expected_init=expected,
                      dry_run=args.dry_run, manifest_registry=manifest_registry, data_version=data_version,
                      schema_id=schema.schema_id, schema_registry=str(args.schema_registry), log_every=args.log_every)
    tr = CentralTrainer(cfg, schema, tp, ep, run_dir, device=str(dev), disk_usage_fn=disk_usage_fn, **build_kw)
    if tr.built.schema_id != schema.schema_id or tr.built.schema_self_sha256 != schema.self_sha256:
        raise FSRefused("the trainer's model was not built from the requested derived schema")
    want_label = "SMOKE_NOT_EVIDENCE" if args.dry_run else RUN_LABEL
    if support["labels_rows"] and cfg.run_label != want_label:
        raise FSRefused(f"the trainer would label this run {cfg.run_label!r}, not {want_label!r}")
    launch = {"task": "SCHEMA-01 run_fs", "run_label": RUN_LABEL, "run_id": run_id, "preflight": pre,
              "environment": env_rec, "history_policy": HISTORY_POLICY, "schema": schema.record(),
              "data_paths": paths_rec, "importance": do_imp,
              "manifests": {"train": tp.manifest_id, "eval": ep.manifest.manifest_id,
                            "train_fingerprint": tp.fingerprint, "eval_fingerprint": ep.fingerprint},
              "trainer_support": support, "config_sha256": tr.config_sha256, "config": cfg.as_json(),
              "init_sha256": tr.built.init_sha256, "expected_init_sha256": expected,
              "shared_numel": tr.built.manifest.shared_numel, "trainer_code_sha256": trainer_code_sha256(),
              "torch": torch.__version__, "numpy": np.__version__, "python": sys.version.split()[0],
              "device": str(dev), "snapshot_files": len(snap["files"]), "providers_build_s": round(t_prov, 2),
              "utc": utc()}
    write_json_atomic(run_dir / ("FS_LAUNCH_IMPORTANCE_ONLY.json" if args.importance_only else "FS_LAUNCH.json"),
                      launch)
    say(f"[fs] {run_id}: theta0 {tr.built.init_sha256[:12]} numel {launch['shared_numel']}; "
        f"{'importance only' if args.importance_only else 'training'}")
    t1 = time.time()
    retention = []
    imp = None
    imp_done = False
    ep_rec = None
    loaded = None

    def drop(kinds, why, stage):
        for kind in kinds:
            try:
                d = tr.ckpt.delete(kind, why)
            except Exception as e:                           # a foreign / mismatching file: never removed, recorded
                retention.append({"kind": kind, "error": f"{type(e).__name__}: {e}", "utc": utc(), "stage": stage})
                continue
            if d:
                retention.append({"kind": kind, "sha256": d, "utc": utc(), "stage": stage})

    try:
        if args.importance_only:
            summary = read_json(run_dir / "RUN_SUMMARY.json")
            if summary.get("status") != "DONE_MAIN" or summary.get("config_sha256") != tr.config_sha256:
                raise FSRefused("--importance-only: the run is not a finished main run of this exact config")
        else:
            summary = tr.run()
        train_s = time.time() - t1
        drop(("latest", "best", "best_extended"), f"{RUN_LABEL}: only endpoint_6.0 is kept, until the importance "
             "is computed", "after_training")
        cb = summary["controlled_budget"]
        if abs(float(cb["mark"]) - STATISTIC_MARK) > 1e-9:
            raise FSRefused(f"the CONTROLLED_BUDGET mark is {cb['mark']}, not {STATISTIC_MARK}")
        ranks = np.load(run_dir / "evals" / f"eval_{STATISTIC_MARK:.1f}.ranks_int32.npy").astype(np.int64)
        end_stats = stats_from_ranks(ranks, ep.manifest)
        end_v = point(end_stats, "macro_e2e")
        if abs(end_v - float(cb["value"])) > 1e-12:
            raise FSRefused(f"the 6.0 ranks give {end_v}, the trainer's CONTROLLED_BUDGET value is {cb['value']}")
        stats = {"endpoint": end_stats}
        if do_imp:
            ep_rec = dict(tr.ckpt.index["endpoint_6.0"])
            obj = tr.ckpt.load("endpoint_6.0")               # sha256 vs CHECKPOINTS.json + validation
            tr.module.load_state_dict(obj["model"], strict=True)
            loaded = state_digest(tr.adapter.broadcast_state())
            if loaded != obj["meta"].get("state_digest"):
                raise FSRefused("the loaded endpoint_6.0 state digest differs from the checkpoint's")
            imp = grouped_permutation_importance(tr.adapter, ep, schema, device=dev, model_sha256=loaded,
                                                 alignment=args.alignment, run_id=run_id, log=say)
            if imp.baseline["value"] != float(cb["value"]):
                raise FSRefused(f"importance baseline {imp.baseline['value']} != the trainer's CONTROLLED_BUDGET "
                                f"value {cb['value']} (same weights, same frozen evaluator)")
            stats.update(imp.stats)
        imp_done = True
    finally:
        # endpoint_6.0 is kept only when the importance failed (so --importance-only can finish it)
        drop(("latest", "best", "best_extended") + (("endpoint_6.0",) if imp_done else ()),
             f"{RUN_LABEL}: weights are deleted once the importance is computed", "final")
    left = tr.ckpt.weight_files()
    if left:
        raise FSRefused(f"weight files remain after the FS run: {left}")
    us_sha = save_stats(run_dir / USERSTATS_FILE, stats, {"run_label": RUN_LABEL, "run_id": run_id,
                                                          "schema_id": schema.schema_id, "family": args.family,
                                                          "seed": int(args.seed), "statistic": "CONTROLLED_BUDGET"})
    rows = [label_row(r) for r in read_json(run_dir / "RESULT_ROWS.json")["rows"]]
    trainer_rows_labelled = all(r.get("run_label") == RUN_LABEL
                                for r in read_json(run_dir / "RESULT_ROWS.json")["rows"])
    extra = dict(LABEL_FIELDS, selection_rule=None, family=args.family, seed=int(args.seed))
    imp_rows = [dict(r, **extra, kind="GROUP_PERMUTATION_IMPORTANCE") for r in (imp.rows if imp else [])]
    base_rows = [dict(imp.baseline, **extra, kind="IMPORTANCE_BASELINE")] if imp else []
    payload = {"run_label": RUN_LABEL, "run_id": run_id, "family": args.family, "seed": int(args.seed),
               "schema_id": schema.schema_id, "schema_self_sha256": schema.self_sha256,
               "history_policy": HISTORY_POLICY, "statistic": "CONTROLLED_BUDGET 6.0 client-macro MRR@20 (E2E)",
               "rows": rows, "importance_baseline": base_rows, "importance": imp_rows,
               "importance_meta": imp.meta if imp else None,
               "note": "FEATURE_SEARCH_SCREEN rows: screening evidence only; refused by every selection path"}
    res_sha = write_rows(run_dir / RESULT_FILE, payload, rows_keys=("rows", "importance_baseline", "importance"))
    fs_summary = {"run_label": RUN_LABEL, "run_id": run_id, "family": args.family, "seed": int(args.seed),
                  "schema": schema.record(), "manifests": launch["manifests"], "history_policy": HISTORY_POLICY,
                  "config_sha256": tr.config_sha256, "init_sha256": tr.built.init_sha256,
                  "shared_numel": launch["shared_numel"], "statistic": "CONTROLLED_BUDGET",
                  "controlled_budget": summary.get("controlled_budget"),
                  "practical_best": summary.get("practical_best"), "still_improving": summary.get("still_improving"),
                  "updates": summary["updates"], "efe": summary["efe"], "train_wall_s": round(train_s, 1),
                  "endpoint_stats_reproduce_trainer_value": True,
                  "importance": None if imp is None else
                  [{k: r[k] for k in ("group", "importance", "importance_ci95", "within_noise",
                                      "relative_importance")} for r in imp.rows],
                  "importance_baseline_equals_controlled_budget": None if imp is None else True,
                  "endpoint_checkpoint": None if ep_rec is None else
                  {"sha256": ep_rec["sha256"], "bytes": ep_rec["bytes"],
                   "state_digest": ep_rec["meta"].get("state_digest"), "loaded_state_digest": loaded},
                  "retention": retention, "weight_files_left": left,
                  "retention_log_sha256": sha256_file(run_dir / "RETENTION_LOG.jsonl")
                  if (run_dir / "RETENTION_LOG.jsonl").exists() else None,
                  "trainer_rows_labelled_by_trainer": trainer_rows_labelled,
                  "result_file": RESULT_FILE, "result_sha256": res_sha,
                  "userstats_file": USERSTATS_FILE, "userstats_sha256": us_sha, "utc_end": utc()}
    write_json_atomic(run_dir / SUMMARY_FILE, fs_summary)
    say(f"[fs] {run_id}: done; CONTROLLED_BUDGET {cb['value']:.6f}; importance "
        f"{'%d groups' % len(imp.rows) if imp else 'off'}; weights left {left}")
    return fs_summary


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        pre = preflight(args)
        env_rec = export_environment(args.gpu, args.dry_run)
        run(args, pre, env_rec)
    except FSRefused as e:
        print(f"[fs] REFUSED: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
