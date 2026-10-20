"""v4int.launch_central — the production entry point of one central run (INT-01; SYS-03's supervisor registers it).

  python -B code/v4int/launch_central.py --family GRU --seed 2026 --phase CALIBRATION --peak-lr 1e-3 --gpu 0 \
         --run-dir <dir> [--smoke --smoke-updates 600 --checkpoint-every 300 [--smoke-interrupt-at 300]] [--resume]

Order (fail-closed; nothing heavy is imported before step 3):
  1. preflight: population COHORT only (FULL_TRAIN refuses until M_FULL_TRAIN / M_FULL_VALIDATION_WARM exist); an
     explicit physical GPU id (argparse-required, >= 0); phase CALIBRATION or MAIN; a non-smoke launch refuses unless
     company/gates/G1_APPROVED.json exists (strict JSON; its sha256 is recorded) — this launcher never writes it; a
     non-smoke fresh MAIN run needs --calibration-record; an inherited CUDA_VISIBLE_DEVICES that differs from --gpu
     refuses; GPU ownership (v4sup.gpu.gpu_ready: any compute process on that GPU, or any nvidia-smi failure, means
     refuse) plus a memory guard (memory.used above GPU_MEM_FOREIGN_MIB on that GPU with no visible process — a
     process of another container — also refuses);
  2. export CUDA_VISIBLE_DEVICES=<physical id>, CUBLAS_WORKSPACE_CONFIG (read from PREG1_LOCK_VALUES T3 flags) and
     PYTHONDONTWRITEBYTECODE=1, BEFORE torch or the snapshot is imported;
  3. import v4train (it captures the launch CUDA_VISIBLE_DEVICES, then imports ppsi_env first), verify the snapshot and
     the isolation, import torch, use cuda:0; the process must see exactly one CUDA device;
  4. build the providers (v4int.providers, population-agnostic: --train-manifest / --eval-manifest, defaulting to
     v4train.provider.expected_manifests(population, phase) — the ids the locked trainer enforces; the data paths
     default to the locked data_v2 tree and can be overridden for a new data version), the RunConfig (exposure_point
     "end", determinism_mode, cuda_visible_devices = the physical id, the comparison key from the lock and manifests)
     and the CentralTrainer; write LAUNCH_<k>.json; run / resume / continue.
--smoke (SMOKE_NOT_EVIDENCE): test_overrides=True; run id prefixed SMOKE_NOT_EVIDENCE__; at most 600 updates, and
the cap must end before the first 0.5-EFE mark (so the trainer writes no mark row, no RESULT_ROWS, no RUN_SUMMARY);
optional --smoke-interrupt-at k (a multiple of --checkpoint-every) stops the leg after the `latest` save at update k
(the resume test's first leg; its `latest` is kept for the resume leg of the same smoke job, which deletes it). At the
cap: the state digests (model, AdamW state, RNG, train state) are recorded, every weight file is deleted right away
(logged with its sha256; Operations disk rule), then ONE evaluation on the phase's evaluation manifest runs through the
trainer's own frozen evaluation path from the in-memory model, its rows are relabelled SMOKE_NOT_EVIDENCE (v4int.smoke)
into SMOKE_EVAL.json, and SMOKE_SUMMARY.json is written (free disk recorded at start, stop, deletion and end). A leg
that fails deletes its weights before re-raising. A launch refuses when free disk - the trainer's planned peak < 8 GiB.
No smoke row is valid for any selection.
--history-policy (DATA-05; REQUIRED, no default; v4hist.policy): USER_HISTORY_L50 (locked, DECISION_LOG 36) or
SESSION_L50 (the v1 diagnostic). The same policy is passed to both providers, recorded in LAUNCH_<k>.json and bound
into the comparison key (v4int.providers.comparison_key_base).
TRAIN-02 (P2 search, PREG1_AMENDMENT_02R1): `--role TUNING --hp-id <P2_<FAMILY>_Tnn>` (phase CALIBRATION, so the
REGISTRY_V2 trainer view's CALIBRATION role gives CALIB_TRAIN / CALIB_VALIDATION). After the environment export the
launcher reads the trial from the HP registry (v4train.hp_registry, hash-checked; --hp-registry overrides the path) and
passes its peak LR (an explicit --peak-lr must equal it), dropout, schedule and model variant to the RunConfig, whose
validate() accepts them only as that trial; theta0_init_sha256 comes from THETA0_VARIANTS.json (v4train.theta0_variants,
self-hash verified; keyed by the S* schema, the running torch build, the variant and the seed) and is required for a
non-smoke TUNING launch. The run id defaults to <hp_id>_s<seed>. --hp-id without --role TUNING, --role TUNING without
--hp-id, a TUNING launch with --calibration-record / --continue-run / another phase are refused in the preflight.
TRAIN-02R1: the HP registry default is HP_REGISTRY_P2R2.json (02R2; the superseded P2 file is refused); a TUNING launch
also refuses --train-manifest / --eval-manifest (F2: nothing but the CALIBRATION role's manifests is ever loaded);
F6 (P3): `--phase MAIN --recipe-id 02B_<FAMILY>` takes the tuned recipe of TUNED_RECIPES_02B.json (peak LR, dropout,
schedule, model variant; theta0 from THETA0_VARIANTS.json, required non-smoke) and needs no --calibration-record.
INT-03: (a) `--allow-shared-gpu` (default off; recorded in LAUNCH_<k>.json under preflight.gpu_check.shared_gpu):
the GPU-ownership refusal and the foreign-memory guard are recorded instead of refusing (nvidia-smi failure / an
unenumerated GPU still refuse). PREG1_AMENDMENT_06 §4: central MAIN-scale training keeps one job per card, so the flag
is refused for a non-smoke --phase MAIN launch unless `--coschedule-06g` is also given (PREG1_AMENDMENT_06G §5 / 06K §1,
owner "100 % utilisation": STRICT numerics do not depend on co-scheduling; recorded as preflight.coschedule_06g;
--coschedule-06g without --allow-shared-gpu is refused). (b) v4int.small_recipe.install() runs right after the environment
export: a SMALL-format --hp-registry (the SASREC_D64_B2 mini-ladder) or --tuned-recipes (the SMALL recipe, C_SMALL) is
verified by v4int.small_recipe; every other file goes to the locked loaders unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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

G1_GATE = V4 / "company" / "gates" / "G1_APPROVED.json"
PREG1_LOCK = V4 / "contracts" / "PREG1_LOCK_VALUES.json"
GPU_MEM_FOREIGN_MIB = 256
# TRAIN-01R1: RunConfig has a REQUIRED `determinism_mode` in {"STRICT", "WARN_ONLY"}; the proposed lock value is
# STRICT (PREG1_AMENDMENT_01 item 4, Fable to confirm at G1). This one constant is the switch; it is passed whenever the
# RunConfig in use declares the field (a pre-R1 trainer runs warn-only), and the mode actually in effect inside the
# training loop is recorded either way.
DETERMINISM_MODE = "STRICT"
FAMILIES = ("GRU", "SASREC")
PHASES = ("CALIBRATION", "MAIN")


class LaunchRefused(RuntimeError):
    pass


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PPSI V4 central run launcher (INT-01)")
    ap.add_argument("--family", required=True, type=str.upper, choices=FAMILIES)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--phase", required=True, type=str.upper, choices=PHASES)
    ap.add_argument("--population", default="COHORT", type=str.upper, choices=("COHORT", "FULL_TRAIN"))
    ap.add_argument("--peak-lr", default=None, type=float,
                    help="required unless --role TUNING (then taken from the HP trial; if given it must equal it)")
    ap.add_argument("--role", default=None, type=str.upper, choices=("TUNING",),
                    help="TRAIN-02: TUNING (the P2 search; phase CALIBRATION, needs --hp-id)")
    ap.add_argument("--hp-id", default=None, help="TRAIN-02: the HP registry trial of a TUNING run")
    ap.add_argument("--hp-registry", type=Path, default=None,
                    help="default contracts/hp_registry/HP_REGISTRY_P2.json")
    ap.add_argument("--theta0-variants", type=Path, default=None,
                    help="default contracts/model_variants/THETA0_VARIANTS.json")
    ap.add_argument("--recipe-id", default=None,
                    help="TRAIN-02R1 F6: a tuned recipe (02B_GRU / 02B_SASREC) for a --phase MAIN run (P3)")
    ap.add_argument("--tuned-recipes", type=Path, default=None,
                    help="default contracts/hp_registry/TUNED_RECIPES_02B.json")
    ap.add_argument("--gpu", required=True, type=int, help="PHYSICAL GPU index (exported as CUDA_VISIBLE_DEVICES)")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-updates", type=int, default=600)
    ap.add_argument("--smoke-interrupt-at", type=int, default=0)
    ap.add_argument("--checkpoint-every", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--continue-run", action="store_true")
    ap.add_argument("--calibration-record", default=None)
    ap.add_argument("--continuation-budget", default=None)
    ap.add_argument("--expected-init-sha256", default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--eval-prefetch", type=int, default=2)
    ap.add_argument("--train-manifest", default=None, help="default: v4train.provider.expected_manifests(...)[0]")
    ap.add_argument("--eval-manifest", default=None, help="default: v4train.provider.expected_manifests(...)[1]")
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--lock-path", type=Path, default=None)
    ap.add_argument("--manifest-hashes", type=Path, default=None)
    ap.add_argument("--header-list", type=Path, default=None)
    ap.add_argument("--transform-state", type=Path, default=None)
    ap.add_argument("--population-label", default=None, help="recorded in every record, e.g. HISTORICAL_POPULATION")
    ap.add_argument("--allow-shared-gpu", action="store_true",
                    help="INT-03: do not refuse a GPU that already runs a process (recorded; refused for MAIN)")
    ap.add_argument("--coschedule-06g", action="store_true",
                    help="PREG1_AMENDMENT_06G §5 (owner, 2026-09-29): with --allow-shared-gpu, allow a central MAIN run to share "
                         "its card (STRICT numerics are independent of co-scheduling; recorded in the launch record)")
    from v4hist.policy import HISTORY_POLICIES                       # import-free module (no torch, no snapshot)
    ap.add_argument("--history-policy", required=True, choices=HISTORY_POLICIES,
                    help="REQUIRED input-history policy (DATA-05): USER_HISTORY_L50 (locked) or SESSION_L50")
    return ap.parse_args(argv)


def data_paths(args):
    """The locked data_v3 tree (DATA_PROTOCOL_LOCK_V2) by default; any field may be overridden (a new data version)."""
    from dataclasses import replace
    from v4int.verify import DataPaths
    p = DataPaths.default()
    over = {k: getattr(args, a) for k, a in (("manifest_dir", "manifest_dir"), ("lock_path", "lock_path"),
                                             ("manifest_hashes_path", "manifest_hashes"),
                                             ("header_list_path", "header_list"),
                                             ("transform_state_path", "transform_state"))
            if getattr(args, a, None) is not None}
    return replace(p, **over) if over else p


def check_lock_pin(lock_self: str, *, smoke: bool, gate_path: Path = G1_GATE) -> Optional[str]:
    """TRAIN-02R2 F3: a non-smoke run binds only the lock pinned at G1: the recomputed lock self-hash must equal
    company/gates/G1_APPROVED.json's `lock_v2_self` (a missing gate file or field refuses). Smoke runs are not pinned."""
    if smoke:
        return None
    try:
        gate = json.loads(Path(gate_path).read_text(encoding="utf-8"),
                          parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))
    except (OSError, ValueError) as e:
        raise LaunchRefused(f"lock pin: {gate_path} unreadable ({e})") from e
    pinned = gate.get("lock_v2_self") if isinstance(gate, dict) else None
    if not isinstance(pinned, str) or not pinned:
        raise LaunchRefused(f"lock pin: {gate_path} carries no `lock_v2_self`")
    if pinned != lock_self:
        raise LaunchRefused(f"the DATA_PROTOCOL_LOCK self-hash {lock_self} is not the lock pinned at G1 ({pinned})")
    return pinned


def bind_data_protocol_lock_v2(paths, *, smoke: bool, gate_path: Path = G1_GATE) -> dict:
    """INT-02 item 1: binds `contracts/DATA_PROTOCOL_LOCK_V2.json` (via `paths`) and reads, EXPLICITLY from the lock
    (no code default), the production data_version, the REGISTRY_V2 manifest-role registry (path + file sha256, cross-
    checked against the lock's own binding) and the S* production schema (id, self-hash, and the derived-schema
    registry the trainer will re-verify it against — again cross-checked against the lock's binding)."""
    from v4int.verify import ManifestVerificationError, verify_lock
    try:
        lock, lock_self, lock_file = verify_lock(paths)
    except ManifestVerificationError as e:
        raise LaunchRefused(f"DATA_PROTOCOL_LOCK refused: {e}") from e
    pinned = check_lock_pin(lock_self, smoke=smoke, gate_path=gate_path)      # TRAIN-02R2 F3
    try:
        data_version = lock["data_version"]["name"]
        mr = lock["population_and_manifests"]["manifest_registry"]
        prod = lock["features_and_transforms"]["production_schema"]
        mr_path, mr_file, mr_self = str(mr["path"]), mr["file_sha256"], mr["self_sha256"]
        schema_id, schema_self = prod["schema_id"], prod["self_sha256"]
        schema_reg_file = prod["derived_schema_registry_file_sha256"]
    except (KeyError, TypeError) as e:                                           # a v1 or other lock shape
        raise LaunchRefused(f"{paths.lock_path} is not a DATA_PROTOCOL_LOCK_V2-shaped lock (missing {e})") from e
    manifest_registry_path = V4 / mr_path
    mr_file_sha = hashlib.sha256(manifest_registry_path.read_bytes()).hexdigest()
    if mr_file_sha != mr_file:
        raise LaunchRefused(f"{manifest_registry_path}: file sha256 differs from the DATA_PROTOCOL_LOCK_V2 binding")
    schema_registry_path = V4 / "contracts" / "derived_schemas" / "REGISTRY.json"
    schema_registry_file_sha = hashlib.sha256(schema_registry_path.read_bytes()).hexdigest()
    if schema_registry_file_sha != schema_reg_file:
        raise LaunchRefused(f"{schema_registry_path}: file sha256 differs from the DATA_PROTOCOL_LOCK_V2 binding")
    return {"lock": lock, "lock_self_sha256": lock_self, "lock_file_sha256": lock_file, "data_version": data_version,
            "lock_pinned_at_g1": pinned,
            "manifest_registry_path": manifest_registry_path, "manifest_registry_rel_path": mr_path,
            "manifest_registry_file_sha256": mr_file_sha, "manifest_registry_self_sha256": mr_self,
            "schema_id": schema_id, "schema_self_sha256": schema_self,
            "schema_registry_path": schema_registry_path, "schema_registry_file_sha256": schema_registry_file_sha}


def build_trainer_registry_view(manifest_registry_path: Path, run_dir: Path, *, registry_rel_path: str,
                                registry_file_sha256: str) -> dict:
    """v4train.registry.load_manifest_registry (the trainer's OWN registry loader, RunConfig.manifest_registry / a
    resume's binding) refuses any role name outside v4train.recipe.ROLES ("MAIN", "CALIBRATION", "CONTINUATION",
    "FULL_TRAIN", "FEATURE_SEARCH_SCREEN"); REGISTRY_V2 also carries DATA-06's other population-band roles
    (UNSEEN_VAL, UNSEEN_HOLDOUT, SPARSE_PANEL, HEAVY_PANEL, TRUE_COLD, COLD_H, WARM_TRUNCATION), so the trainer cannot
    load REGISTRY_V2 whole. v4int (this launcher) is not v4train and does not edit it, so it derives a trainer-facing
    VIEW instead: the SAME data_version, only the MAIN / CALIBRATION roles this launcher can start (+ CONTINUATION
    mirroring MAIN, exactly as the historical registry does — v4train.trainer.continue_run looks it up by that name)
    and the manifest entries they name, every field copied byte-for-byte from REGISTRY_V2 (no fabricated value),
    re-hashed with v4train.registry's own canonical self-hash rule. The providers (StoreTrainProvider /
    StoreEvalProvider) still verify against REGISTRY_V2 itself, unfiltered; this view is for the trainer's role
    lookup only.
    INT-02R1 F2 (Ops decision: option a): `derived_from` records the LOCK's own RELATIVE registry path
    (`registry_rel_path`, e.g. "contracts/manifest_registry/REGISTRY_V2.json") and BOTH REGISTRY_V2's file_sha256
    (`registry_file_sha256`, already verified against the lock's binding by `bind_data_protocol_lock_v2`) and its
    self_sha256 — so the view's own self_sha256 is deterministic and host/root-independent (it never encodes an
    absolute path); only the absolute `manifest_registry_path` passed in for reading the file, and the `path` this
    function returns for the caller to record in LAUNCH_<k>.json, are host-specific."""
    from v4train.registry import with_self_sha256
    reg = json.loads(Path(manifest_registry_path).read_text(encoding="utf-8"))
    roles_view = {r: dict(reg["roles"][r]) for r in ("MAIN", "CALIBRATION") if r in reg["roles"]}
    if "MAIN" in roles_view:
        roles_view["CONTINUATION"] = dict(roles_view["MAIN"])
    needed = {mid for pair in roles_view.values() for mid in pair.values()}
    manifests_view = {mid: dict(reg["manifests"][mid]) for mid in needed}
    obj = with_self_sha256({
        "data_version": reg["data_version"], "roles": roles_view, "manifests": manifests_view,
        "derived_from": {"path": registry_rel_path, "file_sha256": registry_file_sha256,
                         "self_sha256": reg.get("self_sha256"),
                         "note": "v4int.launch_central.build_trainer_registry_view: MAIN / CALIBRATION (+ mirrored "
                                 "CONTINUATION) roles only, every field byte-identical to the source registry"}})
    run_dir.mkdir(parents=True, exist_ok=True)
    p = run_dir / "REGISTRY_V2_TRAINER_VIEW.json"
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return {"path": p, "self_sha256": obj["self_sha256"], "roles": sorted(roles_view),
            "source_registry_self_sha256": reg.get("self_sha256"), "source_registry_file_sha256": registry_file_sha256}


def load_production_schema(schema_id: str, schema_self_sha256: str, registry_path: Path):
    """The S* production schema (a v4models DerivedSchema), loaded and verified through the derived-schema registry
    (v4fs.schemas.load_registered: registry self-hash, file sha256 and schema self-hash all verified), then bound to
    the exact self-hash DATA_PROTOCOL_LOCK_V2 records for it."""
    from v4fs.schemas import load_registered
    ds = load_registered(schema_id, registry_path)
    if ds.self_sha256 != schema_self_sha256:
        raise LaunchRefused(f"schema {schema_id}: self-hash {ds.self_sha256} != the DATA_PROTOCOL_LOCK_V2 binding "
                            f"{schema_self_sha256}")
    return ds


# ------------------------------------------------------------------------------------------------ preflight
def _gpu_memory_used(gpu: int, runner: Callable) -> dict:
    try:
        p = runner(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"])
    except Exception as e:                                       # fail closed
        raise LaunchRefused(f"GPU check fail-closed: nvidia-smi unavailable ({e})") from e
    if p.returncode != 0:
        raise LaunchRefused(f"GPU check fail-closed: nvidia-smi exit {p.returncode}")
    rows = {}
    for line in p.stdout.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) != 6:
            raise LaunchRefused(f"GPU check fail-closed: unparsable nvidia-smi line {line!r}")
        rows[parts[0]] = {"uuid": parts[1], "name": parts[2], "memory_used_mib": float(parts[3]),
                          "memory_total_mib": float(parts[4]), "utilization_pct": float(parts[5])}
    if str(gpu) not in rows:
        raise LaunchRefused(f"GPU check fail-closed: physical GPU {gpu} is not enumerated ({sorted(rows)})")
    return rows


def preflight(args: argparse.Namespace, *, runner: Optional[Callable] = None, environ=None,
              gate_path: Path = G1_GATE) -> dict:
    """Every refusal that needs no torch / snapshot import. Returns the record written into LAUNCH_<k>.json."""
    from v4sup.gpu import default_runner                             # no torch, no snapshot
    from v4int.smoke import MAX_SMOKE_UPDATES
    env = os.environ if environ is None else environ
    runner = runner or default_runner
    rec = {"utc": utc(), "argv": list(sys.argv), "host": platform.node(), "smoke": bool(args.smoke)}
    from v4hist.policy import HistoryPolicyError, require_policy
    try:
        rec["history_policy"] = require_policy(getattr(args, "history_policy", None))
    except HistoryPolicyError as e:
        raise LaunchRefused(str(e)) from e
    if args.population != "COHORT":
        raise LaunchRefused("population FULL_TRAIN refused: the tie-break manifests M_FULL_TRAIN / "
                            "M_FULL_VALIDATION_WARM are not built (PREG1_AMENDMENT_01 item 2)")
    if args.gpu is None or int(args.gpu) < 0:
        raise LaunchRefused("an explicit physical GPU id (--gpu >= 0) is required")
    if args.phase not in PHASES or args.family not in FAMILIES:
        raise LaunchRefused(f"phase must be one of {PHASES} and family one of {FAMILIES}")
    if args.resume and args.continue_run:
        raise LaunchRefused("--resume and --continue-run are exclusive")
    _role_preflight(args, rec)
    if args.smoke:
        if not (1 <= int(args.smoke_updates) <= MAX_SMOKE_UPDATES):
            raise LaunchRefused(f"--smoke-updates must be in [1, {MAX_SMOKE_UPDATES}]")
        ce = args.checkpoint_every if args.checkpoint_every is not None else 0
        k = int(args.smoke_interrupt_at)
        if k < 0 or k >= int(args.smoke_updates) or (k and (not ce or k % ce)):
            raise LaunchRefused("--smoke-interrupt-at must be 0, or below --smoke-updates and a multiple of "
                                "--checkpoint-every (the leg stops after a `latest` save)")
        if args.continue_run:
            raise LaunchRefused("a smoke run has no continuation")
        rec["g1_gate"] = {"required": False, "reason": "smoke (SMOKE_NOT_EVIDENCE)"}
    else:
        if not Path(gate_path).is_file():
            raise LaunchRefused(f"non-smoke run refused: {gate_path} does not exist (G1 not approved)")
        try:                                                  # (v4int.verify would import v4train / the snapshot)
            obj = json.loads(Path(gate_path).read_text(encoding="utf-8"),
                             parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))
        except Exception as e:
            raise LaunchRefused(f"non-smoke run refused: {gate_path} is not strict JSON ({e})") from e
        if not isinstance(obj, dict):
            raise LaunchRefused(f"non-smoke run refused: {gate_path} is not a JSON object")
        rec["g1_gate"] = {"required": True, "path": str(gate_path),
                          "sha256": hashlib.sha256(Path(gate_path).read_bytes()).hexdigest()}
        if int(args.smoke_interrupt_at):
            raise LaunchRefused("--smoke-interrupt-at is a smoke-only option")
        if args.phase == "MAIN" and not (args.resume or args.continue_run) and not args.calibration_record \
                and not getattr(args, "recipe_id", None):
            raise LaunchRefused("a MAIN run needs --calibration-record (the main peak comes from the calibration) or "
                                "--recipe-id (a tuned recipe of TUNED_RECIPES_02B.json)")
    if getattr(args, "coschedule_06g", False) and not getattr(args, "allow_shared_gpu", False):
        raise LaunchRefused("--coschedule-06g takes effect only together with --allow-shared-gpu (DECISION_LOG 208)")
    if getattr(args, "allow_shared_gpu", False) and args.phase == "MAIN" and not args.smoke \
            and not getattr(args, "coschedule_06g", False):
        raise LaunchRefused("--allow-shared-gpu is refused for a central MAIN run (PREG1_AMENDMENT_06 §4: one "
                            "central MAIN-scale job per card) unless --coschedule-06g (06G §5 / 06K §1) is given")
    rec["gpu_check"] = gpu_checks(args, runner, env)
    rec["coschedule_06g"] = bool(getattr(args, "coschedule_06g", False))
    return rec


def gpu_checks(args, runner: Callable, env) -> dict:
    """INT-03: the GPU part of the preflight, shared by launch_central / launch_fl / launch_device. With
    args.allow_shared_gpu the ownership refusal and the foreign-memory guard are RECORDED, not refused; an nvidia-smi
    failure or an unenumerated GPU refuses either way (fail-closed)."""
    from v4sup.gpu import gpu_ready
    shared = bool(getattr(args, "allow_shared_gpu", False))
    inherited = env.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None and inherited != str(int(args.gpu)):
        raise LaunchRefused(f"inherited CUDA_VISIBLE_DEVICES={inherited!r} differs from --gpu {args.gpu}")
    ok, why = gpu_ready([int(args.gpu)], runner=runner)
    if not ok and (not shared or "fail-closed" in why):
        raise LaunchRefused(f"GPU ownership check refused physical GPU {args.gpu}: {why}")
    mem = _gpu_memory_used(int(args.gpu), runner)
    used = mem[str(int(args.gpu))]["memory_used_mib"]
    if used > GPU_MEM_FOREIGN_MIB and not shared:
        raise LaunchRefused(f"GPU {args.gpu} has {used} MiB in use with no visible compute process of ours "
                            f"(> {GPU_MEM_FOREIGN_MIB} MiB): a foreign process; refused")
    out = {"gpu_ready": why, "nvidia_smi": mem, "inherited_cuda_visible_devices": inherited}
    if shared:
        out["shared_gpu"] = {"allowed": True, "gpu_ready_ok": bool(ok), "memory_used_mib_at_launch": used,
                             "authority": "PREG1_AMENDMENT_06 §4 (FL / device jobs may share one A16)"}
    return out


def _role_preflight(args, rec: dict) -> None:
    """TRAIN-02 / TRAIN-02R1 (torch-free): the TUNING role, the tuned-recipe MAIN path and their flag refusals."""
    role = getattr(args, "role", None)
    hp_id = getattr(args, "hp_id", None)
    recipe_id = getattr(args, "recipe_id", None)
    if role == "TUNING":
        if not hp_id:
            raise LaunchRefused("--role TUNING needs --hp-id (a trial of the HP registry)")
        if args.phase != "CALIBRATION":
            raise LaunchRefused("a TUNING run uses --phase CALIBRATION (the registry's CALIBRATION manifests)")
        if args.calibration_record or args.continue_run or args.continuation_budget:
            raise LaunchRefused("a TUNING run has no calibration record and no continuation")
        if args.train_manifest or args.eval_manifest:                  # F2: never another manifest (MAIN_VALIDATION)
            raise LaunchRefused("a TUNING run takes its manifests from the registry's CALIBRATION role only: "
                                "--train-manifest / --eval-manifest are refused")
        if recipe_id:
            raise LaunchRefused("a TUNING run uses --hp-id, never --recipe-id")
        rec["tuning"] = {"role": "TUNING", "hp_id": hp_id}
        return
    if hp_id:
        raise LaunchRefused("--hp-id is valid with --role TUNING only")
    if recipe_id:                                                       # F6: the P3 tuned-recipe MAIN path
        if args.phase != "MAIN":
            raise LaunchRefused("--recipe-id is valid for a --phase MAIN run only")
        if args.calibration_record:
            raise LaunchRefused("a tuned-recipe run takes its peak LR from the recipe, not --calibration-record")
        rec["tuned_recipe"] = {"recipe_id": recipe_id}
        return
    if args.peak_lr is None:
        raise LaunchRefused("--peak-lr is required (unless --role TUNING or --recipe-id)")


def export_environment(gpu: int, environ=None, lock_path: Path = PREG1_LOCK) -> dict:
    """CUDA_VISIBLE_DEVICES, CUBLAS_WORKSPACE_CONFIG (the T3 lock value) and PYTHONDONTWRITEBYTECODE, set before torch
    and the snapshot are imported."""
    env = os.environ if environ is None else environ
    t3 = json.loads(Path(lock_path).read_text(encoding="utf-8"))["T3_eval_numerics"]["flags"]["CUBLAS_WORKSPACE_CONFIG"]
    prev = env.get("CUBLAS_WORKSPACE_CONFIG")
    if prev is not None and prev != t3:
        raise LaunchRefused(f"inherited CUBLAS_WORKSPACE_CONFIG={prev!r} differs from the T3 value {t3!r}")
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    env["CUBLAS_WORKSPACE_CONFIG"] = t3
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return {"CUDA_VISIBLE_DEVICES": env["CUDA_VISIBLE_DEVICES"], "CUBLAS_WORKSPACE_CONFIG": t3,
            "PYTHONDONTWRITEBYTECODE": "1", "cuda_imported_before_export": "torch" in sys.modules}


# ------------------------------------------------------------------------------------------------ helpers
def _json_safe(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _next_launch_path(run_dir: Path) -> Path:
    k = 0
    while (run_dir / f"LAUNCH_{k}.json").exists():
        k += 1
    return run_dir / f"LAUNCH_{k}.json"


def tensor_tree_digest(obj) -> str:
    """sha256 over a nested state (dicts / lists / tensors / numbers), tensors by dtype, shape and raw bytes."""
    import torch
    h = hashlib.sha256()

    def walk(x, path):
        if isinstance(x, dict):
            for k in sorted(x, key=lambda z: str(z)):
                walk(x[k], f"{path}/{k}")
        elif isinstance(x, (list, tuple)):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")
        elif torch.is_tensor(x):
            t = x.detach().cpu().contiguous()
            h.update(f"{path}|{t.dtype}|{tuple(t.shape)}".encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        else:
            h.update(f"{path}|{type(x).__name__}|{x!r}".encode())
    walk(obj, "")
    return h.hexdigest()


DISK_FLOOR_BYTES = 8 * 2 ** 30


def disk_record(path: Path, planned_peak_bytes: float = 0.0) -> dict:
    import shutil
    p = Path(path)
    while not p.exists():
        p = p.parent
    free = int(shutil.disk_usage(str(p)).free)
    return {"utc": utc(), "free_bytes": free, "planned_peak_bytes": int(planned_peak_bytes),
            "headroom_bytes": int(free - planned_peak_bytes), "floor_bytes": DISK_FLOOR_BYTES}


def determinism_mode_in_effect() -> dict:
    import torch
    on = bool(torch.are_deterministic_algorithms_enabled())
    warn = bool(torch.is_deterministic_algorithms_warn_only_enabled())
    return {"determinism_mode_in_effect": ("STRICT" if on and not warn else "WARN_ONLY" if on else "OFF"),
            "deterministic_algorithms": on, "warn_only": warn,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def state_digests(tr) -> dict:
    import torch
    from fl.numerics import state_digest
    ts = tr.state.as_dict()
    return {"model_state_dict": state_digest(tr.module.state_dict()),
            "optimizer_state_dict": tensor_tree_digest(tr.opt.state_dict()),
            "rng_torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "rng_cuda": tensor_tree_digest(torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
            "train_state": hashlib.sha256(json.dumps(ts, sort_keys=True).encode()).hexdigest(),
            "stop_state": hashlib.sha256(json.dumps(tr.tracker.state(), sort_keys=True).encode()).hexdigest(),
            "updates": int(tr.state.updates), "exposures": int(tr.state.exposures),
            "pass_index": int(tr.state.pass_index), "pos": int(tr.state.pos)}


# ------------------------------------------------------------------------------------------------ run
def build_launch_dict(*, args, key, train_id, eval_id, want_train, want_eval, data_paths, lockctx,
                      manifest_registry_path, trainer_view, pre, env_rec, verified_files, isolated_modules,
                      device_name, device_uuid, device_total_memory, torch_version, numpy_version, python_version,
                      run_id, config_sha256, init_sha256, trainer_code_sha, t_prov, train_provider, eval_provider,
                      det_passed, determinism_at_init, disk, tuned, tuning, theta0_init_sha256, run_label,
                      stop_at, interrupt_at, ce) -> dict:
    """Build the exact LAUNCH record from already observed inputs, without I/O or training."""
    return {"task": "INT-01 launch_central", "label": SMOKE_LABEL if args.smoke else "PRODUCTION",
            "population_label": args.population_label, "history_policy": args.history_policy,
            "comparison_key": dict(key),
            "manifests": {"train": train_id, "eval": eval_id, "trainer_expects": [want_train, want_eval],
                          "source": "argument" if (args.train_manifest or args.eval_manifest)
                          else "v4train.provider.expected_manifests (REGISTRY_V2 role)"},
            "data_paths": data_paths,
            "data_protocol_lock_v2": {"path": str(lockctx["lock_path"]),
                                     "lock_self_sha256": lockctx["lock_self_sha256"],
                                     "lock_file_sha256": lockctx["lock_file_sha256"],
                                     "data_version": lockctx["data_version"],
                                     "manifest_registry": {"path": str(manifest_registry_path),
                                                           "file_sha256": lockctx["manifest_registry_file_sha256"],
                                                           "self_sha256": lockctx["manifest_registry_self_sha256"]},
                                     "trainer_registry_view": {"path": str(trainer_view["path"]),
                                                               "self_sha256": trainer_view["self_sha256"],
                                                               "roles": trainer_view["roles"]},
                                     "schema": {"schema_id": lockctx["schema_id"],
                                               "schema_self_sha256": lockctx["schema_self_sha256"],
                                               "registry_path": str(lockctx["schema_registry_path"]),
                                               "registry_file_sha256": lockctx["schema_registry_file_sha256"]}},
            "utc_start": pre["utc"], "preflight": pre, "environment": env_rec,
            "snapshot": {"verified_files": verified_files, "isolated_modules": isolated_modules},
            "device": {"physical_gpu": int(args.gpu), "in_process": "cuda:0", "name": device_name,
                       "uuid": device_uuid, "total_memory_bytes": device_total_memory},
            "torch": torch_version, "numpy": numpy_version, "python": python_version,
            "run_id": run_id, "config_sha256": config_sha256, "init_sha256": init_sha256,
            "trainer_code_sha256": trainer_code_sha,
            "providers_build_s": round(t_prov, 2), "train_provider": train_provider,
            "eval_provider": eval_provider,
            "mode": "resume" if args.resume else ("continue" if args.continue_run else "fresh"),
            "determinism": {"constant": DETERMINISM_MODE, "passed_to_runconfig": det_passed,
                            "at_trainer_init": determinism_at_init},
            "disk_at_start": disk,
            "tuned_recipe": None if not tuned else dict(tuned, theta0_init_sha256=theta0_init_sha256),
            "tuning": None if not tuning else dict(tuning, theta0_init_sha256=theta0_init_sha256,
                                                   run_label=run_label),
            "smoke": {"updates_cap": stop_at, "interrupt_at": interrupt_at, "checkpoint_every": ce} if args.smoke
            else None}


def run(args: argparse.Namespace, pre: dict, env_rec: dict, say: Callable = print) -> int:
    import v4train                                   # captures CUDA_VISIBLE_DEVICES, then imports ppsi_env first
    import ppsi_env
    snap = ppsi_env.verify_snapshot()
    iso = ppsi_env.assert_isolated()
    if v4train.LAUNCH_CUDA_VISIBLE_DEVICES != str(int(args.gpu)):
        raise LaunchRefused(f"v4train captured CUDA_VISIBLE_DEVICES={v4train.LAUNCH_CUDA_VISIBLE_DEVICES!r}, not "
                            f"{args.gpu!r}: the export did not precede the import")
    import numpy as np
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise LaunchRefused(f"expected exactly one visible CUDA device, got available={torch.cuda.is_available()} "
                            f"count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    torch.set_num_threads(int(args.n_threads))
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    props = torch.cuda.get_device_properties(dev)
    from v4data.common import dataset_root
    from v4int.providers import StoreEvalProvider, StoreTrainProvider, comparison_key_base
    from v4int.smoke import SMOKE_LABEL, SMOKE_RUN_PREFIX, smoke_cap_before_first_mark, smoke_rows
    from v4train.checkpoint import write_json_atomic
    from v4train.provider import expected_manifests
    from v4train.recipe import RunConfig
    from v4train.schedule import mark_exposures
    from v4train.trainer import CentralTrainer, InjectedInterruption, trainer_code_sha256
    from v4sup.status import write_status_atomic

    class SmokeStop(InjectedInterruption):
        pass

    run_dir = Path(args.run_dir)
    fam, phase = args.family, args.phase
    from v4int.small_recipe import install as _small_install          # INT-03: SMALL-format files only
    _small_install()
    tuning = _tuning_values(args, fam) if getattr(args, "role", None) == "TUNING" else None
    tuned = _recipe_values(args, fam) if getattr(args, "recipe_id", None) else None
    paths = data_paths(args)
    lockctx = bind_data_protocol_lock_v2(paths, smoke=bool(args.smoke))         # LOCK_V2 (item 1); pinned (02R2 F3)
    manifest_registry_path = lockctx["manifest_registry_path"]                  # the REAL REGISTRY_V2 (for the providers)
    trainer_view = build_trainer_registry_view(                                 # MAIN/CALIBRATION view (the trainer)
        manifest_registry_path, run_dir, registry_rel_path=lockctx["manifest_registry_rel_path"],
        registry_file_sha256=lockctx["manifest_registry_file_sha256"])
    # REGISTRY_V2 roles (item 1): CALIBRATION -> CALIB_TRAIN/CALIB_VALIDATION; MAIN -> MAIN_TRAIN/MAIN_VALIDATION
    want_train, want_eval = expected_manifests(args.population, phase, registry=str(trainer_view["path"]))
    train_id = args.train_manifest or want_train
    eval_id = args.eval_manifest or want_eval
    t0 = time.time()
    tp = StoreTrainProvider(train_id, history_policy=args.history_policy, paths=paths, log=say,
                           manifest_registry=manifest_registry_path, registry_root=V4)
    ep = StoreEvalProvider(eval_id, history_policy=args.history_policy, paths=paths, prefetch=int(args.eval_prefetch),
                           log=say, manifest_registry=manifest_registry_path, registry_root=V4)
    t_prov = time.time() - t0
    key = comparison_key_base(paths, tp, ep)
    peak_lr = tuning["peak_lr"] if tuning else (tuned["peak_lr"] if tuned else float(args.peak_lr))
    lr_tag = f"{peak_lr:.0e}".replace("-0", "-")
    default_id = f"{tuning['hp_id']}_s{args.seed}" if tuning else (
        f"{tuned['recipe_id']}_{phase}_s{args.seed}" if tuned else f"{fam}_{phase}_s{args.seed}_lr{lr_tag}")
    run_id = args.run_id or f"{SMOKE_RUN_PREFIX if args.smoke else ''}{default_id}"
    if args.smoke and not run_id.startswith(SMOKE_RUN_PREFIX):
        raise LaunchRefused(f"a smoke run id must start with {SMOKE_RUN_PREFIX}")
    ce = int(args.checkpoint_every) if args.checkpoint_every is not None else 0
    extra = {}
    det_passed = "determinism_mode" in getattr(RunConfig, "__dataclass_fields__", {})
    if det_passed:
        extra["determinism_mode"] = DETERMINISM_MODE
    f = getattr(RunConfig, "__dataclass_fields__", {})                          # TRAIN-01R2: REQUIRED keys
    if "manifest_registry" in f:
        extra["manifest_registry"] = str(trainer_view["path"])
    if "data_version" in f:
        extra["data_version"] = lockctx["data_version"]
    if "schema_registry" in f:
        extra["schema_registry"] = str(lockctx["schema_registry_path"])
    if tuning:                                                                  # TRAIN-02: the HP trial's values
        extra.update(role="TUNING", hp_id=tuning["hp_id"], hp_registry=tuning["hp_registry"],
                     model_variant=tuning["model_variant"], dropout=tuple(tuning["dropout"]),
                     schedule=tuning["schedule"],
                     theta0_init_sha256=_theta0_for(args, lockctx["schema_id"], tuning, int(args.seed)))
    if tuned:                                                                   # TRAIN-02R1 F6: the tuned recipe
        extra.update(recipe_id=tuned["recipe_id"], recipe_registry=tuned["recipe_registry"],
                     model_variant=tuned["model_variant"], dropout=tuple(tuned["dropout"]),
                     schedule=tuned["schedule"],
                     theta0_init_sha256=_theta0_for(args, lockctx["schema_id"], tuned, int(args.seed)))
    cfg = RunConfig(run_id=run_id, family=fam, seed=int(args.seed), exposure_point="end",
                    cuda_visible_devices=str(int(args.gpu)), phase=phase, population="COHORT",
                    peak_lr=peak_lr, log_every=int(args.log_every), checkpoint_every=ce,
                    calibration_record=args.calibration_record, continuation_budget=args.continuation_budget,
                    expected_init_sha256=args.expected_init_sha256, key=key, test_overrides=bool(args.smoke), **extra)
    if args.smoke and not (smoke_cap_before_first_mark(int(args.smoke_updates), cfg.effective_batch, tp.n_decisions)
                           and int(args.smoke_updates) * cfg.effective_batch < mark_exposures(0.5, tp.n_decisions)):
        raise LaunchRefused("the smoke cap must end before the first 0.5-EFE mark (no mark row may be written)")
    # the S* production schema (item 1), never the base FEATURE_SCHEMA_CLEAN_V2 (schema_id and self-hash come from the
    # DATA_PROTOCOL_LOCK_V2 binding above, not a code default)
    schema = load_production_schema(lockctx["schema_id"], lockctx["schema_self_sha256"], lockctx["schema_registry_path"])
    clock: dict = {}
    mode_seen: dict = {}
    stop_at, interrupt_at = int(args.smoke_updates), int(args.smoke_interrupt_at)

    def fault(u: int) -> None:
        if not mode_seen:                            # the flags in force inside the training loop (not the eval's)
            mode_seen.update(determinism_mode_in_effect(), at_update=u)
        if not args.smoke:
            return
        clock[u] = time.perf_counter()
        if interrupt_at and u == interrupt_at:
            raise SmokeStop(f"SMOKE_INTERRUPT after update {u} (resume test, first leg)")
        if u >= stop_at:
            raise SmokeStop(f"SMOKE_BUDGET_REACHED at update {u}")

    tr = CentralTrainer(cfg, schema, tp, ep, run_dir, device="cuda:0", v1_root=dataset_root(), fault=fault)
    disk = disk_record(run_dir, tr._planned_peak_gib(cfg.train_phase if hasattr(cfg, "train_phase") else phase)
                       * 2 ** 30)                                           # the trainer's own planned-peak rule
    if disk["headroom_bytes"] < DISK_FLOOR_BYTES:
        raise LaunchRefused(f"disk: free {disk['free_bytes']} B - planned peak {disk['planned_peak_bytes']} B < the "
                            f"8 GiB floor: stopped before any weight is written (Operations disk rule)")
    launch = build_launch_dict(args=args, key=key, train_id=train_id, eval_id=eval_id, want_train=want_train,
                               want_eval=want_eval, data_paths=paths.as_json(),
                               lockctx=dict(lockctx, lock_path=paths.lock_path),
                               manifest_registry_path=manifest_registry_path, trainer_view=trainer_view,
                               pre=pre, env_rec=env_rec, verified_files=len(snap["files"]),
                               isolated_modules=len(iso), device_name=torch.cuda.get_device_name(dev),
                               device_uuid=str(getattr(props, "uuid", "")),
                               device_total_memory=int(props.total_memory), torch_version=torch.__version__,
                               numpy_version=np.__version__, python_version=sys.version.split()[0], run_id=run_id,
                               config_sha256=tr.config_sha256, init_sha256=tr.built.init_sha256,
                               trainer_code_sha=trainer_code_sha256(), t_prov=t_prov,
                               train_provider=tp.describe(), eval_provider=ep.describe(), det_passed=det_passed,
                               determinism_at_init=determinism_mode_in_effect(), disk=disk, tuned=tuned,
                               tuning=tuning, theta0_init_sha256=cfg.theta0_init_sha256, run_label=cfg.run_label,
                               stop_at=stop_at, interrupt_at=interrupt_at, ce=ce)
    if tr.device_pin_sha256:
        launch["device_pin_sha256"] = tr.device_pin_sha256
    run_dir.mkdir(parents=True, exist_ok=True)
    lp = _next_launch_path(run_dir)
    write_json_atomic(lp, _json_safe(launch))
    say(f"[launch] {run_id}: providers {t_prov:.1f} s; {lp.name} written; mode {launch['mode']}")
    torch.cuda.reset_peak_memory_stats(dev)
    t_train = time.perf_counter()
    try:
        if args.resume:
            out = tr.resume()
        elif args.continue_run:
            out = tr.continue_run()
        else:
            out = tr.run()
    except SmokeStop as e:
        if not args.smoke:
            raise
        launch["determinism"]["in_training_loop"] = dict(mode_seen)
        try:
            return _finish_smoke(args, tr, ep, run_dir, clock, t_train, str(e), launch, smoke_rows, write_json_atomic,
                                 write_status_atomic, dev, say)
        except BaseException:
            _drop_smoke_weights(tr, "INT-01 smoke: leg failed after its stop; weights deleted by the same job")
            raise
    except BaseException:
        if args.smoke:
            _drop_smoke_weights(tr, "INT-01 smoke: leg failed; weights deleted by the same job")
        raise
    say(f"[launch] {run_id}: {out.get('status')}")          # the status word only (TRAIN-03 seal: no value)
    return 0 if str(out.get("status", "")).startswith("DONE") else 1


def _tuning_values(args, fam: str) -> dict:
    """TRAIN-02: the HP trial named by --hp-id (hash-checked registry), after the environment export."""
    from v4train.hp_registry import REGISTRY_PATH, HPRegistryError, load_hp_registry, trial
    path = Path(args.hp_registry) if getattr(args, "hp_registry", None) else REGISTRY_PATH
    try:
        reg = load_hp_registry(path)
        t = trial(reg, args.hp_id)
    except HPRegistryError as e:
        raise LaunchRefused(f"HP registry refused: {e}") from e
    if t["family"] != fam:
        raise LaunchRefused(f"--hp-id {args.hp_id} is a {t['family']} trial, not {fam}")
    if args.peak_lr is not None and float(args.peak_lr) != float(t["peak_lr"]):
        raise LaunchRefused(f"--peak-lr {args.peak_lr} differs from trial {t['hp_id']}'s {t['peak_lr']!r}")
    return {"hp_id": t["hp_id"], "hp_sha256": t["sha256"], "hp_registry": str(path),
            "hp_registry_file_sha256": reg["file_sha256"], "hp_registry_self_sha256": reg["self_sha256"],
            "peak_lr": float(t["peak_lr"]), "dropout": list(t["dropout"]), "schedule": t["schedule"],
            "model_variant": t["model_variant"]}


def _recipe_values(args, fam: str) -> dict:
    """TRAIN-02R1 F6: the tuned recipe named by --recipe-id (hash-checked TUNED_RECIPES_02B.json)."""
    from v4train.hp_registry import REGISTRY_PATH as HP_PATH
    from v4train import tuned_recipes as TRM
    path = Path(args.tuned_recipes) if getattr(args, "tuned_recipes", None) else TRM.REGISTRY_PATH
    hp_path = Path(args.hp_registry) if getattr(args, "hp_registry", None) else HP_PATH
    try:
        reg = TRM.load_tuned_recipes(path, hp_path)
        if not args.smoke:                                         # TRAIN-02R2 F2: the production file, pinned at G1b
            TRM.check_production_registry(path, reg)
        r = TRM.recipe(reg, args.recipe_id)
    except TRM.TunedRecipeError as e:
        raise LaunchRefused(f"tuned recipe refused: {e}") from e
    if r["family"] != fam:
        raise LaunchRefused(f"--recipe-id {args.recipe_id} is a {r['family']} recipe, not {fam}")
    if args.peak_lr is not None and float(args.peak_lr) != float(r["peak_lr"]):
        raise LaunchRefused(f"--peak-lr {args.peak_lr} differs from recipe {r['recipe_id']}'s {r['peak_lr']!r}")
    return {"recipe_id": r["recipe_id"], "recipe_sha256": r["sha256"], "recipe_registry": str(path),
            "peak_lr": float(r["peak_lr"]), "dropout": list(r["dropout"]), "schedule": r["schedule"],
            "model_variant": r["model_variant"]}


def _theta0_for(args, schema_id: str, tuning: dict, seed: int):
    """TRAIN-02: the registered theta0 of (S*, torch build, variant, seed); required for a non-smoke TUNING launch."""
    import torch
    from v4train.theta0_variants import OUT_PATH, Theta0VariantsError, load_theta0_variants, lookup
    path = Path(args.theta0_variants) if getattr(args, "theta0_variants", None) else OUT_PATH
    try:
        obj = load_theta0_variants(path)
        h = lookup(obj, schema_id, torch.__version__, tuning["model_variant"], seed)
    except Theta0VariantsError as e:
        if args.smoke:
            return None
        raise LaunchRefused(f"theta0 refused: {e}") from e
    tuning["theta0_variants"] = {"path": str(path), "file_sha256": obj["_file_sha256"],
                                 "self_sha256": obj["self_sha256"]}
    return h


def _drop_smoke_weights(tr, reason: str) -> None:
    try:
        for kind in list(tr.ckpt.index):
            tr.ckpt.delete(kind, reason)
    except Exception:
        pass
    for q in tr.ckpt.dir.glob("*.pt*"):                       # a partial temp file of an interrupted save
        try:
            q.unlink()
        except OSError:
            pass


def _rate(clock: dict, lo: int, hi: int, exclude=()) -> Optional[float]:
    us = sorted(u for u in clock if lo <= u <= hi)
    if len(us) < 2:
        return None
    wall = clock[us[-1]] - clock[us[0]]
    n = us[-1] - us[0]
    for x in exclude:
        if x in clock and (x - 1) in clock and us[0] < x <= us[-1]:
            wall -= clock[x] - clock[x - 1]
            n -= 1
    return n / wall if wall > 0 else None


def _finish_smoke(args, tr, ep, run_dir, clock, t_train, why, launch, smoke_rows, write_json_atomic,
                  write_status_atomic, dev, say) -> int:
    import numpy as np
    import torch
    from v4int.cache import peak_rss_bytes, rss_bytes
    torch.cuda.synchronize(dev)
    train_wall = time.perf_counter() - t_train
    gpu_train = {"peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
                 "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev))}
    first = min(clock) if clock else None
    last = max(clock) if clock else None
    ce = int(args.checkpoint_every or 0)
    ck_updates = [u for u in clock if ce and u % ce == 0]
    metrics = []
    art = getattr(tr, "art", run_dir)                 # TRAIN-03: a sealed contest run keeps values in sealed/
    mp = art / "metrics.jsonl"
    if mp.exists():
        metrics = [json.loads(x) for x in mp.read_text(encoding="utf-8").splitlines() if x.strip()]
    mon = [m for m in metrics if m.get("kind") == "monitor" and m.get("attempt") == tr.attempt]
    timing = {"leg_first_update": first, "leg_last_update": last, "train_wall_s_incl_start": round(train_wall, 2),
              "updates_per_s_leg": _rate(clock, first or 0, last or 0),
              "updates_per_s_after_warmup_excl_checkpoints": _rate(clock, (first or 0) + 20, last or 0,
                                                                     exclude=ck_updates),
              "monitor_data_wait_s": round(sum(m.get("data_wait_s", 0.0) for m in mon), 3),
              "monitor_step_wall_s": round(sum(m.get("step_wall_s", 0.0) for m in mon), 3),
              "monitor_windows": len(mon), "checkpoint_updates": sorted(ck_updates),
              "loss_mean_by_window": [(m["update"], m.get("loss_mean")) for m in mon]}
    summ = {"task": "INT-01 smoke", "label": "SMOKE_NOT_EVIDENCE", "population_label": launch.get("population_label"),
            "manifests": launch.get("manifests"), "run_id": tr.cfg.run_id, "family": tr.family,
            "phase": tr.cfg.phase, "seed": tr.cfg.seed, "stop_reason": why, "attempt": tr.attempt,
            "config_sha256": tr.config_sha256, "init_sha256": tr.built.init_sha256,
            "trainer_code_sha256": launch.get("trainer_code_sha256"), "timing": timing,
            "determinism": launch.get("determinism"), "gpu_memory_train": gpu_train, "digests": state_digests(tr),
            "utc_stop": utc(), "disk_at_start": launch.get("disk_at_start"), "disk_at_stop": disk_record(run_dir)}
    interrupted = "SMOKE_INTERRUPT" in why
    if interrupted:
        summ["weights_retained_for_the_resume_leg"] = {"kinds": sorted(tr.ckpt.index),
                                                       "files": tr.ckpt.weight_files(),
                                                       "deleted_by": "the resume leg (B2) of the same smoke job, right "
                                                                     "after its digests are recorded"}
    else:
        deleted = {}                                         # weights go right after the digests (Operations disk rule)
        for kind in list(tr.ckpt.index):
            d = tr.ckpt.delete(kind, "INT-01 smoke: weights deleted right after the state digests were recorded")
            if d:
                deleted[kind] = d
        left = tr.ckpt.weight_files()
        summ["weights_deleted"] = {"deleted_sha256": deleted, "weight_files_left": left, "utc": utc(),
                                   "disk_after_delete": disk_record(run_dir)}
        if left:
            raise RuntimeError(f"smoke weights left behind: {left}")
        torch.cuda.reset_peak_memory_stats(dev)
        te = time.perf_counter()
        out = tr._evaluate(round(tr.state.exposures / tr.N, 6))      # the trainer's own frozen evaluation path
        torch.cuda.synchronize(dev)
        ev_wall = time.perf_counter() - te
        ranks = np.asarray(out.ranks, dtype=np.int64)
        res = out.result
        sha = lambda a: hashlib.sha256(np.ascontiguousarray(a, dtype="<i8").tobytes()).hexdigest()  # noqa: E731
        ev = {"manifest_id": ep.manifest_id, "eval_wall_s": round(ev_wall, 2),
              "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(dev)),
              "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(dev)),
              "eval_stream": res["eval_config"].get("eval_stream"), "eval_config": res["eval_config"],
              "ranks_sha256": sha(ranks), "SMOKE_value_E2E_macro_mrr@20": out.value,
              "secondary": {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in out.secondary.items()},
              "E2E_n_decisions": res["E2E"]["n_decisions"], "E2E_n_users": res["E2E"]["n_users"]}
        write_json_atomic(art / "SMOKE_EVAL.json", {"label": "SMOKE_NOT_EVIDENCE", "run_id": tr.cfg.run_id,
                                                        "rows": smoke_rows(out.rows), "summary": ev})
        summ["eval"] = ev
    summ["disk_at_end"] = disk_record(run_dir)
    summ["host_rss_bytes"] = rss_bytes()
    summ["host_peak_rss_bytes"] = peak_rss_bytes()
    name = "SMOKE_LEG_INTERRUPTED.json" if interrupted else "SMOKE_SUMMARY.json"
    write_json_atomic(art / name, summ)
    write_status_atomic(run_dir / "STATUS.json", {"run_id": tr.cfg.run_id, "status": (
        "SMOKE_INTERRUPTED_NOT_EVIDENCE" if interrupted else "SMOKE_DONE_NOT_EVIDENCE"), "label": "SMOKE_NOT_EVIDENCE",
        "updates": int(tr.state.updates), "utc": utc()})
    say(f"[launch] {tr.cfg.run_id}: {name} written ({why})")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        pre = preflight(args)
        env_rec = export_environment(int(args.gpu))
        return run(args, pre, env_rec)
    except LaunchRefused as e:
        print(f"LAUNCH REFUSED: {e}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
