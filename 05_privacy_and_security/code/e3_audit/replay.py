"""Replay the first 50 CALIB rounds without compressing the full FL schedule.

Inputs bind the actual completed PB100 tau-fixed recipe and verified pretrain.
The frozen counterpart differs only in the item-table freeze and item LR group.
There is no real TEST/MAIN evaluation, smoke path, DP mode or resume option.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

from .capture import UploadCapture, canonical_sha, declare_population, write_json


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def code_inventory(code):
    code = Path(code)
    return {str(p.relative_to(code)): sha_file(p) for p in sorted(code.rglob("*.py"))
            if "__pycache__" not in p.parts}


def normalize_isolated_path(value, *, review=None):
    """Map launcher-owned physical/virtual aliases to this execution's copy root.

    This uses the explicit EID launcher environment only, then the ordinary
    resolved-path review guard still applies. Arbitrary outside paths are never
    remapped into an allowed output directory.
    """
    path = Path(value).expanduser()
    review = Path(review) if review is not None else Path(__file__).resolve().parents[3]
    project_root = os.environ.get("EID_PROJECT_ROOT")
    if os.environ.get("EID_ISOLATED") == "1" and project_root:
        physical = Path(project_root).parent
        aliases = (physical, Path("/home/jovyan/ahmedabdelhamed01"), Path("/home/jovyan/ahmed"))
        execution_root = review.parent
        if execution_root not in aliases:
            raise ValueError("EID execution root is outside the launcher's declared aliases")
        for alias in aliases:
            if path.is_absolute() and path.is_relative_to(alias):
                path = execution_root / path.relative_to(alias)
                break
    return path.resolve()


def build_audit_spec(source_record, *, frozen=False, n_clients=32768):
    """Copy donor numerical recipe, replace only CALIB size/role and freeze flag."""
    spec = copy.deepcopy(source_record["arm_spec"])
    cfg = spec["cfg"]
    recipe = spec["p2_recipe"]
    if spec.get("arm") != "P2_W" or recipe.get("items") != "unfrozen" or recipe.get("upload") != "dense":
        raise ValueError("E3 donor must be a dense unfrozen P2 warm recipe")
    if cfg.get("dp") is not None or cfg.get("pf") is not None or cfg.get("method") != "FA":
        raise ValueError("E3 donor refuses DP or personal state")
    if cfg.get("server_opt", {}).get("tau") != 1e-6 or cfg.get("solver", {}).get("weight_decay") != 0:
        raise ValueError("E3 requires tau-fixed recipe with zero weight decay for the support attack")
    if spec.get("upload_codec") is not None or recipe.get("lora_rank") or recipe.get("personal_head"):
        raise ValueError("E3 does not capture encoded/LoRA/personal uploads")
    if spec.get("plan") != {"group_size": 64, "sampling": "sweep"} or cfg.get("dropout_p") != 0.1:
        raise ValueError("E3 requires the real64-client deterministic sweep/dropout recipe")
    if int(cfg["seed"]) != 2026:
        raise ValueError("E3 CALIB replay is bound to seed2026")
    passes = int(cfg["solver"]["passes"])
    efe = float(cfg["endpoint_efe"])
    rounds = math.ceil(10 * efe * n_clients / (576 * passes))
    cfg["endpoint_rounds"] = rounds
    spec.update(role="CALIBRATION", max_rounds=rounds, audit_label="POST_HOC_EXPLORATORY")
    spec["exposure_budget"].update(n_clients=n_clients, rounds=rounds)
    if frozen:
        spec["frozen_item_tables"] = ["item_embed.weight", "output_embed", "output_bias"]
        cfg["solver"].pop("item_lr_mult", None)
        recipe.update(items="frozen", item_lr_mult=None)
        spec["p2_label"] = spec["p2_label"].split("|items=")[0]
    else:
        spec.pop("frozen_item_tables", None)
    return spec


def validate_args(args):
    if args.role != "CALIBRATION" or args.resume or args.dp or args.personal_state:
        raise ValueError("E3 admits CALIBRATION only and refuses DP, personal state or resume")
    if args.rounds != 50 or args.sample_size != 512:
        raise ValueError("E3 real-data audit is bound to50 rounds/512 users")
    if not 1 <= args.threads <= 96 or args.gpu not in (0,1):
        raise ValueError("E3 needs threads1..96 and GPU0 or1")
    for name in ("output", "source_spec", "init_record", "data_root"):
        if hasattr(args, name):
            setattr(args, name, normalize_isolated_path(getattr(args, name)))
    output = args.output
    review = Path(__file__).resolve().parents[3]
    if not output.is_relative_to(review):
        raise ValueError("E3 writes must stay in the unfrozen review workspace")
    if output.exists() and any(output.iterdir()):
        raise ValueError("E3 output directory must be empty (no resume/overwrite)")
    return output


def build_inputs(args, output):
    """Read and hash TRAIN-only bindings before the first local training step."""
    from dataclasses import replace
    import torch
    from fl.client import ClientData
    from fl.numerics import enforce_strict_fp32, numerics_receipt
    from fl.participation import ParticipationPlan
    from fl.runtime import FLRun, build_lr_table, freeze_item_tables
    from fl.server import Server
    from v4data.common import dataset_root
    from v4int import launch_fl as F
    from v4int.launch_central import bind_data_protocol_lock_v2, data_paths, load_production_schema
    from v4int.providers import StoreTrainProvider
    from v4models import build
    from v4train.provider import expected_manifests
    from .analysis import PROTOCOL

    import ppsi_env
    snapshot = ppsi_env.verify_snapshot()
    ppsi_env.assert_isolated()

    torch.set_num_threads(args.threads)
    enforce_strict_fp32(warn_only=False)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("E3 replay needs exactly one isolated visible CUDA device")
    torch.cuda.set_device(0)
    source = json.loads(Path(args.source_spec).read_text())
    if source.get("arm_spec_sha256") != F.spec_digest(source.get("arm_spec", {})):
        raise ValueError("donor ARM_SPEC content differs from its registered source hash")
    spec = build_audit_spec(source, frozen=args.variant == "frozen")
    paths = data_paths(args)
    # The verified tables are read from the server copy; code/contracts remain in
    # the unfrozen work root. The handover intentionally did not ship big tables.
    root = Path(args.data_root).resolve()
    paths = replace(paths, manifest_dir=root / "data_v3" / "manifests",
                    transform_state_path=root / "data_v2" / "TRANSFORM_STATE_CLEAN_V2.json")
    lockctx = bind_data_protocol_lock_v2(paths, smoke=False)
    reg = json.loads(Path(lockctx["manifest_registry_path"]).read_text())
    train_id = reg["roles"]["CALIBRATION"]["train"]
    if "CALIB" not in train_id.upper() or "TEST" in train_id.upper():
        raise ValueError("E3 locked registry does not identify CALIB TRAIN")
    provider = StoreTrainProvider(train_id, history_policy="USER_HISTORY_L50", paths=paths,
                                  manifest_registry=lockctx["manifest_registry_path"], registry_root=root,
                                  n_threads=args.threads, log=lambda *msg: print(*msg, flush=True))
    keys, positions = F.client_index(provider, paths.manifest_dir)
    print(f"[E3] verified CALIB TRAIN provider: {len(keys)} clients, {provider.n_decisions} rows", flush=True)
    if len(keys) != 32768:
        raise ValueError("E3 real CALIB manifest must contain32768 distinct clients")
    n_valid = {k: int(v.size) for k, v in positions.items()}
    cfg = F.fl_run_config(spec, "E3_CALIB_" + args.variant.upper() + "_s2026", ckpt_every=10**9)
    plan = ParticipationPlan.build(keys, seed=cfg.seed, manifest_hash=provider.fingerprint,
                                   group_size=64, sampling="sweep")
    population = declare_population(plan, cfg, rounds=args.rounds, sample_size=args.sample_size,
                                    seed=args.audit_seed, holdout_rounds=15)
    schema = load_production_schema(lockctx["schema_id"], lockctx["schema_self_sha256"],
                                     lockctx["schema_registry_path"])
    small = F._small_recipe(argparse.Namespace(small_recipe=None, smoke=False))
    init_state, identity, init_record = F.load_pretrain_init(Path(args.init_record), model_variant=small["model_variant"])
    if identity != spec["init_from"]:
        raise ValueError("bound donor recipe and init checkpoint identities differ")
    server_adapter = build(F.FAMILY, schema, 2026, "cuda:0", v1_root=dataset_root(),
                           variant=small["model_variant"], dropout=tuple(small["dropout"])).adapter
    init_sha = F.apply_init_state(server_adapter, init_state, identity["state_digest"])
    worker = build(F.FAMILY, schema, 2026, "cuda:0", v1_root=dataset_root(),
                   variant=small["model_variant"], dropout=tuple(small["dropout"])).adapter
    freeze_record = None
    if args.variant == "frozen":
        freeze_record = freeze_item_tables(server_adapter)
        freeze_item_tables(worker, source=server_adapter.broadcast_state())
    server = Server(server_adapter, init_sha256=init_sha)
    get_client = lambda k: ClientData(k, provider.batch(positions[k]))
    binding = {"format": "PPSI_E3_REPLAY_BINDING_1", "label": "POST_HOC_EXPLORATORY", "role": "CALIBRATION",
               "variant": args.variant, "rounds_replayed": args.rounds, "full_endpoint_rounds": cfg.endpoint_rounds,
               "attack_protocol_sha256": canonical_sha(PROTOCOL),
               "schedule_compressed": False, "source_spec_sha256": sha_file(args.source_spec),
               "source_spec": str(args.source_spec), "audit_spec": spec, "audit_spec_sha256": canonical_sha(spec),
               "init": init_record, "provider": provider.describe(), "schema_self_sha256": lockctx["schema_self_sha256"],
               "population_sha256": population["self_sha256"], "freeze_record": freeze_record,
               "lr_table_digest": build_lr_table(plan, n_valid, provider.n_decisions, cfg)["digest"],
               "numerical_code": code_inventory(Path(__file__).resolve().parents[1]),
               "numerical_code_scope": "complete_work_Python_tree_including_snapshot_and_tests",
               "snapshot": {"manifest_sha256": sha_file(ppsi_env.SNAP / "SNAPSHOT_MANIFEST.json"),
                            "verified_files": len(snapshot["files"]), "all_declared_files_verified": True},
               "known_catalogue_count": int(server_adapter.K),
               "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_name": torch.cuda.get_device_name(0)}
    binding["numerics"] = numerics_receipt()
    binding["self_sha256"] = canonical_sha(binding)
    write_json(output / "POPULATION.json", population)
    write_json(output / "REPLAY_BINDING.json", binding)
    return dict(spec=spec, cfg=cfg, plan=plan, server=server, workers=[worker], get_client=get_client,
                provider=provider, positions=positions, n_valid=n_valid, population=population, binding=binding)


def run_replay(args, *, truth_factory, analyse, secagg_factory=None):
    """Execute + analyze inside one guaranteed capture-cleanup lifetime."""
    from fl.runtime import FLRun
    output = validate_args(args)
    output.mkdir(parents=True, exist_ok=True)
    from .analysis import PROTOCOL
    write_json(output / "ATTACK_PROTOCOL.json", dict(PROTOCOL, self_sha256=canonical_sha(PROTOCOL)))
    start = time.monotonic()
    state = build_inputs(args, output)
    secagg_fn, projector = secagg_factory(state) if secagg_factory is not None else (None, None)
    if secagg_factory is not None:
        if "secagg_binding" not in state:
            raise ValueError("SecAgg factory did not provide its pretraining coordinate/protocol binding")
        crypto_binding = state["secagg_binding"]
        crypto_body = {k:v for k,v in crypto_binding.items() if k!="self_sha256"}
        crypto_sha = canonical_sha(crypto_body)
        if crypto_binding.get("self_sha256", crypto_sha) != crypto_sha:
            raise ValueError("SecAgg factory coordinate/protocol binding hash does not verify")
        write_json(output / "SECAGG_BINDING.json", dict(crypto_body, self_sha256=crypto_sha))
    if getattr(args, "prepare_only", False):
        result = {"status": "PREPARED_NOT_RUN", "binding_sha256": state["binding"]["self_sha256"],
                  "population_sha256": state["population"]["self_sha256"], "role": "CALIBRATION"}
        write_json(output / "PREPARED_NOT_RUN.json", result)
        return result
    truth_fn = truth_factory(state["provider"], state["positions"], state["binding"])
    with UploadCapture(parent=output, population=state["population"], truth_fn=truth_fn,
                       manifest=state["server"].manifest, secagg_fn=secagg_fn, secagg_projector=projector) as capture:
        run = FLRun(state["cfg"], state["plan"], state["server"], state["get_client"],
                    state["provider"].n_decisions, workers=state["workers"], n_valid_by_key=state["n_valid"],
                    upload_observer=capture)
        if state["binding"]["lr_table_digest"] != run.lr_table["digest"]:
            raise ValueError("run LR table differs from its pretraining binding")
        write_json(output / "LR_TABLE_BINDING.json", {"digest": run.lr_table["digest"],
                   "full_n_rounds": run.lr_table["n_rounds"], "replayed_rows": run.lr_table["rows"][:args.rounds]})
        for r in range(args.rounds):
            rec = run.step()
            with (output / "ROUND_LOG.jsonl").open("a") as f:
                f.write(json.dumps({"round": r, "lr": rec["lr"], "efe": rec["efe"], "weight": rec["weight"],
                                    "state_digest": run.server.digest(), "elapsed_s": time.monotonic()-start}) + "\n")
            print(f"[E3] {args.variant} round{r+1}/{args.rounds}, captures={len(capture.record_paths)}", flush=True)
        complete = capture.verify_complete()
        result = analyse(capture, state["binding"], output)
        replay = {"label": "POST_HOC_EXPLORATORY", "status": "CALIB_REPLAY_COMPLETE", "capture": complete,
                  "replay_elapsed_s": time.monotonic()-start, "summary": run.summary(),
                  "final_state_digest": run.server.digest(), "result": result}
        write_json(output / "REPLAY_SUMMARY.json", replay)
    return replay


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-spec", required=True, type=Path)
    ap.add_argument("--init-record", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--variant", choices=("unfrozen", "frozen"), required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--role", default="CALIBRATION")
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--sample-size", type=int, default=512)
    ap.add_argument("--audit-seed", type=int, default=20261013)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dp", action="store_true")
    ap.add_argument("--personal-state", action="store_true")
    ap.add_argument("--prepare-only", action="store_true", help="bind plan/init/code without training")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # This file imports torch only through capture; launcher must export before
    # Python starts, and the check is explicit to prevent an already imported CUDA.
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        raise ValueError("export CUDA_VISIBLE_DEVICES=<gpu> before starting the E3 Python process")
    from .analysis import ground_truth, analyze_capture
    from .secagg_features import secagg_factory
    def truth_factory(provider, positions, binding):
        cache = {}
        def truth(key):
            if key not in cache:
                cache[key] = ground_truth(provider.batch(positions[key]))
            return cache[key]
        return truth
    def analyze(capture, binding, output):
        result = analyze_capture(capture, catalogue_known_count=binding["known_catalogue_count"])
        write_json(output / "ANALYSIS_RESULT.json", result)
        return result
    run_replay(args, truth_factory=truth_factory, analyse=analyze, secagg_factory=secagg_factory)


if __name__ == "__main__":
    main()
