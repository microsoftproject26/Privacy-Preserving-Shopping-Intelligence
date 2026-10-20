"""v4int.p2_probe — PREG2 §3.2 criterion (b) (functional no-op) of a warm-FL run on the CALIB_TRAIN probe (P2-01).

  python -B code/v4int/p2_probe.py --run-dir <FL run dir> --theta-pre <PRE endpoint_6.0.pt> --gpu 0 \
         --history-policy USER_HISTORY_L50 --out <NEW P2_PROBE.json> [--chunk 128] [--allow-shared-gpu]

Value-free (inputs only, no labels): 4,096 CALIB_TRAIN query rows chosen by `probe_positions` (numpy
default_rng(20261004).choice over the manifest's canonical positions, without replacement, sorted) are served by
v4int.providers.StoreTrainProvider (the registered CALIBRATION train manifest of DATA_PROTOCOL_LOCK_V2's registry; any
other manifest — MAIN / VALIDATION / TEST — is refused) and scored with adapter.scores (eval mode, no grad) by two copies
of the SMALL-recipe model: theta_pre (the PRE run's v4train endpoint_6.0 `model`) and theta (the run's best.pt /
endpoint_6.0.pt `theta`, digest-verified). The target column is never read: `strip_labels` drops target_class /
loss_mask / decision ids before the batch reaches the model. Per row: top-20 Jaccard, |s_theta[i*] - s_pre[i*]| (i* =
argmax s_pre) and the top-1 / top-2 gap of theta_pre (fl.update_log.functional_rows), aggregated by
fl.update_log.functional_noop_from_rows. The output (self-sha256, written exclusively) feeds tools/p2_noop_check.py
--probe. Cost: about 0.1 GPU-h per probed checkpoint pair (INFERENCE; plan P2-01). Server-only (real data + the model).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Mapping

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
sys.dont_write_bytecode = True

FORMAT = "PPSI_V4_P2_PROBE_1"
PROBE_ROLE = ("COHORT", "CALIBRATION")                    # v4train.provider.expected_manifests(population, role)
LABEL_FIELDS = ("target_class", "loss_mask", "decision_id")
CHECKPOINTS = {"PRACTICAL_BEST": "best.pt", "CONTROLLED_BUDGET": "endpoint_6.0.pt"}


class ProbeRefused(RuntimeError):
    pass


def probe_positions(n_rows: int, *, rows: int = None, seed: int = None):
    """The registered probe: `rows` (4,096) distinct canonical positions of [0, n_rows), numpy default_rng(seed =
    20261004).choice without replacement, sorted ascending (a pure function of n_rows)."""
    import numpy as np
    from fl.update_log import PROBE_ROWS, PROBE_SEED
    rows = PROBE_ROWS if rows is None else int(rows)
    seed = PROBE_SEED if seed is None else int(seed)
    if n_rows < rows:
        raise ProbeRefused(f"the probe needs {rows} rows; the manifest has {n_rows}")
    return np.sort(np.random.default_rng(seed).choice(int(n_rows), size=rows, replace=False)).astype(np.int64)


def strip_labels(batch: Mapping) -> dict:
    """The query part of a provider batch: every label / id field removed (criterion (b) uses inputs only)."""
    return {k: v for k, v in batch.items() if k not in LABEL_FIELDS}


def score_chunks(adapter_a, adapter_b, batches, k: int = None) -> dict:
    """Per-row (b) statistics over an iterable of label-free batches: both adapters in eval mode under no_grad (the
    models' dropout is off: no RNG is drawn); returns the concatenated fl.update_log.functional_rows tensors."""
    import torch
    from fl.update_log import NOOP_TOPK, functional_rows
    k = NOOP_TOPK if k is None else int(k)
    out = {"jaccard": [], "dscore": [], "gap": []}
    for ad in (adapter_a, adapter_b):
        ad.module.eval()
    with torch.no_grad():
        for b in batches:
            r = functional_rows(adapter_a.scores(b), adapter_b.scores(b), k)
            for f in out:
                out[f].append(r[f])
    return {f: torch.cat(v) for f, v in out.items()}


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _canon(obj: Mapping) -> str:
    body = {k: v for k, v in obj.items() if k != "self_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def parse_args(argv=None):
    from v4hist.policy import HISTORY_POLICIES
    ap = argparse.ArgumentParser(description="PREG2 §3.2 (b): functional no-op probe on CALIB_TRAIN (value-free)")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--theta-pre", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--history-policy", required=True, choices=HISTORY_POLICIES)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--allow-shared-gpu", action="store_true")
    ap.add_argument("--small-recipe", type=Path, default=None)
    ap.add_argument("--manifest-dir", type=Path, default=None)
    ap.add_argument("--lock-path", type=Path, default=None)
    ap.add_argument("--manifest-hashes", type=Path, default=None)
    ap.add_argument("--header-list", type=Path, default=None)
    ap.add_argument("--transform-state", type=Path, default=None)
    return ap.parse_args(argv)


def run(args, say: Callable = print) -> dict:
    """Server path: data + two models on one GPU (see the module docstring)."""
    if args.out.exists():
        raise ProbeRefused(f"{args.out} exists (the probe record is written once)")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(args.gpu))
    import v4train  # noqa: F401  (ppsi_env first)
    import torch
    from fl.checkpoint import load_checkpoint
    from fl.numerics import state_digest
    from fl.update_log import NOOP_TOPK, PROBE_ROWS, PROBE_SEED, functional_noop_from_rows
    from v4data.common import dataset_root
    from v4int.launch_central import (bind_data_protocol_lock_v2, build_trainer_registry_view, data_paths,
                                      load_production_schema)
    from v4int.launch_fl import FAMILY, _small_recipe
    from v4int.providers import StoreTrainProvider
    from v4models import build
    from v4train.provider import expected_manifests
    if not torch.cuda.is_available():
        raise ProbeRefused("the probe runs on one visible CUDA device")
    dev = "cuda:0"
    spec_rec = json.loads((Path(args.run_dir) / "ARM_SPEC.json").read_text(encoding="utf-8"))
    recipe = _small_recipe(argparse.Namespace(small_recipe=args.small_recipe, smoke=False))
    paths = data_paths(args)
    lockctx = bind_data_protocol_lock_v2(paths, smoke=False)
    view = build_trainer_registry_view(lockctx["manifest_registry_path"], Path(args.out).parent / "_probe_view",
                                       registry_rel_path=lockctx["manifest_registry_rel_path"],
                                       registry_file_sha256=lockctx["manifest_registry_file_sha256"])
    train_id, _eval_id = expected_manifests(*PROBE_ROLE, registry=str(view["path"]))
    if "CALIB_TRAIN" not in str(train_id):
        raise ProbeRefused(f"the probe manifest {train_id} is not the CALIB_TRAIN manifest")
    tp = StoreTrainProvider(train_id, history_policy=args.history_policy, paths=paths, log=say,
                            manifest_registry=lockctx["manifest_registry_path"], registry_root=V4)
    pos = probe_positions(int(tp.n_decisions))
    schema = load_production_schema(lockctx["schema_id"], lockctx["schema_self_sha256"], lockctx["schema_registry_path"])

    def model():
        return build(FAMILY, schema, int(spec_rec["arm_spec"]["cfg"]["seed"]), dev, v1_root=dataset_root(),
                     variant=recipe["model_variant"], dropout=tuple(recipe["dropout"]))

    pre_obj = torch.load(args.theta_pre, map_location="cpu", weights_only=True)
    bm_pre = model()
    bm_pre.module.load_state_dict(pre_obj["model"], strict=True)
    pre_digest = state_digest(bm_pre.adapter.broadcast_state())
    want = (spec_rec["arm_spec"].get("init_from") or {}).get("state_digest")
    if want is not None and pre_digest != want:
        raise ProbeRefused(f"theta_pre digest {pre_digest[:12]}… != the run's init_from {want[:12]}…")
    ckdirs = [p for p in (Path(args.run_dir) / "ckpt").iterdir() if p.is_dir()]
    if len(ckdirs) != 1:
        raise ProbeRefused(f"{args.run_dir}/ckpt must hold exactly one checkpoint directory")
    batches = [strip_labels({k: (v.to(dev) if torch.is_tensor(v) and k != "lengths" else v)
                             for k, v in tp.batch(pos[i:i + int(args.chunk)]).items()})
               for i in range(0, pos.size, int(args.chunk))]
    points = {}
    for rule, f in CHECKPOINTS.items():
        p = ckdirs[0] / f
        payload = load_checkpoint(p)
        bm = model()
        bm.adapter.load_state_(payload["theta"])
        d = state_digest(bm.adapter.broadcast_state())
        if d != state_digest(payload["theta"]):
            raise ProbeRefused(f"{p}: the loaded theta does not reproduce its own digest")
        rows = score_chunks(bm.adapter, bm_pre.adapter, batches)
        points[rule] = dict(functional_noop_from_rows(rows, NOOP_TOPK), checkpoint=str(p), checkpoint_sha256=_sha(p),
                            state_digest=d)
        del bm
        torch.cuda.empty_cache()
    rec = {"format": FORMAT, "run_dir": str(args.run_dir), "run_id": spec_rec.get("run_id"),
           "arm_spec_sha256": spec_rec.get("arm_spec_sha256"), "probe_manifest": train_id,
           "probe_rows": PROBE_ROWS, "probe_seed": PROBE_SEED, "probe_positions_sha256":
               hashlib.sha256(pos.tobytes()).hexdigest(), "train_provider_fingerprint": tp.fingerprint,
           "theta_pre": str(args.theta_pre), "theta_pre_sha256": _sha(args.theta_pre),
           "theta_pre_state_digest": pre_digest, "points": points,
           "labels_used": False, "label": "DESCRIPTIVE (PREG2 §3.2 (b); inputs only)"}
    rec["self_sha256"] = _canon(rec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(rec, indent=1, sort_keys=True, allow_nan=False) + "\n")
    verdict_text = ", ".join(f"{r}: holds={p['holds']}" for r, p in points.items())
    say(f"[p2_probe] {args.out} written ({verdict_text})")
    return rec


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except ProbeRefused as e:
        print(f"PROBE REFUSED: {e}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
