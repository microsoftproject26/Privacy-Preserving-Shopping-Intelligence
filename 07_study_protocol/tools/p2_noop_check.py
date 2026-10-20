"""tools/p2_noop_check.py — PREG2 §3.2 criterion (a) (parameter no-op) of one warm-FL run, from checkpoints only (P2-01).

  python -B tools/p2_noop_check.py --run-dir <FL run dir> --theta-pre <PRE endpoint_6.0.pt> --out <NEW json>
         [--probe <P2_PROBE.json of v4int/p2_probe.py>] [--calib-value v --pre-calib v0] [--code-root <V4>/code]

Value-free: reads theta_pre (the PRE run's v4train endpoint_6.0 checkpoint, `model`), the run's best.pt and
endpoint_6.0.pt (fl.checkpoint payloads, `theta`, digest-verified by fl.checkpoint.load_checkpoint) and the run's
ARM_SPEC.json; never a row, a rank or a metric. The trained subset T = the run's uploaded keys: RUN_SUMMARY.json
`update_norm_log.trained_keys` when the run wrote it (Phase-2 runs), else every floating-point key of theta_pre minus the
ARM_SPEC frozen item tables (`frozen_item_tables`); the keys left out are reported (`excluded_keys`; a float buffer of
the family, if any, would stay in T and only enlarge the denominator — disclosed). The frozen tables must be
bit-identical to theta_pre (checked and reported).
Criterion (b) (functional) is v4int/p2_probe.py's output (`--probe`); with it, the run is classified NO_OP / MOVED /
DRIFT / INCOMPLETE (fl.update_log.classify_noop; DRIFT needs --calib-value and --pre-calib, the run's CALIB value at
PRACTICAL_BEST and PRE_CALIB, given by Ops from the selection record — this tool reads no value itself).
Ops' D1 script (DL 407), if it exists on the server, is the D1 reading; this tool is the registered Phase-2 check.
The output file is written exclusively (an existing path is refused).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
import ppsi_paths

V4 = ppsi_paths.PROJECT_ROOT.resolve()   # the ppsi_v4 workspace
CHECKPOINTS = {"PRACTICAL_BEST": "best.pt", "CONTROLLED_BUDGET": "endpoint_6.0.pt"}
FORMAT = "PPSI_V4_P2_NOOP_CHECK_1"


class NoopCheckRefused(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_theta_pre(path: Path) -> dict:
    """theta_pre from the PRE run's v4train weights checkpoint (`model`) or from an fl payload (`theta`)."""
    import torch
    obj = torch.load(Path(path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and isinstance(obj.get("model"), dict):
        return dict(obj["model"])
    if isinstance(obj, dict) and isinstance(obj.get("theta"), dict):
        return dict(obj["theta"])
    raise NoopCheckRefused(f"{path}: neither a v4train weights checkpoint (model) nor an fl payload (theta)")


def find_checkpoints(run_dir: Path) -> dict:
    """{rule: path} of best.pt / endpoint_6.0.pt under <run_dir>/ckpt/<run_id>/ (exactly one checkpoint directory)."""
    dirs = [p for p in (Path(run_dir) / "ckpt").iterdir() if p.is_dir()] if (Path(run_dir) / "ckpt").is_dir() else []
    if len(dirs) != 1:
        raise NoopCheckRefused(f"{run_dir}/ckpt must hold exactly one checkpoint directory, found {len(dirs)}")
    out = {r: dirs[0] / f for r, f in CHECKPOINTS.items()}
    missing = [str(p) for p in out.values() if not p.is_file()]
    if missing:
        raise NoopCheckRefused(f"missing checkpoint(s): {missing}")
    return out


def trained_keys(run_dir: Path, theta_pre: dict, thetas: dict, frozen: list) -> tuple:
    """(T, source, excluded): see the module docstring."""
    import torch
    rs = Path(run_dir) / "RUN_SUMMARY.json"
    if rs.is_file():
        un = _json(rs).get("update_norm_log") or {}
        if un.get("trained_keys"):
            keys = list(un["trained_keys"])
            bad = [k for k in keys if k in frozen or k not in theta_pre]
            if bad:
                raise NoopCheckRefused(f"RUN_SUMMARY trained_keys contain frozen / unknown keys {bad[:5]}")
            return keys, "RUN_SUMMARY.update_norm_log.trained_keys", []
    keys, excluded = [], []
    for k, v in theta_pre.items():
        if k in frozen or not torch.is_tensor(v) or not v.is_floating_point():
            excluded.append(k)                             # frozen tables / integer buffers: never in T
            continue
        if any(k not in t for t in thetas.values()):
            raise NoopCheckRefused(f"theta_pre key {k} is absent from a run checkpoint")
        keys.append(k)
    return keys, "theta_pre float keys minus ARM_SPEC frozen_item_tables (fallback: no trained_keys record)", excluded


def check(run_dir: Path, theta_pre_path: Path, *, probe: dict = None, calib_value=None, pre_calib=None) -> dict:
    import torch
    from fl.checkpoint import load_checkpoint
    from fl.numerics import state_digest
    from fl.update_log import classify_noop, param_noop_criterion
    spec_rec = _json(Path(run_dir) / "ARM_SPEC.json")
    spec = spec_rec.get("arm_spec") or {}
    frozen = list(spec.get("frozen_item_tables") or [])
    theta_pre = load_theta_pre(theta_pre_path)
    init = (spec.get("init_from") or {}).get("state_digest")
    pre_digest = state_digest({k: v for k, v in theta_pre.items() if torch.is_tensor(v)})
    paths = find_checkpoints(run_dir)
    thetas = {}
    for rule, p in paths.items():
        payload = load_checkpoint(p)
        if payload.get("kind") != "weights":
            raise NoopCheckRefused(f"{p}: kind {payload.get('kind')!r} is not an fl weights payload")
        thetas[rule] = dict(payload["theta"])
    keys, source, excluded = trained_keys(run_dir, theta_pre, thetas, frozen)
    frozen_ok = {k: all(torch.equal(t[k], theta_pre[k]) for t in thetas.values()) for k in frozen}
    points = {}
    for rule, th in thetas.items():
        a = param_noop_criterion(th, theta_pre, keys)
        b = None if probe is None else (probe.get("points") or {}).get(rule)
        points[rule] = {"a": a, "b": b, "checkpoint": str(paths[rule]), "checkpoint_sha256": _sha(paths[rule]),
                        "state_digest": state_digest(th)}
    verdict = classify_noop({r: {"a": p["a"], "b": p["b"]} for r, p in points.items()}, calib_value=calib_value,
                            pre_calib=pre_calib)
    return {"format": FORMAT, "run_dir": str(run_dir), "arm": spec_rec.get("arm"), "run_id": spec_rec.get("run_id"),
            "arm_spec_sha256": spec_rec.get("arm_spec_sha256"), "theta_pre": str(theta_pre_path),
            "theta_pre_sha256": _sha(theta_pre_path), "theta_pre_state_digest": pre_digest,
            "theta_pre_matches_init_from": None if init is None else init == pre_digest,
            "trained_keys_source": source, "trained_keys": keys, "excluded_keys": excluded,
            "frozen_item_tables": frozen, "frozen_tables_bit_identical": frozen_ok,
            "points": points, "classification": verdict,
            "probe_sha256": None if probe is None else probe.get("self_sha256"),
            "label": "DESCRIPTIVE (PREG2 §3.2; value-free: checkpoints only)"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PREG2 §3.2 (a) no-op check of one warm-FL run")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--theta-pre", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--probe", type=Path, default=None)
    ap.add_argument("--calib-value", type=float, default=None)
    ap.add_argument("--pre-calib", type=float, default=None)
    ap.add_argument("--code-root", type=Path, default=V4 / "code")
    a = ap.parse_args(argv)
    sys.dont_write_bytecode = True
    if str(a.code_root) not in sys.path:
        sys.path.insert(0, str(a.code_root))
    if (a.calib_value is None) != (a.pre_calib is None):
        print("REFUSED: --calib-value and --pre-calib go together", file=sys.stderr)
        return 2
    try:
        probe = None if a.probe is None else _json(a.probe)
        rec = check(a.run_dir, a.theta_pre, probe=probe, calib_value=a.calib_value, pre_calib=a.pre_calib)
    except (NoopCheckRefused, OSError, KeyError, ValueError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    import os
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(a.out, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(rec, indent=1, sort_keys=True, allow_nan=False) + "\n")
    print(f"[p2_noop_check] {a.out}: {rec['classification']['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
