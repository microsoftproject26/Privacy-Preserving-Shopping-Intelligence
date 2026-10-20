"""v4fs.theta0 — theta0 hash registration of derived schemas for the running torch build (SCHEMA-01, server: seconds).

  python -B -m v4fs.theta0 --schemas L0_CLEAN_V2 L3_NEAR_DUPLICATE --seeds 2026 2027 --v1-root <v1> --out <receipt>
         [--device cuda:0] [--merge-into contracts/derived_schemas/THETA0_REGISTRY.json]

For each (schema, family, seed): v4models.build(family, registered derived schema, seed, "cpu", v1_root=...) at the
full catalogue (the mapping files are sha256-checked against FILES.json; no dataset row is read). Records init_sha256
(SASRec: after the one C7 recentering, on CPU before any device move), raw_init_sha256, manifest digest, shared
numel. The L0 bitwise guard: the L0_CLEAN_V2 hashes must equal the build of the plain Clean V2 schema (same process),
and, when a registration exists for the running torch build (MODEL-01 inventory / FL-02 GPU receipt), that value.
With --device cuda:0 each model is then moved to the (single visible) GPU and one synthetic full-K forward through
the adapter (the channel projection included) is checked for shape and finiteness; the moved state digest must equal
the CPU theta0. No training, no optimizer, no weights written.
SCHEMA-01R1: the default schema list adds L0S / L3S (+ session_offset_tokens); the synthetic GPU batch carries a
synthetic session_offset_tokens key (ignored by L0 / L3 models). A re-run over L0 / L3 must reproduce their registered
values (theta0_obj refuses a changed value).
FS-02: `--merge-into` now goes through `v4fs.schemas.merge_theta0`, which is safe to run on the server (a short
exclusive lock file serializes a concurrent merge into the same THETA0_REGISTRY.json; the self-hash rule and the
refusal of a changed existing value are unchanged).
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import sys
import time
from pathlib import Path

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
import ppsi_env  # noqa: E402,F401

import torch  # noqa: E402

from fl.numerics import enforce_strict_fp32, state_digest  # noqa: E402
from v4models.build import build  # noqa: E402
from v4models.schema import load_schema  # noqa: E402
from v4models.synthetic import synthetic_examples_v2  # noqa: E402

from v4fs import schemas as S  # noqa: E402

FAMILIES = ("GRU", "SASREC")
INVENTORY_DIR = _CODE / "v4models" / "inventory"
FL02_GPU_RECEIPT = _CODE / "tests_fl" / "receipts" / "gpu" / "init.json"


def registered_l0(family: str, seed: int) -> tuple:
    """(init_sha256, raw_init_sha256, source) of the Clean V2 build registered for the RUNNING torch build: the
    MODEL-01 inventory (its own build) or the FL-02 GPU receipt init.json (torch 2.13.0+cu130); (None, None, why)."""
    v = torch.__version__
    rec = json.loads((INVENTORY_DIR / f"{family}_CLEAN_V2_MANIFEST_INVENTORY.json").read_text(encoding="utf-8"))
    if rec["environment"]["torch"] == v and str(seed) in rec["init_state"]:
        e = rec["init_state"][str(seed)]
        return e["init_sha256"], e["raw_init_sha256"], f"MODEL-01 inventory (torch {v})"
    if FL02_GPU_RECEIPT.is_file():
        r = json.loads(FL02_GPU_RECEIPT.read_text(encoding="utf-8"))
        e = r["families"].get(family, {}).get(str(seed))
        if r["env"]["torch"] == v and e:
            return e["init_sha256"], e["raw_init_sha256"], f"FL-02 GPU receipt init.json (torch {v})"
    return None, None, f"no registered Clean V2 theta0 for torch {v}"


def measure(schema_ids, seeds, v1_root: Path, device: str = "cpu", log=print) -> dict:
    enforce_strict_fp32()
    base = load_schema()
    utc = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  # noqa: E731
    out = {"task": "SCHEMA-01 theta0 registration", "torch": torch.__version__, "python": sys.version.split()[0],
           "host": platform.node(), "platform": platform.platform(), "device": device, "utc_start": utc(),
           "threads": torch.get_num_threads(), "records": [], "l0_guard": {}, "gpu_forward": []}
    if device != "cpu":
        out["cuda"] = {"available": torch.cuda.is_available(), "count": torch.cuda.device_count(),
                       "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    plain = {}
    for fam in FAMILIES:
        for seed in seeds:
            bm = build(fam, base, int(seed), "cpu", v1_root=v1_root)
            plain[(fam, int(seed))] = (bm.init_sha256, bm.raw_init_sha256, bm.manifest_digest)
            del bm
            gc.collect()
    for sid in schema_ids:
        ds = S.load_registered(sid)
        for fam in FAMILIES:
            for seed in seeds:
                t0 = time.time()
                bm = build(fam, ds, int(seed), "cpu", v1_root=v1_root)
                rec = {"schema_id": sid, "schema_self_sha256": ds.self_sha256, "torch": torch.__version__,
                       "family": fam, "seed": int(seed), "init_sha256": bm.init_sha256,
                       "raw_init_sha256": bm.raw_init_sha256, "manifest_digest": bm.manifest_digest,
                       "shared_numel": bm.manifest.shared_numel, "K": bm.K, "host": platform.node(), "utc": utc(),
                       "build_s": round(time.time() - t0, 2)}
                out["records"].append(rec)
                log(f"[theta0] {sid} {fam} seed {seed}: {bm.init_sha256} numel {rec['shared_numel']}")
                if ds.is_full:
                    p = plain[(fam, int(seed))]
                    g = {"equals_plain_clean_v2_build": (bm.init_sha256, bm.raw_init_sha256, bm.manifest_digest) == p}
                    want, want_raw, src = registered_l0(fam, int(seed))
                    g.update(registered=want, registered_raw=want_raw, registered_source=src,
                             equals_registered=(None if want is None else
                                                (bm.init_sha256, bm.raw_init_sha256) == (want, want_raw)))
                    out["l0_guard"][f"{fam}/{seed}"] = g
                if device != "cpu" and int(seed) == int(seeds[0]):
                    theta = bm.theta0()
                    m = bm.module.to(torch.device(device))
                    from v4models.build import make_adapter
                    a = make_adapter(m, ds)
                    moved = state_digest({k: v.cpu() for k, v in a.broadcast_state().items()})
                    m.eval()
                    b = synthetic_examples_v2(8, bm.K, base.widths, torch.Generator().manual_seed(7),
                                              vocab=bm.vocab_sizes, with_names=True, with_session_intent=True)
                    b = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
                    with torch.no_grad():
                        y = a.scores(b)
                    out["gpu_forward"].append({"schema_id": sid, "family": fam, "seed": int(seed),
                                               "moved_state_equals_theta0": moved == state_digest(theta),
                                               "logits_shape": list(y.shape),
                                               "finite": bool(torch.isfinite(y).all())})
                    del theta, m, a, y, b
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                del bm
                gc.collect()
    out["utc_end"] = utc()
    ok = all(g["equals_plain_clean_v2_build"] and g.get("equals_registered") in (True, None)
             for g in out["l0_guard"].values())
    ok = ok and all(r["moved_state_equals_theta0"] and r["finite"] and r["logits_shape"][1] == 158486
                    for r in out["gpu_forward"])
    out["ok"] = bool(ok)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemas", nargs="+", default=list(S.WRITTEN_SCHEMAS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027])
    ap.add_argument("--v1-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--merge-into", default=None, help="THETA0_REGISTRY.json to create / extend, local or server "
                    "(FS-02: v4fs.schemas.merge_theta0, lock-guarded against a concurrent merge into the same file)")
    args = ap.parse_args(argv)
    torch.set_num_threads(int(args.threads))
    if args.device != "cpu":
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise SystemExit("expected exactly one visible CUDA device (export CUDA_VISIBLE_DEVICES)")
    res = measure(args.schemas, args.seeds, Path(args.v1_root), args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    S.write_file(Path(args.out), json.dumps(res, indent=1, sort_keys=True, allow_nan=False) + "\n")
    if args.merge_into and res["ok"]:
        S.merge_theta0(Path(args.merge_into), res["records"])
    print(json.dumps({"ok": res["ok"], "torch": res["torch"], "l0_guard": res["l0_guard"],
                      "n_records": len(res["records"]), "gpu_forward": res["gpu_forward"]}, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
