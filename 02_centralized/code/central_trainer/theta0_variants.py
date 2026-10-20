"""v4train.theta0_variants — theta0 per model variant x seed (TRAIN-02 item 0b; PREG1_AMENDMENT_02R1 §1). Server, CPU.

  python -B -m v4train.theta0_variants --v1-root <rees46_nextitem_v1> --out <receipt.json> \
         --write contracts/model_variants/THETA0_VARIANTS.json [--schema <id>] [--seeds 2026 2027] [--variants ...]

For each (variant, seed) of the MODEL_VARIANT_REGISTRY (verified against v4models.variants): v4models.build(family,
the registered derived schema, seed, "cpu", v1_root=..., variant=...) at the full catalogue, TWICE in the same process
(theta0 determinism: both builds must give the same init_sha256, raw_init_sha256 and manifest digest). SASRec theta0 is
the C7-recentered init (v4models.build recenters once on CPU). Recorded per build: init_sha256, raw_init_sha256,
manifest_digest, shared_numel, K, host, utc. Per variant (the inventory): the parameter inventory by component, the
shared numel / bytes, fl.comm.visit_bytes, the AdamW state bytes and the 02R1 §4 admissibility proxy (visit
send + receive <= 2 x 665 MB; per-client AdamW state <= 4 GiB) — recorded, never a refusal here.
Guards (the file is written only when all hold; nothing is written otherwise; TRAIN-02R1 F3: in production mode a
missing THETA0_REGISTRY entry fails, and the real run needs EQUAL for GRU_D128 / SASREC_D256_B2 x seeds 2026 / 2027):
  * the current builds (GRU_D128, SASREC_D256_B2) must equal THETA0_REGISTRY.json's entry for the same schema and
    torch build when it has one (bit-identical to today's builds); the absence of an entry is recorded;
  * contracts/derived_schemas/REGISTRY.json and THETA0_REGISTRY.json are only read: their sha256 before and after the
    run must be equal (and the derived-schema registry must still be the one DATA_PROTOCOL_LOCK_V2 binds);
  * the output file is NEW (O_EXCL; an existing THETA0_VARIANTS.json is never overwritten).
The default schema is DATA_PROTOCOL_LOCK_V2's production schema S*. No dataset row is read (only the v1 mapping files,
sha256-checked against FILES.json by v4models.build); no training, no optimizer, no weights written.
`--K n` (tests only) builds a synthetic catalogue arange(3, 3 + n) (at n = 158,486 it is the v1 catalogue map, so the
guard still applies; below it the guard is not applicable); a K-override file is refused by load_theta0_variants.
`load_theta0_variants(path)` verifies the file's self-hash; `lookup(obj, schema_id, torch_version, variant, seed)`
returns the registered init_sha256 (the launcher passes it as RunConfig.theta0_init_sha256).
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
import ppsi_env  # noqa: E402,F401

import torch  # noqa: E402

from fl.comm import visit_bytes  # noqa: E402
from fl.numerics import enforce_strict_fp32  # noqa: E402
from v4models import variants as V  # noqa: E402
from v4models.build import build  # noqa: E402

V4 = _CODE.parent
OUT_PATH = V4 / "contracts" / "model_variants" / "THETA0_VARIANTS.json"
DERIVED_REGISTRY = V4 / "contracts" / "derived_schemas" / "REGISTRY.json"
THETA0_REGISTRY = V4 / "contracts" / "derived_schemas" / "THETA0_REGISTRY.json"
LOCK_V2 = V4 / "contracts" / "DATA_PROTOCOL_LOCK_V2.json"
FORMAT = "PPSI_V4_THETA0_VARIANTS_1"
FULL_K = 158_486
VISIT_PROXY_BYTES = 2 * 665_000_000          # 02R1 §4: visit bytes <= 2 x 665 MB (SYS-02 SASRec visit)
ADAMW_PROXY_BYTES = 4 * 2 ** 30              # 02R1 §4: per-client AdamW state <= 4 GiB


class Theta0VariantsError(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def self_sha256(obj) -> str:
    return V.self_sha256(obj)


def production_schema_id(lock_path: Path = LOCK_V2) -> str:
    return json.loads(Path(lock_path).read_text(encoding="utf-8"))["features_and_transforms"]["production_schema"][
        "schema_id"]


def registered_current(schema_id: str, family: str, seed: int, theta0_registry: Path = THETA0_REGISTRY) -> Optional[dict]:
    """THETA0_REGISTRY.json's entry for (schema, the running torch build, family, seed), or None."""
    reg = json.loads(Path(theta0_registry).read_text(encoding="utf-8"))
    return (((reg.get("schemas") or {}).get(schema_id) or {}).get("builds") or {}).get(torch.__version__, {}) \
        .get(family, {}).get(str(seed))


def _one(fam, schema, seed, variant, K, v1_root) -> tuple:
    kw = {"K": K} if K is not None else {"v1_root": v1_root}
    bm = build(fam, schema, int(seed), "cpu", variant=variant, **kw)
    return bm, (bm.init_sha256, bm.raw_init_sha256, bm.manifest_digest)


def measure(schema_id: str, seeds: Iterable[int], variants: Iterable[str], *, v1_root: Optional[Path] = None,
            K: Optional[int] = None, schema=None, log=print, theta0_registry: Path = THETA0_REGISTRY,
            derived_registry: Path = DERIVED_REGISTRY) -> dict:
    enforce_strict_fp32()
    if (K is None) == (v1_root is None):
        raise Theta0VariantsError("give exactly one of v1_root (production) or K (tests)")
    utc = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  # noqa: E731
    watched = {"derived_schema_registry": Path(derived_registry), "theta0_registry": Path(theta0_registry)}
    before = {k: _sha(p) for k, p in watched.items()}
    vreg = V.load_variant_registry()
    if schema is None:
        from v4fs.schemas import load_registered
        schema = load_registered(schema_id, Path(derived_registry))
    if getattr(schema, "schema_id", None) != schema_id:
        raise Theta0VariantsError(f"schema object {getattr(schema, 'schema_id', None)!r} is not {schema_id!r}")
    out = {"task": "TRAIN-02 item 0b theta0 per variant", "torch": torch.__version__,
           "python": sys.version.split()[0], "host": platform.node(), "platform": platform.platform(),
           "utc_start": utc(), "schema_id": schema_id, "schema_self_sha256": schema.self_sha256, "K_override": K,
           "records": [], "inventory": {}, "guard": {}, "determinism": {}}
    for name in variants:
        fam = V.VARIANTS[name]["family"]
        for seed in seeds:
            t0 = time.time()
            bm, h1 = _one(fam, schema, seed, name, K, v1_root)
            rec = {"model_variant": name, "family": fam, "seed": int(seed), "init_sha256": h1[0],
                   "raw_init_sha256": h1[1], "manifest_digest": h1[2], "shared_numel": bm.manifest.shared_numel,
                   "K": bm.K, "recentered": bm.recentered, "host": platform.node(), "utc": utc()}
            if name not in out["inventory"]:
                man = bm.manifest
                vb = visit_bytes(man)
                out["inventory"][name] = {
                    "family": fam, "parameter_inventory": bm.inventory, "shared_numel": man.shared_numel,
                    "shared_bytes": man.shared_bytes, "weights_fp32_bytes": 4 * man.shared_numel,
                    "adamw_state_bytes": 2 * man.shared_bytes, "visit_bytes": vb, "query_dim": bm.adapter.query_dim,
                    "admissibility_proxy": {"visit_send_plus_receive_le_2x665MB":
                                            vb["send_plus_receive_bytes"] <= VISIT_PROXY_BYTES,
                                            "adamw_state_le_4GiB": 2 * man.shared_bytes <= ADAMW_PROXY_BYTES}}
            del bm
            gc.collect()
            bm2, h2 = _one(fam, schema, seed, name, K, v1_root)       # theta0 determinism: a second, fresh build
            del bm2
            gc.collect()
            out["determinism"][f"{name}/{seed}"] = h1 == h2
            rec["build_s_two_builds"] = round(time.time() - t0, 2)
            out["records"].append(rec)
            log(f"[theta0_variants] {schema_id} {name} seed {seed}: {h1[0]} numel {rec['shared_numel']} "
                f"(two builds equal: {h1 == h2})")
            if V.is_current_build(name):
                if K is not None and int(K) != FULL_K:
                    out["guard"][f"{name}/{seed}"] = {"status": "NOT_APPLICABLE_K_OVERRIDE"}
                else:
                    want = registered_current(schema_id, fam, seed, theta0_registry)
                    if want is None:
                        out["guard"][f"{name}/{seed}"] = {"status": "NO_THETA0_REGISTRY_ENTRY_FOR_THIS_TORCH_BUILD"}
                    else:
                        same = (h1[0], h1[1], h1[2], rec["shared_numel"]) == (
                            want["init_sha256"], want["raw_init_sha256"], want["manifest_digest"], want["shared_numel"])
                        out["guard"][f"{name}/{seed}"] = {"status": "EQUAL" if same else "DIFFERENT",
                                                          "registered": want}
    after = {k: _sha(p) for k, p in watched.items()}
    out["read_only_files"] = {k: {"path": str(watched[k]), "sha256_before": before[k], "sha256_after": after[k],
                                  "unchanged": before[k] == after[k]} for k in watched}
    if Path(derived_registry).resolve() == DERIVED_REGISTRY.resolve():     # the production registry: LOCK_V2 binds it
        prod = json.loads(LOCK_V2.read_text(encoding="utf-8"))["features_and_transforms"]["production_schema"]
        out["read_only_files"]["derived_schema_registry"]["bound_by_lock_v2"] = \
            after["derived_schema_registry"] == prod["derived_schema_registry_file_sha256"]
    out["model_variant_registry"] = {k: vreg[k] for k in ("path", "file_sha256", "self_sha256")}
    out["utc_end"] = utc()
    production = v1_root is not None or (K is not None and int(K) == FULL_K)
    out["guard_verdict"] = guard_verdict(out["guard"], production=production, require_coverage=v1_root is not None)
    out["ok"] = bool(all(out["determinism"].values())
                     and out["guard_verdict"]["ok"]
                     and all(v["unchanged"] and v.get("bound_by_lock_v2", True)
                             for v in out["read_only_files"].values()))
    return out


REQUIRED_GUARD = tuple(f"{v}/{s}" for v in ("GRU_D128", "SASREC_D256_B2") for s in (2026, 2027))


def guard_verdict(guard: dict, *, production: bool, require_coverage: bool) -> dict:
    """TRAIN-02R1 F3: a DIFFERENT current build always fails. In production mode (the v1 mapping files, or the full
    catalogue K) every guarded current build must be EQUAL to THETA0_REGISTRY (a missing entry for the running torch
    build is a failure, not a pass); with the v1 mapping files (the real run) both current builds x both seeds
    (REQUIRED_GUARD) must have been measured and be EQUAL."""
    problems = [f"{k}: {g['status']}" for k, g in guard.items() if g["status"] == "DIFFERENT"]
    if production:
        problems += [f"{k}: {g['status']} (production needs EQUAL)" for k, g in guard.items()
                     if g["status"] not in ("EQUAL", "DIFFERENT")]
    if require_coverage:
        problems += [f"{k}: not measured (production needs EQUAL)" for k in REQUIRED_GUARD if k not in guard]
    return {"ok": not problems, "problems": problems, "production": production, "require_coverage": require_coverage}


def registry_obj(res: dict) -> dict:
    """The THETA0_VARIANTS.json object of one measurement (schema -> torch build -> variant -> seed)."""
    builds = {}
    for r in res["records"]:
        builds.setdefault(r["model_variant"], {})[str(r["seed"])] = {
            k: r[k] for k in ("init_sha256", "raw_init_sha256", "manifest_digest", "shared_numel", "K", "host", "utc",
                              "recentered")}
    obj = {"format": FORMAT, "task": "TRAIN-02 item 0b",
           "definition": "fl.numerics.state_digest(adapter.broadcast_state()) of v4models.build(family, derived schema, "
                         "seed, 'cpu', variant=...) at the full catalogue; SASRec after the one C7 recentering",
           "schemas": {res["schema_id"]: {"schema_self_sha256": res["schema_self_sha256"],
                                          "builds": {res["torch"]: builds}}},
           "inventory": {res["schema_id"]: res["inventory"]},
           "guard_current_builds_vs_THETA0_REGISTRY": res["guard"], "determinism_two_builds": res["determinism"],
           "read_only_files": res["read_only_files"], "model_variant_registry": res["model_variant_registry"],
           "K_override": res["K_override"]}
    obj["self_sha256"] = self_sha256(obj)
    return obj


def write(res: dict, path: Path = OUT_PATH) -> dict:
    if not res.get("ok"):
        raise Theta0VariantsError("the measurement failed a guard; THETA0_VARIANTS.json is not written")
    obj = registry_obj(res)
    text = json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    return {"path": str(path), "file_sha256": V.write_exclusive(Path(path), text), "self_sha256": obj["self_sha256"]}


def load_theta0_variants(path: Path = OUT_PATH) -> dict:
    p = Path(path)
    if not p.is_file():
        raise Theta0VariantsError(f"{p} does not exist (run v4train.theta0_variants on the server first)")
    data = p.read_bytes()
    obj = json.loads(data.decode("utf-8"),
                     parse_constant=lambda c: (_ for _ in ()).throw(Theta0VariantsError(f"non-strict constant {c}")))
    if not isinstance(obj, dict) or obj.get("format") != FORMAT:
        raise Theta0VariantsError(f"{p} is not a {FORMAT} object")
    if obj.get("self_sha256") != self_sha256(obj):
        raise Theta0VariantsError(f"{p}: self-hash mismatch")
    if obj.get("K_override") is not None:
        raise Theta0VariantsError(f"{p}: a K-override (test) registry is not a production theta0 registry")
    return dict(obj, _file_sha256=hashlib.sha256(data).hexdigest(), _path=str(p))


def lookup(obj: dict, schema_id: str, torch_version: str, variant: str, seed: int) -> str:
    try:
        return obj["schemas"][schema_id]["builds"][torch_version][variant][str(int(seed))]["init_sha256"]
    except KeyError:
        raise Theta0VariantsError(f"no registered theta0 for schema {schema_id}, torch {torch_version}, variant "
                                  f"{variant}, seed {seed}") from None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="theta0 per model variant x seed (TRAIN-02 item 0b)")
    ap.add_argument("--schema", default=None, help="default: DATA_PROTOCOL_LOCK_V2's production schema (S*)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[2026, 2027])
    ap.add_argument("--variants", nargs="+", default=list(V.VARIANTS))
    ap.add_argument("--v1-root", default=None)
    ap.add_argument("--K", type=int, default=None, help="tests only: a small synthetic catalogue")
    ap.add_argument("--out", required=True, help="the receipt (full measurement)")
    ap.add_argument("--write", default=None, help="create THETA0_VARIANTS.json here (O_EXCL) when every guard holds")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(int(args.threads))
    sid = args.schema or production_schema_id()
    res = measure(sid, args.seeds, args.variants, v1_root=Path(args.v1_root) if args.v1_root else None, K=args.K)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
                              newline="\n")
    wrote = write(res, Path(args.write)) if (args.write and res["ok"]) else None
    print(json.dumps({"ok": res["ok"], "torch": res["torch"], "schema_id": sid, "guard": res["guard"],
                      "determinism": res["determinism"], "read_only_files": res["read_only_files"],
                      "written": wrote}, indent=1))
    return 0 if res["ok"] else 1


__all__ = ["FORMAT", "OUT_PATH", "Theta0VariantsError", "measure", "registry_obj", "write", "load_theta0_variants",
           "lookup", "production_schema_id", "registered_current"]

if __name__ == "__main__":
    raise SystemExit(main())
