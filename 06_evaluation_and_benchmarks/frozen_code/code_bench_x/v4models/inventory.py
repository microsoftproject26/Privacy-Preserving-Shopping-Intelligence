"""v4models.inventory — parameter-role manifest + parameter inventory per family at the Clean V2 widths (full K).

  python -B -m v4models.inventory --v1-root "<...>/rees46_nextitem_v1" --out code/v4models/inventory

Reads ONLY the v1 mapping files (catalogue + vocabularies, sha256-checked against FILES.json) and the schema; never a
dataset row. For each family it builds the Clean V2 model at the full catalogue K on CPU and writes
<FAMILY>_CLEAN_V2_MANIFEST_INVENTORY.json with: the role of every state_dict key (fl.adapter.build_manifest), the
parameter inventory per component, byte accounting per visit (fl.comm.visit_bytes), the initial-state sha256 for the
two registered seeds (SASRec: raw and C7-recentered), the v1-width reference count (must equal SYS-01's measured
count) and the exact V2 - v1 delta, plus the code / schema hashes it was produced from.
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

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))
import ppsi_env  # noqa: E402,F401

import torch  # noqa: E402

from fl.comm import visit_bytes  # noqa: E402
from fl.numerics import enforce_strict_fp32, state_digest  # noqa: E402
from v4models.build import TOPOLOGY, build, vocab_sizes_from_mappings  # noqa: E402
from v4models.schema import load_schema, v1_reference_widths  # noqa: E402

SEEDS = (2026, 2027)
SYS01_V1_PARAMS = {"GRU": 20_964_062, "SASREC": 83_094_368}     # company/handoffs/SYS-01.md (measured on the A16)
WIDTH_MATRICES = {"GRU": ("numeric_flags_proj.0.weight", "user_ctx_proj.0.weight"),
                  "SASREC": ("numflag_proj.weight", "user_mlp.weight")}
SOURCES = {"GRU": ("v4models/gru_v2.py", "snap_v1/reuse/model.py"),
           "SASREC": ("v4models/sasrec_v2.py", "snap_v1/sasrec_model.py")}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def family_record(family: str, schema, v1_root: Path) -> dict:
    vocab, poc = vocab_sizes_from_mappings(v1_root)
    rec = {"task": "MODEL-01", "evidence": "PROJECT_MEASURED: local CPU build (strict FP32), no dataset rows read",
           "family": family, "topology": TOPOLOGY[family],
           "schema": {"schema_id": schema.schema_id, "schema_version": schema.schema_version,
                      "self_sha256": schema.self_sha256, "file_sha256": schema.file_sha256},
           "widths": schema.widths.as_json(), "K": int(poc.shape[0]), "vocab_sizes": vocab,
           "class_map": "product_idx_of_class = arange(3, 3 + K) from V1 mappings/catalogue.parquet "
                        "(FILES.json sha256 verified)"}
    init = {}
    first = None
    for seed in SEEDS:
        bm = build(family, schema, seed, "cpu", v1_root=v1_root)
        init[str(seed)] = {"init_sha256": bm.init_sha256, "raw_init_sha256": bm.raw_init_sha256,
                           "recentered_theta0": bm.recentered}
        if first is None:
            first = bm
        else:
            del bm
            gc.collect()
    bm = first
    man = bm.manifest
    rec.update({"parameter_inventory": bm.inventory, "shared_numel": man.shared_numel,
                "shared_bytes": man.shared_bytes, "visit_bytes": visit_bytes(man),
                "manifest_digest": man.digest(), "manifest": man.as_json(), "init_state": init,
                "init_hash_definition": "fl.numerics.state_digest over adapter.broadcast_state() (shared canonical "
                                        "keys + persistent buffers); SASRec: after the one C7 recentering on CPU"})
    shapes_v2 = {k: list(e.shape) for k, e in man.entries.items()}
    if family == "SASREC":
        s0 = bm.theta0()
        bm.adapter.server_post_aggregate()
        s1 = bm.adapter.broadcast_state()
        rec["recentering_idempotence"] = {
            "max_abs_change_after_second_recentering": max(float((s0[k] - s1[k]).abs().max())
                                                           for k in ("output_embed", "output_bias")),
            "state_sha256_after_second_recentering": state_digest(s1),
            "bitwise_equal": state_digest(s1) == init[str(SEEDS[0])]["init_sha256"]}
    del bm, first
    gc.collect()
    ref = build(family, v1_reference_widths(schema), SEEDS[0], "cpu", v1_root=v1_root)
    n1 = ref.manifest.shared_numel
    shapes_v1 = {k: list(e.shape) for k, e in ref.manifest.entries.items()}
    del ref
    gc.collect()
    changed = {k: {"v1": shapes_v1[k], "v2": shapes_v2[k]} for k in shapes_v2 if shapes_v1.get(k) != shapes_v2[k]}
    rec["v1_reference"] = {"shared_numel_at_v1_widths": n1, "sys01_measured_v1": SYS01_V1_PARAMS[family],
                           "equals_sys01": n1 == SYS01_V1_PARAMS[family],
                           "delta_v2_minus_v1": rec["shared_numel"] - n1, "changed_shapes": changed,
                           "only_width_matrices_changed": sorted(changed) == sorted(WIDTH_MATRICES[family]),
                           "same_key_set": sorted(shapes_v1) == sorted(shapes_v2)}
    v2src, snapsrc = SOURCES[family]
    rec["code_sha256"] = {v2src: _sha(_CODE / v2src), snapsrc: _sha(_CODE / snapsrc),
                          "v4models/build.py": _sha(_CODE / "v4models/build.py"),
                          "v4models/schema.py": _sha(_CODE / "v4models/schema.py")}
    rec["environment"] = {"torch": torch.__version__, "python": sys.version.split()[0], "platform": platform.platform(),
                          "threads": torch.get_num_threads(), "device": "cpu",
                          "note": "GRU init uses PyTorch default init from the seeded global CPU RNG; the hashes are "
                                  "tied to this torch build and must be re-measured on the execution host"}
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    enforce_strict_fp32()
    schema = load_schema()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = {}
    for fam in ("GRU", "SASREC"):
        rec = family_record(fam, schema, Path(args.v1_root))
        rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        p = out / f"{fam}_CLEAN_V2_MANIFEST_INVENTORY.json"
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=1, allow_nan=False), encoding="utf-8")
        tmp.replace(p)
        summary[fam] = {"file": p.name, "shared_numel": rec["shared_numel"],
                        "delta_v2_minus_v1": rec["v1_reference"]["delta_v2_minus_v1"],
                        "equals_sys01_at_v1_widths": rec["v1_reference"]["equals_sys01"],
                        "manifest_digest": rec["manifest_digest"],
                        "init_sha256_2026": rec["init_state"]["2026"]["init_sha256"]}
    print(json.dumps({"wall_s": round(time.time() - t0, 1), "families": summary}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
