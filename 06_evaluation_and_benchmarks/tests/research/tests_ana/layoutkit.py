"""ANA-02: a synthetic runs_ds3-style layout (REEVAL_A40 / FL / DEV / BASE / ONNX / TRUNC + records) built from an
anakit block, for tools/ana_block_index.py. Every evidence file is a byte copy of the anakit block's, so the index the
builder writes must give the same retention numbers as the hand-written anakit index.

ANA-03 (REV-DELTA-01 F1): the REEVAL_A40 dirs are written by the REAL `v4int.reeval.write_outputs` (read-only import), so
the REEVAL_RECORD is exactly the format the deployed tool writes (top-level hashes + `outputs`), not a hand-written copy
of the ds3r2 field names. `record_form` rewrites the record afterwards: "real" (default), "outputs_only" (the hashes only
under `outputs`), "top_level_only" (the ds3r2 records)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import anakit
import retention_report as R

PIN_OBJ = {"device_class": anakit.DEV, "self_sha256": anakit.PIN, "format": "PPSI_V4_EVAL_DEVICE_PIN_1"}
REEVAL_ARMS = ("C_BIG", "C_SMALL", "C_FULL", "FP", "PF", "FA_1024", "DP8", "DP4", "DP1", "FA_Q8")


def _jd(p: Path, obj) -> None:
    anakit.jdump(p, obj)


def _cp(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def ckpt_sha(arm, seed, rule) -> str:
    return R._sha_bytes(f"ckpt|{arm}|{seed}|{rule}".encode())


def central_source_record(arm: str) -> dict:
    """FIX-DELTA-03 M1: the REEVAL_RECORD.central_source v4int.reeval writes for a registered source (GATE-B B1)."""
    lab = anakit.SOURCE_LABELS[arm]
    return {"source": arm, "arm_label": arm, "ext_role": lab["source_manifest_role"],
            "pair": {"train": lab["source_train_manifest_id"], "eval": lab["source_eval_manifest_id"]},
            "ext_self_sha256": "e" * 64}


def write_reeval_dir(d: Path, src_rows: Path, src_ranks: Path, record: dict, role: str, form: str = "real") -> None:
    """One REEVAL_A40 dir through the real v4int.reeval.write_outputs (tag = role), then the record form of `form`."""
    import numpy as np
    from v4int.reeval import write_outputs
    rows = json.loads(Path(src_rows).read_text(encoding="utf-8"))["rows"]
    write_outputs(d, tag=role, run_id=f"REEVAL_{record['arm_label']}_{role}_s{record['seed']}_{record['selection_rule']}",
                  role=role, rows=rows, result={"n_rows": len(rows)}, ranks=np.load(src_ranks), record=record)
    if form != "real":
        p = d / "REEVAL_RECORD.json"
        rec = json.loads(p.read_text(encoding="utf-8"))
        if form == "outputs_only":
            rec.pop("result_rows_sha256")
            rec.pop("ranks_sha256")
        elif form == "top_level_only":
            rec.pop("outputs")
        else:
            raise ValueError(form)
        _jd(p, rec)


def build_layout(tmp: Path, blk, record_form: str = "real") -> dict:
    """-> {"R": runs root, "args": the builder argv (without --out), "pin": path}."""
    Rr = tmp / "runs_ds3"
    src = blk.root
    idx = blk.index()
    for e in idx["entries"]:
        arm, seed, rule, role = e["arm"], e["seed"], e["rule"], e["role"]
        if e.get("comparison_block") or arm == "POP" or (arm == "FA" and role == "MAIN"):
            continue
        if arm in anakit.DEVICE_ARMS:
            name = f"{arm}_s{seed}" if role == "MAIN" else f"{arm}_{role}_s{seed}"
            d = Rr / "DEV" / name
            _cp(src / e["rows"], d / "RESULT_ROWS.json")
            _cp(src / e["ranks"], d / "cells" / f"{anakit.DEVICE_CELL}.ranks_int32.npy")
            _jd(d / "RUN_SUMMARY.json", {"status": "DONE", "arm": arm, "role": role, "selected_cell": anakit.DEVICE_CELL,
                                         "spec": {"seed": seed, "select": False}, "device_pin_sha256": anakit.PIN})
            _jd(d / "LAUNCH_0.json", {"label": "PRODUCTION", "arm": arm})
            continue
        tag = "PB" if rule == "PRACTICAL_BEST" else "CB"
        name = f"{arm}_s{seed}_{tag}" if role == "MAIN" else f"{arm}_{role}_s{seed}_{tag}"
        d = Rr / "REEVAL_A40" / name
        write_reeval_dir(d, src / e["rows"], src / e["ranks"], {
            "format": "PPSI_V4_REEVAL_RECORD_1", "arm_label": arm, "seed": seed,
            "role": role, "selection_rule": rule, "checkpoint_id": anakit.ckpt_id(arm, seed, rule),
            "checkpoint_sha256": ckpt_sha(arm, seed, rule), "device_class": anakit.DEV,
            "device_pin_sha256": anakit.PIN, "eval_manifest_id": idx["roles"][role]["eval_manifest_id"],
            **({"central_source": central_source_record(arm)} if arm in anakit.SOURCE_LABELS else {})},
            role, record_form)
    for seed in (2026, 2027):                                          # FL: the anakit FA run dirs
        s, d = src / "MAIN" / f"FL_FA_s{seed}", Rr / "FL" / f"FA_s{seed}"
        shutil.copytree(s, d)
        summ = json.loads((d / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
        summ["arm_spec"]["cfg"]["seed"] = seed
        _jd(d / "RUN_SUMMARY.json", summ)
        _jd(d / "LAUNCH_0.json", {"label": "PRODUCTION", "arm": "FA", "fl05": {"role": "MAIN"},
                                  "device_pin_sha256": anakit.PIN})
    _cp(src / "MAIN" / "POP_rows.json", Rr / "BASE" / "POP" / "rows.json")
    _jd(Rr / "BASE" / "POP" / "RECEIPT.json", {"baseline": "POP", "method": "POP"})
    _cp(src / "ONNX" / "evals" / "onnx_int8.ranks_int32.npy", Rr / "ONNX" / "evals" / "onnx_int8.ranks_int32.npy")
    meta = json.loads((src / "ONNX" / "rows.meta.json").read_text(encoding="utf-8"))
    _jd(Rr / "ONNX" / "rows.meta.json", meta)
    bench = anakit.bench_result({"onnx_int8": R._sha_file(Rr / "ONNX" / "evals" / "onnx_int8.ranks_int32.npy")})
    bench["retention"]["onnx_int8"]["rows_meta_sha256"] = R._sha_file(Rr / "ONNX" / "rows.meta.json")
    bench["checkpoint"] = "/srv/runs_ds/C_SMALL_s2026/sealed/checkpoints/best.pt"
    _jd(Rr / "ONNX" / "BENCH_RESULT.json", bench)
    d = Rr / "TRUNC" / "SA_C_SMALL_s2026"                               # Study A (C_BIG PB checkpoint -> PB)
    _cp(src / "records" / "TRUNC.json", d / "RESULT_ROWS.json")
    _jd(d / "STATUS.json", {"status": "DONE"})
    _jd(d / "LAUNCH_0.json", {"study": "A", "label": "PRODUCTION",
                              "ckpt": {"file_sha256": ckpt_sha("C_BIG", 2026, "PRACTICAL_BEST")}})
    (Rr / "TRUNC" / "T41_WT").mkdir(parents=True)                       # a T41 dir: skipped
    rec = tmp / "records_src"
    for f in ("STRATA_CUTPOINTS_MAIN.json", "S_RECORD.json", "LR_SELECTION.json"):
        _cp(src / "records" / f, rec / f)
    _cp(src / "records" / "SEL_FT_C.json", rec / "SELECTION_FT_C.json")
    _cp(src / "records" / "DP_WOR.json", rec / "DP_WOR_SENSITIVITY_FL06.json")
    pin = tmp / "EVAL_DEVICE_PIN_A40.json"
    _jd(pin, PIN_OBJ)
    args = ["--split", "VALIDATION", "--device-class", anakit.DEV, "--pin", str(pin),
            "--reeval", str(Rr / "REEVAL_A40"), "--fl", str(Rr / "FL"), "--dev", str(Rr / "DEV"),
            "--base", str(Rr / "BASE"), "--onnx", str(Rr / "ONNX"), "--trunc", str(Rr / "TRUNC"),
            "--records", *[str(p) for p in sorted(rec.iterdir())]]
    return {"R": Rr, "args": args, "pin": pin}
