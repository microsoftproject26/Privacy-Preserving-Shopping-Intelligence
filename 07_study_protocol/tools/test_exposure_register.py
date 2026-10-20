"""test_exposure_register.py (owner: Operations Director) — the append-only, hash-chained TEST_EXPOSURE_REGISTER of
MANIFEST_SPEC_V4 `registers.TEST_EXPOSURE_REGISTER`. Location: company/registers/TEST_EXPOSURE_REGISTER.json.

  append --entry entry.json   validates the fields and enums, chains prev_entry_sha256, computes entry_sha256,
                              and writes atomically
  verify                      re-derives every hash and the chain, and exits 1 on any break

Canonical JSON: json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":")). Values of TEST metrics are
never written here: observations are enums plus a free-text description without values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

V = Path(__file__).resolve().parents[1]
REG = V / "company" / "registers" / "TEST_EXPOSURE_REGISTER.json"
SPLITS = {"RETRO_TEST", "LEGACY_TEST"}
PURPOSES = {"HISTORICAL_PRIOR_EXPOSURE", "STRUCTURAL_COUNT", "MANIFEST_BUILD", "LOCKED_EVALUATION",
            "RESTART_INCOMPLETE", "INTEGRITY_CHECK"}
OBS = {"NONE", "COUNTS_ONLY", "METRICS"}
LOCK_EXEMPT = {"HISTORICAL_PRIOR_EXPOSURE", "STRUCTURAL_COUNT"}
FIELDS = ("actor_role", "source", "split", "access_purpose", "observations", "observations_description", "decision",
          "all_methods_lock_sha256", "manifest_header_sha256", "model_checkpoint_sha256", "code_sha256", "receipt_path")


def canon(o) -> bytes:
    return json.dumps(o, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()


def load() -> dict:
    if REG.exists():
        return json.loads(REG.read_text(encoding="utf-8"))
    return {"register_id": "TEST_EXPOSURE_REGISTER", "spec_version": "MANIFEST_SPEC_V4 1.2.0", "entries": []}


def verify(reg: dict) -> list:
    errs, prev = [], None
    for i, e in enumerate(reg["entries"], 1):
        body = {k: v for k, v in e.items() if k != "entry_sha256"}
        if e["entry_id"] != i:
            errs.append(f"entry {i}: id {e['entry_id']}")
        if e["prev_entry_sha256"] != prev:
            errs.append(f"entry {i}: chain break")
        if hashlib.sha256(canon(body)).hexdigest() != e["entry_sha256"]:
            errs.append(f"entry {i}: hash mismatch")
        prev = e["entry_sha256"]
    return errs


def append(new: dict) -> dict:
    reg = load()
    errs = verify(reg)
    if errs:
        raise SystemExit(f"register broken before append: {errs}")
    missing = [f for f in FIELDS if f not in new]
    if missing:
        raise SystemExit(f"missing fields {missing}")
    if new["split"] not in SPLITS or new["access_purpose"] not in PURPOSES or new["observations"] not in OBS:
        raise SystemExit("bad enum")
    if new["access_purpose"] not in LOCK_EXEMPT and not new["all_methods_lock_sha256"]:
        raise SystemExit("all_methods_lock_sha256 required for this purpose")
    if new["observations"] == "METRICS" and re.search(r"\d\.\d|\d%", new["observations_description"]):
        raise SystemExit("no metric values (decimals or percentages) in a METRICS description")
    e = {"entry_id": len(reg["entries"]) + 1, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         **{k: new[k] for k in FIELDS}, "prev_entry_sha256": reg["entries"][-1]["entry_sha256"] if reg["entries"] else None}
    e["entry_sha256"] = hashlib.sha256(canon(e)).hexdigest()
    reg["entries"].append(e)
    REG.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(REG.parent))
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(reg, indent=1, ensure_ascii=False, allow_nan=False))
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, REG)
    return e


def main() -> int:
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="op", required=True)
    a = sp.add_parser("append"); a.add_argument("--entry", required=True)
    sp.add_parser("verify")
    x = ap.parse_args()
    if x.op == "verify":
        errs = verify(load()); print(json.dumps({"entries": len(load()["entries"]), "errors": errs})); return 1 if errs else 0
    e = append(json.loads(Path(x.entry).read_text(encoding="utf-8")))
    print(json.dumps({"entry_id": e["entry_id"], "entry_sha256": e["entry_sha256"]})); return 0


if __name__ == "__main__":
    raise SystemExit(main())
