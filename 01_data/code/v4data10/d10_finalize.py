"""v4data10.d10_finalize (owner: Data engineer, DATA-10) - the registry extension and the final server receipt
(the v4data08.d8_finalize procedure for role PRETRAIN).

  registry  verifies DATA10_BUILD_RECEIPT (self-hash, PASS, full build, n_failed 0), re-verifies the manifest on disk
            (v4data06.d6_finalize.verify_manifest_on_disk, unedited: header self-hash, every table's file / content
            hash and row count, the .OK marker), verifies REGISTRY_V2 against DATA_PROTOCOL_LOCK_V2's pin (file sha256
            AND self_sha256), then writes REGISTRY_V2_PRETRAIN_EXT.json (v4data10.d10_registry.build_ext_registry) and
            re-reads it through verify_ext_registry (all REGISTRY_V2 entries canonical-JSON identical). It refuses to
            overwrite an existing extension file whose bytes differ.
  final     test receipt + RSS-guard record + `date -u` stamps -> data_pretrain/receipts/DATA10_SERVER_RECEIPT.json.
Reads only data_pretrain/, contracts/ (lock + REGISTRY_V2) and data_v2/TRANSFORM_STATE; never the dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import ppsi_env  # noqa: E402,F401  (import isolation first, T43)

from v4data.common import atomic_write_bytes, json_bytes_strict, sha256_file  # noqa: E402
from v4data06.d6_finalize import verify_manifest_on_disk  # noqa: E402
from v4data08.d8_finalize import base_pin_checks  # noqa: E402  (REGISTRY_V2 vs the LOCK_V2 pin; pure read)
from v4data10 import d10_common as D10  # noqa: E402
from v4data10 import d10_registry as R  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

V4 = D10.V4
BASE_REL = "contracts/manifest_registry/REGISTRY_V2.json"
EXT_REL = "contracts/manifest_registry/REGISTRY_V2_PRETRAIN_EXT.json"
LOCK_V2_REL = "contracts/DATA_PROTOCOL_LOCK_V2.json"
BUILD_RECEIPT = "receipts/DATA10_BUILD_RECEIPT.json"


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def registry_entry(v4: Path, out_rel: str) -> dict:
    """The REGISTRY_V2-shaped entry of the PRETRAIN manifest (the d6_finalize.build_registry field set)."""
    mid = D10.TRAIN_ID
    v = verify_manifest_on_disk(v4, out_rel, mid)
    if not v["all_ok"]:
        raise SystemExit(f"{mid}: on-disk verification failed: {v['checks']}")
    h = v["header"]
    return {"path": f"{out_rel}/manifests/{mid}.header.json", "manifest_dir": f"{out_rel}/manifests", "kind": "TRAIN",
            "definition": D10.DEFINITION, "status": h["status"], "role": h["run_label"],
            "content_sha256": FC.manifest_content_sha256(h["tables"]), "header_sha256": v["header_file_sha256"],
            "header_content_sha256": h["header_content_sha256"], "n_users": h["n_users"],
            "n_decisions": h["n_decisions"], "n_eval_rows": h["n_eval_rows"], "n_rankable_rows": h["n_rankable_rows"],
            "population_sha256": h["population_sha256"],
            "population_sha256_selection_key_v2": h["population_sha256_selection_key_v2"],
            "ordered_decision_id_sha256": h["ordered_decision_id_sha256"],
            "eval_ordered_decision_id_sha256": h["eval_ordered_decision_id_sha256"],
            "rankable_ordered_decision_id_sha256": h["rankable_ordered_decision_id_sha256"],
            "parent_manifests": h["parent_manifests"], "tables": v["tables"], "marker": f"{out_rel}/manifests/{mid}.OK"}


def build_extension(v4: Path = V4, out_rel: str = D10.OUT_REL, *, base_rel: str = BASE_REL,
                    lock_rel: str = LOCK_V2_REL, require_full: bool = True) -> dict:
    root = Path(v4) / out_rel
    brec = _load(root / BUILD_RECEIPT)
    pre = {"build_receipt_self_hash": FC.verify_self_hash(brec, "receipt_sha256"),
           "build_receipt_pass": brec.get("status") == "PASS" and (brec.get("full_build") is True or not require_full),
           "build_receipt_n_failed_zero": brec.get("n_failed") == 0}
    if not all(pre.values()):
        raise SystemExit(f"DATA10_BUILD_RECEIPT does not verify: {pre}")
    pins = base_pin_checks(v4, base_rel, lock_rel)
    bad = [k for k, v in pins.items() if not k.startswith("_") and v is not True]
    if bad:
        raise SystemExit(f"REGISTRY_V2 is not the registry DATA_PROTOCOL_LOCK_V2 pins: {bad}")
    entry = registry_entry(v4, out_rel)
    h = _load(Path(v4) / entry["path"])
    if FC.hashes_entry(h, h["tables"], entry["header_sha256"]) != brec["manifests"][D10.TRAIN_ID]:
        raise SystemExit(f"{D10.TRAIN_ID}: header / files differ from the build receipt")
    pop = brec["population"]
    band = {k: pop[k] for k in ("rank_range", "n_users", "population_sha256_selection_key_v2", "definition",
                                "excluded_bands", "excluded_panels")}
    ext = R.build_ext_registry(
        pins["_base"], base_rel_path=base_rel, base_file_sha256=pins["_file_sha256"], entry=entry, band=band,
        build_receipt={"receipt": f"{out_rel}/{BUILD_RECEIPT}",
                       "receipt_file_sha256": sha256_file(root / BUILD_RECEIPT),
                       "receipt_sha256": brec["receipt_sha256"]},
        code={"d10_code_sha256": brec["code"]["d10_code_sha256"],
              "v4data_builder_code_sha256": brec["code"]["v4data_builder_code_sha256"],
              "d10_registry_sha256": sha256_file(HERE / "d10_registry.py"),
              "d10_finalize_sha256": sha256_file(HERE / "d10_finalize.py")})
    chk = R.verify_ext_registry(ext, pins["_base"], pins["_file_sha256"])
    if not chk["all_ok"]:
        raise SystemExit(f"extension registry fails its own rule: {[k for k, v in chk.items() if not v]}")
    return ext


def write_extension(v4: Path, ext: dict, ext_rel: str = EXT_REL, base_rel: str = BASE_REL) -> dict:
    data = json_bytes_strict(ext)
    p = Path(v4) / ext_rel
    if p.exists() and p.read_bytes() != data:
        raise SystemExit(f"{ext_rel} exists with different bytes: refused (no overwrite)")
    p = atomic_write_bytes(Path(v4), ext_rel, data)
    back = _load(p)
    chk = R.verify_ext_registry(back, _load(Path(v4) / base_rel), sha256_file(Path(v4) / base_rel))
    if not chk["all_ok"]:
        raise SystemExit(f"{ext_rel}: re-read verification failed: {[k for k, v in chk.items() if not v]}")
    return {"registry": ext_rel, "file_sha256": sha256_file(p), "self_sha256": back["self_sha256"], "checks": chk}


def final_receipt(v4: Path, out_rel: str, tests_receipt: Path, rss_logs: list, run_dir: Path) -> dict:
    root = v4 / out_rel
    brec = _load(root / BUILD_RECEIPT)
    ext = _load(v4 / EXT_REL)
    chk = R.verify_ext_registry(ext, _load(v4 / BASE_REL), sha256_file(v4 / BASE_REL))
    pins = base_pin_checks(v4)
    tests = _load(Path(tests_receipt))
    guards = [[json.loads(x) for x in Path(g).read_text(encoding="utf-8").splitlines() if x.strip()][-1]
              for g in rss_logs]
    stamps = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted(run_dir.glob("*_utc.txt"))}
    df = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted(run_dir.glob("df_*.txt"))}
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and "_run" not in p.parts)
    checks = {"build_receipt_pass": brec["status"] == "PASS" and FC.verify_self_hash(brec, "receipt_sha256")
                                    and brec["n_failed"] == 0 and brec["full_build"] is True,
              "manifest_on_disk": verify_manifest_on_disk(v4, out_rel, D10.TRAIN_ID)["all_ok"],
              "ext_registry_rule": chk["all_ok"],
              "base_registry_pinned_by_lock_v2": all(v is True for k, v in pins.items() if not k.startswith("_")),
              "tests_exit_0": tests.get("exit_code") == 0,
              "tests_negative_controls_failed_as_intended": tests.get("negative_controls", {}).get(
                  "all_failed_as_intended") is True,
              "tests_real_data_ran": tests.get("counts", {}).get("skipped", 0) == 0,
              "rss_guards_rc_0": bool(guards) and all(g["rc"] == 0 and not g["killed_rss_guard"] for g in guards),
              "no_pycache_in_owned_code": not any(True for d in ("v4data10", "tests_data10")
                                                  for _ in (D10.CODE / d).rglob("__pycache__")),
              "no_test_manifest_or_role": chk["no_TEST_id_or_role"]}
    e = ext["manifests"][D10.TRAIN_ID]
    rec = {"receipt": "DATA10_SERVER_RECEIPT", "task": D10.TASK, "overall_pass": all(checks.values()),
           "checks": checks, "date_u": stamps, "df": df, "rss_guards": guards, "data_pretrain_bytes": int(size),
           "ext_registry": {"path": EXT_REL, "file_sha256": sha256_file(v4 / EXT_REL),
                            "self_sha256": ext["self_sha256"], "extends": ext["extends"]},
           "build_receipt_sha256": brec["receipt_sha256"],
           "tests": {k: tests.get(k) for k in ("exit_code", "counts", "negative_controls", "summary_line")},
           "tests_receipt_file_sha256": sha256_file(Path(tests_receipt)),
           "manifest": {D10.TRAIN_ID: {k: e[k] for k in ("n_users", "n_decisions", "content_sha256", "header_sha256",
                                                         "header_content_sha256", "population_sha256",
                                                         "population_sha256_selection_key_v2")}},
           "code": brec["code"], "finalize_code_sha256": sha256_file(HERE / "d10_finalize.py"),
           "test_split_read": False}
    p, rec = FC.write_json_self_hashed(root, "receipts/DATA10_SERVER_RECEIPT.json", rec)
    rec["_file_sha256"] = sha256_file(p)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("registry", "final"))
    ap.add_argument("--v4", default=str(V4))
    ap.add_argument("--out-rel", default=D10.OUT_REL)
    ap.add_argument("--tests-receipt", default=None)
    ap.add_argument("--rss-log", action="append", default=[])
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args(argv)
    v4 = Path(a.v4)
    if a.step == "registry":
        out = write_extension(v4, build_extension(v4, a.out_rel))
        print(json.dumps({k: out[k] for k in ("registry", "file_sha256", "self_sha256")}))
        return 0
    rec = final_receipt(v4, a.out_rel, Path(a.tests_receipt), a.rss_log, Path(a.run_dir))
    print(json.dumps({"overall_pass": rec["overall_pass"], "receipt_sha256": rec["receipt_sha256"],
                      "file_sha256": rec["_file_sha256"], "checks": rec["checks"]}))
    return 0 if rec["overall_pass"] else 4


if __name__ == "__main__":
    sys.exit(main())
