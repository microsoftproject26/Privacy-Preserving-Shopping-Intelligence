"""v4data09.d9_finalize (owner: Data engineer, DATA-09) - the TEST registry extension, the loader-path verification, the
post-build G0 integrity check and the final server receipt. Every step calls d9_gate.check_test_gate FIRST.

  registry  verifies DATA09_BUILD_RECEIPT (self-hash, PASS, full build, n_failed 0, its gate binding == the open gate),
            re-verifies every RETRO_TEST manifest on disk (header self-hash, gate fields, assertions, every table's file
            / content hash (extended field dictionary) and row count, the .OK marker), verifies REGISTRY_V2 against the
            DATA_PROTOCOL_LOCK_V2 pin (v4data08.d8_finalize.base_pin_checks, unedited), then writes
            contracts/manifest_registry/REGISTRY_V2_TEST_EXT.json with O_EXCL (an existing file is accepted only if its
            bytes are identical; different bytes are refused) and re-reads it through verify_test_ext_registry.
  verify    the LOADER path: for every TEST role, v4int.test_roles.verify_test_manifest (registry + header + tables +
            rows under the gate) and v4eval.manifest.build_eval_manifest with the ALL_METHODS_LOCK path
            -> receipts/DATA09_LOADER_VERIFY.json.
  g0        AFTER the build: the builder's full-population RETRO_TEST counters vs the G0 census (SPLIT_COUNTS.json
            per_old_split.TEST and v4_retro_test, CLIENT_CENSUS.json s5.test_users; TEST_EXPOSURE_REGISTER entry 1),
            counts only -> receipts/DATA09_G0_INTEGRITY.json.
  final     tests receipt + RSS-guard record + `date -u` stamps -> data_test/receipts/DATA09_SERVER_RECEIPT.json.
Reads only data_test/, contracts/ (locks + registries), g0/data_census (the two G0 files) and data_v2/TRANSFORM_STATE
(through the loader); never the dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
import ppsi_env  # noqa: E402,F401  (import isolation first, T43)

import pyarrow.parquet as pq  # noqa: E402

from v4data.common import json_bytes_strict, sha256_file, table_content_sha256  # noqa: E402
from v4data08.d8_finalize import base_pin_checks  # noqa: E402
from v4data09 import d9_common as D9  # noqa: E402
from v4data09 import d9_registry as R  # noqa: E402
from v4data09.d9_gate import GateRecord, TestGateRefused, check_test_gate, header_matches_gate  # noqa: E402
from v4fs_data import fs_common as FC  # noqa: E402

V4 = D9.V4
BASE_REL = "contracts/manifest_registry/REGISTRY_V2.json"
EXT_REL = "contracts/manifest_registry/REGISTRY_V2_TEST_EXT.json"
LOCK_V2_REL = "contracts/DATA_PROTOCOL_LOCK_V2.json"


def _load(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


# ------------------------------------------------------------------------------------------------ on-disk verification
def verify_manifest_on_disk_d9(v4: Path, out_rel: str, mid: str, gate: GateRecord) -> dict:
    """v4data06.d6_finalize.verify_manifest_on_disk for a RETRO_TEST manifest (gate instead of the id refusal)."""
    if mid not in D9.TEST_IDS:
        raise TestGateRefused(f"{mid} is not a registered RETRO_TEST id")
    d = Path(v4) / out_rel / "manifests"
    hp = d / f"{mid}.header.json"
    h = _load(hp)
    ok = {"header.content_self_hash": h["header_content_sha256"] == FC.header_content_sha256(h),
          "header.manifest_id": h["manifest_id"] == mid,
          "header.gate_eq_open_gate": header_matches_gate(h, gate),
          "header.v4_split_RETRO_TEST": h.get("v4_split") == "RETRO_TEST",
          "header.status_is_role": h.get("status") == D9.ROLE_OF_ID[mid] == h.get("run_label"),
          "header.assertions_all_true": all(v is True for v in h["assertions"].values())}
    tables = {}
    for t in h["tables"]:
        f = d / f"{mid}.{t['table']}.parquet"
        fsha = sha256_file(f)
        tbl = pq.ParquetFile(f).read()
        c = table_content_sha256(D9.reread_columns_d9(tbl))
        ok[f"{t['table']}.file_sha256"] = fsha == t["file_sha256"]
        ok[f"{t['table']}.content_sha256"] = c == t["table_content_sha256"]
        ok[f"{t['table']}.row_count"] = tbl.num_rows == t["row_count"]
        tables[t["table"]] = {"file": f"{out_rel}/manifests/{f.name}", "file_sha256": fsha,
                              "table_content_sha256": c, "row_count": int(tbl.num_rows)}
        del tbl
    mk = _load(d / f"{mid}.OK")["verified_sha256"]
    ok["ok_marker_matches_disk"] = bool(mk) and all(sha256_file(Path(v4) / out_rel / rel) == s for rel, s in mk.items())
    return {"header": h, "header_file_sha256": sha256_file(hp), "tables": tables, "checks": ok,
            "all_ok": all(ok.values())}


def registry_entry_d9(v4: Path, out_rel: str, mid: str, gate: GateRecord) -> dict:
    """The REGISTRY_V2-shaped entry of a RETRO_TEST manifest (+ the two M07 gate fields and v4_split)."""
    v = verify_manifest_on_disk_d9(v4, out_rel, mid, gate)
    if not v["all_ok"]:
        raise SystemExit(f"{mid}: on-disk verification failed: {v['checks']}")
    h = v["header"]
    return {"path": f"{out_rel}/manifests/{mid}.header.json", "manifest_dir": f"{out_rel}/manifests", "kind": "EVAL",
            "definition": D9.DEFINITION[mid], "status": h["status"], "role": h["run_label"],
            "v4_split": "RETRO_TEST", "all_methods_lock_sha256": h["all_methods_lock_sha256"],
            "test_exposure_register_entry_id": h["test_exposure_register_entry_id"],
            "content_sha256": FC.manifest_content_sha256(h["tables"]), "header_sha256": v["header_file_sha256"],
            "header_content_sha256": h["header_content_sha256"], "n_users": h["n_users"],
            "n_decisions": h["n_decisions"], "n_eval_rows": h["n_eval_rows"], "n_rankable_rows": h["n_rankable_rows"],
            "population_sha256": h["population_sha256"],
            "population_sha256_selection_key_v2": h["population_sha256_selection_key_v2"],
            "ordered_decision_id_sha256": h["ordered_decision_id_sha256"],
            "eval_ordered_decision_id_sha256": h["eval_ordered_decision_id_sha256"],
            "rankable_ordered_decision_id_sha256": h["rankable_ordered_decision_id_sha256"],
            "parent_manifests": h["parent_manifests"], "tables": v["tables"], "marker": f"{out_rel}/manifests/{mid}.OK"}


def verify_build_receipt(root: Path, gate: GateRecord, require_full: bool = True) -> dict:
    brec = _load(Path(root) / "receipts" / "DATA09_BUILD_RECEIPT.json")
    pre = {"build_receipt_self_hash": FC.verify_self_hash(brec, "receipt_sha256"),
           "build_receipt_pass": brec.get("status") == "PASS" and (brec.get("full_build") is True or not require_full),
           "build_receipt_n_failed_zero": brec.get("n_failed") == 0,
           "build_receipt_gate_eq_open_gate": all((brec.get("test_gate") or {}).get(k) == v
                                                  for k, v in gate.binding().items() if k != "rule")}
    if not all(pre.values()):
        raise SystemExit(f"DATA09_BUILD_RECEIPT does not verify: {pre}")
    return brec


def build_extension(v4: Path, gate: GateRecord, out_rel: str = D9.OUT_REL, *, base_rel: str = BASE_REL,
                    lock_rel: str = LOCK_V2_REL, require_full: bool = True) -> dict:
    root = Path(v4) / out_rel
    brec = verify_build_receipt(root, gate, require_full)
    pins = base_pin_checks(v4, base_rel, lock_rel)
    bad = [k for k, v in pins.items() if not k.startswith("_") and v is not True]
    if bad:
        raise SystemExit(f"REGISTRY_V2 is not the registry DATA_PROTOCOL_LOCK_V2 pins: {bad}")
    entries = {}
    for mid in D9.TEST_IDS:
        e = registry_entry_d9(v4, out_rel, mid, gate)
        h = _load(Path(v4) / e["path"])
        if FC.hashes_entry(h, h["tables"], e["header_sha256"]) != brec["manifests"][mid]:
            raise SystemExit(f"{mid}: header / files differ from the build receipt")
        entries[mid] = e
    ext = R.build_test_ext_registry(
        pins["_base"], base_rel_path=base_rel, base_file_sha256=pins["_file_sha256"], entries=entries,
        gate_binding={k: v for k, v in gate.binding().items()},
        build_receipt={"receipt": f"{out_rel}/receipts/DATA09_BUILD_RECEIPT.json",
                       "receipt_file_sha256": sha256_file(root / "receipts" / "DATA09_BUILD_RECEIPT.json"),
                       "receipt_sha256": brec["receipt_sha256"]},
        code={"d9_code_sha256": brec["code"]["d9_code_sha256"],
              "v4data_builder_code_sha256": brec["code"]["v4data_builder_code_sha256"],
              "d9_registry_sha256": sha256_file(HERE / "d9_registry.py"),
              "d9_finalize_sha256": sha256_file(HERE / "d9_finalize.py")})
    chk = R.verify_test_ext_registry(ext, pins["_base"], pins["_file_sha256"])
    if not chk["all_ok"]:
        raise SystemExit(f"extension registry fails its own rule: {[k for k, v in chk.items() if not v]}")
    return ext


def write_extension(v4: Path, ext: dict, ext_rel: str = EXT_REL, base_rel: str = BASE_REL) -> dict:
    """O_EXCL create; an existing file is accepted only with identical bytes (never overwritten)."""
    data = json_bytes_strict(ext)
    p = Path(v4) / ext_rel
    if p.exists():
        if p.read_bytes() != data:
            raise SystemExit(f"{ext_rel} exists with different bytes: refused (no overwrite)")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    back = _load(p)
    chk = R.verify_test_ext_registry(back, _load(Path(v4) / base_rel), sha256_file(Path(v4) / base_rel))
    if not chk["all_ok"]:
        raise SystemExit(f"{ext_rel}: re-read verification failed: {[k for k, v in chk.items() if not v]}")
    return {"registry": ext_rel, "file_sha256": sha256_file(p), "self_sha256": back["self_sha256"], "checks": chk}


# ------------------------------------------------------------------------------------------------ G0 integrity
def _cmp_block(obs: dict, exp: dict, prefix: str) -> dict:
    out = {}
    for k, ev in exp.items():
        if isinstance(ev, dict):
            out.update(_cmp_block(obs.get(k) or {}, ev, f"{prefix}.{k}"))
        else:
            ov = obs.get(k)
            out[f"{prefix}.{k}"] = {"expected": ev, "observed": ov, "match": ov == ev}
    for k, ov in obs.items():                            # an observed status the census never listed must be 0
        if not isinstance(ov, dict) and k not in exp:
            out[f"{prefix}.{k}"] = {"expected": "absent (0)", "observed": ov, "match": ov == 0}
    return out


def g0_integrity(counters: dict, split_counts: dict, client_census: dict) -> dict:
    rows = {}
    rows.update(_cmp_block(counters["per_old_split_TEST"], split_counts["per_old_split"]["TEST"],
                           "SPLIT_COUNTS.per_old_split.TEST"))
    rows.update(_cmp_block(counters["v4_retro_test"], split_counts["v4_retro_test"], "SPLIT_COUNTS.v4_retro_test"))
    rows.update(_cmp_block(counters["test_users"], client_census["s5"]["test_users"], "CLIENT_CENSUS.s5.test_users"))
    return {"all_match": all(r["match"] for r in rows.values()), "n_compared": len(rows), "rows": rows}


def g0_step(v4: Path, gate: GateRecord, out_rel: str = D9.OUT_REL, *, split_counts_path: Path = D9.G0_SPLIT_COUNTS,
            client_census_path: Path = D9.G0_CLIENT_CENSUS, require_full: bool = True) -> dict:
    root = Path(v4) / out_rel
    brec = verify_build_receipt(root, gate, require_full)
    res = g0_integrity(brec["full_population_retro_test_counters"], _load(split_counts_path),
                       _load(client_census_path))
    rec = {"receipt": "DATA09_G0_INTEGRITY", "task": D9.TASK, "all_match": res["all_match"],
           "n_compared": res["n_compared"], "rows": res["rows"], "build_receipt_sha256": brec["receipt_sha256"],
           "g0_files": {"split_counts": {"path": str(split_counts_path), "file_sha256": sha256_file(split_counts_path)},
                        "client_census": {"path": str(client_census_path),
                                          "file_sha256": sha256_file(client_census_path)}},
           "register_entry": "TEST_EXPOSURE_REGISTER entry 1 (STRUCTURAL_COUNT, G0 census)",
           "note": "INTEGRITY_CHECK after the build: counts only, no model output; a mismatch stops the look (Ops)"}
    FC.write_json_self_hashed(root, "receipts/DATA09_G0_INTEGRITY.json", rec)
    return rec


# ------------------------------------------------------------------------------------------------ loader verification
def verify_step(v4: Path, gate: GateRecord, ext_rel: str = EXT_REL) -> dict:
    """Every TEST role through the loader path of v4int.test_roles (gated), then the v4eval EvalManifest."""
    from v4int import test_roles as TR
    from v4int.verify import DataPaths
    ext_path = Path(v4) / ext_rel
    out = {}
    for role, pair in D9.TEST_ROLES.items():
        res = TR.resolve_role(role, ext_path=ext_path, lock_path=Path(gate.lock_path),
                              register_path=Path(gate.register_path))
        vm = TR.verify_test_manifest(pair["eval"], ext_path=ext_path, paths=DataPaths.default(Path(v4)), root=Path(v4),
                                     lock_path=Path(gate.lock_path), register_path=Path(gate.register_path))
        em = TR.test_eval_manifest(vm, lock_path=Path(gate.lock_path), register_path=Path(gate.register_path))
        out[role] = {"resolved": {"train": res.get("train"), "eval": res["eval"]}, "n_checks": len(vm.checks),
                     "n_decisions": int(vm.canonical_decision_id.size), "eval_manifest_n": int(em.n),
                     "n_eval_rows": int(em.eval_mask.sum()), "n_rankable_rows": int(em.rankable.sum()),
                     "header_content_sha256": vm.header["header_content_sha256"]}
        vm.columns = None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("registry", "verify", "g0", "final"))
    ap.add_argument("--v4", default=str(V4))
    ap.add_argument("--out-rel", default=D9.OUT_REL)
    ap.add_argument("--lock", type=Path, default=None)
    ap.add_argument("--register", type=Path, default=None)
    ap.add_argument("--tests-receipt", default=None)
    ap.add_argument("--rss-log", action="append", default=[])
    ap.add_argument("--run-dir", default=None)
    a = ap.parse_args(argv)
    v4 = Path(a.v4)
    try:
        gate = check_test_gate(a.lock, a.register)       # FIRST, for every step
    except TestGateRefused as e:
        print(f"DATA-09 {a.step} REFUSED (TEST gate closed): {e}", file=sys.stderr, flush=True)
        return 2
    root = v4 / a.out_rel
    if a.step == "registry":
        out = write_extension(v4, build_extension(v4, gate, a.out_rel))
        print(json.dumps({k: out[k] for k in ("registry", "file_sha256", "self_sha256")}))
        return 0
    if a.step == "verify":
        res = verify_step(v4, gate)
        p, rec = FC.write_json_self_hashed(root, "receipts/DATA09_LOADER_VERIFY.json",
                                           {"receipt": "DATA09_LOADER_VERIFY", "task": D9.TASK, "roles": res,
                                            "test_gate": gate.binding()})
        print(json.dumps({"loader_verify": "PASS", "receipt_sha256": rec["receipt_sha256"]}))
        return 0
    if a.step == "g0":
        rec = g0_step(v4, gate, a.out_rel)
        print(json.dumps({"g0_all_match": rec["all_match"], "n_compared": rec["n_compared"]}))
        return 0 if rec["all_match"] else 4
    run = Path(a.run_dir)
    brec = verify_build_receipt(root, gate)
    ext = _load(v4 / EXT_REL)
    chk = R.verify_test_ext_registry(ext, _load(v4 / BASE_REL), sha256_file(v4 / BASE_REL))
    pins = base_pin_checks(v4)
    tests = _load(Path(a.tests_receipt))
    loader = _load(root / "receipts" / "DATA09_LOADER_VERIFY.json")
    g0 = _load(root / "receipts" / "DATA09_G0_INTEGRITY.json")
    guards = [[json.loads(x) for x in Path(g).read_text(encoding="utf-8").splitlines() if x.strip()][-1]
              for g in a.rss_log]
    stamps = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted(run.glob("*_utc.txt"))}
    df = {p.stem: p.read_text(encoding="utf-8").strip() for p in sorted(run.glob("df_*.txt"))}
    size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and "_run" not in p.parts)
    checks = {"build_receipt_pass": True,
              "manifests_on_disk": all(verify_manifest_on_disk_d9(v4, a.out_rel, m, gate)["all_ok"] for m in D9.TEST_IDS),
              "ext_registry_rule": chk["all_ok"],
              "ext_gate_eq_open_gate": all(ext.get("extension_test_gate", {}).get(k) == v
                                           for k, v in gate.binding().items() if k != "rule"),
              "base_registry_pinned_by_lock_v2": all(v is True for k, v in pins.items() if not k.startswith("_")),
              "loader_verify_self_hash": FC.verify_self_hash(loader, "receipt_sha256")
                                         and set(loader.get("roles", {})) == set(D9.TEST_ROLES),
              "g0_integrity_all_match": FC.verify_self_hash(g0, "receipt_sha256") and g0.get("all_match") is True,
              "tests_exit_0": tests.get("exit_code") == 0,
              "tests_negative_controls_failed_as_intended": tests.get("negative_controls", {}).get(
                  "all_failed_as_intended") is True,
              "rss_guards_rc_0": bool(guards) and all(g["rc"] == 0 and not g["killed_rss_guard"] for g in guards),
              "no_pycache_in_owned_code": not any(True for d in ("v4data09", "tests_data09")
                                                  for _ in (D9.CODE / d).rglob("__pycache__"))}
    rec = {"receipt": "DATA09_SERVER_RECEIPT", "task": D9.TASK, "overall_pass": all(checks.values()), "checks": checks,
           "date_u": stamps, "df": df, "rss_guards": guards, "data_test_bytes": int(size),
           "ext_registry": {"path": EXT_REL, "file_sha256": sha256_file(v4 / EXT_REL), "self_sha256": ext["self_sha256"],
                            "extends": ext["extends"], "extension_test_gate": ext["extension_test_gate"]},
           "build_receipt_sha256": brec["receipt_sha256"], "test_gate": gate.binding(),
           "tests": {k: tests.get(k) for k in ("exit_code", "counts", "negative_controls", "summary_line")},
           "tests_receipt_file_sha256": sha256_file(Path(a.tests_receipt)),
           "manifests": {m: {k: ext["manifests"][m][k] for k in ("n_users", "n_decisions", "n_eval_rows",
                                                                 "n_rankable_rows", "content_sha256", "header_sha256",
                                                                 "header_content_sha256")} for m in D9.TEST_IDS},
           "code": brec["code"], "finalize_code_sha256": sha256_file(HERE / "d9_finalize.py"),
           "test_split_read": "gated (build only)"}
    p, rec = FC.write_json_self_hashed(root, "receipts/DATA09_SERVER_RECEIPT.json", rec)
    print(json.dumps({"overall_pass": rec["overall_pass"], "receipt_sha256": rec["receipt_sha256"],
                      "file_sha256": sha256_file(p), "checks": checks}))
    return 0 if rec["overall_pass"] else 4


if __name__ == "__main__":
    sys.exit(main())
