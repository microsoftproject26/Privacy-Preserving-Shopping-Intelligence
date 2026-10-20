"""tools/g2_run.py — the PPSI V4 G2 runner (PREG1 amendment 05 §4-§5; task company/tasks/G2-01_RUNNER.md).

  python -B tools/g2_run.py --code-root <V4>/code_p3b --runs-root <V4>/runs_p3 --out <dir> [--preflight]
         [--print-values] [--init-hash-receipt <cross-check receipt.json>] [--expect-script-sha256 <hex>]

Every rule lives in the --code-root library (put FIRST on sys.path; each module's origin is checked): the unseal gate
(v4train.unseal), the within-family choice and T1 (v4train.selection), the paired user bootstrap
(v4train.bootstrap_ci), the lock draft (selection.write_lock_draft), the metric arithmetic (v4eval.metrics) and the
manifest loading path of the launcher (v4int). This script only discovers, checks bindings, reads and records.

Discovery: the pinned tuned-recipes registry (self-hash = the G1B current pin) names the recipes (C1 + C2 per family, or
C1 only for a family in no_c2); each recipe x seeds 2026 / 2027 must exist at <runs-root>/<FAMILY>/<recipe_id>_s<seed>
(v4train.p3_driver.run_dir_of); any other run directory under <runs-root>/GRU or /SASREC is refused.
Public checks (preflight AND full run, before unsealing; public lock / registry / manifest / RUN_START files only):
  manifests  MAIN_VALIDATION and CALIB_VALIDATION loaded through the launcher's path. eval_user_key is a per-manifest
             dense key, so CALIB users are mapped to MAIN through decision_id: every CALIB decision_id must be in MAIN,
             the CALIB -> MAIN user map must be 1:1, and the counts must be the registered 51,211 / 12,715 / 38,496.
  runs       each RUN_START: key_base.eval_manifest_id = MAIN_VALIDATION; key_base.manifest_registry_sha256 = the
             loaded REGISTRY_V2 file sha256, or the sha256 of the run's trainer view whose derived_from.file_sha256 is
             the loaded REGISTRY_V2 and whose MAIN_VALIDATION entry is identical; manifest_registry.bindings.eval =
             the loaded registry entry (id, n_decisions, bound hashes).
  init       the ARCHITECTURE_LOCK init-hash receipt is built from the public RUN_START.json of each recipe's two seeds
             (theta0_init_sha256 = the theta0_sha256 in key_base.initialization_block, torch, device, file sha256),
             bound to the recipe id by this script (the library's check is per family only).
--preflight: the public checks, then unseal_gate(...) whose token is discarded WITHOUT any sealed read; prints counts,
statuses and hashes only; writes nothing; exit 0 iff everything passes.
Full run: refuse an existing G2_STARTED.json / G2_RESULT.json / lock draft; the public checks (refusal = no unsealing);
G2_STARTED.json (O_EXCL attempt record) BEFORE unseal_gate; then the sealed reads (RESULT_ROWS.json, never
RESULT_ROWS_EXTENDED.json; RUN_SUMMARY.json; the PRACTICAL_BEST and 6.0 rank files), sealed binding checks (row
identity, query_order_hash, row value = library metric of the rank file), the CALIB-disjoint diagnostic rows (never
selection), within-family -> delta_bar_ci on the selected recipes -> select_g2_candidates, the diagnostic T1 (flip
report), and G2_RESULT.json (strict JSON, sort_keys, LF, O_EXCL). TIEBREAK_REQUIRED is a HOLD (never auto-resolved).
Only DECISIVE_STABLE writes ARCHITECTURE_LOCK.DRAFT.json.
Exit: 0 DECISIVE_STABLE / preflight pass, 2 refused before unsealing, 3 HOLD, 4 NEEDS_FABLE, 5 error after unsealing.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Optional

RESULT_NAME = "G2_RESULT.json"
STARTED_NAME = "G2_STARTED.json"
RESULT_SCHEMA = "PPSI_V4_G2_RESULT_2"
SEEDS = (2026, 2027)
PRIMARY_MANIFEST = "MAIN_VALIDATION_5_100_131072"
CALIB_MANIFEST = "CALIB_VALIDATION_5_100_32768"
DIAGNOSTIC_TAG = "CALIB_DISJOINT"
REGISTERED_COUNTS = {"main_users": 51211, "calib_users": 12715, "disjoint_users": 38496}   # amendment 02 §2
LIB_FILES = ("selection", "unseal", "bootstrap_ci", "tuned_recipes")
ROW_VALUE_TOL = 1e-9
REFUSED, EXIT = 2, {"DECISIVE_STABLE": 0, "HOLD": 3, "NEEDS_FABLE": 4, "ERROR": 5}
ENDPOINT_RANKS = "eval_6.0.ranks_int32.npy"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class G2Refused(RuntimeError):
    """Refused before any sealed value was read."""


class G2Error(RuntimeError):
    """A binding check failed after unsealing (no selection is made)."""


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def script_sha256() -> str:
    return _sha_file(Path(__file__).resolve())


# ------------------------------------------------------------------------------------------------ library binding
def bootstrap(code_root) -> SimpleNamespace:
    """--code-root first on sys.path; every library module used must come from that tree."""
    root = Path(code_root).resolve()
    if not (root / "v4train" / "selection.py").is_file():
        raise G2Refused(f"{root} is not a V4 code tree (no v4train/selection.py)")
    while str(root) in sys.path:
        sys.path.remove(str(root))
    sys.path.insert(0, str(root))
    if (root / "ppsi_env.py").is_file():
        import ppsi_env  # noqa: F401  (mandatory import isolation of the V4 tree)
    import numpy as np
    import v4eval.metrics as M
    import v4train.bootstrap_ci as BC
    import v4train.p3_driver as P3
    import v4train.selection as S
    import v4train.tuned_recipes as TR
    import v4train.unseal as U
    lib = SimpleNamespace(root=root, np=np, S=S, U=U, BC=BC, TR=TR, P3=P3, M=M)
    for name, mod in (("selection", S), ("unseal", U), ("bootstrap_ci", BC), ("tuned_recipes", TR),
                      ("p3_driver", P3), ("metrics", M)):
        check_origin(lib, mod, name)
    lib.code_files_sha256 = {f"v4train/{n}.py": _sha_file(root / "v4train" / f"{n}.py") for n in LIB_FILES}
    return lib


def check_origin(lib, mod, name: str) -> None:
    origin = Path(mod.__file__).resolve()
    if lib.root not in origin.parents:
        raise G2Refused(f"library module {name} was imported from {origin}, not from the code root {lib.root}")


# ------------------------------------------------------------------------------------------------ discovery
def load_pinned_registry(lib, gate_path=None, tuned_path=None, hp_path=None) -> dict:
    gate = lib.TR.load_g1b(gate_path)
    kw = {} if hp_path is None else {"hp_path": hp_path}
    reg = lib.TR.load_tuned_recipes(tuned_path or lib.TR.REGISTRY_PATH, **kw)
    if reg["self_sha256"] != gate["tuned_recipes_self_sha256"]:
        raise G2Refused(f"the tuned-recipes registry self-hash {reg['self_sha256']} is not the G1B current pin "
                        f"{gate['tuned_recipes_self_sha256']}")
    return reg


def expected_runs(lib, runs_root: Path, reg: Mapping) -> dict:
    out = {}
    for rid, r in sorted(reg["recipes"].items()):
        fam = str(r.get("family", "")).upper()
        if fam not in lib.S.FAMILIES or rid not in (lib.TR.recipe_id_of(fam), lib.TR.c2_id_of(fam)):
            raise G2Refused(f"registry recipe {rid!r} (family {r.get('family')!r}) is not a contest candidate id")
        for seed in SEEDS:
            out[(fam, rid, seed)] = Path(lib.P3.run_dir_of(Path(runs_root), fam, rid, seed))
    for fam in lib.S.FAMILIES:
        if lib.TR.recipe_id_of(fam) not in reg["recipes"]:
            raise G2Refused(f"{fam}: the pinned registry has no C1 recipe {lib.TR.recipe_id_of(fam)}")
        if (lib.TR.c2_id_of(fam) in reg["recipes"]) == (fam in reg.get("no_c2", [])):
            raise G2Refused(f"{fam}: exactly one of a C2 recipe or a no_c2 record is required")
    return out


def _extra_run_dirs(lib, runs_root: Path, expected: Mapping) -> list:
    want = {Path(d).resolve() for d in expected.values()}
    extra = []
    for fam in lib.S.FAMILIES:
        fd = Path(runs_root) / fam
        if fd.is_dir():
            extra += [str(c) for c in sorted(fd.iterdir()) if c.is_dir() and c.resolve() not in want]
    return extra


def discover(lib, runs_root, reg) -> dict:
    runs = expected_runs(lib, Path(runs_root), reg)
    missing = [f"{f}/{r}/{s}: {d}" for (f, r, s), d in runs.items() if not d.is_dir()]
    extra = _extra_run_dirs(lib, Path(runs_root), runs)
    if missing or extra:
        raise G2Refused(f"the contest run set is not exactly the {len(runs)} registered runs: missing {missing}, "
                        f"extra {extra}")
    return runs


def _key(k) -> str:
    return "/".join(str(x) for x in k)


# ------------------------------------------------------------------------------------------------ manifests
def production_context(lib) -> SimpleNamespace:
    """The launcher's data path: DATA_PROTOCOL_LOCK_V2 (G1-pinned) -> REGISTRY_V2 -> verify -> build_eval_manifest."""
    import v4eval.manifest as VM
    import v4int.launch_central as LC
    import v4int.providers as PR
    import v4int.verify as VF
    for name, mod in (("v4eval.manifest", VM), ("v4int.launch_central", LC), ("v4int.providers", PR),
                      ("v4int.verify", VF)):
        check_origin(lib, mod, name)
    paths = VF.DataPaths.default()
    ctx = LC.bind_data_protocol_lock_v2(paths, smoke=False)
    reg_path = Path(ctx["manifest_registry_path"])
    return SimpleNamespace(VM=VM, LC=LC, PR=PR, VF=VF, paths=paths, ctx=ctx, reg_path=reg_path,
                           reg_obj=_read_json(reg_path), reg_file_sha256=_sha_file(reg_path))


def load_eval_manifest(lib, pc, mid: str) -> tuple:
    vm = pc.VF.verify_manifest_by_registry(mid, pc.reg_path, pc.paths, root=pc.LC.V4, expect_kind="EVAL")
    m = pc.VM.build_eval_manifest(vm.header, {c: vm.columns[c] for c in pc.PR.EVAL_MANIFEST_COLUMNS if c in vm.columns})
    try:
        binding = dict(pc.PR.manifest_binding_of(vm))
    except Exception as e:                                     # recorded, the registry entry is the reference
        binding = {"error": f"{type(e).__name__}: {e}"}
    return m, binding


def _registry_record(pc, mids) -> dict:
    return {"manifest_registry_path": str(pc.reg_path), "manifest_registry_file_sha256": pc.reg_file_sha256,
            "lock_manifest_registry_file_sha256": pc.ctx.get("manifest_registry_file_sha256"),
            "entries": {mid: pc.reg_obj["manifests"][mid] for mid in mids}}


def load_manifests_production(lib) -> tuple:
    pc = production_context(lib)
    main, b_main = load_eval_manifest(lib, pc, PRIMARY_MANIFEST)
    calib, b_calib = load_eval_manifest(lib, pc, CALIB_MANIFEST)
    rec = dict(_registry_record(pc, (PRIMARY_MANIFEST, CALIB_MANIFEST)),
               bindings={PRIMARY_MANIFEST: b_main, CALIB_MANIFEST: b_calib})
    if rec["lock_manifest_registry_file_sha256"] not in (None, rec["manifest_registry_file_sha256"]):
        raise G2Refused("REGISTRY_V2 changed between the lock binding and this read")
    return main, calib, rec


def calib_mapping(lib, main, calib) -> dict:
    """CALIB users in MAIN keys, through decision_id (eval_user_key is dense per manifest). Refuses unless every CALIB
    decision is in MAIN and the CALIB -> MAIN user map is 1:1."""
    np = lib.np
    order = np.argsort(main.decision_id, kind="stable")
    sd = main.decision_id[order]
    pos = np.clip(np.searchsorted(sd, calib.decision_id), 0, max(sd.size - 1, 0))
    found = sd.size > 0 and bool(np.all(sd[pos] == calib.decision_id))
    if not found:
        n_missing = int(np.count_nonzero(~np.isin(calib.decision_id, main.decision_id)))
        raise G2Refused(f"{n_missing} CALIB_VALIDATION decision_id(s) are not in MAIN_VALIDATION")
    mapped = main.eval_user_key[order[pos]]
    ck = np.asarray(calib.eval_user_key)
    pairs = np.unique(np.stack([ck, mapped], axis=1), axis=0)
    n_c, n_m = int(np.unique(ck).size), int(np.unique(mapped).size)
    if not (pairs.shape[0] == n_c == n_m):
        raise G2Refused(f"the CALIB -> MAIN user map is not 1:1 ({n_c} CALIB keys, {n_m} MAIN keys, "
                        f"{pairs.shape[0]} pairs)")
    users = np.unique(mapped)
    main_users = np.unique(main.eval_user_key)
    return {"calib_users": users,
            "counts": {"main_users": int(main_users.size), "calib_users": int(users.size),
                       "disjoint_users": int(main_users.size - users.size)},
            "calib_users_main_keys_sha256": hashlib.sha256(np.ascontiguousarray(users, dtype=np.int64)
                                                           .tobytes()).hexdigest()}


def check_counts(mapping: Mapping) -> None:
    if dict(mapping["counts"]) != dict(REGISTERED_COUNTS):
        raise G2Refused(f"MAIN / CALIB / disjoint user counts {mapping['counts']} are not the registered "
                        f"{REGISTERED_COUNTS}")


def diagnostic_row_mask(lib, main, calib_users):
    return ~lib.np.isin(main.eval_user_key, calib_users)


def manifest_record(lib, main, mapping, mrec) -> dict:
    rows = diagnostic_row_mask(lib, main, mapping["calib_users"])
    np = lib.np
    return {"manifest_id": main.manifest_id, "n_rows": main.n, "hashes": main.hashes, "counts": mapping["counts"],
            "registered_counts": dict(REGISTERED_COUNTS), "loader": mrec,
            "calib_users_main_keys_sha256": mapping["calib_users_main_keys_sha256"],
            "diagnostic": {"tag": DIAGNOSTIC_TAG, "mapping": "CALIB decision_id -> MAIN eval_user_key (1:1)",
                           "n_eval_rows_disjoint": int(np.count_nonzero(main.eval_mask & rows)),
                           "n_eval_users_disjoint": int(np.unique(main.eval_user_key[main.eval_mask & rows]).size)}}


# ------------------------------------------------------------------------------------------------ public RUN_START checks
def _registry_route(start: Mapping, kb: Mapping, mrec: Mapping) -> str:
    loaded = mrec["manifest_registry_file_sha256"]
    got = kb.get("manifest_registry_sha256")
    if got == loaded:
        return "DIRECT"
    mr = start.get("manifest_registry") or {}
    view = Path(mr.get("path") or "")
    if mr.get("path") and view.is_file() and _sha_file(view) == got == mr.get("file_sha256"):
        v = _read_json(view)
        if (v.get("derived_from") or {}).get("file_sha256") == loaded and \
                (v.get("manifests") or {}).get(PRIMARY_MANIFEST) == mrec["entries"][PRIMARY_MANIFEST]:
            return "TRAINER_VIEW"
    raise G2Refused(f"key_base.manifest_registry_sha256 {got} is neither the loaded REGISTRY_V2 ({loaded}) nor a "
                    "trainer view derived from it with the same MAIN_VALIDATION entry")


def _check_eval_binding(start: Mapping, main, mrec: Mapping) -> None:
    b = ((start.get("manifest_registry") or {}).get("bindings") or {}).get("eval") or {}
    entry = mrec["entries"][PRIMARY_MANIFEST]
    if b.get("manifest_id") != PRIMARY_MANIFEST or b.get("n_decisions") is None \
            or int(b["n_decisions"]) != main.n or int(entry.get("n_decisions", -1)) != main.n:
        raise G2Refused("RUN_START manifest_registry.bindings.eval does not name the loaded MAIN_VALIDATION "
                        "(id / n_decisions)")
    compared = 0
    loaded = (mrec.get("bindings") or {}).get(PRIMARY_MANIFEST) or {}
    for h in ("content_sha256", "header_sha256", "fingerprint"):
        if entry.get(h) is not None:
            if b.get(h) != entry[h]:
                raise G2Refused(f"RUN_START bindings.eval.{h} differs from the loaded registry entry")
            compared += 1
        if loaded.get(h) is not None and b.get(h) is not None and loaded[h] != b[h]:
            raise G2Refused(f"RUN_START bindings.eval.{h} differs from the verified manifest")
    if not compared:
        raise G2Refused("the loaded MAIN_VALIDATION registry entry binds no hash to compare")


def run_start_checks(lib, runs: Mapping, main, mrec: Mapping) -> dict:
    routes = {}
    for key, rd in runs.items():
        start = _read_json(Path(rd) / "RUN_START.json")
        kb = start.get("key_base") or {}
        if kb.get("eval_manifest_id") != PRIMARY_MANIFEST:
            raise G2Refused(f"{_key(key)}: RUN_START key_base.eval_manifest_id is {kb.get('eval_manifest_id')!r}")
        routes[_key(key)] = _registry_route(start, kb, mrec)
        _check_eval_binding(start, main, mrec)
    return routes


def init_receipt(lib, runs: Mapping, reg: Mapping, fam: str, rid: str) -> dict:
    """The theta0 hashes of recipe `rid` from its two public RUN_START.json (measured by the server trainer),
    bound to the recipe here (selection._check_init_receipt looks up by family only)."""
    hashes, shas, torch, devices = {}, {}, set(), set()
    for seed in SEEDS:
        p = Path(runs[(fam, rid, seed)]) / "RUN_START.json"
        start = _read_json(p)
        cfg = start.get("config") or {}
        if cfg.get("recipe_id") != rid or str(cfg.get("family")).upper() != fam or int(cfg.get("seed", -1)) != seed:
            raise G2Refused(f"{fam}/{rid}/{seed}: RUN_START is not a run of this recipe and seed")
        if (start.get("tuned_recipe") or {}).get("recipe_sha256") != reg["recipes"][rid]["sha256"]:
            raise G2Refused(f"{fam}/{rid}/{seed}: RUN_START recipe_sha256 is not the pinned recipe's")
        h = start.get("theta0_init_sha256")
        prefix = f"I_META_RANDOM|{fam}|seed={seed}|theta0_sha256="
        blk = str((start.get("key_base") or {}).get("initialization_block") or "")
        if not (isinstance(h, str) and _HEX64.match(h)) or blk != prefix + h \
                or start.get("init_sha256") not in (None, h):
            raise G2Refused(f"{fam}/{rid}/{seed}: theta0_init_sha256 is missing or differs from "
                            "key_base.initialization_block / init_sha256")
        hashes[str(seed)], shas[str(seed)] = h, _sha_file(p)
        torch.add(start.get("torch"))
        devices.add(str(start.get("device") or ""))
    if len(torch) != 1 or not next(iter(torch)) or "" in devices:
        raise G2Refused(f"{fam}/{rid}: RUN_START torch {sorted(map(str, torch))} / device {sorted(devices)} are not "
                        "one recorded build")
    return {"source": "SERVER_MEASURED", "host": ",".join(sorted(devices)), "host_field": "RUN_START.device",
            "torch": next(iter(torch)), "hashes": {fam: hashes}, "recipe_id": rid, "family": fam,
            "run_start_sha256": shas,
            "derived_from": "public RUN_START.json theta0_init_sha256 of the selected recipe's seeds (tools/g2_run.py)"}


def public_checks(lib, runs, reg, main, calib, mrec) -> dict:
    """Everything that is checked from public files before any unsealing (refusal = G2Refused)."""
    if main.manifest_id != PRIMARY_MANIFEST or calib.manifest_id != CALIB_MANIFEST:
        raise G2Refused(f"loaded manifests {main.manifest_id!r} / {calib.manifest_id!r} are not "
                        f"{PRIMARY_MANIFEST} / {CALIB_MANIFEST}")
    mapping = calib_mapping(lib, main, calib)
    check_counts(mapping)
    routes = run_start_checks(lib, runs, main, mrec)
    receipts = {rid: init_receipt(lib, runs, reg, str(r["family"]).upper(), rid) for rid, r in reg["recipes"].items()}
    return {"mapping": mapping, "registry_routes": routes, "init_receipts": receipts}


# ------------------------------------------------------------------------------------------------ preflight
def _format_run_line(key, state: Mapping, sealed: bool, finished: bool) -> str:
    return (f"RUN {_key(key)}: status={state.get('status')} still_improving={state.get('still_improving')} "
            f"STATUS={state.get('status_file')} sealed_dir={sealed} complete={finished}")


def _discard_token(lib, token, runs) -> None:
    """The preflight token is dropped unused: no sealed file is opened."""
    del token


def preflight(lib, runs: Mapping, reg, *, gate_path=None, tuned_path=None, hp_path=None, out: Optional[Path] = None,
              manifest_loader=None, say=print) -> int:
    ok = True
    for key, rd in runs.items():
        state = lib.P3.public_state(rd)
        sealed = lib.U.is_sealed(rd)
        fin = lib.P3.finished(state) and state.get("status_file") == state.get("status")
        ok &= bool(sealed and fin)
        say(_format_run_line(key, state, sealed, fin))
    for name in (RESULT_NAME, STARTED_NAME):
        if out is not None and (Path(out) / name).exists():
            ok = False
            say(f"OUT: {Path(out) / name} already exists (a full run would refuse)")
    try:
        main, calib, mrec = (manifest_loader or load_manifests_production)(lib)
        say(f"MANIFEST {main.manifest_id}: rows {main.n} eval_ordered {main.hashes['eval_ordered_decision_id_sha256']}")
        say(f"MANIFEST {calib.manifest_id}: rows {calib.n} eval_ordered "
            f"{calib.hashes['eval_ordered_decision_id_sha256']}")
        say(f"REGISTRY {mrec['manifest_registry_file_sha256']}")
        pub = public_checks(lib, runs, reg, main, calib, mrec)
        c = pub["mapping"]["counts"]
        say(f"USERS main {c['main_users']} calib {c['calib_users']} (1:1 via decision_id) disjoint "
            f"{c['disjoint_users']}: registered counts OK")
        say(f"RUN_START bindings OK for {len(pub['registry_routes'])} runs "
            f"(registry {sorted(set(pub['registry_routes'].values()))})")
        say(f"INIT RECEIPTS buildable for {sorted(pub['init_receipts'])}")
    except G2Refused as e:
        ok = False
        say(f"PUBLIC CHECKS: REFUSED ({e})")
    except Exception as e:
        ok = False
        say(f"PUBLIC CHECKS: REFUSED ({type(e).__name__}: {e})")
    try:
        token = lib.U.unseal_gate(runs, gate_path=gate_path, tuned_path=tuned_path, hp_path=hp_path)
    except lib.U.UnsealRefused as e:
        say(f"GATE: REFUSED ({e})")
        return REFUSED
    n = len(token.record.get("configs", {}))
    _discard_token(lib, token, runs)
    say(f"GATE: PASS for {n} runs (token discarded; no sealed file read)")
    say(f"SCRIPT sha256 {script_sha256()}")
    say(f"PREFLIGHT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else REFUSED


# ------------------------------------------------------------------------------------------------ sealed evidence
def read_evidence(lib, token, runs: Mapping) -> dict:
    """The ONLY sealed reads: RESULT_ROWS.json, RUN_SUMMARY.json and two rank files per run, through the token."""
    ev = {"rows": {}, "summary": {}, "pb_ranks": {}, "end_ranks": {}, "files": {}}
    for key, rd in runs.items():
        root = lib.U.sealed_root(rd, token)
        ev["rows"][key] = list(lib.U.read_sealed_json(token, rd, "RESULT_ROWS.json")["rows"])
        ev["summary"][key] = lib.U.read_sealed_json(token, rd, "RUN_SUMMARY.json")
        ev["pb_ranks"][key] = lib.BC.practical_best_ranks(rd, token)
        ev["end_ranks"][key] = lib.np.load(root / "evals" / ENDPOINT_RANKS).astype(lib.np.int64)
        mark = float(ev["summary"][key]["practical_best"]["mark"])
        ev["files"][_key(key)] = {
            "RESULT_ROWS.json": _sha_file(root / "RESULT_ROWS.json"),
            "RUN_SUMMARY.json": _sha_file(root / "RUN_SUMMARY.json"),
            f"evals/eval_{mark:.1f}.ranks_int32.npy": _sha_file(root / "evals" / f"eval_{mark:.1f}.ranks_int32.npy"),
            f"evals/{ENDPOINT_RANKS}": _sha_file(root / "evals" / ENDPOINT_RANKS),
            "rows_sha256": lib.S.rows_sha256(ev["rows"][key])}
    return ev


def _is_macro_e2e(r: Mapping) -> bool:
    return r.get("aggregation") == "macro" and r.get("target_denominator") == "E2E"


def _ranks_for(ev, key, row):
    rule = row.get("selection_rule")
    if rule == "PRACTICAL_BEST":
        return ev["pb_ranks"].get(key)
    if rule == "CONTROLLED_BUDGET":
        return ev["end_ranks"].get(key)
    return None


def _macro(lib, ranks, mask, users, metric: str, k: int):
    credits = lib.M.credits_from_ranks(ranks, (int(k),))
    v = lib.M.population_block({f"{metric}@{k}": credits[f"{metric}@{k}"]}, mask, users)
    return v["macro"][f"{metric}@{k}"], v["n_decisions"], v["n_users"]


def check_bindings(lib, ev, runs, manifest, *, manifest_id: str = PRIMARY_MANIFEST,
                   check_identity: bool = True) -> dict:
    """Row identity (optional: TUNING rows carry no recipe_id), the manifest binding (eval_manifest_id,
    query_order_hash) and rank-file / row-value agreement of every defined macro-E2E mrr@20 / ndcg@10 row."""
    qhash = manifest.hashes["eval_ordered_decision_id_sha256"]
    worst, n_checked, n_skipped = 0.0, 0, 0
    for key in runs:
        pb = (ev["summary"].get(key) or {}).get("practical_best") or {}
        for r in ev["rows"][key]:
            if check_identity:
                fam, rid, seed = key
                if (str(r.get("family", "")).upper(), r.get("recipe_id"), int(r.get("seed", -1))) != (fam, rid, seed):
                    raise G2Error(f"{_key(key)}: a row names family/recipe/seed {r.get('family')}/"
                                  f"{r.get('recipe_id')}/{r.get('seed')}")
            if r.get("eval_manifest_id") != manifest_id or r.get("query_order_hash") != qhash:
                raise G2Error(f"{_key(key)}: a row is not bound to the loaded {manifest_id} manifest "
                              "(eval_manifest_id / query_order_hash)")
            if r.get("selection_rule") == "PRACTICAL_BEST" and abs(float(r["efe"]) - float(pb["efe"])) > 1e-9:
                raise G2Error(f"{_key(key)}: a PRACTICAL_BEST row is not at the summary's practical-best mark")
            ranks = _ranks_for(ev, key, r)
            if ranks is None or not _is_macro_e2e(r) or r.get("metric") not in ("mrr", "ndcg") \
                    or int(r.get("k", -1)) not in (10, 20):
                continue
            if ranks.shape != (manifest.n,):
                raise G2Error(f"{_key(key)}: rank file has {ranks.shape[0]} entries, manifest {manifest.n}")
            if r.get("value") is None or r.get("value_status") != "OK" or not math.isfinite(float(r["value"])):
                n_skipped += 1                      # an undefined / non-finite row: the library's exclusion rule
                continue
            got = _macro(lib, ranks, manifest.eval_mask, manifest.eval_user_key, r["metric"], r["k"])[0]
            if not isinstance(got, float):
                raise G2Error(f"{_key(key)}: the recomputation of a defined evidence row is undefined")
            d = abs(got - float(r["value"]))
            if not d <= ROW_VALUE_TOL:
                raise G2Error(f"{_key(key)}: a {r['selection_rule']} {r['metric']}@{r['k']} row disagrees with its "
                              "rank file")
            worst, n_checked = max(worst, d), n_checked + 1
    return {"query_order_hash": qhash, "rank_row_rows_checked": n_checked, "rank_row_rows_undefined": n_skipped,
            "rank_row_max_abs_diff": worst, "rank_row_tolerance": ROW_VALUE_TOL}


def diagnostic_rows(lib, ev, runs, main, calib_users) -> list:
    """The CALIB-disjoint MAIN_VALIDATION rows (diagnostic, never selection): the macro E2E rows of each run
    recomputed on the eval rows of MAIN users that are not CALIB users, from the same rank files."""
    sub = main.eval_mask & diagnostic_row_mask(lib, main, calib_users)
    out = []
    for key in runs:
        for r in ev["rows"][key]:
            ranks = _ranks_for(ev, key, r)
            if ranks is None or not _is_macro_e2e(r):
                continue
            v, n_dec, n_users = _macro(lib, ranks, sub, main.eval_user_key, r["metric"], r["k"])
            ok = isinstance(v, float) and math.isfinite(v)
            out.append(dict(r, value=(v if ok else None), value_status=("OK" if ok else "UNDEFINED"),
                            n_decisions=n_dec, n_users=n_users,
                            eval_manifest_id=f"{PRIMARY_MANIFEST}#{DIAGNOSTIC_TAG}",
                            diagnostic=f"{DIAGNOSTIC_TAG}: never selection evidence"))
    return out


def _diag_values(rows) -> dict:
    out = {}
    for r in rows:
        if r.get("metric") == "mrr" and int(r.get("k", -1)) == 20:
            out.setdefault(r["selection_rule"], {}).setdefault(r["recipe_id"], {})[str(r["seed"])] = r.get("value")
    return out


def _selected_ranks(ev, selected: Mapping) -> dict:
    return {(fam, seed): ev["pb_ranks"][(fam, rid, seed)] for fam, rid in selected.items() for seed in SEEDS}


# ------------------------------------------------------------------------------------------------ writing
def _strict(obj):
    """JSON-safe copy: str keys, numpy scalars / arrays to Python, non-finite floats as explicit strings."""
    if isinstance(obj, Mapping):
        return {str(_key(k) if isinstance(k, tuple) else k): _strict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        items = sorted(obj) if isinstance(obj, (set, frozenset)) else obj
        return [_strict(v) for v in items]
    if hasattr(obj, "tolist") and not isinstance(obj, (str, bytes)):
        return _strict(obj.tolist())
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else f"NON_FINITE:{obj!r}"
    return str(obj)


def dumps(obj) -> str:
    return json.dumps(_strict(obj), sort_keys=True, allow_nan=False, indent=1, ensure_ascii=True) + "\n"


def _write_new(path: Path, text: str) -> None:
    """O_EXCL: an existing file is never replaced."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o644)
    with os.fdopen(fd, "wb") as f:
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------------------------------------------ full run
def _lock_draft(lib, decision, selected, out: Path, receipts: Mapping, cross_path, sha: str) -> dict:
    fam = decision.get("winner_family")
    rid = selected[fam]
    receipt = receipts[rid]
    if receipt.get("recipe_id") != rid or receipt.get("family") != fam or set(receipt["hashes"]) != {fam}:
        return {"written": False, "reason": f"the init receipt is not bound to the selected recipe {rid}"}
    extra = {"selected_recipe_id": rid, "g2_result": RESULT_NAME, "g2_script_sha256": sha,
             "init_hash_receipt": receipt}
    if cross_path is not None:
        try:
            other = _read_json(cross_path)
            if (other.get("hashes") or {}).get(fam) != receipt["hashes"][fam]:
                return {"written": False, "reason": "the --init-hash-receipt cross-check disagrees with RUN_START"}
            extra["init_hash_cross_check_sha256"] = _sha_file(cross_path)
        except (OSError, ValueError) as e:
            return {"written": False, "reason": f"--init-hash-receipt unreadable: {type(e).__name__}: {e}"}
    try:
        path = out / lib.S.DRAFT_NAME
        fsha = lib.S.write_lock_draft(decision, path, init_hash_receipt=receipt, extra=extra)
        return {"written": True, "path": str(path), "file_sha256": fsha, "recipe_id": rid}
    except (lib.S.SelectionError, OSError, ValueError, KeyError) as e:
        return {"written": False, "reason": f"{type(e).__name__}: {e}"}


def _decide(lib, token, ev, runs, reg, main, calib_users, rec) -> dict:
    S, BC = lib.S, lib.BC
    rec["checks"] = check_bindings(lib, ev, runs, main)
    rows = [r for key in runs for r in ev["rows"][key]]
    diag = diagnostic_rows(lib, ev, runs, main, calib_users)
    rec["inputs"]["primary_rows_sha256"] = S.rows_sha256(rows)
    rec["inputs"]["diagnostic_rows_sha256"] = S.rows_sha256(diag)
    recipes = reg["recipes"]
    within = S.within_family_candidates(rows, recipes, unseal_token=token, diagnostic_rows=diag)
    selected = {fam: within[fam]["selected_recipe_id"] for fam in S.FAMILIES}
    ci = BC.delta_bar_ci(main, _selected_ranks(ev, selected))
    res = S.select_g2_candidates(rows, recipes, unseal_token=token, delta_bar_ci=ci["ci95"], diagnostic_rows=diag)
    if res["within_family"] != within:
        raise G2Error("select_g2_candidates chose other recipes than its own within-family step")
    rec.update(within_family=res["within_family"], between_families=res["between_families"], delta_bar_ci=ci,
               selected_recipe_ids=selected)
    dmain = dataclasses.replace(main, eval_mask=main.eval_mask & diagnostic_row_mask(lib, main, calib_users))
    dd = {"values_mrr20_macro_e2e": _diag_values(diag),
          "within_family": {f: {k: res["within_family"][f].get(k) for k in
                                ("diagnostic_selected_recipe_id", "diagnostic_flip", "diagnostic_error")}
                            for f in S.FAMILIES}}
    try:
        dci = BC.delta_bar_ci(dmain, _selected_ranks(ev, selected))
        sel = set(selected.values())
        ddec = S.select_family([r for r in diag if r.get("recipe_id") in sel
                                and r.get("selection_rule") == "PRACTICAL_BEST"], delta_bar_ci=dci["ci95"])
        main_dec = res["between_families"]
        dd["between_families"] = {"decision": ddec, "delta_bar_ci": dci,
                                  "flip": (ddec["status"], ddec["winner_family"]) != (main_dec["status"],
                                                                                      main_dec["winner_family"])}
    except Exception as e:                                       # a diagnostic never blocks the decision
        dd["between_families"] = {"error": f"{type(e).__name__}: {e}", "flip": None}
    rec["diagnostic"] = dd
    return selected


def full_run(lib, runs, reg, args, *, gate_path=None, tuned_path=None, hp_path=None, manifest_loader=None,
             say=print) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result_path, started_path = out / RESULT_NAME, out / STARTED_NAME
    for p in (started_path, result_path, out / lib.S.DRAFT_NAME):
        if p.exists():
            raise G2Refused(f"{p} already exists: G2 never overwrites (O_EXCL); use a new --out")
    sha = script_sha256()
    if args.expect_script_sha256 and args.expect_script_sha256 != sha:
        raise G2Refused(f"the G2 script sha256 {sha} is not the frozen {args.expect_script_sha256}")
    try:
        main, calib, mrec = (manifest_loader or load_manifests_production)(lib)
    except G2Refused:
        raise
    except Exception as e:
        raise G2Refused(f"the eval manifests could not be loaded: {type(e).__name__}: {e}") from None
    pub = public_checks(lib, runs, reg, main, calib, mrec)
    mapping = pub["mapping"]
    rec = {"schema": RESULT_SCHEMA, "utc_start": _utc(),
           "script": {"path": str(Path(__file__).resolve()), "sha256": sha}, "code_root": str(lib.root),
           "code_files_sha256": lib.code_files_sha256, "runs_root": str(Path(args.runs_root).resolve()),
           "runs": {_key(k): str(d) for k, d in runs.items()},
           "tuned_recipes": {"self_sha256": reg["self_sha256"], "recipes": sorted(reg["recipes"]),
                             "no_c2": reg.get("no_c2", [])},
           "manifest": manifest_record(lib, main, mapping, mrec),
           "public_checks": {"registry_routes": pub["registry_routes"], "init_receipts": pub["init_receipts"]},
           "never_selection": ["PRACTICAL_BEST_EXTENDED (RESULT_ROWS_EXTENDED.json is never read)",
                               f"{DIAGNOSTIC_TAG} diagnostic rows"]}
    _write_new(started_path, dumps({"schema": "PPSI_V4_G2_STARTED_1", "utc": _utc(), "script_sha256": sha,
                                    "code_root": str(lib.root), "code_files_sha256": lib.code_files_sha256,
                                    "runs": rec["runs"], "note": "attempt record written BEFORE unseal_gate"}))
    rec["started_record_sha256"] = _sha_file(started_path)
    try:
        token = lib.U.unseal_gate(runs, gate_path=gate_path, tuned_path=tuned_path, hp_path=hp_path)
    except lib.U.UnsealRefused as e:
        raise G2Refused(f"unseal gate refused: {e} (attempt recorded in {started_path})") from None
    rec["unseal_token"] = {"record": dict(token.record), "run_dirs": sorted(token.run_dirs)}
    selected = None
    try:
        ev = read_evidence(lib, token, runs)
        rec["inputs"] = {"per_run": ev["files"]}
        selected = _decide(lib, token, ev, runs, reg, main, mapping["calib_users"], rec)
        status = rec["between_families"]["status"]
        outcome = "DECISIVE_STABLE" if status == "DECISIVE_STABLE" else "HOLD"
        rec["hold"] = None if outcome != "HOLD" else {
            "status": status, "note": "TIEBREAK_REQUIRED / ARCHITECTURE_EQUIVALENT is a HOLD: never auto-resolved "
                                      "(no tie-break run, no owner ratification is made by this script)"}
    except lib.S.NeedsFable as e:
        outcome, rec["needs_fable"] = "NEEDS_FABLE", str(e)
    except Exception as e:
        outcome, rec["error"] = "ERROR", f"{type(e).__name__}: {e}"
    rec["outcome"] = outcome
    rec["lock_draft"] = (_lock_draft(lib, rec["between_families"], selected, out, pub["init_receipts"],
                                     args.init_hash_receipt, sha)
                         if outcome == "DECISIVE_STABLE" else
                         {"written": False, "reason": f"outcome {outcome}: no ARCHITECTURE_LOCK draft"})
    rec["utc"] = _utc()
    _write_new(result_path, dumps(rec))
    for line in console_lines(rec, print_values=args.print_values):
        say(line)
    say(f"G2 RESULT: {result_path} sha256 {_sha_file(result_path)}")
    return EXIT[outcome]


def console_lines(rec: Mapping, *, print_values: bool = False) -> list:
    lines = []
    for fam, w in sorted((rec.get("within_family") or {}).items()):
        labs = " | ".join(f"{lab} {rid}" for rid, lab in sorted(w["labels"].items(), key=lambda x: x[1] != "SELECTED"))
        s = (f"G2 WITHIN {fam}: {labs} | excluded {sorted(w['excluded'].items()) or 'none'} | "
             f"diagnostic_flip {w.get('diagnostic_flip')}")
        if print_values:
            s += f" | means {w['means']}"
        lines.append(s)
    b = rec.get("between_families")
    if b:
        fired = [t for t, v in b["cohort"]["triggers"].items() if v]
        s = (f"G2 BETWEEN: {b['status']} winner {b['winner_family']} basis {b['selection_basis']} "
             f"triggers {fired or 'none'}")
        if print_values:
            s += f" | delta_bar {b['cohort']['delta_bar']} ci95 {b['cohort']['delta_bar_ci95']}"
        lines.append(s)
        d = (rec.get("diagnostic") or {}).get("between_families") or {}
        dec = d.get("decision") or {}
        lines.append(f"G2 DIAGNOSTIC {DIAGNOSTIC_TAG} BETWEEN: {dec.get('status', d.get('error'))} winner "
                     f"{dec.get('winner_family')} flip {d.get('flip')}")
    detail = rec.get("error") or rec.get("needs_fable") or ""
    if detail and not print_values:
        detail = detail.split(":")[0] + " (message in G2_RESULT.json)"
    lines.append(f"G2 OUTCOME: {rec['outcome']}" + (f" [{detail}]" if detail else ""))
    ld = rec.get("lock_draft") or {}
    lines.append(f"G2 LOCK DRAFT: {'written ' + ld['path'] if ld.get('written') else 'not written'}")
    return lines


# ------------------------------------------------------------------------------------------------ CLI
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--code-root", required=True, type=Path)
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--print-values", action="store_true")
    ap.add_argument("--init-hash-receipt", type=Path, default=None,
                    help="optional cross-check of the RUN_START-derived theta0 hashes")
    ap.add_argument("--expect-script-sha256", default=None)
    return ap.parse_args(argv)


def main(argv=None, *, gate_path=None, tuned_path=None, hp_path=None, manifest_loader=None, say=print) -> int:
    """CLI entry. The keyword overrides exist for the synthetic tests only (the CLI always uses the library
    defaults: company/gates/G1B_APPROVED.json, contracts/hp_registry/TUNED_RECIPES_02B.json, the locked manifests)."""
    args = parse_args(argv)
    try:
        lib = bootstrap(args.code_root)
        reg = load_pinned_registry(lib, gate_path, tuned_path, hp_path)
        runs = discover(lib, args.runs_root, reg)
        if args.preflight:
            return preflight(lib, runs, reg, gate_path=gate_path, tuned_path=tuned_path, hp_path=hp_path,
                             out=args.out, manifest_loader=manifest_loader, say=say)
        return full_run(lib, runs, reg, args, gate_path=gate_path, tuned_path=tuned_path, hp_path=hp_path,
                        manifest_loader=manifest_loader, say=say)
    except G2Refused as e:
        say(f"G2 REFUSED: {e}")
        return REFUSED
    except Exception as e:                  # registry / gate file errors before unsealing
        name = type(e).__name__
        if name in ("TunedRecipeError", "UnsealRefused", "HPRegistryError"):
            say(f"G2 REFUSED: {name}: {e}")
            return REFUSED
        raise


if __name__ == "__main__":
    sys.exit(main())
