"""v4int.test_roles (owner: Data engineer, DATA-09) - the LOADER gate and the TEST-role resolution shim of the single TEST
look (PREG1_AMENDMENT_06D §2; EVALUATION_CONTRACT_V4 E10.1 "the builder and the loader both refuse otherwise").

Every resolution of a role or manifest id containing "TEST" calls v4data09.d9_gate.check_test_gate FIRST (ALL_METHODS_LOCK
exists, its sha256 is the first MANIFEST_BUILD / RETRO_TEST entry of the hash-chain-verified TEST_EXPOSURE_REGISTER) and
refuses otherwise - before the registry extension, a TEST header or a TEST table is opened. The gate is re-checked on every
call (no cached gate). A TEST manifest is served only under the gate it was built under (its header's
all_methods_lock_sha256 / test_exposure_register_entry_id == the open gate's).

Public API (INT-06 / INT-07 call these; nothing here edits launch_central / launch_fl / launch_device / v4base / v4dev):
  resolve_role(role, *, ext_path, lock_path, register_path) -> {"role", "train", "eval", "gate", "registry", ...}
  test_gate_ok(role, *, lock_path, register_path, ext_path) -> bool      (reeval.py hook; never raises)
  verify_test_manifest(mid, *, ext_path, paths, root, ...) -> v4int.verify.VerifiedManifest   (gated registry path)
  test_eval_manifest(vm, *, lock_path, register_path) -> v4eval EvalManifest (build_eval_manifest + ALL_METHODS_LOCK)
  install(ext_path, *, role, lock_path, register_path, LC=None, eager=True) / uninstall()
      in-process wrappers (the v4int.c_full pattern), each delegating UNCHANGED for every non-TEST path / id:
        W1 launch_central.bind_data_protocol_lock_v2   original bind (LOCK_V2 pin, REGISTRY_V2 file sha); then the gate;
           then REGISTRY_V2_TEST_EXT.json verified against that exact REGISTRY_V2 (d9_registry.verify_test_ext_registry)
           and handed to the providers as the manifest registry;
        W2 launch_central.build_trainer_registry_view  for the extension only: REGISTRY_V2_TRAINER_VIEW.json with MAIN /
           CONTINUATION = the installed TEST role's {train, eval} pair (eval-only TRUE_COLD_TEST is refused here);
        W3 v4int.verify.verify_registry                 for the extension path only: gate -> the ORIGINAL function on
           REGISTRY_V2 (F3 pin + B1 route) -> the extension rule;
        W4 v4int.verify.verify_manifest_by_registry and v4int.providers.{verify_manifest_by_registry,
           refuse_test_manifest, build_eval_manifest}   for TEST ids / RETRO_TEST headers only: gate ->
           verify_test_manifest / the ALL_METHODS_LOCK path handed to v4eval's own TEST gate;
        W5 v4train.registry.load_manifest_registry     for the views W2 wrote only: gate -> the loader's checks with the
           TEST-id refusal lifted for the registered RETRO_TEST ids (so expected_manifests(...) resolves). The
           CentralTrainer binds the loader at import and therefore still refuses a TEST view (no training on TEST).
      With eager=False, W3-W5 are installed by W1 (after the launcher's CUDA export), exactly like c_full.

NOT covered (documented for INT-07, see company/handoffs/DATA-09.md): the Clean V2 feature store (v4int.cache /
v4hist.policy / v4data.store) has no TEST path - StoreEvalProvider(...) resolves, verifies and builds the EvalManifest of
a RETRO_TEST manifest, then build_subset_store(splits=VALIDATION_SPLITS) refuses its TEST decision ids (fail-closed).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
V4 = CODE.parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from v4data.common import TestGateError  # noqa: E402
from v4data09 import d9_common as D9  # noqa: E402
from v4data09 import d9_registry as R  # noqa: E402
from v4data09.d9_gate import GateRecord, TestGateRefused, check_test_gate, header_matches_gate  # noqa: E402

DEFAULT_EXT = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2_TEST_EXT.json"
BASE_REGISTRY = V4 / "contracts" / "manifest_registry" / "REGISTRY_V2.json"
MARK = "__d9_test_roles__"
TEST_ROLES = D9.TEST_ROLES
TEST_IDS = D9.TEST_IDS


class TestRoleRefused(TestGateRefused):
    """A TEST role / manifest was requested and the gate, the extension or the header binding refused it."""


def is_test_name(x) -> bool:
    return "TEST" in str(x).upper()


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _strict(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"),
                          parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"non-strict constant {c}")))
    except (OSError, ValueError) as e:
        raise TestRoleRefused(f"{p}: not readable strict JSON ({e})") from None


def gate(lock_path: Optional[Path] = None, register_path: Optional[Path] = None) -> GateRecord:
    return check_test_gate(lock_path, register_path)


def load_test_ext(ext_path: Path, g: GateRecord, *, base_path: Optional[Path] = None,
                  base_file_sha256: Optional[str] = None) -> dict:
    """The TEST registry extension, read only under an open gate: self-hash, the gate binding == the open gate, and
    (when a base is given) the full extension rule against that REGISTRY_V2."""
    if not isinstance(g, GateRecord):
        raise TestRoleRefused("an open TEST gate (GateRecord) is required before the TEST extension is read")
    p = Path(ext_path)
    if not p.is_file():
        raise TestRoleRefused(f"the TEST registry extension {p} does not exist (run the DATA-09 server jobs)")
    ext = _strict(p)
    if not R.self_ok(ext):
        raise TestRoleRefused(f"{p}: self_sha256 does not verify")
    eg = ext.get("extension_test_gate") or {}
    if eg.get("all_methods_lock_sha256") != g.lock_sha256 or eg.get("test_exposure_register_entry_id") != g.entry_id:
        raise TestRoleRefused(f"{p}: built under another TEST gate (lock / register entry) than the open one")
    if base_path is not None:
        base = _strict(base_path)
        chk = R.verify_test_ext_registry(ext, base, base_file_sha256 or _sha(base_path))
        if not chk["all_ok"]:
            raise TestRoleRefused(f"{p}: fails the extension rule: {[k for k, v in chk.items() if not v]}")
    return ext


# ------------------------------------------------------------------------------------------------ resolution
def resolve_role(role: str, *, ext_path: Path = DEFAULT_EXT, lock_path: Optional[Path] = None,
                 register_path: Optional[Path] = None, base_path: Path = BASE_REGISTRY) -> dict:
    """(train, eval) of a role. A TEST role: gate first, then the extension (verified against REGISTRY_V2). Any other
    role: REGISTRY_V2 itself, no gate."""
    if is_test_name(role):
        g = check_test_gate(lock_path, register_path)
        if role not in TEST_ROLES:
            raise TestRoleRefused(f"{role!r} is not a registered TEST role {sorted(TEST_ROLES)}")
        ext = load_test_ext(ext_path, g, base_path=base_path)
        pair = (ext.get("roles") or {}).get(role)
        if pair != TEST_ROLES[role]:
            raise TestRoleRefused(f"the extension maps {role} to {pair!r}, not {TEST_ROLES[role]!r}")
        return {"role": role, "train": pair.get("train"), "eval": pair["eval"], "gate": g,
                "registry": str(ext_path), "registry_self_sha256": ext["self_sha256"]}
    base = _strict(base_path)
    pair = (base.get("roles") or {}).get(role)
    if not isinstance(pair, dict):
        raise KeyError(f"role {role!r} is not in {base_path}")
    return {"role": role, "train": pair.get("train"), "eval": pair.get("eval"), "panel": pair.get("panel"),
            "gate": None, "registry": str(base_path), "registry_self_sha256": base.get("self_sha256")}


def test_gate_ok(role: str, *, lock_path: Optional[Path] = None, register_path: Optional[Path] = None,
                 ext_path: Optional[Path] = None) -> bool:
    """The reeval.py hook (INT-06): True for any non-TEST role; for a TEST role True iff it is registered and the gate is
    open (and, when ext_path is given, the extension was built under that same gate). Never raises."""
    if not is_test_name(role):
        return True
    if role not in TEST_ROLES:
        return False
    try:
        g = check_test_gate(lock_path, register_path)
        if ext_path is not None:
            load_test_ext(ext_path, g)
        return True
    except (TestGateError, OSError, ValueError):
        return False


# ------------------------------------------------------------------------------------------------ gated verification
def _orig(mod, name):
    f = getattr(mod, name)
    while getattr(f, MARK, False) or getattr(f, "__c_full__", False):
        f = f.__wrapped__
    return f


def verify_test_manifest(manifest_id: str, *, ext_path: Path = DEFAULT_EXT, paths=None, root: Optional[Path] = None,
                         expect_kind: Optional[str] = "EVAL", lock_path: Optional[Path] = None,
                         register_path: Optional[Path] = None, log=None):
    """v4int.verify.verify_manifest_by_registry for a RETRO_TEST manifest of the extension (the same checks; the TEST
    refusals replaced by: the open gate, header gate fields == the gate == the registry entry, v4_split RETRO_TEST,
    status == the TEST role, rows old_split TEST / v4_split RETRO_TEST; tables re-read with the extended field
    dictionary). Refuses before any file is opened unless the gate is open."""
    g = check_test_gate(lock_path, register_path)                       # FIRST
    mid = str(manifest_id)
    if mid not in TEST_IDS:
        raise TestRoleRefused(f"{mid} is not a registered RETRO_TEST manifest {TEST_IDS}")
    import numpy as np
    import pyarrow.parquet as pq
    import v4int.verify as VF
    from v4data.common import ordered_ids_sha256, table_content_sha256
    say = log or (lambda *_: None)
    t0 = time.time()
    paths = paths or VF.DataPaths.default()
    root = Path(root) if root is not None else V4
    checks: dict = {}
    lock, lself, lfile = VF.verify_lock(paths)
    checks["lock_self_sha256_recomputed"] = True
    ext = load_test_ext(ext_path, g)
    base_path = root / str((ext.get("extends") or {}).get("path", ""))
    base, base_file, base_self, v1_ctx = _orig(VF, "verify_registry")(base_path, lself, lfile, lock=lock, root=root)
    chk = R.verify_test_ext_registry(ext, base, base_file)
    if not chk["all_ok"]:
        raise TestRoleRefused(f"{ext_path}: fails the extension rule: {[k for k, v in chk.items() if not v]}")
    checks["registry.base_verified_and_extension_rule"] = True
    ref_lock, ref_self, ref_file = v1_ctx if v1_ctx is not None else (lock, lself, lfile)
    entry = ext["manifests"].get(mid)
    kind = VF.manifest_kind(entry)
    if kind != "EVAL" or entry.get("kind") != "EVAL" or (expect_kind is not None and kind != expect_kind):
        raise VF.ManifestVerificationError(f"{mid}: a {kind} manifest, not EVAL")
    ts_self, ts_file = VF.verify_transform_state(paths, lock)
    rts = ext.get("transform_state") or {}
    checks["registry.transform_state_eq_lock"] = rts.get("transform_state_sha256") == ts_self and \
        rts.get("file_sha256") == ts_file
    t_lock = time.time()
    tables, files = {}, {}
    for t in VF.TABLES[kind]:
        e = entry["tables"].get(t)
        f = root / str((e or {}).get("file", ""))
        if not e or not f.is_file():
            raise VF.ManifestVerificationError(f"{mid}: table {t} missing from the registry entry or from disk ({f})")
        fsha = VF.sha256_file(f)
        checks[f"{t}.file_sha256_eq_registry"] = fsha == e.get("file_sha256")
        tables[t] = {"file": f.name, "file_sha256": fsha, "row_count": int(e["row_count"]),
                     "table_content_sha256": e["table_content_sha256"]}
        files[t] = f
    hp = root / str(entry.get("path", ""))
    if not hp.is_file():
        raise VF.ManifestVerificationError(f"{mid}: header file {hp} is missing")
    hsha = VF.sha256_file(hp)
    header = VF.strict_json(hp)
    role = D9.ROLE_OF_ID[mid]
    checks.update({
        "registry.header_sha256": hsha == entry.get("header_sha256"),
        "header.manifest_id": header.get("manifest_id") == mid,
        "header.status_is_test_role": header.get("status") == role == entry.get("status") == header.get("run_label"),
        "header.data_version_eq_registry": header.get("data_version") == ext.get("data_version"),
        "header.v4_split_RETRO_TEST": header.get("v4_split") == "RETRO_TEST",
        "header.gate_eq_open_gate": header_matches_gate(header, g),
        "header.gate_eq_registry_entry": all(header.get(k) == entry.get(k) for k in R.GATE_KEYS),
        "header.data_protocol_lock_sha256_eq_lock_self": header.get("data_protocol_lock_sha256") == ref_self,
        "header.data_protocol_lock_file_sha256_eq_lock_file": header.get("data_protocol_lock_file_sha256") == ref_file,
        "header.content_self_hash": header.get("header_content_sha256") == VF.header_content_sha256(header),
        "registry.header_content_sha256": header.get("header_content_sha256") == entry.get("header_content_sha256"),
        "header.tables_eq_registry": {x["table"]: {k: x[k] for k in ("row_count", "table_content_sha256", "file_sha256")}
                                      for x in header.get("tables", [])} == {
            t: {k: v[k] for k in ("row_count", "table_content_sha256", "file_sha256")} for t, v in entry["tables"].items()},
        "header.builder_code_sha256_eq_lock": header.get("builder_code_sha256") == (ref_lock.get("code") or {}).get(
            "v4data_builder_code_sha256"),
        "header.schema_self_sha256_eq_lock": header.get("feature_schema_self_sha256") == (
            ref_lock.get("features_and_transforms") or {}).get("schema_self_sha256"),
        "header.dataset_version_eq_lock": header.get("dataset_version_hash") == (ref_lock.get("data_version") or {}).get(
            "base_version_hash"),
        "header.candidate_order_eq_lock": header.get("candidate_order_hash") == ((ref_lock.get(
            "population_and_manifests") or {}).get("manifest_hashes") or {}).get("candidate_order_hash"),
        "header.catalogue_K_eq_lock": header.get("catalogue_K") == (ref_lock.get("catalogue") or {}).get("K"),
    })
    for k in ("ordered_decision_id_sha256", "eval_ordered_decision_id_sha256", "rankable_ordered_decision_id_sha256",
              "population_sha256", "n_users", "n_decisions", "n_eval_rows", "n_rankable_rows"):
        checks[f"header.{k}_eq_registry"] = header.get(k) == entry.get(k)
    try:
        checks["registry.content_sha256"] = VF.manifest_content_sha256(header.get("tables", []), tables) == \
            entry.get("content_sha256")
    except VF.ManifestVerificationError:
        checks["registry.content_sha256"] = False
    mk_path = root / str(entry.get("marker", ""))
    if entry.get("marker") and mk_path.is_file():
        mk = VF.strict_json(mk_path).get("verified_sha256", {})
        mbase = mk_path.parent.parent
        checks["ok_marker_matches_disk"] = bool(mk) and all(
            (mbase / rel).is_file() and VF.sha256_file(mbase / rel) == s for rel, s in mk.items())
    else:
        checks["ok_marker_matches_disk"] = False
    t_head = time.time()
    VF._fail_if_bad(mid, checks)
    cols = None
    for t in VF.TABLES[kind]:
        tbl = pq.read_table(files[t])
        checks[f"{t}.row_count_eq_registry"] = tbl.num_rows == tables[t]["row_count"]
        checks[f"{t}.table_content_sha256_eq_registry"] = table_content_sha256(D9.reread_columns_d9(tbl)) == \
            tables[t]["table_content_sha256"]
        if t == "decisions":
            cols = {name: tbl[name].to_numpy(zero_copy_only=False) for name in tbl.column_names}
        del tbl
    did = np.asarray(cols["decision_id"], dtype=np.int64)
    n = did.size
    checks["decisions.order_hash_eq_header"] = ordered_ids_sha256(did) == header.get("ordered_decision_id_sha256")
    checks["decisions.strictly_ascending"] = bool(n <= 1 or np.all(did[1:] > did[:-1]))
    checks["decisions.old_split_all_TEST"] = bool(np.all(np.asarray(cols["old_split"], dtype=object) == "TEST"))
    checks["decisions.v4_split_all_RETRO_TEST"] = bool(np.all(np.asarray(cols["v4_split"], dtype=object)
                                                              == "RETRO_TEST"))
    checks["decisions.grace_matured_column"] = "grace_matured" in cols
    checks["n_decisions_eq_header"] = n == int(header.get("n_decisions", -1))
    checks["n_decisions_eq_registry"] = n == int(entry.get("n_decisions", -1))
    VF._fail_if_bad(mid, checks)
    t_end = time.time()
    say(f"[v4int.test_roles] {mid}: {len(checks)} gated registry checks passed ({t_end - t0:.1f} s)")
    rec = {"path": str(ext_path), "file_sha256": _sha(ext_path), "self_sha256": ext["self_sha256"],
           "data_version": ext.get("data_version"), "run_label": ext.get("run_label"), "root": str(root),
           "entry_content_sha256": entry.get("content_sha256"), "entry_header_sha256": entry.get("header_sha256"),
           "test_gate": g.binding()}
    return VF.VerifiedManifest(manifest_id=mid, kind=kind, header=header, header_file_sha256=hsha, tables=tables,
                               columns=cols, canonical_decision_id=np.ascontiguousarray(did, dtype=np.int64),
                               canonical_rows=np.arange(n, dtype=np.int64), lock_self_sha256=lself,
                               lock_file_sha256=lfile, transform_state_self_sha256=ts_self,
                               transform_state_file_sha256=ts_file, checks=checks,
                               timings={"lock_state_registry": round(t_lock - t0, 2),
                                        "files_and_header": round(t_head - t_lock, 2),
                                        "tables": round(t_end - t_head, 2), "total": round(t_end - t0, 2)},
                               registry=rec)


def _is_test_header(header, fields=None) -> bool:
    h = header or {}
    if str(h.get("v4_split", "")).upper() == "RETRO_TEST" or is_test_name(h.get("manifest_id", "")):
        return True
    if h.get("all_methods_lock_sha256") or h.get("test_exposure_register_entry_id"):
        return True
    return False


def test_eval_manifest(vm, *, lock_path: Optional[Path] = None, register_path: Optional[Path] = None, columns=None):
    """v4eval.manifest.build_eval_manifest of a gated VerifiedManifest, handing v4eval its ALL_METHODS_LOCK path."""
    g = check_test_gate(lock_path, register_path)
    if not header_matches_gate(vm.header, g):
        raise TestRoleRefused(f"{vm.manifest_id}: header gate fields differ from the open gate")
    from v4eval.manifest import build_eval_manifest
    from v4int.providers import EVAL_MANIFEST_COLUMNS
    src = vm.columns if columns is None else columns
    cols = {c: src[c] for c in EVAL_MANIFEST_COLUMNS if c in src}
    return build_eval_manifest(vm.header, cols, all_methods_lock_path=Path(g.lock_path))


# ------------------------------------------------------------------------------------------------ in-process wrappers
_STATE: dict = {"patched": {}, "views": set(), "ext": None, "role": None, "lock": None, "register": None,
                "records": []}


def _gate_now() -> GateRecord:
    return check_test_gate(_STATE["lock"], _STATE["register"])


def _patch(mod, name: str, new) -> None:
    key = (id(mod), name)
    cur = getattr(mod, name)
    orig = _orig(mod, name) if getattr(cur, MARK, False) else cur
    if key not in _STATE["patched"]:
        _STATE["patched"][key] = (mod, name, cur)
    new.__wrapped__ = orig
    setattr(new, MARK, True)
    setattr(mod, name, new)


def uninstall() -> None:
    for mod, name, orig in _STATE["patched"].values():
        setattr(mod, name, orig)
    _STATE.update(patched={}, views=set(), ext=None, role=None, lock=None, register=None, records=[])


def _refuse(LC, msg: str, cause=None):
    exc = getattr(LC, "LaunchRefused", TestRoleRefused) if LC is not None else TestRoleRefused
    raise exc(msg) from cause


def install(ext_path: Path = DEFAULT_EXT, *, role: Optional[str] = None, lock_path: Optional[Path] = None,
            register_path: Optional[Path] = None, LC=None, eager: bool = True) -> dict:
    """Install W1-W5 (module docstring). `role` = the TEST role a launcher's MAIN role becomes (W2); None = no trainer
    view (eval-only consumers: reeval / baselines / dump_rows call resolve_role + StoreEvalProvider directly)."""
    if role is not None and role not in TEST_ROLES:
        raise TestRoleRefused(f"{role!r} is not a registered TEST role {sorted(TEST_ROLES)}")
    ext_path = Path(ext_path).resolve()
    _STATE.update(ext=ext_path, role=role, lock=lock_path, register=register_path)
    if LC is None:
        from v4int import launch_central as LC                  # stdlib-only at import
    orig_bind = _orig(LC, "bind_data_protocol_lock_v2")
    orig_view = _orig(LC, "build_trainer_registry_view")

    def bind(paths, *, smoke, **kw):
        ctx = orig_bind(paths, smoke=smoke, **kw)
        try:
            g = _gate_now()
            base_path = Path(ctx["manifest_registry_path"])
            ext = load_test_ext(ext_path, g, base_path=base_path, base_file_sha256=ctx["manifest_registry_file_sha256"])
        except TestGateError as e:
            _refuse(LC, f"TEST registry extension refused: {e}", e)
        if (ext.get("extends") or {}).get("path") != ctx["manifest_registry_rel_path"]:
            _refuse(LC, "the TEST extension does not extend the lock-pinned REGISTRY_V2 path")
        install_loader_wrappers()
        try:
            rel = str(ext_path.relative_to(Path(LC.V4).resolve())).replace("\\", "/")
        except ValueError:
            rel = str(ext_path)
        rec = {"ext_registry": {"path": rel, "file_sha256": _sha(ext_path), "self_sha256": ext["self_sha256"]},
               "base_registry": {"path": ctx["manifest_registry_rel_path"],
                                 "file_sha256": ctx["manifest_registry_file_sha256"],
                                 "self_sha256": ctx["manifest_registry_self_sha256"]},
               "test_gate": g.binding(), "test_role": _STATE["role"]}
        _STATE["records"].append(rec)
        return dict(ctx, manifest_registry_path=ext_path, manifest_registry_rel_path=rel,
                    manifest_registry_file_sha256=_sha(ext_path), manifest_registry_self_sha256=ext["self_sha256"],
                    test_base_registry=rec["base_registry"], test_gate=rec["test_gate"], test_role=_STATE["role"])

    def view(manifest_registry_path, run_dir, *, registry_rel_path, registry_file_sha256):
        if Path(manifest_registry_path).resolve() != ext_path:
            return orig_view(manifest_registry_path, run_dir, registry_rel_path=registry_rel_path,
                             registry_file_sha256=registry_file_sha256)
        try:
            g = _gate_now()
            ext = load_test_ext(ext_path, g)
        except TestGateError as e:
            _refuse(LC, f"TEST trainer view refused: {e}", e)
        role_ = _STATE["role"]
        if role_ is None:
            _refuse(LC, "the TEST extension is bound but no TEST role was installed (test_roles.install(role=...))")
        try:
            obj = R.trainer_view_for_role(ext, role_, ext_rel_path=registry_rel_path, ext_file_sha256=registry_file_sha256)
        except R.ExtRegistryError as e:
            _refuse(LC, str(e), e)
        chk = R.check_trainer_view(obj, ext, role_)
        if not chk["all_ok"]:
            _refuse(LC, f"TEST trainer view refused: {chk}")
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        p = run_dir / R.VIEW_FILE
        p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        _STATE["views"].add(str(p.resolve()))
        return {"path": p, "self_sha256": obj["self_sha256"], "roles": sorted(obj["roles"]),
                "source_registry_self_sha256": ext["self_sha256"], "source_registry_file_sha256": registry_file_sha256,
                "test_role": role_}

    _patch(LC, "bind_data_protocol_lock_v2", bind)
    _patch(LC, "build_trainer_registry_view", view)
    if eager:
        install_loader_wrappers()
    return {"ext": str(ext_path), "role": role, "eager": eager,
            "wrappers": ["launch_central.bind_data_protocol_lock_v2", "launch_central.build_trainer_registry_view",
                         "v4int.verify.verify_registry", "v4int.verify.verify_manifest_by_registry",
                         "v4int.providers.{verify_manifest_by_registry,refuse_test_manifest,build_eval_manifest}",
                         "v4train.registry.load_manifest_registry (TEST views only)"]}


def install_loader_wrappers() -> None:
    """W3-W5 (imports v4int.verify / v4int.providers / v4train: call after a launcher's CUDA export)."""
    if _STATE["ext"] is None:
        raise TestRoleRefused("test_roles.install(...) first")
    import v4int.providers as PV
    import v4int.verify as VF
    import v4train.registry as TRG
    ext_path = _STATE["ext"]
    o_vr, o_vmr_vf = _orig(VF, "verify_registry"), _orig(VF, "verify_manifest_by_registry")
    o_vmr_pv, o_rt, o_bem = (_orig(PV, "verify_manifest_by_registry"), _orig(PV, "refuse_test_manifest"),
                             _orig(PV, "build_eval_manifest"))
    o_lmr = _orig(TRG, "load_manifest_registry")

    def verify_registry(registry_path, lock_self, lock_file, *, lock=None, root=None):
        if Path(registry_path).resolve() != ext_path:
            return o_vr(registry_path, lock_self, lock_file, lock=lock, root=root)
        g = _gate_now()
        ext = load_test_ext(ext_path, g)
        base_path = (Path(root) if root is not None else V4) / str((ext.get("extends") or {}).get("path", ""))
        base, base_file, base_self, v1_ctx = o_vr(base_path, lock_self, lock_file, lock=lock, root=root)
        chk = R.verify_test_ext_registry(ext, base, base_file)
        if not chk["all_ok"]:
            raise VF.ManifestVerificationError(f"TEST extension {ext_path} fails the extension rule: "
                                               f"{[k for k, v in chk.items() if not v]}")
        return ext, _sha(ext_path), ext["self_sha256"], v1_ctx

    def make_vmr(orig):
        def verify_manifest_by_registry(manifest_id, registry_path, paths=None, *, root=None, expect_kind=None,
                                        log=None):
            if not is_test_name(manifest_id):
                return orig(manifest_id, registry_path, paths, root=root, expect_kind=expect_kind, log=log)
            _gate_now()                                             # before the registry path is even compared
            if Path(registry_path).resolve() != ext_path:
                raise TestRoleRefused(f"{manifest_id}: a TEST manifest resolves only through the installed extension "
                                      f"{ext_path} (got {registry_path})")
            return verify_test_manifest(manifest_id, ext_path=ext_path, paths=paths, root=root,
                                        expect_kind=expect_kind, lock_path=_STATE["lock"],
                                        register_path=_STATE["register"], log=log)
        return verify_manifest_by_registry

    def refuse_test_manifest(manifest_id, header=None, columns=None):
        if is_test_name(manifest_id) and header is None and columns is None:
            _gate_now()
            if str(manifest_id) not in TEST_IDS:
                raise TestRoleRefused(f"{manifest_id}: not a registered RETRO_TEST manifest")
            return None
        return o_rt(manifest_id, header=header, columns=columns)

    def build_eval_manifest(header, columns, *, all_methods_lock_path=None):
        if not _is_test_header(header):
            return o_bem(header, columns, all_methods_lock_path=all_methods_lock_path)
        g = _gate_now()
        if not header_matches_gate(header, g):
            raise TestRoleRefused(f"{header.get('manifest_id')}: header gate fields differ from the open gate")
        return o_bem(header, columns, all_methods_lock_path=Path(g.lock_path))

    def load_manifest_registry(path):
        if str(Path(path).resolve()) not in _STATE["views"]:
            return o_lmr(path)
        _gate_now()
        return _load_test_view(Path(path))

    _patch(VF, "verify_registry", verify_registry)
    _patch(VF, "verify_manifest_by_registry", make_vmr(o_vmr_vf))
    _patch(PV, "verify_manifest_by_registry", make_vmr(o_vmr_pv))
    _patch(PV, "refuse_test_manifest", refuse_test_manifest)
    _patch(PV, "build_eval_manifest", build_eval_manifest)
    _patch(TRG, "load_manifest_registry", load_manifest_registry)


def _load_test_view(p: Path):
    """v4train.registry.load_manifest_registry's checks for a gated TEST view: identical, except that the registered
    RETRO_TEST eval ids are not refused as TEST ids (every other id containing TEST still is)."""
    from v4train import registry as TRG
    from v4train.recipe import ROLES
    obj, file_sha = TRG._read_object(p, "manifest registry")
    self_sha = TRG._verify_self_hash(obj, f"manifest registry {p}")
    dv = obj.get("data_version")
    roles, manifests = obj.get("roles"), obj.get("manifests")
    if not isinstance(dv, str) or not dv or not isinstance(roles, dict) or not roles or not isinstance(manifests, dict):
        raise TRG.RegistryError(f"TEST view {p}: data_version / roles / manifests required")
    bad_ids = [m for m in manifests if TRG._is_test_id(m) and m not in TEST_IDS]
    if bad_ids:
        raise TRG.RegistryError(f"TEST view {p}: unregistered TEST manifest(s) {bad_ids} refused")
    for role, pair in roles.items():
        if role not in ROLES:
            raise TRG.RegistryError(f"TEST view {p}: unknown role {role!r}")
        if not isinstance(pair, dict) or set(pair) != set(TRG.KINDS):
            raise TRG.RegistryError(f"TEST view {p}: role {role} must map exactly 'train' and 'eval'")
        if TRG._is_test_id(pair["train"]):
            raise TRG.RegistryError(f"TEST view {p}: a TEST manifest can never be a train manifest")
        for kind in TRG.KINDS:
            e = manifests.get(pair[kind])
            if not isinstance(e, dict) or not isinstance(e.get("n_decisions"), int) or e["n_decisions"] < 1 or not (
                    TRG._hash_binding(e) or TRG._fp_binding(e)):
                raise TRG.RegistryError(f"TEST view {p}: entry {pair[kind]} is missing or binds nothing")
    return TRG.ManifestRegistry(path=str(p), file_sha256=file_sha, self_sha256=self_sha, data_version=dv,
                                roles={k: dict(v) for k, v in roles.items()},
                                manifests={k: dict(v) for k, v in manifests.items()}, raw=obj)


__all__ = ["TestRoleRefused", "resolve_role", "test_gate_ok", "verify_test_manifest", "test_eval_manifest", "install",
           "install_loader_wrappers", "uninstall", "load_test_ext", "gate", "is_test_name", "TEST_ROLES", "TEST_IDS",
           "DEFAULT_EXT"]
