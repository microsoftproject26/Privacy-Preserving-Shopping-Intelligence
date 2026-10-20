"""v4data09.d9_registry (owner: Data engineer, DATA-09) - the TEST registry extension REGISTRY_V2_TEST_EXT.json and the
TEST-role trainer view, as PURE functions (no file I/O, no v4train import). Pattern: v4data08.d8_registry.

Extension rule (`verify_test_ext_registry`): the extension is REGISTRY_V2 plus additions only -
  * every top-level key of REGISTRY_V2 except `self_sha256` is present and canonical-JSON identical, except the two
    containers `roles` and `manifests`, whose every REGISTRY_V2 entry is canonical-JSON identical and which gain exactly
    the five TEST roles (d9_common.TEST_ROLES) and the five RETRO_TEST manifest entries (d9_common.TEST_IDS);
  * the added top-level keys are exactly ADDED_TOP_KEYS; `extends` names REGISTRY_V2's path, file sha256 and self_sha256;
    `extension_test_gate` is the GateRecord binding (all_methods_lock_sha256, test_exposure_register_entry_id, entry
    sha256) and every RETRO_TEST entry repeats the same two values (from its header);
  * each TEST role's train manifest is the REGISTRY_V2 entry itself (unchanged support manifests); each TEST eval entry
    is kind EVAL, status = its role, and has the EVAL table layout;
  * the extension's own `self_sha256` verifies (the REGISTRY_V2 rule).
TEST-role trainer view (`trainer_view_for_role`): the shape launch_central.build_trainer_registry_view writes, with MAIN
and CONTINUATION = the chosen TEST role's {train, eval} pair; entries copied byte-for-byte from the extension. Only the
v4int.test_roles loader wrapper (gated) can load it: the unedited v4train loader refuses its TEST eval id, by design.
"""
from __future__ import annotations

import copy
from typing import Mapping

from v4data.common import canonical_json_bytes, self_hash

from . import d9_common as D9

SELF = "self_sha256"
ADDED_TOP_KEYS = ("extends", "extension_task", "extension_code", "extension_build_receipt", "extension_note",
                  "extension_test_gate")
NESTED_ADDITIONS = {"roles": tuple(D9.TEST_ROLES), "manifests": tuple(D9.TEST_IDS)}
VIEW_FILE = "REGISTRY_V2_TRAINER_VIEW.json"
VIEW_ROLES = ("MAIN", "CONTINUATION")
GATE_KEYS = ("all_methods_lock_sha256", "test_exposure_register_entry_id")


class ExtRegistryError(ValueError):
    pass


def canon(x) -> bytes:
    return canonical_json_bytes(x)


def with_self(obj: Mapping) -> dict:
    d = {k: v for k, v in obj.items() if k != SELF}
    d[SELF] = self_hash(d, SELF)
    return d


def self_ok(obj: Mapping) -> bool:
    return isinstance(obj.get(SELF), str) and obj.get(SELF) == self_hash(obj, SELF)


def build_test_ext_registry(base: Mapping, *, base_rel_path: str, base_file_sha256: str, entries: Mapping,
                            gate_binding: Mapping, build_receipt: Mapping, code: Mapping) -> dict:
    """REGISTRY_V2 (`base`, verified by the caller) + the five TEST roles and their RETRO_TEST manifest entries."""
    if not self_ok(base):
        raise ExtRegistryError("the base registry's self_sha256 does not verify")
    if any(r in base.get("roles", {}) for r in D9.TEST_ROLES) or any(m in base.get("manifests", {}) for m in D9.TEST_IDS):
        raise ExtRegistryError("the base registry already carries a TEST role / RETRO_TEST manifest")
    if set(entries) != set(D9.TEST_IDS):
        raise ExtRegistryError(f"entries must be exactly {D9.TEST_IDS} (got {sorted(entries)})")
    for role, pair in D9.TEST_ROLES.items():
        if "train" in pair and pair["train"] not in base.get("manifests", {}):
            raise ExtRegistryError(f"the base registry has no {pair['train']} entry (support of {role})")
        e = entries[pair["eval"]]
        if e.get("kind") != "EVAL" or e.get("status") != role or e.get("role") != role:
            raise ExtRegistryError(f"{pair['eval']}: must be an EVAL entry with status / role {role}")
        if any(e.get(k) != gate_binding.get(k) for k in GATE_KEYS):
            raise ExtRegistryError(f"{pair['eval']}: its header gate fields differ from the extension's gate")
    ext = copy.deepcopy({k: v for k, v in base.items() if k != SELF})
    for role, pair in D9.TEST_ROLES.items():
        ext["roles"][role] = dict(pair)
    for mid in D9.TEST_IDS:
        ext["manifests"][mid] = copy.deepcopy(dict(entries[mid]))
    ext["extends"] = {"path": base_rel_path, "file_sha256": base_file_sha256, "self_sha256": base[SELF],
                      "rule": "every REGISTRY_V2 key / role / manifest is canonical-JSON identical here (v4data09."
                              "d9_registry.verify_test_ext_registry); additions only: the five TEST roles, the five "
                              "RETRO_TEST manifests and the extension_* keys"}
    ext["extension_task"] = D9.TASK
    ext["extension_code"] = dict(code)
    ext["extension_build_receipt"] = dict(build_receipt)
    ext["extension_test_gate"] = dict(gate_binding)
    ext["extension_note"] = {
        "test_look": "PREG1_AMENDMENT_06D §2: the single TEST look's registered manifests; the support manifests are "
                     "REGISTRY_V2's, unchanged; resolved only through v4int.test_roles under the TEST gate",
        "roles": {r: dict(p) for r, p in D9.TEST_ROLES.items()},
        "m09": "M09 (TEST cold-h) is NOT registered (amendment 06D §2)"}
    return with_self(ext)


def verify_test_ext_registry(ext: Mapping, base: Mapping, base_file_sha256: str) -> dict:
    """Every check of the extension rule (module docstring) -> {name: bool, ..., 'all_ok': bool}."""
    c: dict = {}
    c["ext.self_sha256"] = self_ok(ext)
    c["base.self_sha256"] = self_ok(base)
    ex = ext.get("extends") or {}
    c["extends.self_sha256_eq_base"] = ex.get("self_sha256") == base.get(SELF)
    c["extends.file_sha256_eq_base"] = ex.get("file_sha256") == base_file_sha256
    base_keys = set(base) - {SELF}
    c["top_keys.exact"] = set(ext) - {SELF} == base_keys | set(ADDED_TOP_KEYS)
    for k in sorted(base_keys):
        if k in NESTED_ADDITIONS:
            b, e = base.get(k) or {}, ext.get(k) or {}
            c[f"{k}.key_set"] = isinstance(e, Mapping) and set(e) == set(b) | set(NESTED_ADDITIONS[k])
            c[f"{k}.base_entries_byte_identical"] = isinstance(e, Mapping) and all(
                sk in e and canon(e[sk]) == canon(v) for sk, v in b.items())
        else:
            c[f"top.{k}.byte_identical"] = k in ext and canon(ext[k]) == canon(base[k])
    roles, mans = ext.get("roles") or {}, ext.get("manifests") or {}
    gate = ext.get("extension_test_gate") or {}
    c["extension_test_gate.fields"] = all(gate.get(k) not in (None, "") for k in GATE_KEYS)
    for role, pair in D9.TEST_ROLES.items():
        c[f"roles.{role}"] = roles.get(role) == pair
        e = mans.get(pair["eval"]) or {}
        c[f"manifests.{pair['eval']}.kind_status"] = e.get("kind") == "EVAL" and e.get("status") == role
        c[f"manifests.{pair['eval']}.eval_layout"] = set((e.get("tables") or {})) == {"users", "decisions"}
        c[f"manifests.{pair['eval']}.gate_eq_extension"] = all(e.get(k) == gate.get(k) for k in GATE_KEYS)
        if "train" in pair:
            c[f"roles.{role}.train_is_base_entry"] = pair["train"] in (base.get("manifests") or {}) and canon(
                mans.get(pair["train"])) == canon(base["manifests"][pair["train"]])
    c["no_base_role_or_id_named_TEST"] = not any("TEST" in str(x).upper() for x in
                                                 list(base.get("roles") or {}) + list(base.get("manifests") or {}))
    c["data_version_eq_base"] = ext.get("data_version") == base.get("data_version")
    c["all_ok"] = all(c.values())
    return c


def trainer_view_for_role(ext: Mapping, role: str, *, ext_rel_path: str, ext_file_sha256: str) -> dict:
    pair = dict((ext.get("roles") or {}).get(role) or {})
    if role not in D9.TEST_ROLES or pair != D9.TEST_ROLES[role]:
        raise ExtRegistryError(f"{role!r} is not a registered TEST role of the extension (got {pair!r})")
    if "train" not in pair:
        raise ExtRegistryError(f"{role} is eval-only (no support manifest): a trainer view needs a train + eval pair; "
                               "evaluate it through StoreEvalProvider under v4int.test_roles instead")
    mans = ext["manifests"]
    return with_self({
        "data_version": ext["data_version"], "roles": {r: dict(pair) for r in VIEW_ROLES},
        "manifests": {mid: copy.deepcopy(mans[mid]) for mid in (pair["train"], pair["eval"])},
        "derived_from": {"path": ext_rel_path, "file_sha256": ext_file_sha256, "self_sha256": ext.get(SELF),
                         "test_role": role,
                         "note": "v4int.test_roles (DATA-09): MAIN / CONTINUATION = the extension's TEST role pair, "
                                 "every entry byte-identical to the extension registry; loadable only through the "
                                 "gated test_roles loader wrapper"}})


def check_trainer_view(view: Mapping, ext: Mapping, role: str) -> dict:
    pair = D9.TEST_ROLES.get(role) or {}
    c = {"view.self_sha256": self_ok(view)}
    c["view.roles"] = set(view.get("roles") or {}) == set(VIEW_ROLES) and all(
        (view.get("roles") or {}).get(r) == pair for r in VIEW_ROLES)
    c["view.entries_byte_identical"] = set(view.get("manifests") or {}) == set(pair.values()) and all(
        canon(view["manifests"][m]) == canon((ext.get("manifests") or {}).get(m)) for m in pair.values())
    c["view.data_version"] = view.get("data_version") == ext.get("data_version")
    c["view.derived_from_ext"] = (view.get("derived_from") or {}).get("self_sha256") == ext.get(SELF)
    c["all_ok"] = all(c.values())
    return c


__all__ = ["ExtRegistryError", "build_test_ext_registry", "verify_test_ext_registry", "trainer_view_for_role",
           "check_trainer_view", "with_self", "self_ok", "canon", "ADDED_TOP_KEYS", "NESTED_ADDITIONS", "VIEW_FILE",
           "VIEW_ROLES", "GATE_KEYS"]
