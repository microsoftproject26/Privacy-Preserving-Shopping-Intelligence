"""v4data08.d8_registry (owner: Data engineer, DATA-08) - the registry extension REGISTRY_V2_FULL_EXT.json and the C_FULL
trainer view, as PURE functions (no file I/O, no v4train import; used by d8_finalize on the server and by the
launcher shim v4int.c_full).

Extension rule (checked by `verify_ext_registry`): the extension is REGISTRY_V2 plus additions only -
  * every top-level key of REGISTRY_V2 except `self_sha256` is present with canonical-JSON-identical bytes, except the
    three containers `roles`, `manifests`, `population_bands`, whose every REGISTRY_V2 entry is canonical-JSON
    identical and which gain exactly roles.C_FULL, manifests.FULL_TRAIN_5_100_TIEBREAK, population_bands.TIEBREAK;
  * the added top-level keys are exactly ADDED_TOP_KEYS; `extends` names REGISTRY_V2's path, file sha256 and
    self_sha256 (so a LOCK_V2-pinned REGISTRY_V2 anchors the extension);
  * roles.C_FULL = {train: FULL_TRAIN_5_100_TIEBREAK, eval: MAIN_VALIDATION_5_100_131072}; the extension's own
    `self_sha256` verifies (the REGISTRY_V2 rule: v4data.common.self_hash, field self_sha256).
Trainer view (`trainer_view`): the same shape launch_central.build_trainer_registry_view writes, but its MAIN (and
mirrored CONTINUATION) role is the C_FULL pair, so the unedited trainer (role MAIN, recipe 02B_SASREC) is bound to
FULL_TRAIN_5_100_TIEBREAK -> MAIN_VALIDATION_5_100_131072; every entry is copied byte-for-byte from the extension.
"""
from __future__ import annotations

import copy
from typing import Mapping

from v4data.common import canonical_json_bytes, self_hash

from . import d8_common as D8

SELF = "self_sha256"
ADDED_TOP_KEYS = ("extends", "extension_task", "extension_code", "extension_build_receipt", "extension_note")
NESTED_ADDITIONS = {"roles": (D8.ROLE,), "manifests": (D8.TRAIN_ID,), "population_bands": (D8.BAND,)}
VIEW_FILE = "REGISTRY_V2_TRAINER_VIEW.json"          # the launcher's own view file name (config-hash parity with dry runs)
VIEW_ROLES = ("MAIN", "CONTINUATION")


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


def build_ext_registry(base: Mapping, *, base_rel_path: str, base_file_sha256: str, entry: Mapping, band: Mapping,
                       build_receipt: Mapping, code: Mapping) -> dict:
    """REGISTRY_V2 (`base`, verified by the caller) + the C_FULL role, its train manifest entry and the TIEBREAK band."""
    if not self_ok(base):
        raise ExtRegistryError("the base registry's self_sha256 does not verify")
    if D8.ROLE in base.get("roles", {}) or D8.TRAIN_ID in base.get("manifests", {}):
        raise ExtRegistryError("the base registry already carries C_FULL / FULL_TRAIN_5_100_TIEBREAK")
    if D8.EVAL_ID not in base.get("manifests", {}):
        raise ExtRegistryError(f"the base registry has no {D8.EVAL_ID} entry")
    if entry.get("kind") != "TRAIN" or entry.get("status") != D8.ROLE:
        raise ExtRegistryError("the C_FULL entry must be a TRAIN manifest with status C_FULL")
    ext = copy.deepcopy({k: v for k, v in base.items() if k != SELF})
    ext["roles"][D8.ROLE] = {"train": D8.TRAIN_ID, "eval": D8.EVAL_ID}
    ext["manifests"][D8.TRAIN_ID] = copy.deepcopy(dict(entry))
    ext.setdefault("population_bands", {})[D8.BAND] = copy.deepcopy(dict(band))
    ext["extends"] = {"path": base_rel_path, "file_sha256": base_file_sha256, "self_sha256": base[SELF],
                      "rule": "every REGISTRY_V2 key / role / manifest / population band is canonical-JSON identical "
                              "here (v4data08.d8_registry.verify_ext_registry); additions only: roles.C_FULL, "
                              "manifests.FULL_TRAIN_5_100_TIEBREAK, population_bands.TIEBREAK and the extension_* keys"}
    ext["extension_task"] = D8.TASK
    ext["extension_code"] = dict(code)
    ext["extension_build_receipt"] = dict(build_receipt)
    ext["extension_note"] = {"c_full": "PREG1_AMENDMENT_06 §5 / 06B §6.3: role C_FULL trains on the locked TIEBREAK "
                                       "band and evaluates on MAIN_VALIDATION_5_100_131072 unchanged",
                             "population_discrepancy": D8.DISCREPANCY_NOTE}
    return with_self(ext)


def expected_population() -> dict:
    """The locked TIEBREAK population facts the extension must carry (REV-ANA-01 C-F1): rank range
    `BANDS['TIEBREAK']`, the TB selection-key-v2 sha256, n_users and n_decisions (all from v4data08.d8_common)."""
    return {"rank_range": list(D8.BANDS[D8.BAND]), "population_sha256_selection_key_v2": D8.TB_SHA_V2,
            "n_users": int(D8.TB_N_USERS), "n_decisions": int(D8.TB_N_DECISIONS)}


def verify_ext_registry(ext: Mapping, base: Mapping, base_file_sha256: str) -> dict:
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
    c["roles.C_FULL"] = roles.get(D8.ROLE) == {"train": D8.TRAIN_ID, "eval": D8.EVAL_ID}
    ent = mans.get(D8.TRAIN_ID) or {}
    c["manifests.C_FULL_train.kind_status"] = ent.get("kind") == "TRAIN" and ent.get("status") == D8.ROLE
    c["manifests.C_FULL_eval_is_base_entry"] = D8.EVAL_ID in (base.get("manifests") or {}) and canon(
        mans.get(D8.EVAL_ID)) == canon(base["manifests"][D8.EVAL_ID])
    exp = expected_population()
    band = (ext.get("population_bands") or {}).get(D8.BAND) or {}
    rr = band.get("rank_range")
    c["population_bands.TIEBREAK.rank_range"] = (isinstance(rr, (list, tuple)) and list(rr) == exp["rank_range"]
                                                 and all(type(x) is int for x in rr))
    for fld in ("n_users", "population_sha256_selection_key_v2"):
        if fld in band:                              # the DATA-08 finalizer always writes both; checked when present
            c[f"population_bands.TIEBREAK.{fld}"] = band[fld] == exp[fld] and type(band[fld]) is type(exp[fld])
    c["manifests.C_FULL_train.n_users"] = type(ent.get("n_users")) is int and ent.get("n_users") == exp["n_users"]
    c["manifests.C_FULL_train.n_decisions"] = (type(ent.get("n_decisions")) is int
                                               and ent.get("n_decisions") == exp["n_decisions"])
    c["manifests.C_FULL_train.population_sha256_selection_key_v2"] = (
        ent.get("population_sha256_selection_key_v2") == exp["population_sha256_selection_key_v2"])
    c["no_TEST_id_or_role"] = not any("TEST" in str(x).upper() for x in list(roles) + list(mans))
    c["data_version_eq_base"] = ext.get("data_version") == base.get("data_version")
    c["all_ok"] = all(c.values())
    return c


def trainer_view(ext: Mapping, *, ext_rel_path: str, ext_file_sha256: str) -> dict:
    """The C_FULL trainer view (module docstring): roles MAIN / CONTINUATION = the extension's C_FULL pair."""
    pair = dict((ext.get("roles") or {}).get(D8.ROLE) or {})
    if pair != {"train": D8.TRAIN_ID, "eval": D8.EVAL_ID}:
        raise ExtRegistryError(f"the extension's C_FULL role is {pair!r}")
    mans = ext["manifests"]
    return with_self({
        "data_version": ext["data_version"], "roles": {r: dict(pair) for r in VIEW_ROLES},
        "manifests": {mid: copy.deepcopy(mans[mid]) for mid in (D8.TRAIN_ID, D8.EVAL_ID)},
        "derived_from": {"path": ext_rel_path, "file_sha256": ext_file_sha256, "self_sha256": ext.get(SELF),
                         "note": "v4int.c_full (DATA-08): MAIN / CONTINUATION = the extension's C_FULL role "
                                 "(FULL_TRAIN_5_100_TIEBREAK -> MAIN_VALIDATION_5_100_131072), every entry "
                                 "byte-identical to the extension registry"}})


def check_trainer_view(view: Mapping, ext: Mapping) -> dict:
    c = {"view.self_sha256": self_ok(view)}
    pair = {"train": D8.TRAIN_ID, "eval": D8.EVAL_ID}
    c["view.roles"] = set(view.get("roles") or {}) == set(VIEW_ROLES) and all(
        view["roles"][r] == pair for r in VIEW_ROLES)
    c["view.entries_byte_identical"] = set(view.get("manifests") or {}) == set(pair.values()) and all(
        canon(view["manifests"][m]) == canon(ext["manifests"][m]) for m in pair.values())
    c["view.data_version"] = view.get("data_version") == ext.get("data_version")
    c["view.derived_from_ext"] = (view.get("derived_from") or {}).get("self_sha256") == ext.get(SELF)
    c["all_ok"] = all(c.values())
    return c


__all__ = ["ExtRegistryError", "build_ext_registry", "expected_population", "verify_ext_registry", "trainer_view", "check_trainer_view",
           "with_self", "self_ok", "ADDED_TOP_KEYS", "NESTED_ADDITIONS", "VIEW_FILE", "VIEW_ROLES"]
