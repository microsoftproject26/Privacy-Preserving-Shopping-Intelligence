"""v4data10.d10_registry (owner: Data engineer, DATA-10) - the registry extension REGISTRY_V2_PRETRAIN_EXT.json and the
PRETRAIN trainer view, as PURE functions (no file I/O, no v4train import; used by d10_finalize on the server and by the
launcher shim v4int.pretrain_small). Same rule as v4data08.d8_registry (DATA-08), for role PRETRAIN.

Extension rule (checked by `verify_ext_registry`): the extension is REGISTRY_V2 plus additions only -
  * every top-level key of REGISTRY_V2 except `self_sha256` is present with canonical-JSON-identical bytes, except the
    three containers `roles`, `manifests`, `population_bands`, whose every REGISTRY_V2 entry is canonical-JSON
    identical and which gain exactly roles.PRETRAIN, manifests.PRETRAIN_TRAIN_5_100_NONMAIN,
    population_bands.PRETRAIN_NONMAIN;
  * the added top-level keys are exactly ADDED_TOP_KEYS; `extends` names REGISTRY_V2's path, file sha256 and
    self_sha256 (so the LOCK_V2-pinned REGISTRY_V2 anchors the extension);
  * roles.PRETRAIN = {train: PRETRAIN_TRAIN_5_100_NONMAIN, eval: UNEW_VAL_VALIDATION_32768} (the eval entry is the
    REGISTRY_V2 entry, byte-identical); the band / entry carry the registered rank range [131,072, 1,348,682),
    n_users 1,217,610, n_decisions 24,089,824 and one population hash (band == entry; == PRETRAIN_SHA_V2 once measured);
  * the extension's own `self_sha256` verifies (the REGISTRY_V2 rule: v4data.common.self_hash, field self_sha256).
Trainer view (`trainer_view`): the shape launch_central.build_trainer_registry_view writes, with its MAIN (and
mirrored CONTINUATION) role = the PRETRAIN pair, so the unedited trainer (role MAIN) is bound to
PRETRAIN_TRAIN_5_100_NONMAIN -> UNEW_VAL_VALIDATION_32768; every entry is copied byte-for-byte from the extension.
"""
from __future__ import annotations

import copy
from typing import Mapping

from v4data.common import canonical_json_bytes, self_hash

from . import d10_common as D10

SELF = "self_sha256"
ADDED_TOP_KEYS = ("extends", "extension_task", "extension_code", "extension_build_receipt", "extension_note")
NESTED_ADDITIONS = {"roles": (D10.ROLE,), "manifests": (D10.TRAIN_ID,), "population_bands": (D10.BAND,)}
VIEW_FILE = "REGISTRY_V2_TRAINER_VIEW.json"          # the launcher's own view file name (config-hash parity)
VIEW_ROLES = ("MAIN", "CONTINUATION")
PAIR = {"train": D10.TRAIN_ID, "eval": D10.EVAL_ID}


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
    """REGISTRY_V2 (`base`, verified by the caller) + the PRETRAIN role, its train manifest entry and its band."""
    if not self_ok(base):
        raise ExtRegistryError("the base registry's self_sha256 does not verify")
    if D10.ROLE in base.get("roles", {}) or D10.TRAIN_ID in base.get("manifests", {}) \
            or D10.BAND in (base.get("population_bands") or {}):
        raise ExtRegistryError("the base registry already carries PRETRAIN / PRETRAIN_TRAIN_5_100_NONMAIN")
    if D10.EVAL_ID not in base.get("manifests", {}):
        raise ExtRegistryError(f"the base registry has no {D10.EVAL_ID} entry")
    if entry.get("kind") != "TRAIN" or entry.get("status") != D10.ROLE:
        raise ExtRegistryError("the PRETRAIN entry must be a TRAIN manifest with status PRETRAIN")
    ext = copy.deepcopy({k: v for k, v in base.items() if k != SELF})
    ext["roles"][D10.ROLE] = dict(PAIR)
    ext["manifests"][D10.TRAIN_ID] = copy.deepcopy(dict(entry))
    ext.setdefault("population_bands", {})[D10.BAND] = copy.deepcopy(dict(band))
    ext["extends"] = {"path": base_rel_path, "file_sha256": base_file_sha256, "self_sha256": base[SELF],
                      "rule": "every REGISTRY_V2 key / role / manifest / population band is canonical-JSON identical "
                              "here (v4data10.d10_registry.verify_ext_registry); additions only: roles.PRETRAIN, "
                              "manifests.PRETRAIN_TRAIN_5_100_NONMAIN, population_bands.PRETRAIN_NONMAIN and the "
                              "extension_* keys"}
    ext["extension_task"] = D10.TASK
    ext["extension_code"] = dict(code)
    ext["extension_build_receipt"] = dict(build_receipt)
    ext["extension_note"] = {"pretrain": "FA_WARM (owner decision 2026-09-29, DECISION_LOG 195): role PRETRAIN trains "
                                         "the SMALL model centrally on the non-MAIN users of the TIEBREAK band; its "
                                         "UNEW_VAL_VALIDATION_32768 marks are DESCRIPTIVE and never select anything; "
                                         "the FL init is the endpoint_6.0 checkpoint",
                             "selection_refusal": D10.SELECTION_REFUSAL}
    return with_self(ext)


def expected_population() -> dict:
    """The registered PRETRAIN population facts the extension must carry (v4data10.d10_common)."""
    return {"rank_range": list(D10.RANKS), "n_users": int(D10.N_USERS), "n_decisions": int(D10.N_DECISIONS),
            "population_sha256_selection_key_v2": D10.PRETRAIN_SHA_V2}


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
    c["roles.PRETRAIN"] = roles.get(D10.ROLE) == PAIR
    ent = mans.get(D10.TRAIN_ID) or {}
    c["manifests.PRETRAIN_train.kind_status"] = ent.get("kind") == "TRAIN" and ent.get("status") == D10.ROLE
    c["manifests.PRETRAIN_eval_is_base_entry"] = D10.EVAL_ID in (base.get("manifests") or {}) and canon(
        mans.get(D10.EVAL_ID)) == canon(base["manifests"][D10.EVAL_ID])
    exp = expected_population()
    band = (ext.get("population_bands") or {}).get(D10.BAND) or {}
    rr = band.get("rank_range")
    c["population_bands.PRETRAIN_NONMAIN.rank_range"] = (isinstance(rr, (list, tuple)) and list(rr) == exp["rank_range"]
                                                         and all(type(x) is int for x in rr))
    c["population_bands.PRETRAIN_NONMAIN.n_users"] = type(band.get("n_users")) is int and band["n_users"] == exp["n_users"]
    c["manifests.PRETRAIN_train.n_users"] = type(ent.get("n_users")) is int and ent.get("n_users") == exp["n_users"]
    c["manifests.PRETRAIN_train.n_decisions"] = (type(ent.get("n_decisions")) is int
                                                 and ent.get("n_decisions") == exp["n_decisions"])
    psha = ent.get("population_sha256_selection_key_v2")
    c["population_sha256.band_eq_entry"] = isinstance(psha, str) and len(psha) == 64 and \
        band.get("population_sha256_selection_key_v2") == psha
    if exp["population_sha256_selection_key_v2"] is not None:
        c["population_sha256.eq_registered"] = psha == exp["population_sha256_selection_key_v2"]
    c["population_sha256.not_a_registry_v2_band"] = psha not in {
        (v or {}).get("population_sha256_selection_key_v2") for v in (base.get("population_bands") or {}).values()}
    c["no_TEST_id_or_role"] = not any("TEST" in str(x).upper() for x in list(roles) + list(mans))
    c["data_version_eq_base"] = ext.get("data_version") == base.get("data_version")
    c["all_ok"] = all(c.values())
    return c


def trainer_view(ext: Mapping, *, ext_rel_path: str, ext_file_sha256: str) -> dict:
    """The PRETRAIN trainer view (module docstring): roles MAIN / CONTINUATION = the extension's PRETRAIN pair."""
    pair = dict((ext.get("roles") or {}).get(D10.ROLE) or {})
    if pair != PAIR:
        raise ExtRegistryError(f"the extension's PRETRAIN role is {pair!r}")
    mans = ext["manifests"]
    return with_self({
        "data_version": ext["data_version"], "roles": {r: dict(pair) for r in VIEW_ROLES},
        "manifests": {mid: copy.deepcopy(mans[mid]) for mid in (D10.TRAIN_ID, D10.EVAL_ID)},
        "derived_from": {"path": ext_rel_path, "file_sha256": ext_file_sha256, "self_sha256": ext.get(SELF),
                         "note": "v4int.pretrain_small (DATA-10): MAIN / CONTINUATION = the extension's PRETRAIN role "
                                 "(PRETRAIN_TRAIN_5_100_NONMAIN -> UNEW_VAL_VALIDATION_32768), every entry "
                                 "byte-identical to the extension registry"}})


def check_trainer_view(view: Mapping, ext: Mapping) -> dict:
    c = {"view.self_sha256": self_ok(view)}
    c["view.roles"] = set(view.get("roles") or {}) == set(VIEW_ROLES) and all(
        view["roles"][r] == PAIR for r in VIEW_ROLES)
    c["view.entries_byte_identical"] = set(view.get("manifests") or {}) == set(PAIR.values()) and all(
        canon(view["manifests"][m]) == canon(ext["manifests"][m]) for m in PAIR.values())
    c["view.data_version"] = view.get("data_version") == ext.get("data_version")
    c["view.derived_from_ext"] = (view.get("derived_from") or {}).get("self_sha256") == ext.get(SELF)
    c["all_ok"] = all(c.values())
    return c


__all__ = ["ExtRegistryError", "build_ext_registry", "expected_population", "verify_ext_registry", "trainer_view",
           "check_trainer_view", "with_self", "self_ok", "canon", "ADDED_TOP_KEYS", "NESTED_ADDITIONS", "VIEW_FILE",
           "VIEW_ROLES", "PAIR"]
