"""v4train.registry — the manifest-role registry and derived-schema acceptance (TRAIN-01R2; G1 residual R1).

Manifest-role registry (FS-01 writes contracts/manifest_registry/REGISTRY_FS.json; the HISTORICAL population's file
is registries/REGISTRY_CLEAN_V2_HISTORICAL_262144.json next to this module):
    {"data_version": str,
     "roles": {ROLE: {"train": manifest_id, "eval": manifest_id}},          ROLE in recipe.ROLES
     "manifests": {manifest_id: {"n_decisions", and a binding: "content_sha256" + "header_sha256" and / or
                                 "fingerprint"; FS-01 also records "path", "n_users", "n_eval_rows"}},
     "self_sha256": sha256 of the canonical JSON of the object without "self_sha256"}
  canonical JSON = json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False) — the
  v4models.derived.self_hash / FEATURE_SCHEMA_CLEAN_V2 rule; the DATA_PROTOCOL_LOCK variant that also leaves out a
  "self_sha256_rule" key is accepted too. The file is verified by its self-hash; its file sha256 and self-hash are
  recorded in RUN_START, every checkpoint and every row key, and checked on resume. A TEST manifest id anywhere in a
  registry is refused.
Provider binding: the trainer refuses a provider whose manifest id is not the registry's id for the run's role, or
  whose identity does not match the registry entry:
    * the provider's row count (train: n_decisions; eval: manifest.n) must equal the entry's n_decisions;
    * entry "fingerprint" given -> the provider's fingerprint must equal it;
    * entry "content_sha256" AND "header_sha256" given -> the provider must declare `manifest_binding` =
      {"manifest_id", "content_sha256", "header_sha256"} (the hashes of the manifest files it verified, with the
      registry's meaning) and both hashes must equal the entry's;
    * an entry that states neither binding is refused (fail-closed).
Derived schemas (SCHEMA-01 writes contracts/derived_schemas/): REGISTRY.json maps schema_id -> {path, self_sha256,
  optional file_sha256}, either under a "schemas" object (SCHEMA-01's format) or at the top level, and carries its own
  self_sha256 (same canonical rule); each schema JSON carries "schema_id" and "self_sha256" (same rule). The trainer
  accepts:
    BASE     the locked Clean V2 schema (v4models.schema.CleanV2Schema whose id and self-hash are the base file's),
             re-verified from its file;
    DERIVED  any other schema object exposing `schema_id` and `self_sha256`: the registry must list the id with the
             same self-hash, the registered file must hash to it and state the same id, and when the object carries
             `raw` with a "self_sha256", the raw content must hash to it too.
  schema_id and schema_self_sha256 are recorded in RUN_START, every checkpoint and every row key; a resume with
  another schema is refused.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from v4sup.artifacts import parse_strict_json

from .recipe import HISTORICAL_DATA_VERSION, ROLES

_HERE = Path(__file__).resolve().parent
V4 = _HERE.parents[1]                                     # .../ppsi_v4
HISTORICAL_REGISTRY_PATH = _HERE / "registries" / f"REGISTRY_{HISTORICAL_DATA_VERSION}.json"
DERIVED_SCHEMA_REGISTRY_PATH = V4 / "contracts" / "derived_schemas" / "REGISTRY.json"
SELF_KEY = "self_sha256"
BINDING_HASHES = ("content_sha256", "header_sha256")
ENTRY_REQUIRED = ("n_decisions",)
KINDS = ("train", "eval")
LOCK_STYLE_RULE_KEY = SELF_KEY + "_rule"


class RegistryError(ValueError):
    pass


def canonical_self_sha256(obj: Mapping, key: str = SELF_KEY, exclude: tuple = ()) -> str:
    body = {k: v for k, v in obj.items() if k != key and k not in exclude}
    enc = json.dumps(body, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(enc.encode("utf-8")).hexdigest()


def with_self_sha256(obj: Mapping, key: str = SELF_KEY) -> dict:
    """A copy of `obj` carrying its self-hash (for writers and test fixtures)."""
    out = {k: v for k, v in obj.items() if k != key}
    out[key] = canonical_self_sha256(out, key)
    return out


def _read_object(path: Path, what: str) -> tuple:
    p = Path(path)
    if not p.is_file():
        raise RegistryError(f"{what} {p} does not exist")
    data = p.read_bytes()
    try:
        obj = parse_strict_json(data.decode("utf-8"))
    except ValueError as e:
        raise RegistryError(f"{what} {p} is not strict JSON ({e})") from None
    if not isinstance(obj, dict):
        raise RegistryError(f"{what} {p} is not a JSON object")
    return obj, hashlib.sha256(data).hexdigest()


def _verify_self_hash(obj: Mapping, what: str) -> str:
    stated = obj.get(SELF_KEY)
    got = canonical_self_sha256(obj)
    alt = canonical_self_sha256(obj, exclude=(LOCK_STYLE_RULE_KEY,)) if LOCK_STYLE_RULE_KEY in obj else None
    if not isinstance(stated, str) or stated not in (got, alt):
        raise RegistryError(f"{what}: self-hash mismatch (stated {stated!r}, recomputed {got})")
    return stated


def _is_test_id(mid: str) -> bool:
    return "TEST" in str(mid).upper()


@dataclass(frozen=True)
class ManifestRegistry:
    path: str
    file_sha256: str
    self_sha256: str
    data_version: str
    roles: Mapping
    manifests: Mapping
    raw: Mapping = field(repr=False, compare=False, default_factory=dict)

    def role(self, role: str) -> tuple:
        if role not in self.roles:
            raise RegistryError(f"role {role!r} is not in the manifest registry {self.path} "
                                f"(data_version {self.data_version}; roles {sorted(self.roles)})")
        r = self.roles[role]
        return r["train"], r["eval"]

    def record(self) -> dict:
        return {"path": self.path, "file_sha256": self.file_sha256, "self_sha256": self.self_sha256,
                "data_version": self.data_version}


def load_manifest_registry(path) -> ManifestRegistry:
    """Read and verify a manifest-role registry (self-hash, shape, role -> entry references, no TEST id)."""
    p = Path(path)
    obj, file_sha = _read_object(p, "manifest registry")
    self_sha = _verify_self_hash(obj, f"manifest registry {p}")
    dv = obj.get("data_version")
    if not isinstance(dv, str) or not dv:
        raise RegistryError(f"manifest registry {p}: data_version missing")
    roles, manifests = obj.get("roles"), obj.get("manifests")
    if not isinstance(roles, dict) or not roles or not isinstance(manifests, dict):
        raise RegistryError(f"manifest registry {p}: 'roles' and 'manifests' objects are required")
    for mid in manifests:
        if _is_test_id(mid):
            raise RegistryError(f"manifest registry {p}: TEST manifest {mid!r} refused (VALIDATION-only training)")
    for role, pair in roles.items():
        if role not in ROLES:
            raise RegistryError(f"manifest registry {p}: unknown role {role!r} (roles are {ROLES})")
        if not isinstance(pair, dict) or set(pair) != set(KINDS):
            raise RegistryError(f"manifest registry {p}: role {role} must map exactly 'train' and 'eval'")
        for kind in KINDS:
            mid = pair[kind]
            if not isinstance(mid, str) or not mid or _is_test_id(mid):
                raise RegistryError(f"manifest registry {p}: role {role} {kind} manifest {mid!r} refused")
            e = manifests.get(mid)
            if not isinstance(e, dict):
                raise RegistryError(f"manifest registry {p}: role {role} names {mid!r}, which has no manifest entry")
            miss = [f for f in ENTRY_REQUIRED if e.get(f) in (None, "")]
            if miss:
                raise RegistryError(f"manifest registry {p}: entry {mid} lacks {miss}")
            if not isinstance(e["n_decisions"], int) or isinstance(e["n_decisions"], bool) or e["n_decisions"] < 1:
                raise RegistryError(f"manifest registry {p}: entry {mid} n_decisions must be a positive int")
            if not (_hash_binding(e) or _fp_binding(e)):
                raise RegistryError(f"manifest registry {p}: entry {mid} binds nothing (it needs content_sha256 + "
                                    "header_sha256 and / or a fingerprint)")
    return ManifestRegistry(path=str(p), file_sha256=file_sha, self_sha256=self_sha, data_version=dv,
                            roles={k: dict(v) for k, v in roles.items()},
                            manifests={k: dict(v) for k, v in manifests.items()}, raw=obj)


def _hash_binding(entry: Mapping) -> bool:
    return all(isinstance(entry.get(h), str) and entry.get(h) for h in BINDING_HASHES)


def _fp_binding(entry: Mapping) -> bool:
    return isinstance(entry.get("fingerprint"), str) and bool(entry.get("fingerprint"))


def role_of(population: str, phase: str, role: Optional[str] = None) -> str:
    if role is not None:
        return role
    return "FULL_TRAIN" if population == "FULL_TRAIN" else phase


def check_provider(reg: ManifestRegistry, role: str, kind: str, provider) -> dict:
    """Refuse a provider whose manifest id or identity does not match the registry entry of `role`."""
    want = reg.role(role)[KINDS.index(kind)]
    got = provider.manifest_id if kind == "train" else provider.manifest.manifest_id
    if got != want:
        raise RegistryError(f"{role} {kind} manifest {got!r} is not the registry's {want!r} "
                            f"(data_version {reg.data_version}, {reg.path})")
    entry = reg.manifests[want]
    n = int(provider.n_decisions) if kind == "train" else int(provider.manifest.n)
    if n != int(entry["n_decisions"]):
        raise RegistryError(f"{want}: the provider serves {n} decisions, the registry entry states "
                            f"{entry['n_decisions']}")
    bound = []
    if _fp_binding(entry):
        if entry["fingerprint"] != provider.fingerprint:
            raise RegistryError(f"{want}: the provider fingerprint differs from the registry entry's")
        bound.append("fingerprint")
    if _hash_binding(entry):
        b = getattr(provider, "manifest_binding", None)
        if not isinstance(b, Mapping):
            raise RegistryError(f"the {kind} provider of {want} declares no manifest_binding: it cannot be checked "
                                "against the registry entry's content / header hashes")
        if b.get("manifest_id") != want:
            raise RegistryError(f"the {kind} provider's binding names {b.get('manifest_id')!r}, not {want!r}")
        bad = [h for h in BINDING_HASHES if b.get(h) != entry[h]]
        if bad:
            raise RegistryError(f"{want}: the provider's {bad} differ from the registry entry (fingerprint mismatch)")
        bound.append("content_header_sha256")
    if not bound:                                             # load_manifest_registry refuses this already
        raise RegistryError(f"{want}: the registry entry binds nothing")
    return {"manifest_id": want, "content_sha256": entry.get("content_sha256"),
            "header_sha256": entry.get("header_sha256"), "n_decisions": n, "fingerprint": provider.fingerprint,
            "bound_by": bound}


# ------------------------------------------------------------------------------------------------ derived schemas
def _base_schema():
    from v4models.schema import load_schema
    return load_schema()


def _schema_entries(reg: Mapping) -> dict:
    if isinstance(reg.get("schemas"), Mapping):
        return dict(reg["schemas"])
    return {k: v for k, v in reg.items() if isinstance(v, Mapping) and "path" in v}


def _derived_entry(reg: Mapping, schema_id: str, where: str) -> Mapping:
    e = _schema_entries(reg).get(schema_id)
    if not isinstance(e, Mapping) or not e.get("path") or not e.get("self_sha256"):
        raise RegistryError(f"schema {schema_id!r} is not in the derived-schema registry {where}")
    return e


def _resolve(entry_path: str, reg_path: Path) -> Path:
    p = Path(entry_path)
    for c in ([p] if p.is_absolute() else [reg_path.parent / p, V4 / p]):
        if c.is_file():
            return c
    raise RegistryError(f"registered schema file {entry_path!r} not found (next to {reg_path} or under {V4})")


def verify_schema(schema, registry_path: Optional[str] = None) -> dict:
    """The schema record of a run (kind BASE or DERIVED), after verifying it; raises RegistryError otherwise."""
    from v4models.schema import CleanV2Schema, load_schema
    sid, ssh = getattr(schema, "schema_id", None), getattr(schema, "self_sha256", None)
    if not isinstance(sid, str) or not sid or not isinstance(ssh, str) or len(ssh) != 64:
        raise RegistryError("the schema object must expose schema_id and a 64-hex self_sha256")
    base = _base_schema()
    if isinstance(schema, CleanV2Schema) and ssh == base.self_sha256 and sid == base.schema_id:
        again = load_schema(schema.path)                  # re-verified from its file
        if again.self_sha256 != ssh or again.widths != schema.widths:
            raise RegistryError(f"the base schema object differs from its file {schema.path}")
        return {"kind": "BASE", "schema_id": sid, "schema_self_sha256": ssh, "file": str(schema.path),
                "file_sha256": again.file_sha256, "registry": None}
    rp = Path(registry_path) if registry_path else DERIVED_SCHEMA_REGISTRY_PATH
    reg, reg_file_sha = _read_object(rp, "derived-schema registry")
    reg_self = _verify_self_hash(reg, f"derived-schema registry {rp}")
    e = _derived_entry(reg, sid, str(rp))
    if e["self_sha256"] != ssh:
        raise RegistryError(f"schema {sid}: self_sha256 {ssh} differs from the registry's {e['self_sha256']}")
    fp = _resolve(str(e["path"]), rp)
    obj, file_sha = _read_object(fp, f"schema {sid}")
    if e.get("file_sha256") not in (None, "") and e["file_sha256"] != file_sha:
        raise RegistryError(f"schema {sid}: the file sha256 of {fp} differs from the registry entry's")
    if _verify_self_hash(obj, f"schema file {fp}") != ssh or obj.get("schema_id") != sid:
        raise RegistryError(f"the registered file {fp} is not schema {sid} with self_sha256 {ssh}")
    raw = getattr(schema, "raw", None)
    if isinstance(raw, Mapping) and SELF_KEY in raw:
        if raw.get(SELF_KEY) != ssh or canonical_self_sha256(raw) != ssh:
            raise RegistryError(f"the schema object's content does not hash to its self_sha256 {ssh}")
    return {"kind": "DERIVED", "schema_id": sid, "schema_self_sha256": ssh, "file": str(fp), "file_sha256": file_sha,
            "registry": {"path": str(rp), "file_sha256": reg_file_sha, "self_sha256": reg_self}}


__all__ = ["RegistryError", "ManifestRegistry", "load_manifest_registry", "check_provider", "role_of",
           "verify_schema", "canonical_self_sha256", "with_self_sha256", "HISTORICAL_REGISTRY_PATH",
           "DERIVED_SCHEMA_REGISTRY_PATH", "SELF_KEY", "BINDING_HASHES", "ENTRY_REQUIRED"]
