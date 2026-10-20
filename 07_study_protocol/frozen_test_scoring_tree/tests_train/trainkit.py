"""Shared helpers for the TRAIN-01 tests (synthetic in-memory providers only; no dataset rows).

TRAIN-01R2: every config states `manifest_registry` and `data_version` explicitly. The default registry is written
once per process into a temporary directory OUTSIDE the code tree (removed at exit): data version
CLEAN_V2_HISTORICAL_262144 with the HISTORICAL registry file's role map (MAIN / CONTINUATION -> M01 / M02, CALIBRATION
-> M05 / M06) plus the synthetic-only FULL_TRAIN role (M_FULL_TRAIN / M_FULL_VALIDATION_WARM, never built), each entry
bound (content / header hashes, n_decisions) to the default synthetic providers. FS_REGISTRY has FS-01's shape (data
version FS_SCREEN_5_100_v1, role FEATURE_SEARCH_SCREEN; a config selects it with role="FEATURE_SEARCH_SCREEN"). `write_registry` binds any other providers; `derived_schema` / `write_schema_registry` build
SCHEMA-01-shaped derived schemas (base widths, so the model builds unchanged) and their registry.
"""
from __future__ import annotations

import atexit
import collections
import dataclasses
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Mapping, Optional

import pytest

import v4train  # noqa: F401  (ppsi_env first)
from v4eval.rows import make_row
from v4models.schema import load_schema
from v4train.provider import SyntheticEvalProvider, SyntheticTrainProvider
from v4train.recipe import HISTORICAL_DATA_VERSION, RunConfig
from v4train.registry import HISTORICAL_REGISTRY_PATH, load_manifest_registry, with_self_sha256
from v4train.trainer import CentralTrainer, EvalOutcome, InjectedInterruption

SCHEMA = load_schema()
WIDTHS = SCHEMA.widths
K, N, B, NEVAL = 64, 200, 32, 60          # 7 updates per pass (6 x 32 + a tail of 8)
KEY = {"data_protocol_hash": "SYN-DATA-PROTOCOL", "feature_contract_hash": "SYN-FEATURES", "split": "VALIDATION",
       "label_policy_hash": "SYN-LABEL-POLICY", "population_hash": "SYN-POPULATION",
       "candidate_order_hash": "SYNTHETIC-CANDIDATES"}
_Usage = collections.namedtuple("_Usage", "total used free")
GIB = 2 ** 30

# ---------------------------------------------------------------------------------------- registries (TRAIN-01R2)
FULL_TRAIN_MANIFEST, FULL_EVAL_MANIFEST = "M_FULL_TRAIN", "M_FULL_VALIDATION_WARM"    # synthetic ids only
FS_DATA_VERSION = "FS_SCREEN_5_100_v1"                                                 # FS-01's shape
FS_TRAIN_MANIFEST, FS_EVAL_MANIFEST = "FS_TRAIN_5_100_32768", "FS_VALIDATION_5_100_32768"
NEW_DATA_VERSION = "SYN_POP_5_100_TEST_V1"                  # a non-historical synthetic data version (PRODUCTION)
REG_DIR = Path(tempfile.mkdtemp(prefix="v4train_r2_"))
atexit.register(shutil.rmtree, REG_DIR, True)
HISTORICAL_ROLES = load_manifest_registry(HISTORICAL_REGISTRY_PATH).roles
ROLE_IDS = dict(HISTORICAL_ROLES, FULL_TRAIN={"train": FULL_TRAIN_MANIFEST, "eval": FULL_EVAL_MANIFEST},
                FEATURE_SEARCH_SCREEN={"train": FS_TRAIN_MANIFEST, "eval": FS_EVAL_MANIFEST})
ROLE_IDS["TUNING"] = dict(ROLE_IDS["CALIBRATION"])         # TRAIN-02: TUNING runs on the registry's CALIBRATION role


def role_of(phase: str, population: str, role: Optional[str] = None) -> str:
    return role if role is not None else ("FULL_TRAIN" if population == "FULL_TRAIN" else phase)


def registry_entry(provider, kind: str) -> dict:
    b = provider.manifest_binding
    n = provider.n_decisions if kind == "train" else provider.manifest.n
    return {"path": f"synthetic/{b['manifest_id']}", "content_sha256": b["content_sha256"],
            "header_sha256": b["header_sha256"], "n_users": 0, "n_decisions": int(n),
            "n_eval_rows": 0 if kind == "train" else int(provider.manifest.eval_mask.sum())}


def write_json_file(obj, name: Optional[str] = None) -> str:
    text = json.dumps(obj, indent=1, sort_keys=True)
    p = REG_DIR / (name or f"R_{hashlib.sha256(text.encode()).hexdigest()[:20]}.json")
    p.write_text(text, encoding="utf-8")
    return str(p)


def registry_obj(by_role: Mapping, *, data_version: str = HISTORICAL_DATA_VERSION, extra: Optional[Mapping] = None,
                 entry_extra: Optional[Mapping] = None) -> dict:
    """A self-hashed manifest-role registry binding each role to the given (train, eval) providers."""
    roles, manifests = {}, {}
    for role, (tp, ep) in by_role.items():
        roles[role] = {"train": tp.manifest_id, "eval": ep.manifest.manifest_id}
        for kind, p in (("train", tp), ("eval", ep)):
            mid, e = roles[role][kind], registry_entry(p, kind)
            e.update((entry_extra or {}).get(mid, {}))
            if mid in manifests and manifests[mid] != e:
                raise ValueError(f"two different providers registered as {mid}")
            manifests[mid] = e
    return with_self_sha256({"data_version": data_version, "roles": roles, "manifests": manifests, **(extra or {})})


def write_registry(by_role: Mapping, *, name: Optional[str] = None, **kw) -> str:
    return write_json_file(registry_obj(by_role, **kw), name)


# ---------------------------------------------------------------------------------------- derived schemas
def derived_schema(schema_id: str = "L0_SYNTH_TEST", *, write: bool = True):
    """A SCHEMA-01-shaped derived schema with the base widths (a CleanV2Schema carrying the derived id and hash)."""
    w = SCHEMA.widths
    obj = with_self_sha256({
        "schema_id": schema_id, "base": f"FEATURE_SCHEMA_CLEAN_V2 2.1.0 self {SCHEMA.self_sha256}",
        "keep": {"categorical": list(w.categorical), "event_numeric_features": list(w.numeric_names),
                 "event_quality_flags": list(w.flag_names), "user_context": list(w.user_context_names),
                 "user_context_masks": list(w.user_mask_names)},
        "widths": {"event_numeric_features": w.numeric_dim, "event_quality_flags": w.flags_dim,
                   "user_context": w.user_context_dim, "user_context_masks": w.user_mask_dim}})
    path = REG_DIR / f"{schema_id}.json"
    if write:
        path.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    return dataclasses.replace(SCHEMA, schema_id=schema_id, self_sha256=obj["self_sha256"], raw=obj, path=str(path))


def write_schema_registry(*schemas, top_level: bool = False, name: Optional[str] = None) -> str:
    entries = {s.schema_id: {"path": Path(s.path).name, "self_sha256": s.self_sha256} for s in schemas}
    body = dict(entries) if top_level else {"schemas": entries}
    return write_json_file(with_self_sha256(dict(body, registry_id="SYNTH_DERIVED_SCHEMAS")), name)


def nc(reason: str):
    """Negative control: the wrapped key-property check must raise AssertionError (strict xfail)."""
    return pytest.mark.xfail(strict=True, raises=AssertionError, reason="NEGATIVE CONTROL: " + reason)


def assert_raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc}")


def disk(free_gib: float):
    return lambda _path: _Usage(int(2000 * GIB), int(2000 * GIB - free_gib * GIB), int(free_gib * GIB))


BIG_DISK = disk(1000.0)


def providers(phase: str = "MAIN", seed: int = 3, n: int = N, n_eval: int = NEVAL, population: str = "COHORT",
              role: Optional[str] = None):
    ids = ROLE_IDS[role_of(phase, population, role)]
    return (SyntheticTrainProvider(n, K, WIDTHS, seed=seed, manifest_id=ids["train"]),
            SyntheticEvalProvider(n_eval, K, WIDTHS, seed=seed + 1, manifest_id=ids["eval"]))


DEFAULT_REGISTRY = write_registry({"MAIN": providers("MAIN"), "CONTINUATION": providers("MAIN"),
                                   "CALIBRATION": providers("CALIBRATION"),
                                   "FULL_TRAIN": providers(population="FULL_TRAIN")}, name="REGISTRY_DEFAULT.json")
FS_REGISTRY = write_registry({"FEATURE_SEARCH_SCREEN": providers(role="FEATURE_SEARCH_SCREEN")},
                             data_version=FS_DATA_VERSION, name="REGISTRY_FS.json")


def registry_keys(role: Optional[str] = None) -> dict:
    """The explicit TRAIN-01R2 keys of the default synthetic registries."""
    if role == "FEATURE_SEARCH_SCREEN":
        return {"manifest_registry": FS_REGISTRY, "data_version": FS_DATA_VERSION}
    return {"manifest_registry": DEFAULT_REGISTRY, "data_version": HISTORICAL_DATA_VERSION}


def cfg(run_id: str, family: str, *, seed: int = 2026, phase: str = "MAIN", peak: float = 1e-3, **kw) -> RunConfig:
    kw.setdefault("effective_batch", B)
    kw.setdefault("log_every", 5)
    kw.setdefault("key", dict(KEY))
    kw.setdefault("exposure_point", "end")                   # always stated explicitly (PREG1 T2 P1)
    kw.setdefault("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES"))   # explicit, equal to the env
    kw.setdefault("determinism_mode", "STRICT")              # always stated explicitly (PREG1_AMENDMENT_01 item 4)
    for k, v in registry_keys(kw.get("role")).items():
        kw.setdefault(k, v)                                   # always stated explicitly (TRAIN-01R2)
    return RunConfig(run_id=run_id, family=family, seed=seed, phase=phase, peak_lr=peak, test_overrides=True, **kw)


def trainer(run_dir: Path, config: RunConfig, tp=None, ep=None, *, schema=None, **kw) -> CentralTrainer:
    if tp is None or ep is None:
        tp2, ep2 = providers(config.phase, population=config.population, role=config.role)
        tp, ep = tp or tp2, ep or ep2
    kw.setdefault("disk_usage_fn", BIG_DISK)
    return CentralTrainer(config, SCHEMA if schema is None else schema, tp, ep, run_dir, K=K, **kw)


def interrupt_after(update: int):
    def f(u: int) -> None:
        if u == update:
            raise InjectedInterruption(f"injected crash after update {u}")
    return f


def scripted(values: Mapping[float, float], ndcg: Optional[Mapping[float, float]] = None):
    """An evaluator stand-in returning pre-scripted primary values per mark, with real v4eval result rows."""
    def ev(tr: CentralTrainer, mark: float) -> EvalOutcome:
        v = float(values[round(mark, 6)])
        nd = float((ndcg or {}).get(round(mark, 6), v / 2))
        base = dict(tr.key_base, query_order_hash="SCRIPTED-QUERY-ORDER", evaluator_code_sha256=tr.evaluator_sha,
                    eval_score_block=tr.frozen.score_block, eval_batch=tr.frozen.eval_batch,
                    eval_device_class=tr.eval_device_class, eval_stream="ALL_ROWS",
                    aggregation="macro", target_denominator="E2E")
        efe = tr.state.exposures / tr.N
        rows = [make_row(dict(base, metric="mrr", k=20), method="C", seed=tr.cfg.seed, value=v, n_decisions=10,
                         n_users=5, oov_count=0, coverage=1.0, efe=efe),
                make_row(dict(base, metric="ndcg", k=10), method="C", seed=tr.cfg.seed, value=nd, n_decisions=10,
                         n_users=5, oov_count=0, coverage=1.0, efe=efe)]
        rows = [dict(r, **tr._row_labels()) for r in rows]
        return EvalOutcome(value=v, rows=rows, secondary={"ndcg@10": nd})
    return ev


def ramp(marks, peak_at: float, top: float = 0.5, step: float = 0.01) -> dict:
    """Values that rise by `step` per mark up to `peak_at`, then stay flat (no further strict improvement)."""
    out, v = {}, 0.1
    for m in marks:
        if m <= peak_at + 1e-9:
            v = min(top, 0.1 + step * (m / 0.5))
        out[round(m, 6)] = v
    return out


def art(run_dir) -> Path:
    """TRAIN-03: the artifact root of a run — <run_dir>/sealed for a sealed MAIN contest run, else run_dir."""
    d = Path(run_dir)
    return d / "sealed" if (d / "sealed").is_dir() else d


def read(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


TIME_FIELDS = ("data_wait_s", "step_wall_s", "utc", "attempt")


def logical_metrics(run_dir: Path) -> list:
    """metrics.jsonl without timing fields, de-duplicated so a replayed window after a resume counts once."""
    recs = {}
    for line in (art(run_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        key = (r["kind"], r.get("update"), r.get("mark"))
        recs[key] = {k: v for k, v in r.items() if k not in TIME_FIELDS}
    return [recs[k] for k in sorted(recs, key=lambda k: (k[1] or 0, k[0]))]


def raises_with(exc, text, fn, *a, **kw):
    """fn must raise `exc` and its message must name the intended cause (not merely any refusal)."""
    try:
        fn(*a, **kw)
    except exc as e:
        err = e
    else:
        raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__} ({text!r})")
    assert text in str(err), f"refused for another reason: {err}"
    return err


__all__ = ["SCHEMA", "WIDTHS", "K", "N", "B", "NEVAL", "KEY", "GIB", "nc", "assert_raises", "disk", "BIG_DISK",
           "providers", "cfg", "trainer", "interrupt_after", "scripted", "ramp", "read", "logical_metrics",
           "FULL_TRAIN_MANIFEST", "FULL_EVAL_MANIFEST", "FS_DATA_VERSION", "FS_TRAIN_MANIFEST", "FS_EVAL_MANIFEST",
           "NEW_DATA_VERSION", "REG_DIR", "HISTORICAL_ROLES", "ROLE_IDS", "DEFAULT_REGISTRY", "FS_REGISTRY",
           "registry_keys", "registry_entry", "registry_obj", "write_registry", "write_json_file", "derived_schema",
           "write_schema_registry", "raises_with", "art"]
