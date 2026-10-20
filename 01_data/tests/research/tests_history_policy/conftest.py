"""DATA-05 test configuration: ppsi_env FIRST (snapshot isolation), strict FP32, 1 CPU thread, no byte-code.

Negative controls are `xfail(strict=True, raises=AssertionError)` tests named test_nc_* (data_testkit.nc): each runs a
key-property check against a deliberately wrong variant and MUST fail at its assertion (an XPASS fails the suite).
Store-level fixtures read the frozen v1 dataset read-only: bucket 0, TRAIN and the two VALIDATION splits (never TEST;
every feature scan is split-filtered by the snapshot store or by v4data.manifests.read_non_test).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for _p in (str(CODE), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import ppsi_env  # noqa: E402,F401  (mandatory isolation, before any snapshot import)

import pytest  # noqa: E402

import v4train  # noqa: E402,F401
import torch  # noqa: E402

from fl.numerics import enforce_strict_fp32  # noqa: E402


@pytest.fixture(autouse=True)
def _strict_numerics():
    torch.set_num_threads(1)
    enforce_strict_fp32()
    yield
    enforce_strict_fp32()


def _need_ds():
    from histkit import REAL_DS
    if not (Path(REAL_DS) / "model_ready").is_dir():
        pytest.skip("the frozen v1 dataset is not present on this host")
    return REAL_DS


@pytest.fixture(scope="session")
def real_train_store():
    """The INT-01 / v1 streaming store of TRAIN decisions (decision-split store; the SESSION_L50 reference)."""
    from v4data.store import CleanV2Store
    return CleanV2Store.build(splits=("TRAIN",), buckets=[0], ds_root=_need_ds(), log=lambda *_: None)


@pytest.fixture(scope="session")
def real_val_store():
    from v4data.common import VALIDATION_SPLITS
    from v4data.store import CleanV2Store
    return CleanV2Store.build(splits=VALIDATION_SPLITS, buckets=[0], ds_root=_need_ds(), log=lambda *_: None)


@pytest.fixture(scope="session")
def hist_train():
    """USER_HISTORY_L50 streaming path for TRAIN decisions (store: TRAIN)."""
    from v4hist.stream import HistoryStore
    return HistoryStore.build("USER_HISTORY_L50", ("TRAIN",), buckets=[0], ds_root=_need_ds(), log=lambda *_: None)


@pytest.fixture(scope="session")
def hist_val():
    """USER_HISTORY_L50 streaming path for VALIDATION decisions (store: TRAIN + VALIDATION_SELECT + VALIDATION_CONFIRM)."""
    from v4data.common import VALIDATION_SPLITS
    from v4hist.stream import HistoryStore
    return HistoryStore.build("USER_HISTORY_L50", VALIDATION_SPLITS, buckets=[0], ds_root=_need_ds(),
                              log=lambda *_: None)


@pytest.fixture(scope="session")
def hist_train_on_full(hist_val):
    """TRAIN decisions served from the TRAIN + VALIDATION store (same store object as hist_val)."""
    from v4hist.stream import HistoryStore
    return HistoryStore(hist_val.store, hist_val.index, "USER_HISTORY_L50", ("TRAIN",))


DIRECT_ENV_DATA_VERSION = "TESTS_HIST_DIRECT_ENV"          # INT-02R1 (OPS-10): set on the headers AT CONSTRUCTION
# time (never mutated afterwards — that would invalidate the R1 list / .OK marker `write_env` already wrote) so a
# registry bound to this SAME env can match `header.data_version_eq_registry`.


@pytest.fixture(scope="session")
def real_env(tmp_path_factory, real_train_store, real_val_store):
    """A locked TEST manifest environment over real bucket-0 rows (intkit, read-only import): M05 (1,500 TRAIN clients,
    >= 20,000 decisions) and M06 (120 VALIDATION users + fabricated censored rows). Headers carry `data_version`
    (INT-02R1 OPS-10) so `real_env_registry` below can bind a registry to them without ever rewriting a header after
    `write_env` has fixed its R1 list / .OK marker."""
    from intkit import lock_and_reissue, real_eval_tables, real_train_tables, write_manifest_files
    root = tmp_path_factory.mktemp("data05_env")
    d2 = root / "data_v2"
    tabs = {"M05_CALIB_TRAIN": real_train_tables(real_train_store, n_clients=1500, seed=7),
            "M06_CALIB_VALIDATION_WARM": real_eval_tables(real_val_store, n_users=120, seed=1)}
    entries = {mid: write_manifest_files(d2 / "manifests", mid, t, header_extra={"data_version": DIRECT_ENV_DATA_VERSION})
              for mid, t in tabs.items()}
    return {"root": root, "paths": lock_and_reissue(d2, entries), "tables": tabs}


def _direct_registry_for(real_env) -> tuple:
    """INT-02R1 (OPS-10): v4int.providers.comparison_key_base now requires registry-verified providers (R4). This
    binds a small TEST manifest-role registry DIRECTLY to real_env['paths'].lock_path's own hash (the v1-shaped TEST
    lock `real_env` already built) — the DIRECT route, no B1 needed, exactly mirroring how FS-01 / DATA-06's real
    registries bind to the v1 lock. Returns (registry_path, root)."""
    import json as _json
    from v4int.verify import manifest_content_sha256, registry_self_sha256, verify_lock, verify_transform_state
    paths, root = real_env["paths"], real_env["root"]
    lock, lself, lfile = verify_lock(paths)
    ts_self, ts_file = verify_transform_state(paths, lock)
    rel_dir = str(paths.manifest_dir.relative_to(root)).replace("\\", "/")
    mids = {"M05_CALIB_TRAIN": "TRAIN", "M06_CALIB_VALIDATION_WARM": "EVAL"}
    manifests, status = {}, None
    for mid, kind in mids.items():
        hp = paths.manifest_dir / f"{mid}.header.json"
        h = _json.loads(hp.read_text(encoding="utf-8"))
        status = h["status"]
        tables_meta = {t["table"]: {"file": f"{rel_dir}/{mid}.{t['table']}.parquet", "file_sha256": t["file_sha256"],
                                    "row_count": t["row_count"], "table_content_sha256": t["table_content_sha256"]}
                       for t in h["tables"]}
        manifests[mid] = {"path": f"{rel_dir}/{mid}.header.json", "kind": kind, "status": status,
                          "content_sha256": manifest_content_sha256(h["tables"], tables_meta),
                          "header_sha256": _sha256_file(hp), "header_content_sha256": h["header_content_sha256"],
                          "n_decisions": h["n_decisions"], "tables": tables_meta, "marker": f"{rel_dir}/{mid}.OK"}
    reg = {"data_version": DIRECT_ENV_DATA_VERSION, "roles": {status: {"train": "M05_CALIB_TRAIN",
                                                                       "eval": "M06_CALIB_VALIDATION_WARM"}},
          "manifests": manifests, "transform_state": {"transform_state_sha256": ts_self, "file_sha256": ts_file},
          "source_data_protocol_lock": {"lock_self_sha256": lself, "file_sha256": lfile}}
    reg["self_sha256"] = registry_self_sha256(reg)
    reg_path = root / "REGISTRY_DIRECT_TESTENV.json"
    reg_path.write_text(_json.dumps(reg, indent=1, sort_keys=True), encoding="utf-8")
    return reg_path, root


def _sha256_file(p) -> str:
    import hashlib
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture(scope="session")
def real_env_registry(real_env):
    return _direct_registry_for(real_env)


def _providers(real_env, policy, registry=None):
    from histkit import REAL_DS, Q
    from v4int.providers import StoreEvalProvider, StoreTrainProvider
    reg_kw = {}
    if registry is not None:
        reg_kw = {"manifest_registry": registry[0], "registry_root": registry[1]}
    tp = StoreTrainProvider("M05_CALIB_TRAIN", history_policy=policy, paths=real_env["paths"], ds_root=REAL_DS,
                            buckets=[0], log=Q, **reg_kw)
    ep = StoreEvalProvider("M06_CALIB_VALIDATION_WARM", history_policy=policy, paths=real_env["paths"],
                           ds_root=REAL_DS, buckets=[0], log=Q, **reg_kw)
    return tp, ep


@pytest.fixture(scope="session")
def providers_user(real_env):
    """Deliberately NOT registry-bound (INT-02R1 OPS-10): several tests compare `.fingerprint` against an
    independently, freshly rebuilt provider over the SAME manifest (test_determinism.py), and `_registry_fp` folds
    the registry's identity into the fingerprint — so any registry binding here would make those rebuilds diverge.
    Tests that need comparison_key_base's R4 registry-verified path build their OWN registry-bound provider pair
    (`registered_providers`) alongside these, over the exact same manifest / policy / transform state."""
    return _providers(real_env, "USER_HISTORY_L50")


@pytest.fixture(scope="session")
def providers_session(real_env):
    return _providers(real_env, "SESSION_L50")


def registered_providers(real_env, real_env_registry, policy, *, transform_state=None):
    """A registry-bound provider pair (INT-02R1 OPS-10 / R4) over the SAME manifest / policy / transform state as
    `_providers` — for comparison_key_base calls only; never compared by `.fingerprint` against a non-registry pair
    built over the same manifest (see `providers_user`'s note)."""
    from histkit import REAL_DS, Q
    from v4int.providers import StoreEvalProvider, StoreTrainProvider
    reg_path, root = real_env_registry
    kw = dict(paths=real_env["paths"], ds_root=REAL_DS, buckets=[0], log=Q, manifest_registry=reg_path,
             registry_root=root)
    if transform_state is not None:
        kw["transform_state"] = transform_state
    tp = StoreTrainProvider("M05_CALIB_TRAIN", history_policy=policy, **kw)
    ep = StoreEvalProvider("M06_CALIB_VALIDATION_WARM", history_policy=policy, **kw)
    return tp, ep
