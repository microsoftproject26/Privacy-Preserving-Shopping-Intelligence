"""INT-01 test configuration: ppsi_env FIRST (snapshot isolation, T43; the pattern of code/tests_train/conftest.py, so
v4train records the launch value UNKNOWN_PPSI_ENV_ALREADY_IMPORTED), strict FP32 with deterministic algorithms,
1 CPU thread, no byte-code. (The launcher itself imports v4train first, whose first act is importing ppsi_env, so v4train
can capture the exported CUDA_VISIBLE_DEVICES.)

Negative controls are `xfail(strict=True, raises=AssertionError)` tests named test_nc_* (data_testkit.nc): each runs a
key-property check against a deliberately wrong variant and MUST fail at its assertion (an XPASS fails the suite).
Store-level fixtures read the frozen v1 dataset read-only: bucket 0, TRAIN and the two VALIDATION splits (never TEST).
Every environment is written under the test's own tmp directory.
"""
from __future__ import annotations

import json
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

import v4eval.frozen_config  # noqa: E402,F401
from fl.numerics import enforce_strict_fp32  # noqa: E402


@pytest.fixture(autouse=True)
def _strict_numerics():
    torch.set_num_threads(1)
    enforce_strict_fp32()
    yield
    enforce_strict_fp32()


@pytest.fixture(scope="session")
def real_train_store():
    from intkit import REAL_DS
    from v4data.store import CleanV2Store
    if not (Path(REAL_DS) / "model_ready").is_dir():
        pytest.skip("the frozen v1 dataset is not present on this host")
    return CleanV2Store.build(splits=("TRAIN",), buckets=[0], ds_root=REAL_DS, log=lambda *_: None)


@pytest.fixture(scope="session")
def real_val_store():
    from intkit import REAL_DS
    from v4data.common import VALIDATION_SPLITS
    from v4data.store import CleanV2Store
    if not (Path(REAL_DS) / "model_ready").is_dir():
        pytest.skip("the frozen v1 dataset is not present on this host")
    return CleanV2Store.build(splits=VALIDATION_SPLITS, buckets=[0], ds_root=REAL_DS, log=lambda *_: None)


@pytest.fixture(scope="session")
def real_env(tmp_path_factory, real_train_store, real_val_store):
    """A locked TEST environment: M05 (real TRAIN rows of 60 clients), M06 (real VALIDATION rows of 120 users + censored
    rows), M01 (80 clients) and M02 (150 users), all from bucket 0."""
    from intkit import real_eval_tables, real_train_tables, write_env
    root = tmp_path_factory.mktemp("int01_env")
    tabs = {"M05_CALIB_TRAIN": real_train_tables(real_train_store, n_clients=60, seed=0),
            "M01_COHORT_TRAIN": real_train_tables(real_train_store, n_clients=80, seed=3),
            "M06_CALIB_VALIDATION_WARM": real_eval_tables(real_val_store, n_users=120, seed=1),
            "M02_VALIDATION_WARM": real_eval_tables(real_val_store, n_users=150, seed=4)}
    paths = write_env(root, tabs)
    return {"root": root, "paths": paths, "tables": tabs}


@pytest.fixture(scope="session")
def providers(real_env):
    from intkit import REAL_DS
    from v4int.providers import StoreEvalProvider, StoreTrainProvider
    q = lambda *_: None  # noqa: E731
    tp = StoreTrainProvider("M05_CALIB_TRAIN", history_policy="SESSION_L50", paths=real_env["paths"], ds_root=REAL_DS, buckets=[0], log=q)
    ep = StoreEvalProvider("M06_CALIB_VALIDATION_WARM", history_policy="SESSION_L50", paths=real_env["paths"], ds_root=REAL_DS, buckets=[0], log=q)
    return tp, ep


@pytest.fixture(scope="session")
def int_registry(tmp_path_factory, providers):
    """A TRAIN-01R2 manifest-role registry (v4train.registry) for the INT-01 test-kit's own M05/M06 pair, bound by
    fingerprint (the providers' pure, deterministic identity) so a fresh `StoreTrainProvider`/`StoreEvalProvider` pair
    over the same manifest + history policy verifies against it exactly like `providers` does. A RunConfig now REQUIRES
    manifest_registry / data_version (TRAIN-01R2); this is the test kit's own registry, never contracts/*."""
    from v4train.registry import with_self_sha256
    tp, ep = providers
    obj = {"data_version": "PPSI_V4_INT01_HISTORICAL_HARNESS",
           "roles": {"CALIBRATION": {"train": "M05_CALIB_TRAIN", "eval": "M06_CALIB_VALIDATION_WARM"}},
           "manifests": {"M05_CALIB_TRAIN": {"n_decisions": int(tp.n_decisions), "fingerprint": tp.fingerprint},
                        "M06_CALIB_VALIDATION_WARM": {"n_decisions": int(ep.manifest.n), "fingerprint": ep.fingerprint}}}
    obj = with_self_sha256(obj)
    p = tmp_path_factory.mktemp("int01_registry") / "REGISTRY_INT01_HARNESS.json"
    p.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return p
