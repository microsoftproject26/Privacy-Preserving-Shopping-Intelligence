"""history_policy is a REQUIRED enum everywhere (no default): the policy module, the cache, both providers and the
launcher; it is bound into the provider fingerprint, describe(), the build record and the comparison key, and a
train / eval pair with different policies is refused. Negative control: a provider without history_policy."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from histkit import REAL_DS, Q, assert_raises, nc
from v4hist.policy import (HISTORY_POLICIES, LOCKED_POLICY, SESSION_L50, USER_HISTORY_L50, HistoryPolicyError,
                           history_splits, needs_row_index, require_policy, store_splits)
from v4hist.windows import input_contract_sha256
import ppsi_paths

CODE = ppsi_paths.code_dir("v4int").parent   # the code folder that holds v4int
VAL = ("VALIDATION_SELECT", "VALIDATION_CONFIRM")


def policy_bound(p) -> None:
    """The key property: a provider carries a registered history_policy, bound into its fingerprint record."""
    pol = getattr(p, "history_policy", None)
    assert pol in HISTORY_POLICIES, "a provider without a registered history_policy"
    d = p.describe()
    assert d["history_policy"] == pol and d["history"]["history_policy"] == pol, "describe() does not record it"
    assert d["build"]["history_policy"] == pol, "the build record does not carry it"


def test_enum_is_required_and_closed():
    assert HISTORY_POLICIES == ("USER_HISTORY_L50", "SESSION_L50") and LOCKED_POLICY == USER_HISTORY_L50
    for p in HISTORY_POLICIES:
        assert require_policy(p) == p
    for bad in (None, "", "USER_HISTORY", "user_history_l50", "SESSION", "ALL", 50, b"SESSION_L50"):
        assert_raises(HistoryPolicyError, require_policy, bad)
    assert needs_row_index(USER_HISTORY_L50) and not needs_row_index(SESSION_L50)


def test_store_partitions_per_policy_and_test_refused():
    assert store_splits(USER_HISTORY_L50, ("TRAIN",)) == ("TRAIN",)
    assert store_splits(USER_HISTORY_L50, VAL) == ("TRAIN",) + VAL
    assert store_splits(USER_HISTORY_L50, ("VALIDATION_SELECT",)) == ("TRAIN", "VALIDATION_SELECT")
    assert store_splits(SESSION_L50, VAL) == VAL and store_splits(SESSION_L50, ("TRAIN",)) == ("TRAIN",)
    for bad in (("TEST",), ("RETRO_TEST",), ("TRAIN", "TEST"), ("LABEL_GRACE",), ()):
        assert_raises(HistoryPolicyError, history_splits, bad)
        assert_raises(HistoryPolicyError, store_splits, USER_HISTORY_L50, bad)


def test_cache_builder_requires_the_policy():
    from v4int.cache import build_subset_store
    ids = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(TypeError):
        build_subset_store(ids, splits=("TRAIN",), kind="loss", state_sha256=None)     # no default
    for bad in (None, "SESSION"):
        assert_raises(HistoryPolicyError, build_subset_store, ids, splits=("TRAIN",), kind="loss", state_sha256=None,
                      history_policy=bad, ds_root=Path("does-not-exist"))


def test_providers_require_the_policy_before_any_file_is_opened(tmp_path):
    from v4int.providers import StoreEvalProvider, StoreTrainProvider
    from v4int.verify import DataPaths
    nowhere = DataPaths(manifest_dir=tmp_path / "none", lock_path=tmp_path / "none.json",
                        manifest_hashes_path=tmp_path / "none2.json", header_list_path=tmp_path / "none.txt",
                        transform_state_path=tmp_path / "none3.json", production=False)
    for cls in (StoreTrainProvider, StoreEvalProvider):
        with pytest.raises(TypeError):
            cls("M05_CALIB_TRAIN", paths=nowhere, log=Q)                               # the argument is required
        for bad in (None, "", "USER", "session_l50"):
            assert_raises(HistoryPolicyError, cls, "M05_CALIB_TRAIN", history_policy=bad, paths=nowhere, log=Q)


def test_policy_bound_into_fingerprint_describe_and_build(providers_user, providers_session):
    tu, eu = providers_user
    ts, es = providers_session
    for p, pol in ((tu, USER_HISTORY_L50), (eu, USER_HISTORY_L50), (ts, SESSION_L50), (es, SESSION_L50)):
        policy_bound(p)
        assert p.history_policy == pol and p.cache_info.history_policy == pol
    assert tu.fingerprint != ts.fingerprint and eu.fingerprint != es.fingerprint
    assert np.array_equal(tu.decision_ids(), ts.decision_ids())                  # same decisions, same targets
    assert tu.describe()["history"]["store_splits"] == ["TRAIN"]
    assert eu.describe()["history"]["store_splits"] == ["TRAIN", "VALIDATION_SELECT", "VALIDATION_CONFIRM"]
    assert es.describe()["history"]["store_splits"] == list(VAL)
    assert tu.cache_info.mean_window_length > ts.cache_info.mean_window_length


def test_comparison_key_binds_the_policy_and_refuses_a_mixed_pair(real_env, real_env_registry):
    from conftest import registered_providers
    from v4eval.rows import comparison_status
    from v4int.providers import ProviderError, comparison_key_base
    from intkit import real_lock
    # R4 (INT-02R1 OPS-10): comparison_key_base needs registry-verified providers; providers_user / providers_session
    # stay deliberately unregistered (their .fingerprint is compared against independently rebuilt providers
    # elsewhere in this suite — v4int.providers._registry_fp folds the registry into it), so this test builds its own
    # registry-bound pair per policy, over the exact same manifests.
    tu, eu = registered_providers(real_env, real_env_registry, USER_HISTORY_L50)
    ts, es = registered_providers(real_env, real_env_registry, SESSION_L50)
    ku = comparison_key_base(real_env["paths"], tu, eu)
    ks = comparison_key_base(real_env["paths"], ts, es)
    schema = real_lock()["features_and_transforms"]["schema_self_sha256"]
    assert ku["history_policy"] == USER_HISTORY_L50 and ks["history_policy"] == SESSION_L50
    assert ku["feature_contract_hash"] == input_contract_sha256(schema, USER_HISTORY_L50)
    assert ks["feature_contract_hash"] == input_contract_sha256(schema, SESSION_L50)
    assert ku["feature_contract_hash"] != ks["feature_contract_hash"]
    assert ku["feature_schema_self_sha256"] == ks["feature_schema_self_sha256"] == schema
    assert {k: v for k, v in ku.items() if k not in ("history_policy", "feature_contract_hash")} == \
        {k: v for k, v in ks.items() if k not in ("history_policy", "feature_contract_hash")}
    st, diff = comparison_status(ku, ks)                  # the LOCKED row comparison separates the two policies
    assert st == "HISTORICAL_REFERENCE" and diff == ["feature_contract_hash"]
    assert_raises(ProviderError, comparison_key_base, real_env["paths"], tu, es)
    assert_raises(ProviderError, comparison_key_base, real_env["paths"], ts, eu)


def _args(*extra, policy=USER_HISTORY_L50):
    import v4int.launch_central as L
    a = ["--family", "GRU", "--seed", "2026", "--phase", "CALIBRATION", "--peak-lr", "1e-3", "--run-dir", "r",
         "--gpu", "0", "--smoke", "--checkpoint-every", "300"]
    if policy is not None:
        a += ["--history-policy", policy]
    return L.parse_args(a + list(extra))


def test_launcher_requires_the_policy():
    for pol in HISTORY_POLICIES:
        assert _args(policy=pol).history_policy == pol
    with pytest.raises(SystemExit):
        _args(policy=None)
    with pytest.raises(SystemExit):
        _args(policy="USER_HISTORY")
    import v4int.launch_central as L
    from types import SimpleNamespace
    a = _args(policy=SESSION_L50)
    bad = SimpleNamespace(**{**vars(a), "history_policy": None})
    assert_raises(L.LaunchRefused, L.preflight, bad, runner=lambda *_a, **_k: None, environ={})


def test_launcher_cli_refuses_without_the_policy(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    base = ["--family", "GRU", "--seed", "2026", "--phase", "CALIBRATION", "--peak-lr", "1e-3", "--gpu", "0",
            "--run-dir", str(tmp_path / "run")]
    p = subprocess.run([sys.executable, "-B", str(CODE / "v4int" / "launch_central.py"), *base], cwd=str(CODE),
                       env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode == 2 and "--history-policy" in p.stderr, p.stderr[-600:]
    assert not (tmp_path / "run").exists()


def test_comparison_key_refuses_a_provider_without_policy(real_env, providers_user):
    from v4int.providers import ProviderError, comparison_key_base
    tu, eu = providers_user

    class NoPolicy:                                        # an INT-01-era provider object (no history_policy)
        verified, manifest = eu.verified, eu.manifest
    assert_raises(ProviderError, comparison_key_base, real_env["paths"], tu, NoPolicy())


class _LegacyProvider:
    """An INT-01-era provider: same protocol surface, no history_policy anywhere (deliberately wrong)."""
    manifest_id, split, fingerprint = "M05_CALIB_TRAIN", "TRAIN", "0" * 64

    def describe(self):
        return {"provider": "v4int.StoreTrainProvider/1", "build": {}}


@nc("a provider without history_policy must fail the policy-binding check")
def test_nc_provider_without_history_policy():
    policy_bound(_LegacyProvider())
