"""The optional provider `transform_state` (DECISION_LOG 42; v4int.renorm): NORMALIZERS ONLY.

* The default equals an explicitly passed Clean V2 state, bitwise (every TRAIN and evaluation batch tested, same
  fingerprints, same comparison key).
* A synthetic state matches the direct formula on the RAW stored values (clip -> z-score -> rezero, float32) for all
  17 channels, within the float32 precision of the served z; everything else in the batch (tokens, flags incl.
  item_age_unavailable, masks, non-normalized channels, targets) is bitwise the default.
* A bad state is refused (self-hash, schema, missing channel, std <= 0, non-strict JSON); a train / eval pair with
  different states is refused by the comparison key; a state that widens a clip range is disclosed.
* Negative control: a provider that ignores the state fails the direct-formula check.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from histkit import REAL_DS, Q, assert_raises, batches_equal, nc, raw_rows
from v4data.common import VALIDATION_SPLITS, self_hash
from v4data.transform import build_spec
from v4eval.predict import stream_rows
from v4hist.windows import window_rows
from v4int.providers import ProviderError, StoreEvalProvider, StoreTrainProvider, comparison_key_base
from v4int.renorm import TransformStateError, load_transform_state

B = 256
USER = "USER_HISTORY_L50"
NON_TEST = ("TRAIN",) + tuple(VALIDATION_SPLITS)
TOL = 1e-5            # absolute, in z units: the float32 precision of the served z carried through the inverse


def _clean_state_path():
    from intkit import transform_state_source
    return transform_state_source()


def _write_state(tmp_path, name, mutate):
    st = json.loads(_clean_state_path().read_text(encoding="utf-8"))
    mutate(st)
    st.pop("transform_state_sha256", None)
    st["transform_state_sha256"] = self_hash(st, "transform_state_sha256")
    p = tmp_path / name
    p.write_text(json.dumps(st, indent=1), encoding="utf-8")
    return p


def _synthetic(st):
    """Every channel changes: clip range shrunk 10% inward on each side (inside the Clean V2 range, so the inverse is
    exact), mean shifted by +0.25 std, std scaled by 1.3."""
    for e in st["normalized_channels_v2"].values():
        lo, hi, mu, sd = e["clip_lo"], e["clip_hi"], e["mean"], e["std"]
        w = hi - lo
        e.update(clip_lo=lo + 0.1 * w, clip_hi=hi - 0.1 * w, mean=mu + 0.25 * sd, std=sd * 1.3)
    st["fit"] = {"population": "SYNTHETIC (DATA-05 test)"}


@pytest.fixture(scope="module")
def states(tmp_path_factory):
    d = tmp_path_factory.mktemp("states")
    return {"clean": _clean_state_path(), "syn": _write_state(d, "SYN.json", _synthetic), "dir": d}


@pytest.fixture(scope="module")
def prov_clean(real_env, states):
    kw = dict(history_policy=USER, paths=real_env["paths"], ds_root=REAL_DS, buckets=[0], log=Q,
              transform_state=states["clean"])
    return (StoreTrainProvider("M05_CALIB_TRAIN", **kw), StoreEvalProvider("M06_CALIB_VALIDATION_WARM", **kw))


@pytest.fixture(scope="module")
def prov_syn(real_env, states):
    kw = dict(history_policy=USER, paths=real_env["paths"], ds_root=REAL_DS, buckets=[0], log=Q,
              transform_state=states["syn"])
    return (StoreTrainProvider("M05_CALIB_TRAIN", **kw), StoreEvalProvider("M06_CALIB_VALIDATION_WARM", **kw))


def _raw(splits):
    spec = build_spec()
    cols = sorted(set(spec.norm_v2) | {"price_missing", "price_conflict", "prev_gap_unavailable",
                                       "item_age_unavailable", "recent_price_supported", "prev_user_event_available",
                                       "prev_session_available"})
    return raw_rows(0, splits, cols)


@pytest.fixture(scope="module")
def raw_train(hist_train):
    r = _raw(("TRAIN",))
    assert np.array_equal(r["decision_id"].astype(np.int64), hist_train.index.decision_id)
    return r


@pytest.fixture(scope="module")
def raw_full(hist_val):
    r = _raw(NON_TEST)
    assert np.array_equal(r["decision_id"].astype(np.int64), hist_val.index.decision_id)
    return r


def direct_formula(raw, idx, valid, rows, norm, spec):
    """The formula of record on RAW values (independent of v4int.renorm): clip -> (x - mean) / std -> float32 ->
    rezero by the frozen gates; padded positions 0. Returns (numeric [B, L, 15] on the normalized columns, uctx)."""
    v = np.asarray(valid, dtype=bool)
    ic = np.where(v, idx, 0)
    z = lambda x, p: ((np.clip(np.asarray(x, dtype=np.float64), p[0], p[1]) - p[2]) / p[3]).astype(np.float32)  # noqa
    pm = (raw["price_missing"].astype(bool) | raw["price_conflict"].astype(bool))
    ev_gate = {"log1p_price": pm, "log1p_prev_raw_view_gap_s": raw["prev_gap_unavailable"].astype(bool),
               "log1p_item_observed_age_s": raw["item_age_unavailable"].astype(bool)}
    num = {}
    for j, name in enumerate(spec.v2_numeric):
        if name in norm:
            g = ~v | (ev_gate[name][ic] if name in ev_gate else False)
            num[j] = np.where(g, np.float32(0.0), z(raw[name][ic], norm[name]))
    nsp = raw["recent_price_supported"][rows] == 0
    u_gate = {"recent_mean_log_price": nsp, "current_minus_recent_mean_log_price": nsp | pm[rows],
              "log1p_time_since_prev_user_event_s": raw["prev_user_event_available"][rows] == 0,
              "log1p_time_since_prev_session_last_event_s": raw["prev_session_available"][rows] == 0}
    uc = {}
    for j, name in enumerate(spec.v2_uctx):
        if name in norm:
            zz = z(raw[name][rows], norm[name])
            uc[j] = np.where(u_gate[name], np.float32(0.0), zz) if name in u_gate else zz
    return num, uc


def _max_diff(batch, num, uc):
    n = batch["event_numeric_features"].numpy()
    u = batch["user_context"].numpy()
    d = [float(np.max(np.abs(n[..., j].astype(np.float64) - a))) for j, a in num.items()]
    d += [float(np.max(np.abs(u[:, j].astype(np.float64) - a))) for j, a in uc.items()]
    zeros_ok = all(np.array_equal(n[..., j] == 0, a == 0) for j, a in num.items())
    return max(d), zeros_ok


def _check_train(tp, hs, raw, norm, positions):
    spec = build_spec()
    ds = hs.decision_set("loss")
    worst = 0.0
    for pos in positions:
        b = tp.batch(pos)
        p = hs.positions_for(ds, tp.decision_ids()[pos])
        idx, valid = window_rows(ds.end_row[p], ds.length[p])
        num, uc = direct_formula(raw, idx, valid, ds.end_row[p], norm, spec)
        m, zeros_ok = _max_diff(b, num, uc)
        assert zeros_ok, "the re-zeroed / padded positions differ from the formula"
        assert m <= TOL, f"served values differ from the state's direct formula by {m}"
        worst = max(worst, m)
    return worst


def _positions(n, k=12, seed=3):
    rng = np.random.default_rng(seed)
    return [rng.choice(n, B, replace=False) for _ in range(k)]


# ------------------------------------------------------------------------------------------------ default
def test_default_equals_explicit_clean_v2_state_bitwise(providers_user, prov_clean, real_env, real_env_registry, states):
    from conftest import registered_providers
    tu, eu = providers_user
    tc, ec = prov_clean
    assert tu.renorm is None and tc.renorm is not None and tc.renorm.identity
    assert tc.fingerprint == tu.fingerprint and ec.fingerprint == eu.fingerprint
    # R4 (INT-02R1 OPS-10): comparison_key_base needs registry-verified providers; tc/tu themselves are deliberately
    # NOT registry-bound (their .fingerprint is compared above against independently rebuilt providers elsewhere in
    # this suite), so a SEPARATE, registry-bound pair over the exact same manifest / policy / transform state proves
    # the same "default == explicit clean state" property at the comparison-key level.
    rc = registered_providers(real_env, real_env_registry, USER, transform_state=states["clean"])
    ru = registered_providers(real_env, real_env_registry, USER)
    assert comparison_key_base(real_env["paths"], *rc) == comparison_key_base(real_env["paths"], *ru)
    for pos in _positions(tu.n_decisions, 40):
        ok, why = batches_equal(tc.batch(pos), tu.batch(pos))
        assert ok, why
    n = 0
    for a, b in zip(ec.batches(ec.eval_batch), eu.batches(eu.eval_batch)):
        ok, why = batches_equal(a, b)
        assert ok, why
        n += int(a["decision_id"].shape[0])
    assert n == eu.n_stream_rows


def test_default_served_values_equal_the_clean_v2_formula_on_raw(providers_user, hist_train, raw_train):
    """Sanity of the raw reference: the default served values ARE the Clean V2 formula on the raw columns."""
    tu, _ = providers_user
    worst = _check_train(tu, hist_train, raw_train, build_spec().norm_v2, _positions(tu.n_decisions, 6))
    assert worst <= TOL
    print(f"DATA05_DEFAULT_VS_RAW_FORMULA_MAX_ABS_DIFF_Z={worst:.3e}")


# ------------------------------------------------------------------------------------------------ synthetic
def test_synthetic_state_matches_the_direct_formula(prov_syn, states, hist_train, hist_val, raw_train, raw_full):
    ts, es = prov_syn
    norm = load_transform_state(states["syn"], build_spec())["norm"]
    assert not ts.renorm.identity and ts.renorm.clip_extends == []
    worst = _check_train(ts, hist_train, raw_train, norm, _positions(ts.n_decisions, 12))
    spec = build_spec()
    ds = hist_val.decision_set("eval")
    ids = es.manifest.decision_id[stream_rows(es.manifest)]
    for k, b in enumerate(es.batches(es.eval_batch)):
        p = hist_val.positions_for(ds, ids[k * es.eval_batch:(k + 1) * es.eval_batch])
        idx, valid = window_rows(ds.end_row[p], ds.length[p])
        num, uc = direct_formula(raw_full, idx, valid, ds.end_row[p], norm, spec)
        m, zeros_ok = _max_diff(b, num, uc)
        assert zeros_ok and m <= TOL, (k, m)
        worst = max(worst, m)
    print(f"DATA05_SYNTHETIC_STATE_MAX_ABS_DIFF_Z={worst:.3e}")


def test_only_the_17_channels_change(prov_syn, providers_user):
    ts, _ = prov_syn
    tu, _ = providers_user
    spec = build_spec()
    jn = [j for j, n in enumerate(spec.v2_numeric) if n in spec.norm_v2]
    ju = [j for j, n in enumerate(spec.v2_uctx) if n in spec.norm_v2]
    for pos in _positions(tu.n_decisions, 8, seed=9):
        a, b = ts.batch(pos), tu.batch(pos)
        assert set(a) == set(b)
        for k in a:
            if k in ("event_numeric_features", "user_context", "clean_v2_state_sha256"):
                continue
            if hasattr(a[k], "numpy"):
                assert np.array_equal(a[k].numpy(), b[k].numpy()), k
            else:
                assert a[k] == b[k], k
        keep_n = [j for j in range(15) if j not in jn]
        keep_u = [j for j in range(12) if j not in ju]
        assert np.array_equal(a["event_numeric_features"].numpy()[..., keep_n],
                              b["event_numeric_features"].numpy()[..., keep_n])
        assert np.array_equal(a["user_context"].numpy()[:, keep_u], b["user_context"].numpy()[:, keep_u])
        assert not np.array_equal(a["event_numeric_features"].numpy()[..., jn],
                                  b["event_numeric_features"].numpy()[..., jn])
        assert a["clean_v2_state_sha256"] == ts.transform_state["transform_state_sha256"]


def test_state_recorded_and_bound(prov_syn, providers_user, real_env, real_env_registry, states):
    from conftest import registered_providers
    ts, es = prov_syn
    tu, eu = providers_user
    rec = ts.describe()["transform_state"]
    assert rec["transform_state_sha256"] == json.loads(states["syn"].read_text())["transform_state_sha256"]
    assert ts.build_record["transform_state"] == rec and es.describe()["transform_state"]["transform_state_sha256"] \
        == rec["transform_state_sha256"]
    assert ts.fingerprint != tu.fingerprint and es.fingerprint != eu.fingerprint
    # R4 (INT-02R1 OPS-10): a registry-bound pair per state, over the exact same manifest / policy, for
    # comparison_key_base (see test_default_equals_explicit_clean_v2_state_bitwise for why tu/ts are not reused here).
    rs = registered_providers(real_env, real_env_registry, USER, transform_state=states["syn"])
    ru = registered_providers(real_env, real_env_registry, USER)
    ks, ku = comparison_key_base(real_env["paths"], *rs), comparison_key_base(real_env["paths"], *ru)
    assert ks["transform_state_sha256"] == rec["transform_state_sha256"] != ku["transform_state_sha256"]
    assert ks["feature_contract_hash"] != ku["feature_contract_hash"]
    assert_raises(ProviderError, comparison_key_base, real_env["paths"], rs[0], ru[1])      # mixed states refused
    assert_raises(ProviderError, comparison_key_base, real_env["paths"], ru[0], rs[1])


def test_bad_states_are_refused(states, real_env):
    spec = build_spec()
    d = states["dir"]
    bad_hash = d / "BAD_HASH.json"
    st = json.loads(states["syn"].read_text())
    st["normalized_channels_v2"]["log1p_price"]["mean"] += 1.0            # edited without re-hashing
    bad_hash.write_text(json.dumps(st), encoding="utf-8")
    cases = [bad_hash,
             _write_state(d, "MISSING.json", lambda s: s["normalized_channels_v2"].pop("log1p_price")),
             _write_state(d, "STD0.json", lambda s: s["normalized_channels_v2"]["recent_history_length"]
                          .update(std=0.0)),
             _write_state(d, "SCHEMA.json", lambda s: s.update(schema_self_sha256="0" * 64)),
             _write_state(d, "SWAP.json", lambda s: s["normalized_channels_v2"]["log1p_price"]
                          .update(clip_lo=9.0, clip_hi=1.0))]
    nan = d / "NAN.json"
    nan.write_text(states["syn"].read_text().replace('"status"', '"nan_field": NaN, "status"', 1), encoding="utf-8")
    cases.append(nan)
    for p in cases + [d / "does_not_exist.json"]:
        assert_raises(TransformStateError, load_transform_state, p, spec)
    assert_raises(TransformStateError, StoreTrainProvider, "M05_CALIB_TRAIN", history_policy=USER,
                  paths=real_env["paths"], ds_root=REAL_DS, buckets=[0], log=Q, transform_state=bad_hash)


def test_a_widened_clip_range_is_disclosed(states):
    def widen(s):
        e = s["normalized_channels_v2"]["log1p_price"]
        e.update(clip_hi=e["clip_hi"] + 1.0)
    p = _write_state(states["dir"], "WIDE.json", widen)
    from v4int.renorm import Renormalizer
    r = Renormalizer(p, build_spec())
    assert r.clip_extends == ["log1p_price"] and r.record()["clip_extends_beyond_clean_v2"] == ["log1p_price"]


@nc("a provider that ignores its transform state (serves the Clean V2 normalization) fails the direct formula")
def test_nc_ignoring_the_state_is_caught(prov_syn, states, hist_train, raw_train):
    ts, _ = prov_syn
    bad = copy.copy(ts)
    bad.renorm = None                                                      # the deliberately wrong provider
    norm = load_transform_state(states["syn"], build_spec())["norm"]
    _check_train(bad, hist_train, raw_train, norm, _positions(ts.n_decisions, 2))
