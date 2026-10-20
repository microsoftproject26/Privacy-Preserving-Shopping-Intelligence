"""Package test list items 20 and 22 -- fo_eval.py (owner: Opus). Never modifies the code
under test. Uses the real reused `eval_full.aggregate` (snapshot copy), not a re-derivation.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import conftest_fo  # noqa: F401
import fo_env
import fo_eval as E


def _hand_ds():
    """5 rows: an OOV-target row (unrankable, rank=-1) mixed with normal rankable rows and one
    eval_mask=False row that must be excluded from the population entirely. One user per row so
    macro == micro here, which keeps the hand computation unambiguous."""
    ranks = np.array([-1, 1, 25, -1, 5], dtype=np.int64)
    ds = {
        "target_class": np.array([-1, 7, 7, 3, 3], dtype=np.int64),
        "eval_mask": np.array([True, True, True, False, True], dtype=bool),
        "eval_user_key": np.array([0, 1, 2, 3, 4], dtype=np.int64),
        "client_idx": np.array([3, 3, 3, 3, 3], dtype=np.int64),
        "prior_view_count": np.zeros(5, dtype=np.int64),
        "prefix_position": np.ones(5, dtype=np.int64),
    }
    return ranks, ds


# --------------------------------------------------------------------- (20) OOV zero-credit in denominator
def test_metrics_counts_oov_rank_minus1_rows_in_denominator_hand_computation():
    ranks, ds = _hand_ds()
    m = E.metrics(ranks, ds)
    # population = rows {0,1,2,4} (row 3 excluded by eval_mask=False); 4 distinct users.
    # row0: rank=-1 (OOV target) -> zero credit, but STILL a denominator member.
    # row1: rank=1  -> mrr@20 = 1.0
    # row2: rank=25 -> outside k=20 -> mrr@20 = 0.0
    # row4: rank=5  -> mrr@20 = 0.2
    expected_macro_mrr20 = (0.0 + 1.0 + 0.0 + 0.2) / 4.0
    assert m["users"] == 4
    assert m["n"] == 5
    assert m["macro_mrr@20"] == pytest.approx(expected_macro_mrr20, abs=1e-6)
    assert m["micro_mrr@20"] == pytest.approx(expected_macro_mrr20, abs=1e-6)


def test_metrics_negative_control_excluding_oov_row_gives_a_different_number():
    """Negative control: if the OOV row were (incorrectly) dropped from the denominator instead
    of counted with zero credit, the macro MRR would be visibly different (0.4 vs 0.3). This
    proves the hand computation above is actually sensitive to the OOV-inclusion rule."""
    ranks, ds = _hand_ds()
    m_real = E.metrics(ranks, ds)
    # Build the "OOV row excluded" alternative ds by hand and confirm it disagrees.
    ranks_excl = np.array([1, 25, 5], dtype=np.int64)
    ds_excl = {
        "target_class": np.array([7, 7, 3], dtype=np.int64),
        "eval_mask": np.array([True, True, True], dtype=bool),
        "eval_user_key": np.array([1, 2, 4], dtype=np.int64),
        "client_idx": np.array([3, 3, 3], dtype=np.int64),
        "prior_view_count": np.zeros(3, dtype=np.int64),
        "prefix_position": np.ones(3, dtype=np.int64),
    }
    m_excl = E.metrics(ranks_excl, ds_excl)
    assert abs(m_real["macro_mrr@20"] - m_excl["macro_mrr@20"]) > 1e-3
    assert m_excl["macro_mrr@20"] == pytest.approx((1.0 + 0.0 + 0.2) / 3.0, abs=1e-6)
    assert m_real["macro_mrr@20"] == pytest.approx((1.0 + 0.0 + 0.2) / 4.0, abs=1e-6)


# --------------------------------------------------------------------- (22) write_once
def test_write_once_writes_valid_json(tmp_path):
    p = tmp_path / "out.json"
    E.write_once(p, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(p.read_text()) == {"a": 1, "b": [1, 2, 3]}


def test_write_once_refuses_to_overwrite(tmp_path):
    p = tmp_path / "out.json"
    E.write_once(p, {"a": 1})
    with pytest.raises(FileExistsError):
        E.write_once(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 1}, "the first write must survive the refused second write"


@pytest.mark.parametrize("writer", [E.write_once, fo_env.write_json_once], ids=["fo_eval.write_once", "fo_env.write_json_once"])
def test_write_json_once_nan_leaves_no_file_and_retry_succeeds(tmp_path, writer):
    """Regression test (2026-09-25: converted from defect-documenting test after lead fix).
    write_once() used to open the destination with mode "x" BEFORE validating the object, so a
    NaN-rejected write left a stray empty file behind and permanently blocked any retry at the
    same path with FileExistsError. The lead fixed this in fo_env.write_json_once (serialise via
    json.dumps(allow_nan=False) FIRST, only then open "x"); fo_eval.write_once now delegates to
    it. Covers both entry points."""
    p = tmp_path / "out.json"
    with pytest.raises(ValueError):
        writer(p, {"a": float("nan")})
    assert not p.exists(), "a rejected NaN write must leave no file behind"
    # A caller who catches the ValueError and retries with corrected data must now succeed.
    writer(p, {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}


@pytest.mark.parametrize("writer", [E.write_once, fo_env.write_json_once], ids=["fo_eval.write_once", "fo_env.write_json_once"])
def test_write_json_once_negative_control_overwrite_still_refused(tmp_path, writer):
    """Negative control for the regression above: the NaN-retry fix must not have loosened the
    exclusive-create guarantee -- an overwrite attempt on an EXISTING (valid) file must still
    raise FileExistsError and leave the original content untouched."""
    p = tmp_path / "out.json"
    writer(p, {"a": 1})
    with pytest.raises(FileExistsError):
        writer(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 1}
