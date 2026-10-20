"""BASE-02: strata — a hand-computed, user-de-duplicated MAIN-TRAIN scan (n_train, category diversity, brand
loyalty, including the zero-known-evidence edge case and the "keep only the longest window per user" dedup), cut
points checked against an INDEPENDENT `numpy.percentile` call (not a re-invocation of `_percentile_cuts`), the
O_EXCL write, the self-hash tamper check, bucket assignment, the `client_key_hex` key space (never a raw dense
`client_ord`/`eval_user_key`), and the explicit NO_TRAIN sentinel for a user absent from the registry.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from basekit import nc
from v4base import strata as base_strata

# Each row is one TRAIN decision; "user" is the client_key_hex this test assigns it (passed as `user_keys`, NEVER
# read from the batch — the real batch carries no such column, see v4base.identity). U0 and U1 each have TWO
# decisions of DIFFERENT length: the shorter one must be discarded entirely by the per-user window dedup (the
# longer one is a superset of it, up to the L cap), not merged with it.
ROWS = [
    {"user": "U0", "length": 1, "cats": [5, 0], "brands": [10, 0]},     # U0, shorter -> discarded
    {"user": "U0", "length": 2, "cats": [6, 6], "brands": [11, 11]},    # U0, longer -> kept
    {"user": "U1", "length": 2, "cats": [5, 5], "brands": [10, 10]},    # U1, longer -> kept
    {"user": "U1", "length": 1, "cats": [9, 0], "brands": [12, 0]},     # U1, shorter -> discarded
    {"user": "U2", "length": 2, "cats": [5, 6], "brands": [10, 11]},    # U2, only row
    {"user": "U3", "length": 2, "cats": [1, 1], "brands": [1, 1]},      # U3, only row, ALL unknown (MISSING=1)
]
USER_KEYS = [r["user"] for r in ROWS]


class _Provider:
    n_decisions = len(ROWS)

    def batch(self, positions):
        idx = [int(i) for i in positions]
        return {"item_tokens": torch.full((len(idx), 2), 3, dtype=torch.int64),
               "lengths": torch.tensor([ROWS[i]["length"] for i in idx], dtype=torch.int64),
               "decision_id": torch.tensor(idx, dtype=torch.int64),
               "main_category_tokens": torch.tensor([ROWS[i]["cats"] for i in idx], dtype=torch.int64),
               "brand_tokens": torch.tensor([ROWS[i]["brands"] for i in idx], dtype=torch.int64)}


# expected AFTER dedup: U0/U1 keep only their LONGER row.
EXPECTED_STATS = {
    "U0": {"n_train": 2, "n_distinct_main_categories": 1, "top_brand_share": 1.0},
    "U1": {"n_train": 2, "n_distinct_main_categories": 1, "top_brand_share": 1.0},
    "U2": {"n_train": 1, "n_distinct_main_categories": 2, "top_brand_share": 0.5},
    "U3": {"n_train": 1, "n_distinct_main_categories": 0, "top_brand_share": 0.0},
}


def test_scan_user_train_stats_hand_computed_and_deduped():
    stats = base_strata.scan_user_train_stats(_Provider(), user_keys=USER_KEYS)
    assert stats == EXPECTED_STATS


def test_cutpoints_match_an_independent_numpy_percentile_call():
    reg = base_strata.build_registry(EXPECTED_STATS, source_manifest_id="FIXTURE_MAIN_TRAIN")
    n_train_vals = np.asarray([2.0, 2.0, 1.0, 1.0])
    div_vals = np.asarray([1.0, 1.0, 2.0, 0.0])
    loy_vals = np.asarray([1.0, 1.0, 0.5, 0.0])
    exp_activity = list(np.percentile(n_train_vals, [25, 50, 75], method="linear"))
    exp_div = list(np.percentile(div_vals, [100 / 3, 200 / 3], method="linear"))
    exp_loy = list(np.percentile(loy_vals, [100 / 3, 200 / 3], method="linear"))
    assert reg["cuts"]["activity_quantile_cuts"] == pytest.approx(exp_activity)
    assert reg["cuts"]["category_diversity_tercile_cuts"] == pytest.approx(exp_div)
    assert reg["cuts"]["brand_loyalty_tercile_cuts"] == pytest.approx(exp_loy)
    assert reg["n_users"] == 4
    assert reg["user_key_field"] == "client_key_hex"


def test_write_then_load_round_trips_and_verifies_the_self_hash(tmp_path):
    path = tmp_path / "strata.json"
    receipt = base_strata.write_registry(path, EXPECTED_STATS, source_manifest_id="FIXTURE_MAIN_TRAIN")
    assert receipt["registry_self_sha256"]
    loaded = base_strata.load_registry(path)
    assert loaded["registry_self_sha256"] == receipt["registry_self_sha256"]
    assert loaded["users"]["U3"]["n_distinct_main_categories"] == 0


def test_tampering_the_file_is_refused_by_the_self_hash_check(tmp_path):
    path = tmp_path / "strata.json"
    base_strata.write_registry(path, EXPECTED_STATS)
    text = path.read_text(encoding="utf-8").replace('"n_train": 2', '"n_train": 999')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        base_strata.load_registry(path)


def test_assign_stratum_bucketing_and_no_train_sentinel():
    reg = base_strata.build_registry(EXPECTED_STATS)
    s0 = base_strata.assign_stratum_for_user(reg, "U0")
    s3 = base_strata.assign_stratum_for_user(reg, "U3")
    assert s0["status"] == "BUCKETED"
    assert s0["persona"] == s0["category_diversity_tercile"] * base_strata.LOYALTY_N_BINS + s0["brand_loyalty_tercile"]
    assert s3["category_diversity_tercile"] == 0            # the lowest diversity value (0) is always bin 0
    assert s3["brand_loyalty_tercile"] == 0                 # the lowest loyalty value (0.0) is always bin 0
    # a user with no MAIN-TRAIN row in this registry (e.g. cold/U_NEW): an explicit sentinel, never bucketed.
    absent = base_strata.assign_stratum_for_user(reg, "U_NEVER_SEEN")
    assert absent["status"] == base_strata.NO_TRAIN_STRATUM
    assert absent["activity_quantile"] is None and absent["persona"] is None


def test_registry_is_keyed_by_client_key_hex_not_a_dense_rank():
    """The HIGH review fix: client_ord (TRAIN) and eval_user_key (EVAL) are different per-manifest dense ranks; the
    registry must be looked up by client_key_hex, translated separately on each side."""
    hex_by_train_client_ord = {0: "A" * 64, 1: "B" * 64}
    stats = {hex_by_train_client_ord[0]: {"n_train": 10, "n_distinct_main_categories": 1, "top_brand_share": 1.0},
            hex_by_train_client_ord[1]: {"n_train": 99, "n_distinct_main_categories": 1, "top_brand_share": 1.0}}
    reg = base_strata.build_registry(stats)
    # the EVAL manifest densifies the SAME two real users in the OPPOSITE order: eval_user_key 0 is the person TRAIN
    # called client_ord 1 ("B..."), not client_ord 0 -- the realistic case MANIFEST_SPEC warns about.
    hex_by_eval_user_key = {0: "B" * 64, 1: "A" * 64}
    correct = base_strata.assign_stratum_for_user(reg, hex_by_eval_user_key[0])
    assert correct["status"] == "BUCKETED"
    assert reg["users"][hex_by_eval_user_key[0]]["n_train"] == 99


@nc("joining eval_user_key directly against the registry's dense TRAIN client_ord (ignoring client_key_hex) picks "
   "the WRONG user whenever the two manifests densify their users in a different order")
def test_nc_joining_by_dense_key_picks_the_wrong_user():
    hex_by_train_client_ord = {0: "A" * 64, 1: "B" * 64}
    stats = {hex_by_train_client_ord[0]: {"n_train": 10, "n_distinct_main_categories": 1, "top_brand_share": 1.0},
            hex_by_train_client_ord[1]: {"n_train": 99, "n_distinct_main_categories": 1, "top_brand_share": 1.0}}
    reg = base_strata.build_registry(stats)
    hex_by_eval_user_key = {0: "B" * 64, 1: "A" * 64}
    correct_n_train = reg["users"][hex_by_eval_user_key[0]]["n_train"]                  # correct: 99
    buggy_n_train = reg["users"][hex_by_train_client_ord[0]]["n_train"]                 # buggy dense-key join: 10
    assert buggy_n_train == correct_n_train


@nc("without O_EXCL, a second write_registry call would silently overwrite the frozen cut points")
def test_nc_write_registry_would_overwrite_without_o_excl(tmp_path):
    path = tmp_path / "strata.json"
    base_strata.write_registry(path, EXPECTED_STATS)
    other_stats = {"U0": {"n_train": 1, "n_distinct_main_categories": 1, "top_brand_share": 1.0}}
    overwritten = False
    try:
        base_strata.write_registry(path, other_stats)
    except FileExistsError:
        pass
    else:
        overwritten = True
    assert overwritten
