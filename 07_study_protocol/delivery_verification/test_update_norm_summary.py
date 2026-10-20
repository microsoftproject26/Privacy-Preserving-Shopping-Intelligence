"""Scientific checks: norms, missing values, malformed logs and the tau bound."""
import json
import math

import pytest

from summarize_update_norms import coordinate_lower_bound, derive, quantile, read_log


def group(l2=5.0, n=25):
    return dict(numel=n, agg_delta_l2=l2, step_l2=l2 / 2, pre_dist_l2=l2 / 3,
                pre_l2=10.0, rel_pre_dist=l2 / 30)


def record(r=0):
    return dict(round=r, server_step=True, finite=True, groups={'encoder': group()}, total_T=group(), cos_prev=None)


def write(tmp_path, rows):
    p = tmp_path / 'log.jsonl'
    p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return p


def test_rms_and_quantile_are_not_group_l2():
    d = derive(group(), 2.0)
    assert d['agg_delta_rms'] == 1.0
    assert quantile([0.0, 10.0], .99) == 9.9
    assert quantile([], .5) is None


def test_bound_is_valid_for_dense_sparse_and_threshold_coordinates():
    for coords in [[0, 0, 0, 0], [0, 0, 0, 3], [1, 1, 1, 1], [.1, .1, .1, .1], [-2, .5, 0, 2]]:
        for tau in [.1, 1, 2]:
            bound = coordinate_lower_bound(math.sqrt(sum(x*x for x in coords)), len(coords), tau)
            actual = sum(abs(x) < tau for x in coords) / len(coords)
            assert 0 <= bound <= actual + 1e-15
    # Same L2, different exact coordinate fractions: logs cannot identify that fraction.
    assert sum(x < 1 for x in [0, 0, 0, 2]) != sum(x < 1 for x in [1, 1, 1, 1])


def test_zero_updates_and_missing_values():
    d = derive(group(0), .001)
    assert d['fraction_coordinates_below_tau_lower_bound'] == 1.0
    g = group()
    g['agg_delta_l2'] = None
    d = derive(g, .001)
    assert d['agg_delta_rms'] is None and d['fraction_coordinates_below_tau_lower_bound'] is None


def test_read_log_accepts_noop_round_without_norms(tmp_path):
    p = write(tmp_path, [record(), dict(round=1, server_step=False), record(2)])
    assert [bool(g) for _, g in read_log(p, .001)] == [True, False, True]


@pytest.mark.parametrize('rounds', [[0, 0], [0, 2], [1]])
def test_duplicate_missing_or_nonzero_first_round_is_refused(tmp_path, rounds):
    with pytest.raises(ValueError, match='rounds must'):
        list(read_log(write(tmp_path, [record(r) for r in rounds]), .001))


def test_inconsistent_total_is_refused(tmp_path):
    r = record()
    r['total_T']['step_l2'] = 20.0
    with pytest.raises(ValueError, match='does not equal'):
        list(read_log(write(tmp_path, [r]), .001))


def test_changing_tensor_dimensions_is_refused(tmp_path):
    r = record(1)
    r['groups']['encoder']['numel'] = r['total_T']['numel'] = 26
    with pytest.raises(ValueError, match='dimensions changed'):
        list(read_log(write(tmp_path, [record(), r]), .001))


@pytest.mark.parametrize('x', [float('nan'), float('inf'), -1])
def test_invalid_numeric_value_is_refused(x):
    with pytest.raises(ValueError, match='invalid'):
        derive(group(x), .001)
