from fractions import Fraction
import math
import numpy as np
import pytest
from scipy.stats import binom
from . import canary_statistics as C


@pytest.mark.parametrize('m,r,v', [(1,1,1), (5,3,2), (10,5,5), (10,2,0), (10,0,0)])
def test_theorem_tail_matches_exhaustive_fair_coin_probabilities(m,r,v):
    masses = [Fraction(math.comb(r,j), 2**r) for j in range(r+1)]
    tail = sum(masses[v:])
    alpha = max(Fraction(2,i)*sum(masses[max(0,v-i):v]) for i in range(1,m+1))
    expected = min(1., float(tail + Fraction(1,100)*m*alpha))
    assert C.audit_tail_bound(0, .01, m,r,v) == pytest.approx(expected, abs=1e-14)


def test_pure_dp_reduces_to_binomial_tail():
    for eps in [0, .3, 2, 8]:
        assert C.audit_tail_bound(eps,0,1000,500,420) == binom.sf(419,500,1/(1+math.exp(-eps)))


def test_empirical_lower_bound_rejects_below_and_not_above_bracket():
    a=C.empirical_epsilon_lower_bound(1000,1000,800)
    e=a['empirical_epsilon_lower_bound'];assert 1 < e < 1.4
    assert C.audit_tail_bound(e-1e-8,5e-6,1000,1000,800) < .05
    assert C.audit_tail_bound(e+1e-8,5e-6,1000,1000,800) > .05
    assert a['privacy_certified'] is False and a['bound_type']=='LOWER_BOUND_ONLY'


def test_chance_no_guesses_and_zero_correct_have_zero_lower_bound():
    for r,v in [(1000,500),(0,0),(1000,0)]:
        assert C.empirical_epsilon_lower_bound(1000,r,v)['empirical_epsilon_lower_bound']==0


def test_delta_and_multiple_testing_make_lower_bound_more_conservative():
    pure=C.empirical_epsilon_lower_bound(1000,500,410,delta=0)['empirical_epsilon_lower_bound']
    approx=C.empirical_epsilon_lower_bound(1000,500,410)['empirical_epsilon_lower_bound']
    multi=C.empirical_epsilon_lower_bound(1000,500,410,n_predeclared_tests=3)['empirical_epsilon_lower_bound']
    assert 0 < multi < approx < pure


def test_guesses_have_fixed_counts_and_ignore_truth_labels():
    scores=np.array([1,4,2,2,0,3.]);g=C.fixed_tail_guesses(scores,2,2)
    np.testing.assert_array_equal(g,[-1,1,0,0,-1,1])
    assert C.correct_guesses(g,[-1,1,1,-1,-1,1])==4
    assert C.correct_guesses(g,[1,-1,-1,1,1,-1])==0
    np.testing.assert_array_equal(C.fixed_tail_guesses(np.ones(5),2,2),[-1,-1,0,1,1])


@pytest.mark.parametrize('args', [(-1,.01,10,5,3),(math.inf,0,10,5,3),(0,-.01,10,5,3),
                                    (0,1,10,5,3),(0,0,10,11,3),(0,0,10,5,6),(0,0,10,5,-1),
                                    (0,0,0,0,0),(0,0,10,2.5,1),(0,0,10,True,0)])
def test_invalid_null_inputs_refused(args):
    with pytest.raises(ValueError):C.audit_tail_bound(*args)


@pytest.mark.parametrize('scores,kp,kn', [([1,math.nan],1,0),([1,2],2,1),([1,2],-1,0)])
def test_invalid_guess_rule_refused(scores,kp,kn):
    with pytest.raises(ValueError):C.fixed_tail_guesses(scores,kp,kn)
