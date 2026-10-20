"""One-run canary auditing statistics; this module does not train a model.

Implements Theorem 3.1, equations (2)-(3), Steinke/Nasr/Jagielski (2023):
https://papers.nips.cc/paper_files/paper/2023/file/9a6f6e0d6781d1cb8689192408946d73-Paper-Conference.pdf
The result is an empirical LOWER bound, never an upper privacy certificate.
Inclusion bits must be independent fair coins; guesses cannot inspect those bits.
Guess counts and any multiple-testing correction are fixed before scores exist.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import expit
from scipy.stats import binom

REFERENCE = 'Steinke, Nasr & Jagielski 2023, Theorem 3.1, equations (2)-(3)'


def integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(label + ' must be an integer')
    return int(value)


def audit_tail_bound(epsilon, delta, n_canaries, n_guesses, n_correct):
    """Upper bound on the null tail probability for one fixed guessing rule."""
    m = integer(n_canaries, 'n_canaries')
    r = integer(n_guesses, 'n_guesses')
    v = integer(n_correct, 'n_correct')
    if not 0 <= v <= r <= m or m < 1:
        raise ValueError('require 0 <= correct <= guesses <= canaries, canaries >= 1')
    epsilon, delta = float(epsilon), float(delta)
    if not math.isfinite(epsilon) or epsilon < 0 or not math.isfinite(delta) or not 0 <= delta < 1:
        raise ValueError('epsilon must be finite nonnegative and 0 <= delta < 1')
    if v == 0:
        return 1.0
    p = float(expit(epsilon))
    tail = float(binom.sf(v - 1, r, p))
    if not delta:
        return tail
    # P(v > W >= v-i). Counts below zero are impossible. For i >= v,
    # the window is constant while 2/i decreases, so i > v cannot maximize.
    widths = np.arange(1, min(m, v) + 1, dtype=np.int64)
    upper_cdf = float(binom.cdf(v - 1, r, p))
    lower_cdf = binom.cdf(v - widths - 1, r, p)
    # Use survival differences on the high-CDF side to avoid subtracting two
    # values close to one; CDF differences are stable on the other side.
    windows = upper_cdf - lower_cdf
    high = lower_cdf > 0.5
    windows[high] = binom.sf(v - widths[high] - 1, r, p) - tail
    alpha = float(np.max(2.0 * np.clip(windows, 0.0, 1.0) / widths))
    return min(1.0, max(0.0, tail + delta * m * alpha))


def empirical_epsilon_lower_bound(n_canaries, n_guesses, n_correct, *, delta=5e-6,
                                  confidence=0.95, n_predeclared_tests=1):
    """Invert fixed-rule null tests with Bonferroni coverage if several are used."""
    tests = integer(n_predeclared_tests, 'n_predeclared_tests')
    if tests < 1 or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError('positive test count and confidence in (0,1) required')
    threshold = (1.0 - confidence) / tests
    def bound(eps):
        return audit_tail_bound(eps, delta, n_canaries, n_guesses, n_correct)
    null_at_zero = bound(0.0)
    if null_at_zero >= threshold:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        while bound(hi) < threshold:
            hi *= 2
            if hi >= 64:
                # Logistic p rounds to one in float64; every valid tail is one.
                hi = 64.0
                break
        for _ in range(80):
            middle = (lo + hi) / 2
            if bound(middle) < threshold:
                lo = middle
            else:
                hi = middle
        lower = lo  # conservative rejected side of the numerical bracket
    return dict(empirical_epsilon_lower_bound=lower, delta=float(delta),
                confidence=float(confidence), per_test_significance=threshold,
                n_predeclared_tests=tests, null_tail_at_epsilon0=null_at_zero,
                bound_type='LOWER_BOUND_ONLY', privacy_certified=False,
                reference=REFERENCE)


def fixed_tail_guesses(scores, k_positive, k_negative):
    """Guess largest/smallest fixed numbers, deterministic ID ties; no labels."""
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('scores must be a finite vector')
    kp, kn = integer(k_positive, 'k_positive'), integer(k_negative, 'k_negative')
    if kp < 0 or kn < 0 or kp + kn > len(values):
        raise ValueError('fixed tail counts must be nonnegative and disjoint')
    ids = np.arange(len(values))
    order = np.lexsort((ids, values))
    guesses = np.zeros(len(values), dtype=np.int8)
    guesses[order[:kn]] = -1
    if kp:
        guesses[order[-kp:]] = 1
    return guesses


def correct_guesses(guesses, inclusion_bits):
    g, s = np.asarray(guesses), np.asarray(inclusion_bits)
    if g.ndim != 1 or g.shape != s.shape or not np.isin(g, [-1, 0, 1]).all() or not np.isin(s, [-1, 1]).all():
        raise ValueError('aligned ternary guesses and binary signs required')
    return int(np.count_nonzero((g != 0) & (g == s)))
