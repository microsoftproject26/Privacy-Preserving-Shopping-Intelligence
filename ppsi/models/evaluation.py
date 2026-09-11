"""The T1 evaluator - one implementation, shared by every lane that scores T1.

`S2-PR-07` owns the official T1 evaluation harness and `S2-PR-08` the official baselines.
This module exists so that lane imports an evaluator rather than writing a second one.
Two independent implementations would disagree by a small amount that nobody could
explain, and "one rule written in two places" has already produced three wrong numbers in
this project.

The baselines are already measured, on the frozen VALIDATION examples:

    trivial - repeat the current category      0.8232   MRR@20, overall
    globally most popular category             0.2841
    TRAIN transition table                     0.8560
    transition table, category-change slice    0.3085   micro     0.3144  macro

`ADR-001` makes **macro the headline** - one vote per client - with micro printed beside
it, and the comparison metric the **category-change slice**, because the overall number is
82.32% free: a rule that answers "the same category again" is right that often without
learning anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOP_K = 20
STRATA_EDGES = [10, 20, 50, 100, np.inf]
STRATA_LABELS = ["10-19", "20-49", "50-99", "100+"]


def rank_of_truth(
    scores: np.ndarray,
    targets: np.ndarray,
    *,
    k: int = TOP_K,
    suppress: np.ndarray | None = None,
    legacy_clamp: bool = False,
) -> np.ndarray:
    """1-based rank of the true class within the top `k`, or 0 when it is absent.

    `suppress` removes one class per row before ranking. It exists to match how the
    upstream baseline was built: on the category-change slice the transition table is
    evaluated off-diagonal, so it is handed the fact that the answer is not the current
    category. Leaving that advantage one-sided is not a comparison - the model would burn
    rank 1 on an answer the slice definition already excludes. It is worth about
    **+0.13 MRR**, so getting it wrong does not look like a bug, it looks like a result.

    `legacy_clamp` reproduces a defect in S2-SMOKE. There, `suppress` is the current
    category, which is `-1` for the 138 VALIDATION decisions whose category is unseen in
    TRAIN, and `clamp(min=0)` turned those into class 0 - actively removing a real class
    instead of suppressing nothing. It is kept only so the calibration run can reproduce
    0.3348 exactly; every real run leaves it off and the difference is reported.

    Ties are broken by lowest class index, and the rank is **counted rather than sorted**:
    how many classes score strictly higher, plus how many tie with the truth at a lower
    index. Sorting would leave tie order up to the kernel, and the transition-table
    baseline is mostly ties - the great majority of category pairs were never observed and
    all score zero - so an unspecified tie rule would move the very number the model is
    compared against.
    """
    scores = np.array(scores, dtype="float64", copy=True)
    if suppress is not None:
        suppress = np.asarray(suppress)
        rows = np.arange(len(scores))
        if legacy_clamp:
            scores[rows, np.clip(suppress, 0, None)] = -np.inf
        else:
            valid = suppress >= 0
            scores[rows[valid], suppress[valid]] = -np.inf

    truth_score = np.take_along_axis(scores, targets[:, None], axis=1)
    higher = (scores > truth_score).sum(axis=1)
    index = np.arange(scores.shape[1])[None, :]
    earlier_tie = ((scores == truth_score) & (index < targets[:, None])).sum(axis=1)
    rank = higher + earlier_tie + 1
    return np.where(rank <= k, rank, 0)


def reciprocal_rank(ranks: np.ndarray) -> np.ndarray:
    """1/rank for a hit inside the top k, 0 for a miss."""
    return np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)


def hit_rates(
    ranks: np.ndarray, mask: np.ndarray | None = None, at: tuple = (1, 3, 5, 10, 20)
) -> dict:
    """How often the right answer lands inside the first k - what a user actually sees.

    MRR is the protocol metric and the right one for comparing regimes, because it is
    sensitive to *where* in the list the answer sits. But nobody experiences an MRR. A
    person sees a shelf of five things and either what they wanted is on it or it is not,
    so `HR@5` is the number that describes the product rather than the measurement.

    Reported alongside MRR, never instead of it: hit rate is blind to the difference
    between rank 1 and rank 5, which is exactly the difference a better model buys.
    """
    if mask is not None:
        ranks = ranks[mask]
    return {f"HR@{k}": round(float(((ranks > 0) & (ranks <= k)).mean()), 4) for k in at}


def micro_and_macro(
    values: np.ndarray, clients: np.ndarray, mask: np.ndarray | None = None
) -> tuple[float, float]:
    """Micro weights every decision equally; macro weights every client equally.

    The mask is applied **first**, then clients are grouped. A client with no decision on
    the slice does not appear at all, so the macro denominator is the number of clients
    with at least one slice decision - 16,096, not the 31,576 with any decision. Grouping
    before masking silently changes the number.
    """
    if mask is not None:
        values, clients = values[mask], clients[mask]
    frame = pd.DataFrame({"client": clients, "value": values})
    return float(frame["value"].mean()), float(frame.groupby("client")["value"].mean().mean())


def by_client_history(
    values: np.ndarray, clients: np.ndarray, train_events: pd.Series, mask: np.ndarray | None = None
) -> pd.DataFrame:
    """Macro score inside each TRAIN-history bucket.

    Clients are bucketed by how many TRAIN events they have, because that is what a device
    would have to learn from. If an advantage exists only for clients with long histories,
    the device-local regime has nothing to offer the rest - which is the question `R4`
    exists to answer.
    """
    if mask is not None:
        values, clients = values[mask], clients[mask]
    frame = pd.DataFrame({"client": clients, "value": values})
    frame["bucket"] = pd.cut(
        frame["client"].map(train_events).fillna(0),
        bins=STRATA_EDGES,
        labels=STRATA_LABELS,
        right=False,
    )
    per_client = frame.groupby(["bucket", "client"], observed=True)["value"].mean()
    counts = frame.groupby("bucket", observed=True)["client"].nunique()
    return pd.DataFrame(
        {
            "clients": counts,
            "decisions": frame.groupby("bucket", observed=True).size(),
            "macro": per_client.groupby("bucket", observed=True).mean(),
        }
    ).reindex(STRATA_LABELS)


# --- diagnostics: is it a recommender, or a popularity table with extra steps? ------


def category_coverage(scores: np.ndarray, *, top: int = 5) -> dict:
    """How much of the vocabulary the model is willing to name.

    A model that only ever proposes the handful of most common categories scores well -
    those categories genuinely are common - and recommends nothing. The headline metric
    cannot see this, which is why `S2-PR-07` lists coverage among its required
    diagnostics.
    """
    proposed = np.argpartition(-scores, kth=top - 1, axis=1)[:, :top]
    distinct, counts = np.unique(proposed, return_counts=True)
    share = counts / counts.sum()
    return {
        "distinct_categories_in_top": int(distinct.size),
        "vocabulary_size": int(scores.shape[1]),
        "coverage": round(float(distinct.size / scores.shape[1]), 4),
        # Gini: 0 means every category is proposed equally often, 1 means one takes all.
        "concentration_gini": round(float(_gini(np.sort(share))), 4),
    }


def coverage_of_top5(top5: np.ndarray) -> dict:
    """Coverage from already-extracted top-5 indices.

    The full score matrix for a validation split is 438,185 x 588 floats - a gigabyte held
    only to be reduced. Scoring accumulates the top-5 per batch instead, and this reads
    that.
    """
    distinct, counts = np.unique(top5, return_counts=True)
    share = np.sort(counts / counts.sum())
    return {
        "distinct_categories_in_top5": int(distinct.size),
        "coverage": round(float(distinct.size / 588), 4),
        "concentration_gini": round(_gini(share), 4),
    }


def divergence_of_top1(top1: np.ndarray, clients: np.ndarray) -> dict:
    """Per-client divergence from already-extracted top-1 predictions."""
    frame = pd.DataFrame({"client": clients, "top1": top1})
    per_client = frame.groupby("client")["top1"].nunique()
    _, counts = np.unique(top1, return_counts=True)
    return {
        "distinct_top1_overall": len(counts),
        "most_common_top1_share": round(float(counts.max() / counts.sum()), 4),
        "mean_distinct_top1_per_client": round(float(per_client.mean()), 3),
    }


def per_client_divergence(scores: np.ndarray, clients: np.ndarray) -> dict:
    """Do different clients get different answers, or does everyone get the same list?

    This is the diagnostic that matters most to this project. `R3` personalizes and `R4`
    trains on the device; both assume a client's own history changes what they are shown.
    If the top-1 prediction barely varies across clients, neither regime has anything to
    personalize and the strongest claim in the proposal quietly loses its basis.
    """
    top1 = scores.argmax(axis=1)
    frame = pd.DataFrame({"client": clients, "top1": top1})
    per_client = frame.groupby("client")["top1"].nunique()
    _, counts = np.unique(top1, return_counts=True)
    return {
        "distinct_top1_overall": len(counts),
        "most_common_top1_share": round(float(counts.max() / counts.sum()), 4),
        "mean_distinct_top1_per_client": round(float(per_client.mean()), 3),
    }


def popularity_correlation(scores: np.ndarray, popularity: np.ndarray) -> float:
    """Correlation between the model's mean score per class and global popularity.

    Near 1.0 means we built an expensive popularity baseline.
    """
    mean_score = scores.mean(axis=0)
    return round(float(np.corrcoef(mean_score, popularity)[0, 1]), 4)


def _gini(sorted_shares: np.ndarray) -> float:
    n = sorted_shares.size
    if n == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * index - n - 1).dot(sorted_shares) / (n * sorted_shares.sum()))


# --- the model-free baseline -------------------------------------------------------


def transition_table(current: np.ndarray, following: np.ndarray, categories: int) -> np.ndarray:
    """Counts of "category A is followed by category B", over decision rows only.

    Over **decision rows**, never all events. Fed every row instead, the `-1` that marks
    "no later different item" competes as if it were a category - and because it is what
    every session ends on, it takes rank 1 away from real answers. That drops the baseline
    from 0.3085 to 0.1866 and inflates the model's apparent gain to +0.1454. A bug that
    flatters the result is the most dangerous kind, because nothing about it looks wrong.
    """
    assert current.min() >= 0 and following.min() >= 0, (
        "the transition table must be built from decision rows only; a negative code "
        "means unfiltered rows were passed in"
    )
    table = np.zeros((categories, categories), dtype="float64")
    np.add.at(table, (current, following), 1.0)
    return table


def paired_client_bootstrap(per_client_model: np.ndarray, per_client_baseline: np.ndarray,
                            *, resamples: int = 2000, seed: int = 13) -> dict:
    """A confidence interval for the **gain**, resampling clients rather than seeds.

    The noise floors this project has been quoting - 0.0033 on T1, 0.0029 on T2, 0.0048 on
    T3 - are seed spreads: the max-minus-min of three runs. That number answers *"how much
    does this move if I re-run it?"*, which is a real question but not the one a reader of
    a headline asks. They want *"would this gain survive a different sample of shoppers?"*,
    and three seeds on one fixed set of clients cannot answer it: every run saw exactly the
    same people.

    Resampling clients answers it directly, and pairing matters as much as resampling.
    Taking two independent intervals - one for the model, one for the baseline - and
    checking whether they overlap is a weaker and wronger test, because the two are
    measured on the *same* clients and move together. A client whose session is hard drags
    both scores down at once. Differencing inside each resample cancels that shared
    difficulty and leaves only what the model actually changed, which is why a paired
    interval is usually far tighter than the two separate ones suggest.

    Both arrays are one value per client, in the same client order.
    """
    import numpy as np

    assert per_client_model.shape == per_client_baseline.shape, (
        f"{per_client_model.shape} against {per_client_baseline.shape}; a paired test needs "
        "the same clients in the same order on both sides")

    difference = per_client_model - per_client_baseline
    clients = len(difference)
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, clients, size=(resamples, clients))
    means = difference[draws].mean(axis=1)

    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "clients": int(clients),
        "gain": round(float(difference.mean()), 6),
        "ci_low": round(float(low), 6),
        "ci_high": round(float(high), 6),
        "half_width": round(float((high - low) / 2), 6),
        "resamples": int(resamples),
        # The only claim that matters: does the interval clear zero entirely?
        "above_zero": bool(low > 0),
        "method": "paired client bootstrap, 2.5/97.5 percentiles of the mean difference",
    }
