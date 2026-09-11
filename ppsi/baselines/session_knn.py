"""Recent-session cosine kNN, projected to the frozen T1 category universe.

TRAIN sessions contain binary item and category sets. Retrieval uses item overlap;
each selected neighbor votes once per category. This category projection is an
explicit project adaptation of session-kNN, not an exact copy of an item scorer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from numbers import Integral

import numpy as np
from scipy import sparse


class SessionKNN:
    """A frozen TRAIN index; inference never adds a query to this index."""

    def __init__(self, *, category_count: int = 588, candidate_limit: int = 2000) -> None:
        for name, value in (
            ("category_count", category_count),
            ("candidate_limit", candidate_limit),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.category_count = int(category_count)
        self.candidate_limit = int(candidate_limit)
        self._fitted = False

    @staticmethod
    def _ids(values: Sequence[int], name: str) -> list[int]:
        if any(
            isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) or x < 0 for x in values
        ):
            raise ValueError(f"{name} must contain nonnegative integer IDs, not floats")
        return sorted({int(x) for x in values})

    def fit(
        self,
        session_ids: Sequence[str],
        end_times_ns: Sequence[int],
        item_sets: Sequence[Sequence[int]],
        category_sets: Sequence[Sequence[int]],
        popularity: Sequence[float],
        *,
        split: str,
    ) -> SessionKNN:
        if split != "TRAIN":
            raise ValueError("the session index accepts TRAIN only")
        n = len(session_ids)
        if n == 0 or not (len(end_times_ns) == len(item_sets) == len(category_sets) == n):
            raise ValueError("nonempty aligned session inputs are required")
        if any(not isinstance(x, str) or not x for x in session_ids) or len(set(session_ids)) != n:
            raise ValueError("session IDs must be unique nonempty strings")
        if any(
            isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in end_times_ns
        ):
            raise ValueError("session end times must be integer UTC nanoseconds")
        pop = np.asarray(popularity, dtype=np.float64)
        if (
            pop.shape != (self.category_count,)
            or not np.isfinite(pop).all()
            or (pop < 0).any()
            or pop.sum() <= 0
        ):
            raise ValueError("popularity must be a nonnegative nonempty full-vocabulary vector")
        pop = pop / pop.sum()
        # Row order is itself the retrieval tie-break: newest session, then stable ID.
        order = sorted(range(n), key=lambda i: (-int(end_times_ns[i]), session_ids[i]))
        self.session_ids_ = tuple(session_ids[i] for i in order)
        self.end_times_ns_ = np.asarray([end_times_ns[i] for i in order], dtype=np.int64)
        items = [self._ids(item_sets[i], "items") for i in order]
        cats = [self._ids(category_sets[i], "categories") for i in order]
        if any(not x for x in items) or any(not x for x in cats):
            raise ValueError("TRAIN sessions must have items and categories")
        if any(c >= self.category_count for row in cats for c in row):
            raise ValueError("a TRAIN category is outside the frozen vocabulary")
        vocab = sorted({x for row in items for x in row})
        self.item_to_column_ = {item: i for i, item in enumerate(vocab)}
        indptr = np.zeros(n + 1, dtype=np.int64)
        indptr[1:] = np.cumsum([len(x) for x in items])
        indices = np.fromiter(
            (self.item_to_column_[x] for row in items for x in row), dtype=np.int32
        )
        self.items_ = sparse.csr_matrix(
            (np.ones(len(indices)), indices, indptr), shape=(n, len(vocab))
        )
        catptr = np.zeros(n + 1, dtype=np.int64)
        catptr[1:] = np.cumsum([len(x) for x in cats])
        catidx = np.fromiter((c for row in cats for c in row), dtype=np.int32)
        self.categories_ = sparse.csr_matrix(
            (np.ones(len(catidx)), catidx, catptr), shape=(n, self.category_count)
        )
        self.item_counts_ = np.asarray([len(x) for x in items], dtype=np.float64)
        postings: dict[int, list[int]] = defaultdict(list)
        for row_id, row in enumerate(items):
            for item in row:
                if len(postings[item]) < self.candidate_limit:
                    postings[item].append(row_id)
        self.postings_ = {item: np.asarray(rows, dtype=np.int32) for item, rows in postings.items()}
        self.popularity_ = pop
        self._fitted = True
        return self

    def score_many_k(
        self, query_items: Sequence[int], ks: Sequence[int]
    ) -> tuple[dict[int, np.ndarray], dict[str, int | bool]]:
        if not self._fitted:
            raise ValueError("fit the TRAIN index before scoring")
        if not ks or any(isinstance(k, bool) or not isinstance(k, Integral) or k <= 0 for k in ks):
            raise ValueError("ks must contain positive integers")
        query = self._ids(query_items, "query items")
        lists = [self.postings_[x] for x in query if x in self.postings_]
        if not lists:
            return {int(k): self.popularity_.copy() for k in ks}, {
                "backoff": True,
                "candidate_count": 0,
                "neighbor_count": 0,
            }
        candidates = np.unique(np.concatenate(lists))[: self.candidate_limit]
        columns = [self.item_to_column_[x] for x in query if x in self.item_to_column_]
        # Use full session membership, not truncated posting counts, for overlap.
        overlap = np.asarray(self.items_[candidates][:, columns].sum(axis=1)).ravel()
        sims = overlap / np.sqrt(len(query) * self.item_counts_[candidates])
        neighbor_order = np.lexsort((candidates, -sims))
        neighbor_order = neighbor_order[sims[neighbor_order] > 0][: max(ks)]
        neighbors = candidates[neighbor_order]
        weights = sims[neighbor_order]
        scores = {}
        for k in ks:
            limit = min(int(k), len(neighbors))
            scores[int(k)] = np.asarray(
                self.categories_[neighbors[:limit]].T.dot(weights[:limit])
            ).ravel()
        return scores, {
            "backoff": False,
            "candidate_count": len(candidates),
            "neighbor_count": len(neighbors),
        }
