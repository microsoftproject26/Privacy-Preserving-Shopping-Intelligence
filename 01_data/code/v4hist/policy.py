"""v4hist.policy (owner: Data/integration engineer, DATA-05) — the REQUIRED history-policy enum. Import-free.

Owner decision (DECISION_LOG 36): the input window of every decision is USER_HISTORY_L50, the last 50 representative
events of the user's whole prior stream, across model segments and partition fences, strictly causal, identical for
TRAIN / VALIDATION / TEST. The v1 session-scoped window SESSION_L50 is a named diagnostic ablation only.

There is no default anywhere: every caller names the policy, and `require_policy` refuses None, a wrong type and any
unregistered value.
"""
from __future__ import annotations

USER_HISTORY_L50 = "USER_HISTORY_L50"
SESSION_L50 = "SESSION_L50"
HISTORY_POLICIES = (USER_HISTORY_L50, SESSION_L50)
LOCKED_POLICY = USER_HISTORY_L50                  # DECISION_LOG 36 (owner)
DIAGNOSTIC_POLICIES = (SESSION_L50,)
CONTEXT_LENGTH = 50                               # L of both policies (the v1 frozen context length)
RESOLVER_ID = "v4hist.windows/1"

# chronological partition order of the non-TEST stream (splits/splits.json; half-open intervals)
TRAIN = "TRAIN"
VALIDATION_SELECT = "VALIDATION_SELECT"
VALIDATION_CONFIRM = "VALIDATION_CONFIRM"
NON_TEST_ORDER = (TRAIN, VALIDATION_SELECT, VALIDATION_CONFIRM)


class HistoryPolicyError(ValueError):
    """A missing, None or unregistered history policy (there is no default)."""


def require_policy(policy) -> str:
    if policy is None:
        raise HistoryPolicyError("history_policy is REQUIRED (no default): one of " + ", ".join(HISTORY_POLICIES))
    if not isinstance(policy, str):
        raise HistoryPolicyError(f"history_policy must be a string, got {type(policy).__name__}")
    if policy not in HISTORY_POLICIES:
        raise HistoryPolicyError(f"history_policy {policy!r} is not registered: one of " + ", ".join(HISTORY_POLICIES))
    return policy


def history_splits(decision_splits) -> tuple:
    """The partitions a USER_HISTORY window may read for decisions of `decision_splits`: every non-TEST partition up
    to and including the latest decision partition, in chronological order (a later partition is never needed, and a
    causal window never reaches one). TEST decisions are refused here (no TEST row before ALL_METHODS_LOCK)."""
    ds = tuple(decision_splits)
    if not ds:
        raise HistoryPolicyError("decision_splits must be non-empty")
    for s in ds:
        if "TEST" in str(s).upper():
            raise HistoryPolicyError(f"split {s!r}: no TEST decision is served before ALL_METHODS_LOCK")
        if s not in NON_TEST_ORDER:
            raise HistoryPolicyError(f"unknown decision split {s!r}")
    last = max(NON_TEST_ORDER.index(s) for s in ds)
    return NON_TEST_ORDER[:last + 1]


def store_splits(policy, decision_splits) -> tuple:
    """The partitions the snapshot CompactStore must retain to serve `decision_splits` under `policy`:
    SESSION_L50 = the decision splits themselves (the v1 / INT-01 path, unchanged); USER_HISTORY_L50 = history_splits."""
    p = require_policy(policy)
    ds = tuple(decision_splits)
    history_splits(ds)                                      # validates (and refuses TEST)
    return ds if p == SESSION_L50 else history_splits(ds)


def needs_row_index(policy) -> bool:
    """USER_HISTORY_L50 needs the per-row user index; SESSION_L50 is served by the store's own lengths."""
    return require_policy(policy) == USER_HISTORY_L50


__all__ = ["USER_HISTORY_L50", "SESSION_L50", "HISTORY_POLICIES", "LOCKED_POLICY", "DIAGNOSTIC_POLICIES",
           "CONTEXT_LENGTH", "RESOLVER_ID", "NON_TEST_ORDER", "HistoryPolicyError", "require_policy",
           "history_splits", "store_splits", "needs_row_index"]
