"""Small temporal-leakage checks for Phase 1 task examples.

The frozen notebooks persist decision metadata but not encoder-history rows.  A
producer can call :func:`validate_temporal_example` with its selected history
and the source events for that logical session before batching an example.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any


class TemporalLeakageError(ValueError):
    """Raised when an example uses information unavailable at decision time."""


def validate_temporal_example(
    task: str,
    example: Mapping[str, Any],
    history_events: Iterable[Mapping[str, Any]],
    session_events: Iterable[Mapping[str, Any]],
    *,
    split_start: str | datetime,
    split_end: str | datetime,
    t3_gain_rule: Mapping[str, int] | None = None,
) -> None:
    """Validate one T1/T2/T3 example against its source session.

    Event records use the names already emitted inside the frozen notebooks:
    ``session``, ``order``, ``event_time``, ``event_type``, ``item`` and
    ``category``.  Example records use their persisted TaskExample fields.
    Canonical ``order`` is authoritative when two events share a timestamp.

    This deliberately does not build examples, fit a catalogue, or decide a
    protocol.  The caller supplies the frozen T3 gain mapping when validating
    T3, so this helper cannot silently create a second gain rule.
    """

    normalized_task = task.upper()
    if normalized_task not in {"T1", "T2", "T3"}:
        raise ValueError("task must be T1, T2, or T3")

    events = list(session_events)
    history = list(history_events)
    if not events:
        raise TemporalLeakageError("session events must not be empty")

    session = _required(example, "session")
    decision_order = _as_order(_required(example, "decision_order"), "decision_order")
    start = _as_utc(split_start, "split_start")
    end = _as_utc(split_end, "split_end")
    if start >= end:
        raise ValueError("split_start must be before split_end")

    indexed: dict[int, Mapping[str, Any]] = {}
    prior_time: datetime | None = None
    for event in sorted(events, key=lambda row: _as_order(_required(row, "order"), "order")):
        if _required(event, "session") != session:
            raise TemporalLeakageError("session events must belong to the example session")
        order = _as_order(_required(event, "order"), "order")
        if order in indexed:
            raise TemporalLeakageError(f"duplicate canonical order {order} in session")
        event_time = _event_time(event)
        if prior_time is not None and event_time < prior_time:
            raise TemporalLeakageError("event time goes backwards in canonical order")
        indexed[order] = event
        prior_time = event_time

    decision = indexed.get(decision_order)
    if decision is None:
        raise TemporalLeakageError("decision_order does not identify a source event")
    decision_time = _event_time(decision)
    _require_inside_split(decision_time, start, end, "decision event")

    for event in history:
        if _required(event, "session") != session:
            raise TemporalLeakageError("history event belongs to a different session")
        order = _as_order(_required(event, "order"), "history order")
        if order not in indexed:
            raise TemporalLeakageError("history event is not present in the source session")
        if order >= decision_order:
            raise TemporalLeakageError(
                "history must be strictly before decision_order; future target/outcome leaked"
            )
        _require_inside_split(_event_time(event), start, end, "history event")

    ordered = [indexed[order] for order in sorted(indexed)]
    if normalized_task == "T1":
        _validate_t1(example, decision, ordered, decision_order, start, end)
    elif normalized_task == "T2":
        _validate_t2(example, decision, ordered, decision_order, start, end)
    else:
        if t3_gain_rule is None:
            raise ValueError("t3_gain_rule must come from the frozen T3 protocol")
        _validate_t3(
            example,
            decision,
            ordered,
            decision_order,
            start,
            end,
            t3_gain_rule,
        )


def _validate_t1(
    example: Mapping[str, Any],
    decision: Mapping[str, Any],
    events: list[Mapping[str, Any]],
    decision_order: int,
    start: datetime,
    end: datetime,
) -> None:
    current_category = _required(example, "current_category")
    if _required(decision, "category") != current_category:
        raise TemporalLeakageError("T1 current_category does not match the decision event")

    current_item = _required(decision, "item")
    target = next(
        (
            event
            for event in events
            if _as_order(_required(event, "order"), "order") > decision_order
            and _required(event, "item") != current_item
        ),
        None,
    )
    if target is None:
        raise TemporalLeakageError("T1 requires a later different-item target in the session")
    _require_inside_split(_event_time(target), start, end, "T1 target")
    if _required(example, "label_value") != _required(target, "category"):
        raise TemporalLeakageError("T1 label is not the first later different-item category")
    _require_maturation(example, _event_time(target), "T1 target")
    _require_observed(example, "T1")


def _validate_t2(
    example: Mapping[str, Any],
    decision: Mapping[str, Any],
    events: list[Mapping[str, Any]],
    decision_order: int,
    start: datetime,
    end: datetime,
) -> None:
    item = _required(example, "item")
    if _required(decision, "event_type") != "view" or _required(decision, "item") != item:
        raise TemporalLeakageError("T2 decision must be the query item's first view")
    _require_first_view(events, decision_order, item, "T2")

    later = [
        event
        for event in events
        if _as_order(_required(event, "order"), "order") > decision_order
    ]
    later_purchase = any(
        _required(event, "event_type") == "purchase" and _required(event, "item") == item
        for event in later
    )
    session_end = events[-1]
    _require_inside_split(_event_time(session_end), start, end, "T2 outcome horizon")
    _require_maturation(example, _event_time(session_end), "T2 outcome horizon")

    if not later:
        _require_censored(example, "T2 decision without an observable outcome horizon")
        return

    _require_observed(example, "T2")
    label = _required(example, "label_value")
    if label not in {0, 1} or bool(label) != later_purchase:
        raise TemporalLeakageError(
            "T2 positive requires a later same-session purchase; mature negatives must be zero"
        )


def _validate_t3(
    example: Mapping[str, Any],
    decision: Mapping[str, Any],
    events: list[Mapping[str, Any]],
    decision_order: int,
    start: datetime,
    end: datetime,
    gain_rule: Mapping[str, int],
) -> None:
    query_item = _required(example, "query_item")
    if _required(decision, "event_type") != "view" or _required(decision, "item") != query_item:
        raise TemporalLeakageError("T3 decision must be the query item's first view")
    _require_first_view(events, decision_order, query_item, "T3")

    positive_item = _required(example, "positive_item")
    positive_type = _required(example, "event_type")
    if positive_item == query_item:
        raise TemporalLeakageError("T3 positive must be a different item")
    if positive_type not in gain_rule:
        raise TemporalLeakageError("T3 event type is absent from the frozen gain rule")

    positives = [
        event
        for event in events
        if _as_order(_required(event, "order"), "order") > decision_order
        and _required(event, "item") == positive_item
        and _required(event, "event_type") == positive_type
    ]
    if not positives:
        raise TemporalLeakageError("T3 positive must occur later in the same session")
    if any(_required(event, "category") != _required(decision, "category") for event in positives):
        raise TemporalLeakageError("T3 frozen eligibility requires the same category")

    expected_gain = gain_rule[positive_type]
    if _required(example, "label_value") != expected_gain:
        raise TemporalLeakageError("T3 label does not match the frozen gain rule")
    session_end = events[-1]
    _require_inside_split(_event_time(session_end), start, end, "T3 outcome horizon")
    _require_maturation(example, _event_time(session_end), "T3 outcome horizon")
    _require_observed(example, "T3")


def _require_first_view(
    events: list[Mapping[str, Any]], decision_order: int, item: Any, task: str
) -> None:
    if any(
        _as_order(_required(event, "order"), "order") < decision_order
        and _required(event, "event_type") == "view"
        and _required(event, "item") == item
        for event in events
    ):
        raise TemporalLeakageError(f"{task} decision is not the query item's first view")


def _require_observed(example: Mapping[str, Any], task: str) -> None:
    if example.get("task_mask") is not True or example.get("status") != "OBSERVED":
        raise TemporalLeakageError(f"{task} observed outcome must have task_mask=true")
    if example.get("label_value") is None:
        raise TemporalLeakageError(f"{task} observed outcome must carry a label")


def _require_censored(example: Mapping[str, Any], reason: str) -> None:
    if (
        example.get("task_mask") is not False
        or example.get("status") != "CENSORED"
        or example.get("label_value") is not None
    ):
        raise TemporalLeakageError(f"{reason} must be censored, never a negative label")


def _require_maturation(
    example: Mapping[str, Any], expected: datetime, description: str
) -> None:
    actual = _as_utc(_required(example, "label_matures_at"), "label_matures_at")
    if actual != expected:
        raise TemporalLeakageError(f"label_matures_at must equal the {description} time")


def _require_inside_split(
    value: datetime, start: datetime, end: datetime, description: str
) -> None:
    if not start <= value < end:
        raise TemporalLeakageError(f"{description} crosses the approved temporal split")


def _event_time(event: Mapping[str, Any]) -> datetime:
    return _as_utc(_required(event, "event_time"), "event_time")


def _required(record: Mapping[str, Any], field: str) -> Any:
    if field not in record:
        raise TemporalLeakageError(f"missing required field: {field}")
    return record[field]


def _as_order(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TemporalLeakageError(f"{field} must be an integer")
    return value


def _as_utc(value: Any, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as error:
            raise TemporalLeakageError(f"{field} must be an ISO 8601 timestamp") from error
    elif hasattr(value, "to_pydatetime"):
        parsed = value.to_pydatetime()
    else:
        raise TemporalLeakageError(f"{field} must be a timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TemporalLeakageError(f"{field} must include a UTC offset")
    return parsed.astimezone(UTC)
