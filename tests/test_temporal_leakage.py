from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from ppsi.temporal import TemporalLeakageError, validate_temporal_example

SPLIT_START = "2019-10-22T00:00:00+00:00"
SPLIT_END = "2019-10-27T00:00:00+00:00"


def event(
    order: int,
    minute: int,
    *,
    item: int,
    category: int,
    event_type: str = "view",
    session: str = "session-1",
) -> dict[str, object]:
    return {
        "session": session,
        "order": order,
        "event_time": f"2019-10-23T10:{minute:02d}:00+00:00",
        "event_type": event_type,
        "item": item,
        "category": category,
    }


def validate(task, example, history, events, **kwargs) -> None:
    validate_temporal_example(
        task,
        example,
        history,
        events,
        split_start=SPLIT_START,
        split_end=SPLIT_END,
        **kwargs,
    )


@pytest.fixture
def t1_case():
    events = [
        event(10, 0, item=1, category=100),
        event(11, 1, item=1, category=100),
        event(12, 2, item=2, category=200, event_type="cart"),
    ]
    example = {
        "session": "session-1",
        "decision_order": 11,
        "current_category": 100,
        "label_value": 200,
        "task_mask": True,
        "status": "OBSERVED",
        "label_matures_at": events[2]["event_time"],
        "split": "VALIDATION",
    }
    return example, [events[0]], events


@pytest.fixture
def t2_case():
    events = [
        event(20, 0, item=9, category=100, event_type="cart"),
        event(21, 1, item=10, category=100),
        event(22, 2, item=11, category=100, event_type="cart"),
        event(23, 3, item=10, category=100, event_type="purchase"),
    ]
    example = {
        "session": "session-1",
        "decision_order": 21,
        "item": 10,
        "category": 100,
        "label_value": 1.0,
        "task_mask": True,
        "status": "OBSERVED",
        "label_matures_at": events[-1]["event_time"],
        "split": "VALIDATION",
    }
    return example, [events[0]], events


@pytest.fixture
def t3_case():
    events = [
        event(30, 0, item=9, category=100, event_type="cart"),
        event(31, 1, item=10, category=100),
        event(32, 2, item=20, category=100, event_type="purchase"),
    ]
    example = {
        "session": "session-1",
        "decision_order": 31,
        "category": 100,
        "query_item": 10,
        "positive_item": 20,
        "event_type": "purchase",
        "label_value": 2,
        "task_mask": True,
        "status": "OBSERVED",
        "label_matures_at": events[-1]["event_time"],
        "split": "VALIDATION",
    }
    return example, [events[0]], events


def frozen_t3_gain_rule() -> dict[str, int]:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "evidence"
        / "s1-d1-ds-07"
        / "t3_protocol_v1.proposed.json"
    )
    protocol = json.loads(path.read_text(encoding="utf-8"))
    return protocol["gain_rule"]["gains"]


def test_known_good_examples_pass_repeatedly(t1_case, t2_case, t3_case) -> None:
    for _ in range(2):
        validate("T1", *t1_case)
        validate("T2", *t2_case)
        validate("T3", *t3_case, t3_gain_rule=frozen_t3_gain_rule())


@pytest.mark.parametrize("task,fixture_name", [("T1", "t1_case"), ("T2", "t2_case")])
def test_future_event_in_history_is_rejected(task, fixture_name, request) -> None:
    example, history, events = request.getfixturevalue(fixture_name)
    history.append(events[-1])

    with pytest.raises(TemporalLeakageError, match="history must be strictly before"):
        validate(task, example, history, events)


def test_t1_target_and_maturation_are_the_first_later_different_item(t1_case) -> None:
    example, history, events = t1_case
    leaking = deepcopy(example)
    leaking["label_matures_at"] = events[1]["event_time"]

    with pytest.raises(TemporalLeakageError, match="T1 target time"):
        validate("T1", leaking, history, events)


def test_t2_query_itself_cannot_enter_encoder_history(t2_case) -> None:
    example, history, events = t2_case
    history.append(events[1])

    with pytest.raises(TemporalLeakageError, match="history must be strictly before"):
        validate("T2", example, history, events)


def test_t2_positive_requires_a_later_purchase_in_the_same_session(t2_case) -> None:
    example, history, events = t2_case
    events[-1] = event(23, 3, item=10, category=100, event_type="cart")
    example["label_matures_at"] = events[-1]["event_time"]

    with pytest.raises(TemporalLeakageError, match="later same-session purchase"):
        validate("T2", example, history, events)


def test_censored_t2_outcome_is_unknown_not_negative() -> None:
    only_view = event(40, 0, item=10, category=100)
    censored = {
        "session": "session-1",
        "decision_order": 40,
        "item": 10,
        "category": 100,
        "label_value": None,
        "task_mask": False,
        "status": "CENSORED",
        "label_matures_at": only_view["event_time"],
        "split": "VALIDATION",
    }
    validate("T2", censored, [], [only_view])

    censored["label_value"] = 0.0
    with pytest.raises(TemporalLeakageError, match="censored, never a negative"):
        validate("T2", censored, [], [only_view])


def test_event_on_cutoff_belongs_only_to_the_later_split(t1_case) -> None:
    example, history, events = t1_case
    # Move the decision and everything after it onto the cutoff instant, keeping the
    # session chronological so the split rule is what rejects it.
    for row in events[1:]:
        row["event_time"] = SPLIT_END
    example["label_matures_at"] = SPLIT_END

    with pytest.raises(TemporalLeakageError, match="crosses the approved temporal split"):
        validate("T1", example, history, events)


def test_t3_uses_the_frozen_same_category_and_gain_protocol(t3_case) -> None:
    gate_path = (
        Path(__file__).parents[1]
        / "docs"
        / "evidence"
        / "s1-ds-09"
        / "g1_gate_v1.frozen.json"
    )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["decisions"]["t3_eligibility"]["rule"] == "same category only"

    example, history, events = t3_case
    events[-1]["category"] = 999
    with pytest.raises(TemporalLeakageError, match="same category"):
        validate(
            "T3",
            example,
            history,
            events,
            t3_gain_rule=frozen_t3_gain_rule(),
        )


def test_t3_positive_cannot_precede_its_query() -> None:
    # The comparison purchase happens before the query is ever viewed, so it cannot be
    # a positive for a decision that had not been taken yet.
    events = [
        event(30, 0, item=20, category=100, event_type="purchase"),
        event(31, 1, item=10, category=100),
    ]
    example = {
        "session": "session-1",
        "decision_order": 31,
        "category": 100,
        "query_item": 10,
        "positive_item": 20,
        "event_type": "purchase",
        "label_value": 2,
        "task_mask": True,
        "status": "OBSERVED",
        "label_matures_at": events[-1]["event_time"],
        "split": "VALIDATION",
    }

    with pytest.raises(TemporalLeakageError, match="must occur later"):
        validate(
            "T3",
            example,
            [events[0]],
            events,
            t3_gain_rule=frozen_t3_gain_rule(),
        )
