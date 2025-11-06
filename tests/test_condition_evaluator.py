from __future__ import annotations

from typing import Any, Dict

import pytest

from phase_runner.engine.executor import ConditionEvaluator


def _payload(**kwargs) -> Dict[str, Any]:
    return kwargs


@pytest.mark.parametrize(
    "payload, condition, expected",
    [
        (_payload(value=i), {"field": "value", "op": "eq", "value": i}, True)
        for i in range(0, 10)
    ]
    + [
        (_payload(value=i), {"field": "value", "op": "ne", "value": i + 1}, True)
        for i in range(0, 10)
    ]
    + [
        (_payload(value=i), {"field": "value", "op": "lt", "value": i + 1}, True)
        for i in range(0, 10)
    ]
    + [
        (_payload(value=i), {"field": "value", "op": "lte", "value": i}, True)
        for i in range(0, 10)
    ]
    + [
        (_payload(value=i), {"field": "value", "op": "gt", "value": i - 1}, True)
        for i in range(1, 11)
    ]
    + [
        (_payload(value=i), {"field": "value", "op": "gte", "value": i}, True)
        for i in range(0, 10)
    ]
)
def test_comparison_operators(payload: Dict[str, Any], condition: Dict[str, Any], expected: bool) -> None:
    evaluator = ConditionEvaluator(payload)
    assert evaluator.evaluate(condition) is expected


@pytest.mark.parametrize(
    "payload, condition, expected",
    [
        (
            _payload(colors=["red", "green"], value=i),
            {"field": "value", "op": "in", "value": list(range(0, 5))},
            i in list(range(0, 5)),
        )
        for i in range(0, 10)
    ]
    + [
        (
            _payload(colors=["red", "green"], value=i),
            {"field": "value", "op": "not_in", "value": list(range(5, 10))},
            i not in list(range(5, 10)),
        )
        for i in range(0, 10)
    ]
)
def test_membership_operators(payload: Dict[str, Any], condition: Dict[str, Any], expected: bool) -> None:
    evaluator = ConditionEvaluator(payload)
    assert evaluator.evaluate(condition) is expected


@pytest.mark.parametrize(
    "payload, condition, expected",
    [
        (_payload(name=f"alpha-{i}"), {"field": "name", "op": "matches", "value": r"alpha-\d"}, True)
        for i in range(0, 10)
    ]
    + [
        (_payload(name=f"beta-{i}"), {"field": "name", "op": "starts_with", "value": "beta"}, True)
        for i in range(0, 10)
    ]
    + [
        (
            _payload(name=f"gamma-{i}"),
            {"field": "name", "op": "ends_with", "value": str(i)},
            True,
        )
        for i in range(0, 10)
    ]
)
def test_string_operators(payload: Dict[str, Any], condition: Dict[str, Any], expected: bool) -> None:
    evaluator = ConditionEvaluator(payload)
    assert evaluator.evaluate(condition) is expected


@pytest.mark.parametrize(
    "payload, condition, expected",
    [
        (_payload(optional=i), {"field": "optional", "op": "exists"}, True)
        for i in range(0, 10)
    ]
    + [
        ({}, {"field": "missing", "op": "missing"}, True)
        for _ in range(0, 10)
    ]
)
def test_presence_operators(payload: Dict[str, Any], condition: Dict[str, Any], expected: bool) -> None:
    evaluator = ConditionEvaluator(payload)
    assert evaluator.evaluate(condition) is expected


@pytest.mark.parametrize(
    "payload, condition, expected",
    [
        (
            {"a": i, "b": {"c": i}},
            {"all": [{"field": "a", "op": "gte", "value": i}, {"field": "b.c", "op": "eq", "value": i}]},
            True,
        )
        for i in range(0, 10)
    ]
    + [
        (
            {"a": i, "b": {"c": i}},
            {"any": [{"field": "a", "op": "lt", "value": -1}, {"field": "b.c", "op": "eq", "value": i}]},
            True,
        )
        for i in range(0, 10)
    ]
    + [
        (
            {"flag": True},
            {"not": {"field": "flag", "op": "eq", "value": False}},
            True,
        )
        for _ in range(0, 10)
    ]
)
def test_boolean_combinators(payload: Dict[str, Any], condition: Dict[str, Any], expected: bool) -> None:
    evaluator = ConditionEvaluator(payload)
    assert evaluator.evaluate(condition) is expected


def test_missing_field_returns_none() -> None:
    evaluator = ConditionEvaluator({"nested": {}})
    assert evaluator._resolve_field("nested.value") is None


def test_unknown_operator_raises() -> None:
    evaluator = ConditionEvaluator({})
    with pytest.raises(ValueError):
        evaluator.evaluate({"field": "value", "op": "unknown", "value": 1})
