from __future__ import annotations

from pathlib import Path

import pytest

from phase_runner.engine.loader import load_and_validate, validate_pack
from phase_runner.engine.schema import RULE_PACK_SCHEMA


@pytest.mark.parametrize(
    "pack_name",
    ["sunrise", "horizon", "opencare"],
)
def test_rule_packs_valid(rule_pack_paths: dict[str, Path], pack_name: str) -> None:
    load_and_validate(rule_pack_paths[pack_name])


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("metadata", None, "None is not of type 'object'"),
        ("rules", [], "should be non-empty"),
        ("metadata", {"payer": "Only"}, "'pack_version' is a required property"),
        (
            "rules",
            [
                {
                    "id": "RULE-1",
                    "when": {"field": "a", "op": "eq", "value": 1},
                }
            ],
            "'then' is a required property",
        ),
        (
            "rules",
            [
                {
                    "id": "RULE-1",
                    "when": {"field": "a", "op": "unsupported", "value": 1},
                    "then": {"decision": "approve"},
                }
            ],
            "is not valid under any of the given schemas",
        ),
        (
            "rules",
            [
                {
                    "id": "RULE-1",
                    "when": {"any": [{"field": "a", "op": "eq", "value": 1}]},
                    "then": {"decision": "approve"},
                    "extra": True,
                }
            ],
            "Additional properties are not allowed",
        ),
        (
            "metadata",
            {
                "payer": "A",
                "pack_version": "1",
                "engine_version": "0.1",
                "tags": ["dup", "dup"],
            },
            "['dup', 'dup'] has non-unique elements",
        ),
        (
            "rules",
            [
                {
                    "id": "RULE-1",
                    "when": {"field": "a", "op": "eq"},
                    "then": {"decision": "approve"},
                }
            ],
            "is not valid under any of the given schemas",
        ),
        (
            "rules",
            [
                {
                    "id": "RULE-1",
                    "when": {"field": "a", "op": "exists", "value": 1},
                    "then": {"decision": "approve"},
                }
            ],
            "is not valid under any of the given schemas",
        ),
        (
            "rules",
            [
                {
                    "id": "RULE-1",
                    "when": {"field": "a", "op": "matches", "value": 1},
                    "then": {"decision": "approve", "severity": "unknown"},
                }
            ],
            "'unknown' is not one of",
        ),
    ],
)
def test_schema_validation_errors(field, value, message):
    payload = {"metadata": {"payer": "P", "pack_version": "1", "engine_version": "0.1"}, "rules": []}
    payload[field] = value
    errors = validate_pack(payload)
    assert any(message in error for error in errors)


def test_schema_structure_keys() -> None:
    assert set(RULE_PACK_SCHEMA["properties"].keys()) == {"metadata", "rules"}
