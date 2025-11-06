from __future__ import annotations

import copy
from pathlib import Path

import pytest

from phase_runner.engine.loader import (
    RulePack,
    compare_packs,
    iter_rule_pack_paths,
    load_yaml,
    validate_pack,
)


def test_load_yaml_requires_mapping(tmp_path: Path) -> None:
    path = tmp_path / "rule.yaml"
    path.write_text("- not-a-mapping")
    with pytest.raises(TypeError):
        load_yaml(path)


def test_validate_pack_sorted_errors() -> None:
    payload = {"metadata": {"payer": "P"}, "rules": []}
    errors = validate_pack(payload)
    assert errors[0].startswith("$")


def test_compare_packs(rule_packs) -> None:
    base = rule_packs["sunrise"]
    modified_data = copy.deepcopy(base.data)
    modified_data["rules"][0]["id"] = "SUN-004"
    other = RulePack(source=base.source, data=modified_data)
    diff = compare_packs(base, other)
    assert diff["added_rules"][0]["id"] == "SUN-004"


def test_iter_rule_pack_paths(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    first = tmp_path / "first.yaml"
    second = tmp_path / "nested" / "second.yml"
    third = tmp_path / "ignore.txt"
    first.write_text("{}")
    second.write_text("{}")
    third.write_text("{}")
    paths = list(iter_rule_pack_paths([tmp_path]))
    assert first in paths and second in paths and third not in paths
