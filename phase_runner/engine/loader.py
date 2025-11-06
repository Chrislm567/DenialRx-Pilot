"""Utilities for loading and validating rule packs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from jsonschema import Draft202012Validator

from .schema import RULE_PACK_SCHEMA


@dataclass(frozen=True)
class RulePack:
    """In-memory representation of a rule pack."""

    source: Path
    data: Dict[str, Any]

    @property
    def payer(self) -> str:
        return self.data["metadata"]["payer"]

    @property
    def pack_version(self) -> str:
        return self.data["metadata"]["pack_version"]

    @property
    def engine_version(self) -> str:
        return self.data["metadata"]["engine_version"]

    @property
    def rules(self) -> List[Dict[str, Any]]:
        return list(self.data["rules"])

    def to_json(self) -> str:
        return json.dumps(self.data, sort_keys=True, indent=2)


_validator = Draft202012Validator(RULE_PACK_SCHEMA)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load YAML from ``path`` with safe semantics."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(payload).__name__}")
    return payload


def validate_pack(data: Dict[str, Any], *, source: Optional[Path] = None) -> List[str]:
    """Validate ``data`` using the rule pack schema.

    Returns a list of human friendly error messages sorted deterministically.
    """

    errors = sorted(_validator.iter_errors(data), key=lambda err: err.json_path)
    messages = [format_validation_error(error, source=source) for error in errors]
    return messages


def format_validation_error(error, *, source: Optional[Path] = None) -> str:
    location = error.json_path or "$"
    src = f"{source}: " if source else ""
    return f"{src}{location} -> {error.message}"


def load_and_validate(path: Path) -> RulePack:
    """Load YAML from ``path`` and validate it, raising if invalid."""

    data = load_yaml(path)
    errors = validate_pack(data, source=path)
    if errors:
        raise ValueError("\n".join(errors))
    return RulePack(source=path, data=data)


def compare_packs(base: RulePack, other: RulePack) -> Dict[str, Any]:
    """Return a deterministic diff between two rule packs."""

    diff: Dict[str, Any] = {
        "payer": base.payer,
        "from_version": base.pack_version,
        "to_version": other.pack_version,
        "added_rules": [],
        "removed_rules": [],
        "changed_rules": [],
    }

    base_rules = {rule["id"]: rule for rule in base.rules}
    other_rules = {rule["id"]: rule for rule in other.rules}

    for rule_id in sorted(set(other_rules) - set(base_rules)):
        diff["added_rules"].append(other_rules[rule_id])

    for rule_id in sorted(set(base_rules) - set(other_rules)):
        diff["removed_rules"].append(base_rules[rule_id])

    for rule_id in sorted(set(base_rules) & set(other_rules)):
        if base_rules[rule_id] != other_rules[rule_id]:
            diff["changed_rules"].append(
                {
                    "id": rule_id,
                    "from": base_rules[rule_id],
                    "to": other_rules[rule_id],
                }
            )

    return diff


def iter_rule_pack_paths(paths: Iterable[Path]) -> Iterable[Path]:
    """Yield YAML files from ``paths`` recursively."""

    for path in paths:
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*.yml"))
            yield from sorted(p for p in path.rglob("*.yaml"))
        elif path.suffix.lower() in {".yml", ".yaml"}:
            yield path


__all__ = [
    "RulePack",
    "load_yaml",
    "validate_pack",
    "load_and_validate",
    "compare_packs",
    "iter_rule_pack_paths",
]
