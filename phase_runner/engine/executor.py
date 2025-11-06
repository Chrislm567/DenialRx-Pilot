"""Rule pack execution primitives."""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from .loader import RulePack


class ConditionEvaluator:
    """Evaluate rule conditions against an input payload."""

    OPERATORS = {
        "eq": operator.eq,
        "ne": operator.ne,
        "lt": operator.lt,
        "lte": operator.le,
        "gt": operator.gt,
        "gte": operator.ge,
        "in": lambda a, b: a in b if isinstance(b, Iterable) else False,
        "not_in": lambda a, b: a not in b if isinstance(b, Iterable) else False,
        "exists": lambda a, _: a is not None,
        "missing": lambda a, _: a is None,
        "matches": lambda a, b: bool(re.search(str(b), str(a))),
        "starts_with": lambda a, b: str(a).startswith(str(b)),
        "ends_with": lambda a, b: str(a).endswith(str(b)),
    }

    def __init__(self, payload: Mapping[str, Any]):
        self.payload = payload

    def evaluate(self, condition: Mapping[str, Any]) -> bool:
        if "all" in condition:
            return all(self.evaluate(child) for child in condition["all"])
        if "any" in condition:
            return any(self.evaluate(child) for child in condition["any"])
        if "not" in condition:
            return not self.evaluate(condition["not"])

        field_path = condition.get("field")
        op_name = condition.get("op")
        value = condition.get("value")
        resolver = self.OPERATORS.get(op_name)
        if resolver is None:
            raise ValueError(f"Unsupported operator: {op_name}")

        actual = self._resolve_field(field_path)
        if op_name in {"exists", "missing"}:
            return resolver(actual, None)
        if actual is None and op_name in {"lt", "lte", "gt", "gte"}:
            return False
        if actual is None and op_name in {"matches", "starts_with", "ends_with"}:
            return False
        return resolver(actual, value)

    def _resolve_field(self, field_path: str) -> Any:
        if not field_path:
            return None
        parts = field_path.split(".")
        current: Any = self.payload
        for part in parts:
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                return None
        return current


@dataclass
class RuleResult:
    rule_id: str
    decision: str
    reason: str | None
    severity: str
    applied: bool
    references: List[str]
    flags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "decision": self.decision,
            "reason": self.reason,
            "severity": self.severity,
            "applied": self.applied,
            "references": list(self.references),
            "flags": list(self.flags),
        }


def execute(pack: RulePack, payload: Mapping[str, Any], *, dry_run: bool = False) -> Dict[str, Any]:
    """Execute ``pack`` against ``payload`` returning deterministic results."""

    evaluator = ConditionEvaluator(payload)
    results: List[RuleResult] = []
    for rule in sorted(pack.rules, key=lambda item: item["id"]):
        if evaluator.evaluate(rule["when"]):
            outcome = rule["then"]
            result = RuleResult(
                rule_id=rule["id"],
                decision=outcome["decision"],
                reason=outcome.get("reason"),
                severity=outcome.get("severity", "warning"),
                applied=not dry_run,
                references=sorted(outcome.get("references", [])),
                flags=sorted(rule.get("flags", [])),
            )
            results.append(result)
    summary = {
        "payer": pack.payer,
        "pack_version": pack.pack_version,
        "engine_version": pack.engine_version,
        "dry_run": dry_run,
        "decisions": [result.to_dict() for result in results],
    }
    return summary


__all__ = ["execute", "RuleResult", "ConditionEvaluator"]
