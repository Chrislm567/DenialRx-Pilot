from __future__ import annotations

import copy
from typing import Dict

import pytest

from phase_runner.engine.executor import execute


GOLDEN_CASES = {
    "sunrise_patient_a": ("sunrise", "patient_a"),
    "sunrise_patient_child": ("sunrise", "patient_child"),
    "horizon_er_deny": ("horizon", "er_visit"),
    "horizon_er_pregnant": ("horizon", "er_visit_pregnant"),
    "opencare_visit_15": ("opencare", "pt_visit_15"),
    "opencare_visit_25": ("opencare", "pt_visit_25"),
}


@pytest.mark.parametrize("golden", sorted(GOLDEN_CASES))
def test_execute_matches_golden(
    golden: str, rule_packs: Dict[str, any], payloads: Dict[str, dict], golden_outputs: Dict[str, dict]
) -> None:
    pack_name, payload_name = GOLDEN_CASES[golden]
    result = execute(rule_packs[pack_name], payloads[payload_name], dry_run=True)
    assert result == golden_outputs[golden]


def test_execute_dry_run_false(rule_packs, payloads) -> None:
    result = execute(rule_packs["opencare"], payloads["pt_visit_25"], dry_run=False)
    assert result["decisions"][0]["applied"] is True


def test_execute_deterministic(rule_packs, payloads) -> None:
    payload = copy.deepcopy(payloads["patient_a"])
    payload["claim"]["charge_amount"] = 9000
    result1 = execute(rule_packs["sunrise"], payload, dry_run=True)
    result2 = execute(rule_packs["sunrise"], payload, dry_run=True)
    assert result1 == result2
