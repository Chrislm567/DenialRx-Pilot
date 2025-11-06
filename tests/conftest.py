from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phase_runner.engine.loader import RulePack, load_and_validate


FIXTURE_ROOT = Path(__file__).parent / "fixtures"
RULE_PACK_ROOT = Path(__file__).resolve().parent.parent / "examples" / "rule_packs"
INPUT_ROOT = Path(__file__).resolve().parent.parent / "examples" / "inputs"
GOLDEN_ROOT = FIXTURE_ROOT / "golden"


@pytest.fixture(scope="session")
def rule_pack_paths() -> Dict[str, Path]:
    return {
        "sunrise": RULE_PACK_ROOT / "sunrise_health.yaml",
        "horizon": RULE_PACK_ROOT / "horizon_medicaid.yaml",
        "opencare": RULE_PACK_ROOT / "opencare_value.yaml",
    }


@pytest.fixture(scope="session")
def rule_packs(rule_pack_paths: Dict[str, Path]) -> Dict[str, RulePack]:
    return {name: load_and_validate(path) for name, path in rule_pack_paths.items()}


@pytest.fixture(scope="session")
def payloads() -> Dict[str, Dict]:
    payload_data = {}
    for path in INPUT_ROOT.glob("*.yaml"):
        payload_data[path.stem] = yaml.safe_load(path.read_text())
    return payload_data


@pytest.fixture(scope="session")
def golden_outputs() -> Dict[str, Dict]:
    outputs = {}
    for path in GOLDEN_ROOT.glob("*.json"):
        outputs[path.stem] = json.loads(path.read_text())
    return outputs
