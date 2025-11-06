from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from phase_runner.cli import cli


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def test_cli_validate_success(runner: CliRunner, rule_pack_paths) -> None:
    result = runner.invoke(cli, ["validate", str(rule_pack_paths["sunrise"]), "--as-json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["metadata"]["payer"] == "Sunrise Health Commercial"


def test_cli_run_dry_run(runner: CliRunner, rule_pack_paths, payloads, tmp_path: Path) -> None:
    payload_file = tmp_path / "payload.yaml"
    payload_file.write_text("claim:\n  service_line:\n    cpt: 70450\n  charge_amount: 10000\n")
    result = runner.invoke(
        cli,
        [
            "run",
            str(rule_pack_paths["sunrise"]),
            str(payload_file),
            "--dry-run",
            "--as-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["dry_run"] is True
    assert payload["decisions"][0]["rule_id"] == "SUN-001"


def test_cli_diff_json(runner: CliRunner, rule_pack_paths, tmp_path: Path) -> None:
    modified = tmp_path / "sunrise_mod.yaml"
    content = rule_pack_paths["sunrise"].read_text().replace("pend", "manual_review", 1)
    modified.write_text(content)
    result = runner.invoke(
        cli,
        [
            "diff",
            str(rule_pack_paths["sunrise"]),
            str(modified),
            "--as-json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["changed_rules"][0]["id"] == "SUN-001"


def test_cli_validate_all(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["validate-all", "examples/rule_packs"])
    assert result.exit_code == 0
