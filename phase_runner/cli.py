"""Command line interface for Phase Runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click
import yaml

from .engine.executor import execute
from .engine.loader import compare_packs, iter_rule_pack_paths, load_and_validate


@click.group()
def cli() -> None:
    """Phase Runner rules engine commands."""


@cli.command()
@click.argument("pack", type=click.Path(path_type=Path))
@click.option("--as-json", is_flag=True, help="Emit normalized JSON on success.")
def validate(pack: Path, as_json: bool) -> None:
    """Validate a rule pack against the schema."""

    rule_pack = load_and_validate(pack)
    if as_json:
        click.echo(rule_pack.to_json())
    else:
        click.echo(f"{pack} is valid for payer {rule_pack.payer} {rule_pack.pack_version}")


@cli.command()
@click.argument("pack", type=click.Path(path_type=Path))
@click.argument("payload", type=click.Path(path_type=Path))
@click.option("--dry-run/--apply", default=True, help="Do not mark results as applied.")
@click.option("--as-json", is_flag=True, help="Emit JSON results.")
def run(pack: Path, payload: Path, dry_run: bool, as_json: bool) -> None:
    """Execute a rule pack for the supplied payload."""

    rule_pack = load_and_validate(pack)
    with payload.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) if payload.suffix in {".yml", ".yaml"} else json.load(handle)
    result = execute(rule_pack, data, dry_run=dry_run)
    if as_json:
        click.echo(json.dumps(result, indent=2, sort_keys=True))
    else:
        click.echo(
            f"{pack.name}@{rule_pack.pack_version} -> {len(result['decisions'])} decision(s)"
        )
        for decision in result["decisions"]:
            click.echo(
                f"- {decision['rule_id']} {decision['decision']} (applied={decision['applied']})"
            )


@cli.command()
@click.argument("base", type=click.Path(path_type=Path))
@click.argument("other", type=click.Path(path_type=Path))
@click.option("--as-json", is_flag=True, help="Emit JSON diff for tooling consumption.")
def diff(base: Path, other: Path, as_json: bool) -> None:
    """Show differences between two rule packs."""

    base_pack = load_and_validate(base)
    other_pack = load_and_validate(other)
    delta = compare_packs(base_pack, other_pack)
    if as_json:
        click.echo(json.dumps(delta, indent=2, sort_keys=True))
        return
    click.echo(
        f"{delta['payer']} {delta['from_version']} -> {delta['to_version']}"
    )
    if delta["added_rules"]:
        click.echo("Added rules:")
        for rule in delta["added_rules"]:
            click.echo(f"+ {rule['id']}: {rule.get('description', '')}")
    if delta["removed_rules"]:
        click.echo("Removed rules:")
        for rule in delta["removed_rules"]:
            click.echo(f"- {rule['id']}: {rule.get('description', '')}")
    if delta["changed_rules"]:
        click.echo("Changed rules:")
        for change in delta["changed_rules"]:
            click.echo(f"* {change['id']}")


@cli.command()
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
def validate_all(paths: tuple[Path, ...]) -> None:
    """Validate all rule packs discovered under ``paths``."""

    any_errors = False
    for path in iter_rule_pack_paths(paths):
        try:
            load_and_validate(path)
        except ValueError as exc:  # pragma: no cover - sanity output
            any_errors = True
            click.echo(str(exc))
    if any_errors:
        raise SystemExit(1)


def main(argv: Optional[list[str]] = None) -> None:  # pragma: no cover - exercised via CLI tests
    cli.main(args=argv, prog_name="phase-runner")


if __name__ == "__main__":  # pragma: no cover
    main()
