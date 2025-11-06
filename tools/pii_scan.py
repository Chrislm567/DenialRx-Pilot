"""Lightweight PHI/PII scanner to protect regulated data in logs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IGNORED_DIRS = {
    "node_modules",
    ".git",
    ".venv",
    "__pycache__",
    "coverage",
}

PATTERNS = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "MRN": re.compile(r"\bMRN\d{6,}\b", re.IGNORECASE),
    "CC": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "DOB": re.compile(r"\b\d{2}/\d{2}/\d{4}\b"),
}


def is_text_file(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            chunk = handle.read(512)
        chunk.decode("utf-8")
    except (UnicodeDecodeError, OSError):
        return False
    return True


def main() -> int:
    violations: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.name.startswith("."):
            continue
        if not is_text_file(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                snippet = match.group(0)
                violations.append(f"{path}: {label} -> {snippet}")
    if violations:
        print("Potential PHI/PII markers detected:")
        for violation in violations:
            print(f" - {violation}")
        return 1
    print("PII scan completed without findings.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
