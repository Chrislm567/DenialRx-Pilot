#!/usr/bin/env python3
"""Simple PHI scanner used locally and in CI."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

DEFAULT_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "dob": re.compile(r"(?i)\b(?:dob|birth(?:date)?)[:\s]+(19|20)\d{2}-[01]\d-[0-3]\d\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}

EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", "tests/data"}


def iter_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_dir():
            if path.name in EXCLUDE_DIRS:
                continue
        elif path.is_file():
            if path.suffix in {".pyc", ".png", ".jpg", ".jpeg", ".gif"}:
                continue
            yield path


def scan_file(path: Path) -> List[Tuple[int, str, str]]:
    findings: List[Tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings
    for idx, line in enumerate(text.splitlines(), start=1):
        if "PHI_OK" in line:
            continue
        for label, pattern in DEFAULT_PATTERNS.items():
            if pattern.search(line):
                findings.append((idx, label, line.strip()))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan repository for PHI patterns")
    parser.add_argument("path", nargs="?", default=".", help="Root path to scan")
    args = parser.parse_args()
    root = Path(args.path).resolve()
    findings_total: List[Tuple[Path, int, str, str]] = []
    for file_path in iter_files(root):
        if any(part in EXCLUDE_DIRS for part in file_path.parts):
            continue
        findings = scan_file(file_path)
        for line_no, label, line in findings:
            findings_total.append((file_path.relative_to(root), line_no, label, line))
    if findings_total:
        for relative, line_no, label, line in findings_total:
            print(f"{relative}:{line_no} [{label}] {line}")
        return 1
    print("No PHI patterns detected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
