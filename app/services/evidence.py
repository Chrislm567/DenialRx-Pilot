from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable, List

from PyPDF2 import PdfMerger


class EvidencePackError(RuntimeError):
    pass


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_evidence_pack(output_path: Path, files: Iterable[Path]) -> Dict[str, str]:
    paths: List[Path] = [Path(f) for f in files]
    if not paths:
        raise EvidencePackError("At least one PDF must be supplied")
    merger = PdfMerger()
    hashes: Dict[str, str] = {}
    for path in paths:
        if not path.exists():
            raise EvidencePackError(f"Attachment missing: {path}")
        if path.suffix.lower() != ".pdf":
            raise EvidencePackError(f"Attachment must be PDF: {path.name}")
        merger.append(str(path))
        hashes[path.name] = _hash_file(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        merger.write(handle)
    merger.close()
    hashes[output_path.name] = _hash_file(output_path)
    return hashes
