"""Phase Runner rules engine package."""

from importlib import metadata


try:  # pragma: no cover - version metadata is environment dependent
    __version__ = metadata.version("phase-runner")
except metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = ["__version__"]
