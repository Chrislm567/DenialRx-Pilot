"""Lightweight scoring primitives for fast feedback in CI pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Scorecard:
    """Represents a collection of numeric rule scores."""

    scores: tuple[float, ...]

    def average(self) -> float:
        """Return the arithmetic mean of the stored scores."""

        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)


def evaluate_threshold(scores: Iterable[float], *, threshold: float) -> bool:
    """Evaluate whether the provided scores exceed a safety threshold.

    The helper intentionally avoids any external dependencies so it can be
    executed as part of quick smoke-tests inside the monorepo CI.
    """

    normalized = tuple(float(score) for score in scores)
    card = Scorecard(scores=normalized)
    return card.average() >= threshold
