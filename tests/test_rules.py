"""Unit tests for the lightweight rules scoring helper."""

from __future__ import annotations

import pathlib
import sys
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "packages" / "rules" / "src"
if str(MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(MODULE_PATH))

from rules import Scorecard, evaluate_threshold  # noqa: E402


class ScorecardTestCase(unittest.TestCase):
    def test_average_handles_empty(self) -> None:
        card = Scorecard(scores=())
        self.assertEqual(card.average(), 0.0)

    def test_average_handles_positive_scores(self) -> None:
        card = Scorecard(scores=(1.0, 3.0, 5.0))
        self.assertAlmostEqual(card.average(), 3.0)


class ThresholdEvaluationTestCase(unittest.TestCase):
    def test_threshold_pass(self) -> None:
        passed = evaluate_threshold((0.9, 0.95, 1.0), threshold=0.9)
        self.assertTrue(passed)

    def test_threshold_fail(self) -> None:
        passed = evaluate_threshold((0.4, 0.6, 0.8), threshold=0.75)
        self.assertFalse(passed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
