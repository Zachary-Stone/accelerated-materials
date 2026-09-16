"""Tests for pure NEB outcome calculations."""

import unittest

from uma_catalysis.calculations.reaction_paths import NEBOutcome


class NEBOutcomeTests(unittest.TestCase):
    def test_barriers_are_relative_to_the_endpoint_energies(self) -> None:
        """Report forward and reverse barriers from a relative path profile."""
        outcome = NEBOutcome(
            images=[object(), object(), object()],
            converged=True,
            steps=10,
            relative_energies=(0.0, 1.2, -0.3),
        )

        self.assertEqual(outcome.forward_barrier, 1.2)
        self.assertEqual(outcome.reverse_barrier, 1.5)


if __name__ == "__main__":
    unittest.main()
