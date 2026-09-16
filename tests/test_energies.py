"""Unit tests for pure UMA catalysis energy calculations."""

import unittest

from uma_catalysis.calculations.energies import (
    EV_PER_ANGSTROM_SQUARED_TO_J_PER_M2,
    adsorption_energy,
    apply_zero_point_correction,
    linear_fit,
    reaction_energy,
    surface_energy_from_intercept,
)
from uma_catalysis.structs import EnergyComponents


class EnergyCalculationTests(unittest.TestCase):
    """Validate notebook energy conventions without loading scientific models."""

    def test_surface_energy_uses_two_exposed_slab_faces(self) -> None:
        """Divide the linear-fit intercept by twice the slab face area."""
        energy_ev, energy_si = surface_energy_from_intercept(4.0, 20.0)

        self.assertEqual(energy_ev, 0.1)
        self.assertAlmostEqual(
            energy_si, 0.1 * EV_PER_ANGSTROM_SQUARED_TO_J_PER_M2
        )

    def test_adsorption_and_reaction_energies_preserve_notebook_signs(self) -> None:
        """Use final-minus-initial and slab-plus-adsorbate conventions."""
        adsorbed = EnergyComponents(-10.0, -0.5)
        clean = EnergyComponents(-8.0, -0.25)
        reference = EnergyComponents(-2.0, -0.1)

        calculated_adsorption = adsorption_energy(adsorbed, clean, reference, 0.5)
        calculated_reaction = reaction_energy(clean, adsorbed)

        self.assertAlmostEqual(calculated_adsorption, -1.2)
        self.assertAlmostEqual(calculated_reaction, -2.25)
        self.assertAlmostEqual(apply_zero_point_correction(-1.2, 0.05), -1.15)

    def test_invalid_energy_inputs_are_rejected(self) -> None:
        """Reject non-physical areas and reference multipliers."""
        energy = EnergyComponents(-1.0)

        with self.assertRaises(ValueError):
            surface_energy_from_intercept(1.0, 0.0)
        with self.assertRaises(ValueError):
            adsorption_energy(energy, energy, energy, 0.0)

    def test_linear_fit_recovers_a_known_slab_energy_relation(self) -> None:
        """Fit the slope and intercept used by the surface-energy workflow."""
        slope, intercept = linear_fit((4, 6, 8), (-10.0, -14.0, -18.0))

        self.assertEqual(slope, -2.0)
        self.assertEqual(intercept, -2.0)
