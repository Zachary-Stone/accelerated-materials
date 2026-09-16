"""Unit tests for UMA catalysis result structures."""

import unittest

from uma_catalysis.structs import (
    AdsorptionResult,
    BulkOptimizationResult,
    EnergyComponents,
    SurfaceEnergyStudyResult,
    ReactionResult,
    SurfaceEnergyResult,
)


class ResultStructureTests(unittest.TestCase):
    """Validate numerical result objects and derived energy properties."""

    def test_energy_and_derived_result_values(self) -> None:
        """Calculate component totals, adsorption energy, and reaction energy."""
        adsorbed = EnergyComponents(-10.0, -0.5)
        clean = EnergyComponents(-8.0, -0.25)
        reference = EnergyComponents(-2.0, -0.1)
        adsorption = AdsorptionResult(
            adsorbate="H*",
            adsorbed_energy=adsorbed,
            clean_slab_energy=clean,
            reference_energy=reference,
            reference_multiplier=0.5,
            zero_point_correction=0.1,
        )
        reaction = ReactionResult(
            reaction_name="C* + O* -> CO*",
            initial_energy=EnergyComponents(-15.0, -0.2),
            final_energy=EnergyComponents(-16.0, -0.3),
            zero_point_correction=0.05,
        )

        self.assertEqual(adsorbed.total_energy, -10.5)
        self.assertAlmostEqual(adsorption.electronic_adsorption_energy, -1.2)
        self.assertAlmostEqual(adsorption.total_adsorption_energy, -1.1)
        self.assertAlmostEqual(reaction.electronic_reaction_energy, -1.1)
        self.assertAlmostEqual(reaction.total_reaction_energy, -1.05)

    def test_invalid_results_are_rejected(self) -> None:
        """Reject impossible lattice and surface-fit result data."""
        with self.assertRaises(ValueError):
            BulkOptimizationResult(3.52, 0.0, 3.524, True)
        with self.assertRaises(ValueError):
            SurfaceEnergyResult(
                facet=(1, 1, 1),
                atom_counts=(4,),
                slab_energies=(-20.0,),
                fit_slope=-5.0,
                fit_intercept=1.0,
                surface_energy_ev_per_angstrom_squared=0.1,
                surface_energy_j_per_m2=1.6,
            )

    def test_surface_study_rejects_duplicate_facet_results(self) -> None:
        """Keep one unambiguous fit result for every configured facet."""
        facet_result = SurfaceEnergyResult(
            facet=(1, 1, 1),
            atom_counts=(4, 6),
            slab_energies=(-10.0, -14.0),
            fit_slope=-2.0,
            fit_intercept=-2.0,
            surface_energy_ev_per_angstrom_squared=-0.1,
            surface_energy_j_per_m2=-1.6,
        )

        with self.assertRaises(ValueError):
            SurfaceEnergyStudyResult(
                bulk_energy_per_atom=-2.0,
                facet_results=(facet_result, facet_result),
            )
