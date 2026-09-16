"""Construct bulk, slab, gas, and adsorbate-candidate structures."""

from collections.abc import Sequence
from math import sqrt
from typing import Any

from uma_catalysis.structs.config import Facet, MaterialConfig


def build_bulk_structure(
    config: MaterialConfig, lattice_constant: float | None = None
) -> Any:
    """
    Construct the tutorial's cubic bulk crystal structure.

    Parameters
    ----------
    config : uma_catalysis.structs.config.MaterialConfig
        Element, crystal structure, and default lattice constant.
    lattice_constant : float or None, optional
        Lattice constant in angstrom. When omitted, uses
        ``config.initial_lattice_constant``.

    Returns
    -------
    Any
        ASE atoms object for the requested cubic bulk structure.

    Raises
    ------
    ValueError
        If the lattice constant is not positive.
    """
    lattice_constant = (
        config.initial_lattice_constant
        if lattice_constant is None
        else lattice_constant
    )
    if lattice_constant <= 0:
        raise ValueError("lattice_constant must be positive.")

    from ase.build import bulk

    return bulk(
        config.element,
        config.crystal_structure,
        a=lattice_constant,
        cubic=True,
    )


def build_surface_energy_slab(
    bulk_atoms: Any,
    facet: Facet,
    layer_count: int,
    lattice_constant: float,
    vacuum_size: float,
) -> Any:
    """
    Construct one Pymatgen-generated slab for a surface-energy calculation.

    Parameters
    ----------
    bulk_atoms : Any
        ASE-compatible optimized bulk structure.
    facet : Facet
        Surface Miller index.
    layer_count : int
        Approximate number of crystal layers used to size the slab.
    lattice_constant : float
        Optimized bulk lattice constant in angstrom.
    vacuum_size : float
        Vacuum spacing in angstrom.

    Returns
    -------
    Any
        ASE-compatible slab atoms object centered along its third lattice axis.

    Raises
    ------
    ValueError
        If the facet, layer count, lattice constant, or vacuum size is invalid.
    """
    if len(facet) != 3 or not any(facet):
        raise ValueError("facet must contain three Miller indices, not all zero.")
    if layer_count < 1:
        raise ValueError("layer_count must be at least 1.")
    if lattice_constant <= 0:
        raise ValueError("lattice_constant must be positive.")
    if vacuum_size <= 0:
        raise ValueError("vacuum_size must be positive.")

    from pymatgen.core.surface import SlabGenerator
    from pymatgen.io.ase import AseAtomsAdaptor

    adaptor = AseAtomsAdaptor()
    structure = adaptor.get_structure(bulk_atoms)
    minimum_slab_size = (
        layer_count * lattice_constant / sqrt(sum(index**2 for index in facet))
    )
    slab_generator = SlabGenerator(
        structure,
        facet,
        min_slab_size=minimum_slab_size,
        min_vacuum_size=vacuum_size,
        center_slab=True,
    )
    slab = adaptor.get_atoms(slab_generator.get_slabs()[0])
    slab.center(vacuum=vacuum_size, axis=2)
    return slab


def build_adsorption_slab(
    config: MaterialConfig,
    lattice_constant: float,
    facet: Facet | None = None,
) -> Any:
    """
    Construct a FAIR Chemistry slab object for adsorption workflows.

    Parameters
    ----------
    config : uma_catalysis.structs.config.MaterialConfig
        Bulk and adsorption-facet settings.
    lattice_constant : float
        Optimized bulk lattice constant in angstrom.
    facet : Facet or None, optional
        Adsorption facet. When omitted, uses ``config.adsorption_facet``.

    Returns
    -------
    Any
        FAIR Chemistry ``Slab`` object whose ``atoms`` field contains ASE atoms.
    """
    selected_facet = config.adsorption_facet if facet is None else facet
    if len(selected_facet) != 3 or not any(selected_facet):
        raise ValueError("facet must contain three Miller indices, not all zero.")

    from fairchem.data.oc.core import Bulk, Slab

    bulk_reference = Bulk(
        bulk_atoms=build_bulk_structure(config, lattice_constant=lattice_constant)
    )
    return Slab.from_bulk_get_specific_millers(
        bulk=bulk_reference,
        specific_millers=selected_facet,
    )[0]


def build_diatomic_reference(
    symbols: tuple[str, str], bond_length: float, vacuum_size: float
) -> Any:
    """
    Construct a periodic vacuum-box reference for a diatomic molecule.

    Parameters
    ----------
    symbols : tuple[str, str]
        Chemical symbols for the two atoms in bond order.
    bond_length : float
        Initial bond length in angstrom.
    vacuum_size : float
        Vacuum spacing in angstrom.

    Returns
    -------
    Any
        ASE atoms object centered in a periodic vacuum box.
    """
    if len(symbols) != 2 or not all(symbol.strip() for symbol in symbols):
        raise ValueError("symbols must contain two non-empty chemical symbols.")
    if bond_length <= 0:
        raise ValueError("bond_length must be positive.")
    if vacuum_size <= 0:
        raise ValueError("vacuum_size must be positive.")

    from ase import Atoms

    molecule = Atoms(
        "".join(symbols),
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, bond_length]],
    )
    molecule.center(vacuum=vacuum_size)
    molecule.set_pbc([True, True, True])
    return molecule


def generate_single_adsorbate_candidates(
    slab: Any, adsorbate_smiles: str, count: int
) -> list[Any]:
    """
    Generate heuristic single-adsorbate configurations for one FAIR slab.

    Parameters
    ----------
    slab : Any
        FAIR Chemistry slab object.
    adsorbate_smiles : str
        FAIR Chemistry adsorbate SMILES, including the adsorption marker.
    count : int
        Number of candidate sites to generate.

    Returns
    -------
    list[Any]
        ASE-compatible candidate atoms objects.
    """
    if not adsorbate_smiles.strip():
        raise ValueError("adsorbate_smiles must not be empty.")
    if count < 1:
        raise ValueError("count must be at least 1.")

    from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig

    adsorbate = Adsorbate(adsorbate_smiles_from_db=adsorbate_smiles)
    configurations = AdsorbateSlabConfig(
        slab,
        adsorbate,
        mode="random_site_heuristic_placement",
        num_sites=count,
    )
    return list(configurations.atoms_list)


def generate_multiple_adsorbate_candidates(
    slab: Any, adsorbate_smiles: Sequence[str], count: int
) -> list[Any]:
    """
    Generate multi-adsorbate configurations for one FAIR Chemistry slab.

    Parameters
    ----------
    slab : Any
        FAIR Chemistry slab object.
    adsorbate_smiles : collections.abc.Sequence[str]
        FAIR Chemistry adsorbate SMILES entries, one for each adsorbate.
    count : int
        Number of candidate configurations to generate.

    Returns
    -------
    list[Any]
        ASE-compatible candidate atoms objects.
    """
    if not adsorbate_smiles or not all(smiles.strip() for smiles in adsorbate_smiles):
        raise ValueError("adsorbate_smiles must contain non-empty entries.")
    if count < 1:
        raise ValueError("count must be at least 1.")

    from fairchem.data.oc.core import Adsorbate, MultipleAdsorbateSlabConfig

    adsorbates = [
        Adsorbate(adsorbate_smiles_from_db=smiles) for smiles in adsorbate_smiles
    ]
    configurations = MultipleAdsorbateSlabConfig(
        slab,
        adsorbates,
        num_configurations=count,
    )
    return list(configurations.atoms_list)
