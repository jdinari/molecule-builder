"""molbuilder - transition metal complex builder -> POSCAR / XYZ (VASP)"""

__version__ = "0.1.0"

from molbuilder.api import (
    build, build_isomers, dimer, trimer, poscar, xyz, info,
    load_ligand_from_poscar, CustomLigand,
)
from molbuilder.combinatorics import (
    enumerate_complexes,
    enumerate_monomers,
    enumerate_dimers,
    enumerate_trimers,
    enumerate_heteroleptic_dimers,
    enumerate_heteroleptic_trimers,
    MULTI_BRIDGE_CASES,
    combo_label,
)
from molbuilder.output.writer import write_all, write_poscar, write_xyz, write_csv, write_json
from molbuilder.relaxation import (
    relax, compute_energy, compute_gibbs, thermochemistry,
    compare_backends, check_bonds_intact,
    RelaxResult, ThermResult,
)
from molbuilder.energetics import (
    run_energetics, molecule_name, BondStatus, write_broken_report,
)
from molbuilder.reactions import ReactionNetwork, ReactionType
from molbuilder.graph import (
    canonical_hash, MolGraph, deduplicate, DeduplicationResult,
)
from molbuilder.exceptions import (
    MolbuilderError,
    InvalidLigandError,
    GeometryError,
    ClashError,
    CoordinationError,
    ChargeError,
    ValidationError,
)
from molbuilder.ligands.models import Ligand
from molbuilder.ligands.library import get_ligand_obj


__all__ = [
    # single-structure builders
    "build", "build_isomers", "dimer", "trimer",
    "poscar", "xyz", "info",
    "load_ligand_from_poscar", "CustomLigand",
    # combinatorial enumeration
    "enumerate_complexes",
    "enumerate_monomers",
    "enumerate_dimers",
    "enumerate_trimers",
    "enumerate_heteroleptic_dimers",
    "enumerate_heteroleptic_trimers",
    "MULTI_BRIDGE_CASES",
    "combo_label",
    # I/O helpers
    "write_all", "write_poscar", "write_xyz", "write_csv", "write_json",
    # relaxation
    "relax", "compute_energy", "compute_gibbs", "thermochemistry",
    "compare_backends", "check_bonds_intact",
    "RelaxResult", "ThermResult",
    # energetics pipeline
    "run_energetics", "molecule_name", "BondStatus",
    # reactions
    "ReactionNetwork", "ReactionType",
    # graph (beta)
    "canonical_hash", "MolGraph", "deduplicate", "DeduplicationResult",
    # exceptions
    "MolbuilderError", "InvalidLigandError", "GeometryError",
    "ClashError", "CoordinationError", "ChargeError", "ValidationError",
    # ligand typed models
    "Ligand", "get_ligand_obj",
]
