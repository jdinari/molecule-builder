"""molbuilder – transition metal complex builder → POSCAR / XYZ (VASP)"""

__version__ = "2.1.0"

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
    MULTI_BRIDGE_CASES,
    combo_label,
)
from molbuilder.output.writer import write_all, write_poscar, write_xyz, write_csv
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
    "MULTI_BRIDGE_CASES",
    "combo_label",
    # I/O helpers
    "write_all", "write_poscar", "write_xyz", "write_csv",
    # exceptions
    "MolbuilderError", "InvalidLigandError", "GeometryError",
    "ClashError", "CoordinationError", "ChargeError", "ValidationError",
    # ligand typed models
    "Ligand", "get_ligand_obj",
]
