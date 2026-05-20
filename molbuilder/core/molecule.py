"""
molecule.py
===========
Core Molecule data class for molbuilder.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Atom:
    symbol: str
    position: np.ndarray  # Cartesian coordinates, Å
    label: str = ""       # e.g. "Fe1", "Cl1"
    charge: float = 0.0

    def __repr__(self):
        x, y, z = self.position
        return f"Atom({self.symbol} @ [{x:.4f}, {y:.4f}, {z:.4f}])"


@dataclass
class Molecule:
    """
    A transition-metal complex as a collection of atoms + metadata.
    """
    atoms: List[Atom] = field(default_factory=list)
    metal_indices: List[int] = field(default_factory=list)  # indices into self.atoms
    formula: str = ""
    charge: int = 0
    spin_multiplicity: int = 1
    geometry: str = ""
    metal_symbol: str = ""
    metal_ox: int = 0
    ligand_names: List[str] = field(default_factory=list)

    # ── accessors ────────────────────────────────────────────────────────────
    def get_symbols(self) -> List[str]:
        return [a.symbol for a in self.atoms]

    def get_positions(self) -> np.ndarray:
        return np.array([a.position for a in self.atoms])

    def num_atoms(self) -> int:
        return len(self.atoms)

    def center_of_mass(self) -> np.ndarray:
        """Simple geometric centroid (equal weights)."""
        return self.get_positions().mean(axis=0)

    def translate(self, vector: np.ndarray):
        """Translate all atoms in-place."""
        for a in self.atoms:
            a.position = a.position + np.asarray(vector)

    def center(self):
        """Move centroid to origin."""
        c = self.center_of_mass()
        self.translate(-c)

    def __repr__(self):
        return (f"Molecule({self.formula or 'unnamed'}, "
                f"{self.num_atoms()} atoms, charge={self.charge})")
