"""
molecule.py
===========
Core Molecule data class for molbuilder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class Atom:
    symbol:   str
    position: np.ndarray   # Cartesian coordinates, Angstrom
    label:    str = ""     # e.g. "Fe1", "Cl1"
    charge:   float = 0.0

    def __repr__(self) -> str:
        x, y, z = self.position
        return f"Atom({self.symbol} @ [{x:.4f}, {y:.4f}, {z:.4f}])"

    def to_dict(self) -> dict:
        x, y, z = self.position.tolist()
        return {"symbol": self.symbol, "position": [x, y, z],
                "label": self.label, "charge": self.charge}

    @classmethod
    def from_dict(cls, d: dict) -> "Atom":
        return cls(
            symbol   = d["symbol"],
            position = np.array(d["position"], dtype=float),
            label    = d.get("label", ""),
            charge   = float(d.get("charge", 0.0)),
        )


@dataclass
class Molecule:
    """
    A transition-metal complex as a collection of atoms + metadata.
    """
    atoms:             List[Atom]  = field(default_factory=list)
    metal_indices:     List[int]   = field(default_factory=list)
    formula:           str         = ""
    charge:            int         = 0
    spin_multiplicity: int         = 1
    geometry:          str         = ""
    metal_symbol:      str         = ""
    metal_ox:          int         = 0
    ligand_names:      List[str]   = field(default_factory=list)

    # -- accessors ------------------------------------------------------------

    def get_symbols(self) -> List[str]:
        return [a.symbol for a in self.atoms]

    def get_positions(self) -> np.ndarray:
        return np.array([a.position for a in self.atoms])

    def num_atoms(self) -> int:
        return len(self.atoms)

    def center_of_mass(self) -> np.ndarray:
        """Simple geometric centroid (equal weights)."""
        return self.get_positions().mean(axis=0)

    def translate(self, vector: np.ndarray) -> None:
        """Translate all atoms in-place."""
        for a in self.atoms:
            a.position = a.position + np.asarray(vector)

    def center(self) -> None:
        """Move centroid to origin."""
        self.translate(-self.center_of_mass())

    # -- serialization ---------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialize the molecule to a plain Python dictionary.

        All numpy arrays are converted to plain lists so the result is
        JSON-serializable with the standard library.

        Example
        -------
        >>> d = mol.to_dict()
        >>> mol2 = Molecule.from_dict(d)
        >>> mol2.formula == mol.formula
        True
        """
        return {
            "_molbuilder_version": "2",
            "formula":           self.formula,
            "charge":            self.charge,
            "spin_multiplicity": self.spin_multiplicity,
            "geometry":          self.geometry,
            "metal_symbol":      self.metal_symbol,
            "metal_ox":          self.metal_ox,
            "metal_indices":     list(self.metal_indices),
            "ligand_names":      list(self.ligand_names),
            "atoms":             [a.to_dict() for a in self.atoms],
        }

    def to_json(self, **kwargs) -> str:
        """
        Serialize the molecule to a JSON string.

        All keyword arguments are forwarded to ``json.dumps``.
        ``indent=2`` is a useful option for human-readable output.
        """
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> "Molecule":
        """
        Reconstruct a Molecule from a dict produced by :meth:`to_dict`.

        Raises
        ------
        ValueError
            If the dict is missing required keys.
        """
        try:
            atoms = [Atom.from_dict(a) for a in d["atoms"]]
        except KeyError as exc:
            raise ValueError(f"Molecule.from_dict: missing key {exc}") from exc
        return cls(
            atoms             = atoms,
            metal_indices     = list(d.get("metal_indices", [])),
            formula           = d.get("formula", ""),
            charge            = int(d.get("charge", 0)),
            spin_multiplicity = int(d.get("spin_multiplicity", 1)),
            geometry          = d.get("geometry", ""),
            metal_symbol      = d.get("metal_symbol", ""),
            metal_ox          = int(d.get("metal_ox", 0)),
            ligand_names      = list(d.get("ligand_names", [])),
        )

    @classmethod
    def from_json(cls, s: str) -> "Molecule":
        """Reconstruct a Molecule from a JSON string."""
        return cls.from_dict(json.loads(s))

    def __repr__(self) -> str:
        return (f"Molecule({self.formula or 'unnamed'}, "
                f"{self.num_atoms()} atoms, charge={self.charge:+d})")
