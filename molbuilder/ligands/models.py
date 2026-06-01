"""
models.py
=========
Typed data model for ligands.

All ligand data flows through the ``Ligand`` dataclass internally.
The string-based API (``"HCOO:bi"``, ``"mu-HCOO"``) is still supported at
the public layer and immediately converted to a ``Ligand`` object.

Usage
-----
    from molbuilder.ligands.models import Ligand
    lig = Ligand.from_name("HCOO:bi")
    lig.charge       # -1
    lig.denticity    # 2
    lig.is_bridging  # False
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # Ligand is self-referential; annotations are strings under PEP 563
from typing import Optional, Tuple


@dataclass(frozen=True)
class Ligand:
    """
    Immutable, typed representation of a coordination ligand.

    Attributes
    ----------
    name         : Canonical name as stored in the library (e.g. "HCOO:bi").
    smiles       : SMILES string (empty for simple donors).
    charge       : Formal charge of the free ligand.
    denticity    : Number of donor atoms that bond to a single metal centre.
    donor_atoms  : Element symbols of the donor atom(s), e.g. ("O",) or ("N","N").
    donor_indices: Atom indices within the SMILES that are donors.
    bite_angle   : Preferred O-M-O (or N-M-N) angle for bidentate ligands, degrees.
                   None for monodentate.
    is_bridging  : True if the ligand is designed for bridging mode (mu-X).
    """

    name:          str
    smiles:        str
    charge:        int
    denticity:     int
    donor_atoms:   Tuple[str, ...]
    donor_indices: Tuple[int, ...]
    bite_angle:    Optional[float] = None
    is_bridging:   bool = False

    # -- convenience properties ------------------------------------------------

    @property
    def primary_donor(self) -> str:
        """Element symbol of the first (or only) donor atom."""
        return self.donor_atoms[0] if self.donor_atoms else "C"

    @property
    def vectors_count(self) -> int:
        """Number of coordination vectors consumed at a single metal centre."""
        return self.denticity

    # -- construction helpers --------------------------------------------------

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Ligand":
        """Build a Ligand from a raw library dictionary entry."""
        return cls(
            name          = name,
            smiles        = d.get("smiles", ""),
            charge        = d["charge"],
            denticity     = d["denticity"],
            donor_atoms   = tuple(d.get("donor_atoms", ["C"])),
            donor_indices = tuple(d.get("donors", [0])),
            bite_angle    = d.get("bite_angle", None),
            is_bridging   = bool(d.get("is_bridging", False)),
        )

    @classmethod
    def from_name(cls, name: str) -> "Ligand":
        """
        Look up a ligand by name (including colon-mode suffixes and mu- prefix).

        Raises
        ------
        InvalidLigandError
            If *name* is not found in the library and does not look like a SMILES.
        """
        from molbuilder.ligands.library import get_ligand
        from molbuilder.exceptions import InvalidLigandError
        try:
            d = get_ligand(name)
            # get_ligand returns a plain dict; build from it
            return cls.from_dict(d.get("_canonical_name", name), d)
        except KeyError as exc:
            raise InvalidLigandError(
                f"Unknown ligand '{name}'. "
                "Check molbuilder.ligands.library or pass a SMILES string directly."
            ) from exc

    # -- serialization ---------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "name":          self.name,
            "smiles":        self.smiles,
            "charge":        self.charge,
            "denticity":     self.denticity,
            "donor_atoms":   list(self.donor_atoms),
            "donor_indices": list(self.donor_indices),
            "bite_angle":    self.bite_angle,
            "is_bridging":   self.is_bridging,
        }

    @classmethod
    def from_dict_serialized(cls, d: dict) -> "Ligand":
        """Reconstruct from a dict produced by ``to_dict()``."""
        return cls(
            name          = d["name"],
            smiles        = d.get("smiles", ""),
            charge        = d["charge"],
            denticity     = d["denticity"],
            donor_atoms   = tuple(d.get("donor_atoms", ["C"])),
            donor_indices = tuple(d.get("donor_indices", [0])),
            bite_angle    = d.get("bite_angle"),
            is_bridging   = bool(d.get("is_bridging", False)),
        )

    def __repr__(self) -> str:
        mode = "bridge" if self.is_bridging else f"d{self.denticity}"
        return f"Ligand({self.name!r}, charge={self.charge:+d}, {mode})"
