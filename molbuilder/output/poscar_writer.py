"""
poscar_writer.py
================
Write a Molecule to VASP POSCAR format.

Layout:
 - Atoms centered in a cubic vacuum box (15 Å padding each side → 30 Å cell)
 - Species sorted by atomic number (heaviest first: metal → donor → H)
 - Cartesian coordinates in Ångströms
 - Formal charge and spin multiplicity in the comment line
"""

from collections import Counter, OrderedDict
from typing import List
import numpy as np

from molbuilder.core.molecule import Molecule

# Atomic numbers for sorting (heaviest first within each block)
ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29,
    "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
    "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57,
    "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64,
    "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71,
    "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
    "Rn": 86,
}

PADDING = 15.0   # Å of vacuum on each side of the molecule


def poscar_to_string(mol: Molecule, padding: float = PADDING) -> str:
    """
    Convert a Molecule to a POSCAR string.
    """
    if mol.num_atoms() == 0:
        raise ValueError("Molecule has no atoms.")

    atoms = mol.atoms
    symbols = [a.symbol for a in atoms]

    # ── sort species heaviest-first ────────────────────────────────────────
    unique_elements = list(OrderedDict.fromkeys(symbols))  # preserve first-seen order
    unique_elements.sort(key=lambda s: -ATOMIC_NUMBERS.get(s, 0))

    # reorder atoms by sorted species
    sorted_atoms = []
    for el in unique_elements:
        for a in atoms:
            if a.symbol == el:
                sorted_atoms.append(a)

    sorted_symbols = [a.symbol for a in sorted_atoms]
    sorted_positions = np.array([a.position for a in sorted_atoms])

    # ── centre in box ─────────────────────────────────────────────────────
    if len(sorted_positions) > 0:
        centroid = sorted_positions.mean(axis=0)
        sorted_positions = sorted_positions - centroid

    # ── cell: tight box + padding ──────────────────────────────────────────
    if len(sorted_positions) > 1:
        span = sorted_positions.max(axis=0) - sorted_positions.min(axis=0)
    else:
        span = np.zeros(3)
    cell = span + 2 * padding  # add padding on both sides
    cell = np.maximum(cell, 2 * padding)  # minimum cell = 2*padding

    # shift so that minimum coord is at +padding inside cell
    if len(sorted_positions) > 1:
        min_coords = sorted_positions.min(axis=0)
        sorted_positions = sorted_positions - min_coords + padding
    else:
        sorted_positions = sorted_positions + cell / 2

    # ── count species for header line ─────────────────────────────────────
    species_order = list(OrderedDict.fromkeys(sorted_symbols))
    counts = [sorted_symbols.count(el) for el in species_order]

    # ── comment line ──────────────────────────────────────────────────────
    formula = mol.formula or _make_formula(sorted_symbols)
    charge_str = f"charge={mol.charge:+d}" if mol.charge != 0 else "charge=0"
    spin_str   = f"mult={mol.spin_multiplicity}"
    comment = f"{formula}  {charge_str}  {spin_str}  (molbuilder)"

    # ── assemble POSCAR ───────────────────────────────────────────────────
    lines = []
    lines.append(comment)
    lines.append("1.0")                                        # scale factor
    lines.append(f"  {cell[0]:16.10f}  {0:16.10f}  {0:16.10f}")
    lines.append(f"  {0:16.10f}  {cell[1]:16.10f}  {0:16.10f}")
    lines.append(f"  {0:16.10f}  {0:16.10f}  {cell[2]:16.10f}")
    lines.append("  " + "  ".join(species_order))             # species names
    lines.append("  " + "  ".join(str(c) for c in counts))   # species counts
    lines.append("Cartesian")
    for pos in sorted_positions:
        lines.append(f"  {pos[0]:16.10f}  {pos[1]:16.10f}  {pos[2]:16.10f}")

    return "\n".join(lines) + "\n"


def _make_formula(symbols: List[str]) -> str:
    """Hill-system empirical formula."""
    counter = Counter(symbols)
    parts = []
    # Carbon first, then hydrogen, then alphabetical
    for el in ["C", "H"]:
        if el in counter:
            n = counter.pop(el)
            parts.append(el if n == 1 else f"{el}{n}")
    for el in sorted(counter):
        n = counter[el]
        parts.append(el if n == 1 else f"{el}{n}")
    return "".join(parts)
