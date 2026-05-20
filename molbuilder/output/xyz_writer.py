"""
xyz_writer.py
=============
Write a Molecule to XYZ format.
"""

from molbuilder.core.molecule import Molecule


def xyz_to_string(mol: Molecule) -> str:
    lines = [str(mol.num_atoms()), mol.formula or "molbuilder"]
    for a in mol.atoms:
        x, y, z = a.position
        lines.append(f"{a.symbol:3s}  {x:14.8f}  {y:14.8f}  {z:14.8f}")
    return "\n".join(lines) + "\n"
