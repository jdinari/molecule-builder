"""
build_single.py -- build one complex and write a POSCAR
"""
from molbuilder.api import build, poscar, info

mol = build("Ni", ox=2, ligands=["H2O"] * 6)
info(mol)
poscar(mol, "Ni_H2O6.POSCAR")
