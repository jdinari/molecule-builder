"""
isomers.py -- enumerate all isomers and write one POSCAR each
"""
from molbuilder.api import build, poscar

mols = build("Fe", ox=3, ligands=["Cl", "Cl", "Cl", "H2O", "H2O", "H2O"])
for mol in mols:
    print(f"{mol.label}: {mol.formula}")
    poscar(mol, f"FeCl3_H2O3_{mol.label}.POSCAR")
