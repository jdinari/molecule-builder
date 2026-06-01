"""
cisplatin.py -- cis and trans isomers of [PdCl2(NH3)2]
"""
from molbuilder.api import build, poscar

mols = build("Pd", ox=2, ligands=["Cl", "Cl", "NH3", "NH3"], geometry="sqp")
for mol in mols:
    print(f"{mol.label}: {mol.formula}  (cisplatin is the cis isomer)")
    poscar(mol, f"PdCl2_NH3_2_{mol.label}.POSCAR")
