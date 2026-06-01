"""
Tutorial 02 -- Isomer enumeration and batch POSCAR generation
=============================================================

This tutorial covers:
  1. Automatic isomer generation for octahedral and square-planar complexes
  2. Writing one POSCAR per isomer
  3. Iterating over mixed-ligand sets

Run this script with:
    python tutorials/02_isomers_and_batch.py
"""

from pathlib import Path
from molbuilder.api import build, poscar, info

OUT = Path("poscar_tutorial02")
OUT.mkdir(exist_ok=True)


# -- 1. Single-isomer complex --------------------------------------------------

mol = build("Ni", ox=2, ligands=["H2O"] * 6)
print(f"[Ni(H2O)6]2+  ->  {type(mol).__name__}")  # single Molecule

poscar(mol, OUT / "Ni_H2O6.POSCAR")


# -- 2. Two isomers: fac and mer -----------------------------------------------

mols = build("Fe", ox=3, ligands=["Cl", "Cl", "Cl", "H2O", "H2O", "H2O"])
print(f"\n[FeCl3(H2O)3]  ->  {len(mols)} isomers")

for mol in mols:
    filename = OUT / f"FeCl3_H2O3_{mol.label}.POSCAR"
    poscar(mol, filename)
    print(f"  {mol.label:6s}  ->  {filename.name}")


# -- 3. cis / trans: square planar Pd(II) -------------------------------------

mols = build("Pd", ox=2, ligands=["Cl", "Cl", "NH3", "NH3"], geometry="sqp")
mols = mols if isinstance(mols, list) else [mols]
print(f"\n[PdCl2(NH3)2] sqp  ->  {len(mols)} isomers")

for mol in mols:
    filename = OUT / f"PdCl2_NH3_2_{mol.label}.POSCAR"
    poscar(mol, filename)
    print(f"  {mol.label:6s}  ->  {filename.name}")


# -- 4. Bidentate chelating formate -------------------------------------------

mol = build("Ni", ox=2, ligands=["HCOO:bi", "HCOO:bi", "H2O", "H2O"], geometry="sqp")
mols = mol if isinstance(mol, list) else [mol]
for mol in mols:
    poscar(mol, OUT / f"Ni_HCOObi2_H2O2_{mol.label}.POSCAR")

print(f"\n[Ni(HCOO:bi)2(H2O)2] sqp  ->  {len(mols)} isomers written")


# -- 5. Looping over a ligand pool ---------------------------------------------

print("\nMixed-ligand Ni(II) octahedral sweep:")

ligand_sets = [
    ["HCOO", "HCOO", "H2O", "H2O", "H2O", "H2O"],
    ["HCOO", "HCOO", "HCOO", "H2O", "H2O", "H2O"],
    ["OH",   "HCOO", "H2O", "H2O", "H2O", "H2O"],
]

for ligs in ligand_sets:
    mols = build("Ni", ox=2, ligands=ligs)
    mols = mols if isinstance(mols, list) else [mols]
    label_str = "_".join(sorted(set(ligs)))
    for mol in mols:
        fname = OUT / f"Ni_{label_str}_{mol.label}.POSCAR"
        poscar(mol, fname)
    print(f"  {ligs}  ->  {len(mols)} isomer(s)")

print(f"\nAll POSCARs written to {OUT}/")
