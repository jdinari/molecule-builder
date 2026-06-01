"""
Tutorial 02 -- Isomers and bidentate ligands
============================================

Covers:
  1. Automatic isomer enumeration (fac/mer, cis/trans)
  2. Bidentate chelating ligands (HCOO:bi, en, bpy)
  3. Looping over mixed-ligand sets
  4. How to work with the list/single return value safely

Run:
    python tutorials/02_isomers_and_bidentate.py
"""

from pathlib import Path
from molbuilder.api import build, poscar, info

OUT = Path("out_tutorial02")
OUT.mkdir(exist_ok=True)


# -- 1. fac and mer -----------------------------------------------------------
#
# [FeCl3(H2O)3]: two isomers.
#   fac -- all three Cl on one face of the octahedron
#   mer -- three Cl in a row around the equator

mols = build("Fe", ox=3, ligands=["Cl", "Cl", "Cl", "H2O", "H2O", "H2O"])
print(f"[FeCl3(H2O)3]  -> {len(mols)} isomers")
for mol in mols:
    poscar(mol, OUT / f"FeCl3_H2O3_{mol.label}.POSCAR")
    print(f"  {mol.label:6s}  {mol.formula}  {mol.num_atoms()} atoms")


# -- 2. cis and trans ---------------------------------------------------------
#
# [PdCl2(NH3)2] square planar: the classic cis-platin / trans-platin pair.
# All H atoms in NH3 now point correctly away from the metal.

mols = build("Pd", ox=2, ligands=["Cl", "Cl", "NH3", "NH3"], geometry="sqp")
print(f"\n[PdCl2(NH3)2] sqp  -> {len(mols)} isomers")
for mol in mols:
    poscar(mol, OUT / f"PdCl2_NH3_2_{mol.label}.POSCAR")
    print(f"  {mol.label:6s}  {mol.formula}")


# -- 3. Bidentate chelating formate -------------------------------------------
#
# HCOO:bi = bidentate formate, kappa2O,O' -- both carboxylate oxygens coordinate
# the metal.  The two Ni-O bonds form a 4-membered chelate ring with an
# O-Ni-O bite angle of ~55deg.

mol = build("Ni", ox=2, ligands=["HCOO:bi", "HCOO:bi", "H2O", "H2O"], geometry="sqp")
mols = mol if isinstance(mol, list) else [mol]
print(f"\n[Ni(HCOO:bi)2(H2O)2] sqp  -> {len(mols)} isomer(s)")
for m in mols:
    poscar(m, OUT / f"Ni_HCOObi2_H2O2_{m.label}.POSCAR")
    print(f"  {m.label:6s}  {m.formula}  charge={m.charge:+d}")


# -- 4. Other bidentate ligands ------------------------------------------------

bidentate_examples = [
    ("Ni", 2, ["en",  "en",  "en"],       "oct"),   # ethylenediamine
    ("Fe", 2, ["bpy", "bpy", "bpy"],      "oct"),   # bipyridine (tris)
    ("Ni", 2, ["HCOO:bi", "H2O", "H2O"], "sqpy"),  # bidentate + 2 water, CN=5
]

print("\nBidentate examples:")
for metal, ox, ligs, geom in bidentate_examples:
    result = build(metal, ox=ox, ligands=ligs, geometry=geom)
    n = len(result) if isinstance(result, list) else 1
    m0 = result[0] if isinstance(result, list) else result
    label = f"[{metal}({'+'.join(sorted(set(ligs)))})]"
    poscar(m0, OUT / f"{metal}_{geom}_{'_'.join(sorted(set(ligs)))}.POSCAR")
    print(f"  {label:30s}  {n} isomer(s)  {m0.formula}  CN={len(ligs) if not any(':bi' in l for l in ligs) else sum(2 if ':bi' in l else 1 for l in ligs)}")


# -- 5. Looping over a ligand pool ---------------------------------------------

print("\nNi(II) octahedral sweep -- mixed formate + water:")

ligand_sets = [
    ["HCOO", "HCOO", "H2O", "H2O", "H2O", "H2O"],
    ["HCOO", "HCOO", "HCOO", "H2O", "H2O", "H2O"],
    ["OH",   "HCOO", "H2O", "H2O", "H2O", "H2O"],
    ["HCOO:bi", "HCOO", "H2O", "H2O", "H2O"],   # CN=6 with bidentate
]

for ligs in ligand_sets:
    result = build("Ni", ox=2, ligands=ligs)
    mols_list = result if isinstance(result, list) else [result]
    for m in mols_list:
        tag = "_".join(sorted(set(l.replace(":", "") for l in ligs)))
        poscar(m, OUT / f"Ni_{tag}_{m.label}.POSCAR")
    count_str = f"{len(mols_list)} isomer(s)"
    print(f"  {str(ligs):55s}  {count_str}")

print(f"\nAll POSCARs written to {OUT}/")
