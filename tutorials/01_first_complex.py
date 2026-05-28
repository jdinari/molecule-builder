"""
Tutorial 01 — Building and writing your first complex
======================================================

Covers:
  1. Building a Ni(II) hexaaqua complex
  2. Printing a structure summary
  3. Writing POSCAR and XYZ files
  4. Choosing a geometry explicitly
  5. What .charge, .formula, .spin_multiplicity mean

Run:
    python tutorials/01_first_complex.py
"""

from pathlib import Path
from molbuilder.api import build, poscar, xyz, info

OUT = Path("out_tutorial01")
OUT.mkdir(exist_ok=True)

# ── 1. The simplest possible complex ─────────────────────────────────────────
#
# build() returns either a single Molecule or a list of Molecules.
# It returns a list only when more than one symmetry-distinct isomer exists.
# [Ni(H2O)6]²⁺ has a single arrangement, so we get one Molecule back.

mol = build("Ni", ox=2, ligands=["H2O"] * 6)

print("=== [Ni(H2O)6]²⁺ ===")
info(mol)
# info() prints: formula, charge, spin multiplicity, geometry, atom list

print(f"\nFormula          : {mol.formula}")
print(f"Charge           : {mol.charge:+d}")
print(f"Spin multiplicity: {mol.spin_multiplicity}  (2S+1, unpaired e⁻ = {mol.spin_multiplicity-1})")
print(f"Num atoms        : {mol.num_atoms()}")


# ── 2. Write files ────────────────────────────────────────────────────────────

poscar(mol, OUT / "Ni_H2O6.POSCAR")
xyz(mol,   OUT / "Ni_H2O6.xyz")
print(f"\nWrote {OUT}/Ni_H2O6.POSCAR  and  .xyz")


# ── 3. Explicit geometry ──────────────────────────────────────────────────────
#
# By default the geometry is inferred from the coordination number (CN=6 → oct).
# You can override it explicitly.  Supported geometries: lin, tp, tet, sqp,
# tbp, sqpy, oct, pbp — run  molbuilder --list-geometries  for the full list.

mol_sqp = build("Pd", ox=2, ligands=["Cl", "Cl", "NH3", "NH3"], geometry="sqp")
# sqp = square planar; two isomers exist (cis and trans) → list is returned
print(f"\n[PdCl2(NH3)2] sqp → {len(mol_sqp)} isomers")
for m in mol_sqp:
    poscar(m, OUT / f"PdCl2_NH3_2_{m.label}.POSCAR")
    print(f"  {m.label:6s}  {m.formula}  {m.num_atoms()} atoms")


# ── 4. Different metals and oxidation states ──────────────────────────────────

for spec in [
    ("Fe", 3, ["Cl"]*3 + ["H2O"]*3),     # fac/mer isomers
    ("Co", 3, ["NH3"]*6),                  # single isomer
    ("Ru", 2, ["NH3"]*3 + ["Cl"]*3),       # two isomers
]:
    metal, ox, ligs = spec
    result = build(metal, ox=ox, ligands=ligs)
    n = len(result) if isinstance(result, list) else 1
    m0 = result[0] if isinstance(result, list) else result
    print(f"[{metal}({'/'.join(sorted(set(ligs)))})]  {n} isomer(s)  formula={m0.formula}  charge={m0.charge:+d}")

print(f"\nAll files written to {OUT}/")
