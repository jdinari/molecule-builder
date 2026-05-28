"""
Tutorial 03 — Dimers and trimers with bridging ligands
=======================================================

This tutorial covers:
  1. Di-μ-hydroxo and di-μ-formate Ni(II) dimers
  2. Linear and triangular trimers
  3. Metal–metal bonded dimers (Re quadruple bond example)

Run this script with:
    python tutorials/03_dimers_and_trimers.py
"""

from pathlib import Path
from molbuilder.api import dimer, trimer, poscar, info

OUT = Path("poscar_tutorial03")
OUT.mkdir(exist_ok=True)


# ── 1. Di-μ-hydroxo Ni(II) dimer ─────────────────────────────────────────────
#
# [Ni2(μ-OH)2(H2O)8]²⁺
# Each Ni is octahedral: 4 terminal H2O + 2 bridging OH

mol = dimer("Ni", ox=2, terminal=["H2O", "H2O", "H2O", "H2O"], bridge="mu-OH", n=2)
poscar(mol, OUT / "Ni2_muOH2_H2O8.POSCAR")
print("Ni2 di-μ-OH dimer:")
info(mol)


# ── 2. Di-μ-formate dimer ────────────────────────────────────────────────────
#
# [Ni2(μ-HCOO)2(H2O)6]

mol = dimer("Ni", ox=2, terminal=["H2O", "H2O", "H2O"], bridge="mu-HCOO", n=2)
poscar(mol, OUT / "Ni2_muHCOO2_H2O6.POSCAR")
print("\nNi2 di-μ-HCOO dimer:")
info(mol)


# ── 3. Mixed terminal ligands ─────────────────────────────────────────────────

mol = dimer("Ni", ox=2, terminal=["HCOO", "H2O", "H2O"], bridge="mu-OH", n=2)
poscar(mol, OUT / "Ni2_muOH2_HCOO2_H2O4.POSCAR")
print("\nNi2 mixed terminal (HCOO + H2O):")
info(mol)


# ── 4. Linear Fe3 trimer ─────────────────────────────────────────────────────
#
# Three Fe(III) centres in a row, bridged by μ-OH

mol = trimer("Fe", ox=3, terminal=["H2O", "H2O"], bridge="mu-OH", arrangement="linear")
poscar(mol, OUT / "Fe3_linear_muOH.POSCAR")
print("\nFe3 linear trimer:")
info(mol)


# ── 5. Triangular Ru3 carbonyl cluster ───────────────────────────────────────

mol = trimer("Ru", ox=0, terminal=["CO", "CO", "CO", "CO"], bridge="mu-CO",
             arrangement="triangular")
poscar(mol, OUT / "Ru3_triangular_CO12.POSCAR")
print("\nRu3 triangular carbonyl cluster:")
info(mol)


# ── 6. Metal–metal bonded Re dimer ───────────────────────────────────────────
#
# Rhenium(III) quadruple bond: Re≡Re distance ~2.22 Å

mol = dimer("Re", ox=3, terminal=["Cl", "Cl", "Cl"], bridge="mu-Cl", n=2,
            mm_bond=True, mm_distance=2.22)
poscar(mol, OUT / "Re2_quadruple_bond.POSCAR")
print("\nRe2 quadruple bond:")
info(mol)


print(f"\nAll POSCARs written to {OUT}/")
