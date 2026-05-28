"""
Tutorial 03 — Dinuclear and trinuclear complexes
=================================================

Covers:
  1. Di-μ-hydroxo and di-μ-formate Ni(II) dimers
  2. Heteroleptic dimers (different terminals on each metal)
  3. Linear and triangular trinuclear complexes
  4. Triangular Ni3 with double bridges and terminal water
  5. Paddle-wheel Ni2(HCOO)4 (short Ni-Ni contact)

Structures that did NOT work at the time of writing and why:
  • dimer(..., mm_bond=True) with terminal halides — terminal placement
    along the M-M axis produces bond-length errors.  Use mm_bond for
    bare dimers (no terminals) or open this as a GitHub issue.
  • Ru3(μ-CO) triangular trimers — bridging CO on a triangular Ru3
    cluster causes severe C-C clashes at ideal geometry.  Use
    octahedral Ni/Fe trimers instead.

Run:
    python tutorials/03_dimers_and_trimers.py
"""

from pathlib import Path
from molbuilder.api import dimer, trimer, poscar, info

OUT = Path("out_tutorial03")
OUT.mkdir(exist_ok=True)


# ── 1. Di-μ-hydroxo Ni(II) dimer ─────────────────────────────────────────────
#
# [Ni2(μ-OH)2(H2O)4]²⁺
# CN = 4 per Ni: 2 terminal H2O + 2 bridging OH
# Charge: 2*Ni(II) = +4, 2*OH⁻ = −2  →  total = +2

mol = dimer("Ni", ox=2, terminal=["H2O", "H2O"], bridge="mu-OH", n=2)
poscar(mol, OUT / "Ni2_muOH2_H2O4.POSCAR")
print("Ni2(μ-OH)2(H2O)4:")
info(mol)


# ── 2. Di-μ-formate dimer ────────────────────────────────────────────────────
#
# [Ni2(μ-HCOO)2(H2O)6]
# CN = 5 per Ni: 3 terminal H2O + 2 bridging formate
# Charge: 2*Ni(II) = +4, 2*HCOO⁻ = −2, 6*H2O = 0  →  neutral

mol = dimer("Ni", ox=2, terminal=["H2O", "H2O", "H2O"], bridge="mu-HCOO", n=2)
poscar(mol, OUT / "Ni2_muHCOO2_H2O6.POSCAR")
print("\nNi2(μ-HCOO)2(H2O)6:")
info(mol)


# ── 3. Paddle-wheel: Ni2(HCOO)4 ─────────────────────────────────────────────
#
# 4 bridging formates, no terminals.  Short Ni-Ni distance (~3.98 Å).
# This is a common secondary building unit in metal-organic frameworks.
# Charge: 2*Ni(II) = +4, 4*HCOO⁻ = −4  →  neutral

import numpy as np

mol = dimer("Ni", ox=2, terminal=[], bridge="mu-HCOO", n=4)
ni_pos = [a.position for a in mol.atoms if a.symbol == "Ni"]
ni_ni = float(np.linalg.norm(ni_pos[0] - ni_pos[1]))
poscar(mol, OUT / "Ni2_muHCOO4_paddlewheel.POSCAR")
print(f"\nNi2(μ-HCOO)4 paddle-wheel:  Ni–Ni = {ni_ni:.3f} Å  charge={mol.charge}")


# ── 4. Heteroleptic dimer: asymmetric terminals ───────────────────────────────
#
# [Ni2(μ-HCOO)4(H2O)] — water on one Ni only, the other is bare.
# terminal_m1 / terminal_m2 set independent ligand lists per metal centre.
# Charge: 2*Ni(II) = +4, 4*HCOO⁻ = −4, H2O = 0  →  neutral

mol = dimer("Ni", ox=2,
            terminal_m1=["H2O"], terminal_m2=[],
            bridge="mu-HCOO", n=4)
poscar(mol, OUT / "Ni2_muHCOO4_H2O_hetero.POSCAR")
print(f"\nNi2(μ-HCOO)4(H2O) heteroleptic:  {mol.formula}  charge={mol.charge}")
cn_each = []
for ni in [a for a in mol.atoms if a.symbol == "Ni"]:
    donors = [a for a in mol.atoms if a.symbol == "O"
              and float(np.linalg.norm(a.position - ni.position)) < 2.5]
    cn_each.append(len(donors))
print(f"  CN per Ni: {sorted(cn_each)}  (4 = bare, 5 = with H2O)")


# ── 5. Linear Fe(III) trimer ─────────────────────────────────────────────────
#
# Fe–O–Fe–O–Fe linear chain, bridging OH.
# Charge: 3*Fe(III) = +9, 4*OH⁻ = −4, 6*H2O = 0  →  total +5

mol = trimer("Fe", ox=3, terminal=["H2O", "H2O"], bridge="mu-OH",
             arrangement="linear")
poscar(mol, OUT / "Fe3_linear_muOH.POSCAR")
print(f"\nFe3 linear trimer:  {mol.formula}  charge={mol.charge}")
info(mol)


# ── 6. Triangular Ni3(μ-HCOO)6 ───────────────────────────────────────────────
#
# The triangular double-bridge trimer: 2 syn-syn formate bridges per edge.
# All 3 Ni are CN=4.  Charge: 3*Ni(II) = +6, 6*HCOO⁻ = −6  →  neutral.
# The ±35° tilt scheme ensures all O···O distances > 2.2 Å.

mol = trimer("Ni", ox=2, terminal=[], bridge="mu-HCOO",
             arrangement="triangular", n_bridges_per_pair=2)
poscar(mol, OUT / "Ni3_triangular_muHCOO6.POSCAR")
ni_positions = [a.position for a in mol.atoms if a.symbol == "Ni"]
ni_ni_dists = sorted([
    float(np.linalg.norm(ni_positions[i] - ni_positions[j]))
    for i in range(3) for j in range(i + 1, 3)
])
print(f"\nNi3(μ-HCOO)6 triangular:  {mol.formula}  charge={mol.charge}")
print(f"  Ni–Ni distances: {[round(d, 3) for d in ni_ni_dists]} Å  (equilateral)")


# ── 7. Heteroleptic Ni3 — water on one metal only ────────────────────────────
#
# [Ni3(μ-HCOO)6(H2O)]: H2O on metal-0 only, metals 1 and 2 are bare.
# terminals_per_metal accepts a list of 3 terminal-ligand lists, one per metal.

mol = trimer("Ni", ox=2,
             bridge="mu-HCOO",
             arrangement="triangular",
             n_bridges_per_pair=2,
             terminals_per_metal=[["H2O"], [], []])
poscar(mol, OUT / "Ni3_triangular_muHCOO6_H2O_hetero.POSCAR")
print(f"\nNi3(μ-HCOO)6 + H2O on Ni0:  {mol.formula}  charge={mol.charge}")
for i, ni in enumerate([a for a in mol.atoms if a.symbol == "Ni"]):
    donors = [a for a in mol.atoms if a.symbol == "O"
              and float(np.linalg.norm(a.position - ni.position)) < 2.5]
    print(f"  Ni{i}: CN={len(donors)}")

print(f"\nAll POSCARs written to {OUT}/")
