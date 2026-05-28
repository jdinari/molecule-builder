"""
xtb_delta_g.py — compute ΔG for a ligand substitution reaction with xTB

Reaction:  [Ni(H2O)6] + HCOO⁻  →  [Ni(HCOO)(H2O)5] + H2O

Install:  pip install tblite ase
"""
import numpy as np
from molbuilder.api import build
from molbuilder.core.molecule import Molecule, Atom
from molbuilder.relaxation import thermochemistry

T = 298.15   # K
P = 101325   # Pa

# ── Free-molecule references ──────────────────────────────────────────────────

def make_h2o():
    ah = np.radians(104.5 / 2);  oh = 0.958
    return Molecule(
        atoms=[Atom("O", np.zeros(3)),
               Atom("H", np.array([ oh*np.sin(ah), oh*np.cos(ah), 0.])),
               Atom("H", np.array([-oh*np.sin(ah), oh*np.cos(ah), 0.]))],
        formula="H2O", charge=0, spin_multiplicity=1, metal_symbol="", metal_ox=0,
    )

def make_formate():
    ah = np.radians(126.0 / 2);  co = 1.25;  ch = 1.09
    return Molecule(
        atoms=[Atom("C", np.zeros(3)),
               Atom("O", np.array([ co*np.sin(ah), co*np.cos(ah), 0.])),
               Atom("O", np.array([-co*np.sin(ah), co*np.cos(ah), 0.])),
               Atom("H", np.array([0., -ch, 0.]))],
        formula="HCOO", charge=-1, spin_multiplicity=1, metal_symbol="", metal_ox=0,
    )

# ── Build species ─────────────────────────────────────────────────────────────

reactant1 = build("Ni", ox=2, ligands=["H2O"]*6)
reactant2 = make_formate()
product1  = build("Ni", ox=2, ligands=["HCOO"]+["H2O"]*5)
if isinstance(product1, list): product1 = product1[0]
product2  = make_h2o()

# ── Thermochemistry ───────────────────────────────────────────────────────────

species = {"Ni_H2O6": reactant1, "HCOO": reactant2,
           "Ni_HCOO_H2O5": product1, "H2O": product2}

results = {}
for name, mol in species.items():
    print(f"  {name} ({mol.formula}) ...", flush=True)
    results[name] = thermochemistry(mol, backend="xtb", T=T, P=P, fmax=0.05)
    print(f"    G={results[name].gibbs_eV:.4f} eV")

# ── ΔE and ΔG ─────────────────────────────────────────────────────────────────

r = results
dE = r["Ni_HCOO_H2O5"].energy_eV + r["H2O"].energy_eV \
   - r["Ni_H2O6"].energy_eV      - r["HCOO"].energy_eV
dG = r["Ni_HCOO_H2O5"].gibbs_eV  + r["H2O"].gibbs_eV  \
   - r["Ni_H2O6"].gibbs_eV       - r["HCOO"].gibbs_eV

print(f"\n[Ni(H2O)6] + HCOO⁻  →  [Ni(HCOO)(H2O)5] + H2O")
print(f"  ΔE = {dE:+.4f} eV  ({dE*23.06:+.2f} kcal/mol)")
print(f"  ΔG = {dG:+.4f} eV  ({dG*23.06:+.2f} kcal/mol)  at {T} K, {P/100:.0f} hPa")
print(f"  ΔG(350K) = {r['Ni_HCOO_H2O5'].gibbs_at(350)+r['H2O'].gibbs_at(350)-r['Ni_H2O6'].gibbs_at(350)-r['HCOO'].gibbs_at(350):+.4f} eV")
