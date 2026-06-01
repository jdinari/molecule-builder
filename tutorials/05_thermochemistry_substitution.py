"""
Tutorial 05 — Thermochemistry: ΔG for ligand substitution
==========================================================

Computes Gibbs free energies for stepwise formate substitution at Ni(II):

    Step 1:  [Ni(H2O)6]²⁺  +  HCOO⁻  →  [Ni(HCOO)(H2O)5]⁺  +  H2O
    Step 2:  [Ni(HCOO)(H2O)5]⁺  +  HCOO⁻  →  [Ni(HCOO)2(H2O)4]  +  H2O

and compares the cis and trans isomers of the diforimate product.

Uses the xTB backend (GFN2-xTB via tblite), which handles Ni charge and spin
multiplicity explicitly and is the recommended default for Ni coordination
chemistry.

Install:
    pip install git+https://github.com/jdinari/molecule-builder.git
    pip install tblite ase

Run:
    python tutorials/05_thermochemistry_substitution.py
"""

import numpy as np
from molbuilder.api import build
from molbuilder.relaxation import thermochemistry
from molbuilder.core.molecule import Molecule, Atom

# ── Configuration ─────────────────────────────────────────────────────────────

BACKEND     = "xtb"         # "xtb" (recommended) or "mace"
MACE_MODEL  = None          # only used if BACKEND = "mace"
MACE_DEVICE = "cpu"

T = 298.15   # K
P = 101325   # Pa  (1 atm)

BACKEND_KWARGS = {}
if BACKEND == "mace":
    BACKEND_KWARGS = {"model": MACE_MODEL, "device": MACE_DEVICE}


# ── Free-molecule reference geometries ───────────────────────────────────────
#
# build() requires a metal centre, so free H2O and HCOO⁻ are built directly
# from their experimental geometries.

def make_h2o() -> Molecule:
    """Free water molecule: O-H = 0.958 Å, H-O-H = 104.5°."""
    angle_half = np.radians(104.5 / 2)
    oh = 0.958
    return Molecule(
        atoms=[
            Atom("O", np.zeros(3)),
            Atom("H", np.array([ oh * np.sin(angle_half),  oh * np.cos(angle_half), 0.])),
            Atom("H", np.array([-oh * np.sin(angle_half),  oh * np.cos(angle_half), 0.])),
        ],
        formula="H2O", charge=0, spin_multiplicity=1,
        metal_symbol="", metal_ox=0,
    )


def make_formate() -> Molecule:
    """Free formate anion (HCOO⁻): O-C-O = 126°, C-O = 1.25 Å, C-H = 1.09 Å."""
    angle_half = np.radians(126.0 / 2)
    co, ch = 1.25, 1.09
    return Molecule(
        atoms=[
            Atom("C", np.zeros(3)),
            Atom("O", np.array([ co * np.sin(angle_half),  co * np.cos(angle_half), 0.])),
            Atom("O", np.array([-co * np.sin(angle_half),  co * np.cos(angle_half), 0.])),
            Atom("H", np.array([0., -ch, 0.])),
        ],
        formula="HCOO", charge=-1, spin_multiplicity=1,
        metal_symbol="", metal_ox=0,
    )


# ── Build all species ─────────────────────────────────────────────────────────

print("Building structures ...")

Ni_H2O6      = build("Ni", ox=2, ligands=["H2O"] * 6)
Ni_HCOO_H2O5 = build("Ni", ox=2, ligands=["HCOO"] + ["H2O"] * 5)
Ni_HCOO2_all = build("Ni", ox=2, ligands=["HCOO", "HCOO", "H2O", "H2O", "H2O", "H2O"])

# Normalise single-result cases to plain Molecule objects
if isinstance(Ni_H2O6, list):      Ni_H2O6 = Ni_H2O6[0]
if isinstance(Ni_HCOO_H2O5, list): Ni_HCOO_H2O5 = Ni_HCOO_H2O5[0]
Ni_HCOO2_all = Ni_HCOO2_all if isinstance(Ni_HCOO2_all, list) else [Ni_HCOO2_all]
Ni_HCOO2 = {mol.label: mol for mol in Ni_HCOO2_all}

species = {
    "Ni_H2O6":              Ni_H2O6,
    "Ni_HCOO_H2O5":         Ni_HCOO_H2O5,
    "Ni_HCOO2_H2O4_cis":    Ni_HCOO2.get("cis"),
    "Ni_HCOO2_H2O4_trans":  Ni_HCOO2.get("trans"),
    "H2O":                  make_h2o(),
    "HCOO":                 make_formate(),
}
species = {k: v for k, v in species.items() if v is not None}
print(f"  Species to compute: {list(species.keys())}")


# ── Compute thermochemistry ───────────────────────────────────────────────────

print(f"\nRunning thermochemistry (backend={BACKEND}, T={T} K, P={P} Pa) ...")
print("Each species takes ~1–2 minutes (geometry relaxation + frequencies).\n")

results = {}
for name, mol in species.items():
    print(f"  {name} ({mol.formula}) ...", flush=True)
    try:
        r = thermochemistry(
            mol,
            backend = BACKEND,
            T       = T,
            P       = P,
            fmax    = 0.05,
            steps   = 300,
            **BACKEND_KWARGS,
        )
        results[name] = r
        print(f"    E={r.energy_eV:.4f} eV  G={r.gibbs_eV:.4f} eV  converged={r.converged}")
    except Exception as exc:
        print(f"    FAILED: {exc}")


# ── Isomer comparison: cis vs trans [Ni(HCOO)2(H2O)4] ────────────────────────

print("\n" + "=" * 58)
print("  Isomer comparison: cis vs trans [Ni(HCOO)2(H2O)4]")
print("=" * 58)

r_cis   = results.get("Ni_HCOO2_H2O4_cis")
r_trans = results.get("Ni_HCOO2_H2O4_trans")

if r_cis and r_trans:
    dE = r_trans.energy_eV - r_cis.energy_eV
    dG = r_trans.gibbs_eV  - r_cis.gibbs_eV
    print(f"  ΔE(trans − cis) = {dE:+.4f} eV  ({dE * 23.06:+.2f} kcal/mol)")
    print(f"  ΔG(trans − cis) = {dG:+.4f} eV  ({dG * 23.06:+.2f} kcal/mol)")
    preferred = "trans" if dG < 0 else "cis"
    print(f"\n  → {preferred} isomer is thermodynamically preferred at {T} K, {P/100:.0f} hPa")

    # Re-evaluate ΔG at other temperatures without re-running any calculations
    print(f"\n  ΔG(trans − cis) at other temperatures (no re-run needed):")
    for T2 in [250, 298.15, 350, 400]:
        dG2 = r_trans.gibbs_at(T=T2) - r_cis.gibbs_at(T=T2)
        print(f"    T = {T2:6.1f} K  →  ΔG = {dG2:+.4f} eV")
else:
    print("  (One or both isomers failed — check output above.)")


# ── Step 1: [Ni(H2O)6] + HCOO⁻ → [Ni(HCOO)(H2O)5] + H2O ───────────────────

print("\n" + "=" * 58)
print("  Step 1: [Ni(H2O)6] + HCOO⁻  →  [Ni(HCOO)(H2O)5] + H2O")
print("=" * 58)

needed_step1 = ["Ni_H2O6", "HCOO", "Ni_HCOO_H2O5", "H2O"]
if all(k in results for k in needed_step1):
    dE1 = (results["Ni_HCOO_H2O5"].energy_eV + results["H2O"].energy_eV
           - results["Ni_H2O6"].energy_eV     - results["HCOO"].energy_eV)
    dG1 = (results["Ni_HCOO_H2O5"].gibbs_eV  + results["H2O"].gibbs_eV
           - results["Ni_H2O6"].gibbs_eV      - results["HCOO"].gibbs_eV)
    print(f"  ΔE = {dE1:+.4f} eV  ({dE1 * 23.06:+.2f} kcal/mol)")
    print(f"  ΔG = {dG1:+.4f} eV  ({dG1 * 23.06:+.2f} kcal/mol)  at {T} K, {P/100:.0f} hPa")
else:
    missing = [k for k in needed_step1 if k not in results]
    print(f"  (Missing results for: {missing})")


# ── Step 2: [Ni(HCOO)(H2O)5] + HCOO⁻ → [Ni(HCOO)2(H2O)4] + H2O ────────────
#
# Uses the lower-energy isomer (cis or trans) as the product.

print("\n" + "=" * 58)
print("  Step 2: [Ni(HCOO)(H2O)5] + HCOO⁻  →  [Ni(HCOO)2(H2O)4] + H2O")
print("=" * 58)

product_key = None
if "Ni_HCOO2_H2O4_cis" in results and "Ni_HCOO2_H2O4_trans" in results:
    # Pick the lower-energy isomer as the thermodynamic product
    if results["Ni_HCOO2_H2O4_cis"].gibbs_eV <= results["Ni_HCOO2_H2O4_trans"].gibbs_eV:
        product_key = "Ni_HCOO2_H2O4_cis"
    else:
        product_key = "Ni_HCOO2_H2O4_trans"
elif "Ni_HCOO2_H2O4_cis" in results:
    product_key = "Ni_HCOO2_H2O4_cis"
elif "Ni_HCOO2_H2O4_trans" in results:
    product_key = "Ni_HCOO2_H2O4_trans"

needed_step2 = ["Ni_HCOO_H2O5", "HCOO", "H2O"]
if product_key and all(k in results for k in needed_step2):
    dE2 = (results[product_key].energy_eV + results["H2O"].energy_eV
           - results["Ni_HCOO_H2O5"].energy_eV - results["HCOO"].energy_eV)
    dG2 = (results[product_key].gibbs_eV  + results["H2O"].gibbs_eV
           - results["Ni_HCOO_H2O5"].gibbs_eV  - results["HCOO"].gibbs_eV)
    print(f"  Product: {product_key}")
    print(f"  ΔE = {dE2:+.4f} eV  ({dE2 * 23.06:+.2f} kcal/mol)")
    print(f"  ΔG = {dG2:+.4f} eV  ({dG2 * 23.06:+.2f} kcal/mol)  at {T} K, {P/100:.0f} hPa")
else:
    missing = [k for k in needed_step2 if k not in results] + ([] if product_key else ["Ni_HCOO2"])
    print(f"  (Missing results for: {missing})")

print("\nNote: these are gas-phase xTB energies.")
print("For solution-phase ΔG, add solvation corrections (e.g. COSMO-RS or SMD).")
