"""
Tutorial 05 — Thermochemistry: ΔE and ΔG for ligand substitution
=================================================================

Computes Gibbs free energies for the stepwise formate substitution reaction:

    [Ni(H2O)6]²⁺  +  HCOO⁻  →  [Ni(HCOO)(H2O)5]⁺  +  H2O

and compares the two isomers of the di-formate product:

    cis-[Ni(HCOO)2(H2O)4]  vs  trans-[Ni(HCOO)2(H2O)4]

Uses the xTB backend (GFN2-xTB via tblite) which handles Ni charge and spin
multiplicity explicitly.  xTB is recommended over MACE for charged or
open-shell species.

All species are relaxed to their xTB minimum, then vibrational frequencies
are computed numerically and thermochemical corrections applied using the
ideal-gas rigid-rotor harmonic-oscillator (IGRRHO) model.

Note on the free-molecule references
-------------------------------------
For a rigorous ΔG you need E(HCOO⁻) and E(H2O).  Here we build both
from scratch with molbuilder and relax them with xTB so all energies are
on the same level of theory.

Install:
    pip install tblite ase

Run:
    python tutorials/05_thermochemistry.py
"""

from molbuilder.api import build
from molbuilder.relaxation import thermochemistry
from molbuilder.core.molecule import Molecule, Atom
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

BACKEND     = "xtb"      # "xtb" (recommended) or "mace"
MACE_MODEL  = None       # only used when BACKEND = "mace"
MACE_DEVICE = "cpu"

T = 298.15   # K
P = 101325   # Pa  (1 atm)

BACKEND_KWARGS = {}
if BACKEND == "mace":
    BACKEND_KWARGS = {"model": MACE_MODEL, "device": MACE_DEVICE}


# ── Helper: build a free H2O molecule ────────────────────────────────────────
#
# molbuilder's build() requires a metal centre.
# For a free H2O we construct the geometry directly using known bond parameters:
# O-H = 0.958 Å, H-O-H = 104.5°

def make_h2o() -> Molecule:
    """Return a free water molecule at the experimental geometry."""
    angle_half = np.radians(104.5 / 2)
    oh = 0.958
    o  = np.array([0., 0., 0.])
    h1 = np.array([oh * np.sin(angle_half),  oh * np.cos(angle_half), 0.])
    h2 = np.array([-oh * np.sin(angle_half), oh * np.cos(angle_half), 0.])
    return Molecule(
        atoms=[Atom("O", o), Atom("H", h1), Atom("H", h2)],
        formula="H2O", charge=0, spin_multiplicity=1,
        metal_symbol="", metal_ox=0,
    )


# ── Helper: build a free HCOO⁻ ion ──────────────────────────────────────────
#
# Formate: C at origin, O1 and O2 symmetric, H on C.
# O-C-O = 126°, C-O = 1.25 Å, C-H = 1.09 Å.

def make_formate() -> Molecule:
    """Return a free formate anion (HCOO⁻) at an approximate geometry."""
    angle_half = np.radians(126.0 / 2)
    co = 1.25
    ch = 1.09
    c  = np.array([0., 0., 0.])
    o1 = np.array([co * np.sin(angle_half),  co * np.cos(angle_half), 0.])
    o2 = np.array([-co * np.sin(angle_half), co * np.cos(angle_half), 0.])
    h  = np.array([0., -ch, 0.])
    return Molecule(
        atoms=[Atom("C", c), Atom("O", o1), Atom("O", o2), Atom("H", h)],
        formula="HCOO", charge=-1, spin_multiplicity=1,
        metal_symbol="", metal_ox=0,
    )


# ── Build Ni complexes ────────────────────────────────────────────────────────

print("Building structures ...")

Ni_H2O6        = build("Ni", ox=2, ligands=["H2O"] * 6)
Ni_HCOO_H2O5   = build("Ni", ox=2, ligands=["HCOO"] + ["H2O"] * 5)

Ni_HCOO2_all   = build("Ni", ox=2, ligands=["HCOO", "HCOO", "H2O", "H2O", "H2O", "H2O"])
Ni_HCOO2_all   = Ni_HCOO2_all if isinstance(Ni_HCOO2_all, list) else [Ni_HCOO2_all]
Ni_HCOO2       = {mol.label: mol for mol in Ni_HCOO2_all}

# Normalise the single-result cases
if isinstance(Ni_H2O6, list):      Ni_H2O6 = Ni_H2O6[0]
if isinstance(Ni_HCOO_H2O5, list): Ni_HCOO_H2O5 = Ni_HCOO_H2O5[0]

species = {
    "Ni_H2O6":          Ni_H2O6,
    "Ni_HCOO_H2O5":     Ni_HCOO_H2O5,
    "Ni_HCOO2_cis":     Ni_HCOO2.get("cis"),
    "Ni_HCOO2_trans":   Ni_HCOO2.get("trans"),
    "H2O":              make_h2o(),
    "HCOO":             make_formate(),
}
species = {k: v for k, v in species.items() if v is not None}

print(f"Species to compute: {list(species.keys())}")


# ── Compute thermochemistry ───────────────────────────────────────────────────

print(f"\nRunning thermochemistry  backend={BACKEND}  T={T} K  P={P} Pa")
print("(This runs geometry relaxation + frequency calculation — a few minutes per species)\n")

results = {}
for name, mol in species.items():
    print(f"  {name:30s} ({mol.formula}) ...", flush=True)
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
        print(f"    E={r.energy_eV:.4f} eV  G={r.gibbs_eV:.4f} eV  conv={r.converged}")
    except Exception as exc:
        print(f"    FAILED: {exc}")


# ── ΔG: cis vs trans [Ni(HCOO)2(H2O)4] ──────────────────────────────────────

print("\n" + "=" * 58)
print("  Isomer comparison: cis vs trans [Ni(HCOO)2(H2O)4]")
print("=" * 58)

r_cis   = results.get("Ni_HCOO2_cis")
r_trans = results.get("Ni_HCOO2_trans")

if r_cis and r_trans:
    dE = r_trans.energy_eV - r_cis.energy_eV
    dG = r_trans.gibbs_eV  - r_cis.gibbs_eV
    print(f"  ΔE(trans − cis) = {dE:+.4f} eV  ({dE * 23.06:+.2f} kcal/mol)")
    print(f"  ΔG(trans − cis) = {dG:+.4f} eV  ({dG * 23.06:+.2f} kcal/mol)")
    print()
    preferred = "trans" if dG < 0 else "cis"
    print(f"  → {preferred} isomer is thermodynamically favoured at {T} K, {P/100:.0f} hPa")

    # Re-evaluate at other temperatures without re-running any calculations
    print(f"\n  ΔG(trans − cis) at other temperatures (no re-run needed):")
    for T2 in [250, 298.15, 350, 400]:
        dG2 = r_trans.gibbs_at(T=T2) - r_cis.gibbs_at(T=T2)
        print(f"    T = {T2:6.1f} K  →  ΔG = {dG2:+.4f} eV")
else:
    print("  (One or both isomers failed — check output above.)")


# ── ΔG: first substitution step ──────────────────────────────────────────────
#
# [Ni(H2O)6]²⁺  +  HCOO⁻  →  [Ni(HCOO)(H2O)5]⁺  +  H2O

print("\n" + "=" * 58)
print("  [Ni(H2O)6] + HCOO⁻  →  [Ni(HCOO)(H2O)5] + H2O")
print("=" * 58)

needed = ["Ni_H2O6", "HCOO", "Ni_HCOO_H2O5", "H2O"]
if all(k in results for k in needed):
    dE1 = (results["Ni_HCOO_H2O5"].energy_eV + results["H2O"].energy_eV
           - results["Ni_H2O6"].energy_eV     - results["HCOO"].energy_eV)
    dG1 = (results["Ni_HCOO_H2O5"].gibbs_eV  + results["H2O"].gibbs_eV
           - results["Ni_H2O6"].gibbs_eV      - results["HCOO"].gibbs_eV)
    print(f"  ΔE = {dE1:+.4f} eV  ({dE1 * 23.06:+.2f} kcal/mol)")
    print(f"  ΔG = {dG1:+.4f} eV  ({dG1 * 23.06:+.2f} kcal/mol)  at {T} K, {P/100:.0f} hPa")
    print()
    print("  Note: gas-phase xTB energies only — for solution-phase ΔG,")
    print("  add solvation corrections (e.g. COSMO-RS or SMD post-correction).")
else:
    missing = [k for k in needed if k not in results]
    print(f"  Missing results for: {missing}")
