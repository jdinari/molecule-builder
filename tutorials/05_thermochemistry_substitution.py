"""
Tutorial 05 — Thermochemistry: ΔG for ligand substitution
==========================================================

This tutorial computes Gibbs free energies for the stepwise substitution:

    [Ni(H2O)6]²⁺  +  HCOO⁻  →  [Ni(HCOO)(H2O)5]⁺  +  H2O
    [Ni(HCOO)(H2O)5]⁺  +  HCOO⁻  →  [Ni(HCOO)2(H2O)4]  +  H2O

Using the xTB backend (recommended for Ni coordination chemistry because
it handles charge and spin explicitly).

To use MACE instead, change BACKEND = "mace" and set MACE_MODEL.

Run this script with:
    pip install tblite ase   # for xTB backend
    python tutorials/05_thermochemistry_substitution.py
"""

from molbuilder.api import build
from molbuilder.relaxation import thermochemistry

# ── Configuration ─────────────────────────────────────────────────────────────

BACKEND     = "xtb"         # "xtb" (recommended) or "mace"
MACE_MODEL  = None          # only used if BACKEND = "mace"
MACE_DEVICE = "cpu"

T = 298.15   # K
P = 101325   # Pa  (1 atm)

BACKEND_KWARGS = {}
if BACKEND == "mace":
    BACKEND_KWARGS = {"model": MACE_MODEL, "device": MACE_DEVICE}


# ── Build species ─────────────────────────────────────────────────────────────

species_defs = {
    "Ni_H2O6":         ("Ni", 2, ["H2O", "H2O", "H2O", "H2O", "H2O", "H2O"]),
    "Ni_HCOO_H2O5":    ("Ni", 2, ["HCOO", "H2O", "H2O", "H2O", "H2O", "H2O"]),
    "Ni_HCOO2_H2O4_cis":  ("Ni", 2, ["HCOO", "HCOO", "H2O", "H2O", "H2O", "H2O"]),
    "Ni_HCOO2_H2O4_trans": None,   # populated from isomers below
    "H2O":             None,       # single water molecule
    "HCOO":            None,       # free formate (handled separately below)
}

# Build multi-isomer complexes
print("Building structures ...")

Ni_H2O6       = build("Ni", ox=2, ligands=["H2O"] * 6)
Ni_HCOO_H2O5  = build("Ni", ox=2, ligands=["HCOO"] + ["H2O"] * 5)

Ni_HCOO2_all  = build("Ni", ox=2, ligands=["HCOO", "HCOO", "H2O", "H2O", "H2O", "H2O"])
Ni_HCOO2_all  = Ni_HCOO2_all if isinstance(Ni_HCOO2_all, list) else [Ni_HCOO2_all]
Ni_HCOO2      = {mol.label: mol for mol in Ni_HCOO2_all}

# Free H2O and HCOO⁻ as single-atom/small molecules
# For a rigorous treatment these should be relaxed too; here we use minimal builds.
H2O   = build("O",   ox=0, ligands=[])   # placeholder — use a pre-relaxed geometry in production
HCOO  = build("C",   ox=0, ligands=[])   # placeholder — similarly

# In practice for free-molecule references, you'd load them from a pre-relaxed XYZ/POSCAR.
# This tutorial focuses on the complex energetics workflow.

mols = {
    "Ni_H2O6":             Ni_H2O6,
    "Ni_HCOO_H2O5":        Ni_HCOO_H2O5 if not isinstance(Ni_HCOO_H2O5, list) else Ni_HCOO_H2O5[0],
    "Ni_HCOO2_H2O4_cis":   Ni_HCOO2.get("cis"),
    "Ni_HCOO2_H2O4_trans":  Ni_HCOO2.get("trans"),
}
mols = {k: v for k, v in mols.items() if v is not None}


# ── Compute thermochemistry ───────────────────────────────────────────────────

print(f"\nRunning thermochemistry (backend={BACKEND}, T={T} K, P={P} Pa) ...")
print("This may take several minutes per structure.\n")

results = {}
for name, mol in mols.items():
    print(f"  {name} ...", flush=True)
    try:
        results[name] = thermochemistry(
            mol,
            backend = BACKEND,
            T       = T,
            P       = P,
            **BACKEND_KWARGS,
        )
        r = results[name]
        print(f"    E={r.energy_eV:.4f} eV  G={r.gibbs_eV:.4f} eV  converged={r.converged}")
    except Exception as exc:
        print(f"    FAILED: {exc}")


# ── ΔG for stepwise substitution ─────────────────────────────────────────────
#
# For a proper reaction energy you would also include:
#   HCOO⁻(aq)  and  H2O(l)  reference energies.
# This snippet shows the pattern; substitute real references for production use.

print("\n" + "=" * 55)
print("  Isomer comparison: cis vs trans [Ni(HCOO)2(H2O)4]")
print("=" * 55)

r_cis   = results.get("Ni_HCOO2_H2O4_cis")
r_trans = results.get("Ni_HCOO2_H2O4_trans")

if r_cis and r_trans:
    dE = r_trans.energy_eV - r_cis.energy_eV
    dG = r_trans.gibbs_eV  - r_cis.gibbs_eV
    print(f"  ΔE(trans - cis) = {dE:+.4f} eV  ({dE*23.06:.2f} kcal/mol)")
    print(f"  ΔG(trans - cis) = {dG:+.4f} eV  ({dG*23.06:.2f} kcal/mol)")
    print()
    if dG < 0:
        print("  trans isomer is thermodynamically favoured at this T/P.")
    else:
        print("  cis isomer is thermodynamically favoured at this T/P.")

    # Re-evaluate at different temperatures without re-running
    print("\n  G(cis) and G(trans) at other temperatures:")
    for T2 in [250, 298.15, 350, 400]:
        G_cis   = r_cis.gibbs_at(T=T2)
        G_trans = r_trans.gibbs_at(T=T2)
        dG2 = G_trans - G_cis
        print(f"    T={T2:6.2f} K  ΔG = {dG2:+.4f} eV")

else:
    print("  (one or both isomers failed — check output above)")
