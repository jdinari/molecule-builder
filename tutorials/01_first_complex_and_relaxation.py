"""
Tutorial 01 — Building your first complex and relaxing it with MACE
===================================================================

This tutorial walks through:
  1. Building a Ni(II) hexaaqua complex
  2. Writing it as a POSCAR
  3. Relaxing the geometry with your local MACE model
  4. Inspecting the result

Cluster setup
-------------
On a cluster without internet access (or when using a fine-tuned model),
set MACE_MODEL to the absolute path of your .model file.

On a GPU node, change MACE_DEVICE to "cuda".

Run this script with:
    python tutorials/01_first_complex_and_relaxation.py
"""

from molbuilder.api import build, poscar, info
from molbuilder.relaxation import relax, thermochemistry

# ── Configuration ─────────────────────────────────────────────────────────────
#
# Point this to your local MACE checkpoint.
# None → downloads mh-1 from the internet on first use (not suitable for
#        offline clusters or custom fine-tuned models).
#
MACE_MODEL  = None          # e.g. "/scratch/models/mace-mh-1.model"
MACE_DEVICE = "cpu"         # "cuda" for GPU nodes


# ── 1. Build the complex ──────────────────────────────────────────────────────

mol = build("Ni", ox=2, ligands=["H2O"] * 6)

print("Structure summary:")
info(mol)
# → formula, charge, spin, coordination geometry, atom list

# Write the initial (pre-relaxation) POSCAR
poscar(mol, "Ni_H2O6_initial.POSCAR")
print("Wrote Ni_H2O6_initial.POSCAR")


# ── 2. Relax with MACE ────────────────────────────────────────────────────────

print("\nRelaxing with MACE ...")
result = relax(
    mol,
    backend = "mace",
    model   = MACE_MODEL,   # ← your local path goes here
    device  = MACE_DEVICE,
    fmax    = 0.05,          # convergence threshold in eV/Å
    steps   = 300,
)

print(f"Converged : {result.converged}")
print(f"Steps     : {result.steps}")
print(f"Energy    : {result.energy_eV:.4f} eV")
print(f"Backend   : {result.backend}/{result.model}")

# Write the relaxed structure
poscar(result.mol, "Ni_H2O6_relaxed.POSCAR")
print("Wrote Ni_H2O6_relaxed.POSCAR")


# ── 3. Full thermochemistry (optional, slower) ────────────────────────────────
#
# Uncomment to also compute vibrational frequencies and ΔG(T,P).

# thermo = thermochemistry(
#     mol,
#     backend = "mace",
#     model   = MACE_MODEL,
#     device  = MACE_DEVICE,
#     T       = 298.15,
#     P       = 101325.0,
# )
# print(f"\nZPE     : {thermo.zpe_eV:.4f} eV")
# print(f"H(298K) : {thermo.enthalpy_eV:.4f} eV")
# print(f"G(298K) : {thermo.gibbs_eV:.4f} eV")
#
# # Re-evaluate at a different temperature without re-running:
# print(f"G(350K) : {thermo.gibbs_at(T=350):.4f} eV")
