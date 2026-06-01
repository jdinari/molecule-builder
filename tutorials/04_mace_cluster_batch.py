"""
Tutorial 04 -- MACE energetics on a cluster (no internet, local model file)
===========================================================================

This tutorial shows the full cluster workflow:
  1. Generate a set of Ni complexes
  2. Relax all of them with a local MACE model on GPU
  3. Compare DeltaE relative to the hexaaqua reference
  4. Save results to CSV

Typical cluster submission (SLURM):
------------------------------------
    #!/bin/bash
    #SBATCH --gres=gpu:1
    #SBATCH --mem=16G
    #SBATCH --time=04:00:00
    module load python/3.10
    source venv/bin/activate
    python tutorials/04_mace_cluster_batch.py

Run locally (CPU, slow but functional):
    python tutorials/04_mace_cluster_batch.py

Edit MACE_MODEL below before submitting to your cluster.
"""

from pathlib import Path
from molbuilder.api import build
from molbuilder.relaxation import relax
import csv

# -- Configuration -- set these before running on your cluster ------------------

MACE_MODEL  = None          # <- REQUIRED for offline clusters
                            #   e.g. "/scratch/yourname/models/mace-mh-1.model"
MACE_DEVICE = "cpu"         # <- change to "cuda" on a GPU node

OUT_DIR  = Path("poscar_tutorial04")
CSV_FILE = Path("tutorial04_energetics.csv")

OUT_DIR.mkdir(exist_ok=True)


# -- Structures to compute -----------------------------------------------------

STRUCTURES = {
    "Ni_H2O6":        ("Ni", 2, ["H2O"] * 6),
    "Ni_HCOO2_H2O4":  ("Ni", 2, ["HCOO", "HCOO", "H2O", "H2O", "H2O", "H2O"]),
    "Ni_HCOO4_H2O2":  ("Ni", 2, ["HCOO", "HCOO", "HCOO", "HCOO", "H2O", "H2O"]),
    "Ni_OH2_H2O4":    ("Ni", 2, ["OH",   "OH",   "H2O", "H2O", "H2O", "H2O"]),
}


# -- Build + relax -------------------------------------------------------------

results = {}

for name, (metal, ox, ligands) in STRUCTURES.items():
    mols = build(metal, ox=ox, ligands=ligands)
    mols = mols if isinstance(mols, list) else [mols]

    for mol in mols:
        key = f"{name}_{mol.label}" if mol.label else name
        print(f"Relaxing {key} ...", flush=True)

        try:
            result = relax(
                mol,
                backend = "mace",
                model   = MACE_MODEL,
                device  = MACE_DEVICE,
                fmax    = 0.05,
                steps   = 300,
            )
            results[key] = result
            print(f"  E = {result.energy_eV:.4f} eV  converged={result.converged}")

        except Exception as exc:
            print(f"  FAILED: {exc}")
            results[key] = None


# -- Reference: hexaaqua -------------------------------------------------------

ref_key = "Ni_H2O6"
ref_result = results.get(ref_key)
ref_E = ref_result.energy_eV if ref_result else None


# -- Write CSV -----------------------------------------------------------------

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "energy_eV", "dE_eV", "converged", "steps", "model"])

    for key, res in results.items():
        if res is None:
            writer.writerow([key, "FAILED", "", "", "", ""])
        else:
            dE = (res.energy_eV - ref_E) if ref_E is not None else ""
            writer.writerow([key, f"{res.energy_eV:.6f}",
                             f"{dE:.6f}" if dE != "" else "",
                             res.converged, res.steps, res.model])

print(f"\nResults written to {CSV_FILE}")


# -- Quick summary -------------------------------------------------------------

if ref_E is not None:
    print(f"\nDeltaE relative to {ref_key} ({ref_E:.4f} eV):")
    for key, res in results.items():
        if res is not None and key != ref_key:
            dE = res.energy_eV - ref_E
            print(f"  {key:35s}  DeltaE = {dE:+.4f} eV")
