"""
Tutorial 04 -- MACE geometry relaxation (cluster/GPU workflow)
=============================================================

Shows the full cluster workflow:
  1. Generate a set of Ni complexes
  2. Relax each with a local MACE-MH-1 model
  3. Check for bond dissociation
  4. Write DeltaE CSV relative to the hexaaqua reference

MACE-MH-1 notes
---------------
The "omol" head is used automatically -- this is the molecular head trained on
OMOL/OC20/MATPES data with wB97M-V/R2SCAN references, which is appropriate
for coordination complex geometry optimisation.

MACE does not take charge or spin as explicit inputs.  For charged or open-shell
species, prefer xTB (Tutorial 05) or use MACE results as a sanity check only.

Cluster setup (SLURM example)
------------------------------
    #!/bin/bash
    #SBATCH --gres=gpu:1
    #SBATCH --mem=16G
    #SBATCH --time=04:00:00
    module load python/3.10
    source venv/bin/activate
    python tutorials/04_mace_relaxation.py

Run locally (CPU, slower):
    pip install mace-torch ase
    python tutorials/04_mace_relaxation.py

Edit MACE_MODEL below before submitting to your cluster.
"""

import csv
from pathlib import Path
from molbuilder.api import build
from molbuilder.relaxation import relax, check_bonds_intact

# -- Configuration -------------------------------------------------------------
#
# MACE_MODEL : path to your local .model file.
#   None     -> downloads mace-mh-1 automatically on first run (requires internet).
#              On most cluster compute nodes this will fail; set the path instead.
#   Example  : "/scratch/yourname/models/mace-mh-1.model"
#
# MACE_DEVICE: "cuda" on GPU nodes, "cpu" for local testing.

MACE_MODEL  = None
MACE_DEVICE = "cpu"

OUT_DIR  = Path("out_tutorial04")
CSV_FILE = OUT_DIR / "tutorial04_energetics.csv"
OUT_DIR.mkdir(exist_ok=True)


# -- Structures ----------------------------------------------------------------

STRUCTURES = {
    "Ni_H2O6":         ("Ni", 2, ["H2O"] * 6),
    "Ni_HCOO2_H2O4":   ("Ni", 2, ["HCOO", "HCOO", "H2O", "H2O", "H2O", "H2O"]),
    "Ni_HCOO4_H2O2":   ("Ni", 2, ["HCOO", "HCOO", "HCOO", "HCOO", "H2O", "H2O"]),
    "Ni_OH2_H2O4":     ("Ni", 2, ["OH", "OH", "H2O", "H2O", "H2O", "H2O"]),
}


# -- Build + relax -------------------------------------------------------------

results = {}

for name, (metal, ox, ligands) in STRUCTURES.items():
    result_mols = build(metal, ox=ox, ligands=ligands)
    result_mols = result_mols if isinstance(result_mols, list) else [result_mols]

    for mol in result_mols:
        key = f"{name}_{mol.label}" if mol.label and mol.label != "only" else name
        print(f"Relaxing {key} ({mol.formula}) ...", flush=True)

        try:
            res = relax(
                mol,
                backend = "mace",
                model   = MACE_MODEL,
                device  = MACE_DEVICE,
                fmax    = 0.05,
                steps   = 300,
                # constrain_bonds=False (default): bond dissociation is informative.
                # Set True if you need to preserve the coordination motif.
            )

            # Check whether any M-L bonds broke during relaxation
            bc = check_bonds_intact(mol, res.mol)

            results[key] = {
                "mol_in":   mol,
                "result":   res,
                "bond_check": bc,
            }

            bond_flag = ""
            if bc["max_elongation"] > 1.35:
                bond_flag = "  ! BOND BROKEN"
            elif bc["max_elongation"] > 1.20:
                bond_flag = "  ~ stretched"

            print(f"  E={res.energy_eV:.4f} eV  conv={res.converged}"
                  f"  max_elong={bc['max_elongation']:.2f}x{bond_flag}")

            # Write relaxed POSCAR
            poscar_path = OUT_DIR / f"{key}_relaxed.POSCAR"
            from molbuilder.api import poscar
            poscar(res.mol, poscar_path)

        except Exception as exc:
            print(f"  FAILED: {exc}")
            results[key] = None


# -- Reference: hexaaqua -------------------------------------------------------

ref_E = None
ref_key = next((k for k in results if k.startswith("Ni_H2O6") and results[k]), None)
if ref_key:
    ref_E = results[ref_key]["result"].energy_eV


# -- Write CSV -----------------------------------------------------------------

with open(CSV_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "formula", "energy_eV", "dE_vs_H2O6_eV",
                     "converged", "steps", "bond_status", "max_elongation", "model"])

    for key, data in results.items():
        if data is None:
            writer.writerow([key, "FAILED", "", "", "", "", "", "", ""])
            continue
        res = data["result"]
        mol = data["mol_in"]
        bc  = data["bond_check"]
        dE  = f"{res.energy_eV - ref_E:.6f}" if ref_E is not None else ""

        if bc["max_elongation"] > 1.35:
            bond_status = "BROKEN"
        elif bc["max_elongation"] > 1.20:
            bond_status = "STRETCHED"
        else:
            bond_status = "OK"

        writer.writerow([
            key, mol.formula,
            f"{res.energy_eV:.6f}", dE,
            res.converged, res.steps,
            bond_status, f"{bc['max_elongation']:.3f}",
            res.model,
        ])

print(f"\nCSV written to {CSV_FILE}")


# -- Summary -------------------------------------------------------------------

if ref_E is not None:
    print(f"\nDeltaE relative to {ref_key} ({ref_E:.4f} eV):")
    for key, data in results.items():
        if data and key != ref_key:
            dE = data["result"].energy_eV - ref_E
            bc_str = ""
            if data["bond_check"]["max_elongation"] > 1.35:
                bc_str = "  ! bond broken"
            print(f"  {key:35s}  DeltaE = {dE:+.4f} eV{bc_str}")
