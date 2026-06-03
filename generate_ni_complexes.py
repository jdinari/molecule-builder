"""
generate_ni_complexes.py
========================
Top-level script: edit the settings below, then run.

    python generate_ni_complexes.py

All logic lives in the molbuilder package.  This file is settings + glue only.

Stages (enabled by the flags below):
  0  Enumerate structures -> POSCARs + mol cache  (always runs, fast)
  1  Relax + thermochemistry -> CSV + Excel        (COMPUTE_ENERGY)
  2  Best-isomer filter                            (BEST_ISOMER_ONLY)
  3  Reaction network + DeltaG screening           (COMPUTE_REACTIONS)

Each stage writes results to disk so jobs can be split across cluster runs.
"""

from pathlib import Path

from molbuilder import enumerate_complexes, write_all, MULTI_BRIDGE_CASES
from molbuilder.energetics import run_energetics, write_broken_report
from molbuilder.energetics import filter_duplicate_structures, filter_best_isomers
from molbuilder.output.writer import load_csv, save_mols_cache, load_mols_cache, write_csv
from molbuilder.reactions import (attach_energies_from_rows, run_network_energies,
                                   run_and_export_network)
from molbuilder.cli_utils import print_header, print_settings, print_enumeration_summary


# ==============================================================================
# SETTINGS  --  edit these before running
# ==============================================================================

# --- Chemistry ----------------------------------------------------------------
METAL       = "Ni"
OX_STATES   = [2, 3]
LIGAND_POOL = ["HCOO", "H2O", "OH"]
BI_LIGANDS  = ["HCOO:bi"]
BRIDGE_POOL = ["mu-OH", "mu-HCOO"]

# --- Output paths -------------------------------------------------------------
OUTPUT_DIR     = Path("poscar")
CSV_FILE       = Path("ni_complexes_summary.csv")
EXCEL_FILE     = Path("ni_energetics.xlsx")
REACTIONS_CSV  = Path("ni_reaction_network.csv")
REACTIONS_PLOT = Path("ni_reaction_network.png")
MOLS_CACHE     = Path("ni_mols_cache.pkl")

# --- Stage flags --------------------------------------------------------------
COMPUTE_ENERGY    = False   # relax structures and compute DeltaE / DeltaG
BEST_ISOMER_ONLY  = False   # keep only lowest-energy isomer per stoichiometry
COMPUTE_REACTIONS = False   # build isodesmic reaction network

# --- Energy backend -----------------------------------------------------------
# "xtb"      GFN2-xTB, explicit charge/spin (~5 s/structure, CPU)
# "mace"     MACE-MH-1, better energetics (~0.5 s/structure, GPU)
# "xtb+mace" xTB geometry + MACE single-point -- best quality, needs both
#            requires COMPUTE_ENERGY = True; reaction network falls back to xTB
ENERGY_BACKEND = "xtb+mace"

# --- Thermochemistry ----------------------------------------------------------
COMPUTE_THERMO = True
TEMPERATURE_K  = 298.15
PRESSURE_PA    = 101325.0

# --- Relaxation ---------------------------------------------------------------
RELAX_FMAX      = 0.05   # eV/Angstrom
RELAX_STEPS     = 300
CONSTRAIN_BONDS = False

# --- Model paths (None = built-in defaults) -----------------------------------
# On a cluster set MACE_MODEL to your local .model file to avoid downloading.
# Find the path after first download:  from mace.calculators import mace_mp
#                                      mace_mp("mh-1")  # prints cache path
XTB_MODEL   = None
MACE_MODEL  = None
MACE_DEVICE = "cpu"   # "cuda" on a GPU node

# --- Reaction network ---------------------------------------------------------
REACTION_MAX_DG = 0.5   # eV -- only report reactions with |DeltaG| <= this
REACTION_TYPES  = None  # None = all; or e.g. ["substitution"]


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    print_header()
    print_settings(
        metal=METAL, ox_states=OX_STATES,
        ligand_pool=LIGAND_POOL, bi_ligands=BI_LIGANDS, bridge_pool=BRIDGE_POOL,
        cn_range=(3, 7), nuclearity=[1, 2, 3],
        compute_energy=COMPUTE_ENERGY, best_isomer_only=BEST_ISOMER_ONLY,
        compute_reactions=COMPUTE_REACTIONS, energy_backend=ENERGY_BACKEND,
        compute_thermo=COMPUTE_THERMO, temperature_k=TEMPERATURE_K,
    )

    # --- Stage 0: Enumerate ---------------------------------------------------
    mols_in_order = []

    def _gen():
        for mol, row in enumerate_complexes(
            metal=METAL, ox_states=OX_STATES,
            ligand_pool=LIGAND_POOL, bridge_pool=BRIDGE_POOL,
            bi_ligands=BI_LIGANDS,
            nuclearity=[1, 2, 3], arrangements=["linear", "triangular"],
            cn_range=(3, 7), multi_bridge_cases=MULTI_BRIDGE_CASES,
            include_heteroleptic=True, include_heteroleptic_trimers=True,
            max_terminals_per_metal=1, output_root=OUTPUT_DIR, verbose=True,
        ):
            mols_in_order.append(mol)
            yield mol, row

    rows = write_all(_gen(), output_dir=OUTPUT_DIR, csv_file=None, fmt="poscar")
    print_enumeration_summary(rows)
    save_mols_cache(mols_in_order, rows, MOLS_CACHE)

    # --- Stage 1: Energetics --------------------------------------------------
    if COMPUTE_ENERGY:
        mol_lookup = {r["filename"]: m for m, r in zip(mols_in_order, rows)}
        rows = run_energetics(
            rows=rows, mols=mol_lookup,
            backend=ENERGY_BACKEND, compute_thermo=COMPUTE_THERMO,
            T=TEMPERATURE_K, P=PRESSURE_PA,
            fmax=RELAX_FMAX, steps=RELAX_STEPS,
            xtb_model=XTB_MODEL, mace_model=MACE_MODEL,
            mace_device=MACE_DEVICE, constrain_bonds=CONSTRAIN_BONDS,
            output_dir=OUTPUT_DIR, csv_file=CSV_FILE, excel_file=EXCEL_FILE,
        )
        broken = [(mols_in_order[i], rows[i])
                  for i, r in enumerate(rows) if r.get("bond_status") == "BROKEN"]
        if broken:
            write_broken_report(broken, OUTPUT_DIR, verbose=True)
        rows = filter_duplicate_structures(rows, mol_lookup)

    elif CSV_FILE.exists():
        print(f"  Loading energetics from {CSV_FILE} ...")
        rows = load_csv(CSV_FILE)
        print(f"  Loaded {len(rows)} rows\n")
    else:
        write_csv(rows, CSV_FILE)
        print(f"  CSV -> {CSV_FILE}\n")

    # --- Stage 2: Best-isomer filter ------------------------------------------
    if BEST_ISOMER_ONLY:
        if any(r.get("relax_energy_eV") is not None for r in rows):
            rows = filter_best_isomers(rows)
        else:
            print("  BEST_ISOMER_ONLY skipped -- run COMPUTE_ENERGY = True first.\n")

    # --- Stage 3: Reaction network --------------------------------------------
    if COMPUTE_REACTIONS:
        from molbuilder.reactions import ReactionNetwork

        mol_lookup = (load_mols_cache(MOLS_CACHE) if MOLS_CACHE.exists()
                      else {r["filename"]: m for m, r in zip(mols_in_order, rows)})
        mols_and_rows = [(mol_lookup[r["filename"]], r)
                         for r in rows if r.get("filename") in mol_lookup]

        if not mols_and_rows:
            print(f"ERROR: no molecule objects found. Run Stage 0 first (cache: {MOLS_CACHE}).")
        else:
            net = ReactionNetwork(mols_and_rows, bond_filter=True,
                                  include_geometry_changes=True, verbose=True)
            print(net.summary())

            has_energies = any(r.get("relax_energy_eV") is not None for r in rows)
            if has_energies:
                attach_energies_from_rows(
                    net, rows, backend=ENERGY_BACKEND, compute_thermo=COMPUTE_THERMO,
                    T=TEMPERATURE_K, P=PRESSURE_PA, fmax=RELAX_FMAX, steps=RELAX_STEPS,
                    xtb_model=XTB_MODEL, mace_model=MACE_MODEL, mace_device=MACE_DEVICE,
                )
            else:
                run_network_energies(net, backend=ENERGY_BACKEND,
                                     compute_thermo=COMPUTE_THERMO,
                                     T=TEMPERATURE_K, P=PRESSURE_PA,
                                     fmax=RELAX_FMAX, steps=RELAX_STEPS)

            run_and_export_network(
                net, rows,
                output_csv=REACTIONS_CSV, output_plot=REACTIONS_PLOT,
                reaction_max_dg=REACTION_MAX_DG, reaction_types=REACTION_TYPES,
                compute_thermo=COMPUTE_THERMO, output_dir=OUTPUT_DIR,
            )
