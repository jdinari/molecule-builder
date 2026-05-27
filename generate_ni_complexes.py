"""
generate_ni_complexes.py
========================
Generate all neutral (charge = 0) Ni(II) and Ni(III) mononuclear,
dinuclear, and trinuclear complexes and write them to POSCAR files.

This script is intentionally thin: all chemistry and enumeration logic
lives in molbuilder.combinatorics.  Swapping the metal, adding ligands,
or changing the output format is a one-line edit here.

Usage
-----
    python generate_ni_complexes.py
"""

from pathlib import Path
from molbuilder import enumerate_complexes, write_all, MULTI_BRIDGE_CASES

# ── Chemistry definition ──────────────────────────────────────────────────────

METAL     = "Ni"
OX_STATES = [2, 3]

# Monodentate terminal ligands
LIGAND_POOL = ["HCOO", "HCOOH", "H2O", "OH"]

# Bidentate chelating terminal ligands
BI_LIGANDS  = ["HCOO:bi"]

# Bridging ligands for di- and trinuclear complexes
BRIDGE_POOL = ["mu-OH", "mu-HCOO"]

# ── Output settings ───────────────────────────────────────────────────────────

OUTPUT_DIR = Path("poscar")
CSV_FILE   = Path("ni_complexes_summary.csv")

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = enumerate_complexes(
        metal              = METAL,
        ox_states          = OX_STATES,
        ligand_pool        = LIGAND_POOL,
        bridge_pool        = BRIDGE_POOL,
        bi_ligands         = BI_LIGANDS,
        nuclearity         = [1, 2, 3],
        arrangements       = ["linear", "triangular"],
        cn_range           = (3, 7),
        multi_bridge_cases = MULTI_BRIDGE_CASES,
        # Heteroleptic dimers: both metals get different terminal ligand sets.
        # max_terminals_per_metal=2 keeps runtime fast (~5s extra) while
        # covering the most chemically relevant asymmetric cases.
        # Set to 3 or None to expand the search at the cost of more runtime.
        include_heteroleptic    = True,
        max_terminals_per_metal = 2,
        output_root        = OUTPUT_DIR,
        verbose            = True,
    )

    rows = write_all(
        results,
        output_dir = OUTPUT_DIR,
        csv_file   = CSV_FILE,
        fmt        = "poscar",
    )

    n_mono        = sum(1 for r in rows if r["structure"] == "monomer")
    n_dimer_sym   = sum(1 for r in rows if r["structure"] == "dimer")
    n_dimer_het   = sum(1 for r in rows if r["structure"] == "dimer_hetero")
    n_trimer      = sum(1 for r in rows if r["structure"].startswith("trimer"))
    print(f"\n{'='*60}")
    print(f"  Monomers            : {n_mono}")
    print(f"  Dimers (symmetric)  : {n_dimer_sym}")
    print(f"  Dimers (heteroleptic): {n_dimer_het}")
    print(f"  Trimers             : {n_trimer}")
    print(f"  Total               : {len(rows)} POSCAR files")
    print(f"  CSV                 : {CSV_FILE}")
    print(f"{'='*60}")
