"""
generate_ni_complexes.py
========================
Generate all neutral Ni(II)/Ni(III) complexes and optionally compute
xTB/MACE energetics. Flip COMPUTE_ENERGY = True to enable.
"""

from pathlib import Path
from molbuilder import enumerate_complexes, write_all, MULTI_BRIDGE_CASES
from molbuilder.energetics import run_energetics

# ── Chemistry ─────────────────────────────────────────────────────────────────
METAL       = "Ni"
OX_STATES   = [2, 3]
LIGAND_POOL = ["HCOO", "HCOOH", "H2O", "OH"]
BI_LIGANDS  = ["HCOO:bi"]
BRIDGE_POOL = ["mu-OH", "mu-HCOO"]

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("poscar")
CSV_FILE    = Path("ni_complexes_summary.csv")
EXCEL_FILE  = Path("ni_energetics.xlsx")

# ── Energy computation (off by default) ───────────────────────────────────────
#
# COMPUTE_ENERGY  set True to relax every structure and compute ΔE
# COMPUTE_THERMO  set True (with COMPUTE_ENERGY) to also compute ΔG(T,P)
#
# ENERGY_BACKEND  "xtb"   GFN2-xTB via tblite — best for Ni coordination chemistry,
#                         handles charge and spin explicitly.
#                         pip install tblite ase
#
#                 "mace"  MACE-MH-1 universal MLIP — faster on GPU, downloads
#                         ~500 MB model on first use (or set MACE_MODEL to a
#                         local .model file path).
#                         pip install mace-torch ase
#
#                 "both"  Run both and compare ΔE_mace vs ΔE_xtb.
#
# CONSTRAIN_BONDS False (default) — bond dissociation is physically meaningful
#                 information: if a ligand departs during xTB relaxation, the
#                 coordination is genuinely strained. Broken bonds are flagged
#                 as "BROKEN" in the CSV/Excel so you can review them before DFT.
#                 Set True only if you want to preserve the designed coordination
#                 motif as a DFT seed regardless of whether it's stable.

COMPUTE_ENERGY  = False
COMPUTE_THERMO  = False
ENERGY_BACKEND  = "xtb"
XTB_MODEL       = None          # None → GFN2-xTB
MACE_MODEL      = None          # None → mh-1; or "/path/to/mace-mh-1.model"
MACE_DEVICE     = "cpu"         # "cpu" or "cuda"
CONSTRAIN_BONDS = False
TEMPERATURE_K   = 298.15
PRESSURE_PA     = 101325.0
RELAX_FMAX      = 0.05
RELAX_STEPS     = 300

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mols_in_order = []

    def _results():
        for mol, row in enumerate_complexes(
            metal=METAL, ox_states=OX_STATES,
            ligand_pool=LIGAND_POOL, bridge_pool=BRIDGE_POOL,
            bi_ligands=BI_LIGANDS, nuclearity=[1, 2, 3],
            arrangements=["linear", "triangular"], cn_range=(3, 7),
            multi_bridge_cases=MULTI_BRIDGE_CASES,
            include_heteroleptic=True,
            include_heteroleptic_trimers=True,
            max_terminals_per_metal=1,
            output_root=OUTPUT_DIR, verbose=True,
        ):
            mols_in_order.append(mol)
            yield mol, row

    rows = write_all(_results(), output_dir=OUTPUT_DIR,
                     csv_file=None, fmt="poscar")

    n = lambda s: sum(1 for r in rows if s in r["structure"])
    print(f"\n{'='*55}")
    print(f"  Monomers              : {n('monomer')}")
    print(f"  Dimers (symmetric)    : {n('dimer') - n('hetero')}")
    print(f"  Dimers (heteroleptic) : {n('dimer_hetero')}")
    print(f"  Trimers (symmetric)   : {n('trimer_') - n('hetero')}")
    print(f"  Trimers (heteroleptic): {sum(1 for r in rows if 'hetero' in r['structure'] and 'trimer' in r['structure'])}")
    print(f"  Total                 : {len(rows)}")
    print(f"{'='*55}")

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
    else:
        from molbuilder.output.writer import write_csv
        write_csv(rows, CSV_FILE)
        print(f"  CSV → {CSV_FILE}")
