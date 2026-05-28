"""
Tutorial 06 — Batch enumeration and energetics
===============================================

Shows how the generate_ni_complexes.py script works under the hood,
and how to call run_energetics() directly on a subset of structures.

Covers:
  1. Using enumerate_complexes() directly
  2. Writing POSCARs with write_all()
  3. Running xTB energetics via run_energetics()
  4. Reading bond_status from the output rows
  5. Writing an Excel summary

Run:
    pip install tblite ase openpyxl
    python tutorials/06_enumeration_and_energetics.py
"""

from pathlib import Path
from molbuilder import enumerate_complexes, write_all, MULTI_BRIDGE_CASES
from molbuilder.energetics import run_energetics, BondStatus

OUT_DIR  = Path("out_tutorial06")
CSV_FILE = OUT_DIR / "tutorial06.csv"
XLSX     = OUT_DIR / "tutorial06.xlsx"
OUT_DIR.mkdir(exist_ok=True)


# ── 1. Generate a small set of Ni(II) monomers ───────────────────────────────

print("Generating structures ...")

mols_in_order = []

def _results():
    for mol, row in enumerate_complexes(
        metal       = "Ni",
        ox_states   = [2],
        ligand_pool = ["HCOO", "H2O", "OH"],
        bridge_pool = ["mu-OH", "mu-HCOO"],
        nuclearity  = [1, 2],   # monomers + dimers only for this tutorial
        cn_range    = (4, 6),   # limit to CN 4-6
        output_root = OUT_DIR,
        verbose     = False,
    ):
        mols_in_order.append(mol)
        yield mol, row

rows = write_all(
    _results(),
    output_dir = OUT_DIR,
    csv_file   = None,   # we'll write after energetics
    fmt        = "poscar",
)

n_mono  = sum(1 for r in rows if r["structure"] == "monomer")
n_dimer = sum(1 for r in rows if "dimer" in r["structure"])
print(f"Generated {len(rows)} structures: {n_mono} monomers, {n_dimer} dimers")


# ── 2. Run xTB energetics ─────────────────────────────────────────────────────

print("\nRunning xTB single-point energetics (no freq — just ΔE) ...")

mol_lookup = {row["filename"]: mol for mol, row in zip(mols_in_order, rows)}

rows = run_energetics(
    rows    = rows,
    mols    = mol_lookup,
    backend = "xtb",
    compute_thermo  = False,   # True → adds freq + ΔG, slower
    fmax    = 0.05,
    steps   = 200,
    constrain_bonds = False,   # default: detect bond breaking
    output_dir      = OUT_DIR,
    csv_file        = CSV_FILE,
    excel_file      = XLSX,
    verbose = True,
)


# ── 3. Bond status summary ────────────────────────────────────────────────────

print("\n" + "=" * 50)
n_ok       = sum(1 for r in rows if r.get("bond_status") == BondStatus.OK)
n_stretched= sum(1 for r in rows if r.get("bond_status") == BondStatus.STRETCHED)
n_broken   = sum(1 for r in rows if r.get("bond_status") == BondStatus.BROKEN)
n_error    = sum(1 for r in rows if r.get("bond_status") == "ERROR")

print(f"  OK        : {n_ok}")
print(f"  STRETCHED : {n_stretched}")
print(f"  BROKEN    : {n_broken}  (ligand dissociated during relaxation)")
if n_error:
    print(f"  ERROR     : {n_error}  (xTB failed — check structures)")

# Show broken structures
broken = [r for r in rows if r.get("bond_status") in (BondStatus.BROKEN,)]
if broken:
    print("\n  Broken bond structures (review before DFT):")
    for r in broken:
        print(f"    {r['formula']:20s}  {r['structure']:15s}  "
              f"elong={r.get('bond_max_elongation','?'):.2f}×")

print(f"\n  CSV  → {CSV_FILE}")
print(f"  XLSX → {XLSX}  (colour-coded: red=broken, amber=stretched)")


# ── 4. ΔE relative to lowest-energy monomer ──────────────────────────────────

monomer_rows = [r for r in rows if r["structure"] == "monomer"
                and r.get("relax_energy_eV") is not None]

if monomer_rows:
    ref_row = min(monomer_rows, key=lambda r: r["relax_energy_eV"])
    ref_E   = ref_row["relax_energy_eV"]
    print(f"\n  ΔE relative to lowest monomer ({ref_row['formula']}, E={ref_E:.4f} eV):")
    for r in sorted(monomer_rows, key=lambda r: r["relax_energy_eV"])[:5]:
        dE = r["relax_energy_eV"] - ref_E
        print(f"    {r['formula']:20s}  ΔE = {dE:+.4f} eV"
              f"  {r.get('bond_status','?')}")
