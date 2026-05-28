"""
batch_enumerate.py — generate a set of Ni complexes and compute xTB energetics

Install:  pip install tblite ase openpyxl
"""
from pathlib import Path
from molbuilder import enumerate_complexes, write_all
from molbuilder.energetics import run_energetics, BondStatus

OUT = Path("out_batch")
OUT.mkdir(exist_ok=True)

mols_ordered = []

def _gen():
    for mol, row in enumerate_complexes(
        metal="Ni", ox_states=[2],
        ligand_pool=["HCOO", "H2O", "OH"],
        bridge_pool=["mu-OH", "mu-HCOO"],
        nuclearity=[1, 2],
        cn_range=(4, 6),
        output_root=OUT,
        verbose=False,
    ):
        mols_ordered.append(mol)
        yield mol, row

rows = write_all(_gen(), output_dir=OUT, csv_file=None)
print(f"Generated {len(rows)} structures")

mol_lookup = {row["filename"]: mol for mol, row in zip(mols_ordered, rows)}

rows = run_energetics(
    rows=rows, mols=mol_lookup,
    backend="xtb", compute_thermo=False,
    fmax=0.05, steps=200,
    output_dir=OUT,
    csv_file=OUT / "energetics.csv",
    excel_file=OUT / "energetics.xlsx",
    verbose=True,
)

n_ok      = sum(1 for r in rows if r.get("bond_status") == BondStatus.OK)
n_broken  = sum(1 for r in rows if r.get("bond_status") == BondStatus.BROKEN)
print(f"\nOK={n_ok}  BROKEN={n_broken}")
print(f"Excel → {OUT}/energetics.xlsx  (red rows = broken bonds)")
