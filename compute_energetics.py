"""
compute_energetics.py
=====================
CLI for computing xTB / MACE energetics on an existing POSCAR directory.

The actual logic lives in molbuilder.energetics.run_energetics().

Usage
-----
    python compute_energetics.py --poscar-dir poscar/ --backend xtb
    python compute_energetics.py --poscar-dir poscar/ --backend both --thermo --T 350
    python compute_energetics.py --poscar-dir poscar/ --backend mace --model /path/to/mace.model
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def main():
    p = argparse.ArgumentParser(
        description="Compute energetics for molbuilder POSCAR structures."
    )
    p.add_argument("--poscar-dir",   default="poscar")
    p.add_argument("--csv-in",       default=None,   help="Input CSV from write_all")
    p.add_argument("--backend",      default="xtb",  choices=["xtb","mace","both"])
    p.add_argument("--model",        default=None,   help="Model override for selected backend")
    p.add_argument("--mace-device",  default="cpu")
    p.add_argument("--thermo",       action="store_true", help="Run freq → ΔG")
    p.add_argument("--T",            type=float, default=298.15)
    p.add_argument("--P",            type=float, default=101325.0)
    p.add_argument("--fmax",         type=float, default=0.05)
    p.add_argument("--steps",        type=int,   default=300)
    p.add_argument("--constrain",    action="store_true",
                   help="Constrain M-L bonds to prevent dissociation")
    p.add_argument("--output-dir",   default=None)
    p.add_argument("--csv-out",      default="energetics_summary.csv")
    p.add_argument("--excel-out",    default="energetics_summary.xlsx")
    args = p.parse_args()

    poscar_dir = Path(args.poscar_dir)
    output_dir = Path(args.output_dir) if args.output_dir else poscar_dir

    # ── load structures ───────────────────────────────────────────────────────
    rows, mols = _load_structures(poscar_dir, args.csv_in)
    if not mols:
        print(f"No structures found in {poscar_dir}. Check --poscar-dir.")
        sys.exit(1)
    print(f"Loaded {len(mols)} structures. Running {args.backend} energetics...")

    # ── run ───────────────────────────────────────────────────────────────────
    from molbuilder.energetics import run_energetics

    xtb_model  = args.model if args.backend in ("xtb",  "both") else None
    mace_model = args.model if args.backend in ("mace", "both") else None

    run_energetics(
        rows            = rows,
        mols            = mols,
        backend         = args.backend,
        compute_thermo  = args.thermo,
        T               = args.T,
        P               = args.P,
        fmax            = args.fmax,
        steps           = args.steps,
        xtb_model       = xtb_model,
        mace_model      = mace_model,
        mace_device     = args.mace_device,
        constrain_bonds = args.constrain,
        output_dir      = output_dir,
        csv_file        = output_dir / args.csv_out,
        excel_file      = output_dir / args.excel_out,
        verbose         = True,
    )


def _load_structures(poscar_dir: Path, csv_in=None):
    """Return (rows, mol_dict) by reading a CSV or walking the POSCAR directory."""
    import csv as _csv
    from molbuilder.core.molecule import Molecule, Atom

    rows = []
    if csv_in:
        with open(csv_in) as f:
            rows = list(_csv.DictReader(f))
        for r in rows:
            for k in ("cn","charge","n_atoms","ox"):
                try: r[k] = int(r[k])
                except: pass
    else:
        for pf in sorted(poscar_dir.rglob("*.POSCAR")):
            if "_relaxed" in pf.stem:
                continue
            rows.append({"filename": str(pf), "formula": pf.stem,
                         "structure": "", "cn": 0, "charge": 0,
                         "geometry": "", "ligand_combo": pf.stem, "n_atoms": 0})

    mols = {}
    for r in rows:
        p = Path(r["filename"])
        if not p.is_absolute():
            p = poscar_dir / p if not p.exists() else p
        if p.exists():
            mol = _read_poscar(p)
            if mol:
                mols[r["filename"]] = mol
    return rows, mols


def _read_poscar(path: Path):
    try:
        from ase.io import read as ase_read
        from molbuilder.core.molecule import Molecule, Atom
        atoms = ase_read(str(path), format="vasp")
        mol_atoms = [Atom(symbol=s, position=p.copy())
                     for s, p in zip(atoms.get_chemical_symbols(),
                                     atoms.get_positions())]
        return Molecule(atoms=mol_atoms,
                        formula="".join(atoms.get_chemical_symbols()))
    except Exception:
        return None


if __name__ == "__main__":
    main()
