"""
cli.py
======
Standard molbuilder CLI.
"""

import argparse
import sys
from pathlib import Path

from molbuilder.api import build, build_all_isomers, dimer, trimer, poscar, xyz, info
from molbuilder.output.poscar_writer import poscar_to_string
from molbuilder.output.xyz_writer import xyz_to_string
from molbuilder.ligands.library import list_ligands
from molbuilder.core.geometry import list_geometries


def main():
    parser = argparse.ArgumentParser(
        description="molbuilder – transition metal complex builder → POSCAR"
    )
    parser.add_argument("--metal",    type=str, help="Metal element symbol")
    parser.add_argument("--ox",       type=int, help="Oxidation state")
    parser.add_argument("--ligands",  nargs="*", default=[], help="Ligand names")
    parser.add_argument("--geometry", type=str, default=None, help="Coordination geometry")
    parser.add_argument("--out",      type=str, default=None,
                        help="Output POSCAR file. With --all-isomers, used as base name "
                             "(e.g. Fe.POSCAR → Fe_fac.POSCAR, Fe_mer.POSCAR)")
    parser.add_argument("--xyz",      action="store_true", help="Also write XYZ file")
    parser.add_argument("--print",    action="store_true", dest="print_poscar",
                        help="Print POSCAR to stdout")
    parser.add_argument("--all-isomers", action="store_true",
                        help="Generate all symmetry-distinct isomers")
    parser.add_argument("--dimer",    action="store_true", help="Build dimer")
    parser.add_argument("--trimer",   action="store_true", help="Build trimer")
    parser.add_argument("--bridge",   type=str, default=None, help="Bridging ligand")
    parser.add_argument("--n-bridges", type=int, default=2, help="Number of bridges")
    parser.add_argument("--arrangement", type=str, default="triangular",
                        help="Trimer arrangement: triangular or linear")
    parser.add_argument("--mm-bond",  action="store_true", help="Metal–metal bond")
    parser.add_argument("--mm-distance", type=float, default=None, help="M–M distance (Å)")
    parser.add_argument("--smiles-ligands", nargs="*", default=[], help="SMILES ligands")
    parser.add_argument("--smiles-count",   type=int, default=1, help="Count of SMILES ligands")
    parser.add_argument("--list-ligands",   action="store_true", help="List available ligands")
    parser.add_argument("--list-geometries", action="store_true", help="List available geometries")

    args = parser.parse_args()

    if args.list_ligands:
        ligs = list_ligands()
        print("Available ligands:")
        for l in ligs:
            print(f"  {l}")
        return

    if args.list_geometries:
        geoms = list_geometries()
        print(f"{'Key':<12}  {'Name':<28}  CN")
        print("-" * 46)
        for key, name, cn in geoms:
            print(f"  {key:<10}  {name:<28}  {cn}")
        return

    if not args.metal or args.ox is None:
        parser.error("--metal and --ox are required")

    all_ligands = list(args.ligands)
    for _ in range(args.smiles_count):
        all_ligands.extend(args.smiles_ligands)

    # ── all-isomers mode ──────────────────────────────────────────────────────
    if args.all_isomers:
        isomers = build_all_isomers(args.metal, ox=args.ox,
                                    ligands=all_ligands,
                                    geometry=args.geometry)
        print(f"Found {len(isomers)} symmetry-distinct isomer(s).")
        for iso in isomers:
            mol = iso["molecule"]
            label = iso["label"]
            print(f"\n── Isomer: {label} ──")
            info(mol)

            if args.print_poscar:
                print(poscar_to_string(mol))

            if args.out:
                out = Path(args.out)
                stem = out.stem
                suffix = out.suffix or ".POSCAR"
                iso_path = out.with_name(f"{stem}_{label}{suffix}")
                iso_path.write_text(poscar_to_string(mol))
                print(f"✓ Written to {iso_path}")
                if args.xyz:
                    xyz_path = iso_path.with_suffix(".xyz")
                    xyz_path.write_text(xyz_to_string(mol))
                    print(f"✓ XYZ written to {xyz_path}")
        return

    # ── single-structure mode ─────────────────────────────────────────────────
    if args.dimer:
        mol = dimer(args.metal, ox=args.ox,
                    terminal=all_ligands,
                    bridge=args.bridge,
                    n=args.n_bridges,
                    geometry=args.geometry,
                    mm_bond=args.mm_bond,
                    mm_distance=args.mm_distance)
    elif args.trimer:
        mol = trimer(args.metal, ox=args.ox,
                     terminal=all_ligands,
                     bridge=args.bridge,
                     arrangement=args.arrangement,
                     geometry=args.geometry)
    else:
        mol = build(args.metal, ox=args.ox,
                    ligands=all_ligands,
                    geometry=args.geometry)

    poscar_str = poscar_to_string(mol)

    if args.print_poscar:
        print(poscar_str)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(poscar_str)
        print(f"✓ POSCAR written to {out}")

        if args.xyz:
            xyz_path = out.with_suffix(".xyz")
            xyz_path.write_text(xyz_to_string(mol))
            print(f"✓ XYZ written to {xyz_path}")


if __name__ == "__main__":
    main()
