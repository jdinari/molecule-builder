"""
cli.py
======
molbuilder command-line interface.

Isomers are generated automatically — no flag needed.
When multiple isomers exist, one POSCAR is written per isomer with the
isomer label appended to the filename, e.g.:

    --out Ni_HCOO2_H2O4.POSCAR  →  Ni_HCOO2_H2O4_cis.POSCAR
                                    Ni_HCOO2_H2O4_trans.POSCAR

Custom POSCAR ligands are supported via --custom-ligand.
Denticity modes use colon notation: HCOO:bi, bpy:mono, etc.
"""

import argparse
import sys
from pathlib import Path

from molbuilder.api import (
    build, dimer, trimer, poscar, xyz, info,
    load_ligand_from_poscar,
)
from molbuilder.output.poscar_writer import poscar_to_string
from molbuilder.output.xyz_writer import xyz_to_string
from molbuilder.ligands.library import list_ligands
from molbuilder.core.geometry import list_geometries
from molbuilder.core.molecule import Molecule


def _write(mol: Molecule, out: Path, write_xyz: bool, print_poscar: bool):
    txt = poscar_to_string(mol)
    if print_poscar:
        print(txt)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt)
        print(f"✓ Written to {out}")
        if write_xyz:
            xp = out.with_suffix(".xyz")
            xp.write_text(xyz_to_string(mol))
            print(f"✓ XYZ written to {xp}")


def main():
    p = argparse.ArgumentParser(
        description="molbuilder – transition metal complex builder → POSCAR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Denticity modes (colon notation):
  HCOO        monodentate formate  (single O donor)
  HCOO:bi     bidentate chelating formate (O,O, bite ~55°)
  HCOO:bridge bridging formate
  mu-HCOO     same as HCOO:bridge
  mu-OH       bridging hydroxide  (use with --bridge)
  bpy         bidentate bipyridine (default)
  bpy:mono    monodentate bipyridine

Isomers are generated automatically. With --out, files are named
  <stem>_<label><suffix>  e.g. Ni_HCOO2_H2O4_cis.POSCAR

Custom POSCAR ligands:
  --custom-ligand path/to/lig.POSCAR --donor-atoms 0,2 --ligand-charge -1
""",
    )

    # ── required ──────────────────────────────────────────────────────────────
    p.add_argument("--metal",    required=False, help="Metal element symbol, e.g. Ni")
    p.add_argument("--ox",       type=int, required=False, help="Oxidation state, e.g. 2")

    # ── ligands ───────────────────────────────────────────────────────────────
    p.add_argument("--ligands",  nargs="*", default=[],
                   help="Ligand names or SMILES. Use colon modes: HCOO:bi, bpy:mono")

    # ── custom POSCAR ligand ──────────────────────────────────────────────────
    p.add_argument("--custom-ligand",  type=str, metavar="PATH",
                   help="Path to POSCAR file for a custom ligand")
    p.add_argument("--donor-atoms",    type=str, default="0",
                   help="Comma-separated donor atom indices in custom ligand (default: 0)")
    p.add_argument("--ligand-charge",  type=int, default=0,
                   help="Formal charge of custom ligand (default: 0)")
    p.add_argument("--ligand-name",    type=str, default=None,
                   help="Name for custom ligand (default: filename stem)")

    # ── geometry / structure ──────────────────────────────────────────────────
    p.add_argument("--geometry", type=str, default=None,
                   help="Coordination geometry: oct, sqp, tet, tbp, … (auto-inferred if omitted)")
    p.add_argument("--dimer",    action="store_true", help="Build dinuclear complex")
    p.add_argument("--trimer",   action="store_true", help="Build trinuclear complex")
    p.add_argument("--bridge",   type=str, default=None,
                   help="Bridging ligand for dimer/trimer, e.g. mu-OH, mu-HCOO")
    p.add_argument("--n-bridges", type=int, default=2,
                   help="Number of bridging ligand units (default: 2)")
    p.add_argument("--arrangement", type=str, default="triangular",
                   choices=["triangular", "linear"],
                   help="Trimer arrangement (default: triangular)")
    p.add_argument("--mm-bond",  action="store_true", help="Include metal–metal bond")
    p.add_argument("--mm-distance", type=float, default=None,
                   help="Override M–M distance in Å")

    # ── output ────────────────────────────────────────────────────────────────
    p.add_argument("--out",   type=str, default=None,
                   help="Output POSCAR file. Multiple isomers → one file per isomer.")
    p.add_argument("--xyz",   action="store_true", help="Also write XYZ file")
    p.add_argument("--print", action="store_true", dest="print_poscar",
                   help="Print POSCAR to stdout")

    # ── info ──────────────────────────────────────────────────────────────────
    p.add_argument("--list-ligands",    action="store_true", help="List all available ligands")
    p.add_argument("--list-geometries", action="store_true", help="List all supported geometries")

    args = p.parse_args()

    # ── info modes ────────────────────────────────────────────────────────────
    if args.list_ligands:
        print("Available ligands:")
        for l in list_ligands():
            print(f"  {l}")
        return

    if args.list_geometries:
        print(f"  {'Key':<12}  {'Name':<28}  CN")
        print("  " + "-" * 46)
        for key, name, cn in list_geometries():
            print(f"  {key:<12}  {name:<28}  {cn}")
        return

    if not args.metal or args.ox is None:
        p.error("--metal and --ox are required")

    # ── assemble ligand list ──────────────────────────────────────────────────
    all_ligands = list(args.ligands)

    if args.custom_ligand:
        donor_indices = [int(x.strip()) for x in args.donor_atoms.split(",")]
        custom = load_ligand_from_poscar(
            args.custom_ligand,
            donor_atom_indices=donor_indices,
            charge=args.ligand_charge,
            name=args.ligand_name,
        )
        all_ligands.append(custom)

    out = Path(args.out) if args.out else None

    # ── build ─────────────────────────────────────────────────────────────────
    if args.dimer:
        mol = dimer(args.metal, ox=args.ox,
                    terminal=all_ligands,
                    bridge=args.bridge,
                    n=args.n_bridges,
                    geometry=args.geometry,
                    mm_bond=args.mm_bond,
                    mm_distance=args.mm_distance)
        _write(mol, out, args.xyz, args.print_poscar)
        return

    if args.trimer:
        mol = trimer(args.metal, ox=args.ox,
                     terminal=all_ligands,
                     bridge=args.bridge,
                     arrangement=args.arrangement,
                     geometry=args.geometry)
        _write(mol, out, args.xyz, args.print_poscar)
        return

    # mononuclear — may return one Molecule or a list
    result = build(args.metal, ox=args.ox,
                   ligands=all_ligands,
                   geometry=args.geometry)

    if isinstance(result, Molecule):
        # single isomer
        if args.print_poscar:
            print(poscar_to_string(result))
        if out:
            _write(result, out, args.xyz, False)
    else:
        # multiple isomers
        print(f"Found {len(result)} symmetry-distinct isomer(s).")
        for idx, mol in enumerate(result):
            label = getattr(mol, "label", f"isomer-{idx+1}")
            print(f"\n── {label} ──")
            info(mol)
            if args.print_poscar:
                print(poscar_to_string(mol))
            if out:
                iso_out = out.with_name(f"{out.stem}_{label}{out.suffix or '.POSCAR'}")
                _write(mol, iso_out, args.xyz, False)


if __name__ == "__main__":
    main()
