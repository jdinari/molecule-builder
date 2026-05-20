"""
cli_extended.py
===============
Extended CLI supporting:
- Custom POSCAR ligands
- Variable denticity modes
- Bridging ligands for polynuclear complexes
"""

import argparse
from pathlib import Path
from molbuilder.api_extended import (
    build_with_custom_ligands,
    build_with_denticity_modes,
    dimer_with_bridging_ligands,
)
from molbuilder.ligands.custom_poscar import load_custom_ligand
from molbuilder.output.poscar_writer import poscar_to_string


def setup_extended_cli_parser(parser: argparse.ArgumentParser):
    """
    Add extended arguments to an existing CLI parser.
    """
    # Custom ligand arguments
    parser.add_argument(
        "--custom-ligand",
        type=str,
        action="append",
        dest="custom_ligands",
        help="Path to POSCAR file for custom ligand (can use multiple times)"
    )
    
    parser.add_argument(
        "--custom-donor-atoms",
        type=str,
        action="append",
        dest="custom_donor_indices",
        help="Comma-separated atom indices (0-indexed) that donate in the custom ligand. "
             "Must correspond 1-to-1 with --custom-ligand arguments. "
             "Example: --custom-donor-atoms 0,2 for atoms 0 and 2"
    )
    
    parser.add_argument(
        "--custom-ligand-charge",
        type=int,
        default=0,
        help="Formal charge of custom ligand (default: 0)"
    )
    
    parser.add_argument(
        "--custom-ligand-name",
        type=str,
        help="Name for the custom ligand (default: POSCAR filename)"
    )
    
    # Denticity mode arguments
    parser.add_argument(
        "--ligand-mode",
        type=str,
        action="append",
        dest="ligand_modes",
        help="Specify binding mode for a ligand. Format: ligand_name:mode "
             "Example: --ligand-mode bpy:bi --ligand-mode OH:bridge"
    )
    
    # Bridging ligand arguments
    parser.add_argument(
        "--bridge-ligand",
        type=str,
        action="append",
        dest="bridge_ligands",
        help="Bridging ligand with mode. Format: ligand_name:mode "
             "Example: --bridge-ligand HCOO:bi --bridge-ligand OH:mono"
    )
    
    parser.add_argument(
        "--bridge-count",
        type=int,
        default=1,
        help="Number of bridging ligand units (default: 1)"
    )
    
    return parser


def parse_custom_ligands(custom_ligand_paths, donor_indices_strs, charge, name):
    """
    Parse custom ligand arguments and create CustomLigandPOSCAR objects.
    """
    if not custom_ligand_paths:
        return []
    
    if not donor_indices_strs or len(donor_indices_strs) != len(custom_ligand_paths):
        raise ValueError(
            f"Must provide --custom-donor-atoms for each --custom-ligand. "
            f"Got {len(custom_ligand_paths)} ligands but {len(donor_indices_strs or [])} donor specs."
        )
    
    custom_ligs = []
    for lig_path, donor_str in zip(custom_ligand_paths, donor_indices_strs):
        donor_indices = [int(x.strip()) for x in donor_str.split(",")]
        lig = load_custom_ligand(
            lig_path,
            donor_atom_indices=donor_indices,
            charge=charge,
            name=name,
        )
        custom_ligs.append(lig)
    
    return custom_ligs


def parse_ligand_modes(mode_strings):
    """
    Parse --ligand-mode arguments into a dictionary.
    
    Input: ["bpy:bi", "OH:bridge", "Cl"]
    Output: {"bpy": "bi", "OH": "bridge", "Cl": None}
    """
    modes = {}
    if not mode_strings:
        return modes
    
    for spec in mode_strings:
        if ":" in spec:
            lig, mode = spec.split(":", 1)
            modes[lig] = mode
        else:
            modes[spec] = None
    
    return modes


def parse_bridge_ligands(bridge_strings):
    """
    Parse --bridge-ligand arguments into a dictionary.
    
    Input: ["HCOO:bi", "OH:mono"]
    Output: {"HCOO": "bi", "OH": "mono"}
    """
    bridges = {}
    if not bridge_strings:
        return bridges
    
    for spec in bridge_strings:
        if ":" in spec:
            lig, mode = spec.split(":", 1)
            bridges[lig] = mode
        else:
            bridges[spec] = None
    
    return bridges


def main_extended():
    """
    Extended CLI main function with custom ligand and bridging support.
    """
    parser = argparse.ArgumentParser(
        description="Extended molbuilder CLI with custom ligands and bridging support"
    )
    
    # Standard arguments (from original CLI)
    parser.add_argument("--metal", type=str, required=True, help="Metal element")
    parser.add_argument("--ox", type=int, required=True, help="Oxidation state")
    parser.add_argument("--ligands", nargs="*", help="Ligand names")
    parser.add_argument("--geometry", type=str, help="Coordination geometry")
    parser.add_argument("--out", type=str, required=True, help="Output POSCAR file")
    parser.add_argument("--dimer", action="store_true", help="Build dimer")
    parser.add_argument("--n-metals", type=int, default=1, help="Number of metal centers")
    
    # Extended arguments
    parser = setup_extended_cli_parser(parser)
    
    args = parser.parse_args()
    
    try:
        # Parse custom ligands
        custom_ligs = parse_custom_ligands(
            args.custom_ligands,
            args.custom_donor_indices,
            args.custom_ligand_charge,
            args.custom_ligand_name,
        )
        
        # Parse ligand modes
        lig_modes = parse_ligand_modes(args.ligand_modes)
        
        # Parse bridge ligands
        bridge_ligs = parse_bridge_ligands(args.bridge_ligands)
        
        # Build complex
        if args.dimer or len(bridge_ligs) > 0:
            mol = dimer_with_bridging_ligands(
                args.metal,
                ox=args.ox,
                terminal_ligands=args.ligands,
                bridging_ligands=bridge_ligs,
                bridging_count=1,
                n_metals=args.n_metals,
                geometry=args.geometry,
            )
        elif custom_ligs:
            mol = build_with_custom_ligands(
                args.metal,
                ox=args.ox,
                ligands=args.ligands,
                custom_ligands=custom_ligs,
                geometry=args.geometry,
            )
        elif lig_modes:
            mol = build_with_denticity_modes(
                args.metal,
                ox=args.ox,
                ligands_with_modes=lig_modes,
                geometry=args.geometry,
            )
        else:
            from molbuilder.api import build
            mol = build(
                args.metal,
                ox=args.ox,
                ligands=args.ligands,
                geometry=args.geometry,
            )
        
        # Write output
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(poscar_to_string(mol))
        print(f"✓ Complex written to {output_path}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        raise


if __name__ == "__main__":
    main_extended()
