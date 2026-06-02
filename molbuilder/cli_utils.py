"""
cli_utils.py
============
Terminal output helpers shared by generate_ni_complexes.py and any other
top-level scripts.  Nothing here touches chemistry -- just printing.
"""

import datetime

__all__ = ["print_header", "print_settings", "print_enumeration_summary"]

# Option A -- compact box letters
_LOGO = """
  ╔╦╗╔═╗╦  ╔╗ ╦ ╦╦╦  ╔╦╗╔═╗╦═╗
  ║║║║ ║║  ╠╩╗║ ║║║   ║║║╣ ╠╦╝   v{version}
  ╩ ╩╚═╝╩═╝╚═╝╚═╝╩╩═╝═╩╝╚═╝╩╚═   {tagline}
  {line}
"""

_VERSION = "3.0.0"
_TAGLINE = "Ni complex generator"


def print_header(version: str = _VERSION, tagline: str = _TAGLINE) -> None:
    """Print the molbuilder ASCII banner to stdout."""
    now = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    line = "-" * 42
    logo = _LOGO.format(version=version, tagline=tagline, line=line)
    print(logo.rstrip())
    print(f"  Started : {now}")
    print()


def print_settings(metal, ox_states, ligand_pool, bi_ligands, bridge_pool,
                   cn_range, nuclearity,
                   compute_energy, best_isomer_only, compute_reactions,
                   energy_backend, compute_thermo, temperature_k) -> None:
    """Print a compact settings summary so every run is self-documenting."""
    stages = []
    if compute_energy:    stages.append("energetics")
    if best_isomer_only:  stages.append("best-isomer filter")
    if compute_reactions: stages.append("reaction network")
    stages_str = " -> ".join(stages) if stages else "enumeration only"

    print("  Settings:")
    print(f"    metal        {metal}  ox={ox_states}  CN{cn_range}  nuc={nuclearity}")
    print(f"    ligands      {ligand_pool}")
    if bi_ligands:
        print(f"    bidentate    {bi_ligands}")
    if bridge_pool:
        print(f"    bridges      {bridge_pool}")
    print(f"    stages       {stages_str}")
    if compute_energy:
        thermo_str = f"DeltaG at {temperature_k} K" if compute_thermo else "DeltaE only"
        print(f"    backend      {energy_backend}  ({thermo_str})")
    print()


def print_enumeration_summary(rows: list) -> None:
    """Print structure counts after enumeration."""
    n_mono = sum(1 for r in rows if "monomer" in r["structure"])
    n_dim  = sum(1 for r in rows if "dimer"   in r["structure"])
    n_tri  = sum(1 for r in rows if "trimer"  in r["structure"])
    print(f"  {'='*35}")
    print(f"  Monomers : {n_mono}")
    if n_dim:  print(f"  Dimers   : {n_dim}")
    if n_tri:  print(f"  Trimers  : {n_tri}")
    print(f"  Total    : {len(rows)}")
    print(f"  {'='*35}")
    print()
