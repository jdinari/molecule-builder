"""
api_extended.py
===============
Extended API with support for:
- Custom ligands from POSCAR files
- Variable denticity binding modes
- Bridging ligands in polynuclear complexes
"""

from typing import List, Optional, Union, Dict
from pathlib import Path
from molbuilder.ligands.custom_poscar import CustomLigandPOSCAR, load_custom_ligand
from molbuilder.ligands.library_extended import get_ligand_with_mode, EXTENDED_LIGAND_LIBRARY


def build_with_custom_ligands(metal: str,
                              ox: int,
                              ligands: Optional[List[str]] = None,
                              custom_ligands: Optional[List[Union[str, CustomLigandPOSCAR]]] = None,
                              geometry: Optional[str] = None,
                              **kwargs):
    """
    Extended build function that accepts custom ligands from POSCAR files.
    
    Args:
        metal: Metal element symbol
        ox: Oxidation state
        ligands: List of standard ligand names
        custom_ligands: List of custom ligand paths or CustomLigandPOSCAR objects
        geometry: Coordination geometry
        **kwargs: Additional arguments passed to base build function
    
    Returns:
        Molecule object
    
    Example:
        # Load custom ligand from POSCAR
        custom = load_custom_ligand("ligand.POSCAR", donor_atom_indices=[0, 1])
        
        # Build complex with custom + standard ligands
        mol = build_with_custom_ligands(
            "Fe", ox=2,
            ligands=["Cl", "Cl", "Cl"],
            custom_ligands=[custom],
            geometry="oct"
        )
    """
    # Import here to avoid circular imports
    from molbuilder.api import build as base_build
    
    if custom_ligands is None:
        custom_ligands = []
    
    # Convert string paths to CustomLigandPOSCAR objects
    processed_custom = []
    for lig in custom_ligands:
        if isinstance(lig, str):
            # Assume it's a file path - user must provide donor indices separately
            raise ValueError(
                f"Custom ligand path {lig} provided without donor indices. "
                "Use load_custom_ligand() to specify donor atoms."
            )
        processed_custom.append(lig)
    
    # Combine standard and custom ligands
    all_ligands = ligands or []
    all_ligands = list(all_ligands) + processed_custom
    
    # Call base build with all ligands
    return base_build(metal, ox=ox, ligands=all_ligands, geometry=geometry, **kwargs)


def build_with_denticity_modes(metal: str,
                               ox: int,
                               ligands_with_modes: Dict[str, Optional[str]] = None,
                               geometry: Optional[str] = None,
                               **kwargs):
    """
    Build a complex where ligands can be specified with binding mode modifiers.
    
    Args:
        metal: Metal element symbol
        ox: Oxidation state
        ligands_with_modes: Dictionary mapping ligand names to binding modes.
                           Format: {"bpy": "bi", "OH": "bridge", "Cl": None}
                           None means use default mode.
        geometry: Coordination geometry
        **kwargs: Additional arguments
    
    Returns:
        Molecule object
    
    Example:
        # Build [Fe(bpy)2(Cl)2] with bidentate bpy
        mol = build_with_denticity_modes(
            "Fe", ox=2,
            ligands_with_modes={
                "bpy": "bi",  # bidentate
                "Cl": None,   # use default (monodentate)
            },
            geometry="oct"
        )
        
        # Build complex with bridging formate
        mol_dimer = build_dimer_with_bridges(
            "Fe", ox=2,
            terminal_ligands=["H2O"],
            bridging_ligands={"HCOO": "bi"},  # bidentate bridging formate
        )
    """
    from molbuilder.api import build as base_build
    
    if ligands_with_modes is None:
        ligands_with_modes = {}
    
    # Expand ligand specs with modes into full ligand list
    expanded_ligands = []
    denticity_info = {}  # Track binding modes for reference
    
    for lig_name, mode in ligands_with_modes.items():
        if mode:
            spec = f"{lig_name}:{mode}"
        else:
            spec = lig_name
        expanded_ligands.append(spec)
        denticity_info[spec] = mode
    
    # Call base build
    return base_build(metal, ox=ox, ligands=expanded_ligands, geometry=geometry, **kwargs)


def dimer_with_bridging_ligands(metal: str,
                                ox: int,
                                terminal_ligands: Optional[List[str]] = None,
                                bridging_ligands: Optional[Dict[str, str]] = None,
                                bridging_count: int = 1,
                                n_metals: int = 2,
                                geometry: Optional[str] = None,
                                mm_bond: bool = False,
                                mm_distance: Optional[float] = None,
                                **kwargs):
    """
    Build a dinuclear or polynuclear complex with bridging ligands.
    
    Args:
        metal: Metal element symbol
        ox: Oxidation state per metal
        terminal_ligands: List of terminal ligand names
        bridging_ligands: Dict of {ligand_name: binding_mode} for bridging ligands
                         e.g., {"HCOO": "bi", "OH": "mono"}
        bridging_count: Number of bridging ligand units
        n_metals: Number of metal centers (default 2 for dimer)
        geometry: Coordination geometry per metal
        mm_bond: Whether metals have M-M bonding
        mm_distance: M-M bond distance if mm_bond=True
        **kwargs: Additional arguments
    
    Returns:
        Molecule object (dinuclear or polynuclear complex)
    
    Example:
        # [Fe2(μ-HCOO)2(H2O)4] dimer with bridging formate
        mol = dimer_with_bridging_ligands(
            "Fe", ox=3,
            terminal_ligands=["H2O", "H2O"],
            bridging_ligands={"HCOO": "bi"},
            bridging_count=2,
            geometry="oct"
        )
        
        # [Rh2(μ-OH)2(CO)2] with bridging hydroxide
        mol = dimer_with_bridging_ligands(
            "Rh", ox=1,
            terminal_ligands=["CO"],
            bridging_ligands={"OH": "bridge"},
            bridging_count=2,
            mm_bond=True,
        )
    """
    from molbuilder.api import dimer as base_dimer
    
    if terminal_ligands is None:
        terminal_ligands = []
    if bridging_ligands is None:
        bridging_ligands = {}
    
    # Build bridge specifications with modes
    # Convert to format expected by base dimer function
    bridge_specs = []
    for bridge_lig, mode in bridging_ligands.items():
        if mode:
            spec = f"{bridge_lig}:{mode}"
        else:
            spec = bridge_lig
        bridge_specs.append(spec)
    
    # Use the first bridge if multiple provided
    # (base dimer typically uses single bridge type)
    bridge_spec = bridge_specs[0] if bridge_specs else None
    
    # Call base dimer function
    return base_dimer(
        metal,
        ox=ox,
        terminal=terminal_ligands,
        bridge=bridge_spec,
        n=bridging_count,
        geometry=geometry,
        mm_bond=mm_bond,
        mm_distance=mm_distance,
        **kwargs
    )
