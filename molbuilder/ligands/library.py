"""
library.py
====================
Extended ligand library including:
- Formate (HCOO-) and formic acid (HCOOH)
- Variable denticity modes (monodentate vs bidentate)
- Support for bridging in polynuclear complexes
"""

from .denticity_modes import DenticitySite, DenticityMode, create_variable_denticity_ligand

# ============================================================================
# NEW LIGANDS: Formate and Formic Acid
# ============================================================================

FORMATE_MONODENTATE = DenticityMode(
    name="monodentate-O",
    donor_sites=[DenticitySite(atom_index=1, atom_symbol="O", is_bridging=False)],
    charge=-1,
    is_bridging=False,
)

FORMATE_BIDENTATE = DenticityMode(
    name="bidentate-O,O",
    donor_sites=[
        DenticitySite(atom_index=1, atom_symbol="O"),
        DenticitySite(atom_index=3, atom_symbol="O"),
    ],
    bite_angle=55.0,  # Typical chelating carboxylate angle
    charge=-1,
    is_bridging=False,
)

FORMATE_BRIDGING = DenticityMode(
    name="bridging-O,O",
    donor_sites=[
        DenticitySite(atom_index=1, atom_symbol="O", is_bridging=True),
        DenticitySite(atom_index=3, atom_symbol="O", is_bridging=True),
    ],
    bite_angle=60.0,
    charge=-1,
    is_bridging=True,  # Can bridge between two metals
)

# HCOO- formate ion
HCOO = create_variable_denticity_ligand(
    name="HCOO",
    smiles="C(=O)[O-]",  # Formate anion
    possible_modes=[FORMATE_MONODENTATE, FORMATE_BIDENTATE, FORMATE_BRIDGING],
    charge=-1,
)

# HCOOH formic acid (neutral)
HCOOH_MONODENTATE = DenticityMode(
    name="monodentate-O",
    donor_sites=[DenticitySite(atom_index=1, atom_symbol="O")],
    charge=0,
    is_bridging=False,
)

HCOOH_BIDENTATE = DenticityMode(
    name="bidentate-O,O",
    donor_sites=[
        DenticitySite(atom_index=1, atom_symbol="O"),
        DenticitySite(atom_index=3, atom_symbol="O"),
    ],
    bite_angle=52.0,
    charge=0,
    is_bridging=False,
)

HCOOH = create_variable_denticity_ligand(
    name="HCOOH",
    smiles="C(=O)O",  # Formic acid
    possible_modes=[HCOOH_MONODENTATE, HCOOH_BIDENTATE],
    charge=0,
)

# ============================================================================
# VARIABLE DENTICITY MODES FOR EXISTING LIGANDS
# ============================================================================
# These allow existing ligands to be specified with binding mode modifiers
# E.g., use "bpy:mono" for monodentate binding or "bpy:bi" for bidentate

BIPY_MONODENTATE = DenticityMode(
    name="monodentate-N",
    donor_sites=[DenticitySite(atom_index=2, atom_symbol="N")],
    charge=0,
    is_bridging=False,
)

BIPY_BIDENTATE = DenticityMode(
    name="bidentate-N,N",
    donor_sites=[
        DenticitySite(atom_index=2, atom_symbol="N"),
        DenticitySite(atom_index=9, atom_symbol="N"),
    ],
    bite_angle=72.0,  # Typical bpy chelation angle
    charge=0,
    is_bridging=False,
)

BIPY_BRIDGING = DenticityMode(
    name="bridging-N,N",
    donor_sites=[
        DenticitySite(atom_index=2, atom_symbol="N", is_bridging=True),
        DenticitySite(atom_index=9, atom_symbol="N", is_bridging=True),
    ],
    bite_angle=72.0,
    charge=0,
    is_bridging=True,
)

BIPY_MODES = {
    "bpy:mono": BIPY_MONODENTATE,
    "bpy:bi": BIPY_BIDENTATE,
    "bpy:bridge": BIPY_BRIDGING,
    "bipy:mono": BIPY_MONODENTATE,
    "bipy:bi": BIPY_BIDENTATE,
    "bipy:bridge": BIPY_BRIDGING,
}

# EN (ethylenediamine) modes
EN_BIDENTATE = DenticityMode(
    name="bidentate-N,N",
    donor_sites=[
        DenticitySite(atom_index=0, atom_symbol="N"),
        DenticitySite(atom_index=5, atom_symbol="N"),
    ],
    bite_angle=85.0,  # Typical en chelation angle
    charge=0,
    is_bridging=False,
)

EN_BRIDGING = DenticityMode(
    name="bridging-N,N",
    donor_sites=[
        DenticitySite(atom_index=0, atom_symbol="N", is_bridging=True),
        DenticitySite(atom_index=5, atom_symbol="N", is_bridging=True),
    ],
    bite_angle=85.0,
    charge=0,
    is_bridging=True,
)

EN_MODES = {
    "en:bi": EN_BIDENTATE,
    "en:bridge": EN_BRIDGING,
}

# OH (hydroxide) modes
OH_MONODENTATE = DenticityMode(
    name="monodentate-O",
    donor_sites=[DenticitySite(atom_index=0, atom_symbol="O")],
    charge=-1,
    is_bridging=False,
)

OH_BRIDGING = DenticityMode(
    name="bridging-O",
    donor_sites=[DenticitySite(atom_index=0, atom_symbol="O", is_bridging=True)],
    charge=-1,
    is_bridging=True,
)

OH_MODES = {
    "OH:mono": OH_MONODENTATE,
    "OH:bridge": OH_BRIDGING,
    "mu-OH": OH_BRIDGING,  # Traditional notation
}

# CO modes
CO_MONODENTATE = DenticityMode(
    name="monodentate-C",
    donor_sites=[DenticitySite(atom_index=0, atom_symbol="C")],
    charge=0,
    is_bridging=False,
)

CO_BRIDGING = DenticityMode(
    name="bridging-C",
    donor_sites=[DenticitySite(atom_index=0, atom_symbol="C", is_bridging=True)],
    charge=0,
    is_bridging=True,
)

CO_MODES = {
    "CO:mono": CO_MONODENTATE,
    "CO:bridge": CO_BRIDGING,
    "mu-CO": CO_BRIDGING,  # Traditional notation
}

# ============================================================================
# LIGAND REGISTRY
# ============================================================================

EXTENDED_LIGAND_LIBRARY = {
    # New formate and formic acid ligands
    "HCOO": HCOO,
    "formate": HCOO,
    "HCOOH": HCOOH,
    "formic_acid": HCOOH,
    
    # Variable denticity mode specifiers for existing ligands
    # Bipyridine
    "bpy:mono": {"base_ligand": "bpy", "mode": BIPY_MONODENTATE},
    "bpy:bi": {"base_ligand": "bpy", "mode": BIPY_BIDENTATE},
    "bpy:bridge": {"base_ligand": "bpy", "mode": BIPY_BRIDGING},
    "bipy:mono": {"base_ligand": "bpy", "mode": BIPY_MONODENTATE},
    "bipy:bi": {"base_ligand": "bpy", "mode": BIPY_BIDENTATE},
    "bipy:bridge": {"base_ligand": "bpy", "mode": BIPY_BRIDGING},
    
    # Ethylenediamine
    "en:bi": {"base_ligand": "en", "mode": EN_BIDENTATE},
    "en:bridge": {"base_ligand": "en", "mode": EN_BRIDGING},
    
    # Hydroxide
    "OH:mono": {"base_ligand": "OH", "mode": OH_MONODENTATE},
    "OH:bridge": {"base_ligand": "OH", "mode": OH_BRIDGING},
    "mu-OH": {"base_ligand": "OH", "mode": OH_BRIDGING},
    
    # Carbonyl
    "CO:mono": {"base_ligand": "CO", "mode": CO_MONODENTATE},
    "CO:bridge": {"base_ligand": "CO", "mode": CO_BRIDGING},
    "mu-CO": {"base_ligand": "CO", "mode": CO_BRIDGING},
}


def get_ligand_with_mode(ligand_spec: str, base_get_ligand_fn):
    """
    Get a ligand with an optional denticity mode modifier.
    
    Args:
        ligand_spec: Ligand name, optionally with mode (e.g., "bpy", "bpy:mono", "bpy:bi")
        base_get_ligand_fn: Function to get base ligand from original library
    
    Returns:
        Tuple of (ligand_obj, denticity_mode or None)
    """
    # Check if this is in extended library
    if ligand_spec in EXTENDED_LIGAND_LIBRARY:
        entry = EXTENDED_LIGAND_LIBRARY[ligand_spec]
        if isinstance(entry, dict):
            if "base_ligand" in entry:
                # This is a mode modifier entry
                base_lig = base_get_ligand_fn(entry["base_ligand"])
                return base_lig, entry["mode"]
            elif "variable_denticity" in entry:
                # This is a new variable denticity ligand
                return entry, None
        return entry, None
    
    # Fall back to base library
    return base_get_ligand_fn(ligand_spec), None
