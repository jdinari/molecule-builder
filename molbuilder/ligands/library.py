"""
library.py
==========
Complete ligand library for molbuilder.

Ligand names support a colon-mode suffix for denticity:
  HCOO        → monodentate formate (default)
  HCOO:bi     → bidentate chelating formate
  HCOO:bridge → bridging formate (same as mu-HCOO)
  bpy:bi      → bidentate bipyridine (default for bpy)
  bpy:mono    → monodentate bipyridine

Each entry: {smiles, donors, charge, denticity, donor_atoms, [bite_angle], [is_bridging]}
  donors      – atom indices in SMILES that are the donor atoms
  donor_atoms – element symbols of those donors (used for bond-length lookup)
"""

LIGAND_LIBRARY = {
    # ── monodentate neutral ────────────────────────────────────────────────────
    "CO":       {"smiles": "[C-]#[O+]",  "donors": [0], "charge":  0, "denticity": 1, "donor_atoms": ["C"]},
    "H2O":      {"smiles": "O",           "donors": [0], "charge":  0, "denticity": 1, "donor_atoms": ["O"]},
    "aqua":     {"smiles": "O",           "donors": [0], "charge":  0, "denticity": 1, "donor_atoms": ["O"]},
    "NH3":      {"smiles": "N",           "donors": [0], "charge":  0, "denticity": 1, "donor_atoms": ["N"]},
    "ammine":   {"smiles": "N",           "donors": [0], "charge":  0, "denticity": 1, "donor_atoms": ["N"]},
    "py":       {"smiles": "c1ccncc1",    "donors": [3], "charge":  0, "denticity": 1, "donor_atoms": ["N"]},
    "pyridine": {"smiles": "c1ccncc1",    "donors": [3], "charge":  0, "denticity": 1, "donor_atoms": ["N"]},
    "MeCN":     {"smiles": "CC#N",        "donors": [2], "charge":  0, "denticity": 1, "donor_atoms": ["N"]},
    "acetonitrile": {"smiles": "CC#N",    "donors": [2], "charge":  0, "denticity": 1, "donor_atoms": ["N"]},
    "PPh3":     {"smiles": "P(c1ccccc1)(c1ccccc1)c1ccccc1", "donors": [0], "charge": 0, "denticity": 1, "donor_atoms": ["P"]},
    "PMe3":     {"smiles": "CP(C)C",      "donors": [1], "charge":  0, "denticity": 1, "donor_atoms": ["P"]},
    "PH3":      {"smiles": "P",           "donors": [0], "charge":  0, "denticity": 1, "donor_atoms": ["P"]},
    "dmso":     {"smiles": "CS(=O)C",     "donors": [1], "charge":  0, "denticity": 1, "donor_atoms": ["S"]},
    "DMSO":     {"smiles": "CS(=O)C",     "donors": [1], "charge":  0, "denticity": 1, "donor_atoms": ["S"]},
    "NO":       {"smiles": "[N]=O",       "donors": [0], "charge":  1, "denticity": 1, "donor_atoms": ["N"]},

    # ── monodentate anionic ────────────────────────────────────────────────────
    "Cl":       {"smiles": "[Cl-]",       "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["Cl"]},
    "Br":       {"smiles": "[Br-]",       "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["Br"]},
    "I":        {"smiles": "[I-]",        "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["I"]},
    "F":        {"smiles": "[F-]",        "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["F"]},
    "CN":       {"smiles": "[C-]#N",      "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["C"]},
    "OH":       {"smiles": "[OH-]",       "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["O"]},
    "O2-":      {"smiles": "[O-2]",       "donors": [0], "charge": -2, "denticity": 1, "donor_atoms": ["O"]},
    "SCN":      {"smiles": "[S-]C#N",     "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["S"]},
    "NCS":      {"smiles": "N=C=[S]",     "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["N"]},
    "N3":       {"smiles": "[N-]=[N+]=[N-]", "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["N"]},
    "NO2":      {"smiles": "[N+](=O)[O-]","donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["N"]},
    "ONO":      {"smiles": "[O-]N=O",     "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["O"]},
    "H":        {"smiles": "[H-]",        "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["H"]},
    "hydride":  {"smiles": "[H-]",        "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["H"]},
    "Me":       {"smiles": "[CH3-]",      "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["C"]},
    "Ph":       {"smiles": "[c-]1ccccc1", "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["C"]},

    # ── carboxylates (formate and acetate) ────────────────────────────────────
    # Formate HCOO- : monodentate (default), bidentate chelating (:bi), bridging (:bridge / mu-HCOO)
    "HCOO":         {"smiles": "[O-]C=O", "donors": [0],    "charge": -1, "denticity": 1, "donor_atoms": ["O"]},
    "formate":      {"smiles": "[O-]C=O", "donors": [0],    "charge": -1, "denticity": 1, "donor_atoms": ["O"]},
    "HCOO:mono":    {"smiles": "[O-]C=O", "donors": [0],    "charge": -1, "denticity": 1, "donor_atoms": ["O"]},
    "HCOO:bi":      {"smiles": "[O-]C=O", "donors": [0, 2], "charge": -1, "denticity": 2, "donor_atoms": ["O", "O"], "bite_angle": 55.0},
    "HCOO:bridge":  {"smiles": "[O-]C=O", "donors": [0],    "charge": -1, "denticity": 1, "donor_atoms": ["O"], "is_bridging": True},
    "mu-HCOO":      {"smiles": "[O-]C=O", "donors": [0],    "charge": -1, "denticity": 1, "donor_atoms": ["O"], "is_bridging": True},
    # Formic acid HCOOH: neutral, mono or bidentate
    # Formic acid HCOOH: coordinates via carbonyl O (C=O), index 2 in SMILES OC=O
    "HCOOH":        {"smiles": "OC=O",    "donors": [2],    "charge":  0, "denticity": 1, "donor_atoms": ["O"]},
    "HCOOH:mono":   {"smiles": "OC=O",    "donors": [2],    "charge":  0, "denticity": 1, "donor_atoms": ["O"]},
    "HCOOH:bi":     {"smiles": "OC=O",    "donors": [0, 2], "charge":  0, "denticity": 2, "donor_atoms": ["O", "O"], "bite_angle": 52.0},
    # Acetate OAc-
    "OAc":          {"smiles": "CC(=O)[O-]", "donors": [2], "charge": -1, "denticity": 1, "donor_atoms": ["O"]},
    "acetate":      {"smiles": "CC(=O)[O-]", "donors": [2], "charge": -1, "denticity": 1, "donor_atoms": ["O"]},
    "OAc:bi":       {"smiles": "CC(=O)[O-]", "donors": [2, 3], "charge": -1, "denticity": 2, "donor_atoms": ["O", "O"], "bite_angle": 59.0},
    "mu-OAc":       {"smiles": "CC(=O)[O-]", "donors": [2], "charge": -1, "denticity": 1, "donor_atoms": ["O"], "is_bridging": True},

    # ── bidentate neutral ──────────────────────────────────────────────────────
    "en":       {"smiles": "NCCN",        "donors": [0, 3], "charge":  0, "denticity": 2, "donor_atoms": ["N", "N"], "bite_angle": 85.0},
    "bpy":      {"smiles": "c1ccnc(-c2ccccn2)c1", "donors": [3, 10], "charge": 0, "denticity": 2, "donor_atoms": ["N", "N"], "bite_angle": 72.0},
    "bpy:bi":   {"smiles": "c1ccnc(-c2ccccn2)c1", "donors": [3, 10], "charge": 0, "denticity": 2, "donor_atoms": ["N", "N"], "bite_angle": 72.0},
    "bpy:mono": {"smiles": "c1ccnc(-c2ccccn2)c1", "donors": [3],     "charge": 0, "denticity": 1, "donor_atoms": ["N"]},
    "bipy":     {"smiles": "c1ccnc(-c2ccccn2)c1", "donors": [3, 10], "charge": 0, "denticity": 2, "donor_atoms": ["N", "N"], "bite_angle": 72.0},
    "phen":     {"smiles": "c1cnc2ccc3cccnc3c2c1", "donors": [2, 9], "charge": 0, "denticity": 2, "donor_atoms": ["N", "N"], "bite_angle": 78.0},
    "dppm":     {"smiles": "P(CP(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc1", "donors": [0, 2], "charge": 0, "denticity": 2, "donor_atoms": ["P", "P"], "bite_angle": 72.0},
    "dppe":     {"smiles": "P(CCP(c1ccccc1)c1ccccc1)(c1ccccc1)c1ccccc1", "donors": [0, 3], "charge": 0, "denticity": 2, "donor_atoms": ["P", "P"], "bite_angle": 85.0},

    # ── bidentate anionic ──────────────────────────────────────────────────────
    "acac":         {"smiles": "CC(=O)CC(=O)C", "donors": [1, 4], "charge": -1, "denticity": 2, "donor_atoms": ["O", "O"], "bite_angle": 90.0},
    "ox":           {"smiles": "[O-]C(=O)C(=O)[O-]", "donors": [0, 5], "charge": -2, "denticity": 2, "donor_atoms": ["O", "O"], "bite_angle": 80.0},
    "oxalate":      {"smiles": "[O-]C(=O)C(=O)[O-]", "donors": [0, 5], "charge": -2, "denticity": 2, "donor_atoms": ["O", "O"], "bite_angle": 80.0},
    "glycinate":    {"smiles": "[NH2]CC(=O)[O-]", "donors": [0, 4], "charge": -1, "denticity": 2, "donor_atoms": ["N", "O"], "bite_angle": 82.0},
    "catecholate":  {"smiles": "[O-]c1ccccc1[O-]", "donors": [0, 7], "charge": -2, "denticity": 2, "donor_atoms": ["O", "O"], "bite_angle": 82.0},

    # ── tridentate ────────────────────────────────────────────────────────────
    "tpy":      {"smiles": "c1ccnc(-c2cc(-c3ccccn3)ccn2)c1", "donors": [3, 6, 14], "charge": 0, "denticity": 3, "donor_atoms": ["N", "N", "N"], "bite_angle": 72.0},
    "terpy":    {"smiles": "c1ccnc(-c2cc(-c3ccccn3)ccn2)c1", "donors": [3, 6, 14], "charge": 0, "denticity": 3, "donor_atoms": ["N", "N", "N"], "bite_angle": 72.0},
    "dien":     {"smiles": "NCCNCCN", "donors": [0, 3, 6], "charge": 0, "denticity": 3, "donor_atoms": ["N", "N", "N"], "bite_angle": 85.0},

    # ── hexadentate ───────────────────────────────────────────────────────────
    "EDTA":     {"smiles": "[N](CC[N](CC(=O)[O-])CC(=O)[O-])(CC(=O)[O-])CC(=O)[O-]",
                 "donors": [0, 2, 5, 10, 15, 20], "charge": -4, "denticity": 6,
                 "donor_atoms": ["N", "N", "O", "O", "O", "O"]},

    # ── cyclopentadienyl ──────────────────────────────────────────────────────
    "Cp":       {"smiles": "[cH-]1cccc1",  "donors": [0, 1, 2, 3, 4], "charge": -1, "denticity": 5, "donor_atoms": ["C","C","C","C","C"]},
    "Cp*":      {"smiles": "[c-]1(C)c(C)c(C)c(C)c1C", "donors": [0,1,2,3,4], "charge": -1, "denticity": 5, "donor_atoms": ["C","C","C","C","C"]},

    # ── bridging ligands ──────────────────────────────────────────────────────
    "mu-Cl":    {"smiles": "[Cl-]",   "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["Cl"], "is_bridging": True},
    "mu-OH":    {"smiles": "[OH-]",   "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["O"],  "is_bridging": True},
    "mu-O":     {"smiles": "[O-2]",   "donors": [0], "charge": -2, "denticity": 1, "donor_atoms": ["O"],  "is_bridging": True},
    "mu-CO":    {"smiles": "[C-]#[O+]","donors": [0], "charge":  0, "denticity": 1, "donor_atoms": ["C"], "is_bridging": True},
    "mu-H":     {"smiles": "[H-]",    "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["H"],  "is_bridging": True},
    "mu-CN":    {"smiles": "[C-]#N",  "donors": [0], "charge": -1, "denticity": 1, "donor_atoms": ["C"],  "is_bridging": True},
}


def get_ligand(name: str) -> dict:
    """Return ligand data by name, raising KeyError with helpful message."""
    if name in LIGAND_LIBRARY:
        return dict(LIGAND_LIBRARY[name])
    lower = name.lower()
    for k, v in LIGAND_LIBRARY.items():
        if k.lower() == lower:
            return dict(v)
    available = ", ".join(sorted(LIGAND_LIBRARY.keys()))
    raise KeyError(
        f"Unknown ligand '{name}'.\nAvailable: {available}\n"
        f"You can also pass a SMILES string directly."
    )


def list_ligands() -> list:
    """Return sorted list of all ligand names."""
    return sorted(LIGAND_LIBRARY.keys())
