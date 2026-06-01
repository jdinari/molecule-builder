"""
bond_lengths.py
===============
CSD-averaged M-L bond lengths keyed by (metal, oxidation_state, donor_atom, geometry).
Fallback hierarchy:
  1. Exact match
  2. Same metal/ox/donor, any geometry
  3. Same metal/donor, any oxidation state (averaged)
  4. Sum of Alvarez (2008) covalent radii
"""

# ------------------------------------------------------------------
# Bond length database  {(metal, ox, donor, geometry): length_Angstrom}
# geometry keys: 'oct', 'sqp', 'tet', 'tbp', 'lin', 'tp', 'sqpy',
#                'pbp', 'sapr', 'tpr', 'bent', 'tshaped', 'seesaw', None
# ------------------------------------------------------------------

BOND_DB = {
    # ---- Iron ----
    ("Fe", 2, "O", "oct"): 2.11,
    ("Fe", 3, "O", "oct"): 2.01,
    ("Fe", 2, "N", "oct"): 2.18,
    ("Fe", 3, "N", "oct"): 2.09,
    ("Fe", 2, "Cl", "oct"): 2.35,
    ("Fe", 3, "Cl", "oct"): 2.27,
    ("Fe", 2, "C", "oct"): 1.91,
    ("Fe", 3, "C", "oct"): 1.87,
    ("Fe", 2, "S", "oct"): 2.44,
    ("Fe", 3, "S", "oct"): 2.30,
    ("Fe", 2, "P", "oct"): 2.27,
    # ---- Cobalt ----
    ("Co", 2, "O", "oct"): 2.08,
    ("Co", 3, "O", "oct"): 1.92,
    ("Co", 2, "N", "oct"): 2.14,
    ("Co", 3, "N", "oct"): 1.97,
    ("Co", 2, "Cl", "oct"): 2.50,
    ("Co", 3, "Cl", "oct"): 2.27,
    ("Co", 2, "C", "oct"): 1.96,
    ("Co", 3, "C", "oct"): 1.89,
    # ---- Nickel ----
    ("Ni", 2, "O", "oct"): 2.06,
    ("Ni", 2, "N", "oct"): 2.11,
    ("Ni", 2, "Cl", "oct"): 2.40,
    ("Ni", 2, "S", "oct"): 2.43,
    ("Ni", 2, "P", "oct"): 2.21,
    ("Ni", 2, "O", "sqp"): 1.94,
    ("Ni", 2, "N", "sqp"): 1.90,
    ("Ni", 2, "Cl", "sqp"): 2.20,
    ("Ni", 2, "P", "sqp"): 2.14,
    ("Ni", 2, "C", "sqp"): 1.88,
    ("Ni", 2, "O", "tet"): 2.00,
    ("Ni", 2, "Cl", "tet"): 2.26,
    # ---- Copper ----
    ("Cu", 1, "Cl", "tet"): 2.25,
    ("Cu", 1, "N", "tet"): 2.00,
    ("Cu", 1, "P", "tet"): 2.22,
    ("Cu", 1, "C", "lin"): 1.86,
    ("Cu", 2, "O", "sqp"): 1.97,
    ("Cu", 2, "N", "sqp"): 2.02,
    ("Cu", 2, "Cl", "sqp"): 2.28,
    ("Cu", 2, "O", "oct"): 1.97,
    ("Cu", 2, "N", "oct"): 2.00,
    ("Cu", 2, "Cl", "oct"): 2.28,
    # ---- Zinc ----
    ("Zn", 2, "O", "tet"): 1.95,
    ("Zn", 2, "N", "tet"): 2.01,
    ("Zn", 2, "Cl", "tet"): 2.24,
    ("Zn", 2, "S", "tet"): 2.35,
    ("Zn", 2, "O", "oct"): 2.07,
    ("Zn", 2, "N", "oct"): 2.14,
    ("Zn", 2, "Cl", "oct"): 2.35,
    # ---- Manganese ----
    ("Mn", 2, "O", "oct"): 2.18,
    ("Mn", 2, "N", "oct"): 2.24,
    ("Mn", 2, "Cl", "oct"): 2.50,
    ("Mn", 3, "O", "oct"): 2.01,
    ("Mn", 3, "N", "oct"): 2.09,
    ("Mn", 3, "Cl", "oct"): 2.33,
    ("Mn", 4, "O", "oct"): 1.90,
    # ---- Chromium ----
    ("Cr", 0, "C", "oct"): 1.91,
    ("Cr", 2, "O", "oct"): 2.09,
    ("Cr", 3, "O", "oct"): 1.97,
    ("Cr", 3, "N", "oct"): 2.08,
    ("Cr", 3, "Cl", "oct"): 2.31,
    ("Cr", 6, "O", "tet"): 1.65,
    # ---- Vanadium ----
    ("V",  2, "O", "oct"): 2.15,
    ("V",  3, "O", "oct"): 2.00,
    ("V",  4, "O", "oct"): 1.91,
    ("V",  5, "O", "tet"): 1.70,
    ("V",  3, "Cl", "oct"): 2.35,
    ("V",  3, "N", "oct"): 2.10,
    # ---- Titanium ----
    ("Ti", 4, "O", "oct"): 1.96,
    ("Ti", 4, "Cl", "oct"): 2.35,
    ("Ti", 4, "N", "oct"): 2.08,
    # ---- Molybdenum ----
    ("Mo", 0, "C", "oct"): 2.05,
    ("Mo", 2, "Cl", "oct"): 2.47,
    ("Mo", 4, "O", "oct"): 1.96,
    ("Mo", 6, "O", "tet"): 1.72,
    # ---- Tungsten ----
    ("W",  0, "C", "oct"): 2.06,
    ("W",  4, "O", "oct"): 1.97,
    ("W",  6, "O", "tet"): 1.72,
    # ---- Palladium ----
    ("Pd", 2, "Cl", "sqp"): 2.31,
    ("Pd", 2, "N", "sqp"): 2.03,
    ("Pd", 2, "O", "sqp"): 2.02,
    ("Pd", 2, "P", "sqp"): 2.30,
    ("Pd", 2, "C", "sqp"): 2.02,
    # ---- Platinum ----
    ("Pt", 2, "Cl", "sqp"): 2.30,
    ("Pt", 2, "N", "sqp"): 2.02,
    ("Pt", 2, "P", "sqp"): 2.25,
    ("Pt", 2, "C", "sqp"): 2.00,
    ("Pt", 4, "Cl", "oct"): 2.30,
    ("Pt", 4, "N", "oct"): 2.04,
    # ---- Rhodium ----
    ("Rh", 1, "Cl", "sqp"): 2.37,
    ("Rh", 1, "C", "sqp"): 1.84,
    ("Rh", 1, "P", "sqp"): 2.30,
    ("Rh", 3, "Cl", "oct"): 2.34,
    ("Rh", 3, "N", "oct"): 2.06,
    ("Rh", 3, "C", "oct"): 1.97,
    # ---- Iridium ----
    ("Ir", 1, "Cl", "sqp"): 2.36,
    ("Ir", 1, "C", "sqp"): 1.90,
    ("Ir", 3, "Cl", "oct"): 2.35,
    ("Ir", 3, "N", "oct"): 2.07,
    ("Ir", 3, "C", "oct"): 2.03,
    # ---- Ruthenium ----
    ("Ru", 0, "C", "oct"): 1.92,
    ("Ru", 2, "N", "oct"): 2.06,
    ("Ru", 2, "Cl", "oct"): 2.40,
    ("Ru", 2, "P", "oct"): 2.36,
    ("Ru", 3, "Cl", "oct"): 2.34,
    ("Ru", 3, "N", "oct"): 2.10,
    # ---- Osmium ----
    ("Os", 0, "C", "oct"): 1.93,
    ("Os", 2, "N", "oct"): 2.07,
    ("Os", 2, "Cl", "oct"): 2.39,
    # ---- Gold ----
    ("Au", 1, "P", "lin"): 2.28,
    ("Au", 1, "Cl", "lin"): 2.26,
    ("Au", 3, "Cl", "sqp"): 2.27,
    ("Au", 3, "N", "sqp"): 2.03,
    # ---- Silver ----
    ("Ag", 1, "N", "tet"): 2.26,
    ("Ag", 1, "O", "tet"): 2.34,
    ("Ag", 1, "P", "tet"): 2.42,
    ("Ag", 1, "Cl", "tet"): 2.48,
    # ---- Rhenium ----
    ("Re", 1, "C", "oct"): 1.92,
    ("Re", 3, "Cl", "oct"): 2.41,
    # ---- Lanthanides (representative) ----
    ("La", 3, "O", "oct"): 2.52,
    ("La", 3, "N", "oct"): 2.62,
    ("Eu", 3, "O", "oct"): 2.39,
    ("Gd", 3, "O", "oct"): 2.36,
}


# Alvarez 2008 covalent radii (Angstrom) for fallback
COVALENT_RADII = {
    "H":  0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B":  0.84,
    "C":  0.76, "N":  0.71, "O":  0.66, "F":  0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P":  1.07,
    "S":  1.05, "Cl": 1.02, "Ar": 1.06, "K":  2.03, "Ca": 1.76,
    "Sc": 1.70, "Ti": 1.60, "V":  1.53, "Cr": 1.39, "Mn": 1.61,
    "Fe": 1.52, "Co": 1.50, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
    "Kr": 1.16, "Rb": 2.20, "Sr": 1.95, "Y":  1.90, "Zr": 1.75,
    "Nb": 1.64, "Mo": 1.54, "Tc": 1.47, "Ru": 1.46, "Rh": 1.42,
    "Pd": 1.39, "Ag": 1.45, "Cd": 1.44, "In": 1.42, "Sn": 1.39,
    "Sb": 1.39, "Te": 1.38, "I":  1.39, "Xe": 1.40, "Cs": 2.44,
    "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Pr": 2.03, "Nd": 2.01,
    "Pm": 1.99, "Sm": 1.98, "Eu": 1.98, "Gd": 1.96, "Tb": 1.94,
    "Dy": 1.92, "Ho": 1.92, "Er": 1.89, "Tm": 1.90, "Yb": 1.87,
    "Lu": 1.87, "Hf": 1.75, "Ta": 1.70, "W":  1.62, "Re": 1.51,
    "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36, "Hg": 1.32,
    "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50,
    "Rn": 1.50,
}


def get_bond_length(metal: str, ox: int, donor_atom: str, geometry: str = None) -> float:
    """
    Look up M-L bond length with fallback hierarchy.
    """
    # Normalise geometry string
    geom_aliases = {
        "oh": "oct", "octahedral": "oct",
        "sp": "sqp", "square_planar": "sqp", "square-planar": "sqp",
        "td": "tet", "tetrahedral": "tet",
        "linear": "lin",
        "spy": "sqpy", "square_pyramidal": "sqpy",
        "trig_bipy": "tbp", "trigonal_bipyramidal": "tbp",
    }
    if geometry:
        geometry = geom_aliases.get(geometry.lower(), geometry.lower())

    # 1. Exact match
    key = (metal, ox, donor_atom, geometry)
    if key in BOND_DB:
        return BOND_DB[key]

    # 2. Same metal/ox/donor, any geometry
    matches = [v for (m, o, d, g), v in BOND_DB.items()
               if m == metal and o == ox and d == donor_atom]
    if matches:
        return sum(matches) / len(matches)

    # 3. Same metal/donor, any ox (averaged)
    matches = [v for (m, o, d, g), v in BOND_DB.items()
               if m == metal and d == donor_atom]
    if matches:
        return sum(matches) / len(matches)

    # 4. Covalent radii fallback
    r_metal = COVALENT_RADII.get(metal, 1.5)
    r_donor = COVALENT_RADII.get(donor_atom, 0.7)
    return r_metal + r_donor + 0.1   # +0.1 Angstrom empirical correction for dative bonds
