"""
ligand_geometry.py
==================
Place ligand atoms in 3D given:
  - The donor atom's absolute position
  - The metal's absolute position
  - The ligand's internal geometry (bond lengths, angles, torsions)

Convention
----------
All ligands are defined as trees rooted at the donor atom.
The M-donor-C angle determines where the first non-donor atom goes.
The torsion around the M-donor axis is optimised to point the ligand
bulk into the largest gap between adjacent coordination sites.

This eliminates the "bulk along -x" local-frame approach that caused
the C atom to end up too close to the metal.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional

# ── RDKit (optional) ─────────────────────────────────────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:
    _RDKIT = False


# ── Ligand internal geometry definitions ─────────────────────────────────────
# Each entry defines the tree of atoms from the donor outward.
# Format:
#   "NAME": {
#       "donor": element of donor atom,
#       "mdc_angle": M-donor-C angle in degrees (angle at donor between metal and next atom)
#       "atoms": list of (symbol, bond_length, parent_index, angle_from_parent_bond, dihedral_from_grandparent)
#                parent_index is into the atoms list (0 = donor)
#                angle_from_parent_bond = bond angle at the parent atom (degrees)
#                dihedral = dihedral angle (degrees); None = free (optimised)
#   }
#
# The donor is atom 0.  The first child's placement uses mdc_angle.
# Free dihedrals (None) are optimised to maximise distance from adjacent donors.

_LIGAND_DEFS = {

    # ── Water: O-H = 0.957 Å, H-O-H = 104.5° ─────────────────────────────────
    # M-O-H angle: same as H-O-H bisector, ~127.75° from M
    # Both H atoms are free torsion (placed symmetrically)
    "H2O": {
        "donor": "O",
        "mdc_angle": 127.75,   # M-O-(bisector of H2)
        "atoms": [
            # (symbol, bond, parent_idx, angle_at_parent, dihedral)
            ("O",  0.000, None,  None,   None),   # 0: donor
            ("H",  0.957,    0,  127.75, None),   # 1: first H,  free torsion
            ("H",  0.957,    0,  127.75, 180.0),  # 2: second H, opposite torsion
        ],
        "h_oh_angle": 104.5,   # stored for reference
    },
    "aqua": None,   # alias, handled below

    # ── Ammonia: N-H = 1.012 Å, H-N-H = 107.8° ───────────────────────────────
    "NH3": {
        "donor": "N",
        "mdc_angle": 111.5,   # M-N-H angle (tetrahedral-like)
        "atoms": [
            ("N",  0.000, None,  None,   None),
            ("H",  1.012,    0,  111.5,  None),
            ("H",  1.012,    0,  111.5,  120.0),
            ("H",  1.012,    0,  111.5,  240.0),
        ],
    },
    "ammine": None,

    # ── Hydroxide: O-H = 0.960 Å, M-O-H = 120° ───────────────────────────────
    "OH": {
        "donor": "O",
        "mdc_angle": 120.0,
        "atoms": [
            ("O",  0.000, None,  None,   None),
            ("H",  0.960,    0,  120.0,  None),
        ],
    },
    "mu-OH": None,

    # ── Formate HCOO-: monodentate through carboxylate O ─────────────────────
    # M-O-C = 120° (sp2 O); formate is planar; torsion free
    # Atoms: O(donor), C, O2, H
    "HCOO": {
        "donor": "O",
        "mdc_angle": 120.0,   # M-O-C angle
        "atoms": [
            ("O",  0.000, None, None,   None),   # 0: donor O
            ("C",  1.270,    0, 120.0,  None),   # 1: C, free torsion around M-O
            ("O",  1.250,    1, 120.0,  0.0),    # 2: C=O (other O), syn to donor O
            ("H",  1.090,    1, 120.0,  180.0),  # 3: H on C, anti to donor O
        ],
    },
    "formate":   None,
    "HCOO:mono": None,
    "HCOO:bridge": None,
    "mu-HCOO":   None,

    # ── Formic acid HCOOH: coordinates through carbonyl O (C=O) ───────────────
    # M-O=C = 125°; O-H points away
    "HCOOH": {
        "donor": "O",
        "mdc_angle": 125.0,   # M-O=C angle at carbonyl O
        "atoms": [
            ("O",  0.000, None, None,   None),   # 0: carbonyl O (donor)
            ("C",  1.200,    0, 125.0,  None),   # 1: C, free torsion
            ("O",  1.340,    1, 124.0,  0.0),    # 2: hydroxyl O, syn to donor O
            ("H",  1.090,    1, 116.0,  180.0),  # 3: formyl H, anti to donor O
            ("H",  0.972,    2, 109.5,  180.0),  # 4: hydroxyl H, away from C
        ],
    },
    "HCOOH:mono": None,

    # ── Carbon monoxide: C donor, linear ──────────────────────────────────────
    "CO": {
        "donor": "C",
        "mdc_angle": 180.0,
        "atoms": [
            ("C",  0.000, None, None,   None),
            ("O",  1.128,    0, 180.0,  None),
        ],
    },

    # ── Cyanide: C donor ──────────────────────────────────────────────────────
    "CN": {
        "donor": "C",
        "mdc_angle": 180.0,
        "atoms": [
            ("C",  0.000, None, None,   None),
            ("N",  1.160,    0, 180.0,  None),
        ],
    },

    # ── Nitric oxide: N donor, linear ─────────────────────────────────────────
    "NO": {
        "donor": "N",
        "mdc_angle": 180.0,
        "atoms": [
            ("N",  0.000, None, None,   None),
            ("O",  1.150,    0, 180.0,  None),
        ],
    },

    # ── Halides: single atom ───────────────────────────────────────────────────
    "Cl":  {"donor": "Cl", "mdc_angle": None, "atoms": [("Cl", 0.000, None, None, None)]},
    "Br":  {"donor": "Br", "mdc_angle": None, "atoms": [("Br", 0.000, None, None, None)]},
    "I":   {"donor": "I",  "mdc_angle": None, "atoms": [("I",  0.000, None, None, None)]},
    "F":   {"donor": "F",  "mdc_angle": None, "atoms": [("F",  0.000, None, None, None)]},

    # ── Hydride / oxide: single atom ──────────────────────────────────────────
    "H":       {"donor": "H", "mdc_angle": None, "atoms": [("H", 0.000, None, None, None)]},
    "hydride": None,
    "O":       {"donor": "O", "mdc_angle": None, "atoms": [("O", 0.000, None, None, None)]},
    "O2-":     None,

    # ── Thiocyanate / isothiocyanate: linear ──────────────────────────────────
    "SCN": {
        "donor": "S",
        "mdc_angle": 180.0,
        "atoms": [
            ("S", 0.000, None, None,   None),
            ("C", 1.620,    0, 180.0,  None),
            ("N", 1.160,    1, 180.0,  None),
        ],
    },
    "NCS": {
        "donor": "N",
        "mdc_angle": 180.0,
        "atoms": [
            ("N", 0.000, None, None,   None),
            ("C", 1.160,    0, 180.0,  None),
            ("S", 1.620,    1, 180.0,  None),
        ],
    },

    # ── Acetonitrile: N donor, linear chain ───────────────────────────────────
    "MeCN": {
        "donor": "N",
        "mdc_angle": 180.0,
        "atoms": [
            ("N", 0.000, None,  None,   None),
            ("C", 1.160,    0,  180.0,  None),
            ("C", 1.470,    1,  180.0,  None),
            ("H", 1.090,    2,  109.5,  None),
            ("H", 1.090,    2,  109.5,  120.0),
            ("H", 1.090,    2,  109.5,  240.0),
        ],
    },
    "acetonitrile": None,

    # ── Azide: N donor, linear ────────────────────────────────────────────────
    "N3": {
        "donor": "N",
        "mdc_angle": 180.0,
        "atoms": [
            ("N", 0.000, None, None,   None),
            ("N", 1.160,    0, 180.0,  None),
            ("N", 1.160,    1, 180.0,  None),
        ],
    },

    # ── Nitro: N donor, bent ──────────────────────────────────────────────────
    "NO2": {
        "donor": "N",
        "mdc_angle": 120.0,
        "atoms": [
            ("N", 0.000, None, None,   None),
            ("O", 1.240,    0, 120.0,  None),
            ("O", 1.240,    0, 120.0,  180.0),
        ],
    },

    # ── Methyl: C donor, tetrahedral ──────────────────────────────────────────
    "Me": {
        "donor": "C",
        "mdc_angle": 111.5,
        "atoms": [
            ("C", 0.000, None,  None,   None),
            ("H", 1.090,    0,  111.5,  None),
            ("H", 1.090,    0,  111.5,  120.0),
            ("H", 1.090,    0,  111.5,  240.0),
        ],
    },

    # ── Phosphine PH3: P donor ────────────────────────────────────────────────
    "PH3": {
        "donor": "P",
        "mdc_angle": 116.75,  # cone half-angle from M-P bond
        "atoms": [
            ("P", 0.000, None,  None,   None),
            ("H", 1.420,    0,  116.75, None),
            ("H", 1.420,    0,  116.75, 120.0),
            ("H", 1.420,    0,  116.75, 240.0),
        ],
    },

    # ── Bridging ligands ──────────────────────────────────────────────────────
    "mu-Cl": {"donor": "Cl", "mdc_angle": None, "atoms": [("Cl", 0.000, None, None, None)]},
    "mu-O":  {"donor": "O",  "mdc_angle": None, "atoms": [("O",  0.000, None, None, None)]},
    "mu-CO": {
        "donor": "C",
        "mdc_angle": 180.0,
        "atoms": [
            ("C", 0.000, None, None,   None),
            ("O", 1.128,    0, 180.0,  None),
        ],
    },
    "mu-H":  {"donor": "H", "mdc_angle": None, "atoms": [("H", 0.000, None, None, None)]},
    "mu-CN": {
        "donor": "C",
        "mdc_angle": 180.0,
        "atoms": [
            ("C", 0.000, None, None,   None),
            ("N", 1.160,    0, 180.0,  None),
        ],
    },
}

# Resolve aliases
_ALIASES = {
    "aqua":          "H2O",
    "ammine":        "NH3",
    "mu-OH":         "OH",
    "formate":       "HCOO",
    "HCOO:mono":     "HCOO",
    "HCOO:bridge":   "HCOO",
    "mu-HCOO":       "HCOO",
    "HCOOH:mono":    "HCOOH",
    "hydride":       "H",
    "O2-":           "O",
    "acetonitrile":  "MeCN",
}

def _get_def(name: str) -> Optional[dict]:
    name = _ALIASES.get(name, name)
    d = _LIGAND_DEFS.get(name)
    if d is None and name in _ALIASES:
        d = _LIGAND_DEFS.get(_ALIASES[name])
    return d


# ── Rotation utilities ────────────────────────────────────────────────────────

def _rodrigues_rotation(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    v_from = v_from / np.linalg.norm(v_from)
    v_to   = v_to   / np.linalg.norm(v_to)
    axis   = np.cross(v_from, v_to)
    sin_a  = np.linalg.norm(axis)
    cos_a  = np.dot(v_from, v_to)
    if sin_a < 1e-8:
        if cos_a > 0:
            return np.eye(3)
        # 180° rotation around any perpendicular axis
        perp = np.array([1.,0.,0.]) if abs(v_from[0]) < 0.9 else np.array([0.,1.,0.])
        axis = np.cross(v_from, perp); axis /= np.linalg.norm(axis)
        K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
        return np.eye(3) + 2*(K@K)
    axis /= sin_a
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + sin_a*K + (1-cos_a)*(K@K)


def _rot_around_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)


# ── Internal-coordinate placement ────────────────────────────────────────────

def _place_atom(parent_pos: np.ndarray,
                grandparent_pos: Optional[np.ndarray],
                bond_length: float,
                bond_angle_deg: float,
                dihedral_deg: Optional[float],
                default_perp: np.ndarray = np.array([0., 0., 1.])) -> np.ndarray:
    """
    Place a child atom using internal coordinates.

    bond_angle_deg is the TRUE bond angle at parent between
    grandparent->parent->child (standard chemistry convention).
    e.g. 120° for sp2 carbon, 109.5° for sp3, 180° for linear.
    """
    if grandparent_pos is None:
        parent_to_child = np.array([0., 0., 1.])
        return parent_pos + bond_length * parent_to_child

    gp_to_p = parent_pos - grandparent_pos
    gp_to_p_norm = gp_to_p / np.linalg.norm(gp_to_p)

    # Reference perpendicular for dihedral
    perp = np.cross(gp_to_p_norm, default_perp)
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(gp_to_p_norm, np.array([1., 0., 0.]))
    if np.linalg.norm(perp) < 1e-6:
        perp = np.cross(gp_to_p_norm, np.array([0., 1., 0.]))
    perp /= np.linalg.norm(perp)

    # Rotate the gp->parent direction by (180 - bond_angle) to get parent->child.
    # This gives the correct bond angle at parent:
    # angle between (parent->grandparent) and (parent->child) = bond_angle_deg
    supplement = 180.0 - bond_angle_deg
    R_angle = _rot_around_axis(perp, np.radians(supplement))
    child_dir = R_angle @ gp_to_p_norm

    if dihedral_deg is not None:
        R_dihedral = _rot_around_axis(gp_to_p_norm, np.radians(dihedral_deg))
        child_dir = R_dihedral @ child_dir

    return parent_pos + bond_length * child_dir


# ── Best torsion finder ────────────────────────────────────────────────────────

def _best_torsion(donor_abs: np.ndarray,
                  metal_abs: np.ndarray,
                  child_dir_at_zero_torsion: np.ndarray,
                  adjacent_positions: List[np.ndarray],
                  n_steps: int = 72) -> float:
    """
    Find the torsion angle (rotation around metal->donor axis) that places
    child_dir maximally away from all adjacent donor positions.
    Returns best torsion in radians.
    """
    if not adjacent_positions:
        return 0.0

    axis = donor_abs - metal_abs
    if np.linalg.norm(axis) < 1e-6:
        return 0.0
    axis /= np.linalg.norm(axis)

    best_score = -1.0
    best_theta = 0.0

    for step in range(n_steps):
        theta = 2 * np.pi * step / n_steps
        R = _rot_around_axis(axis, theta)
        child_dir = R @ child_dir_at_zero_torsion
        child_abs = donor_abs + child_dir

        # Score = minimum distance from child to any adjacent donor
        min_d = min(np.linalg.norm(child_abs - adj) for adj in adjacent_positions)
        if min_d > best_score:
            best_score = min_d
            best_theta = theta

    return best_theta


# ── Main placement function ────────────────────────────────────────────────────

def place_ligand(ligand_name: str,
                 donor_abs: np.ndarray,
                 metal_abs: np.ndarray,
                 adjacent_donor_positions: List[np.ndarray]) -> List[Tuple[str, np.ndarray]]:
    """
    Place all atoms of a monodentate ligand in absolute coordinates.

    Parameters
    ----------
    ligand_name             : ligand name (or alias) from the library
    donor_abs               : absolute position of donor atom
    metal_abs               : absolute position of metal
    adjacent_donor_positions: absolute positions of all other donor atoms
                              (used to find the best torsion angle)

    Returns
    -------
    List of (element_symbol, absolute_position_Å)
    The first entry is the donor atom.
    """
    defn = _get_def(ligand_name)
    if defn is None:
        # Unknown ligand — just place donor atom
        return [(ligand_name.split(":")[0], donor_abs)]

    atoms_def = defn["atoms"]
    mdc_angle  = defn.get("mdc_angle")

    if len(atoms_def) == 1:
        # Single-atom ligand (halide, hydride, etc.)
        return [(atoms_def[0][0], donor_abs.copy())]

    # ── Step 1: build the ligand in a local frame ─────────────────────────────
    # The donor is at origin, metal at (+bl, 0, 0) in local frame.
    # The first child is placed at mdc_angle from the M-donor bond.
    # All subsequent atoms use standard IC placement.

    # Build initial positions array (local frame)
    local_pos = [np.zeros(3)]  # atom 0 = donor at origin

    # Axis from donor toward metal (local frame = +x)
    m_dir_local = np.array([1., 0., 0.])

    for i, (sym, bond, parent_idx, angle, dihedral) in enumerate(atoms_def[1:], start=1):
        p_pos = local_pos[parent_idx]

        if parent_idx == 0 and mdc_angle is not None:
            # First child: M-donor-child angle = mdc_angle (true bond angle at donor)
            # Metal is at +x in local frame.
            # The O->M direction is +x. The O->C direction is mdc_angle away from O->M.
            # So child_local = [cos(mdc_angle), 0, sin(mdc_angle)]
            # e.g. mdc_angle=120 → [-0.5, 0, 0.866] — pointing away from metal ✓
            angle_rad = np.radians(mdc_angle)
            child_local = np.array([np.cos(angle_rad), 0., np.sin(angle_rad)])
            child_pos = p_pos + bond * child_local
        else:
            gp_pos = local_pos[atoms_def[i][2] - 1] if parent_idx > 0 else None
            # Use grandparent
            if parent_idx > 0:
                gp_idx = atoms_def[i][2]  # parent_idx of this atom's parent
                # Actually: grandparent of atom i is atoms_def[parent_idx][2]
                gp_of_parent = atoms_def[parent_idx][2]
                gp_pos = local_pos[gp_of_parent] if gp_of_parent is not None else None
            else:
                gp_pos = None

            child_pos = _place_atom(
                p_pos, gp_pos,
                bond, angle if angle else 109.5,
                dihedral,
                default_perp=np.array([0., 1., 0.])
            )

        local_pos.append(child_pos)

    # ── Step 2: transform to absolute frame ──────────────────────────────────
    # Rotate so that local +x (toward metal) maps to actual donor->metal direction
    m_to_d = donor_abs - metal_abs
    if np.linalg.norm(m_to_d) < 1e-6:
        return [(sym, donor_abs) for sym, *_ in atoms_def]

    d_to_m_abs = (metal_abs - donor_abs) / np.linalg.norm(metal_abs - donor_abs)
    R_frame = _rodrigues_rotation(m_dir_local, d_to_m_abs)

    # Rotate all local positions and translate to absolute frame
    abs_pos_base = [donor_abs + R_frame @ p for p in local_pos]

    # ── Step 3: find best torsion around metal->donor axis ────────────────────
    if len(abs_pos_base) > 1 and adjacent_donor_positions:
        # The first child (atom 1) defines the torsion reference direction
        child1_dir = abs_pos_base[1] - donor_abs
        if np.linalg.norm(child1_dir) > 1e-6:
            best_theta = _best_torsion(
                donor_abs, metal_abs,
                child1_dir,
                adjacent_donor_positions
            )
            # Apply torsion to all non-donor atoms
            axis = (donor_abs - metal_abs) / np.linalg.norm(donor_abs - metal_abs)
            R_torsion = _rot_around_axis(axis, best_theta)
            abs_pos_final = [abs_pos_base[0]]   # donor stays fixed
            for pos in abs_pos_base[1:]:
                rotated = R_torsion @ (pos - donor_abs) + donor_abs
                abs_pos_final.append(rotated)
        else:
            abs_pos_final = abs_pos_base
    else:
        abs_pos_final = abs_pos_base

    result = [(atoms_def[i][0], abs_pos_final[i]) for i in range(len(atoms_def))]
    return result


# ── Clash detection ───────────────────────────────────────────────────────────

_MIN_NONBONDED = {
    ("H",  "H"):  1.40,
    ("H",  "C"):  1.70,
    ("H",  "N"):  1.70,
    ("H",  "O"):  1.40,
    ("H",  "S"):  1.90,
    ("H",  "P"):  1.90,
    ("H",  "Cl"): 1.90,
    ("C",  "C"):  2.20,
    ("C",  "N"):  2.10,
    ("C",  "O"):  1.80,
    ("N",  "O"):  1.90,
    ("O",  "O"):  1.70,   # intra-chelate O-O in 4-membered rings ~1.8-1.9 Å
}

_MAX_BOND = {
    ("H",  "C"):  1.15, ("H",  "N"):  1.15, ("H",  "O"):  1.10,
    ("H",  "S"):  1.40, ("H",  "P"):  1.50,
    ("C",  "C"):  1.60, ("C",  "N"):  1.55, ("C",  "O"):  1.55,
    ("C",  "S"):  1.85, ("N",  "N"):  1.50, ("N",  "O"):  1.50, ("O",  "O"):  1.55,
}

def _pair_key(s1, s2):
    return tuple(sorted([s1, s2]))

def _is_bonded(s1, s2, d):
    return d <= _MAX_BOND.get(_pair_key(s1, s2), 2.0)

def _is_clash(s1, s2, d):
    if _is_bonded(s1, s2, d):
        return False
    return d < _MIN_NONBONDED.get(_pair_key(s1, s2), 2.00)

def check_clashes(new_atoms, existing_atoms):
    return [(sn, pn, se, pe, np.linalg.norm(pn-pe))
            for sn, pn in new_atoms
            for se, pe in existing_atoms
            if _is_clash(sn, se, np.linalg.norm(pn-pe))]

def _clash_score(new_atoms, existing_atoms):
    score = 0.0
    for sn, pn in new_atoms:
        for se, pe in existing_atoms:
            d = np.linalg.norm(pn - pe)
            if _is_clash(sn, se, d):
                score += _MIN_NONBONDED.get(_pair_key(sn, se), 2.0) - d
    return score

def resolve_clash_by_rotation(new_atoms, donor_abs, metal_pos, existing_atoms, n_steps=72):
    axis = donor_abs - metal_pos
    if np.linalg.norm(axis) < 1e-6:
        return new_atoms
    axis /= np.linalg.norm(axis)
    best = new_atoms
    best_score = _clash_score(new_atoms, existing_atoms)
    for step in range(1, n_steps):
        theta = 2*np.pi*step/n_steps
        R = _rot_around_axis(axis, theta)
        rotated = [(s, R@(p-metal_pos)+metal_pos) for s,p in new_atoms]
        score = _clash_score(rotated, existing_atoms)
        if score < best_score:
            best_score = score
            best = rotated
        if best_score == 0:
            break
    return best


# ── Legacy compatibility shims ────────────────────────────────────────────────
# These are called by api.py; they now delegate to place_ligand()

def get_ligand_atoms(ligand_name: str, smiles: str, donor_symbol: str,
                     donor_index: int = 0) -> List[Tuple[str, np.ndarray]]:
    """
    Legacy interface: return atom list in OLD local-frame convention
    (donor at origin, metal at +x, bulk along -x) for use in api.py.
    This is only used when adjacent positions are unknown; place_ligand()
    is preferred and called directly from _build_single().
    """
    defn = _get_def(ligand_name)
    if defn is None:
        return [(donor_symbol, np.zeros(3))]
    atoms_def = defn["atoms"]
    if len(atoms_def) == 1:
        return [(atoms_def[0][0], np.zeros(3))]

    # Place with metal at +x, no adjacent donors
    donor_abs = np.zeros(3)
    metal_abs = np.array([2.0, 0., 0.])  # dummy, direction only matters
    result = place_ligand(ligand_name, donor_abs, metal_abs, [])
    # Shift so donor is at origin
    d_pos = result[0][1]
    return [(s, p - d_pos) for s, p in result]


def get_ligand_atoms_multidentate(ligand_name, smiles, donor_symbols, donor_indices,
                                   bite_angle_deg=90.0, bond_lengths=None):
    """Legacy shim for bidentate/multidentate — unchanged."""
    if bond_lengths is None:
        bond_lengths = [2.0] * len(donor_symbols)
    if len(donor_symbols) == 1:
        bl = bond_lengths[0]
        return [(donor_symbols[0], np.array([bl, 0., 0.]))]
    half = np.radians(bite_angle_deg / 2)
    result = []
    for i, (sym, bl) in enumerate(zip(donor_symbols, bond_lengths)):
        sign = 1 if i == 0 else -1
        result.append((sym, np.array([bl*np.cos(sign*half), bl*np.sin(sign*half), 0.])))
    return result


# ── Bidentate formate full-atom placement ─────────────────────────────────────

def place_bidentate_formate(donor1_abs: np.ndarray,
                             donor2_abs: np.ndarray,
                             metal_abs: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Place HCOO:bi (bidentate chelating formate) with full atom geometry.
    Both oxygens are already positioned; this adds C and H.

    Returns list of (symbol, position) for all 4 formate atoms.
    """
    # C is the midpoint between the two O donors + a small displacement
    # In bidentate formate: O-C-O angle = 120°, C-O = 1.26 Å
    # The C sits above the midpoint of O1-O2 at a distance determined by geometry
    # O1-O2 distance in chelating formate: 2*1.26*sin(60°) = 2.18 Å

    o1 = donor1_abs
    o2 = donor2_abs

    # Midpoint of O1-O2
    mid = (o1 + o2) / 2.0

    # C is perpendicular to O1-O2, away from metal
    # Direction of O1-O2
    o12 = o2 - o1
    o12_norm = o12 / np.linalg.norm(o12)

    # Direction away from metal (from midpoint)
    to_metal = metal_abs - mid
    if np.linalg.norm(to_metal) > 1e-6:
        to_metal_norm = to_metal / np.linalg.norm(to_metal)
    else:
        to_metal_norm = np.array([1., 0., 0.])

    # C is displaced perpendicular to O1-O2 and away from metal
    # Find perpendicular to o12 in the plane containing metal
    perp = np.cross(o12_norm, np.cross(o12_norm, to_metal_norm))
    if np.linalg.norm(perp) < 1e-6:
        # Fallback: any perpendicular
        perp = np.cross(o12_norm, np.array([0., 0., 1.]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(o12_norm, np.array([0., 1., 0.]))
    perp = perp / np.linalg.norm(perp)

    # C position: C-O = 1.26 Å, O-C-O = 120° -> height from midpoint
    # h = sqrt(1.26^2 - (|O1O2|/2)^2)
    oo_half = np.linalg.norm(o12) / 2.0
    h = np.sqrt(max(1.26**2 - oo_half**2, 0.01))
    # C points AWAY from metal
    c_pos = mid - h * to_metal_norm

    # H on C: C-H = 1.09 Å, O-C-H = 120°
    # H points away from O1-O2 midpoint (away from metal side)
    h_pos = c_pos - 1.09 * to_metal_norm

    return [
        ("O", o1.copy()),
        ("O", o2.copy()),
        ("C", c_pos),
        ("H", h_pos),
    ]


# ── Bidentate ligand placement ────────────────────────────────────────────────

# Full geometry definitions for bidentate ligands.
# Each entry: {atoms_func: callable(v1, v2, bl1, bl2) -> [(sym, pos), ...]}
# v1, v2 are the two geometry unit vectors from the metal;
# bl1, bl2 are the two Ni-donor bond lengths.

def _place_hcoo_bi(v1: np.ndarray, v2: np.ndarray,
                   bl1: float, bl2: float,
                   metal_pos: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Place bidentate formate HCOO-κ2O,O' in absolute coordinates.
    O-C-O = 120°, O-C = 1.26 Å, C-H = 1.09 Å.
    The C and H sit on the bisector of the two M-O vectors.
    """
    o1 = metal_pos + v1 * bl1
    o2 = metal_pos + v2 * bl2

    # Midpoint direction (bisector of the chelate)
    mid_dir = (v1 + v2)
    if np.linalg.norm(mid_dir) < 1e-6:
        mid_dir = np.cross(v1, np.array([0., 0., 1.]))
    mid_dir /= np.linalg.norm(mid_dir)

    # C position: on the bisector, at distance from metal such that O-C = 1.26 Å
    # O1 is at (v1*bl1); C is at (mid_dir * d_c)
    # |O1 - C|^2 = 1.26^2  →  solve for d_c
    # |mid_dir*d_c - v1*bl1|^2 = 1.26^2
    # d_c^2 - 2*(v1·mid_dir)*bl1*d_c + bl1^2 - 1.26^2 = 0
    a_coef = 1.0
    b_coef = -2.0 * np.dot(v1, mid_dir) * bl1
    c_coef = bl1**2 - 1.26**2
    disc = b_coef**2 - 4*a_coef*c_coef
    if disc < 0:
        disc = 0.0
    d_c = (-b_coef + np.sqrt(disc)) / (2*a_coef)  # take the farther root

    c_pos = metal_pos + mid_dir * d_c

    # H position: on the bisector, beyond C
    h_pos = c_pos + mid_dir * 1.09

    return [("O", o1), ("O", o2), ("C", c_pos), ("H", h_pos)]


def _place_hcooh_bi(v1: np.ndarray, v2: np.ndarray,
                    bl1: float, bl2: float,
                    metal_pos: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Place bidentate formic acid HCOOH-κ2O,O' — same geometry as formate
    but neutral; add the O-H on the hydroxyl oxygen.
    """
    base = _place_hcoo_bi(v1, v2, bl1, bl2, metal_pos)
    # base = [O(carbonyl), O(hydroxyl), C, H(formyl)]
    # Add H on second O (hydroxyl), pointing away from C
    o2_pos = base[1][1]
    c_pos  = base[2][1]
    away   = o2_pos - c_pos
    away  /= np.linalg.norm(away)
    h_oh   = o2_pos + 0.972 * away
    return base + [("H", h_oh)]


_BIDENTATE_PLACERS = {
    "HCOO:bi":   _place_hcoo_bi,
    "HCOOH:bi":  _place_hcooh_bi,
    # For other bidentate ligands without a custom placer, fall back to donor-only
}


def place_bidentate_ligand(ligand_name: str,
                            v1: np.ndarray, v2: np.ndarray,
                            bl1: float, bl2: float,
                            metal_pos: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Place a bidentate ligand given two geometry vectors and bond lengths.

    Parameters
    ----------
    ligand_name : str
    v1, v2      : unit vectors from metal to each donor
    bl1, bl2    : Ni-donor bond lengths
    metal_pos   : absolute position of metal

    Returns
    -------
    List of (symbol, absolute_position)
    """
    placer = _BIDENTATE_PLACERS.get(ligand_name)
    if placer is not None:
        return placer(v1, v2, bl1, bl2, metal_pos)

    # Generic fallback: just place donor atoms
    return [("O", metal_pos + v1 * bl1), ("O", metal_pos + v2 * bl2)]
