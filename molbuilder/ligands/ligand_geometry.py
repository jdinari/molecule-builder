"""
ligand_geometry.py
==================
Generate full 3D geometry for ligands (including H atoms) using RDKit.
Donor atom is placed at origin; bulk of ligand points along -x so that
when the metal is at +x the ligand points away from it.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional

# ── RDKit import (optional but strongly preferred) ───────────────────────────
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")   # suppress all RDKit warnings globally
    _RDKIT = True
except ImportError:
    _RDKIT = False

# ── Fallback: hard-coded geometries for common ligands ───────────────────────
# Convention: donor atom at origin, +x points TOWARD the metal,
# so ligand bulk extends into the -x half-space.
#
# All bond lengths and angles are from standard experimental values:
#   O-H = 0.957 Å, H-O-H = 104.5°
#   N-H = 1.012 Å, H-N-H = 107.8°
#   P-H = 1.420 Å, H-P-H = 93.5°
#   O-H (hydroxide) = 0.960 Å, M-O-H ≈ 120°
#   C=O (formate/CO) = 1.128/1.250 Å,  C-O = 1.260 Å

_FALLBACK_GEOMS = {
    # ── water: H-O-H = 104.5°, O-H = 0.957 Å ────────────────────────────────
    # H at ±52.25° from -x axis in xy-plane
    "H2O": ("O", [
        ("O", [ 0.000,  0.000,  0.000]),
        ("H", [-0.586,  0.757,  0.000]),
        ("H", [-0.586, -0.757,  0.000]),
    ]),
    "aqua": ("O", [
        ("O", [ 0.000,  0.000,  0.000]),
        ("H", [-0.586,  0.757,  0.000]),
        ("H", [-0.586, -0.757,  0.000]),
    ]),

    # ── ammonia: H-N-H = 107.8°, N-H = 1.012 Å, trigonal pyramid ─────────────
    # cone half-angle from -x = 68.91°; 3 H at 120° apart around -x axis
    "NH3": ("N", [
        ("N", [ 0.000,  0.000,  0.000]),
        ("H", [-0.364,  0.944,  0.000]),
        ("H", [-0.364, -0.472,  0.818]),
        ("H", [-0.364, -0.472, -0.818]),
    ]),
    "ammine": ("N", [
        ("N", [ 0.000,  0.000,  0.000]),
        ("H", [-0.364,  0.944,  0.000]),
        ("H", [-0.364, -0.472,  0.818]),
        ("H", [-0.364, -0.472, -0.818]),
    ]),

    # ── hydroxide: O-H = 0.960 Å, M-O-H ≈ 120° (sp3-like) ───────────────────
    "OH": ("O", [
        ("O", [ 0.000,  0.000,  0.000]),
        ("H", [-0.480,  0.831,  0.000]),
    ]),
    "mu-OH": ("O", [
        ("O", [ 0.000,  0.000,  0.000]),
        ("H", [-0.480,  0.831,  0.000]),
    ]),

    # ── phosphine: H-P-H = 93.5°, P-H = 1.420 Å ─────────────────────────────
    # cone half-angle from -x = 57.25°
    "PH3": ("P", [
        ("P", [ 0.000,  0.000,  0.000]),
        ("H", [-0.768,  1.194,  0.000]),
        ("H", [-0.768, -0.597,  1.034]),
        ("H", [-0.768, -0.597, -1.034]),
    ]),

    # ── carbon monoxide: C donor, O at -1.128 Å ──────────────────────────────
    "CO": ("C", [
        ("C", [ 0.000,  0.000,  0.000]),
        ("O", [-1.128,  0.000,  0.000]),
    ]),

    # ── cyanide: C donor ──────────────────────────────────────────────────────
    "CN": ("C", [
        ("C", [ 0.000,  0.000,  0.000]),
        ("N", [-1.160,  0.000,  0.000]),
    ]),

    # ── nitric oxide: N donor ─────────────────────────────────────────────────
    "NO": ("N", [
        ("N", [ 0.000,  0.000,  0.000]),
        ("O", [-1.150,  0.000,  0.000]),
    ]),

    # ── halides: single atom ──────────────────────────────────────────────────
    "Cl":  ("Cl", [("Cl", [0., 0., 0.])]),
    "Br":  ("Br", [("Br", [0., 0., 0.])]),
    "I":   ("I",  [("I",  [0., 0., 0.])]),
    "F":   ("F",  [("F",  [0., 0., 0.])]),

    # ── formate HCOO-: O donor ────────────────────────────────────────────────
    # M-O-C angle = 120° (sp2 carboxylate oxygen).
    # Tail (C, O2, H) placed in the xz plane so adjacent-site clashes are
    # resolvable by rotation around the M→donor axis.
    # O-C = 1.26 Å, C=O = 1.25 Å, C-H = 1.09 Å
    "HCOO": ("O", [
        ("O", [ 0.000,  0.000,  0.000]),
        ("C", [-0.630,  0.000,  1.091]),
        ("O", [-0.005,  0.000,  2.174]),
        ("H", [-1.720,  0.000,  1.091]),
    ]),
    "formate": ("O", [
        ("O", [ 0.000,  0.000,  0.000]),
        ("C", [-0.630,  0.000,  1.091]),
        ("O", [-0.005,  0.000,  2.174]),
        ("H", [-1.720,  0.000,  1.091]),
    ]),
    "HCOO:mono": ("O", [
        ("O", [ 0.000,  0.000,  0.000]),
        ("C", [-0.630,  0.000,  1.091]),
        ("O", [-0.005,  0.000,  2.174]),
        ("H", [-1.720,  0.000,  1.091]),
    ]),
    "HCOO:bridge": ("O", [
        ("O", [ 0.000,  0.000,  0.000]),
        ("C", [-0.630,  0.000,  1.091]),
        ("O", [-0.005,  0.000,  2.174]),
        ("H", [-1.720,  0.000,  1.091]),
    ]),
    "mu-HCOO": ("O", [
        ("O", [ 0.000,  0.000,  0.000]),
        ("C", [-0.630,  0.000,  1.091]),
        ("O", [-0.005,  0.000,  2.174]),
        ("H", [-1.720,  0.000,  1.091]),
    ]),

    # ── formic acid HCOOH: carbonyl O donor (C=O oxygen) ─────────────────────
    # M-O=C angle = 125°. Tail in xz plane for rotation-resolvable clashes.
    # C=O = 1.20 Å, C-O(H) = 1.34 Å, O=C-O(H) = 124°, C-H = 1.09 Å, O-H = 0.972 Å
    "HCOOH": ("O", [
        ("O", [ 0.000,  0.000,  0.000]),   # carbonyl O — donor
        ("C", [-0.688,  0.000,  0.983]),
        ("O", [-0.208,  0.000,  2.234]),   # hydroxyl O
        ("H", [-1.765,  0.000,  0.812]),   # formyl H on C
        ("H", [ 0.764,  0.000,  2.209]),   # hydroxyl H
    ]),
    "HCOOH:mono": ("O", [
        ("O", [ 0.000,  0.000,  0.000]),
        ("C", [-0.688,  0.000,  0.983]),
        ("O", [-0.208,  0.000,  2.234]),
        ("H", [-1.765,  0.000,  0.812]),
        ("H", [ 0.764,  0.000,  2.209]),
    ]),

    # ── hydride ───────────────────────────────────────────────────────────────
    "H":       ("H", [("H", [0., 0., 0.])]),
    "hydride": ("H", [("H", [0., 0., 0.])]),

    # ── thiocyanate: S donor or N donor ──────────────────────────────────────
    "SCN": ("S", [
        ("S", [ 0.000,  0.000,  0.000]),
        ("C", [-1.620,  0.000,  0.000]),
        ("N", [-2.780,  0.000,  0.000]),
    ]),
    "NCS": ("N", [
        ("N", [ 0.000,  0.000,  0.000]),
        ("C", [-1.160,  0.000,  0.000]),
        ("S", [-2.780,  0.000,  0.000]),
    ]),

    # ── oxide ─────────────────────────────────────────────────────────────────
    "O":   ("O", [("O", [0., 0., 0.])]),
    "O2-": ("O", [("O", [0., 0., 0.])]),

    # ── acetonitrile MeCN: N donor ────────────────────────────────────────────
    # N-C≡C linear chain; methyl H at tetrahedral angles
    # N at 0; C(nitrile) at -1.16; C(methyl) at -2.63
    # H at tetrahedral (109.5°) from C-C bond axis, 3 H 120° apart
    # cone half-angle = 90° - 70.5° = 19.5° wait...
    # C-H = 1.09 Å, H-C-C angle = 109.5°, cone from -x:
    # Hx = -2.63 - 1.09*cos(70.5°) = -2.63 - 0.364 = -2.994
    # Hr = 1.09*sin(70.5°) = 1.027
    "MeCN": ("N", [
        ("N", [ 0.000,  0.000,  0.000]),
        ("C", [-1.160,  0.000,  0.000]),
        ("C", [-2.630,  0.000,  0.000]),
        ("H", [-2.994,  1.027,  0.000]),
        ("H", [-2.994, -0.514,  0.890]),
        ("H", [-2.994, -0.514, -0.890]),
    ]),
    "acetonitrile": ("N", [
        ("N", [ 0.000,  0.000,  0.000]),
        ("C", [-1.160,  0.000,  0.000]),
        ("C", [-2.630,  0.000,  0.000]),
        ("H", [-2.994,  1.027,  0.000]),
        ("H", [-2.994, -0.514,  0.890]),
        ("H", [-2.994, -0.514, -0.890]),
    ]),

    # ── azide N3-: N donor, linear ────────────────────────────────────────────
    "N3": ("N", [
        ("N", [ 0.000,  0.000,  0.000]),
        ("N", [-1.160,  0.000,  0.000]),
        ("N", [-2.320,  0.000,  0.000]),
    ]),

    # ── nitro NO2-: N donor, O-N-O = 115° ────────────────────────────────────
    # N at origin; 2 O at ±57.5° from -x axis; N-O = 1.24 Å
    "NO2": ("N", [
        ("N", [ 0.000,  0.000,  0.000]),
        ("O", [-0.672,  1.044,  0.000]),
        ("O", [-0.672, -1.044,  0.000]),
    ]),

    # ── nitrito ONO: O donor ──────────────────────────────────────────────────
    "ONO": ("O", [
        ("O", [ 0.000,  0.000,  0.000]),
        ("N", [-1.220,  0.000,  0.000]),
        ("O", [-1.870, -1.090,  0.000]),
    ]),

    # ── methyl Me-: C donor, tetrahedral ─────────────────────────────────────
    # C at origin; 3 H in trigonal pyramid pointing into -x hemisphere
    # C-H = 1.09 Å, H-C-H = 109.5°  → same cone calculation as NH3
    # cone half-angle = 70.5° from -x
    "Me": ("C", [
        ("C", [ 0.000,  0.000,  0.000]),
        ("H", [-0.363,  1.028,  0.000]),
        ("H", [-0.363, -0.514,  0.890]),
        ("H", [-0.363, -0.514, -0.890]),
    ]),

    # ── phenyl Ph-: C(ipso) donor ─────────────────────────────────────────────
    # Benzene ring in the yz-plane; ipso C at origin; ring extending into -x
    "Ph": ("C", [
        ("C", [ 0.000,  0.000,  0.000]),
        ("C", [-1.400,  0.000,  0.000]),
        ("C", [-2.100,  1.210,  0.000]),
        ("C", [-2.100, -1.210,  0.000]),
        ("C", [-3.500,  1.210,  0.000]),
        ("C", [-3.500, -1.210,  0.000]),
        ("C", [-4.200,  0.000,  0.000]),
        ("H", [-1.570,  2.160,  0.000]),
        ("H", [-1.570, -2.160,  0.000]),
        ("H", [-4.050,  2.160,  0.000]),
        ("H", [-4.050, -2.160,  0.000]),
        ("H", [-5.290,  0.000,  0.000]),
    ]),

    # ── bridging ligands ──────────────────────────────────────────────────────
    "mu-Cl": ("Cl", [("Cl", [0., 0., 0.])]),
    "mu-O":  ("O",  [("O",  [0., 0., 0.])]),
    "mu-CO": ("C", [
        ("C", [ 0.000,  0.000,  0.000]),
        ("O", [-1.128,  0.000,  0.000]),
    ]),
    "mu-H":  ("H",  [("H",  [0., 0., 0.])]),
    "mu-CN": ("C", [
        ("C", [ 0.000,  0.000,  0.000]),
        ("N", [-1.160,  0.000,  0.000]),
    ]),
}


def _rodrigues_rotation(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """Rotation matrix that rotates unit vector v_from onto unit vector v_to."""
    v_from = v_from / np.linalg.norm(v_from)
    v_to   = v_to   / np.linalg.norm(v_to)
    axis = np.cross(v_from, v_to)
    sin_a = np.linalg.norm(axis)
    cos_a = np.dot(v_from, v_to)
    if sin_a < 1e-8:
        return np.eye(3) if cos_a > 0 else _rotation_180(v_from)
    axis /= sin_a
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + sin_a * K + (1 - cos_a) * (K @ K)


def _rotation_180(v: np.ndarray) -> np.ndarray:
    """180° rotation matrix around an axis perpendicular to v."""
    perp = np.array([1., 0., 0.]) if abs(v[0]) < 0.9 else np.array([0., 1., 0.])
    axis = np.cross(v, perp)
    axis /= np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + 2 * (K @ K)


def _get_rdkit_geometry(smiles: str,
                        donor_index: int,
                        donor_symbol: str) -> Optional[List[Tuple[str, np.ndarray]]]:
    """
    Use RDKit to generate 3D geometry with H atoms.
    Returns list of (symbol, position) with donor at origin pointing along +x bulk-side.
    """
    if not _RDKIT:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 42
        if AllChem.EmbedMolecule(mol, ps) == -1:
            AllChem.EmbedMolecule(mol, AllChem.ETDG())
        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()

        atoms = [(a.GetSymbol(), np.array(conf.GetAtomPosition(i)))
                 for i, a in enumerate(mol.GetAtoms())]

        # Translate so donor is at origin
        d_pos = atoms[donor_index][1].copy()
        atoms = [(sym, pos - d_pos) for sym, pos in atoms]

        # Find "bulk direction" = centroid of non-donor heavy atoms
        heavy_non_donor = [pos for i, (sym, pos) in enumerate(atoms)
                           if i != donor_index and sym != 'H']
        if heavy_non_donor:
            bulk = np.mean(heavy_non_donor, axis=0)
        else:
            # Just H atoms (e.g. H2O, NH3) — use their centroid
            h_pos = [pos for i, (sym, pos) in enumerate(atoms)
                     if i != donor_index]
            if h_pos:
                bulk = np.mean(h_pos, axis=0)
            else:
                bulk = np.array([-1., 0., 0.])

        # Rotate so bulk points along -x (away from metal which will be at +x)
        if np.linalg.norm(bulk) > 1e-3:
            R = _rodrigues_rotation(bulk / np.linalg.norm(bulk),
                                    np.array([-1., 0., 0.]))
            atoms = [(sym, R @ pos) for sym, pos in atoms]

        return atoms
    except Exception:
        return None


def get_ligand_atoms(ligand_name: str,
                     smiles: str,
                     donor_symbol: str,
                     donor_index: int = 0) -> List[Tuple[str, np.ndarray]]:
    """
    Return list of (element_symbol, relative_position_Å) for a ligand,
    with the donor atom at origin and ligand bulk pointing along -x.

    The metal will be placed in the +x direction from the donor atom.
    The caller translates & rotates these relative positions into the
    global frame using the geometry vectors.
    """
    # 1. Try hard-coded fallback first (faster, deterministic)
    if ligand_name in _FALLBACK_GEOMS:
        _, atom_list = _FALLBACK_GEOMS[ligand_name]
        return [(sym, np.array(pos, dtype=float)) for sym, pos in atom_list]

    # 2. Try RDKit
    if _RDKIT and smiles:
        result = _get_rdkit_geometry(smiles, donor_index, donor_symbol)
        if result is not None:
            return result

    # 3. Last resort: just the donor atom
    return [(donor_symbol, np.zeros(3))]


def get_ligand_atoms_multidentate(
        ligand_name: str,
        smiles: str,
        donor_symbols: List[str],
        donor_indices: List[int],
        bite_angle_deg: float = 90.0,
        bond_lengths: Optional[List[float]] = None,
) -> List[Tuple[str, np.ndarray]]:
    """
    For multidentate ligands: return full atom list with donor atoms
    pre-positioned at ±bite_angle/2 from the +x axis in the xy-plane.
    The metal is at the origin; donor atoms are at their bond lengths.

    Returns list of (symbol, absolute_position) — NOT relative to donor,
    but in the metal-centred frame ready to be added directly.
    """
    n_donors = len(donor_symbols)
    if bond_lengths is None:
        bond_lengths = [2.0] * n_donors

    if n_donors == 1:
        # Just treat as monodentate
        bl = bond_lengths[0]
        atoms = get_ligand_atoms(ligand_name, smiles, donor_symbols[0], donor_indices[0])
        # Translate: donor is at (bl, 0, 0)
        donor_abs = np.array([bl, 0., 0.])
        result = []
        for sym, rel in atoms:
            result.append((sym, rel + donor_abs))
        return result

    # Place donor atoms symmetrically around +x in the xy-plane
    half = np.radians(bite_angle_deg / 2)
    donor_positions = []
    for i, (bl, sym) in enumerate(zip(bond_lengths, donor_symbols)):
        sign = 1 if i == 0 else -1
        angle = sign * half
        donor_positions.append(np.array([bl * np.cos(angle), bl * np.sin(angle), 0.]))

    # Try to get full ligand from RDKit, then superimpose donors
    if _RDKIT and smiles:
        rdkit_atoms = _get_rdkit_geometry(smiles, donor_indices[0], donor_symbols[0])
        if rdkit_atoms is not None:
            # rdkit_atoms has donor0 at origin; find donor1 position
            # and rotate the whole thing so donors match our target positions
            rdkit_d0 = np.zeros(3)  # donor0 is at origin
            # Find donor1 in rdkit_atoms (index by position in original smiles)
            # Since we only stored by order, use index 1 of heavy atoms after donor0
            # Approximate: the second distinct heavy atom that's a donor type
            rdkit_d1 = None
            heavy = [(sym, pos) for sym, pos in rdkit_atoms
                     if sym not in ('H',) and not np.allclose(pos, 0)]
            if heavy:
                rdkit_d1 = heavy[0][1]

            if rdkit_d1 is not None:
                # Compute rotation: align rdkit donor0->donor1 with target d0->d1
                v_rdkit = rdkit_d1 - rdkit_d0
                v_target = donor_positions[1] - donor_positions[0]
                if np.linalg.norm(v_rdkit) > 1e-3 and np.linalg.norm(v_target) > 1e-3:
                    R = _rodrigues_rotation(v_rdkit / np.linalg.norm(v_rdkit),
                                            v_target / np.linalg.norm(v_target))
                    # Scale: adjust donor positions to our bond lengths
                    scale = np.linalg.norm(v_target) / np.linalg.norm(v_rdkit)
                    result = []
                    for sym, pos in rdkit_atoms:
                        new_pos = R @ (pos * scale) + donor_positions[0]
                        result.append((sym, new_pos))
                    return result

    # Fallback: just put donor atoms at computed positions, no H
    result = []
    for sym, pos in zip(donor_symbols, donor_positions):
        result.append((sym, pos))
    return result


# ── clash detection and resolution ───────────────────────────────────────────

# Minimum acceptable non-bonded distances by element-pair
_MIN_NONBONDED = {
    ("H",  "H"):  1.40,
    ("H",  "C"):  1.70,
    ("H",  "N"):  1.70,
    ("H",  "O"):  1.40,   # H...O contacts: relax for bulky ligand initial structures
    ("H",  "S"):  1.90,
    ("H",  "P"):  1.90,
    ("H",  "Cl"): 1.90,
    ("C",  "C"):  2.20,
    ("C",  "N"):  2.10,
    ("C",  "O"):  1.80,   # formate C is ~1.7-1.9 Å from adjacent donor O in crystal structures
    ("N",  "O"):  1.90,
    ("O",  "O"):  2.00,
}

# Maximum bond lengths (to distinguish bonded from non-bonded short contacts)
_MAX_BOND = {
    ("H",  "C"):  1.15,
    ("H",  "N"):  1.15,
    ("H",  "O"):  1.10,
    ("H",  "S"):  1.40,
    ("H",  "P"):  1.50,
    ("C",  "C"):  1.60,
    ("C",  "N"):  1.55,
    ("C",  "O"):  1.55,
    ("C",  "S"):  1.85,
    ("N",  "N"):  1.50,
    ("N",  "O"):  1.50,
    ("O",  "O"):  1.55,
}


def _pair_key(s1: str, s2: str) -> tuple:
    return tuple(sorted([s1, s2]))


def _is_bonded(s1: str, s2: str, d: float) -> bool:
    return d <= _MAX_BOND.get(_pair_key(s1, s2), 2.0)


def _is_clash(s1: str, s2: str, d: float) -> bool:
    """Return True if two non-bonded atoms are closer than the minimum allowed distance."""
    if _is_bonded(s1, s2, d):
        return False
    return d < _MIN_NONBONDED.get(_pair_key(s1, s2), 2.00)


def check_clashes(new_atoms: List[Tuple[str, np.ndarray]],
                  existing_atoms: List[Tuple[str, np.ndarray]]) -> List[tuple]:
    """
    Find all clashes between new_atoms and existing_atoms.
    Returns list of (sym_new, pos_new, sym_exist, pos_exist, distance).
    """
    clashes = []
    for sn, pn in new_atoms:
        for se, pe in existing_atoms:
            d = np.linalg.norm(pn - pe)
            if _is_clash(sn, se, d):
                clashes.append((sn, pn, se, pe, d))
    return clashes


def resolve_clash_by_rotation(new_atoms: List[Tuple[str, np.ndarray]],
                               donor_abs_pos: np.ndarray,
                               metal_pos: np.ndarray,
                               existing_atoms: List[Tuple[str, np.ndarray]],
                               n_steps: int = 36) -> List[Tuple[str, np.ndarray]]:
    """
    Try rotating the ligand around the metal→donor axis in n_steps increments
    to find the orientation with the fewest / least severe clashes.

    Parameters
    ----------
    new_atoms      : list of (symbol, absolute_position) for the new ligand
    donor_abs_pos  : absolute position of the donor atom
    metal_pos      : absolute position of the metal
    existing_atoms : already-placed atoms to check against
    n_steps        : number of rotation increments to try (default 36 = 10° steps)

    Returns
    -------
    Best rotated atom list (fewest clashes, then minimum worst-case distance sum).
    """
    axis = donor_abs_pos - metal_pos
    norm = np.linalg.norm(axis)
    if norm < 1e-6:
        return new_atoms
    axis = axis / norm

    def rotate_about_axis(atoms, theta):
        """Rotate atoms around the metal→donor axis by theta radians."""
        K = np.array([[0,       -axis[2],  axis[1]],
                      [axis[2],  0,        -axis[0]],
                      [-axis[1], axis[0],   0      ]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        rotated = []
        for sym, pos in atoms:
            # rotate around metal position
            p = pos - metal_pos
            rotated.append((sym, R @ p + metal_pos))
        return rotated

    best_atoms = new_atoms
    best_score = _clash_score(new_atoms, existing_atoms)

    for step in range(1, n_steps):
        theta = 2 * np.pi * step / n_steps
        rotated = rotate_about_axis(new_atoms, theta)
        score = _clash_score(rotated, existing_atoms)
        if score < best_score:
            best_score = score
            best_atoms = rotated
        if best_score == 0:
            break

    return best_atoms


def _clash_score(new_atoms: List[Tuple[str, np.ndarray]],
                 existing_atoms: List[Tuple[str, np.ndarray]]) -> float:
    """
    Score = sum of (min_dist - actual_dist) for all clashing pairs.
    0 means no clashes.
    """
    score = 0.0
    for sn, pn in new_atoms:
        for se, pe in existing_atoms:
            d = np.linalg.norm(pn - pe)
            if _is_clash(sn, se, d):
                threshold = _MIN_NONBONDED.get(_pair_key(sn, se), 2.0)
                score += (threshold - d)
    return score
