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
    _RDKIT = True
except ImportError:
    _RDKIT = False

# ── Fallback: hard-coded geometries for common ligands ───────────────────────
# Format: list of (symbol, [dx, dy, dz]) relative to donor atom at origin
# +x points TOWARD the metal; ligand bulk faces -x

_FALLBACK_GEOMS = {
    # H2O: O at origin, two H atoms at ~104.5° H-O-H, O-H = 0.96 Å
    "H2O": ("O", [
        ("O", [0., 0., 0.]),
        ("H", [-0.757,  0.586, 0.]),
        ("H", [-0.757, -0.586, 0.]),
    ]),
    "aqua": ("O", [
        ("O", [0., 0., 0.]),
        ("H", [-0.757,  0.586, 0.]),
        ("H", [-0.757, -0.586, 0.]),
    ]),
    # NH3: N at origin, three H atoms in trigonal pyramid
    "NH3": ("N", [
        ("N", [0., 0., 0.]),
        ("H", [-0.561, -0.561,  0.561]),
        ("H", [-0.561,  0.561, -0.561]),
        ("H", [-0.940,  0.000,  0.000]),
    ]),
    "ammine": ("N", [
        ("N", [0., 0., 0.]),
        ("H", [-0.561, -0.561,  0.561]),
        ("H", [-0.561,  0.561, -0.561]),
        ("H", [-0.940,  0.000,  0.000]),
    ]),
    # PH3
    "PH3": ("P", [
        ("P", [0., 0., 0.]),
        ("H", [-0.820, -0.820,  0.820]),
        ("H", [-0.820,  0.820, -0.820]),
        ("H", [-1.200,  0.000,  0.000]),
    ]),
    # CO: C donor, O at -1.13 Å
    "CO": ("C", [
        ("C", [0., 0., 0.]),
        ("O", [-1.128, 0., 0.]),
    ]),
    # CN: C donor (or N) — just two atoms
    "CN": ("C", [
        ("C", [0., 0., 0.]),
        ("N", [-1.16, 0., 0.]),
    ]),
    # NO: N donor
    "NO": ("N", [
        ("N", [0., 0., 0.]),
        ("O", [-1.15, 0., 0.]),
    ]),
    # Halides — single atom donors, no H
    "Cl":  ("Cl", [("Cl", [0., 0., 0.])]),
    "Br":  ("Br", [("Br", [0., 0., 0.])]),
    "I":   ("I",  [("I",  [0., 0., 0.])]),
    "F":   ("F",  [("F",  [0., 0., 0.])]),
    # OH: O donor, one H at ~109° O-H = 0.96 Å
    "OH": ("O", [
        ("O", [0., 0., 0.]),
        ("H", [-0.819, -0.574, 0.]),
    ]),
    # Hydride
    "H":       ("H", [("H", [0., 0., 0.])]),
    "hydride": ("H", [("H", [0., 0., 0.])]),
    # SCN: S donor
    "SCN": ("S", [
        ("S", [0., 0., 0.]),
        ("C", [-1.62, 0., 0.]),
        ("N", [-2.78, 0., 0.]),
    ]),
    "NCS": ("N", [
        ("N", [0., 0., 0.]),
        ("C", [-1.16, 0., 0.]),
        ("S", [-2.78, 0., 0.]),
    ]),
    # O2- (oxide) — single atom
    "O":  ("O", [("O", [0., 0., 0.])]),
    # MeCN: N donor, linear chain away from metal
    "MeCN": ("N", [
        ("N", [0., 0., 0.]),
        ("C", [-1.16, 0., 0.]),
        ("C", [-2.49, 0., 0.]),
        ("H", [-2.84,  1.027, 0.]),
        ("H", [-2.84, -0.514,  0.890]),
        ("H", [-2.84, -0.514, -0.890]),
    ]),
    "acetonitrile": ("N", [
        ("N", [0., 0., 0.]),
        ("C", [-1.16, 0., 0.]),
        ("C", [-2.49, 0., 0.]),
        ("H", [-2.84,  1.027, 0.]),
        ("H", [-2.84, -0.514,  0.890]),
        ("H", [-2.84, -0.514, -0.890]),
    ]),
    # N3 (azide): N donor, linear
    "N3": ("N", [
        ("N", [0., 0., 0.]),
        ("N", [-1.16, 0., 0.]),
        ("N", [-2.32, 0., 0.]),
    ]),
    # NO2 (nitro): N donor
    "NO2": ("N", [
        ("N", [0., 0., 0.]),
        ("O", [-0.65,  1.09, 0.]),
        ("O", [-0.65, -1.09, 0.]),
    ]),
    # ONO (nitrito-O): O donor
    "ONO": ("O", [
        ("O", [0., 0., 0.]),
        ("N", [-1.22, 0., 0.]),
        ("O", [-1.87, -1.09, 0.]),
    ]),
    # Me: C donor (methyl)
    "Me": ("C", [
        ("C", [0., 0., 0.]),
        ("H", [-0.631,  0.631,  0.631]),
        ("H", [-0.631, -0.631, -0.631]),
        ("H", [-0.631,  0.631, -0.631]),
    ]),
    # Ph: C donor (phenyl) — simplified
    "Ph": ("C", [
        ("C", [0., 0., 0.]),
        ("C", [-1.40, 0., 0.]),
        ("C", [-2.10,  1.21, 0.]),
        ("C", [-2.10, -1.21, 0.]),
        ("C", [-3.50,  1.21, 0.]),
        ("C", [-3.50, -1.21, 0.]),
        ("C", [-4.20,  0., 0.]),
        ("H", [-1.57,  2.16, 0.]),
        ("H", [-1.57, -2.16, 0.]),
        ("H", [-4.05,  2.16, 0.]),
        ("H", [-4.05, -2.16, 0.]),
        ("H", [-5.29,  0., 0.]),
    ]),
    # Bridging (same as non-bridging for placement purposes)
    "mu-Cl": ("Cl", [("Cl", [0., 0., 0.])]),
    "mu-OH": ("O", [
        ("O", [0., 0., 0.]),
        ("H", [-0.819, -0.574, 0.]),
    ]),
    "mu-O":  ("O", [("O", [0., 0., 0.])]),
    "mu-CO": ("C", [
        ("C", [0., 0., 0.]),
        ("O", [-1.128, 0., 0.]),
    ]),
    "mu-H":  ("H", [("H", [0., 0., 0.])]),
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
