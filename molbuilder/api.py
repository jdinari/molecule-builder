"""
api.py
======
Public API for molbuilder.

  from molbuilder.api import build, build_all_isomers, dimer, trimer, poscar, xyz, info
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict, Union
import numpy as np

from molbuilder.core.molecule import Molecule, Atom
from molbuilder.core.geometry import (
    get_geometry_vectors, infer_geometry, resolve_geometry, list_geometries,
)
from molbuilder.core.bond_lengths import get_bond_length
from molbuilder.core.isomers import enumerate_isomers
from molbuilder.ligands.library import get_ligand, list_ligands
from molbuilder.ligands.ligand_geometry import get_ligand_atoms, get_ligand_atoms_multidentate, _rodrigues_rotation
from molbuilder.output.poscar_writer import poscar_to_string
from molbuilder.output.xyz_writer import xyz_to_string


# ──────────────────────────────────────────────────────────────────────────────
# Atomic masses (for spin-state estimation)
# ──────────────────────────────────────────────────────────────────────────────
_ELECTRON_CONFIG: Dict[str, Dict[int, int]] = {
    # d-electron counts {metal: {ox_state: d_electrons}}
    "Sc": {3: 0}, "Ti": {4: 0, 3: 1, 2: 2},
    "V":  {5: 0, 4: 1, 3: 2, 2: 3},
    "Cr": {6: 0, 3: 3, 2: 4, 0: 6},
    "Mn": {7: 0, 4: 3, 3: 4, 2: 5},
    "Fe": {3: 5, 2: 6, 0: 8},
    "Co": {3: 6, 2: 7},
    "Ni": {4: 6, 3: 7, 2: 8},
    "Cu": {3: 8, 2: 9, 1: 10},
    "Zn": {2: 10},
    "Mo": {6: 0, 4: 2, 2: 4, 0: 6},
    "Ru": {8: 0, 4: 4, 3: 5, 2: 6, 0: 8},
    "Rh": {3: 6, 1: 8},
    "Pd": {4: 6, 2: 8, 0: 10},
    "Ag": {1: 10},
    "W":  {6: 0, 4: 2, 0: 6},
    "Re": {7: 0, 3: 4, 1: 6},
    "Os": {8: 0, 4: 4, 2: 6, 0: 8},
    "Ir": {3: 6, 1: 8},
    "Pt": {4: 6, 2: 8, 0: 10},
    "Au": {3: 8, 1: 10},
}


def _d_electrons(metal: str, ox: int) -> int:
    return _ELECTRON_CONFIG.get(metal, {}).get(ox, 0)


def _spin_multiplicity(metal: str, ox: int, geometry: str) -> int:
    """Rough spin-state guess: high-spin for 4/5-coord+, low-spin for strong-field."""
    d = _d_electrons(metal, ox)
    if d == 0 or d == 10:
        return 1
    # Very rough: octahedral 3d metals often high-spin unless CO/CN ligands
    # Default: high-spin
    unpaired = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0}
    return unpaired.get(d, 1) + 1


# ──────────────────────────────────────────────────────────────────────────────
# Helpers: SMILES ligand  → donor atom element
# ──────────────────────────────────────────────────────────────────────────────

def _donor_from_smiles(smiles: str) -> str:
    """Guess donor atom from SMILES (very rough heuristic)."""
    if not smiles:
        return "N"
    # look for explicit heteroatoms in order of priority
    for sym in ["P", "S", "N", "O", "C"]:
        if sym in smiles:
            return sym
    return "C"


# ──────────────────────────────────────────────────────────────────────────────
# Ligand expansion: turn a name/SMILES into atoms + bond length
# ──────────────────────────────────────────────────────────────────────────────

def _expand_ligand(lig_name: str, metal: str, ox: int, geometry: str):
    """
    Resolve a ligand name to a dict carrying full 3D geometry (including H).
    
    Returns dict with keys:
      atoms         – list of (symbol, rel_pos) relative to donor at origin,
                      ligand bulk pointing along -x (metal will be at +x)
      donor_atom    – element symbol of primary donor
      donor_atoms   – list of all donor element symbols
      smiles        – SMILES string
      charge        – formal charge
      denticity     – number of donor atoms
      bite_angle    – chelate bite angle (deg), or None
      vectors_count – number of geometry vectors consumed
      name          – ligand name string
    """
    # Detect raw SMILES
    is_smiles = ("=" in lig_name or "#" in lig_name or
                 "(" in lig_name or "[" in lig_name or
                 "." in lig_name)
    if not is_smiles:
        try:
            lig = get_ligand(lig_name)
        except KeyError:
            is_smiles = True

    if is_smiles:
        donor = _donor_from_smiles(lig_name)
        full_atoms = get_ligand_atoms(lig_name, lig_name, donor, 0)
        return {
            "atoms": full_atoms,
            "donor_atom": donor,
            "donor_atoms": [donor],
            "smiles": lig_name,
            "charge": 0,
            "denticity": 1,
            "bite_angle": None,
            "vectors_count": 1,
            "name": lig_name,
        }

    lig = get_ligand(lig_name)
    donor_atoms = lig.get("donor_atoms", ["N"])
    smiles = lig.get("smiles", "")
    donor_indices = lig.get("donors", [0])

    if lig["denticity"] == 1:
        # Full geometry: donor at origin, bulk along -x
        full_atoms = get_ligand_atoms(lig_name, smiles, donor_atoms[0],
                                      donor_indices[0] if donor_indices else 0)
        return {
            "atoms": full_atoms,
            "donor_atom": donor_atoms[0],
            "donor_atoms": donor_atoms,
            "donors": donor_indices,
            "smiles": smiles,
            "charge": lig["charge"],
            "denticity": 1,
            "bite_angle": None,
            "vectors_count": 1,
            "name": lig_name,
        }

    # Multidentate
    bite = lig.get("bite_angle", 90.0)
    return {
        "atoms": [],          # filled in during placement using get_ligand_atoms_multidentate
        "donor_atom": donor_atoms[0],
        "donor_atoms": donor_atoms,
        "donors": donor_indices,
        "smiles": smiles,
        "charge": lig["charge"],
        "denticity": lig["denticity"],
        "bite_angle": bite,
        "vectors_count": lig["denticity"],
        "name": lig_name,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Geometry vector assignment for multidentate ligands
# ──────────────────────────────────────────────────────────────────────────────

def _place_bidentate(vec1: np.ndarray, vec2: np.ndarray,
                     d1: str, d2: str,
                     metal: str, ox: int, geometry: str) -> List[tuple]:
    """Return [(symbol, position), (symbol, position)] for a bidentate ligand."""
    bl1 = get_bond_length(metal, ox, d1, geometry)
    bl2 = get_bond_length(metal, ox, d2, geometry)
    return [(d1, vec1 * bl1), (d2, vec2 * bl2)]


# ──────────────────────────────────────────────────────────────────────────────
# Formula builder
# ──────────────────────────────────────────────────────────────────────────────

def _make_formula(symbols: List[str]) -> str:
    cnt = Counter(symbols)
    metals = []
    rest = {}
    for el, n in cnt.items():
        rest[el] = n
    parts = []
    for el in sorted(rest, key=lambda s: s):
        n = rest[el]
        parts.append(el if n == 1 else f"{el}{n}")
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# build()
# ──────────────────────────────────────────────────────────────────────────────

def build(metal: str,
          ox: int,
          ligands: Optional[List] = None,
          geometry: Optional[str] = None,
          spin: Optional[int] = None) -> Molecule:
    """
    Build a mononuclear transition-metal complex.

    Parameters
    ----------
    metal : str
        Element symbol, e.g. "Fe"
    ox : int
        Oxidation state, e.g. 3
    ligands : list of str
        Ligand names or SMILES strings, e.g. ["Cl","Cl","Cl","H2O","H2O","H2O"]
    geometry : str, optional
        Coordination geometry key. Auto-inferred from CN if omitted.
    spin : int, optional
        Spin multiplicity. Auto-estimated if omitted.

    Returns
    -------
    Molecule
    """
    if ligands is None:
        ligands = []

    # Expand multidentate ligands to count how many coordination sites we need
    expanded = []      # list of _expand_ligand dicts
    for lname in ligands:
        e = _expand_ligand(str(lname), metal, ox, geometry)
        expanded.append(e)

    # Total coordination number = sum of denticity
    cn = sum(e["vectors_count"] for e in expanded)
    if cn == 0:
        cn = 6  # fallback

    # Resolve geometry
    if geometry:
        geom_canon = resolve_geometry(geometry)
    else:
        geom_canon = infer_geometry(cn)

    vecs = get_geometry_vectors(geom_canon)
    if cn > len(vecs):
        # Pad with extra vectors if needed (unusual case)
        extra = cn - len(vecs)
        angle_step = np.pi / (extra + 1)
        for i in range(extra):
            a = angle_step * (i + 1)
            vecs.append(np.array([np.sin(a), np.cos(a), 0.3]))

    # ── assign vectors to ligands ─────────────────────────────────────────
    mol_atoms: List[Atom] = []
    vec_idx = 0

    # Metal at origin
    mol_atoms.append(Atom(symbol=metal, position=np.zeros(3), label=f"{metal}1"))

    total_charge = ox  # metal formal charge
    lig_names_out = []

    for e in expanded:
        d = e["denticity"]
        charge = e["charge"]
        total_charge += charge
        lig_names_out.append(e["name"])

        if d == 1:
            # Direction vector from geometry
            v = vecs[vec_idx % len(vecs)]
            vec_idx += 1
            bl = get_bond_length(metal, ox, e["donor_atom"], geom_canon)

            # Get full ligand atom list (donor at origin, bulk along -x)
            full_atoms = e["atoms"]   # list of (symbol, rel_pos)

            # Build rotation: rotate ligand so bulk (-x) points AWAY from metal
            # i.e. donor (+x direction from metal) aligns with v
            # The donor is at origin in rel coords; metal is at +x*bl
            # We need to rotate so that +x maps to v
            R = _rodrigues_rotation(np.array([1., 0., 0.]), v)

            for sym, rel_pos in full_atoms:
                # donor is at origin → absolute position = metal_pos + R*(rel_pos) + v*bl
                # But rel_pos already has donor at 0, so donor maps to v*bl exactly
                abs_pos = R @ rel_pos + v * bl
                mol_atoms.append(Atom(symbol=sym, position=abs_pos,
                                      label=f"{sym}{len(mol_atoms)}"))

        elif d == 2:
            v1 = vecs[vec_idx % len(vecs)]
            v2 = vecs[(vec_idx + 1) % len(vecs)]
            vec_idx += 2
            donor_atoms_list = e["donor_atoms"]
            d1 = donor_atoms_list[0]
            d2 = donor_atoms_list[1] if len(donor_atoms_list) > 1 else donor_atoms_list[0]
            bl1 = get_bond_length(metal, ox, d1, geom_canon)
            bl2 = get_bond_length(metal, ox, d2, geom_canon)

            multi_atoms = get_ligand_atoms_multidentate(
                e["name"], e["smiles"],
                donor_atoms_list[:2], e.get("donors", [0, 1])[:2],
                bite_angle_deg=e["bite_angle"] or 90.0,
                bond_lengths=[bl1, bl2],
            )
            # multi_atoms are in a local frame with metal at origin, donors in xy-plane
            # Rotate this frame so the midpoint direction aligns with (v1+v2)/2
            mid = (v1 + v2)
            if np.linalg.norm(mid) > 1e-6:
                mid = mid / np.linalg.norm(mid)
                R = _rodrigues_rotation(np.array([1., 0., 0.]), mid)
                for sym, pos in multi_atoms:
                    mol_atoms.append(Atom(symbol=sym, position=R @ pos,
                                          label=f"{sym}{len(mol_atoms)}"))
            else:
                for sym, pos in multi_atoms:
                    mol_atoms.append(Atom(symbol=sym, position=pos,
                                          label=f"{sym}{len(mol_atoms)}"))

        else:
            # Tridentate or higher
            donor_atoms_list = e["donor_atoms"]
            bls = [get_bond_length(metal, ox, da, geom_canon) for da in donor_atoms_list]
            multi_atoms = get_ligand_atoms_multidentate(
                e["name"], e["smiles"],
                donor_atoms_list, e.get("donors", list(range(len(donor_atoms_list)))),
                bite_angle_deg=e["bite_angle"] or 90.0,
                bond_lengths=bls,
            )
            mid_vec = np.mean([vecs[(vec_idx + i) % len(vecs)] for i in range(d)], axis=0)
            if np.linalg.norm(mid_vec) > 1e-6:
                mid_vec /= np.linalg.norm(mid_vec)
                R = _rodrigues_rotation(np.array([1., 0., 0.]), mid_vec)
                for sym, pos in multi_atoms:
                    mol_atoms.append(Atom(symbol=sym, position=R @ pos,
                                          label=f"{sym}{len(mol_atoms)}"))
            else:
                for sym, pos in multi_atoms:
                    mol_atoms.append(Atom(symbol=sym, position=pos,
                                          label=f"{sym}{len(mol_atoms)}"))
            vec_idx += d

    # ── build formula ─────────────────────────────────────────────────────
    all_symbols = [a.symbol for a in mol_atoms]
    formula = _make_formula(all_symbols)

    spin_mult = spin if spin is not None else _spin_multiplicity(metal, ox, geom_canon)

    mol = Molecule(
        atoms=mol_atoms,
        metal_indices=[0],
        formula=formula,
        charge=total_charge,
        spin_multiplicity=spin_mult,
        geometry=geom_canon,
        metal_symbol=metal,
        metal_ox=ox,
        ligand_names=lig_names_out,
    )
    return mol


# ──────────────────────────────────────────────────────────────────────────────
# build_all_isomers()
# ──────────────────────────────────────────────────────────────────────────────

def build_all_isomers(metal: str,
                      ox: int,
                      ligands: Optional[List] = None,
                      geometry: Optional[str] = None,
                      spin: Optional[int] = None) -> List[dict]:
    """
    Build ALL symmetry-distinct isomers for a given metal + ligand set.

    Returns a list of dicts, one per isomer:
        {
            "molecule"  : Molecule,
            "label"     : str,     e.g. "fac", "mer", "cis", "trans", "only"
            "index"     : int,     0-based
            "n_isomers" : int,
        }

    Example
    -------
    >>> isomers = build_all_isomers("Fe", ox=3,
    ...     ligands=["Cl","Cl","Cl","H2O","H2O","H2O"])
    >>> for iso in isomers:
    ...     print(iso["label"], iso["molecule"].formula)
    fac Cl3FeH6O3
    mer Cl3FeH6O3
    """
    if ligands is None:
        ligands = []

    lig_strs = [str(l) for l in ligands]

    # Determine geometry from CN if not given
    if geometry:
        geom_canon = resolve_geometry(geometry)
    else:
        # Need CN: expand to count denticity
        cn = 0
        for lname in lig_strs:
            is_smiles = ("=" in lname or "#" in lname or
                         "(" in lname or "[" in lname)
            if not is_smiles:
                try:
                    lig = get_ligand(lname)
                    cn += lig["denticity"]
                    continue
                except KeyError:
                    pass
            cn += 1
        geom_canon = infer_geometry(cn) if cn > 0 else "oct"

    isomer_list = enumerate_isomers(lig_strs, geom_canon)
    n = len(isomer_list)

    results = []
    for iso in isomer_list:
        mol = build(metal, ox=ox, ligands=iso["site_assignment"],
                    geometry=geom_canon, spin=spin)
        results.append({
            "molecule":  mol,
            "label":     iso["label"],
            "index":     iso["index"],
            "n_isomers": n,
        })
    return results


# ──────────────────────────────────────────────────────────────────────────────
# dimer()
# ──────────────────────────────────────────────────────────────────────────────

def dimer(metal: str,
          ox: int,
          terminal: Optional[List[str]] = None,
          bridge: Optional[str] = None,
          n: int = 2,
          geometry: Optional[str] = None,
          mm_bond: bool = False,
          mm_distance: Optional[float] = None) -> Molecule:
    """
    Build a dinuclear complex with bridging ligand(s).

    Parameters
    ----------
    metal : str
    ox : int
    terminal : list of str, optional
        Terminal ligand names per metal center
    bridge : str, optional
        Bridging ligand name (e.g. "mu-Cl")
    n : int
        Number of bridging ligand units (default 2)
    geometry : str, optional
    mm_bond : bool
        Whether there is a metal–metal bond
    mm_distance : float, optional
        M–M distance override (Å)
    """
    if terminal is None:
        terminal = []

    # Determine M–M distance
    if mm_distance is not None:
        d_mm = mm_distance
    elif mm_bond:
        from molbuilder.core.bond_lengths import COVALENT_RADII
        r = COVALENT_RADII.get(metal, 1.5)
        d_mm = 2 * r + 0.1
    else:
        # non-bonded dimer: estimate from bridge length
        if bridge:
            try:
                bl = get_ligand(bridge.replace("mu-", ""))
                donor = bl.get("donor_atoms", ["Cl"])[0]
            except KeyError:
                donor = "Cl"
            bridge_bl = get_bond_length(metal, ox, donor, geometry or "oct")
        else:
            bridge_bl = 2.5
        d_mm = 2 * bridge_bl + 0.5

    # Build individual monomers and place them
    terminal_plus_half_bridge = list(terminal)
    if bridge:
        for _ in range(n):
            terminal_plus_half_bridge.append(bridge)

    mol1 = build(metal, ox, terminal_plus_half_bridge, geometry)
    mol2 = build(metal, ox, terminal_plus_half_bridge, geometry)

    # Translate mol2 along x by d_mm
    offset = np.array([d_mm, 0., 0.])
    for a in mol2.atoms:
        a.position = a.position + offset

    # Merge
    n_atoms_m1 = len(mol1.atoms)
    combined_atoms = mol1.atoms + mol2.atoms
    total_charge = mol1.charge + mol2.charge
    formula = _make_formula([a.symbol for a in combined_atoms])
    spin = mol1.spin_multiplicity  # simple approximation

    mol = Molecule(
        atoms=combined_atoms,
        metal_indices=[0, n_atoms_m1],
        formula=formula,
        charge=total_charge,
        spin_multiplicity=spin,
        geometry=mol1.geometry,
        metal_symbol=metal,
        metal_ox=ox,
        ligand_names=mol1.ligand_names,
    )
    return mol


# ──────────────────────────────────────────────────────────────────────────────
# trimer()
# ──────────────────────────────────────────────────────────────────────────────

def trimer(metal: str,
           ox: int,
           terminal: Optional[List[str]] = None,
           bridge: Optional[str] = None,
           arrangement: str = "triangular",
           geometry: Optional[str] = None) -> Molecule:
    """
    Build a trinuclear complex.

    arrangement: 'triangular' or 'linear'
    """
    if terminal is None:
        terminal = []

    terminal_plus_bridge = list(terminal)
    if bridge:
        terminal_plus_bridge.append(bridge)
        terminal_plus_bridge.append(bridge)

    m1 = build(metal, ox, terminal_plus_bridge, geometry)

    # Estimate M–M distance
    bl_ref = 2.5
    if bridge:
        try:
            bl_lig = get_ligand(bridge.replace("mu-", ""))
            donor = bl_lig.get("donor_atoms", ["C"])[0]
            bl_ref = get_bond_length(metal, ox, donor, geometry or "oct")
        except KeyError:
            pass
    d_mm = 2 * bl_ref + 0.4

    if arrangement == "triangular":
        offsets = [
            np.array([0., 0., 0.]),
            np.array([d_mm, 0., 0.]),
            np.array([d_mm / 2, d_mm * np.sqrt(3) / 2, 0.]),
        ]
    else:  # linear
        offsets = [
            np.array([0., 0., 0.]),
            np.array([d_mm, 0., 0.]),
            np.array([2 * d_mm, 0., 0.]),
        ]

    all_atoms = []
    metal_indices = []
    n = 0
    for k, off in enumerate(offsets):
        mk = build(metal, ox, terminal_plus_bridge, geometry)
        for a in mk.atoms:
            a.position = a.position + off
        metal_indices.append(n)
        all_atoms.extend(mk.atoms)
        n += len(mk.atoms)

    total_charge = m1.charge * 3
    formula = _make_formula([a.symbol for a in all_atoms])

    mol = Molecule(
        atoms=all_atoms,
        metal_indices=metal_indices,
        formula=formula,
        charge=total_charge,
        spin_multiplicity=m1.spin_multiplicity,
        geometry=m1.geometry,
        metal_symbol=metal,
        metal_ox=ox,
        ligand_names=m1.ligand_names,
    )
    return mol


# ──────────────────────────────────────────────────────────────────────────────
# poscar(), xyz(), info()
# ──────────────────────────────────────────────────────────────────────────────

def poscar(mol: Molecule, filepath: str) -> None:
    """Write Molecule to a VASP POSCAR file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(poscar_to_string(mol))
    print(f"✓ POSCAR written to {p}")


def xyz(mol: Molecule, filepath: str) -> None:
    """Write Molecule to an XYZ file."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(xyz_to_string(mol))
    print(f"✓ XYZ written to {p}")


def info(mol: Molecule) -> None:
    """Print a summary of the molecule."""
    print(f"Formula          : {mol.formula}")
    print(f"Total charge     : {mol.charge:+d}")
    print(f"Spin multiplicity: {mol.spin_multiplicity}")
    print(f"Metal            : {mol.metal_symbol}({mol.metal_ox:+d})")
    print(f"Geometry         : {mol.geometry}")
    print(f"Num atoms        : {mol.num_atoms()}")
    print(f"Ligands          : {', '.join(mol.ligand_names) if mol.ligand_names else '–'}")
    print("\nAtom list:")
    for i, a in enumerate(mol.atoms):
        x, y, z = a.position
        print(f"  {i+1:3d}  {a.symbol:3s}  {x:10.4f}  {y:10.4f}  {z:10.4f}")
