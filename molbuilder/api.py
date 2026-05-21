"""
api.py
======
Public API for molbuilder.

    from molbuilder.api import build, dimer, trimer, poscar, xyz, info

build() always returns all symmetry-distinct isomers automatically:
  - One isomer  → returns a single Molecule
  - Two or more → returns a list of Molecule objects, one per isomer

Custom POSCAR ligands are supported directly via load_ligand_from_poscar().
Denticity modes use colon notation: "HCOO:bi", "bpy:mono", etc.
"""

from __future__ import annotations

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
from molbuilder.ligands.ligand_geometry import (
    get_ligand_atoms, get_ligand_atoms_multidentate, _rodrigues_rotation,
    check_clashes, resolve_clash_by_rotation, _clash_score,
)
from molbuilder.output.poscar_writer import poscar_to_string
from molbuilder.output.xyz_writer import xyz_to_string


# ── spin-state estimation ────────────────────────────────────────────────────

_D_ELECTRONS: Dict[str, Dict[int, int]] = {
    "Sc": {3: 0},
    "Ti": {4: 0, 3: 1, 2: 2},
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

_UNPAIRED = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 4, 7: 3, 8: 2, 9: 1, 10: 0}

def _spin_multiplicity(metal: str, ox: int) -> int:
    d = _D_ELECTRONS.get(metal, {}).get(ox, 0)
    return _UNPAIRED.get(d, 1) + 1


# ── SMILES donor-atom heuristic ──────────────────────────────────────────────

def _donor_from_smiles(smiles: str) -> str:
    for sym in ["P", "S", "N", "O", "C"]:
        if sym in smiles:
            return sym
    return "C"


# ── custom POSCAR ligand support ─────────────────────────────────────────────

class CustomLigand:
    """
    A ligand loaded from a POSCAR file.

    Parameters
    ----------
    poscar_path : str or Path
    donor_atom_indices : list of int
        0-based atom indices in the POSCAR that coordinate to the metal.
    charge : int
        Formal charge of the free ligand.
    name : str, optional
        Human-readable name (defaults to filename stem).

    Example
    -------
    >>> lig = CustomLigand("my_ligand.POSCAR", donor_atom_indices=[0, 2], charge=0)
    >>> mol = build("Ni", ox=2, ligands=["H2O", "H2O", "H2O", "H2O", lig])
    """

    def __init__(self, poscar_path, donor_atom_indices: List[int],
                 charge: int = 0, name: Optional[str] = None):
        self.poscar_path = Path(poscar_path)
        self.donor_atom_indices = donor_atom_indices
        self.charge = charge
        self.name = name or self.poscar_path.stem
        self.symbols, self.positions = self._parse()

    def _parse(self):
        lines = self.poscar_path.read_text().strip().splitlines()
        scale = float(lines[1])
        cell = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale
        syms_line = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        symbols = [s for s, n in zip(syms_line, counts) for _ in range(n)]
        is_cart = lines[7].strip()[0].upper() == "C"
        positions = []
        for i in range(8, 8 + len(symbols)):
            coords = np.array([float(x) for x in lines[i].split()[:3]])
            positions.append(coords if is_cart else coords @ cell)
        return symbols, np.array(positions) * (scale if is_cart else 1.0)

    @property
    def donor_atoms(self):
        return [self.symbols[i] for i in self.donor_atom_indices]

    @property
    def denticity(self):
        return len(self.donor_atom_indices)

    def __repr__(self):
        return f"CustomLigand({self.name!r}, denticity={self.denticity}, charge={self.charge})"


def load_ligand_from_poscar(poscar_path, donor_atom_indices: List[int],
                             charge: int = 0, name: Optional[str] = None) -> CustomLigand:
    """
    Load a custom ligand from a POSCAR file.

    Parameters
    ----------
    poscar_path : str or Path
    donor_atom_indices : list of int
        0-based atom indices that coordinate to the metal.
    charge : int
        Formal charge of the free ligand (default 0).
    name : str, optional

    Returns
    -------
    CustomLigand

    Example
    -------
    >>> lig = load_ligand_from_poscar("norbornane.POSCAR", [0, 2], charge=0)
    >>> mol = build("Pd", ox=2, ligands=["Cl", "Cl", lig])
    """
    return CustomLigand(poscar_path, donor_atom_indices, charge, name)


# ── ligand expansion ─────────────────────────────────────────────────────────

def _expand_ligand(lig, metal: str, ox: int, geometry: str) -> dict:
    """
    Turn a ligand name, SMILES string, or CustomLigand into a placement dict.

    Returns
    -------
    dict with keys: atoms, donor_atom, donor_atoms, donors, smiles,
                    charge, denticity, bite_angle, vectors_count, name
    """
    # ── CustomLigand ──────────────────────────────────────────────────────────
    if isinstance(lig, CustomLigand):
        d_syms = lig.donor_atoms
        d_idx  = lig.donor_atom_indices
        # Build atom list relative to first donor at origin
        d0_pos = lig.positions[d_idx[0]]
        rel_atoms = [(s, p - d0_pos) for s, p in zip(lig.symbols, lig.positions)]
        # Rotate bulk away from metal (+x)
        non_donor_pos = [p for i, (s, p) in enumerate(rel_atoms)
                         if i not in d_idx and s != "H"]
        if non_donor_pos:
            bulk = np.mean(non_donor_pos, axis=0)
            if np.linalg.norm(bulk) > 1e-3:
                from molbuilder.ligands.ligand_geometry import _rodrigues_rotation
                R = _rodrigues_rotation(bulk / np.linalg.norm(bulk),
                                        np.array([-1., 0., 0.]))
                rel_atoms = [(s, R @ p) for s, p in rel_atoms]
        return {
            "atoms": rel_atoms,
            "donor_atom": d_syms[0],
            "donor_atoms": d_syms,
            "donors": d_idx,
            "smiles": "",
            "charge": lig.charge,
            "denticity": lig.denticity,
            "bite_angle": None,
            "vectors_count": lig.denticity,
            "name": lig.name,
        }

    lig_name = str(lig)

    # ── raw SMILES ────────────────────────────────────────────────────────────
    is_smiles = ("=" in lig_name or "#" in lig_name or
                 "(" in lig_name or "[" in lig_name)
    if not is_smiles:
        try:
            lig_data = get_ligand(lig_name)
        except KeyError:
            is_smiles = True

    if is_smiles:
        donor = _donor_from_smiles(lig_name)
        full_atoms = get_ligand_atoms(lig_name, lig_name, donor, 0)
        return {
            "atoms": full_atoms,
            "donor_atom": donor,
            "donor_atoms": [donor],
            "donors": [0],
            "smiles": lig_name,
            "charge": 0,
            "denticity": 1,
            "bite_angle": None,
            "vectors_count": 1,
            "name": lig_name,
        }

    # ── named ligand ──────────────────────────────────────────────────────────
    lig_data = get_ligand(lig_name)
    donor_atoms  = lig_data.get("donor_atoms", ["N"])
    smiles       = lig_data.get("smiles", "")
    donor_indices = lig_data.get("donors", [0])

    if lig_data["denticity"] == 1:
        full_atoms = get_ligand_atoms(lig_name, smiles, donor_atoms[0],
                                      donor_indices[0] if donor_indices else 0)
        return {
            "atoms": full_atoms,
            "donor_atom": donor_atoms[0],
            "donor_atoms": donor_atoms,
            "donors": donor_indices,
            "smiles": smiles,
            "charge": lig_data["charge"],
            "denticity": 1,
            "bite_angle": None,
            "vectors_count": 1,
            "name": lig_name,
        }

    # multidentate
    return {
        "atoms": [],
        "donor_atom": donor_atoms[0],
        "donor_atoms": donor_atoms,
        "donors": donor_indices,
        "smiles": smiles,
        "charge": lig_data["charge"],
        "denticity": lig_data["denticity"],
        "bite_angle": lig_data.get("bite_angle", 90.0),
        "vectors_count": lig_data["denticity"],
        "name": lig_name,
    }


# ── formula ───────────────────────────────────────────────────────────────────

def _make_formula(symbols: List[str]) -> str:
    cnt = Counter(symbols)
    return "".join(el if n == 1 else f"{el}{n}"
                   for el, n in sorted(cnt.items()))


# ── core builder ─────────────────────────────────────────────────────────────

def _build_single(metal: str, ox: int, ligands: List, geometry: str,
                  spin: Optional[int]) -> Molecule:
    """Build one specific arrangement of ligands (no isomer enumeration)."""
    expanded = [_expand_ligand(l, metal, ox, geometry) for l in ligands]

    cn = sum(e["vectors_count"] for e in expanded) or 6
    geom_canon = resolve_geometry(geometry) if geometry else infer_geometry(cn)

    vecs = get_geometry_vectors(geom_canon)
    # Pad if needed (unusual high-denticity cases)
    while cn > len(vecs):
        a = np.pi * len(vecs) / (cn + 1)
        vecs.append(np.array([np.sin(a), np.cos(a), 0.3]) /
                    np.linalg.norm([np.sin(a), np.cos(a), 0.3]))

    mol_atoms: List[Atom] = [Atom(symbol=metal, position=np.zeros(3), label=f"{metal}1")]
    total_charge = ox
    lig_names_out = []
    vec_idx = 0

    for e in expanded:
        d = e["denticity"]
        total_charge += e["charge"]
        lig_names_out.append(e["name"])

        if d == 1:
            v  = vecs[vec_idx % len(vecs)]
            bl = get_bond_length(metal, ox, e["donor_atom"], geom_canon)
            R  = _rodrigues_rotation(np.array([1., 0., 0.]), v)

            # Place ligand atoms in absolute frame
            placed = [(sym, R @ rel + v * bl) for sym, rel in e["atoms"]]

            # Resolve clashes against all already-placed atoms (including metal)
            existing = [(a.symbol, a.position) for a in mol_atoms]
            if check_clashes(placed, existing):
                donor_abs = v * bl
                metal_pos = mol_atoms[0].position  # metal is always atom 0
                placed = resolve_clash_by_rotation(placed, donor_abs, metal_pos, existing)

            for sym, abs_pos in placed:
                mol_atoms.append(Atom(symbol=sym, position=abs_pos,
                                      label=f"{sym}{len(mol_atoms)}"))
            vec_idx += 1

        elif d == 2:
            v1 = vecs[vec_idx % len(vecs)]
            v2 = vecs[(vec_idx + 1) % len(vecs)]
            vec_idx += 2
            d1, d2 = e["donor_atoms"][0], e["donor_atoms"][1] if len(e["donor_atoms"]) > 1 else e["donor_atoms"][0]
            bl1 = get_bond_length(metal, ox, d1, geom_canon)
            bl2 = get_bond_length(metal, ox, d2, geom_canon)
            multi = get_ligand_atoms_multidentate(
                e["name"], e["smiles"], e["donor_atoms"][:2],
                e.get("donors", [0, 1])[:2],
                bite_angle_deg=e["bite_angle"] or 90.0,
                bond_lengths=[bl1, bl2],
            )
            mid = v1 + v2
            R = _rodrigues_rotation(np.array([1., 0., 0.]),
                                    mid / np.linalg.norm(mid)) if np.linalg.norm(mid) > 1e-6 else np.eye(3)
            for sym, pos in multi:
                mol_atoms.append(Atom(symbol=sym, position=R @ pos,
                                      label=f"{sym}{len(mol_atoms)}"))

        else:
            d_list = e["donor_atoms"]
            bls = [get_bond_length(metal, ox, da, geom_canon) for da in d_list]
            multi = get_ligand_atoms_multidentate(
                e["name"], e["smiles"], d_list,
                e.get("donors", list(range(len(d_list)))),
                bite_angle_deg=e["bite_angle"] or 90.0,
                bond_lengths=bls,
            )
            mid_vecs = [vecs[(vec_idx + i) % len(vecs)] for i in range(d)]
            mid = np.mean(mid_vecs, axis=0)
            R = _rodrigues_rotation(np.array([1., 0., 0.]),
                                    mid / np.linalg.norm(mid)) if np.linalg.norm(mid) > 1e-6 else np.eye(3)
            for sym, pos in multi:
                mol_atoms.append(Atom(symbol=sym, position=R @ pos,
                                      label=f"{sym}{len(mol_atoms)}"))
            vec_idx += d

    formula = _make_formula([a.symbol for a in mol_atoms])
    spin_mult = spin if spin is not None else _spin_multiplicity(metal, ox)

    # ── post-placement clash resolution ───────────────────────────────────────
    # Rotate each non-trivial ligand (those with >1 atom) around its M→donor
    # axis to minimize clashes with all other atoms. Run multiple passes until
    # stable (adjacent ligands can affect each other).
    metal_pos = mol_atoms[0].position

    # Build a map: which atoms belong to which ligand (by index ranges)
    # mol_atoms[0] = metal; then ligands in order
    ligand_slices = []   # list of (donor_abs_pos, [atom_indices])
    atom_idx = 1
    for e in expanded:
        n = len(e["atoms"]) if e["denticity"] == 1 else None
        # count atoms actually added for this ligand
        # We stored them consecutively; count from the placement loop
        # Simpler: rebuild the slice sizes from the placed atom counts
        ligand_slices.append(atom_idx)
        if e["denticity"] == 1:
            atom_idx += len(e["atoms"])
        elif e["denticity"] == 2:
            # multidentate: count atoms from get_ligand_atoms_multidentate
            # We don't know exactly without re-running; skip multidentate for now
            atom_idx += 2  # approximate: just donors
        else:
            atom_idx += e["denticity"]

    # Rebuild ligand_groups properly by scanning the atom list
    ligand_groups = []
    idx = 1
    for e in expanded:
        if e["denticity"] == 1:
            n_atoms = len(e["atoms"])
            ligand_groups.append({
                "indices": list(range(idx, idx + n_atoms)),
                "donor_idx": idx,          # first atom is the donor
                "has_tail": n_atoms > 1,
            })
            idx += n_atoms
        else:
            # multidentate: just skip for now
            n_atoms = e["denticity"]
            ligand_groups.append({
                "indices": list(range(idx, idx + n_atoms)),
                "donor_idx": idx,
                "has_tail": False,
            })
            idx += n_atoms
        if idx >= len(mol_atoms):
            break

    # Run up to 3 passes of rotation optimisation for ligands with tails
    for _pass in range(3):
        improved = False
        for g in ligand_groups:
            if not g["has_tail"]:
                continue
            indices   = g["indices"]
            donor_abs = mol_atoms[g["donor_idx"]].position

            # Collect current positions of this ligand
            lig_atoms = [(mol_atoms[i].symbol, mol_atoms[i].position) for i in indices]

            # All other atoms (not this ligand, not metal)
            other_atoms = [(mol_atoms[i].symbol, mol_atoms[i].position)
                           for i in range(len(mol_atoms)) if i not in indices]

            current_score = _clash_score(lig_atoms, other_atoms)
            if current_score == 0:
                continue

            # Find best rotation
            best_lig = resolve_clash_by_rotation(
                lig_atoms, donor_abs, metal_pos, other_atoms, n_steps=72
            )
            best_score = _clash_score(best_lig, other_atoms)

            if best_score < current_score:
                for k, i in enumerate(indices):
                    mol_atoms[i].position = best_lig[k][1]
                improved = True

        if not improved:
            break

    return Molecule(
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


# ── public API ────────────────────────────────────────────────────────────────

def build(metal: str,
          ox: int,
          ligands: Optional[List] = None,
          geometry: Optional[str] = None,
          spin: Optional[int] = None):
    """
    Build a mononuclear transition-metal complex.

    Automatically generates all symmetry-distinct isomers:
      - One isomer  → returns a single Molecule
      - Two or more → returns a list of Molecule objects

    Each Molecule in a multi-isomer result has a `.label` attribute
    (e.g. "fac", "mer", "cis", "trans", "isomer-1").

    Parameters
    ----------
    metal : str
        Element symbol, e.g. "Ni"
    ox : int
        Oxidation state, e.g. 2
    ligands : list
        Ligand names, SMILES strings, or CustomLigand objects.
        Denticity modes use colon notation: "HCOO:bi", "bpy:mono".
    geometry : str, optional
        Coordination geometry key (oct, sqp, tet, tbp, …).
        Auto-inferred from coordination number if omitted.
    spin : int, optional
        Spin multiplicity. Auto-estimated from d-electron count if omitted.

    Returns
    -------
    Molecule  or  list[Molecule]

    Examples
    --------
    # Single isomer → Molecule
    mol = build("Ni", ox=2, ligands=["H2O"]*6)

    # Two isomers → [Molecule, Molecule] with .label "fac" / "mer"
    mols = build("Ni", ox=2, ligands=["HCOO","HCOO","H2O","H2O","H2O","H2O"])

    # Bidentate formate (colon mode)
    mol = build("Ni", ox=2, ligands=["HCOO:bi","HCOO:bi","H2O","H2O"])

    # Custom POSCAR ligand
    lig = load_ligand_from_poscar("myligand.POSCAR", donor_atom_indices=[0])
    mol = build("Ni", ox=2, ligands=["H2O","H2O","H2O","H2O","H2O", lig])
    """
    if ligands is None:
        ligands = []

    lig_strs = []
    has_custom = False
    for l in ligands:
        if isinstance(l, CustomLigand):
            lig_strs.append(l.name)
            has_custom = True
        else:
            lig_strs.append(str(l))

    # Determine canonical geometry for isomer enumeration
    if geometry:
        geom_canon = resolve_geometry(geometry)
    else:
        cn = 0
        for lname in lig_strs:
            is_smiles = ("=" in lname or "#" in lname or
                         "(" in lname or "[" in lname)
            if not is_smiles:
                try:
                    cn += get_ligand(lname)["denticity"]
                    continue
                except KeyError:
                    pass
            cn += 1
        geom_canon = infer_geometry(cn) if cn > 0 else "oct"

    # Custom ligands can't be meaningfully permuted; skip isomer enumeration
    if has_custom:
        return _build_single(metal, ox, ligands, geom_canon, spin)

    isomers = enumerate_isomers(lig_strs, geom_canon)

    if len(isomers) == 1:
        mol = _build_single(metal, ox, isomers[0]["site_assignment"], geom_canon, spin)
        mol.label = isomers[0]["label"]
        return mol

    results = []
    for iso in isomers:
        mol = _build_single(metal, ox, iso["site_assignment"], geom_canon, spin)
        mol.label = iso["label"]
        results.append(mol)
    return results


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
    terminal : list of str
        Terminal ligands per metal center.
    bridge : str
        Bridging ligand name, e.g. "mu-OH", "mu-HCOO".
    n : int
        Number of bridging units (default 2).
    geometry : str, optional
    mm_bond : bool
        Include a metal–metal bond.
    mm_distance : float, optional
        Override M–M distance (Å).

    Returns
    -------
    Molecule
    """
    if terminal is None:
        terminal = []

    if mm_distance is not None:
        d_mm = mm_distance
    elif mm_bond:
        from molbuilder.core.bond_lengths import COVALENT_RADII
        r = COVALENT_RADII.get(metal, 1.5)
        d_mm = 2 * r + 0.1
    else:
        if bridge:
            try:
                bl_data = get_ligand(bridge.replace("mu-", ""))
                donor = bl_data.get("donor_atoms", ["Cl"])[0]
            except KeyError:
                donor = "Cl"
            bridge_bl = get_bond_length(metal, ox, donor, geometry or "oct")
        else:
            bridge_bl = 2.5
        d_mm = 2 * bridge_bl + 0.5

    per_metal = list(terminal) + ([bridge] * n if bridge else [])
    mol1 = _build_single(metal, ox, per_metal, geometry, None)
    mol2 = _build_single(metal, ox, per_metal, geometry, None)

    offset = np.array([d_mm, 0., 0.])
    for a in mol2.atoms:
        a.position = a.position + offset

    combined = mol1.atoms + mol2.atoms
    return Molecule(
        atoms=combined,
        metal_indices=[0, len(mol1.atoms)],
        formula=_make_formula([a.symbol for a in combined]),
        charge=mol1.charge + mol2.charge,
        spin_multiplicity=mol1.spin_multiplicity,
        geometry=mol1.geometry,
        metal_symbol=metal,
        metal_ox=ox,
        ligand_names=mol1.ligand_names,
    )


def trimer(metal: str,
           ox: int,
           terminal: Optional[List[str]] = None,
           bridge: Optional[str] = None,
           arrangement: str = "triangular",
           geometry: Optional[str] = None) -> Molecule:
    """
    Build a trinuclear complex.

    Parameters
    ----------
    arrangement : "triangular" or "linear"
    """
    if terminal is None:
        terminal = []

    per_metal = list(terminal) + ([bridge, bridge] if bridge else [])
    m1 = _build_single(metal, ox, per_metal, geometry, None)

    bl_ref = 2.5
    if bridge:
        try:
            bl_d = get_ligand(bridge.replace("mu-", ""))
            donor = bl_d.get("donor_atoms", ["C"])[0]
            bl_ref = get_bond_length(metal, ox, donor, geometry or "oct")
        except KeyError:
            pass
    d_mm = 2 * bl_ref + 0.4

    if arrangement == "triangular":
        offsets = [np.zeros(3),
                   np.array([d_mm, 0., 0.]),
                   np.array([d_mm / 2, d_mm * np.sqrt(3) / 2, 0.])]
    else:
        offsets = [np.zeros(3),
                   np.array([d_mm, 0., 0.]),
                   np.array([2 * d_mm, 0., 0.])]

    all_atoms, metal_indices, n = [], [], 0
    for off in offsets:
        mk = _build_single(metal, ox, per_metal, geometry, None)
        for a in mk.atoms:
            a.position = a.position + off
        metal_indices.append(n)
        all_atoms.extend(mk.atoms)
        n += len(mk.atoms)

    return Molecule(
        atoms=all_atoms,
        metal_indices=metal_indices,
        formula=_make_formula([a.symbol for a in all_atoms]),
        charge=m1.charge * 3,
        spin_multiplicity=m1.spin_multiplicity,
        geometry=m1.geometry,
        metal_symbol=metal,
        metal_ox=ox,
        ligand_names=m1.ligand_names,
    )


# ── file output ───────────────────────────────────────────────────────────────

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
    label = f"  [{getattr(mol, 'label', '')}]" if getattr(mol, 'label', '') not in ('', 'only') else ''
    print(f"Formula          : {mol.formula}{label}")
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
