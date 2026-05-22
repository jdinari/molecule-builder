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
    place_ligand, place_bidentate_ligand,
    get_ligand_atoms, get_ligand_atoms_multidentate,
    _rodrigues_rotation, _rot_around_axis,
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
        return {
            "atoms": [],
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
        return {
            "atoms": [],          # not used; place_ligand() handles geometry
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
        "bite_angle": lig_data.get("bite_angle", 55.0),
        "vectors_count": 1,   # bidentate occupies ONE geometry site (bisector)
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
            donor_abs  = v * bl
            metal_abs  = np.zeros(3)

            # All other coordination sites (for torsion optimisation)
            remaining_vecs = [vecs[(vec_idx + k) % len(vecs)] * bl
                              for k in range(1, cn - vec_idx)]
            # Also include already-placed donor atoms
            already_placed = [a.position for a in mol_atoms
                              if a.symbol not in ('H',) and a.symbol != metal]

            adjacent = remaining_vecs + already_placed

            # Place the full ligand with optimal torsion
            placed = place_ligand(e["name"], donor_abs, metal_abs, adjacent)

            for sym, abs_pos in placed:
                mol_atoms.append(Atom(symbol=sym, position=abs_pos,
                                      label=f"{sym}{len(mol_atoms)}"))
            vec_idx += 1

        elif d == 2:
            # Bidentate chelate: use ONE geometry vector as the bisector direction.
            # Fan the two donor atoms out by ±(bite_angle/2) from the bisector.
            # This places the chelate at the correct bite angle regardless of which
            # geometry site it occupies, and avoids the impossible-geometry problem
            # that arises from assigning two 90-deg-apart site vectors.
            v_bisect = vecs[vec_idx % len(vecs)]
            vec_idx += 1

            d1 = e["donor_atoms"][0]
            d2 = e["donor_atoms"][1] if len(e["donor_atoms"]) > 1 else e["donor_atoms"][0]
            bl1 = get_bond_length(metal, ox, d1, geom_canon)
            bl2 = get_bond_length(metal, ox, d2, geom_canon)
            bite = e.get("bite_angle") or 55.0
            half = np.radians(bite / 2)

            # Build a rotation axis perpendicular to v_bisect and lying in a
            # sensible plane (prefer the plane containing v_bisect and z-axis)
            z = np.array([0., 0., 1.])
            perp = np.cross(v_bisect, z)
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(v_bisect, np.array([1., 0., 0.]))
            perp /= np.linalg.norm(perp)

            from molbuilder.ligands.ligand_geometry import _rot_around_axis
            v1 = _rot_around_axis(perp,  half) @ v_bisect
            v2 = _rot_around_axis(perp, -half) @ v_bisect

            placed = place_bidentate_ligand(e["name"], v1, v2, bl1, bl2,
                                            metal_pos=mol_atoms[0].position)
            for sym, pos in placed:
                mol_atoms.append(Atom(symbol=sym, position=pos,
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

    # Custom ligands or multidentate ligands: skip isomer enumeration
    has_multidentate = any(
        not ("=" in l or "#" in l or "(" in l or "[" in l) and
        not isinstance(l, CustomLigand) and
        get_ligand(str(l)).get("denticity", 1) > 1
        for l in ligands
        if not isinstance(l, CustomLigand)
    )

    if has_custom or has_multidentate:
        mol = _build_single(metal, ox, ligands, geom_canon, spin)
        mol.label = "only"
        return mol

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


def _check_assembly_clashes(mol: "Molecule",
                             tol_scale: float = 0.85) -> list:
    """
    Return a list of (i, j, sym_i, sym_j, distance) tuples for every
    non-bonded atom pair closer than tol_scale * min_nonbonded threshold.
    Bonded pairs (d <= MAX_BOND) are excluded automatically.
    """
    from molbuilder.ligands.ligand_geometry import _MIN_NONBONDED, _MAX_BOND, _pair_key
    clashes = []
    atoms = mol.atoms
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            si, sj = atoms[i].symbol, atoms[j].symbol
            d = float(np.linalg.norm(atoms[i].position - atoms[j].position))
            # skip bonded pairs
            max_bond = _MAX_BOND.get(_pair_key(si, sj), 2.0)
            if d <= max_bond:
                continue
            threshold = _MIN_NONBONDED.get(_pair_key(si, sj), 2.0) * tol_scale
            if d < threshold:
                clashes.append((i, j, si, sj, d))
    return clashes


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

    Bridge ligands are placed geometrically spanning both metals:
    each bridge contributes one donor to each metal center, sitting
    in the equatorial plane between them.  Terminal ligands fill the
    remaining coordination sites on each metal.

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

    # ── Determine M-M distance ────────────────────────────────────────────────
    geom = geometry or "oct"

    if bridge:
        try:
            bl_data = get_ligand(bridge.replace("mu-", ""))
            bridge_donor = bl_data.get("donor_atoms", ["O"])[0]
        except KeyError:
            bridge_donor = "O"
        bridge_bl = get_bond_length(metal, ox, bridge_donor, geom)
    else:
        bridge_bl = 2.5
        bridge_donor = "O"

    if mm_distance is not None:
        d_mm = mm_distance
    elif mm_bond:
        from molbuilder.core.bond_lengths import COVALENT_RADII
        r = COVALENT_RADII.get(metal, 1.5)
        d_mm = 2 * r + 0.1
    else:
        # Realistic M-M distance through a single-atom bridge:
        # M-X-M angle ~105° for mu-OH, ~120° for mu-HCOO
        # d_MM = 2 * bl * sin(angle/2)
        bridge_angle_deg = 105.0 if "OH" in (bridge or "") else 120.0
        d_mm = 2.0 * bridge_bl * np.sin(np.radians(bridge_angle_deg / 2))
        d_mm = max(d_mm, 2.8)   # hard floor for sanity

    # ── Place the two metals along the x-axis ────────────────────────────────
    m1_pos = np.array([-d_mm / 2,  0., 0.])
    m2_pos = np.array([ d_mm / 2,  0., 0.])
    m_axis  = np.array([1., 0., 0.])   # unit vector M1 → M2

    # ── Get all geometry vectors for each metal center ────────────────────────
    cn_term = len(terminal)   # terminal ligands per metal (not counting bridges)
    cn_total = cn_term + n    # total CN per metal
    geom_canon = resolve_geometry(geom) if geom else infer_geometry(cn_total)
    vecs_raw = get_geometry_vectors(geom_canon)
    while len(vecs_raw) < cn_total:
        ang = np.pi * len(vecs_raw) / (cn_total + 1)
        vecs_raw.append(np.array([np.sin(ang), np.cos(ang), 0.3]) /
                        np.linalg.norm([np.sin(ang), np.cos(ang), 0.3]))

    # ── Partition vectors: bridges get the ones closest to ±x-axis ───────────
    # For M1 we want bridge vectors pointing toward M2 (+x hemisphere)
    # For M2 we want bridge vectors pointing toward M1 (-x hemisphere)
    # Sort by x-component (descending for M1 bridge sites)
    vecs_sorted_desc = sorted(vecs_raw, key=lambda v: -v[0])
    bridge_vecs_m1 = vecs_sorted_desc[:n]          # point toward +x (M2)
    terminal_vecs_m1 = vecs_sorted_desc[n:][:cn_term]

    # M2 bridge vectors are the mirror of M1's (flip x)
    bridge_vecs_m2 = [np.array([-v[0], v[1], v[2]]) for v in bridge_vecs_m1]
    terminal_vecs_m2 = terminal_vecs_m1   # same local frame, just offset

    # ── Build atom list ───────────────────────────────────────────────────────
    all_atoms: List[Atom] = []
    total_charge = 0

    def _add_metal(pos, idx_label):
        all_atoms.append(Atom(symbol=metal, position=pos.copy(),
                               label=f"{metal}{idx_label}"))

    def _add_terminal_ligands(metal_pos, vecs, charge_accum):
        charge = charge_accum
        for i, lig_name in enumerate(terminal):
            v = vecs[i] if i < len(vecs) else vecs[-1]
            exp = _expand_ligand(lig_name, metal, ox, geom_canon)
            bl  = get_bond_length(metal, ox, exp["donor_atom"], geom_canon)
            donor_abs = metal_pos + v * bl
            other_donors = [metal_pos + vecs[j] * bl
                            for j in range(len(vecs)) if j != i]
            placed = place_ligand(lig_name, donor_abs, metal_pos, other_donors)
            for sym, pos in placed:
                all_atoms.append(Atom(symbol=sym, position=pos,
                                       label=f"{sym}{len(all_atoms)}"))
            charge += exp["charge"]
        return charge

    def _add_bridge_atoms(bridge_name, m1_pos, m2_pos, bv1, bv2, bl):
        """
        Place a single bridging ligand between m1 and m2.
        The donor atom is placed at the midpoint between the two bridge-vector
        tips, then the ligand body is built pointing away from both metals.
        """
        exp = _expand_ligand(bridge_name, metal, ox, geom_canon)
        # Donor sits at the midpoint of the two ideal donor positions
        d1_ideal = m1_pos + bv1 * bl
        d2_ideal = m2_pos + bv2 * bl
        bridge_mid = (d1_ideal + d2_ideal) / 2.0

        # For mu-OH: single O donor bridging both metals — place one O at midpoint
        # For mu-HCOO: single C-bridging formate; O1 toward m1, O2 toward m2
        bridge_name_base = bridge_name.replace("mu-", "")
        if bridge_name_base in ("OH",):
            # Single bridging atom
            donor_pos = bridge_mid
            all_atoms.append(Atom(symbol="O", position=donor_pos,
                                   label=f"O{len(all_atoms)}"))
            # Add the H pointing away from the M-M axis
            away = np.cross(bv1, np.array([0., 0., 1.]))
            if np.linalg.norm(away) < 1e-6:
                away = np.array([0., 1., 0.])
            away /= np.linalg.norm(away)
            h_pos = donor_pos + 0.96 * away
            all_atoms.append(Atom(symbol="H", position=h_pos,
                                   label=f"H{len(all_atoms)}"))
        else:
            # mu-HCOO: formate with O1 on m1 side, O2 on m2 side
            # C sits above the midpoint away from the M-M axis
            o1_pos = m1_pos + bv1 * bl
            o2_pos = m2_pos + bv2 * bl
            # C at midpoint displaced away from M-M axis
            mid_oo = (o1_pos + o2_pos) / 2.0
            away = np.array([0., 0., 1.0])  # out-of-plane
            # Ensure C is away from both metals
            c_dist = np.sqrt(max(1.26**2 - (np.linalg.norm(o2_pos - o1_pos)/2)**2, 0.01))
            c_pos = mid_oo + c_dist * away
            h_pos = c_pos + 1.09 * away
            all_atoms.append(Atom(symbol="O", position=o1_pos,
                                   label=f"O{len(all_atoms)}"))
            all_atoms.append(Atom(symbol="O", position=o2_pos,
                                   label=f"O{len(all_atoms)}"))
            all_atoms.append(Atom(symbol="C", position=c_pos,
                                   label=f"C{len(all_atoms)}"))
            all_atoms.append(Atom(symbol="H", position=h_pos,
                                   label=f"H{len(all_atoms)}"))

    # Place metals
    metal_idx_1 = len(all_atoms)
    _add_metal(m1_pos, 1)
    metal_idx_2_placeholder = None   # will fill after M1 ligands

    # Terminal ligands on M1
    charge_from_terminal = _add_terminal_ligands(m1_pos, terminal_vecs_m1, 0)

    # Place M2
    metal_idx_2 = len(all_atoms)
    _add_metal(m2_pos, 2)

    # Terminal ligands on M2
    _add_terminal_ligands(m2_pos, terminal_vecs_m2, 0)

    # Bridge ligands (between the two metals)
    bridge_charge = 0
    if bridge:
        try:
            bl_data = get_ligand(bridge.replace("mu-", ""))
            bridge_charge_per = bl_data.get("charge", -1)
        except KeyError:
            bridge_charge_per = -1
        bridge_charge = bridge_charge_per * n

        for i in range(n):
            bv1 = bridge_vecs_m1[i] if i < len(bridge_vecs_m1) else bridge_vecs_m1[-1]
            bv2 = bridge_vecs_m2[i] if i < len(bridge_vecs_m2) else bridge_vecs_m2[-1]
            # Alternate bridge ligands above/below the M-M plane
            if i % 2 == 1:
                bv1 = np.array([bv1[0],  bv1[1], -bv1[2]])
                bv2 = np.array([bv2[0],  bv2[1], -bv2[2]])
            _add_bridge_atoms(bridge, m1_pos, m2_pos, bv1, bv2, bridge_bl)

    total_charge = 2 * ox + charge_from_terminal * 2 + bridge_charge
    spin_mult    = _spin_multiplicity(metal, ox)

    mol = Molecule(
        atoms=all_atoms,
        metal_indices=[metal_idx_1, metal_idx_2],
        formula=_make_formula([a.symbol for a in all_atoms]),
        charge=total_charge,
        spin_multiplicity=spin_mult,
        geometry=geom_canon,
        metal_symbol=metal,
        metal_ox=ox,
        ligand_names=list(terminal) + ([bridge] * n if bridge else []),
    )

    clashes = _check_assembly_clashes(mol)
    if clashes:
        mol._clash_warnings = clashes

    return mol


def trimer(metal: str,
           ox: int,
           terminal: Optional[List[str]] = None,
           bridge: Optional[str] = None,
           arrangement: str = "triangular",
           geometry: Optional[str] = None) -> Molecule:
    """
    Build a trinuclear complex with bridging ligands spanning adjacent metals.

    Each metal center has the bridge ligands on its M-M-facing sides, with
    terminal ligands filling the remaining coordination sites.

    Parameters
    ----------
    arrangement : "triangular" or "linear"
        Triangular: three metals at vertices of an equilateral triangle,
        each bridged to its two neighbours (2 bridges per metal).
        Linear: M1-M2-M3 in a line; M1 and M3 each have 1 bridge to M2,
        M2 has 2 bridges.
    """
    if terminal is None:
        terminal = []

    geom = geometry or "oct"

    if bridge:
        try:
            bl_data = get_ligand(bridge.replace("mu-", ""))
            bridge_donor = bl_data.get("donor_atoms", ["O"])[0]
            bridge_charge_per = bl_data.get("charge", -1)
        except KeyError:
            bridge_donor = "O"
            bridge_charge_per = -1
        bridge_bl = get_bond_length(metal, ox, bridge_donor, geom)
    else:
        bridge_bl = 2.5
        bridge_donor = "O"
        bridge_charge_per = 0

    bridge_angle_deg = 105.0 if "OH" in (bridge or "") else 120.0
    d_mm = 2.0 * bridge_bl * np.sin(np.radians(bridge_angle_deg / 2))
    d_mm = max(d_mm, 2.8)

    # ── Metal positions ────────────────────────────────────────────────────────
    if arrangement == "triangular":
        # Equilateral triangle in xy-plane
        metal_positions = [
            np.array([0.,              0.,              0.]),
            np.array([d_mm,            0.,              0.]),
            np.array([d_mm / 2, d_mm * np.sqrt(3) / 2, 0.]),
        ]
        # For each metal, which two others is it bridged to?
        bridge_pairs = [(0, 1), (1, 2), (2, 0)]   # edges of triangle
        n_bridges_per_metal = 2
    else:
        # Linear: M0-M1-M2
        metal_positions = [
            np.array([0.,     0., 0.]),
            np.array([d_mm,   0., 0.]),
            np.array([2*d_mm, 0., 0.]),
        ]
        bridge_pairs = [(0, 1), (1, 2)]
        n_bridges_per_metal = {0: 1, 1: 2, 2: 1}

    cn_term = len(terminal)
    if arrangement == "triangular":
        cn_total = cn_term + n_bridges_per_metal
    else:
        cn_total = cn_term + 2   # worst case for middle metal
    cn_total = max(cn_total, 2)

    geom_canon = resolve_geometry(geom) if geom else infer_geometry(cn_total)
    spin_mult  = _spin_multiplicity(metal, ox)

    all_atoms: List[Atom] = []
    metal_indices: List[int] = []
    total_charge = 0

    # Track placed metal indices for Molecule metadata
    metal_atom_indices: List[int] = []

    # ── Helper: place terminal ligands around one metal ───────────────────────
    def _place_terminals_trimer(metal_pos, exclude_dirs, lig_list):
        """
        Place terminal ligands, choosing geometry vectors that avoid the
        directions already occupied by bridges (exclude_dirs are unit vectors).
        """
        vecs_all = get_geometry_vectors(geom_canon)
        while len(vecs_all) < cn_term + len(exclude_dirs):
            ang = np.pi * len(vecs_all) / (cn_term + len(exclude_dirs) + 1)
            vecs_all.append(np.array([np.sin(ang), np.cos(ang), 0.3]) /
                            np.linalg.norm([np.sin(ang), np.cos(ang), 0.3]))

        # Rotate the raw geometry vectors to align the most-bridge-like vector
        # toward the mean bridge direction
        if exclude_dirs:
            mean_bridge = np.mean(exclude_dirs, axis=0)
            if np.linalg.norm(mean_bridge) > 1e-6:
                mean_bridge /= np.linalg.norm(mean_bridge)
                # Align first geometry vector toward mean bridge direction
                v0 = vecs_all[0] / np.linalg.norm(vecs_all[0])
                R = _rodrigues_rotation(v0, mean_bridge)
                vecs_all = [R @ v for v in vecs_all]

        # Assign vecs farthest from bridge dirs to terminal ligands
        def min_sim_to_bridges(v):
            if not exclude_dirs:
                return 0.0
            return max(np.dot(v / np.linalg.norm(v), bd) for bd in exclude_dirs)

        vecs_sorted = sorted(vecs_all, key=min_sim_to_bridges)
        term_vecs = vecs_sorted[:cn_term]

        charge = 0
        for i, lig_name in enumerate(lig_list):
            v = term_vecs[i] if i < len(term_vecs) else term_vecs[-1]
            exp = _expand_ligand(lig_name, metal, ox, geom_canon)
            bl  = get_bond_length(metal, ox, exp["donor_atom"], geom_canon)
            donor_abs = metal_pos + v * bl
            other_donors = [metal_pos + term_vecs[j] * bl
                            for j in range(len(term_vecs)) if j != i]
            placed = place_ligand(lig_name, donor_abs, metal_pos, other_donors)
            for sym, pos in placed:
                all_atoms.append(Atom(symbol=sym, position=pos,
                                       label=f"{sym}{len(all_atoms)}"))
            charge += exp["charge"]
        return charge

    # ── Helper: place one bridge between two metal positions ─────────────────
    def _place_bridge_trimer(bridge_name, ma_pos, mb_pos):
        bridge_name_base = bridge_name.replace("mu-", "")
        mid = (ma_pos + mb_pos) / 2.0
        m_to_m = mb_pos - ma_pos
        m_to_m_norm = m_to_m / np.linalg.norm(m_to_m)
        perp = np.cross(m_to_m_norm, np.array([0., 0., 1.]))
        if np.linalg.norm(perp) < 1e-6:
            perp = np.cross(m_to_m_norm, np.array([1., 0., 0.]))
        perp /= np.linalg.norm(perp)

        if bridge_name_base in ("OH",):
            donor_pos = mid
            all_atoms.append(Atom(symbol="O", position=donor_pos.copy(),
                                   label=f"O{len(all_atoms)}"))
            h_pos = donor_pos + 0.96 * perp
            all_atoms.append(Atom(symbol="H", position=h_pos,
                                   label=f"H{len(all_atoms)}"))
        else:
            # mu-HCOO: O1 toward ma, O2 toward mb, C above
            o1_pos = ma_pos + m_to_m_norm * bridge_bl
            o2_pos = mb_pos - m_to_m_norm * bridge_bl
            mid_oo = (o1_pos + o2_pos) / 2.0
            # Use alternating out-of-plane direction for successive bridges
            c_dist = np.sqrt(max(1.26**2 - (np.linalg.norm(o2_pos - o1_pos)/2)**2, 0.01))
            c_pos = mid_oo + c_dist * perp
            h_pos = c_pos + 1.09 * perp
            all_atoms.append(Atom(symbol="O", position=o1_pos,
                                   label=f"O{len(all_atoms)}"))
            all_atoms.append(Atom(symbol="O", position=o2_pos,
                                   label=f"O{len(all_atoms)}"))
            all_atoms.append(Atom(symbol="C", position=c_pos,
                                   label=f"C{len(all_atoms)}"))
            all_atoms.append(Atom(symbol="H", position=h_pos,
                                   label=f"H{len(all_atoms)}"))

    # ── Place metals ───────────────────────────────────────────────────────────
    for mi, mpos in enumerate(metal_positions):
        metal_atom_indices.append(len(all_atoms))
        all_atoms.append(Atom(symbol=metal, position=mpos.copy(),
                               label=f"{metal}{mi+1}"))
        total_charge += ox

    # ── Place bridge ligands (one per edge) ────────────────────────────────────
    bridge_dirs_per_metal = {i: [] for i in range(3)}
    if bridge:
        for edge_idx, (ia, ib) in enumerate(bridge_pairs):
            ma = metal_positions[ia]
            mb = metal_positions[ib]
            # Alternate out-of-plane to avoid clashes for multiple bridges
            if edge_idx % 2 == 1:
                # reflect z for alternating bridges
                old_perp_z = 1.0
            _place_bridge_trimer(bridge, ma, mb)
            total_charge += bridge_charge_per
            # Record bridge directions for terminal placement
            d_ab = (mb - ma) / np.linalg.norm(mb - ma)
            bridge_dirs_per_metal[ia].append( d_ab)
            bridge_dirs_per_metal[ib].append(-d_ab)

    # ── Place terminal ligands on each metal ──────────────────────────────────
    for mi, mpos in enumerate(metal_positions):
        charge = _place_terminals_trimer(mpos, bridge_dirs_per_metal[mi], terminal)
        total_charge += charge

    mol = Molecule(
        atoms=all_atoms,
        metal_indices=metal_atom_indices,
        formula=_make_formula([a.symbol for a in all_atoms]),
        charge=total_charge,
        spin_multiplicity=spin_mult,
        geometry=geom_canon,
        metal_symbol=metal,
        metal_ox=ox,
        ligand_names=list(terminal) + ([bridge] * len(bridge_pairs) if bridge else []),
    )

    clashes = _check_assembly_clashes(mol)
    if clashes:
        mol._clash_warnings = clashes

    return mol


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
