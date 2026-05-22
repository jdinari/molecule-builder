"""
api.py
======
Public API for molbuilder.

    from molbuilder.api import build, dimer, trimer, poscar, xyz, info

build() always returns all symmetry-distinct isomers automatically:
  - One isomer  → returns a single Molecule
  - Two or more → returns a list of Molecule objects

Custom POSCAR ligands are supported via load_ligand_from_poscar().
Denticity modes use colon notation: "HCOO:bi", "bpy:mono", etc.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Optional, Dict
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

_UNPAIRED = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
             6: 4, 7: 3, 8: 2, 9: 1, 10: 0}

def _spin_multiplicity(metal: str, ox: int) -> int:
    d = _D_ELECTRONS.get(metal, {}).get(ox, 0)
    return _UNPAIRED.get(d, 1) + 1


# ── SMILES donor-atom heuristic ──────────────────────────────────────────────

def _donor_from_smiles(smiles: str) -> str:
    for sym in ["P", "S", "N", "O", "C"]:
        if sym in smiles:
            return sym
    return "C"


# ── CustomLigand ─────────────────────────────────────────────────────────────

class CustomLigand:
    """A ligand loaded from a POSCAR file."""

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
        cell  = np.array([[float(x) for x in lines[i].split()]
                          for i in range(2, 5)]) * scale
        syms   = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        symbols = [s for s, n in zip(syms, counts) for _ in range(n)]
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


def load_ligand_from_poscar(poscar_path, donor_atom_indices: List[int],
                             charge: int = 0,
                             name: Optional[str] = None) -> CustomLigand:
    """Load a custom ligand from a POSCAR file."""
    return CustomLigand(poscar_path, donor_atom_indices, charge, name)


# ── ligand expansion ─────────────────────────────────────────────────────────

def _expand_ligand(lig, metal: str, ox: int, geometry: str) -> dict:
    if isinstance(lig, CustomLigand):
        d_syms = lig.donor_atoms
        d_idx  = lig.donor_atom_indices
        d0_pos = lig.positions[d_idx[0]]
        rel_atoms = [(s, p - d0_pos)
                     for s, p in zip(lig.symbols, lig.positions)]
        non_donor = [p for i, (s, p) in enumerate(rel_atoms)
                     if i not in d_idx and s != "H"]
        if non_donor:
            bulk = np.mean(non_donor, axis=0)
            if np.linalg.norm(bulk) > 1e-3:
                R = _rodrigues_rotation(bulk / np.linalg.norm(bulk),
                                        np.array([-1., 0., 0.]))
                rel_atoms = [(s, R @ p) for s, p in rel_atoms]
        return {
            "atoms": rel_atoms, "donor_atom": d_syms[0],
            "donor_atoms": d_syms, "donors": d_idx, "smiles": "",
            "charge": lig.charge, "denticity": lig.denticity,
            "bite_angle": None, "vectors_count": lig.denticity,
            "name": lig.name,
        }

    lig_name = str(lig)
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
            "atoms": [], "donor_atom": donor, "donor_atoms": [donor],
            "donors": [0], "smiles": lig_name, "charge": 0, "denticity": 1,
            "bite_angle": None, "vectors_count": 1, "name": lig_name,
        }

    lig_data = get_ligand(lig_name)
    donor_atoms   = lig_data.get("donor_atoms", ["N"])
    smiles        = lig_data.get("smiles", "")
    donor_indices = lig_data.get("donors", [0])

    if lig_data["denticity"] == 1:
        return {
            "atoms": [], "donor_atom": donor_atoms[0],
            "donor_atoms": donor_atoms, "donors": donor_indices,
            "smiles": smiles, "charge": lig_data["charge"], "denticity": 1,
            "bite_angle": None, "vectors_count": 1, "name": lig_name,
        }

    return {
        "atoms": [], "donor_atom": donor_atoms[0],
        "donor_atoms": donor_atoms, "donors": donor_indices,
        "smiles": smiles, "charge": lig_data["charge"],
        "denticity": lig_data["denticity"],
        "bite_angle": lig_data.get("bite_angle", 55.0),
        "vectors_count": 1,
        "name": lig_name,
    }


# ── formula ───────────────────────────────────────────────────────────────────

def _make_formula(symbols: List[str]) -> str:
    cnt = Counter(symbols)
    return "".join(el if n == 1 else f"{el}{n}"
                   for el, n in sorted(cnt.items()))


# ── clash checker ─────────────────────────────────────────────────────────────

# Minimum non-bonded distances — calibrated to allow normal bond lengths
# while flagging genuine overlaps
_NB_MIN = {
    ("H",  "H"):  1.20,
    ("H",  "C"):  1.60,
    ("H",  "N"):  1.60,
    ("H",  "O"):  1.20,   # compact clusters can have H...O ~1.27 Å before relaxation
    ("H",  "Cl"): 1.80,
    ("C",  "C"):  2.00,
    ("C",  "O"):  1.70,
    ("C",  "N"):  1.90,
    ("N",  "O"):  1.80,
    ("O",  "O"):  1.60,
}
# Max bond lengths (to distinguish bonded from non-bonded)
_BOND_MAX = {
    ("H","C"):1.15, ("H","N"):1.15, ("H","O"):1.10,
    ("C","C"):1.60, ("C","N"):1.55, ("C","O"):1.55,
    ("O","O"):1.55, ("N","N"):1.50,
}

def _pair(a, b):
    return tuple(sorted([a, b]))

def _is_bond(s1, s2, d):
    return d <= _BOND_MAX.get(_pair(s1, s2), 2.0)

def _is_overlap(s1, s2, d):
    """True only for genuine problematic clashes, not normal bonds."""
    if _is_bond(s1, s2, d):
        return False
    return d < _NB_MIN.get(_pair(s1, s2), 1.50)

def check_structure(mol: Molecule) -> List[str]:
    """
    Check a molecule for problematic atom overlaps.
    Returns a list of warning strings (empty = clean).
    Skips metal-ligand contacts (those are coordination bonds).
    """
    atoms  = mol.atoms
    metals = {mol.metal_symbol} if mol.metal_symbol else set()
    warns  = []
    for i, a in enumerate(atoms):
        for j, b in enumerate(atoms):
            if j <= i:
                continue
            # Skip metal–anything contacts (coordination bonds)
            if a.symbol in metals or b.symbol in metals:
                continue
            d = np.linalg.norm(a.position - b.position)
            if _is_overlap(a.symbol, b.symbol, d):
                warns.append(
                    f"atom {i+1}({a.symbol}) – atom {j+1}({b.symbol}): {d:.3f} Å"
                )
    return warns


# ── core single-structure builder ─────────────────────────────────────────────

def _build_single(metal: str, ox: int, ligands: List, geometry: str,
                  spin: Optional[int],
                  metal_pos: Optional[np.ndarray] = None) -> Molecule:
    """
    Build one specific arrangement of ligands around a single metal center.
    metal_pos: where to place the metal (default origin).
    """
    if metal_pos is None:
        metal_pos = np.zeros(3)

    expanded = [_expand_ligand(l, metal, ox, geometry) for l in ligands]

    cn = sum(e["vectors_count"] for e in expanded) or 6
    geom_canon = resolve_geometry(geometry) if geometry else infer_geometry(cn)

    vecs = get_geometry_vectors(geom_canon)
    while cn > len(vecs):
        a = np.pi * len(vecs) / (cn + 1)
        v = np.array([np.sin(a), np.cos(a), 0.3])
        vecs.append(v / np.linalg.norm(v))

    mol_atoms: List[Atom] = [
        Atom(symbol=metal, position=metal_pos.copy(), label=f"{metal}1")
    ]
    total_charge = ox
    lig_names_out = []
    vec_idx = 0

    for e in expanded:
        d = e["denticity"]
        total_charge += e["charge"]
        lig_names_out.append(e["name"])

        if d == 1:
            v   = vecs[vec_idx % len(vecs)]
            bl  = get_bond_length(metal, ox, e["donor_atom"], geom_canon)
            donor_abs = metal_pos + v * bl

            # adjacent donor positions for torsion optimisation
            remaining = [metal_pos + vecs[(vec_idx + k) % len(vecs)] * bl
                         for k in range(1, max(1, cn - vec_idx))]
            already   = [a.position for a in mol_atoms
                         if a.symbol not in ("H",) and a.symbol != metal]
            adjacent  = remaining + already

            placed = place_ligand(e["name"], donor_abs, metal_pos, adjacent)
            for sym, pos in placed:
                mol_atoms.append(Atom(symbol=sym, position=pos,
                                      label=f"{sym}{len(mol_atoms)}"))
            vec_idx += 1

        elif d == 2:
            v_bisect = vecs[vec_idx % len(vecs)]
            vec_idx += 1
            d1  = e["donor_atoms"][0]
            d2  = e["donor_atoms"][1] if len(e["donor_atoms"]) > 1 else d1
            bl1 = get_bond_length(metal, ox, d1, geom_canon)
            bl2 = get_bond_length(metal, ox, d2, geom_canon)
            bite = e.get("bite_angle") or 55.0
            half = np.radians(bite / 2)

            z    = np.array([0., 0., 1.])
            perp = np.cross(v_bisect, z)
            if np.linalg.norm(perp) < 1e-6:
                perp = np.cross(v_bisect, np.array([1., 0., 0.]))
            perp /= np.linalg.norm(perp)

            v1 = _rot_around_axis(perp,  half) @ v_bisect
            v2 = _rot_around_axis(perp, -half) @ v_bisect

            placed = place_bidentate_ligand(e["name"], v1, v2, bl1, bl2,
                                            metal_pos=metal_pos)
            for sym, pos in placed:
                mol_atoms.append(Atom(symbol=sym, position=pos,
                                      label=f"{sym}{len(mol_atoms)}"))

        else:
            d_list = e["donor_atoms"]
            bls    = [get_bond_length(metal, ox, da, geom_canon) for da in d_list]
            multi  = get_ligand_atoms_multidentate(
                e["name"], e["smiles"], d_list,
                e.get("donors", list(range(len(d_list)))),
                bite_angle_deg=e["bite_angle"] or 90.0,
                bond_lengths=bls,
            )
            mid_vecs = [vecs[(vec_idx + i) % len(vecs)] for i in range(d)]
            mid      = np.mean(mid_vecs, axis=0)
            if np.linalg.norm(mid) > 1e-6:
                R = _rodrigues_rotation(np.array([1., 0., 0.]),
                                        mid / np.linalg.norm(mid))
            else:
                R = np.eye(3)
            for sym, pos in multi:
                mol_atoms.append(Atom(symbol=sym, position=metal_pos + R @ pos,
                                      label=f"{sym}{len(mol_atoms)}"))
            vec_idx += d

    formula   = _make_formula([a.symbol for a in mol_atoms])
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


# ── public build() ────────────────────────────────────────────────────────────

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

    Each molecule has a .label attribute ("fac", "mer", "cis", "trans", …).
    Isomers only differ when chemically different ligands end up in different
    spatial arrangements — swapping identical ligands never produces a new isomer.

    Parameters
    ----------
    metal : str
    ox : int
    ligands : list
        Ligand names, SMILES strings, or CustomLigand objects.
        Denticity modes use colon notation: "HCOO:bi", "bpy:mono".
    geometry : str, optional
        Auto-inferred from coordination number if omitted.
    spin : int, optional
        Auto-estimated if omitted.

    Returns
    -------
    Molecule  or  list[Molecule]
    """
    if ligands is None:
        ligands = []

    lig_strs   = []
    has_custom = False
    has_multi  = False
    for l in ligands:
        if isinstance(l, CustomLigand):
            lig_strs.append(l.name)
            has_custom = True
        else:
            lig_strs.append(str(l))
            try:
                if get_ligand(str(l)).get("denticity", 1) > 1:
                    has_multi = True
            except KeyError:
                pass

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

    if has_custom or has_multi:
        mol = _build_single(metal, ox, ligands, geom_canon, spin)
        mol.label = "only"
        return mol

    isomers = enumerate_isomers(lig_strs, geom_canon)

    if len(isomers) == 1:
        mol = _build_single(metal, ox, isomers[0]["site_assignment"],
                            geom_canon, spin)
        mol.label = isomers[0]["label"]
        return mol

    results = []
    for iso in isomers:
        mol = _build_single(metal, ox, iso["site_assignment"], geom_canon, spin)
        mol.label = iso["label"]
        results.append(mol)
    return results


def build_isomers(metal: str,
                  ox: int,
                  ligands: Optional[List] = None,
                  geometry: Optional[str] = None,
                  spin: Optional[int] = None) -> List[Molecule]:
    """
    Alias for build() that always returns a list, for convenience.
    """
    result = build(metal, ox=ox, ligands=ligands, geometry=geometry, spin=spin)
    return result if isinstance(result, list) else [result]



# ── shared bridge-placement helpers ──────────────────────────────────────────

def _perp_basis(mm_hat: np.ndarray):
    """Return two orthonormal vectors spanning the plane perpendicular to mm_hat."""
    ref   = np.array([0., 0., 1.]) if abs(mm_hat[2]) < 0.9 else np.array([0., 1., 0.])
    perp1 = np.cross(mm_hat, ref);   perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(mm_hat, perp1); perp2 /= np.linalg.norm(perp2)
    return perp1, perp2


def _best_bridge_direction(mm_hat: np.ndarray,
                           anchor: np.ndarray,
                           placed_atoms: List,
                           metal_symbol: str,
                           n_total: int,
                           bridge_idx: int,
                           chosen_dirs: List[np.ndarray],
                           n_samples: int = 72,
                           extra_anchors: Optional[List[np.ndarray]] = None) -> np.ndarray:
    """Pick the perpendicular-to-mm_hat direction with the most spatial clearance.

    Parameters
    ----------
    mm_hat        : unit vector along M-M axis
    anchor        : primary reference point (M-M midpoint)
    placed_atoms  : list of Atom objects placed so far
    metal_symbol  : skip atoms of this element when measuring clearance
    n_total       : total number of bridges being placed in this call
    bridge_idx    : index of the current bridge (0-based)
    chosen_dirs   : directions already chosen for bridges 0..bridge_idx-1
    n_samples     : angular resolution of the sweep
    extra_anchors : additional reference points (e.g. each metal centre) —
                    clearance is the MINIMUM across all anchors, so the chosen
                    direction avoids crowding near every metal, not just the midpoint

    Strategy
    --------
    1. For each of n_samples candidate directions in the perp plane, compute
       the minimum angular clearance to all already-placed non-metal atoms,
       measured from every anchor point (midpoint + each metal centre).
       Taking the minimum across anchors ensures bridges avoid both the
       region near M1 (where O1 lands) and near M2 (where O2 lands).
    2. Enforce a minimum angular gap between this bridge and previously
       chosen bridges in this call (spacing ≥ 360/n_total × 0.8 deg).
    3. Return the direction with the highest minimum clearance.
       Falls back to uniform distribution if the structure is empty.
    """
    perp1, perp2 = _perp_basis(mm_hat)

    all_anchors = [anchor] + (extra_anchors or [])

    # For each anchor, collect unit vectors to all non-metal placed atoms
    occupied_per_anchor: List[List[np.ndarray]] = []
    for anc in all_anchors:
        dirs: List[np.ndarray] = []
        for a in placed_atoms:
            if a.symbol == metal_symbol:
                continue
            v = a.position - anc
            d = np.linalg.norm(v)
            if d > 0.1:
                dirs.append(v / d)
        occupied_per_anchor.append(dirs)

    # Minimum angular separation we want between consecutive bridges
    min_gap_deg = 360.0 / max(n_total, 1) * 0.8

    best_phi   = 2.0 * np.pi * bridge_idx / max(n_total, 1)  # uniform fallback
    best_score = -1.0

    for k in range(n_samples):
        phi  = 2.0 * np.pi * k / n_samples
        cand = np.cos(phi) * perp1 + np.sin(phi) * perp2

        # Clearance = minimum angular distance across ALL anchors
        clearance = 180.0
        for occ in occupied_per_anchor:
            if occ:
                dots   = [max(-1.0, min(1.0, float(np.dot(cand, o)))) for o in occ]
                angles = [np.degrees(np.arccos(d)) for d in dots]
                clearance = min(clearance, min(angles))

        # Penalise directions too close to already-chosen bridge directions
        for prev_dir in chosen_dirs:
            dot   = max(-1.0, min(1.0, float(np.dot(cand, prev_dir))))
            inter = np.degrees(np.arccos(dot))
            if inter < min_gap_deg:
                clearance = -1.0
                break

        if clearance > best_score:
            best_score = clearance
            best_phi   = phi

    return np.cos(best_phi) * perp1 + np.sin(best_phi) * perp2


def _place_oh_bridge(m1_p: np.ndarray, m2_p: np.ndarray,
                     mm_hat: np.ndarray, d_mm: float, bl: float,
                     perp_dir: np.ndarray) -> List[tuple]:
    """Return [(symbol, position), ...] for a mu-OH bridge.

    O is placed at the geometrically correct position equidistant from both
    metals (displaced perpendicular to M-M by sqrt(bl²−(d_mm/2)²)).
    H points outward along perp_dir.
    """
    perp_offset = np.sqrt(max(bl**2 - (d_mm / 2.0)**2, 0.0))
    mid   = (m1_p + m2_p) / 2.0
    o_pos = mid   + perp_offset * perp_dir
    h_pos = o_pos + 0.960 * perp_dir
    return [("O", o_pos), ("H", h_pos)]


def _place_hcoo_bridge(m1_p: np.ndarray, m2_p: np.ndarray,
                       mm_hat: np.ndarray, d_mm: float, bl: float,
                       perp_dir: np.ndarray,
                       target_oo: float = 2.2) -> List[tuple]:
    """Return [(symbol, position), ...] for a mu-HCOO (syn-syn) bridge.

    Both O atoms are tilted toward perp_dir at the angle that gives the
    target O-O distance (~2.2 Å).  C bridges above the O-O midpoint.
    """
    cos_t = (d_mm - target_oo) / (2.0 * bl)
    cos_t = max(min(cos_t, 0.99), -0.99)
    sin_t = np.sqrt(1.0 - cos_t**2)
    o1    = m1_p + bl * ( cos_t * mm_hat + sin_t * perp_dir)
    o2    = m2_p + bl * (-cos_t * mm_hat + sin_t * perp_dir)
    mid_oo  = (o1 + o2) / 2.0
    oo_half = np.linalg.norm(o2 - o1) / 2.0
    c_dist  = np.sqrt(max(1.26**2 - oo_half**2, 0.1))
    c_pos   = mid_oo + c_dist * perp_dir
    h_pos   = c_pos  + 1.09  * perp_dir
    return [("O", o1), ("O", o2), ("C", c_pos), ("H", h_pos)]


def _place_generic_bridge(m1_p: np.ndarray, m2_p: np.ndarray,
                          bl: float, donor_sym: str) -> List[tuple]:
    """Return [(symbol, position), ...] for a generic single-atom bridge.

    Each donor is placed one bond-length from its metal along the M-M axis.
    """
    mid     = (m1_p + m2_p) / 2.0
    d1 = m1_p + bl * (mid - m1_p) / np.linalg.norm(mid - m1_p)
    d2 = m2_p + bl * (mid - m2_p) / np.linalg.norm(mid - m2_p)
    return [(donor_sym, d1), (donor_sym, d2)]


def _append_bridge_atoms(atoms_list: List, pairs: List[tuple]):
    """Append (symbol, position) pairs to atoms_list as labelled Atoms."""
    for sym, pos in pairs:
        atoms_list.append(Atom(symbol=sym, position=pos,
                               label=f"{sym}{len(atoms_list)}"))


# ── dimer() ───────────────────────────────────────────────────────────────────

def dimer(metal: str,
          ox: int,
          terminal: Optional[List[str]] = None,
          bridge: Optional[str] = None,
          n: int = 2,
          geometry: Optional[str] = None,
          mm_bond: bool = False,
          mm_distance: Optional[float] = None) -> Molecule:
    """
    Build a dinuclear complex.

    The two metal centres are placed symmetrically along the x-axis.
    Each bridge ligand spans the two metals using a realistic M-X-M angle.
    Terminal ligands fill the remaining coordination sites.
    No metal-metal bond is present unless mm_bond=True.
    """
    if terminal is None:
        terminal = []

    geom_canon = resolve_geometry(geometry) if geometry else "oct"

    # ── bond lengths ──────────────────────────────────────────────────────────
    if bridge:
        try:
            bl_data      = get_ligand(bridge.replace("mu-", ""))
            bridge_donor = bl_data.get("donor_atoms", ["O"])[0]
        except KeyError:
            bridge_donor = "O"
        bridge_bl = get_bond_length(metal, ox, bridge_donor, geom_canon)
    else:
        bridge_bl    = 2.5
        bridge_donor = "O"

    term_bl = get_bond_length(metal, ox, "O", geom_canon)  # representative

    # ── M-M distance ──────────────────────────────────────────────────────────
    if mm_distance is not None:
        d_mm = mm_distance
    elif mm_bond:
        from molbuilder.core.bond_lengths import COVALENT_RADII
        d_mm = 2 * COVALENT_RADII.get(metal, 1.5) + 0.1
    else:
        # Realistic M-X-M angle: ~105° for μ-OH, ~150° for μ-HCOO
        # (syn-syn formate bridges in real crystal structures have Ni-Ni ~3.8 Å,
        # corresponding to an effective M-O-M angle of ~150° when accounting
        # for the O-C-O bridging geometry)
        mxm_angle = 105.0 if "OH" in (bridge or "") else 150.0
        d_mm = 2.0 * bridge_bl * np.sin(np.radians(mxm_angle / 2))
        d_mm = max(d_mm, 2 * bridge_bl * 0.5)

    # ── metal positions ───────────────────────────────────────────────────────
    m1_pos = np.array([-d_mm / 2, 0., 0.])
    m2_pos = np.array([ d_mm / 2, 0., 0.])

    # ── coordinate system ─────────────────────────────────────────────────────
    # +x  = M1 → M2 (bridging axis)
    # +y,+z = equatorial / axial directions

    # Total CN per metal
    cn_term  = sum(_expand_ligand(l, metal, ox, geom_canon)["vectors_count"]
                   for l in terminal)
    cn_total = cn_term + n
    if cn_total == 0:
        cn_total = 6
    geom_canon = resolve_geometry(geometry) if geometry else infer_geometry(cn_total)
    vecs = get_geometry_vectors(geom_canon)

    # ── partition geometry vectors ────────────────────────────────────────────
    # Bridge sites: the n vectors with the LARGEST |x| component that point
    # toward the other metal (+x for M1, -x for M2).
    # Terminal sites: everything else.
    vecs_by_abs_x = sorted(range(len(vecs)), key=lambda i: -abs(vecs[i][0]))
    bridge_indices  = vecs_by_abs_x[:n]
    terminal_indices = [i for i in range(len(vecs)) if i not in bridge_indices][:cn_term]

    bridge_vecs_m1  = [vecs[i] if vecs[i][0] >= 0 else -vecs[i]
                       for i in bridge_indices]
    terminal_vecs_m1 = [vecs[i] for i in terminal_indices]
    # M2 mirrors M1: flip x
    bridge_vecs_m2  = [np.array([-v[0], v[1], v[2]]) for v in bridge_vecs_m1]
    terminal_vecs_m2 = [np.array([-v[0], v[1], v[2]]) for v in terminal_vecs_m1]

    # ── build atom list ───────────────────────────────────────────────────────
    all_atoms:    List[Atom] = []
    total_charge: int = 0

    def add_metal(pos, label):
        all_atoms.append(Atom(symbol=metal, position=pos.copy(), label=label))

    def add_terminal(metal_pos, vecs_local):
        charge = 0
        for i, lig_name in enumerate(terminal):
            e   = _expand_ligand(lig_name, metal, ox, geom_canon)
            bl  = get_bond_length(metal, ox, e["donor_atom"], geom_canon)
            v   = vecs_local[i] if i < len(vecs_local) else vecs_local[-1]

            if e["denticity"] == 1:
                donor_abs = metal_pos + v * bl
                adj = [metal_pos + vecs_local[j] * bl
                       for j in range(len(vecs_local)) if j != i]
                placed = place_ligand(lig_name, donor_abs, metal_pos, adj)
                for sym, pos in placed:
                    all_atoms.append(Atom(symbol=sym, position=pos,
                                          label=f"{sym}{len(all_atoms)}"))
            elif e["denticity"] == 2:
                bite = e.get("bite_angle") or 55.0
                half = np.radians(bite / 2)
                z    = np.array([0., 0., 1.])
                perp = np.cross(v, z)
                if np.linalg.norm(perp) < 1e-6:
                    perp = np.cross(v, np.array([1., 0., 0.]))
                perp /= np.linalg.norm(perp)
                v1 = _rot_around_axis(perp,  half) @ v
                v2 = _rot_around_axis(perp, -half) @ v
                d1 = e["donor_atoms"][0]
                d2 = e["donor_atoms"][1] if len(e["donor_atoms"]) > 1 else d1
                bl1 = get_bond_length(metal, ox, d1, geom_canon)
                bl2 = get_bond_length(metal, ox, d2, geom_canon)
                placed = place_bidentate_ligand(lig_name, v1, v2, bl1, bl2,
                                                metal_pos=metal_pos)
                for sym, pos in placed:
                    all_atoms.append(Atom(symbol=sym, position=pos,
                                          label=f"{sym}{len(all_atoms)}"))
            charge += e["charge"]
        return charge

    def add_bridge(bridge_name, m1_p, m2_p, bridge_idx=0, chosen_dirs=None):
        """Place one bridging ligand spanning m1_p and m2_p.

        Uses the module-level spatial-clearance sweep to pick the perpendicular
        direction with the most free space, so bridges avoid terminal ligands
        and each other regardless of geometry or bridge count.
        """
        if chosen_dirs is None:
            chosen_dirs = []

        bl        = get_bond_length(metal, ox, bridge_donor, geom_canon)
        base_name = bridge_name.replace("mu-", "")

        mm_vec = m2_p - m1_p
        d_mm   = np.linalg.norm(mm_vec)
        mm_hat = mm_vec / d_mm
        anchor = (m1_p + m2_p) / 2.0

        perp_dir = _best_bridge_direction(
            mm_hat, anchor, all_atoms, metal,
            n_total=n, bridge_idx=bridge_idx, chosen_dirs=chosen_dirs,
            extra_anchors=[m1_p, m2_p],
        )

        if base_name == "OH":
            pairs = _place_oh_bridge(m1_p, m2_p, mm_hat, d_mm, bl, perp_dir)
        elif base_name in ("HCOO", "formate"):
            pairs = _place_hcoo_bridge(m1_p, m2_p, mm_hat, d_mm, bl, perp_dir)
        else:
            pairs = _place_generic_bridge(m1_p, m2_p, bl, bridge_donor)

        _append_bridge_atoms(all_atoms, pairs)
        return perp_dir  # caller stores this in chosen_dirs

    # Place M1
    add_metal(m1_pos, f"{metal}1")
    tc1 = add_terminal(m1_pos, terminal_vecs_m1)

    # Place M2
    add_metal(m2_pos, f"{metal}2")
    tc2 = add_terminal(m2_pos, terminal_vecs_m2)

    # Place bridge ligands — alternate above/below M-M axis
    bridge_charge = 0
    if bridge:
        try:
            bdata         = get_ligand(bridge.replace("mu-", ""))
            bridge_charge = bdata.get("charge", -1) * n
        except KeyError:
            bridge_charge = -n

        chosen_dirs: List[np.ndarray] = []
        for i in range(n):
            d = add_bridge(bridge, m1_pos, m2_pos, bridge_idx=i,
                           chosen_dirs=chosen_dirs)
            chosen_dirs.append(d)

    total_charge = 2 * ox + tc1 + tc2 + bridge_charge

    mol = Molecule(
        atoms=all_atoms,
        metal_indices=[0, 1 + sum(
            len(place_ligand(l, np.zeros(3), np.array([2.,0.,0.]), []))
            if not isinstance(l, CustomLigand) and
               (lambda e: e["denticity"] == 1)(_expand_ligand(l, metal, ox, geom_canon))
            else 2
            for l in terminal
        )],
        formula=_make_formula([a.symbol for a in all_atoms]),
        charge=total_charge,
        spin_multiplicity=_spin_multiplicity(metal, ox),
        geometry=geom_canon,
        metal_symbol=metal,
        metal_ox=ox,
        ligand_names=list(terminal) + ([bridge] * n if bridge else []),
    )
    return mol


# ── trimer() ──────────────────────────────────────────────────────────────────

def trimer(metal: str,
           ox: int,
           terminal: Optional[List[str]] = None,
           bridge: Optional[str] = None,
           arrangement: str = "triangular",
           geometry: Optional[str] = None) -> Molecule:
    """
    Build a trinuclear complex (linear or triangular).
    Each adjacent pair of metals is connected by the bridge ligand.
    """
    if terminal is None:
        terminal = []

    geom_canon = resolve_geometry(geometry) if geometry else "oct"

    # Build the asymmetric unit (one Ni with its terminal + 2 bridge donors)
    if bridge:
        try:
            bl_data      = get_ligand(bridge.replace("mu-", ""))
            bridge_donor = bl_data.get("donor_atoms", ["O"])[0]
            bridge_charge_per = bl_data.get("charge", -1)
        except KeyError:
            bridge_donor      = "O"
            bridge_charge_per = -1
        bridge_bl = get_bond_length(metal, ox, bridge_donor, geom_canon)
    else:
        bridge_bl         = 2.5
        bridge_donor      = "O"
        bridge_charge_per = 0

    # M-M distance
    mxm_angle = 105.0 if "OH" in (bridge or "") else 120.0
    d_mm = max(2.0 * bridge_bl * np.sin(np.radians(mxm_angle / 2)), 2.8)

    # Metal positions
    if arrangement == "linear":
        m_positions = [
            np.array([-d_mm, 0., 0.]),
            np.zeros(3),
            np.array([ d_mm, 0., 0.]),
        ]
    else:  # triangular
        r = d_mm / np.sqrt(3)
        m_positions = [
            np.array([r,  0.,             0.]),
            np.array([-r/2,  r*np.sqrt(3)/2, 0.]),
            np.array([-r/2, -r*np.sqrt(3)/2, 0.]),
        ]

    # Build each monomer independently then assemble
    cn_term  = sum(_expand_ligand(l, metal, ox, geom_canon)["vectors_count"]
                   for l in terminal)
    cn_total = cn_term + 2  # 2 bridge sites per metal in trimer
    geom_use = resolve_geometry(geometry) if geometry else infer_geometry(cn_total)

    all_atoms: List[Atom] = []
    metal_indices         = []
    total_charge          = 0

    for k, m_pos in enumerate(m_positions):
        # Place metal
        metal_indices.append(len(all_atoms))
        all_atoms.append(Atom(symbol=metal, position=m_pos.copy(),
                              label=f"{metal}{k+1}"))
        total_charge += ox

        # Terminal ligands
        mono = _build_single(metal, ox, terminal, geom_use, None,
                             metal_pos=m_pos)
        # skip the metal atom (index 0), add the rest
        for a in mono.atoms[1:]:
            all_atoms.append(Atom(symbol=a.symbol, position=a.position,
                                  label=f"{a.symbol}{len(all_atoms)}"))
        total_charge += mono.charge - ox

    # Bridge ligands between adjacent pairs
    pairs = [(0, 1), (1, 2)] if arrangement == "linear" else [(0,1),(1,2),(2,0)]
    n_bridges_total = len(pairs)
    bridge_atoms_added = 0
    chosen_dirs_trimer: List[np.ndarray] = []

    for pair_idx, (i, j) in enumerate(pairs):
        if not bridge:
            continue
        mi = m_positions[i]
        mj = m_positions[j]
        mm_vec  = mj - mi
        d_mm    = np.linalg.norm(mm_vec)
        m_m_vec = mm_vec / d_mm
        anchor  = (mi + mj) / 2.0

        perp_dir = _best_bridge_direction(
            m_m_vec, anchor, all_atoms, metal,
            n_total=n_bridges_total,
            bridge_idx=pair_idx,
            chosen_dirs=chosen_dirs_trimer,
            extra_anchors=[mi, mj],
        )
        chosen_dirs_trimer.append(perp_dir)

        base_name = bridge.replace("mu-", "")
        if base_name == "OH":
            pairs_atoms = _place_oh_bridge(mi, mj, m_m_vec, d_mm,
                                           bridge_bl, perp_dir)
        elif base_name in ("HCOO", "formate"):
            pairs_atoms = _place_hcoo_bridge(mi, mj, m_m_vec, d_mm,
                                             bridge_bl, perp_dir)
        else:
            pairs_atoms = _place_generic_bridge(mi, mj, bridge_bl, bridge_donor)

        _append_bridge_atoms(all_atoms, pairs_atoms)
        total_charge += bridge_charge_per
        bridge_atoms_added += 1

    mol = Molecule(
        atoms=all_atoms,
        metal_indices=metal_indices,
        formula=_make_formula([a.symbol for a in all_atoms]),
        charge=total_charge,
        spin_multiplicity=_spin_multiplicity(metal, ox),
        geometry=geom_use,
        metal_symbol=metal,
        metal_ox=ox,
        ligand_names=list(terminal) + ([bridge] * bridge_atoms_added if bridge else []),
    )
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
    label = getattr(mol, "label", "")
    label_str = f"  [{label}]" if label and label != "only" else ""
    print(f"Formula          : {mol.formula}{label_str}")
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
    warns = check_structure(mol)
    if warns:
        print(f"\n⚠ {len(warns)} geometry warning(s):")
        for w in warns:
            print(f"  {w}")
