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
from typing import List, Optional, Dict, Tuple
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
from molbuilder.core.validation import validate, ValidationResult


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

def _best_bidentate_perp(v_bisect: np.ndarray,
                         already_placed: List[np.ndarray],
                         bite_half_rad: float) -> np.ndarray:
    """
    Find the perpendicular (fan-plane) direction for a bidentate chelate that
    maximises the minimum distance from already-placed donor atoms.

    The two donor positions are:
        O1 = rot(perp,  +bite_half) @ v_bisect * bl
        O2 = rot(perp,  -bite_half) @ v_bisect * bl

    We sweep *perp* through 360° in the plane perpendicular to v_bisect and
    return the direction that keeps O1 and O2 furthest from every position in
    *already_placed*.  If *already_placed* is empty we return an arbitrary perp.
    """
    # Build orthonormal basis in the plane perpendicular to v_bisect
    z    = np.array([0., 0., 1.])
    ref  = z if abs(np.dot(v_bisect, z)) < 0.9 else np.array([1., 0., 0.])
    p0   = np.cross(v_bisect, ref);  p0 /= np.linalg.norm(p0)
    p1   = np.cross(v_bisect, p0);   p1 /= np.linalg.norm(p1)

    if not already_placed:
        return p0

    def _donors(angle):
        perp = np.cos(angle) * p0 + np.sin(angle) * p1
        v1   = _rot_around_axis(perp,  bite_half_rad) @ v_bisect
        v2   = _rot_around_axis(perp, -bite_half_rad) @ v_bisect
        return v1, v2   # unit vectors, caller scales by bl

    best_score = -1.0
    best_angle = 0.0
    # Coarse sweep
    for deg in range(0, 360, 10):
        rad = np.radians(float(deg))
        v1, v2 = _donors(rad)
        min_d  = min(
            min(float(np.linalg.norm(v1 - (p / np.linalg.norm(p + 1e-12))))
                for p in already_placed),
            min(float(np.linalg.norm(v2 - (p / np.linalg.norm(p + 1e-12))))
                for p in already_placed),
        )
        if min_d > best_score:
            best_score = min_d
            best_angle = rad
    # Fine sweep ±15° around coarse best
    for deg in range(-15, 16, 1):
        rad = best_angle + np.radians(float(deg))
        v1, v2 = _donors(rad)
        min_d  = min(
            min(float(np.linalg.norm(v1 - (p / np.linalg.norm(p + 1e-12))))
                for p in already_placed),
            min(float(np.linalg.norm(v2 - (p / np.linalg.norm(p + 1e-12))))
                for p in already_placed),
        )
        if min_d > best_score:
            best_score = min_d
            best_angle = rad
    return np.cos(best_angle) * p0 + np.sin(best_angle) * p1


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

    # ── Sort ligands: bidentate before monodentate ───────────────────────────
    # Placing bidentate chelates first ensures they claim the 'best' adjacent
    # vector pairs in geometries like TBP and OCT rather than being assigned
    # leftover (possibly axial-only) vectors that cause fan-plane clashes.
    # Within each denticity group the original ordering is preserved.
    expanded = (sorted([e for e in expanded if e["denticity"] >= 2],
                       key=lambda e: -e["denticity"])
                + [e for e in expanded if e["denticity"] == 1])

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

            # Choose fan-plane direction to maximise clearance from already-placed atoms.
            # Future ligand positions (remaining geometry vectors) are scaled to 2.7 Å
            # (larger than M-donor bl ≈ 2.0 Å) to account for H atoms that will extend
            # ~0.96 Å beyond the donor, so the bidentate fans into a genuinely open quadrant.
            _H_REACH = 2.7   # conservative: M-O bl + O-H bl ≈ 2.06 + 0.96
            already_dirs = (
                [a.position - metal_pos for a in mol_atoms if a.symbol != metal]
                + [vecs[(vec_idx + k) % len(vecs)] * _H_REACH
                   for k in range(len(vecs) - vec_idx)]
            )
            perp = _best_bidentate_perp(v_bisect, already_dirs, half)

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

    # ── Post-placement H torsion re-optimisation ──────────────────────────────
    # For each H-bearing ligand, rotate the H atoms around the M-donor axis to
    # maximise the minimum distance to all neighbouring heavy atoms.
    # This handles cases where the initial torsion produced by place_ligand() still
    # leaves an H clashing because a bidentate O was placed in the same quadrant
    # AFTER the monodentate was tentatively positioned.
    _TORSION_STEPS = 72   # 5° resolution
    _MIN_H_D       = 1.77  # hard O-H minimum from validation
    heavy_pos = [a.position for a in mol_atoms if a.symbol not in ("H",) and a.symbol != metal]

    # Build a map: for each H atom, which donor atom is its immediate neighbour?
    for idx, atom in enumerate(mol_atoms):
        if atom.symbol != "H":
            continue
        # Find the nearest non-H, non-metal atom (= the donor it belongs to)
        nearest_donor = None
        nearest_d     = 9999.
        for a2 in mol_atoms:
            if a2.symbol in ("H", metal) or a2 is atom:
                continue
            d = float(np.linalg.norm(atom.position - a2.position))
            if d < nearest_d:
                nearest_d = d; nearest_donor = a2
        if nearest_donor is None:
            continue

        # Find ALL H atoms bonded to the same donor
        donor_pos = nearest_donor.position
        h_group   = [i for i, a in enumerate(mol_atoms)
                     if a.symbol == "H"
                     and float(np.linalg.norm(a.position - donor_pos)) < 1.15]
        if not h_group or idx != h_group[0]:
            continue   # only process once per donor (when we hit the first H)

        # Rotation axis: metal → donor
        axis = donor_pos - metal_pos
        n    = np.linalg.norm(axis)
        if n < 1e-6:
            continue
        axis /= n

        # Collect current H vectors relative to donor
        h_vecs_rel = [mol_atoms[hi].position - donor_pos for hi in h_group]

        # Obstacles: all heavy atoms that are NOT bonded to this donor and are close
        obstacles  = [a.position for a in mol_atoms
                      if a.symbol not in ("H",) and a.symbol != metal
                      and a is not nearest_donor
                      and float(np.linalg.norm(a.position - donor_pos)) > 1.6]

        if not obstacles:
            continue

        # Sweep torsion angles and find the best one
        best_min_d = -1.
        best_R     = np.eye(3)
        for step in range(_TORSION_STEPS):
            angle = 2.0 * np.pi * step / _TORSION_STEPS
            c, s  = np.cos(angle), np.sin(angle)
            ux, uy, uz = axis
            R = np.array([
                [c + ux*ux*(1-c),   ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),   uy*uz*(1-c) - ux*s],
                [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)],
            ])
            rotated = [donor_pos + R @ hv for hv in h_vecs_rel]
            min_d   = min(
                float(np.linalg.norm(rh - ob))
                for rh in rotated for ob in obstacles
            )
            if min_d > best_min_d:
                best_min_d = min_d; best_R = R

        # Only update if the best torsion is better than the current one
        current_min_d = min(
            float(np.linalg.norm(mol_atoms[hi].position - ob))
            for hi in h_group for ob in obstacles
        )
        if best_min_d > current_min_d + 0.01:
            for hi, hv in zip(h_group, h_vecs_rel):
                mol_atoms[hi].position = donor_pos + best_R @ hv

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
                           extra_anchors: Optional[List[np.ndarray]] = None,
                           blocked_positions: Optional[List[np.ndarray]] = None,
                           phi_offset_deg: float = 0.0) -> np.ndarray:
    """Pick the perpendicular-to-mm_hat direction with the most spatial clearance.

    Parameters
    ----------
    phi_offset_deg : rotate the uniform-fallback starting angle by this many
                     degrees.  Use pair_idx * (180/nbpp) to interleave bridges
                     from successive pairs at their shared metal centre.
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
        # Also add blocked_positions as occupied directions
        if blocked_positions:
            for bp in blocked_positions:
                v = bp - anc
                d = np.linalg.norm(v)
                if d > 0.1:
                    dirs.append(v / d)
        occupied_per_anchor.append(dirs)

    min_gap_deg = 360.0 / max(n_total, 1) * 1.0  # exact uniform spacing prevents O-O overlap

    phi_offset = np.radians(phi_offset_deg)
    best_phi   = 2.0 * np.pi * bridge_idx / max(n_total, 1) + phi_offset
    best_score = -1.0

    for k in range(n_samples):
        phi  = 2.0 * np.pi * k / n_samples
        cand = np.cos(phi) * perp1 + np.sin(phi) * perp2

        clearance = 180.0
        for occ in occupied_per_anchor:
            if occ:
                dots   = [max(-1.0, min(1.0, float(np.dot(cand, o)))) for o in occ]
                angles = [np.degrees(np.arccos(d)) for d in dots]
                clearance = min(clearance, min(angles))

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
          mm_distance: Optional[float] = None,
          terminal_m1: Optional[List[str]] = None,
          terminal_m2: Optional[List[str]] = None) -> Molecule:
    """
    Build a dinuclear complex.

    The two metal centres are placed symmetrically along the x-axis.
    Each bridge ligand spans the two metals using a realistic M-X-M angle.
    Terminal ligands fill the remaining coordination sites.
    No metal-metal bond is present unless mm_bond=True.

    Parameters
    ----------
    terminal : list, optional
        Terminal ligands applied symmetrically to *both* metal centres.
        Ignored if terminal_m1 / terminal_m2 are supplied.
    terminal_m1, terminal_m2 : list, optional
        Independent terminal ligand lists for metal 1 (−x) and metal 2 (+x).
        Use these to build heteroleptic dimers where the two centres differ,
        e.g. terminal_m1=["H2O"], terminal_m2=[] for Ni2(mu-HCOO)4(H2O) where
        only one Ni carries the water ligand.
        When supplied, *terminal* is ignored.
    """
    # Resolve per-metal terminal lists
    if terminal_m1 is not None or terminal_m2 is not None:
        t_m1 = list(terminal_m1) if terminal_m1 is not None else []
        t_m2 = list(terminal_m2) if terminal_m2 is not None else []
    else:
        t_m1 = list(terminal) if terminal is not None else []
        t_m2 = list(t_m1)   # symmetric: same ligands on both metals
    # Keep the old `terminal` name pointing at m1 list for backward-compat
    # internal references (CN inference, ligand_names on Molecule).
    terminal = t_m1

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

    # CN inference uses the larger of the two terminal sets so geometry vectors
    # are not under-provisioned for whichever metal has more ligands.
    cn_t1    = sum(_expand_ligand(l, metal, ox, geom_canon)["vectors_count"]
                   for l in t_m1)
    cn_t2    = sum(_expand_ligand(l, metal, ox, geom_canon)["vectors_count"]
                   for l in t_m2)
    cn_total = max(cn_t1, cn_t2) + n
    if cn_total == 0:
        cn_total = 6
    geom_canon = resolve_geometry(geometry) if geometry else infer_geometry(cn_total)
    vecs = get_geometry_vectors(geom_canon)

    # Since bridges are placed first (their positions come from clearance sweep,
    # not the geometry vectors), we give add_terminal ALL geometry vectors and let
    # the clearance sweep pick the best ones.  We always use octahedral vectors (6
    # directions) for the candidate pool — lower-CN geometries like sqpy only have
    # 5 vectors which can leave terminals with no clean site when bridges occupy 2.
    _oct_vecs = get_geometry_vectors("oct")
    terminal_vecs_m1 = list(_oct_vecs)
    terminal_vecs_m2 = [np.array([-v[0], v[1], v[2]]) for v in _oct_vecs]

    # ── build atom list ───────────────────────────────────────────────────────
    all_atoms:    List[Atom] = []
    total_charge: int = 0

    def add_metal(pos, label):
        all_atoms.append(Atom(symbol=metal, position=pos.copy(), label=label))

    # ── Build order: metals → bridges → terminals ────────────────────────────
    # Bridges are placed first because their positions are determined purely by
    # M-M geometry (not by terminal positions). Terminals then use the clearance
    # sweep to find the remaining open sites, naturally avoiding bridge atoms.

    def add_terminal(metal_pos, vecs_local, terminal_list):
        """Place terminal ligands using clearance-sorted vectors.

        Tries geometry vectors in order of decreasing clearance from all
        already-placed atoms (which now includes bridges). For each ligand,
        tries the least-blocked vector and checks that all ligand atoms avoid
        hard clashes with already-placed atoms.
        """
        from molbuilder.core.validation import _min_nonbonded_error, _is_ligand_bond

        # Collect non-metal atoms already placed (includes bridges placed before us)
        placed_ref = [a for a in all_atoms if a.symbol != metal]

        def _clearance_t(v: np.ndarray) -> float:
            v_n = v / (np.linalg.norm(v) + 1e-12)
            if not placed_ref:
                return 180.0
            return min(
                np.degrees(np.arccos(max(-1.0, min(1.0,
                    float(np.dot(v_n, (a.position - metal_pos) /
                                 max(np.linalg.norm(a.position - metal_pos), 1e-6)))
                ))))
                for a in placed_ref
                if np.linalg.norm(a.position - metal_pos) < 5.0
            ) if any(np.linalg.norm(a.position - metal_pos) < 5.0 for a in placed_ref) else 180.0

        def _clashes(candidate: list) -> bool:
            for sym, pos in candidate:
                pos = np.asarray(pos)
                for a in placed_ref:
                    if np.linalg.norm(a.position - metal_pos) > 5.0:
                        continue
                    d = float(np.linalg.norm(pos - a.position))
                    if _is_ligand_bond(sym, a.symbol, d):
                        continue
                    if d < _min_nonbonded_error(sym, a.symbol):
                        return True
            return False

        vec_pool = sorted(vecs_local, key=_clearance_t, reverse=True)
        charge = 0

        for lig_name in terminal_list:
            e  = _expand_ligand(lig_name, metal, ox, geom_canon)
            bl = get_bond_length(metal, ox, e["donor_atom"], geom_canon)

            placed_ok    = None
            used_vec_idx = None

            if e["denticity"] == 1:
                for vi, v in enumerate(vec_pool):
                    donor_abs = metal_pos + v * bl
                    adj       = [metal_pos + vec_pool[jj] * bl
                                 for jj in range(len(vec_pool)) if jj != vi]
                    candidate = place_ligand(lig_name, donor_abs, metal_pos, adj)
                    if not _clashes(candidate):
                        placed_ok = candidate; used_vec_idx = vi; break
                if placed_ok is None:
                    # All vectors clash — use the least-bad (already sorted best-first)
                    v         = vec_pool[0]; used_vec_idx = 0
                    donor_abs = metal_pos + v * bl
                    adj       = [metal_pos + vec_pool[jj] * bl
                                 for jj in range(len(vec_pool)) if jj != 0]
                    placed_ok = place_ligand(lig_name, donor_abs, metal_pos, adj)

            elif e["denticity"] == 2:
                d1 = e["donor_atoms"][0]
                d2 = e["donor_atoms"][1] if len(e["donor_atoms"]) > 1 else d1
                bl1 = get_bond_length(metal, ox, d1, geom_canon)
                bl2 = get_bond_length(metal, ox, d2, geom_canon)
                bite = e.get("bite_angle") or 55.0
                half = np.radians(bite / 2)
                for vi, v in enumerate(vec_pool):
                    # Use clearance-maximising perp direction for each candidate vector
                    _H_REACH = 2.7
                    already_dirs = (
                        [a.position - metal_pos for a in all_atoms if a.symbol != metal]
                        + [vp * (_H_REACH / max(float(np.linalg.norm(vp)), 1e-9))
                           for jj, vp in enumerate(vec_pool) if jj != vi]
                    )
                    perp = _best_bidentate_perp(v, already_dirs, half)
                    v1 = _rot_around_axis(perp,  half) @ v
                    v2 = _rot_around_axis(perp, -half) @ v
                    candidate = place_bidentate_ligand(
                        lig_name, v1, v2, bl1, bl2, metal_pos=metal_pos,
                    )
                    if not _clashes(candidate):
                        placed_ok = candidate; used_vec_idx = vi; break
                if placed_ok is None and vec_pool:
                    v = vec_pool[0]; used_vec_idx = 0
                    _H_REACH = 2.7
                    already_dirs = (
                        [a.position - metal_pos for a in all_atoms if a.symbol != metal]
                        + [vp * (_H_REACH / max(float(np.linalg.norm(vp)), 1e-9))
                           for jj, vp in enumerate(vec_pool) if jj != 0]
                    )
                    perp = _best_bidentate_perp(v, already_dirs, half)
                    v1 = _rot_around_axis(perp,  half) @ v
                    v2 = _rot_around_axis(perp, -half) @ v
                    placed_ok = place_bidentate_ligand(
                        lig_name, v1, v2, bl1, bl2, metal_pos=metal_pos,
                    )

            if placed_ok and used_vec_idx is not None:
                for sym, pos in placed_ok:
                    all_atoms.append(Atom(symbol=sym, position=pos,
                                          label=f"{sym}{len(all_atoms)}"))
                vec_pool.pop(used_vec_idx)
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

    # ── Assemble: metals first, then bridges, then terminals ─────────────────
    # Placing bridges before terminals ensures terminal clearance sweep sees
    # the real bridge positions rather than relying on dry-run predictions.

    # 1. Metals
    add_metal(m1_pos, f"{metal}1")
    add_metal(m2_pos, f"{metal}2")

    # 2. Bridge ligands
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

    # 3. Terminal ligands (now see real bridge atoms via all_atoms)
    tc1 = add_terminal(m1_pos, terminal_vecs_m1, t_m1)
    tc2 = add_terminal(m2_pos, terminal_vecs_m2, t_m2)

    total_charge = 2 * ox + tc1 + tc2 + bridge_charge

    mol = Molecule(
        atoms=all_atoms,
        metal_indices=[0, 1 + sum(
            len(place_ligand(l, np.zeros(3), np.array([2.,0.,0.]), []))
            if not isinstance(l, CustomLigand) and
               (lambda e: e["denticity"] == 1)(_expand_ligand(l, metal, ox, geom_canon))
            else 2
            for l in t_m1
        )],
        formula=_make_formula([a.symbol for a in all_atoms]),
        charge=total_charge,
        spin_multiplicity=_spin_multiplicity(metal, ox),
        geometry=geom_canon,
        metal_symbol=metal,
        metal_ox=ox,
        ligand_names=list(t_m1) + list(t_m2) + ([bridge] * n if bridge else []),
    )

    # ── validation gate ───────────────────────────────────────────────────────
    mol.validation = validate(mol)
    if not mol.validation.passed:
        raise ValueError(
            f"dimer() produced an invalid structure:\n{mol.validation.summary}"
        )

    return mol


# ── trimer() ──────────────────────────────────────────────────────────────────

def trimer(metal: str,
           ox: int,
           terminal: Optional[List[str]] = None,
           bridge: Optional[str] = None,
           arrangement: str = "triangular",
           geometry: Optional[str] = None,
           n_bridges_per_pair: int = 1,
           terminals_per_metal: Optional[List[Optional[List[str]]]] = None) -> Molecule:
    """
    Build a trinuclear complex (linear or triangular).
    Each adjacent pair of metals is connected by n_bridges_per_pair bridge ligands.

    Parameters
    ----------
    terminal : list, optional
        Terminal ligands applied symmetrically to *all* metal centres.
        Ignored when *terminals_per_metal* is supplied.
    terminals_per_metal : list of 3 lists, optional
        Independent terminal ligand lists for each metal centre, e.g.::

            terminals_per_metal=[["H2O"], [], []]

        places one water on metal-0 only.  Pass ``None`` for a site to inherit
        from *terminal*.  This overrides *terminal* for the specified sites and
        is the recommended way to build heteroleptic trimers such as
        Ni3(mu-HCOO)6 with terminal water on selected Ni centres.
    n_bridges_per_pair : int
        Number of bridges per metal-metal pair (default 1).
        Use 2 for e.g. Ni3(mu-HCOO)6 (2 formates per edge).
    """
    if terminal is None:
        terminal = []

    # Resolve per-metal terminal lists
    if terminals_per_metal is not None:
        if len(terminals_per_metal) != 3:
            raise ValueError("terminals_per_metal must have exactly 3 entries (one per metal).")
        _tpm = [
            list(t) if t is not None else list(terminal)
            for t in terminals_per_metal
        ]
    else:
        _tpm = [list(terminal), list(terminal), list(terminal)]

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

    # CN and geometry: use the MAXIMUM per-metal terminal count so that the
    # geometry vectors cover the most-coordinated site.  (Undercoordinated sites
    # will simply use fewer vectors from the same pool.)
    max_term_per_metal = max(
        sum(_expand_ligand(l, metal, ox, geom_canon)["vectors_count"] for l in t)
        for t in _tpm
    ) if any(_tpm) else 0
    cn_term  = max_term_per_metal
    cn_total = cn_term + 2 * n_bridges_per_pair  # bridge sites per metal
    geom_use = resolve_geometry(geometry) if geometry else infer_geometry(cn_total)

    pairs_list = [(0, 1), (1, 2)] if arrangement == "linear" else [(0,1),(1,2),(2,0)]

    # ── Assemble: metals → bridges → terminals (per metal) ───────────────────
    # Bridges are placed first (positions determined by M-M geometry alone).
    # Terminals are placed after, so the clearance sweep sees real bridge atoms.

    all_atoms: List[Atom]  = []
    metal_indices          = []
    total_charge           = 0

    # 1. All metals
    for k, m_pos in enumerate(m_positions):
        metal_indices.append(len(all_atoms))
        all_atoms.append(Atom(symbol=metal, position=m_pos.copy(),
                              label=f"{metal}{k+1}"))
        total_charge += ox

    # 2. All bridges (n_bridges_per_pair bridges per pair)
    bridge_atoms_added    = 0
    n_bridges_total       = len(pairs_list) * n_bridges_per_pair
    chosen_dirs_trimer: List[np.ndarray] = []
    global_bridge_idx     = 0   # monotonic index across all pair×bridge combinations

    # ── Special case: triangular + 2 bridges per pair ─────────────────────────
    # A triangular arrangement has an odd 3-cycle, making it topologically
    # impossible to assign ±z bridge directions without a pair of adjacent +z
    # bridges meeting at the same metal and clashing.
    #
    # Fix: a coordinated ±α tilt scheme where the sign alternates between pairs
    # so every shared metal receives one +α and one −α bridge from each edge.
    # α is derived analytically per bridge type by maximising the minimum
    # inter-edge O···O distance:
    #
    #   Bridge type  │  α (deg)  │  min O···O (Å)  │  geometry note
    #   ─────────────┼───────────┼─────────────────┼─────────────────────────
    #   mu-HCOO      │   35      │   2.20          │  O-C-O bar, target_oo=2.2
    #   mu-OH        │   70      │   2.36          │  single-atom bridge
    #   mu-OAc       │   35      │   2.20          │  same O-C-O geometry
    #   mu-Cl        │   35      │   2.20          │  donor radius similar to O
    #   (default)    │   35      │    —            │  conservative fallback
    #
    # Triangular nbpp=3 is NOT handled here; packing 6 bridging O donors per
    # metal at Ni-Ni ≈ 3.2–3.6 Å gives irreducible O···O ≈ 1.3 Å regardless
    # of tilt angle, so those cases are excluded from MULTI_BRIDGE_CASES.

    _TRIANGULAR_DOUBLE_BRIDGE_ALPHA: dict = {
        "OH":      70.0,   # single-atom bridge; larger α needed to clear the shared Ni
        "HCOO":    35.0,   # O-C-O bar; optimised for target_oo = 2.2 Å
        "formate": 35.0,
        "OAc":     35.0,   # same carboxylate geometry
        "acetate": 35.0,
        "Cl":      35.0,   # similar donor-atom radius
        "Br":      35.0,
        "CN":      35.0,
    }

    _use_triangular_double_bridge = (
        arrangement == "triangular"
        and n_bridges_per_pair == 2
        and bridge is not None
    )

    if _use_triangular_double_bridge and bridge:
        base_name_tdb = bridge.replace("mu-", "")
        _alpha_deg  = _TRIANGULAR_DOUBLE_BRIDGE_ALPHA.get(base_name_tdb, 35.0)
        _TILT_ALPHA = np.radians(_alpha_deg)
        _ref_z      = np.array([0., 0., 1.])
        for pair_idx, (i, j) in enumerate(pairs_list):
            mi = m_positions[i]
            mj = m_positions[j]
            mm_vec  = mj - mi
            d_mm_ij = np.linalg.norm(mm_vec)
            m_m_vec = mm_vec / d_mm_ij

            # Build a consistent perp basis: perp1 in-plane, perp2 ≈ ±z
            _perp1 = np.cross(m_m_vec, _ref_z)
            if np.linalg.norm(_perp1) < 1e-6:
                _perp1 = np.cross(m_m_vec, np.array([1., 0., 0.]))
            _perp1 /= np.linalg.norm(_perp1)
            _perp2  = np.cross(m_m_vec, _perp1)
            _perp2 /= np.linalg.norm(_perp2)

            # Alternating sign: pairs 0,2 → (+α, −α); pair 1 → (−α, +α).
            # Ensures every shared metal receives one +α and one −α O per edge.
            bridge_signs = [+1, -1] if pair_idx % 2 == 0 else [-1, +1]

            base_name = bridge.replace("mu-", "")
            for b_idx, sign in enumerate(bridge_signs):
                alpha     = sign * _TILT_ALPHA
                perp_dir  = np.cos(alpha) * _perp1 + np.sin(alpha) * _perp2
                perp_dir /= np.linalg.norm(perp_dir)

                if base_name == "OH":
                    pairs_atoms = _place_oh_bridge(mi, mj, m_m_vec, d_mm_ij,
                                                   bridge_bl, perp_dir)
                elif base_name in ("HCOO", "formate"):
                    pairs_atoms = _place_hcoo_bridge(mi, mj, m_m_vec, d_mm_ij,
                                                     bridge_bl, perp_dir)
                else:
                    pairs_atoms = _place_generic_bridge(mi, mj, bridge_bl, bridge_donor)

                _append_bridge_atoms(all_atoms, pairs_atoms)
                chosen_dirs_trimer.append(perp_dir)
                total_charge    += bridge_charge_per
                bridge_atoms_added += 1

    else:
        for pair_idx, (i, j) in enumerate(pairs_list):
            if not bridge:
                continue
            mi = m_positions[i]
            mj = m_positions[j]
            mm_vec  = mj - mi
            d_mm    = np.linalg.norm(mm_vec)
            m_m_vec = mm_vec / d_mm
            anchor  = (mi + mj) / 2.0

            for b_idx in range(n_bridges_per_pair):
                # For n_bridges_per_pair > 1, treat all other metal positions as
                # obstacles so bridges don't direct their C/H atoms toward them.
                blocked = None
                if n_bridges_per_pair > 1:
                    blocked = [m_positions[k] for k in range(len(m_positions))
                               if k != i and k != j]

                # Only enforce gap against bridges on the SAME pair (same perp plane)
                # so that different pairs are free to reuse similar 3D directions.
                same_pair_dirs = chosen_dirs_trimer[pair_idx * n_bridges_per_pair:
                                                    pair_idx * n_bridges_per_pair + b_idx]

                # For pairs sharing the same M-M axis (e.g. both pairs in a linear trimer
                # have mm_hat = +x), offset each pair's starting angle by
                # pair_idx * (180 / n_bridges_per_pair) degrees so that bridges from
                # successive pairs interleave rather than stacking at the shared metal.
                pair_phi_offset_deg = pair_idx * (180.0 / max(n_bridges_per_pair, 1))

                perp_dir = _best_bridge_direction(
                    m_m_vec, anchor, all_atoms, metal,
                    n_total=n_bridges_per_pair,
                    bridge_idx=b_idx,
                    chosen_dirs=same_pair_dirs,
                    extra_anchors=[mi, mj],
                    blocked_positions=blocked,
                    phi_offset_deg=pair_phi_offset_deg,
                )
                chosen_dirs_trimer.append(perp_dir)
                global_bridge_idx += 1

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

    # 3. Terminal ligands for each metal (now all_atoms has real bridge positions)
    if any(_tpm):
        from molbuilder.core.validation import _min_nonbonded_error, _is_ligand_bond

        all_vecs = get_geometry_vectors(geom_use)

        for k, m_pos in enumerate(m_positions):
            terminal_k = _tpm[k]
            if not terminal_k:
                continue
            placed_ref_t = [a for a in all_atoms if a.symbol != metal]

            def _cl_t(v: np.ndarray, _m=m_pos, _ref=placed_ref_t) -> float:
                v_n = v / (np.linalg.norm(v) + 1e-12)
                nearby = [a for a in _ref if np.linalg.norm(a.position - _m) < 5.0]
                if not nearby:
                    return 180.0
                return min(
                    np.degrees(np.arccos(max(-1.0, min(1.0,
                        float(np.dot(v_n, (a.position - _m) /
                                     max(np.linalg.norm(a.position - _m), 1e-6)))
                    ))))
                    for a in nearby
                )

            def _clashes_t(candidate: list, _m=m_pos, _ref=placed_ref_t) -> bool:
                nearby = [a for a in _ref if np.linalg.norm(a.position - _m) < 5.0]
                for sym, pos in candidate:
                    pos = np.asarray(pos)
                    for a in nearby:
                        d = float(np.linalg.norm(pos - a.position))
                        if _is_ligand_bond(sym, a.symbol, d):
                            continue
                        if d < _min_nonbonded_error(sym, a.symbol):
                            return True
                return False

            vec_pool = sorted(all_vecs, key=_cl_t, reverse=True)
            charge_k = 0

            for lig_name in terminal_k:
                e  = _expand_ligand(lig_name, metal, ox, geom_use)
                bl = get_bond_length(metal, ox, e["donor_atom"], geom_use)
                placed_ok = None; used_vi = None

                if e["denticity"] == 1:
                    for vi, v in enumerate(vec_pool):
                        v_n       = v / np.linalg.norm(v)
                        donor_abs = m_pos + v_n * bl
                        adj       = [m_pos + vec_pool[jj] / np.linalg.norm(vec_pool[jj]) * bl
                                     for jj in range(len(vec_pool)) if jj != vi]
                        cand = place_ligand(lig_name, donor_abs, m_pos, adj)
                        if not _clashes_t(cand):
                            placed_ok = cand; used_vi = vi; break
                    if placed_ok is None:
                        v = vec_pool[0]; used_vi = 0
                        donor_abs = m_pos + v / np.linalg.norm(v) * bl
                        adj       = [m_pos + vec_pool[jj] / np.linalg.norm(vec_pool[jj]) * bl
                                     for jj in range(len(vec_pool)) if jj != 0]
                        placed_ok = place_ligand(lig_name, donor_abs, m_pos, adj)

                elif e["denticity"] == 2:
                    d1 = e["donor_atoms"][0]
                    d2 = e["donor_atoms"][1] if len(e["donor_atoms"]) > 1 else d1
                    bl1 = get_bond_length(metal, ox, d1, geom_use)
                    bl2 = get_bond_length(metal, ox, d2, geom_use)
                    bite = e.get("bite_angle") or 55.0
                    half = np.radians(bite / 2)
                    for vi, v in enumerate(vec_pool):
                        _H_REACH = 2.7
                        already_dirs = (
                            [a.position - m_pos for a in all_atoms if a.symbol != metal]
                            + [vp * (_H_REACH / max(float(np.linalg.norm(vp)), 1e-9))
                               for jj, vp in enumerate(vec_pool) if jj != vi]
                        )
                        perp_v = _best_bidentate_perp(v / np.linalg.norm(v),
                                                      already_dirs, half)
                        vn = v / np.linalg.norm(v)
                        v1 = _rot_around_axis(perp_v,  half) @ vn
                        v2 = _rot_around_axis(perp_v, -half) @ vn
                        cand = place_bidentate_ligand(
                            lig_name, v1, v2, bl1, bl2, metal_pos=m_pos,
                        )
                        if not _clashes_t(cand):
                            placed_ok = cand; used_vi = vi; break
                    if placed_ok is None and vec_pool:
                        v = vec_pool[0]; used_vi = 0
                        _H_REACH = 2.7
                        already_dirs = (
                            [a.position - m_pos for a in all_atoms if a.symbol != metal]
                            + [vp * (_H_REACH / max(float(np.linalg.norm(vp)), 1e-9))
                               for jj, vp in enumerate(vec_pool) if jj != 0]
                        )
                        perp_v = _best_bidentate_perp(v / np.linalg.norm(v),
                                                      already_dirs, half)
                        vn = v / np.linalg.norm(v)
                        v1 = _rot_around_axis(perp_v,  half) @ vn
                        v2 = _rot_around_axis(perp_v, -half) @ vn
                        placed_ok = place_bidentate_ligand(
                            lig_name, v1, v2, bl1, bl2, metal_pos=m_pos,
                        )

                if placed_ok and used_vi is not None:
                    for sym, pos in placed_ok:
                        all_atoms.append(Atom(symbol=sym, position=pos,
                                              label=f"{sym}{len(all_atoms)}"))
                    vec_pool.pop(used_vi)
                charge_k += e["charge"]

            total_charge += charge_k

    # ── Post-placement H torsion re-optimisation (trimer terminals) ───────────
    # Same logic as in _build_single: rotate H-bearing terminal ligands around
    # the M-donor axis to maximise clearance from bridge O atoms and other donors.
    if any(_tpm):
        _TORSION_STEPS_T = 72
        for idx, atom in enumerate(all_atoms):
            if atom.symbol != "H":
                continue
            nearest_donor = None
            nearest_d     = 9999.
            for a2 in all_atoms:
                if a2.symbol in ("H", metal) or a2 is atom:
                    continue
                d = float(np.linalg.norm(atom.position - a2.position))
                if d < nearest_d:
                    nearest_d = d; nearest_donor = a2
            if nearest_donor is None or nearest_d > 1.15:
                continue
            # Find the metal this donor is bonded to
            donor_pos = nearest_donor.position
            closest_m = min(
                (a for a in all_atoms if a.symbol == metal),
                key=lambda a: float(np.linalg.norm(a.position - donor_pos)),
                default=None,
            )
            if closest_m is None:
                continue
            m_pos_t = closest_m.position
            h_group = [i for i, a in enumerate(all_atoms)
                       if a.symbol == "H"
                       and float(np.linalg.norm(a.position - donor_pos)) < 1.15]
            if not h_group or idx != h_group[0]:
                continue
            axis = donor_pos - m_pos_t
            n    = np.linalg.norm(axis)
            if n < 1e-6:
                continue
            axis /= n
            h_vecs_rel = [all_atoms[hi].position - donor_pos for hi in h_group]
            obstacles  = [a.position for a in all_atoms
                          if a.symbol not in ("H",) and a.symbol != metal
                          and a is not nearest_donor
                          and float(np.linalg.norm(a.position - donor_pos)) > 1.6]
            if not obstacles:
                continue
            best_min_d = -1.; best_R = np.eye(3)
            for step in range(_TORSION_STEPS_T):
                angle = 2.0 * np.pi * step / _TORSION_STEPS_T
                c, s  = np.cos(angle), np.sin(angle)
                ux, uy, uz = axis
                R = np.array([
                    [c + ux*ux*(1-c),   ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
                    [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),   uy*uz*(1-c) - ux*s],
                    [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s, c + uz*uz*(1-c)],
                ])
                rotated = [donor_pos + R @ hv for hv in h_vecs_rel]
                min_d   = min(float(np.linalg.norm(rh - ob))
                              for rh in rotated for ob in obstacles)
                if min_d > best_min_d:
                    best_min_d = min_d; best_R = R
            current_min_d = min(float(np.linalg.norm(all_atoms[hi].position - ob))
                                for hi in h_group for ob in obstacles)
            if best_min_d > current_min_d + 0.01:
                for hi, hv in zip(h_group, h_vecs_rel):
                    all_atoms[hi].position = donor_pos + best_R @ hv

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

    # ── validation gate ───────────────────────────────────────────────────────
    mol.validation = validate(mol)
    if not mol.validation.passed:
        raise ValueError(
            f"trimer() produced an invalid structure:\n{mol.validation.summary}"
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
