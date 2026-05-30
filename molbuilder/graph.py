"""
graph.py  — Molecular graph and canonical hashing (beta)
=========================================================

Provides a graph-based canonical hash for molbuilder Molecule objects
that distinguishes:

    •  Different geometries   (tet ≠ sqp,  sqpy ≠ tbp)
    •  Different isomers      (cis ≠ trans in sqp / oct)
    •  Different ligand sets  (H2O ≠ OH)
    •  Different nuclearities (monomer ≠ dimer)

Two structures get the **same hash** only when they have identical:
    1.  Atom connectivity (the covalent bond graph)
    2.  Donor-donor angular relationships on every metal centre
    3.  Declared coordination geometry

This means two POSCAR files that are pure rotations / reflections of each
other (arising from different enumeration paths) will hash identically and
can be deduplicated.

Algorithm
---------
We build a *labelled multigraph* with two edge types:

    cov   — covalent bond (inferred from pairwise distance vs sum of radii)
    ang   — donor–donor spatial relationship on the same metal:
                "adj"    60–120° apart  (cis in octahedral)
                "trans"  >120° apart    (trans in octahedral)
                "close"  <60° apart     (equatorial-equatorial in pbp etc.)

Four rounds of Weisfeiler–Lehman relabelling then produce a canonical label
per atom, and the sorted multiset of those labels — together with the geometry
name — is the hash.

The geometry name is included explicitly so that
    Ni(HCOO)2(H2O)2 tetrahedral  ≠  Ni(HCOO)2(H2O)2 square-planar
even though their covalent bond graphs are identical.

Public API
----------
    canonical_hash(mol)                     → tuple
    MolGraph(mol)                           → graph object with .hash attribute
    DeduplicationResult(original, unique, groups) → named result from deduplicate()
    deduplicate(mols_and_rows, verbose)     → DeduplicationResult

The module is intentionally standalone — it imports only from the core
Molecule class and numpy.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from molbuilder.core.molecule import Molecule

# ── covalent radii (pm → Å) used for bond inference ──────────────────────────
# Pyykkö & Atsumi (2009); metals use single-bond values
_COVALENT_RADII: Dict[str, float] = {
    "H":  0.31, "C":  0.76, "N":  0.71, "O":  0.66, "F":  0.57,
    "S":  1.05, "P":  1.07, "Cl": 1.02, "Br": 1.20, "I":  1.39,
    "Ni": 1.24, "Fe": 1.32, "Co": 1.26, "Cu": 1.32, "Pd": 1.20,
    "Pt": 1.28, "Rh": 1.34, "Ru": 1.36, "Re": 1.51, "Mo": 1.54,
    "Cr": 1.39, "Mn": 1.61, "Zn": 1.22, "Ag": 1.45,
}
_COVALENT_BOND_TOL  = 1.30   # multiply sum-of-radii by this to get max bond length
_METAL_DONOR_CUTOFF = 2.90   # Å — M-L bonds identified up to this distance


def _infer_bonds(atoms) -> List[Tuple[int, int]]:
    """Return (i, j) pairs for inferred covalent bonds (i < j)."""
    pos  = np.array([a.position for a in atoms])
    syms = [a.symbol for a in atoms]
    bonds = []
    n = len(atoms)
    for i in range(n):
        ri = _COVALENT_RADII.get(syms[i], 1.00)
        for j in range(i + 1, n):
            rj  = _COVALENT_RADII.get(syms[j], 1.00)
            cut = (ri + rj) * _COVALENT_BOND_TOL
            if float(np.linalg.norm(pos[i] - pos[j])) < cut:
                bonds.append((i, j))
    return bonds


def _donor_angle_label(angle_deg: float) -> str:
    """Bin a donor–donor angle into a coarse spatial label."""
    if angle_deg < 60.0:
        return "close"      # e.g. equatorial pairs in pbp
    if angle_deg < 120.0:
        return "adj"        # cis in sqp / oct
    return "trans"          # trans in sqp / oct


# ── MolGraph ─────────────────────────────────────────────────────────────────

class MolGraph:
    """
    Labelled multigraph representation of a Molecule.

    Attributes
    ----------
    mol        : source Molecule
    hash       : canonical tuple (use == for comparison)
    n_atoms    : number of atoms
    bonds      : list of (i, j) covalent bond pairs
    ang_edges  : list of (i, j, label) donor–donor angular edges

    The graph is built once in __init__; the hash is computed lazily.
    """

    def __init__(self, mol: Molecule) -> None:
        self.mol       = mol
        self.n_atoms   = len(mol.atoms)
        self.bonds     = _infer_bonds(mol.atoms)
        self.ang_edges = self._build_angular_edges()
        self._hash: Optional[tuple] = None

    # ── graph construction ────────────────────────────────────────────────────

    def _build_angular_edges(self) -> List[Tuple[int, int, str]]:
        """
        For every metal centre, find all donor atoms within
        _METAL_DONOR_CUTOFF and add donor–donor angular edges.
        """
        atoms   = self.mol.atoms
        metal   = self.mol.metal_symbol
        if not metal:
            return []

        m_indices = [i for i, a in enumerate(atoms) if a.symbol == metal]
        ang_edges = []

        for mi in m_indices:
            m_pos = atoms[mi].position
            # donors: non-metal, non-H atoms within cutoff
            donors = [
                i for i, a in enumerate(atoms)
                if a.symbol not in (metal, "H")
                and float(np.linalg.norm(a.position - m_pos)) < _METAL_DONOR_CUTOFF
            ]
            for k in range(len(donors)):
                for l in range(k + 1, len(donors)):
                    di, dj = donors[k], donors[l]
                    vi = atoms[di].position - m_pos
                    vj = atoms[dj].position - m_pos
                    ni_ = float(np.linalg.norm(vi))
                    nj_ = float(np.linalg.norm(vj))
                    if ni_ < 1e-6 or nj_ < 1e-6:
                        continue
                    cos_a = float(np.dot(vi, vj)) / (ni_ * nj_)
                    cos_a = max(-1.0, min(1.0, cos_a))
                    angle = float(np.degrees(np.arccos(cos_a)))
                    label = _donor_angle_label(angle)
                    ang_edges.append((di, dj, label))

        return ang_edges

    # ── WL hashing ───────────────────────────────────────────────────────────

    @property
    def hash(self) -> tuple:
        if self._hash is None:
            self._hash = self._compute_hash()
        return self._hash

    def _compute_hash(self, wl_iterations: int = 4) -> tuple:
        """
        Four rounds of Weisfeiler–Lehman relabelling over the multigraph.

        The initial node label is the element symbol.  After each round,
        a node's label becomes a canonical string encoding:
            current_label : sorted( labels of cov-neighbours ) :
                            sorted( label+ang_type of ang-neighbours )

        The geometry name is appended as a suffix to every final label so
        that tet and sqp variants of the same formula get different hashes.
        """
        atoms  = self.mol.atoms
        geom   = getattr(self.mol, "geometry", "") or ""

        # adjacency for cov edges
        cov_adj: Dict[int, List[int]] = defaultdict(list)
        for i, j in self.bonds:
            cov_adj[i].append(j)
            cov_adj[j].append(i)

        # adjacency for angular edges
        ang_adj: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
        for di, dj, lbl in self.ang_edges:
            ang_adj[di].append((dj, lbl))
            ang_adj[dj].append((di, lbl))

        labels = [a.symbol for a in atoms]

        for _ in range(wl_iterations):
            new_labels = []
            for i in range(len(atoms)):
                cov_nbr = tuple(sorted(labels[j] for j in cov_adj[i]))
                ang_nbr = tuple(sorted(f"{labels[j]}_{t}" for j, t in ang_adj[i]))
                new_labels.append(f"{labels[i]}|{cov_nbr}|{ang_nbr}")
            labels = new_labels

        # Append geometry to each label so tet ≠ sqp even with same ligands
        final = tuple(sorted(f"{lbl}@{geom}" for lbl in labels))
        return final

    # ── convenience ──────────────────────────────────────────────────────────

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MolGraph):
            return NotImplemented
        return self.hash == other.hash

    def __hash__(self) -> int:
        return hash(self.hash)

    def __repr__(self) -> str:
        return (f"MolGraph({self.mol.formula}, geom={getattr(self.mol,'geometry','?')}, "
                f"n_bonds={len(self.bonds)}, n_ang={len(self.ang_edges)})")


# ── public canonical_hash function ────────────────────────────────────────────

def canonical_hash(mol: Molecule) -> tuple:
    """
    Return the canonical hash tuple for *mol*.

    Two molecules have the same hash if and only if they have:
      - the same covalent bond graph (element-labelled)
      - the same donor–donor angular relationships on every metal
      - the same declared coordination geometry

    This correctly distinguishes cis from trans isomers, tet from sqp
    variants of the same formula, and different ligand combinations.
    Pure rotations / reflections of the same structure hash identically.

    Parameters
    ----------
    mol : Molecule

    Returns
    -------
    tuple of str — use == for comparison; hashable for dict/set keys.
    """
    return MolGraph(mol).hash


# ── DeduplicationResult ───────────────────────────────────────────────────────

@dataclass
class DeduplicationResult:
    """
    Result of a deduplication run.

    Attributes
    ----------
    original   : all (mol, row) pairs passed in
    unique     : subset with one representative per hash group
    groups     : dict mapping canonical_hash → list of (mol, row) pairs
                 groups with len > 1 are the duplicate sets
    n_removed  : number of structures removed
    """
    original : List[Tuple[Molecule, Dict[str, Any]]]
    unique   : List[Tuple[Molecule, Dict[str, Any]]]
    groups   : Dict[tuple, List[Tuple[Molecule, Dict[str, Any]]]]

    @property
    def n_removed(self) -> int:
        return len(self.original) - len(self.unique)

    @property
    def duplicate_groups(self):
        """Groups with more than one member."""
        return {h: mols for h, mols in self.groups.items() if len(mols) > 1}

    def summary(self) -> str:
        dup_count = sum(len(v) - 1 for v in self.groups.values() if len(v) > 1)
        lines = [
            f"Deduplication summary",
            f"  Input    : {len(self.original)} structures",
            f"  Unique   : {len(self.unique)} structures",
            f"  Removed  : {self.n_removed} duplicates",
            f"  Groups   : {len(self.duplicate_groups)} hash groups contain duplicates",
        ]
        if self.duplicate_groups:
            lines.append("  Duplicate groups:")
            for h, mols in list(self.duplicate_groups.items())[:8]:
                rep = mols[0]
                geom = rep[1].get("geometry", "?")
                lines.append(
                    f"    [{len(mols)}×] {rep[0].formula:15s}"
                    f"  geom={geom:5s}  cn={rep[1].get('cn','?')}"
                    f"  {rep[1].get('ligand_combo','?')}"
                )
            if len(self.duplicate_groups) > 8:
                lines.append(f"    ... and {len(self.duplicate_groups)-8} more groups")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"DeduplicationResult("
                f"{len(self.original)} → {len(self.unique)} unique, "
                f"{self.n_removed} removed)")


# ── deduplicate ───────────────────────────────────────────────────────────────

def deduplicate(
    mols_and_rows: List[Tuple[Molecule, Dict[str, Any]]],
    verbose: bool = False,
) -> DeduplicationResult:
    """
    Remove duplicate structures from an enumeration result.

    For each group of structures that share a canonical hash, only the first
    encountered is kept (the representative).  The order of the input list
    determines which representative is chosen — in practice this is the order
    structures are yielded by enumerate_complexes().

    Parameters
    ----------
    mols_and_rows : list of (Molecule, row_dict) as from enumerate_complexes()
    verbose       : print a line for each duplicate found

    Returns
    -------
    DeduplicationResult with .unique, .groups, .n_removed, and .summary()
    """
    seen:   Dict[tuple, int] = {}   # hash → index in unique list
    unique: List[Tuple[Molecule, Dict[str, Any]]] = []
    groups: Dict[tuple, List[Tuple[Molecule, Dict[str, Any]]]] = defaultdict(list)

    for mol, row in mols_and_rows:
        try:
            h = canonical_hash(mol)
        except Exception as exc:
            warnings.warn(
                f"canonical_hash failed for {mol.formula}: {exc}. "
                "Keeping structure (conservative).",
                stacklevel=2,
            )
            h = (mol.formula, id(mol))   # unique fallback

        groups[h].append((mol, row))

        if h not in seen:
            seen[h] = len(unique)
            unique.append((mol, row))
        else:
            if verbose:
                rep = unique[seen[h]]
                print(
                    f"  dup  {mol.formula:15s} "
                    f"geom={row.get('geometry','?'):5s} "
                    f"{row.get('ligand_combo','')[:30]:30s}"
                    f"  ← same as {rep[1].get('geometry','?')} "
                    f"{rep[1].get('ligand_combo','')[:30]}"
                )

    return DeduplicationResult(
        original = mols_and_rows,
        unique   = unique,
        groups   = dict(groups),
    )
