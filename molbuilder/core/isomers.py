"""
isomers.py
==========
Enumerate all symmetry-distinct coordination isomers for a given set of
ligands and geometry.

The algorithm:
  1. Generate all permutations of ligand types across coordination sites.
  2. Apply a canonical form under the point-group symmetry of the geometry
     to de-duplicate symmetry-equivalent arrangements.
  3. Return one representative site-assignment per distinct isomer.

Currently implemented geometries:
  - oct  (Oh)         : 6 sites, 3 trans-axis pairs
  - sqp  (D4h)        : 4 sites, 2 trans-axis pairs
  - tet  (Td)         : 4 sites, all equivalent (no trans pairs)
  - tbp  (D3h)        : 5 sites, axial pair + 3 equatorial
  - sqpy (C4v)        : 5 sites, axial + 4 equatorial pairs
  - lin  (Dinfh)        : 2 sites, 1 trans pair
  - tp   (D3h)        : 3 sites, all equivalent
  - bent (C2v)        : 2 sites, equivalent
  - tpr  (D3h)        : 6 sites, two trigonal faces
  - pbp  (D5h)        : 7 sites, axial pair + 5 equatorial
  - sapr (D4d)        : 8 sites
"""

from __future__ import annotations
from itertools import permutations
from collections import Counter
from typing import List, Dict


# ------------------------------------------------------------------------------
# Symmetry canonical-form functions per geometry
# Each function takes a dict {site_index: ligand_label} and returns a
# hashable canonical form.  Two assignments that are symmetry-equivalent
# produce the same canonical form.
# ------------------------------------------------------------------------------

def _canon_oct(assign: Dict[int, str]) -> tuple:
    """Oh symmetry canonical form for 6-site octahedron.
    Trans pairs: (0,1), (2,3), (4,5).
    All three axes are equivalent; within each axis the two ends are equivalent.
    Canonical: sorted tuple of sorted (a,b) pairs for each trans axis.
    """
    axes = [tuple(sorted([assign[0], assign[1]])),
            tuple(sorted([assign[2], assign[3]])),
            tuple(sorted([assign[4], assign[5]]))]
    return tuple(sorted(axes))


def _canon_sqp(assign: Dict[int, str]) -> tuple:
    """D4h square-planar canonical form for 4 sites.
    Trans pairs: (0,2) and (1,3).  The two axes are equivalent.
    """
    axes = [tuple(sorted([assign[0], assign[2]])),
            tuple(sorted([assign[1], assign[3]]))]
    return tuple(sorted(axes))


def _canon_tet(assign: Dict[int, str]) -> tuple:
    """Td tetrahedral: all 4 sites equivalent (no trans pairs).
    Canonical = sorted multiset of ligand labels.
    """
    return tuple(sorted(assign.values()))


def _canon_tbp(assign: Dict[int, str]) -> tuple:
    """D3h trigonal-bipyramidal: sites 0,1 are axial (trans pair);
    sites 2,3,4 are equatorial (mutually equivalent).
    """
    axial  = tuple(sorted([assign[0], assign[1]]))
    equat  = tuple(sorted([assign[2], assign[3], assign[4]]))
    return (axial, equat)


def _canon_sqpy(assign: Dict[int, str]) -> tuple:
    """C4v square-pyramidal: site 0 is apical; sites 1,2,3,4 are basal.
    The 4 basal sites are equivalent under C4 rotation.
    """
    apical = assign[0]
    basal  = tuple(sorted([assign[1], assign[2], assign[3], assign[4]]))
    return (apical, basal)


def _canon_lin(assign: Dict[int, str]) -> tuple:
    """Dinfh linear: sites 0,1 are trans-equivalent."""
    return (tuple(sorted([assign[0], assign[1]])),)


def _canon_tp(assign: Dict[int, str]) -> tuple:
    """D3h trigonal-planar: all 3 sites equivalent."""
    return tuple(sorted(assign.values()))


def _canon_bent(assign: Dict[int, str]) -> tuple:
    """C2v bent: sites 0,1 equivalent."""
    return tuple(sorted(assign.values()))


def _canon_tshaped(assign: Dict[int, str]) -> tuple:
    """C2v T-shaped: sites 0,1 are trans (the stem), site 2 is the top.
    Sites 0 and 1 are equivalent to each other.
    """
    stem = tuple(sorted([assign[0], assign[1]]))
    top  = assign[2]
    return (stem, top)


def _canon_seesaw(assign: Dict[int, str]) -> tuple:
    """C2v see-saw: sites 0,3 are axial-like (trans), sites 1,2 are equatorial."""
    axial = tuple(sorted([assign[0], assign[3]]))
    equat = tuple(sorted([assign[1], assign[2]]))
    return (axial, equat)


def _canon_tpr(assign: Dict[int, str]) -> tuple:
    """D3h trigonal-prismatic: two triangular faces (0,1,2) and (3,4,5).
    Sites within each face are equivalent; the two faces are equivalent.
    """
    face1 = tuple(sorted([assign[0], assign[1], assign[2]]))
    face2 = tuple(sorted([assign[3], assign[4], assign[5]]))
    return tuple(sorted([face1, face2]))


def _canon_pbp(assign: Dict[int, str]) -> tuple:
    """D5h pentagonal-bipyramidal: sites 0,1 axial; sites 2-6 equatorial."""
    axial  = tuple(sorted([assign[0], assign[1]]))
    equat  = tuple(sorted([assign[i] for i in range(2, 7)]))
    return (axial, equat)


def _canon_sapr(assign: Dict[int, str]) -> tuple:
    """D4d square-antiprismatic: two square faces (0-3) and (4-7)."""
    face1 = tuple(sorted([assign[i] for i in range(4)]))
    face2 = tuple(sorted([assign[i] for i in range(4, 8)]))
    return tuple(sorted([face1, face2]))


_CANON_FUNCS = {
    "oct":      _canon_oct,
    "sqp":      _canon_sqp,
    "tet":      _canon_tet,
    "tbp":      _canon_tbp,
    "sqpy":     _canon_sqpy,
    "lin":      _canon_lin,
    "bent":     _canon_bent,
    "tp":       _canon_tp,
    "tshaped":  _canon_tshaped,
    "seesaw":   _canon_seesaw,
    "tpr":      _canon_tpr,
    "pbp":      _canon_pbp,
    "sapr":     _canon_sapr,
}


# ------------------------------------------------------------------------------
# Isomer names
# ------------------------------------------------------------------------------

# Trans pairs per geometry (used for labelling)
_TRANS_PAIRS = {
    "oct":  [(0, 1), (2, 3), (4, 5)],
    "sqp":  [(0, 2), (1, 3)],
    "tbp":  [(0, 1)],
    "lin":  [(0, 1)],
    "tshaped": [(0, 1)],
    "seesaw":  [(0, 3)],
    "tpr":  [],
    "pbp":  [(0, 1)],
    "sapr": [],
    "tet":  [],
    "tp":   [],
    "sqpy": [],
    "bent": [],
}


def _is_trans(i: int, j: int, geometry: str) -> bool:
    return any(set([i, j]) == set(p) for p in _TRANS_PAIRS.get(geometry, []))


def _label_isomer(site_assign: List[str], geometry: str, isomer_idx: int,
                  total_isomers: int) -> str:
    """Generate a human-readable isomer label."""
    if total_isomers == 1:
        return "only"

    counts = Counter(site_assign)
    n_types = len(counts)

    # Special case: MA3B3 octahedral -> fac / mer
    if geometry == "oct" and n_types == 2:
        vals = list(counts.values())
        if sorted(vals) == [3, 3]:
            # fac: no two of the majority ligand are trans
            majority = max(counts, key=lambda k: counts[k])
            sites = [i for i, s in enumerate(site_assign) if s == majority]
            has_trans = any(_is_trans(sites[a], sites[b], geometry)
                            for a in range(len(sites))
                            for b in range(a + 1, len(sites)))
            return "mer" if has_trans else "fac"
        # MA4B2 or MA2B4 -> cis / trans
        if sorted(vals) == [2, 4]:
            minority = min(counts, key=lambda k: counts[k])
            sites = [i for i, s in enumerate(site_assign) if s == minority]
            return "trans" if _is_trans(sites[0], sites[1], geometry) else "cis"

    # Square planar MA2B2 -> cis / trans
    if geometry == "sqp" and n_types == 2:
        vals = list(counts.values())
        if sorted(vals) == [2, 2]:
            minority = min(counts, key=lambda k: counts[k])
            sites = [i for i, s in enumerate(site_assign) if s == minority]
            return "trans" if _is_trans(sites[0], sites[1], geometry) else "cis"

    # Generic labelling
    return f"isomer-{isomer_idx + 1}"


# ------------------------------------------------------------------------------
# Main enumeration function
# ------------------------------------------------------------------------------

def enumerate_isomers(ligand_labels: List[str], geometry: str) -> List[dict]:
    """
    Return a list of dicts, one per symmetry-distinct isomer.

    Each dict:
        site_assignment : list[str]  - ligand name at each site index
        label           : str        - human-readable name (fac/mer/cis/trans/...)
        index           : int        - 0-based isomer index
    """
    canon_func = _CANON_FUNCS.get(geometry)
    if canon_func is None:
        # Unknown geometry: just return the single natural assignment
        return [{"site_assignment": list(ligand_labels), "label": "only", "index": 0}]

    n_sites = len(ligand_labels)
    seen: set = set()
    isomers: List[dict] = []

    for perm in set(permutations(ligand_labels)):
        assign = {i: perm[i] for i in range(n_sites)}
        canon = canon_func(assign)
        if canon not in seen:
            seen.add(canon)
            isomers.append({"site_assignment": list(perm), "_canon": canon})

    # Sort for determinism: alphabetical by canonical form string
    isomers.sort(key=lambda x: str(x["_canon"]))

    # Add labels
    n = len(isomers)
    for idx, iso in enumerate(isomers):
        iso["label"] = _label_isomer(iso["site_assignment"], geometry, idx, n)
        iso["index"] = idx
        del iso["_canon"]

    return isomers


def has_multiple_isomers(ligand_labels: List[str], geometry: str) -> bool:
    return len(enumerate_isomers(ligand_labels, geometry)) > 1
