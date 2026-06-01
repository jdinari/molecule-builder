"""
validation.py
=============
Geometry validation for molbuilder Molecule objects.

Performs four independent checks, each returning a list of
ValidationIssue objects:

  1. overlap_check       - atom pairs closer than element-pair minimums
  2. coordination_check  - every metal has the expected number of
                           close-contact donor atoms; no donor is
                           suspiciously far from its metal
  3. connectivity_check  - the molecular graph is fully connected
                           (no detached fragments)
  4. geometry_check      - (optional) metal coordination angles
                           deviate excessively from ideal values

Public API
----------
  validate(mol, checks=..., strict=False) -> ValidationResult
  ValidationResult.passed  bool
  ValidationResult.issues  List[ValidationIssue]
  ValidationResult.summary str

  is_valid(mol, **kw)  -> bool   (convenience wrapper)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Set

import numpy as np

from molbuilder.core.bond_lengths import COVALENT_RADII
# TODO: also import get_bond_length when M-L distance sanity check is added
from molbuilder.core.molecule import Molecule

# -- constants ------------------------------------------------------------------

# Van der Waals radii (Angstrom), used for non-bonded overlap thresholds.
# Source: Bondi 1964 + Alvarez 2013 for metals.
_VDW = {
    "H":  1.20, "C":  1.70, "N":  1.55, "O":  1.52, "F":  1.47,
    "P":  1.80, "S":  1.80, "Cl": 1.75, "Br": 1.85, "I":  1.98,
    # Transition metals -- use Alvarez 2013 values
    "Ni": 1.63, "Fe": 1.52, "Co": 1.52, "Cu": 1.40, "Zn": 1.39,
    "Mn": 1.45, "Cr": 1.41, "V":  1.53, "Ti": 1.60, "Sc": 1.70,
    "Pd": 1.63, "Pt": 1.72, "Ru": 1.46, "Rh": 1.42, "Ir": 1.41,
    "Os": 1.44, "Mo": 1.54, "W":  1.62, "Re": 1.51,
    "Au": 1.66, "Ag": 1.72, "Cd": 1.58, "La": 2.07, "Eu": 1.98,
}

def _vdw(sym: str) -> float:
    return _VDW.get(sym, 1.70)

# Intra-ligand bond length ranges (Angstrom) -- used to identify genuine bonds.
# Pairs not listed fall back to sum-of-covalent-radii x tolerance.
_BOND_RANGE: dict[tuple, tuple] = {
    # (sym_a, sym_b): (min_Angstrom, max_Angstrom)
    ("H", "H"):  (0.60, 0.80),
    ("H", "C"):  (0.90, 1.15),
    ("H", "N"):  (0.85, 1.10),
    ("H", "O"):  (0.85, 1.05),
    ("H", "S"):  (1.20, 1.40),
    ("C", "C"):  (1.15, 1.62),
    ("C", "N"):  (1.10, 1.55),
    ("C", "O"):  (1.10, 1.55),
    ("C", "S"):  (1.55, 1.85),
    ("N", "N"):  (1.05, 1.55),
    ("N", "O"):  (1.05, 1.55),
    ("O", "O"):  (1.20, 1.55),   # peroxo
    ("O", "P"):  (1.40, 1.70),
    ("P", "C"):  (1.70, 1.90),
    ("S", "S"):  (1.90, 2.10),
}

def _pair(a: str, b: str) -> tuple:
    return tuple(sorted([a, b]))

def _is_ligand_bond(s1: str, s2: str, d: float) -> bool:
    """True if the distance d is consistent with a covalent bond between s1-s2."""
    key = _pair(s1, s2)
    if key in _BOND_RANGE:
        lo, hi = _BOND_RANGE[key]
        return lo <= d <= hi
    # Fallback: covalent radii sum +/- 25 %
    r_sum = COVALENT_RADII.get(s1, 0.75) + COVALENT_RADII.get(s2, 0.75)
    return d <= r_sum * 1.25

def _min_nonbonded_error(s1: str, s2: str) -> float:
    """Hard error threshold: distance below this is physically impossible (0.65 x vdW sum)."""
    return (_vdw(s1) + _vdw(s2)) * 0.65

def _min_nonbonded_warn(s1: str, s2: str) -> float:
    """Warning threshold: tight pre-DFT contact that will likely relax (0.76 x vdW sum)."""
    return (_vdw(s1) + _vdw(s2)) * 0.76


# -- data types -----------------------------------------------------------------

@dataclass(frozen=True)
class ValidationIssue:
    """One failed check in a ValidationResult."""
    check:    str          # 'overlap' | 'coordination' | 'connectivity' | 'geometry'
    severity: str          # 'error' | 'warning'
    reason:   str          # human-readable description
    atom_i:   int = -1     # 0-based index of first offending atom (-1 = N/A)
    atom_j:   int = -1     # 0-based index of second offending atom (-1 = N/A)
    distance: float = math.nan   # interatomic distance if applicable

    def __str__(self) -> str:
        parts = [f"[{self.severity.upper()}:{self.check}] {self.reason}"]
        if self.atom_i >= 0 and self.atom_j >= 0:
            parts.append(f"  atoms {self.atom_i+1} & {self.atom_j+1}")
        if not math.isnan(self.distance):
            parts.append(f"  d = {self.distance:.3f} Angstrom")
        return "\n".join(parts)


@dataclass
class ValidationResult:
    """Aggregated result of all checks run against one Molecule."""
    issues: List[ValidationIssue] = field(default_factory=list)

    # -- derived properties ----------------------------------------------------
    @property
    def passed(self) -> bool:
        """True only if no *error*-severity issues were found."""
        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def summary(self) -> str:
        if not self.issues:
            return "OK Structure passed all validation checks."
        lines = [
            f"{'OK' if self.passed else 'FAIL'} "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s):"
        ]
        for iss in self.issues:
            lines.append(f"  {iss}")
        return "\n".join(lines)

    def __bool__(self) -> bool:
        return self.passed


# -- helpers --------------------------------------------------------------------

def _metal_symbols(mol: Molecule) -> Set[str]:
    """All distinct metal element symbols in the molecule.

    Uses mol.metal_symbol as the primary source, then cross-checks
    mol.metal_indices to pick up any additional metals.  Excludes common
    ligand atoms (H, C, N, O, F, P, S, Cl, Br, I) regardless of what
    metal_indices points to -- this guards against stale index bookkeeping.
    """
    _LIGAND_ATOMS = {"H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"}
    syms: Set[str] = set()
    if mol.metal_symbol:
        syms.add(mol.metal_symbol)
    for idx in mol.metal_indices:
        if 0 <= idx < len(mol.atoms):
            sym = mol.atoms[idx].symbol
            if sym not in _LIGAND_ATOMS:
                syms.add(sym)
    # Also accept any atom whose element is not a typical ligand atom
    # (belt-and-suspenders for multi-metal structures)
    return syms if syms else {"Ni"}  # last-resort fallback


def _is_metal(sym: str, metal_syms: Set[str]) -> bool:
    return sym in metal_syms


def _distance_matrix(mol: Molecule) -> np.ndarray:
    pos = mol.get_positions()
    n = len(pos)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = np.linalg.norm(pos[i] - pos[j])
            d[i, j] = d[j, i] = v
    return d


# -- check 1: overlap ----------------------------------------------------------

def _intra_ligand_pairs(mol: Molecule, metals: Set[str]) -> Set[tuple]:
    """
    Return the set of (i,j) index pairs (i<j) that are part of the same
    ligand fragment, defined as atoms connected through <=4 bonds without
    passing through a metal.  These are normal 1-2, 1-3, and 1-4
    interactions that always have short distances by construction.
    """
    atoms = mol.atoms
    n = len(atoms)
    pos = mol.get_positions()

    # Build bond graph (non-metal bonds only)
    adj: List[Set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = atoms[i].symbol, atoms[j].symbol
            if si in metals or sj in metals:
                continue
            d = float(np.linalg.norm(pos[i] - pos[j]))
            if _is_ligand_bond(si, sj, d):
                adj[i].add(j)
                adj[j].add(i)

    # BFS up to depth 4 from each atom; collect all reachable atom pairs
    intra: Set[tuple] = set()
    for start in range(n):
        if atoms[start].symbol in metals:
            continue
        visited = {start: 0}
        queue = [start]
        while queue:
            cur = queue.pop(0)
            depth = visited[cur]
            if depth >= 4:
                continue
            for nb in adj[cur]:
                if nb not in visited:
                    visited[nb] = depth + 1
                    queue.append(nb)
        for other in visited:
            if other != start:
                pair = (min(start, other), max(start, other))
                intra.add(pair)

    return intra


def overlap_check(mol: Molecule) -> List[ValidationIssue]:
    """
    Flag atom pairs that are closer than the element-pair minimum
    non-bonded threshold AND are not a recognised covalent bond
    AND are not part of the same ligand fragment (intra-ligand 1-2
    through 1-4 contacts are skipped -- they are normal by construction).

    Metal-ligand contacts are excluded (those are coordination bonds
    with their own check).
    """
    issues: List[ValidationIssue] = []
    atoms  = mol.atoms
    metals = _metal_symbols(mol)
    dm     = _distance_matrix(mol)
    n      = len(atoms)

    # Pre-compute intra-ligand pairs so we can skip them quickly
    intra = _intra_ligand_pairs(mol, metals)

    for i in range(n):
        for j in range(i + 1, n):
            si, sj = atoms[i].symbol, atoms[j].symbol
            d = dm[i, j]

            # Skip metal-anything (handled by coordination_check)
            if si in metals or sj in metals:
                continue

            # Skip genuine covalent bonds
            if _is_ligand_bond(si, sj, d):
                continue

            # Skip intra-ligand 1-2 / 1-3 / 1-4 contacts (normal by construction)
            if (i, j) in intra:
                continue

            err_threshold  = _min_nonbonded_error(si, sj)
            warn_threshold = _min_nonbonded_warn(si, sj)
            if d < err_threshold:
                issues.append(ValidationIssue(
                    check="overlap",
                    severity="error",
                    reason=(
                        f"{si}({i+1})-{sj}({j+1}): {d:.3f} Angstrom "
                        f"< hard minimum {err_threshold:.2f} Angstrom "
                        f"(impossible overlap, vdW sum x 0.65)"
                    ),
                    atom_i=i, atom_j=j, distance=d,
                ))
            elif d < warn_threshold:
                issues.append(ValidationIssue(
                    check="overlap",
                    severity="warning",
                    reason=(
                        f"{si}({i+1})-{sj}({j+1}): {d:.3f} Angstrom "
                        f"< soft minimum {warn_threshold:.2f} Angstrom "
                        f"(tight pre-DFT contact, vdW sum x 0.76)"
                    ),
                    atom_i=i, atom_j=j, distance=d,
                ))

    return issues


# -- check 2: coordination -----------------------------------------------------

# Maximum M-L distance to count as coordinated (Angstrom).
# Any donor atom beyond this is considered detached.
_MAX_COORD_DISTANCE = 3.20   # generous upper bound; real bonds < 2.8 Angstrom

def coordination_check(mol: Molecule) -> List[ValidationIssue]:
    """
    For each metal centre:
      - Check that every donor listed in metal_indices has at least
        one ligand atom within _MAX_COORD_DISTANCE.
      - Check that no M-L distance is implausibly short (< 1.4 Angstrom).
      - Warn if the coordination number differs from what is implied
        by the molecule's declared geometry (+/-1 tolerance).
    """
    issues: List[ValidationIssue] = []
    atoms  = mol.atoms
    metals = _metal_symbols(mol)
    dm     = _distance_matrix(mol)
    n      = len(atoms)

    _GEOM_CN = {
        "lin": 2, "tp": 3, "tet": 4, "sqp": 4,
        "tbp": 5, "sqpy": 5, "oct": 6, "pbp": 7,
        "sapr": 8, "tpr": 6,
    }

    for mi in mol.metal_indices:
        m_sym = atoms[mi].symbol

        # All non-metal atoms and their distances from this metal
        donor_dists = [
            (j, atoms[j].symbol, dm[mi, j])
            for j in range(n)
            if j != mi and atoms[j].symbol not in metals
        ]

        # Atoms within coordination distance
        coordinated = [(j, sym, d) for j, sym, d in donor_dists
                       if d <= _MAX_COORD_DISTANCE]

        # --- too-short M-L contact (genuine overlap with metal) ---
        for j, sym, d in coordinated:
            r_sum = COVALENT_RADII.get(m_sym, 1.4) + COVALENT_RADII.get(sym, 0.7)
            min_ml = r_sum * 0.65   # 35 % compression is physically impossible
            if d < min_ml:
                issues.append(ValidationIssue(
                    check="coordination",
                    severity="error",
                    reason=(
                        f"{m_sym}({mi+1})-{sym}({j+1}): {d:.3f} Angstrom "
                        f"impossibly short (min {min_ml:.2f} Angstrom for this pair)"
                    ),
                    atom_i=mi, atom_j=j, distance=d,
                ))

        # --- CN vs declared geometry ---
        if mol.geometry and mol.geometry in _GEOM_CN:
            # Count only first-shell donors (closest atom per ligand fragment)
            # Simple proxy: atoms within 2.8 Angstrom
            first_shell = [d for _, _, d in coordinated if d <= 2.80]
            expected_cn = _GEOM_CN[mol.geometry]
            actual_cn   = len(first_shell)
            if abs(actual_cn - expected_cn) > 1:
                issues.append(ValidationIssue(
                    check="coordination",
                    severity="warning",
                    reason=(
                        f"{m_sym}({mi+1}): declared geometry '{mol.geometry}' "
                        f"implies CN={expected_cn} but found {actual_cn} "
                        f"donors within 2.80 Angstrom"
                    ),
                    atom_i=mi,
                ))

        # --- detached metal (no donors at all) ---
        if not coordinated:
            issues.append(ValidationIssue(
                check="coordination",
                severity="error",
                reason=f"{m_sym}({mi+1}): no donor atoms within {_MAX_COORD_DISTANCE} Angstrom -- metal is bare",
                atom_i=mi,
            ))

    return issues


# -- check 3: connectivity -----------------------------------------------------

def connectivity_check(mol: Molecule) -> List[ValidationIssue]:
    """
    Build a molecular graph where two atoms are connected if their
    distance is within 130 % of the sum of their covalent radii
    (generous to cope with pre-relaxation geometries), then verify
    that the graph has exactly one connected component.

    Detached atoms or fragments are reported as errors.
    """
    issues: List[ValidationIssue] = []
    atoms  = mol.atoms
    n      = len(atoms)
    metals = _metal_symbols(mol)
    dm     = _distance_matrix(mol)

    # Build adjacency
    adj: List[Set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            si, sj = atoms[i].symbol, atoms[j].symbol
            d = dm[i, j]
            r_sum = COVALENT_RADII.get(si, 1.0) + COVALENT_RADII.get(sj, 1.0)
            # Use 1.3x for ligand-ligand, 1.4x for metal-involved (dative bonds longer)
            factor = 1.40 if (si in metals or sj in metals) else 1.30
            if d <= r_sum * factor:
                adj[i].add(j)
                adj[j].add(i)

    # BFS from atom 0
    visited: Set[int] = set()
    queue = [0]
    while queue:
        cur = queue.pop()
        if cur in visited:
            continue
        visited.add(cur)
        queue.extend(adj[cur] - visited)

    disconnected = [i for i in range(n) if i not in visited]
    if disconnected:
        frags: List[List[int]] = []
        remaining = set(disconnected)
        while remaining:
            seed = next(iter(remaining))
            comp = []
            q = [seed]
            seen: Set[int] = set()
            while q:
                cur = q.pop()
                if cur in seen:
                    continue
                seen.add(cur)
                comp.append(cur)
                q.extend(adj[cur] - seen)
            frags.append(comp)
            remaining -= seen

        for frag in frags:
            syms = [atoms[i].symbol for i in frag]
            issues.append(ValidationIssue(
                check="connectivity",
                severity="error",
                reason=(
                    f"Disconnected fragment: {len(frag)} atom(s) "
                    f"({', '.join(f'{s}({i+1})' for s, i in zip(syms, frag))}) "
                    f"not bonded to the main structure"
                ),
                atom_i=frag[0],
            ))

    return issues


# -- check 4: geometry (optional) ----------------------------------------------

# Ideal L-M-L angles (deg) for common geometries
_IDEAL_ANGLES: dict[str, List[float]] = {
    "lin":  [180.0],
    "tp":   [120.0],
    "tet":  [109.47],
    "sqp":  [90.0, 180.0],
    "tbp":  [90.0, 120.0],
    "sqpy": [90.0],
    "oct":  [90.0, 180.0],
    "pbp":  [72.0, 90.0],
}

_ANGLE_TOLERANCE = 25.0   # degrees; pre-optimisation structures can deviate a lot

def geometry_check(mol: Molecule) -> List[ValidationIssue]:
    """
    For each metal centre, compute all L-M-L angles between the
    nearest donors and compare them to the ideal angles for the
    declared geometry.  Issues a warning (not error) when no
    measured angle falls within +/-_ANGLE_TOLERANCE of any ideal value.
    """
    issues: List[ValidationIssue] = []
    if not mol.geometry or mol.geometry not in _IDEAL_ANGLES:
        return issues

    atoms  = mol.atoms
    metals = _metal_symbols(mol)
    dm     = _distance_matrix(mol)
    n      = len(atoms)
    ideals = _IDEAL_ANGLES[mol.geometry]

    for mi in mol.metal_indices:
        m_pos = atoms[mi].position

        # First-shell donors: non-metal atoms within 3.0 Angstrom
        donors = [
            j for j in range(n)
            if j != mi
            and atoms[j].symbol not in metals
            and dm[mi, j] <= 3.0
        ]

        if len(donors) < 2:
            continue

        # Compute all L-M-L angles
        measured: List[float] = []
        for a, da in enumerate(donors):
            for b, db in enumerate(donors):
                if b <= a:
                    continue
                va = atoms[da].position - m_pos
                vb = atoms[db].position - m_pos
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                if na < 1e-6 or nb < 1e-6:
                    continue
                cos_t = float(np.clip(np.dot(va / na, vb / nb), -1.0, 1.0))
                angle = math.degrees(math.acos(cos_t))
                measured.append(angle)

        if not measured:
            continue

        # For each measured angle, check if it's within tolerance of any ideal
        badly_distorted = [
            ang for ang in measured
            if not any(abs(ang - ideal) <= _ANGLE_TOLERANCE for ideal in ideals)
        ]

        if len(badly_distorted) > len(measured) * 0.5:
            # More than half the angles are far from ideal -- flag it
            worst = max(
                badly_distorted,
                key=lambda a: min(abs(a - ideal) for ideal in ideals),
            )
            issues.append(ValidationIssue(
                check="geometry",
                severity="warning",
                reason=(
                    f"{atoms[mi].symbol}({mi+1}): "
                    f"{len(badly_distorted)}/{len(measured)} L-M-L angles "
                    f"deviate >+/- {_ANGLE_TOLERANCE}deg from ideal {mol.geometry} angles "
                    f"{ideals}. Worst: {worst:.1f}deg"
                ),
                atom_i=mi,
            ))

    return issues


# -- public API -----------------------------------------------------------------

_ALL_CHECKS = ("overlap", "coordination", "connectivity", "geometry")

def validate(
    mol: Molecule,
    checks: tuple = ("overlap", "coordination", "connectivity"),
    strict: bool = False,
) -> ValidationResult:
    """
    Run the requested validation checks on *mol*.

    Parameters
    ----------
    mol     : Molecule to validate
    checks  : tuple of check names to run.
              Default: ('overlap', 'coordination', 'connectivity')
              Add 'geometry' for optional angle checks.
    strict  : if True, coordination and geometry warnings are promoted to errors.

    Returns
    -------
    ValidationResult
        .passed  - True iff no error-severity issues found
        .issues  - list of ValidationIssue
        .summary - formatted string
    """
    result = ValidationResult()

    dispatch = {
        "overlap":       overlap_check,
        "coordination":  coordination_check,
        "connectivity":  connectivity_check,
        "geometry":      geometry_check,
    }

    for name in checks:
        if name not in dispatch:
            raise ValueError(f"Unknown check '{name}'. Valid: {list(dispatch)}")
        new_issues = dispatch[name](mol)
        if strict:
            new_issues = [
                ValidationIssue(
                    check=iss.check,
                    severity="error",
                    reason=iss.reason,
                    atom_i=iss.atom_i,
                    atom_j=iss.atom_j,
                    distance=iss.distance,
                )
                if iss.severity == "warning" else iss
                for iss in new_issues
            ]
        result.issues.extend(new_issues)

    return result


def is_valid(mol: Molecule, **kw) -> bool:
    """Convenience wrapper -- returns True iff validate() passes."""
    return validate(mol, **kw).passed
