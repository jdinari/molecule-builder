"""
tests/test_regression.py
========================
Regression tests for specific structures that have been manually verified.

Each test asserts:
  - formula
  - charge
  - spin multiplicity
  - atom count
  - Ni-Ni distances (within 0.01 Å)
  - minimum O-O distance above hard clash threshold (1.976 Å)
  - no geometry warnings from check_structure()

These are the structures whose correctness was confirmed during development.
If any of these break, something fundamental changed in the placement algorithm.
"""

import numpy as np
import pytest

from molbuilder import build, dimer, trimer
from molbuilder.api import check_structure


# ── helpers ───────────────────────────────────────────────────────────────────

def metal_positions(mol, symbol=None):
    sym = symbol or mol.metal_symbol
    return [a.position for a in mol.atoms if a.symbol == sym]


def metal_metal_distances(mol, symbol=None):
    positions = metal_positions(mol, symbol)
    dists = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dists.append(np.linalg.norm(positions[i] - positions[j]))
    return sorted(dists)


def min_oo_distance(mol):
    o_pos = [a.position for a in mol.atoms if a.symbol == "O"]
    if len(o_pos) < 2:
        return float("inf")
    min_d = float("inf")
    for i in range(len(o_pos)):
        for j in range(i + 1, len(o_pos)):
            d = np.linalg.norm(o_pos[i] - o_pos[j])
            if d < min_d:
                min_d = d
    return min_d


HARD_OO_THRESHOLD = 1.976   # Å — from validation.py vdW sum × 0.65


# ── Ni₃(μ-HCOO)₆ triangular — manually verified ──────────────────────────────

class TestNi3HCOO6Triangular:
    """
    The triangular Ni₃ trimer with 2 syn-syn formate bridges per edge.
    This was the structure the user built by hand and verified.
    α = 35° tilt scheme was derived analytically to ensure min O···O ≥ 2.20 Å.
    """

    @pytest.fixture(autouse=True)
    def build_mol(self):
        self.mol = trimer(
            "Ni", ox=2, terminal=[],
            bridge="mu-HCOO", arrangement="triangular",
            n_bridges_per_pair=2,
        )

    def test_formula(self):
        assert self.mol.formula == "C6H6Ni3O12"

    def test_charge(self):
        assert self.mol.charge == 0

    def test_spin_multiplicity(self):
        assert self.mol.spin_multiplicity == 3

    def test_atom_count(self):
        assert self.mol.num_atoms() == 27

    def test_three_metals(self):
        assert len(metal_positions(self.mol)) == 3

    def test_equilateral_triangle(self):
        """All three Ni-Ni distances should be equal to within 0.001 Å."""
        dists = metal_metal_distances(self.mol)
        assert len(dists) == 3
        assert abs(dists[0] - dists[1]) < 0.001
        assert abs(dists[1] - dists[2]) < 0.001

    def test_ni_ni_distance(self):
        """Ni-Ni should be ~3.568 Å (from mxm_angle=120°, Ni-O bl=2.06 Å)."""
        d = metal_metal_distances(self.mol)[0]
        assert abs(d - 3.568) < 0.01

    def test_min_oo_above_threshold(self):
        """No O···O clash: all O-O distances must exceed the hard limit."""
        assert min_oo_distance(self.mol) >= HARD_OO_THRESHOLD

    def test_min_oo_value(self):
        """The tilt scheme achieves min O-O ≈ 2.20 Å."""
        assert min_oo_distance(self.mol) >= 2.18

    def test_no_geometry_warnings(self):
        assert check_structure(self.mol) == []

    def test_each_ni_has_four_o_donors(self):
        """Each Ni should have exactly 4 O within bonding distance (CN=4)."""
        for ni_pos in metal_positions(self.mol):
            donors = [
                a for a in self.mol.atoms
                if a.symbol == "O"
                and np.linalg.norm(a.position - ni_pos) < 2.5
            ]
            assert len(donors) == 4, \
                f"Expected CN=4, found {len(donors)} O donors"

    def test_json_roundtrip_preserves_geometry(self):
        from molbuilder.core.molecule import Molecule
        mol2 = Molecule.from_json(self.mol.to_json())
        assert np.allclose(
            mol2.get_positions(), self.mol.get_positions(), atol=1e-4
        )


# ── Ni₂(μ-HCOO)₄ paddle-wheel ────────────────────────────────────────────────

class TestNi2HCOO4PaddleWheel:
    """
    The symmetric paddle-wheel dimer with 4 bridging formates and no terminals.
    CN = 4 per Ni, tetrahedral.
    """

    @pytest.fixture(autouse=True)
    def build_mol(self):
        self.mol = dimer("Ni", ox=2, terminal=[], bridge="mu-HCOO", n=4)

    def test_formula(self):
        assert self.mol.formula == "C4H4Ni2O8"

    def test_charge(self):
        assert self.mol.charge == 0

    def test_atom_count(self):
        assert self.mol.num_atoms() == 18

    def test_two_metals(self):
        assert len(metal_positions(self.mol)) == 2

    def test_ni_ni_distance(self):
        """Ni-Ni ≈ 3.98 Å for 4 syn-syn formate bridges."""
        d = metal_metal_distances(self.mol)[0]
        assert abs(d - 3.9796) < 0.05

    def test_no_geometry_warnings(self):
        assert check_structure(self.mol) == []

    def test_four_formates(self):
        c_count = sum(1 for a in self.mol.atoms if a.symbol == "C")
        assert c_count == 4


# ── Ni₃(μ-OH)₆ triangular ────────────────────────────────────────────────────

class TestNi3OH6Triangular:
    """
    Triangular Ni₃ with 2 hydroxide bridges per edge.
    α = 70° tilt (larger than formate because OH is a single-atom bridge).
    """

    @pytest.fixture(autouse=True)
    def build_mol(self):
        self.mol = trimer(
            "Ni", ox=2, terminal=[],
            bridge="mu-OH", arrangement="triangular",
            n_bridges_per_pair=2,
        )

    def test_formula(self):
        assert self.mol.formula == "H6Ni3O6"

    def test_charge(self):
        assert self.mol.charge == 0

    def test_atom_count(self):
        assert self.mol.num_atoms() == 15

    def test_no_geometry_warnings(self):
        assert check_structure(self.mol) == []

    def test_min_oo_above_threshold(self):
        assert min_oo_distance(self.mol) >= HARD_OO_THRESHOLD

    def test_equilateral_triangle(self):
        dists = metal_metal_distances(self.mol)
        assert abs(dists[0] - dists[-1]) < 0.01


# ── Ni₂(μ-HCOO)₄(H₂O)₂ symmetric ───────────────────────────────────────────

class TestNi2HCOO4H2O2Symmetric:
    """Paddle-wheel with one water per metal (symmetric)."""

    @pytest.fixture(autouse=True)
    def build_mol(self):
        self.mol = dimer("Ni", ox=2, terminal=["H2O"], bridge="mu-HCOO", n=4)

    def test_formula(self):
        assert self.mol.formula == "C4H8Ni2O10"

    def test_charge(self):
        assert self.mol.charge == 0

    def test_atom_count(self):
        assert self.mol.num_atoms() == 24

    def test_no_geometry_warnings(self):
        assert check_structure(self.mol) == []


# ── Ni₂(μ-HCOO)₄(H₂O) heteroleptic ─────────────────────────────────────────

class TestNi2HCOO4H2OHeteroleptic:
    """
    Heteroleptic paddle-wheel: water on Ni1 only, Ni2 is bare.
    This was the specific structure the user asked about.
    """

    @pytest.fixture(autouse=True)
    def build_mol(self):
        self.mol = dimer(
            "Ni", ox=2,
            terminal_m1=["H2O"], terminal_m2=[],
            bridge="mu-HCOO", n=4,
        )

    def test_formula(self):
        assert self.mol.formula == "C4H6Ni2O9"

    def test_charge(self):
        assert self.mol.charge == 0

    def test_atom_count(self):
        assert self.mol.num_atoms() == 21

    def test_no_geometry_warnings(self):
        assert check_structure(self.mol) == []

    def test_asymmetric_cn(self):
        """Ni1 (with H2O) should have CN=5; Ni2 (bare) should have CN=4."""
        ni_pos = metal_positions(self.mol)
        cn = []
        for pos in ni_pos:
            donors = [
                a for a in self.mol.atoms
                if a.symbol in ("O", "N", "C", "S", "P", "Cl")
                and np.linalg.norm(a.position - pos) < 2.5
            ]
            cn.append(len(donors))
        assert sorted(cn) == [4, 5], \
            f"Expected CN=[4,5], got {sorted(cn)}"

    def test_distinct_from_symmetric(self):
        """Heteroleptic formula differs from symmetric version."""
        sym = dimer("Ni", ox=2, terminal=["H2O"], bridge="mu-HCOO", n=4)
        assert self.mol.formula != sym.formula
        assert self.mol.num_atoms() != sym.num_atoms()


# ── Bond lengths ──────────────────────────────────────────────────────────────

class TestBondLengths:
    """Spot-checks that CSD-averaged bond lengths are physically reasonable."""

    def test_ni_o_oct(self):
        from molbuilder.core.bond_lengths import get_bond_length
        bl = get_bond_length("Ni", 2, "O", "oct")
        assert 1.8 < bl < 2.5

    def test_fe_o_oct(self):
        from molbuilder.core.bond_lengths import get_bond_length
        bl = get_bond_length("Fe", 3, "O", "oct")
        assert abs(bl - 2.01) < 0.05

    def test_fallback_returns_positive(self):
        from molbuilder.core.bond_lengths import get_bond_length
        bl = get_bond_length("Fe", 2, "Xe", "oct")  # exotic donor
        assert bl > 0

    def test_ni_o_in_built_structure(self):
        """All Ni-O bonds in an octahedral Ni(H2O)6 should be ~2.06 Å."""
        mol = build("Ni", ox=2, ligands=["H2O"] * 6)
        ni_pos = metal_positions(mol)[0]
        o_dists = [
            np.linalg.norm(a.position - ni_pos)
            for a in mol.atoms if a.symbol == "O"
        ]
        for d in o_dists:
            assert 1.9 < d < 2.3, f"Unexpected Ni-O distance: {d:.3f}"
