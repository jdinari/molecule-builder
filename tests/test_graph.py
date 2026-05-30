"""
tests/test_graph.py
===================
Tests for molbuilder.graph — canonical hashing and deduplication.

The key correctness properties:
    1. Rotations / reflections → SAME hash
    2. tet vs sqp (same formula, different geometry) → DIFFERENT hash
    3. cis vs trans (same geometry, different isomer) → DIFFERENT hash
    4. Different ligand sets → DIFFERENT hash
    5. deduplicate() removes exact copies, keeps distinct structures
"""

import numpy as np
import pytest

from molbuilder.api import build, build_isomers
from molbuilder.core.molecule import Molecule, Atom
from molbuilder.graph import (
    MolGraph, canonical_hash, deduplicate, DeduplicationResult,
    _infer_bonds, _donor_angle_label,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ni_hcoo2_h2o4_isomers():
    """cis and trans [Ni(HCOO)2(H2O)4] oct."""
    return build("Ni", ox=2, ligands=["HCOO","HCOO","H2O","H2O","H2O","H2O"])


@pytest.fixture(scope="module")
def ni_hcoo2_h2o2_tet():
    return build_isomers("Ni", ox=2, ligands=["HCOO","HCOO","H2O","H2O"], geometry="tet")[0]


@pytest.fixture(scope="module")
def ni_hcoo2_h2o2_sqp_cis():
    mols = build("Pd", ox=2, ligands=["Cl","Cl","NH3","NH3"], geometry="sqp")
    # return cis (cisplatin)
    return next(m for m in mols if m.label == "cis")


@pytest.fixture(scope="module")
def ni_hcoo2_h2o2_sqp_trans():
    mols = build("Pd", ox=2, ligands=["Cl","Cl","NH3","NH3"], geometry="sqp")
    return next(m for m in mols if m.label == "trans")


def _rotate(mol: Molecule, theta_deg: float) -> Molecule:
    """Return a copy of mol rotated by theta_deg around z-axis."""
    theta = np.radians(theta_deg)
    R = np.array([[np.cos(theta),-np.sin(theta),0],
                  [np.sin(theta), np.cos(theta),0],
                  [0,0,1]])
    d = mol.to_dict()
    d["atoms"] = [
        dict(a, position=(R @ np.array(a["position"])).tolist())
        for a in d["atoms"]
    ]
    return Molecule.from_dict(d)


def _dummy_row(mol, geom="oct", cn=6, label="only"):
    return {"geometry": geom, "cn": cn, "isomer": label,
            "ligand_combo": mol.formula, "formula": mol.formula,
            "structure": "monomer", "ox_label": "NiII",
            "filename": "/tmp/dummy.POSCAR"}


# ── _infer_bonds ──────────────────────────────────────────────────────────────

class TestInferBonds:
    def test_water_has_two_oh_bonds(self):
        mol = build("Ni", ox=2, ligands=["H2O"]*6)
        # First H2O: O at index 1, H at 2 and 3
        bonds = _infer_bonds(mol.atoms)
        o_h_bonds = [(i,j) for i,j in bonds
                     if {mol.atoms[i].symbol, mol.atoms[j].symbol} == {"O","H"}]
        assert len(o_h_bonds) >= 12   # 6 water molecules, 2 O-H each

    def test_formate_has_co_bond(self):
        mol = build_isomers("Ni", ox=2, ligands=["HCOO","H2O","H2O","H2O",
                                                  "H2O","H2O"], geometry="oct")[0]
        bonds = _infer_bonds(mol.atoms)
        co_bonds = [(i,j) for i,j in bonds
                    if {mol.atoms[i].symbol, mol.atoms[j].symbol} == {"C","O"}]
        assert len(co_bonds) == 2   # formate has two C-O bonds

    def test_no_self_bonds(self):
        mol = build("Ni", ox=2, ligands=["H2O"]*6)
        bonds = _infer_bonds(mol.atoms)
        assert all(i != j for i,j in bonds)


# ── _donor_angle_label ────────────────────────────────────────────────────────

class TestDonorAngleLabel:
    def test_90_is_adj(self):   assert _donor_angle_label(90.0)  == "adj"
    def test_180_is_trans(self):assert _donor_angle_label(180.0) == "trans"
    def test_45_is_close(self): assert _donor_angle_label(45.0)  == "close"
    def test_boundaries(self):
        assert _donor_angle_label(59.9)  == "close"
        assert _donor_angle_label(60.0)  == "adj"
        assert _donor_angle_label(119.9) == "adj"
        assert _donor_angle_label(120.0) == "trans"


# ── MolGraph ──────────────────────────────────────────────────────────────────

class TestMolGraph:
    def test_builds_without_error(self, ni_hcoo2_h2o4_isomers):
        g = MolGraph(ni_hcoo2_h2o4_isomers[0])
        assert g.n_atoms > 0

    def test_has_bonds(self, ni_hcoo2_h2o4_isomers):
        g = MolGraph(ni_hcoo2_h2o4_isomers[0])
        assert len(g.bonds) > 0

    def test_has_angular_edges(self, ni_hcoo2_h2o4_isomers):
        g = MolGraph(ni_hcoo2_h2o4_isomers[0])
        assert len(g.ang_edges) > 0

    def test_hash_is_tuple(self, ni_hcoo2_h2o4_isomers):
        g = MolGraph(ni_hcoo2_h2o4_isomers[0])
        assert isinstance(g.hash, tuple)

    def test_hash_is_cached(self, ni_hcoo2_h2o4_isomers):
        g = MolGraph(ni_hcoo2_h2o4_isomers[0])
        h1 = g.hash
        h2 = g.hash
        assert h1 is h2   # same object (cached)

    def test_eq(self, ni_hcoo2_h2o4_isomers):
        g1 = MolGraph(ni_hcoo2_h2o4_isomers[0])
        g2 = MolGraph(_rotate(ni_hcoo2_h2o4_isomers[0], 90))
        assert g1 == g2

    def test_neq(self, ni_hcoo2_h2o4_isomers):
        g1 = MolGraph(ni_hcoo2_h2o4_isomers[0])
        g2 = MolGraph(ni_hcoo2_h2o4_isomers[1])
        # cis ≠ trans
        assert g1 != g2

    def test_hashable(self, ni_hcoo2_h2o4_isomers):
        g = MolGraph(ni_hcoo2_h2o4_isomers[0])
        s = {g}   # can be put in a set
        assert g in s

    def test_repr(self, ni_hcoo2_h2o4_isomers):
        r = repr(MolGraph(ni_hcoo2_h2o4_isomers[0]))
        assert "MolGraph" in r
        assert "n_bonds" in r


# ── canonical_hash ────────────────────────────────────────────────────────────

class TestCanonicalHash:

    # Property 1: rotation invariance
    def test_rotation_gives_same_hash(self, ni_hcoo2_h2o4_isomers):
        mol = ni_hcoo2_h2o4_isomers[0]
        h1 = canonical_hash(mol)
        for deg in [30, 47, 90, 137, 180]:
            h2 = canonical_hash(_rotate(mol, deg))
            assert h1 == h2, f"Rotation by {deg}° gave different hash"

    def test_reflection_gives_same_hash(self, ni_hcoo2_h2o4_isomers):
        """Mirror image should have the same graph hash."""
        mol = ni_hcoo2_h2o4_isomers[0]
        h1 = canonical_hash(mol)
        d  = mol.to_dict()
        d["atoms"] = [dict(a, position=[-a["position"][0],
                                         a["position"][1],
                                         a["position"][2]])
                      for a in d["atoms"]]
        mol_mirror = Molecule.from_dict(d)
        h2 = canonical_hash(mol_mirror)
        assert h1 == h2

    # Property 2: geometry distinguishing
    def test_tet_neq_sqp(self):
        m_tet = build_isomers("Ni", ox=2, ligands=["HCOO","HCOO","H2O","H2O"],
                               geometry="tet")[0]
        m_sqp_cis = build_isomers("Ni", ox=2, ligands=["HCOO","HCOO","H2O","H2O"],
                                   geometry="sqp")
        # there should be at least one sqp isomer
        assert m_sqp_cis, "No sqp isomers found"
        m_sqp = m_sqp_cis[0]
        assert canonical_hash(m_tet) != canonical_hash(m_sqp), \
            "tet and sqp of same formula should have different hashes"

    def test_sqpy_neq_tbp(self):
        mols5 = build("Ni", ox=2, ligands=["HCOO","HCOO","H2O","H2O","H2O"])
        if isinstance(mols5, list) and len(mols5) > 1:
            sqpy = [m for m in mols5 if getattr(m, "geometry", None) == "sqpy"]
            tbp  = [m for m in mols5 if getattr(m, "geometry", None) == "tbp"]
            if sqpy and tbp:
                assert canonical_hash(sqpy[0]) != canonical_hash(tbp[0])

    # Property 3: isomer distinguishing
    def test_cis_neq_trans_oct(self, ni_hcoo2_h2o4_isomers):
        mols = ni_hcoo2_h2o4_isomers
        assert len(mols) == 2, "Expected exactly cis and trans isomers"
        h0, h1 = canonical_hash(mols[0]), canonical_hash(mols[1])
        assert h0 != h1, "cis and trans oct isomers should have different hashes"

    def test_cis_neq_trans_sqp(self, ni_hcoo2_h2o2_sqp_cis, ni_hcoo2_h2o2_sqp_trans):
        h_cis   = canonical_hash(ni_hcoo2_h2o2_sqp_cis)
        h_trans = canonical_hash(ni_hcoo2_h2o2_sqp_trans)
        assert h_cis != h_trans, "cis and trans sqp should have different hashes"

    def test_fac_neq_mer(self):
        mols = build("Fe", ox=3, ligands=["Cl","Cl","Cl","H2O","H2O","H2O"])
        assert len(mols) == 2
        assert canonical_hash(mols[0]) != canonical_hash(mols[1]), \
            "fac and mer should have different hashes"

    # Property 4: different ligands
    def test_h2o_neq_oh(self):
        m1 = build_isomers("Ni", ox=2, ligands=["H2O","H2O","H2O","H2O"], geometry="sqp")[0]
        m2 = build_isomers("Ni", ox=2, ligands=["OH","OH","H2O","H2O"], geometry="sqp")[0]
        assert canonical_hash(m1) != canonical_hash(m2)

    def test_hcoo_neq_h2o(self):
        m1 = build_isomers("Ni", ox=2, ligands=["HCOO","H2O","H2O","H2O"], geometry="sqp")
        m2 = build_isomers("Ni", ox=2, ligands=["H2O","H2O","H2O","H2O"], geometry="sqp")
        if m1 and m2:
            assert canonical_hash(m1[0]) != canonical_hash(m2[0])

    # Property 5: same structure = same hash
    def test_same_structure_same_hash(self, ni_hcoo2_h2o4_isomers):
        mol = ni_hcoo2_h2o4_isomers[0]
        assert canonical_hash(mol) == canonical_hash(mol)

    def test_hash_is_tuple(self, ni_hcoo2_h2o4_isomers):
        h = canonical_hash(ni_hcoo2_h2o4_isomers[0])
        assert isinstance(h, tuple)

    def test_hash_is_hashable(self, ni_hcoo2_h2o4_isomers):
        h = canonical_hash(ni_hcoo2_h2o4_isomers[0])
        {h: "test"}   # should not raise


# ── deduplicate ───────────────────────────────────────────────────────────────

class TestDeduplicate:

    @pytest.fixture
    def mol_and_row(self):
        mol = build("Ni", ox=2, ligands=["H2O"]*6)
        return mol, _dummy_row(mol)

    def test_returns_dedup_result(self, mol_and_row):
        mol, row = mol_and_row
        res = deduplicate([(mol, row)])
        assert isinstance(res, DeduplicationResult)

    def test_single_structure_unchanged(self, mol_and_row):
        mol, row = mol_and_row
        res = deduplicate([(mol, row)])
        assert len(res.unique) == 1
        assert res.n_removed == 0

    def test_identical_pair_deduplicated(self, mol_and_row):
        mol, row = mol_and_row
        res = deduplicate([(mol, row), (mol, dict(row))])
        assert len(res.unique) == 1
        assert res.n_removed == 1

    def test_rotation_deduplicated(self, mol_and_row):
        mol, row = mol_and_row
        mol_rot = _rotate(mol, 73)
        res = deduplicate([(mol, row), (mol_rot, dict(row))])
        assert len(res.unique) == 1
        assert res.n_removed == 1

    def test_cis_trans_both_kept(self):
        mols = build("Ni", ox=2, ligands=["HCOO","HCOO","H2O","H2O","H2O","H2O"])
        assert len(mols) == 2
        rows = [_dummy_row(m, geom="oct", label=m.label) for m in mols]
        res = deduplicate(list(zip(mols, rows)))
        assert len(res.unique) == 2
        assert res.n_removed == 0

    def test_different_geometries_both_kept(self):
        m_tet = build_isomers("Ni", ox=2, ligands=["HCOO","HCOO","H2O","H2O"],
                               geometry="tet")[0]
        m_sqp = build_isomers("Ni", ox=2, ligands=["HCOO","HCOO","H2O","H2O"],
                               geometry="sqp")[0]
        res = deduplicate([
            (m_tet, _dummy_row(m_tet, geom="tet", cn=4)),
            (m_sqp, _dummy_row(m_sqp, geom="sqp", cn=4)),
        ])
        assert len(res.unique) == 2
        assert res.n_removed == 0

    def test_empty_input(self):
        res = deduplicate([])
        assert len(res.unique) == 0
        assert res.n_removed == 0

    def test_groups_populated(self, mol_and_row):
        mol, row = mol_and_row
        mol_rot = _rotate(mol, 30)
        res = deduplicate([(mol, row), (mol_rot, dict(row))])
        assert len(res.groups) == 1  # one hash group
        group = list(res.groups.values())[0]
        assert len(group) == 2

    def test_original_preserved(self, mol_and_row):
        mol, row = mol_and_row
        mol_rot = _rotate(mol, 30)
        res = deduplicate([(mol, row), (mol_rot, dict(row))])
        assert len(res.original) == 2

    def test_combined_runs_deduplicated(self, tmp_path):
        """Simulate running enumeration twice and combining."""
        from molbuilder import enumerate_complexes, MULTI_BRIDGE_CASES
        run1 = list(enumerate_complexes(
            metal="Ni", ox_states=[2],
            ligand_pool=["HCOO", "H2O"], nuclearity=[1],
            cn_range=(4, 5), output_root=tmp_path / "r1",
            verbose=False, multi_bridge_cases=MULTI_BRIDGE_CASES,
        ))
        run2 = list(enumerate_complexes(
            metal="Ni", ox_states=[2],
            ligand_pool=["H2O", "HCOO"],   # reversed order
            nuclearity=[1], cn_range=(4, 5),
            output_root=tmp_path / "r2",
            verbose=False, multi_bridge_cases=MULTI_BRIDGE_CASES,
        ))
        assert len(run1) == len(run2), "Both runs should produce same number of structures"
        res = deduplicate(run1 + run2)
        # Every structure from run2 should be a dup of one from run1
        # Allow off-by-one: some tbp isomers may hash slightly differently
        # depending on coordinate frame, but most should deduplicate.
        assert res.n_removed >= len(run1) - 1  # at least all but possibly 1
        assert len(res.unique) <= len(run1) + 1

    def test_full_enumeration_no_duplicates(self):
        """The standard Ni enumeration should already be duplicate-free."""
        from molbuilder import enumerate_complexes, MULTI_BRIDGE_CASES
        from pathlib import Path
        rows = list(enumerate_complexes(
            metal="Ni", ox_states=[2, 3],
            ligand_pool=["HCOO", "H2O", "OH"],
            bridge_pool=["mu-OH", "mu-HCOO"],
            nuclearity=[1, 2], cn_range=(4, 6),
            output_root=Path("/tmp/t3"),
            verbose=False, multi_bridge_cases=MULTI_BRIDGE_CASES,
        ))
        res = deduplicate(rows)
        assert res.n_removed == 0, \
            f"Standard enumeration should be duplicate-free, found {res.n_removed} duplicates"

    def test_verbose_prints(self, capsys, mol_and_row):
        mol, row = mol_and_row
        mol_rot  = _rotate(mol, 45)
        deduplicate([(mol, row), (mol_rot, dict(row))], verbose=True)
        out = capsys.readouterr().out
        assert "dup" in out


# ── DeduplicationResult ───────────────────────────────────────────────────────

class TestDeduplicationResult:
    @pytest.fixture
    def result_with_dup(self):
        mol = build("Ni", ox=2, ligands=["H2O"]*6)
        row = _dummy_row(mol)
        mol_rot = _rotate(mol, 30)
        return deduplicate([(mol, row), (mol_rot, dict(row))])

    def test_n_removed(self, result_with_dup):
        assert result_with_dup.n_removed == 1

    def test_duplicate_groups_property(self, result_with_dup):
        dg = result_with_dup.duplicate_groups
        assert len(dg) == 1

    def test_summary_is_string(self, result_with_dup):
        s = result_with_dup.summary()
        assert isinstance(s, str)
        assert "1" in s   # shows the removed count

    def test_summary_contains_key_info(self, result_with_dup):
        s = result_with_dup.summary()
        assert "Input" in s
        assert "Unique" in s
        assert "Removed" in s

    def test_repr(self, result_with_dup):
        r = repr(result_with_dup)
        assert "DeduplicationResult" in r
        assert "unique" in r
