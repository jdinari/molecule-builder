"""
tests/test_validation.py
========================
Tests for the validation pipeline.
"""

import numpy as np
import pytest

from molbuilder import build, dimer, trimer
from molbuilder.api import check_structure
from molbuilder.core.validation import validate


class TestCheckStructure:
    """check_structure() should return empty list for valid structures."""

    def test_monomer_no_warnings(self):
        # Use build_isomers()[0] to get a single Molecule, not a list
        from molbuilder import build_isomers
        mol = build_isomers("Fe", ox=3, ligands=["Cl"] * 3 + ["H2O"] * 3)[0]
        assert check_structure(mol) == []

    def test_dimer_no_warnings(self):
        mol = dimer("Ni", ox=2, terminal=["H2O"], bridge="mu-OH", n=2)
        assert check_structure(mol) == []

    def test_trimer_triangular_no_warnings(self):
        mol = trimer("Ni", ox=2, terminal=[], bridge="mu-HCOO",
                     arrangement="triangular", n_bridges_per_pair=2)
        assert check_structure(mol) == []


class TestValidateFunction:
    def test_valid_structure_passes(self):
        # Fe(H2O)6 is clean; NH3 ligands have inter-ligand H clashes pre-DFT
        mol = build("Fe", ox=2, ligands=["H2O"] * 6)
        result = validate(mol)
        assert result.passed

    def test_result_has_summary(self):
        mol = build("Ni", ox=2, ligands=["H2O"] * 6)
        result = validate(mol)
        assert isinstance(result.summary, str)
        assert len(result.summary) > 0

    def test_result_bool_true_when_valid(self):
        mol = build("Fe", ox=2, ligands=["H2O"] * 6)
        result = validate(mol)
        assert bool(result) is True
