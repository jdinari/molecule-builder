"""
tests/test_exceptions.py
========================
Tests that the right exception types are raised at the public API.
"""

import pytest

from molbuilder import (
    build, dimer, trimer,
    InvalidLigandError, MolbuilderError, ValidationError,
)
from molbuilder.exceptions import ClashError


class TestExceptionHierarchy:
    def test_invalid_ligand_is_molbuilder_error(self):
        assert issubclass(InvalidLigandError, MolbuilderError)

    def test_validation_error_is_molbuilder_error(self):
        assert issubclass(ValidationError, MolbuilderError)

    def test_clash_error_is_geometry_error(self):
        from molbuilder.exceptions import GeometryError
        assert issubclass(ClashError, GeometryError)


class TestInvalidLigandRaised:
    def test_build_unknown_ligand(self):
        with pytest.raises(InvalidLigandError):
            build("Fe", ox=3, ligands=["NOTELIGAND"])

    def test_build_typo(self):
        with pytest.raises(InvalidLigandError):
            build("Ni", ox=2, ligands=["TYPO123"])   # unknown ligand

    def test_get_ligand_obj_unknown(self):
        from molbuilder import get_ligand_obj
        with pytest.raises(InvalidLigandError):
            get_ligand_obj("BOGUS")


class TestValidationError:
    def test_dimer_impossible_geometry_raises(self):
        """
        4 mu-OH bridges on a short Ni-Ni distance is geometrically impossible;
        dimer() should raise (currently as ValueError wrapping validation failure).
        We just verify it raises — the exact type will change when we migrate
        dimer() to raise ValidationError directly.
        """
        with pytest.raises(Exception):  # ValueError today, ValidationError eventually
            dimer("Ni", ox=2, terminal=[], bridge="mu-OH", n=4)
