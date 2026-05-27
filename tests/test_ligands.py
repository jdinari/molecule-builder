"""
tests/test_ligands.py
=====================
Unit tests for the ligand library and typed Ligand model.
"""

import pytest

from molbuilder.ligands.library import get_ligand, get_ligand_obj, list_ligands
from molbuilder.ligands.models import Ligand
from molbuilder.exceptions import InvalidLigandError


class TestGetLigandDict:
    """get_ligand() returns plain dicts for backward compatibility."""

    def test_co_charge(self):
        assert get_ligand("CO")["charge"] == 0

    def test_cl_charge(self):
        assert get_ligand("Cl")["charge"] == -1

    def test_h2o_donor(self):
        assert get_ligand("H2O")["donor_atoms"] == ["O"]

    def test_bpy_bidentate(self):
        d = get_ligand("bpy")
        assert d["denticity"] == 2
        assert d["bite_angle"] is not None

    def test_edta_hexadentate(self):
        d = get_ligand("EDTA")
        assert d["denticity"] == 6
        assert d["charge"] == -4

    def test_alias_aqua_resolves(self):
        d = get_ligand("aqua")
        assert d["donor_atoms"] == ["O"]

    def test_canonical_name_in_dict(self):
        d = get_ligand("aqua")
        assert "_canonical_name" in d

    def test_mu_oh_is_bridging(self):
        d = get_ligand("mu-OH")
        assert d.get("is_bridging") is True

    def test_hcoo_bi_bite_angle(self):
        d = get_ligand("HCOO:bi")
        assert 40 < d["bite_angle"] < 80

    def test_unknown_raises_invalid_ligand_error(self):
        with pytest.raises(InvalidLigandError):
            get_ligand("NOTELIGAND")

    def test_list_ligands_nonempty(self):
        ligs = list_ligands()
        assert len(ligs) > 20
        assert "H2O" in ligs
        assert "NH3" in ligs


class TestGetLigandObj:
    """get_ligand_obj() returns typed Ligand instances."""

    def test_returns_ligand_instance(self):
        lig = get_ligand_obj("H2O")
        assert isinstance(lig, Ligand)

    def test_co_properties(self):
        lig = get_ligand_obj("CO")
        assert lig.charge == 0
        assert lig.denticity == 1
        assert lig.primary_donor == "C"

    def test_bpy_properties(self):
        lig = get_ligand_obj("bpy")
        assert lig.denticity == 2
        assert lig.bite_angle is not None
        assert "N" in lig.donor_atoms

    def test_mu_oh_bridging(self):
        lig = get_ligand_obj("mu-OH")
        assert lig.is_bridging is True

    def test_hcoo_mono_not_bridging(self):
        lig = get_ligand_obj("HCOO")
        assert lig.is_bridging is False

    def test_vectors_count_monodentate(self):
        assert get_ligand_obj("H2O").vectors_count == 1

    def test_vectors_count_bidentate(self):
        assert get_ligand_obj("en").vectors_count == 2

    def test_unknown_raises(self):
        with pytest.raises(InvalidLigandError):
            get_ligand_obj("BOGUS")


class TestLigandModel:
    """Direct tests of the Ligand dataclass."""

    def test_frozen(self):
        lig = get_ligand_obj("H2O")
        with pytest.raises((AttributeError, TypeError)):
            lig.charge = 99  # type: ignore

    def test_repr(self):
        lig = get_ligand_obj("bpy")
        r = repr(lig)
        assert "bpy" in r
        assert "d2" in r

    def test_to_dict_roundtrip(self):
        lig = get_ligand_obj("HCOO:bi")
        d = lig.to_dict()
        lig2 = Ligand.from_dict_serialized(d)
        assert lig2.name == lig.name
        assert lig2.charge == lig.charge
        assert lig2.denticity == lig.denticity
        assert lig2.bite_angle == lig.bite_angle

    def test_from_name_alias(self):
        lig = Ligand.from_name("aqua")
        assert lig.primary_donor == "O"
