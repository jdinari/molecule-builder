"""
tests/test_serialization.py
===========================
Tests for Molecule.to_dict / from_dict / to_json / from_json.
"""

import json
import numpy as np
import pytest

from molbuilder import build, dimer, trimer
from molbuilder.core.molecule import Molecule, Atom


class TestAtomSerialization:
    def test_roundtrip(self):
        a = Atom(symbol="Fe", position=np.array([1.0, 2.0, 3.0]), label="Fe1")
        d = a.to_dict()
        a2 = Atom.from_dict(d)
        assert a2.symbol == "Fe"
        assert a2.label == "Fe1"
        assert np.allclose(a2.position, [1.0, 2.0, 3.0])

    def test_position_is_ndarray(self):
        a = Atom.from_dict({"symbol": "O", "position": [0.1, 0.2, 0.3]})
        assert isinstance(a.position, np.ndarray)


class TestMoleculeToDict:
    def setup_method(self):
        # Ni(H2O)6 has only one isomer (all identical ligands)
        self.mol = build("Ni", ox=2, ligands=["H2O"] * 6)

    def test_keys_present(self):
        d = self.mol.to_dict()
        for key in ("formula", "charge", "spin_multiplicity", "geometry",
                    "metal_symbol", "metal_ox", "atoms"):
            assert key in d, f"Missing key: {key}"

    def test_formula_preserved(self):
        assert self.mol.to_dict()["formula"] == self.mol.formula

    def test_charge_preserved(self):
        assert self.mol.to_dict()["charge"] == self.mol.charge

    def test_atoms_count(self):
        d = self.mol.to_dict()
        assert len(d["atoms"]) == self.mol.num_atoms()

    def test_json_serializable(self):
        d = self.mol.to_dict()
        s = json.dumps(d)   # must not raise
        assert isinstance(s, str)

    def test_version_key(self):
        assert "_molbuilder_version" in self.mol.to_dict()


class TestMoleculeFromDict:
    def _roundtrip(self, mol):
        d = mol.to_dict()
        mol2 = Molecule.from_dict(d)
        return mol2

    def test_formula_roundtrip(self):
        mol = build("Ni", ox=2, ligands=["H2O"] * 6)
        assert self._roundtrip(mol).formula == mol.formula

    def test_charge_roundtrip(self):
        mol = build("Fe", ox=3, ligands=["Cl"] * 6)
        assert self._roundtrip(mol).charge == mol.charge

    def test_spin_roundtrip(self):
        mol = build("Co", ox=3, ligands=["NH3"] * 6)
        assert self._roundtrip(mol).spin_multiplicity == mol.spin_multiplicity

    def test_positions_roundtrip(self):
        # Use build_isomers()[0] to get a single Molecule, not a list
        from molbuilder import build_isomers
        mol = build_isomers("Pd", ox=2, ligands=["Cl", "Cl", "NH3", "NH3"], geometry="sqp")[0]
        mol2 = self._roundtrip(mol)
        orig = mol.get_positions()
        rec  = mol2.get_positions()
        assert np.allclose(orig, rec, atol=1e-4)

    def test_num_atoms_roundtrip(self):
        mol = build("Cr", ox=0, ligands=["CO"] * 6)
        assert self._roundtrip(mol).num_atoms() == mol.num_atoms()

    def test_metal_symbol_roundtrip(self):
        mol = build("Ru", ox=2, ligands=["NH3"] * 6)
        assert self._roundtrip(mol).metal_symbol == "Ru"

    def test_missing_key_raises(self):
        with pytest.raises((ValueError, KeyError)):
            Molecule.from_dict({"formula": "broken"})  # no 'atoms' key

    def test_dimer_roundtrip(self):
        mol = dimer("Ni", ox=2, terminal=["H2O"], bridge="mu-OH", n=2)
        mol2 = self._roundtrip(mol)
        assert mol2.formula == mol.formula
        assert len(mol2.metal_indices) == len(mol.metal_indices)


class TestMoleculeJsonMethods:
    def test_to_json_is_string(self):
        mol = build("Fe", ox=2, ligands=["H2O"] * 6)
        s = mol.to_json()
        assert isinstance(s, str)

    def test_from_json_roundtrip(self):
        mol = build("Ni", ox=2, ligands=["H2O"] * 6)
        s   = mol.to_json()
        mol2 = Molecule.from_json(s)
        assert mol2.formula == mol.formula
        assert np.allclose(mol2.get_positions(), mol.get_positions(), atol=1e-4)

    def test_to_json_indent(self):
        mol = build("Fe", ox=2, ligands=["H2O"] * 6)
        s = mol.to_json(indent=2)
        assert "\n" in s  # indented JSON has newlines

    def test_trimer_json_roundtrip(self):
        mol  = trimer("Ni", ox=2, terminal=[], bridge="mu-HCOO",
                      arrangement="triangular", n_bridges_per_pair=2)
        mol2 = Molecule.from_json(mol.to_json())
        assert mol2.formula == mol.formula
        assert mol2.charge == 0
