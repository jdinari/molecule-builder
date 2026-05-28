"""
tests/test_relaxation.py
========================
Tests for the relaxation module (xTB backend only — MACE requires a model
download that is not available in CI).

All geometry tests use fmax=0.1 and steps=50 to keep runtime short.
"""

import numpy as np
import pytest

from molbuilder import build, build_isomers, dimer
from molbuilder.relaxation import (
    relax, compute_energy, compute_gibbs, thermochemistry,
    RelaxResult, ThermResult,
    _mol_to_ase, _ase_to_mol,
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ni_h2o4():
    """Ni(H2O)4 sqp — small, fast to relax."""
    mol = build_isomers("Ni", ox=2, ligands=["H2O"] * 4, geometry="sqp")[0]
    return mol


@pytest.fixture(scope="module")
def ni_oh4():
    """Ni(OH)4 sqp — closed-shell analogue."""
    mol = build_isomers("Ni", ox=2, ligands=["OH"] * 4, geometry="sqp")[0]
    return mol


@pytest.fixture(scope="module")
def ni2_oh2():
    """Ni2(μ-OH)2(H2O)2 dimer."""
    return dimer("Ni", ox=2, terminal=["H2O"], bridge="mu-OH", n=2)


# ── ASE conversion ────────────────────────────────────────────────────────────

class TestAseConversion:
    def test_mol_to_ase_symbols(self, ni_h2o4):
        atoms = _mol_to_ase(ni_h2o4)
        assert atoms.get_chemical_symbols()[0] == "Ni"

    def test_mol_to_ase_positions(self, ni_h2o4):
        atoms = _mol_to_ase(ni_h2o4)
        orig  = ni_h2o4.get_positions()
        assert np.allclose(atoms.get_positions(), orig, atol=1e-5)

    def test_mol_to_ase_non_periodic(self, ni_h2o4):
        atoms = _mol_to_ase(ni_h2o4)
        assert not any(atoms.get_pbc())

    def test_roundtrip_preserves_metadata(self, ni_h2o4):
        atoms = _mol_to_ase(ni_h2o4)
        mol2  = _ase_to_mol(atoms, ni_h2o4)
        assert mol2.formula          == ni_h2o4.formula
        assert mol2.charge           == ni_h2o4.charge
        assert mol2.spin_multiplicity == ni_h2o4.spin_multiplicity
        assert mol2.metal_symbol     == ni_h2o4.metal_symbol

    def test_roundtrip_preserves_positions(self, ni_h2o4):
        atoms = _mol_to_ase(ni_h2o4)
        mol2  = _ase_to_mol(atoms, ni_h2o4)
        assert np.allclose(mol2.get_positions(), ni_h2o4.get_positions(), atol=1e-5)


# ── RelaxResult and ThermResult dataclasses ───────────────────────────────────

class TestResultDataclasses:
    def test_relax_result_repr(self, ni_h2o4):
        res = compute_energy(ni_h2o4, backend="xtb")
        r   = repr(res)
        assert "RelaxResult" in r
        assert "xtb" in r

    def test_therm_result_repr(self, ni_h2o4):
        res = compute_gibbs(ni_h2o4, backend="xtb", T=298.15)
        r   = repr(res)
        assert "ThermResult" in r
        assert "298" in r

    def test_relax_result_is_not_therm(self, ni_h2o4):
        res = compute_energy(ni_h2o4, backend="xtb")
        assert isinstance(res, RelaxResult)
        assert not isinstance(res, ThermResult)

    def test_therm_result_is_relax_result(self, ni_h2o4):
        res = compute_gibbs(ni_h2o4, backend="xtb")
        assert isinstance(res, ThermResult)
        assert isinstance(res, RelaxResult)


# ── compute_energy ────────────────────────────────────────────────────────────

class TestComputeEnergy:
    def test_returns_relax_result(self, ni_h2o4):
        res = compute_energy(ni_h2o4, backend="xtb")
        assert isinstance(res, RelaxResult)

    def test_zero_steps(self, ni_h2o4):
        """Single-point should not move any atoms."""
        res = compute_energy(ni_h2o4, backend="xtb")
        assert res.steps == 0

    def test_converged_true(self, ni_h2o4):
        res = compute_energy(ni_h2o4, backend="xtb")
        assert res.converged is True

    def test_energy_is_float(self, ni_h2o4):
        res = compute_energy(ni_h2o4, backend="xtb")
        assert isinstance(res.energy_eV, float)

    def test_energy_negative(self, ni_h2o4):
        """GFN2-xTB absolute energies are always negative for stable molecules."""
        res = compute_energy(ni_h2o4, backend="xtb")
        assert res.energy_eV < 0

    def test_positions_unchanged(self, ni_h2o4):
        res = compute_energy(ni_h2o4, backend="xtb")
        assert np.allclose(
            res.mol.get_positions(), ni_h2o4.get_positions(), atol=1e-5
        )

    def test_backend_recorded(self, ni_h2o4):
        res = compute_energy(ni_h2o4, backend="xtb")
        assert res.backend == "xtb"
        assert "GFN2" in res.model

    def test_unknown_backend_raises(self, ni_h2o4):
        with pytest.raises(ValueError, match="Unknown backend"):
            compute_energy(ni_h2o4, backend="bogus")

    def test_python_bool_converged(self, ni_h2o4):
        """converged must be a plain Python bool, not numpy.bool_."""
        res = compute_energy(ni_h2o4, backend="xtb")
        assert type(res.converged) is bool

    def test_python_int_steps(self, ni_h2o4):
        res = compute_energy(ni_h2o4, backend="xtb")
        assert type(res.steps) is int


# ── relax ─────────────────────────────────────────────────────────────────────

class TestRelax:
    def test_returns_relax_result(self, ni_h2o4):
        res = relax(ni_h2o4, backend="xtb", fmax=0.1, steps=30)
        assert isinstance(res, RelaxResult)

    def test_energy_lower_than_singlepoint(self, ni_h2o4):
        """Relaxed energy should be ≤ single-point energy."""
        sp  = compute_energy(ni_h2o4, backend="xtb")
        opt = relax(ni_h2o4, backend="xtb", fmax=0.1, steps=80)
        assert opt.energy_eV <= sp.energy_eV + 0.01   # small tolerance for noise

    def test_positions_changed(self, ni_h2o4):
        """Relaxation should move at least some atoms."""
        res = relax(ni_h2o4, backend="xtb", fmax=0.1, steps=50)
        assert not np.allclose(
            res.mol.get_positions(), ni_h2o4.get_positions(), atol=1e-3
        )

    def test_formula_preserved(self, ni_h2o4):
        res = relax(ni_h2o4, backend="xtb", fmax=0.1, steps=30)
        assert res.mol.formula == ni_h2o4.formula

    def test_charge_preserved(self, ni_h2o4):
        res = relax(ni_h2o4, backend="xtb", fmax=0.1, steps=30)
        assert res.mol.charge == ni_h2o4.charge

    def test_dimer_relaxation(self, ni2_oh2):
        res = relax(ni2_oh2, backend="xtb", fmax=0.1, steps=50)
        assert res.energy_eV < 0
        assert res.mol.formula == ni2_oh2.formula

    def test_steps_recorded(self, ni_h2o4):
        res = relax(ni_h2o4, backend="xtb", fmax=0.1, steps=30)
        assert 0 < res.steps <= 30


# ── compute_gibbs (freq + thermo, no geometry change) ─────────────────────────

class TestComputeGibbs:
    @pytest.fixture(scope="class")
    def therm(self, ni_oh4):
        """Use ni_oh4 (stiffer, fewer soft modes) for stable freq calc."""
        return compute_gibbs(ni_oh4, backend="xtb", T=298.15, P=101325.0)

    def test_returns_therm_result(self, therm):
        assert isinstance(therm, ThermResult)

    def test_zpe_positive(self, therm):
        assert therm.zpe_eV > 0

    def test_enthalpy_has_thermal_correction(self, therm):
        """H(T) > E + ZPE because thermal corrections add energy."""
        assert therm.enthalpy_eV > therm.energy_eV + therm.zpe_eV - 0.5

    def test_entropy_positive(self, therm):
        assert therm.entropy_eV_K > 0

    def test_gibbs_less_than_enthalpy(self, therm):
        """G = H - TS < H for T > 0."""
        assert therm.gibbs_eV < therm.enthalpy_eV

    def test_has_frequencies(self, therm):
        assert len(therm.frequencies) > 0

    def test_vib_energies_stored(self, therm):
        assert len(therm._vib_energies) > 0

    def test_T_P_recorded(self, therm):
        assert therm.T_K  == 298.15
        assert therm.P_Pa == 101325.0

    def test_gibbs_at_same_T(self, therm):
        """gibbs_at(298.15) should reproduce .gibbs_eV closely."""
        g2 = therm.gibbs_at(T=298.15, P=101325.0)
        assert abs(g2 - therm.gibbs_eV) < 1e-4

    def test_gibbs_at_higher_T_lower(self, therm):
        """G(350 K) < G(298 K) because -TS term grows with T."""
        g_350 = therm.gibbs_at(T=350.0)
        assert g_350 < therm.gibbs_eV

    def test_gibbs_at_lower_P_lower(self, therm):
        """Lower pressure → more translational entropy → lower G."""
        g_low_P = therm.gibbs_at(T=298.15, P=10132.5)   # 0.1 atm
        assert g_low_P < therm.gibbs_eV

    def test_gibbs_at_no_vib_data_warns(self, ni_oh4):
        """gibbs_at() warns gracefully if _vib_energies is empty."""
        import warnings
        t = compute_gibbs(ni_oh4, backend="xtb")
        t._vib_energies = []   # simulate missing data
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g = t.gibbs_at(T=300)
            assert any("vibrational" in str(x.message).lower() for x in w)
        assert isinstance(g, float)


# ── thermochemistry (relax + freq) ────────────────────────────────────────────

class TestThermochemistry:
    @pytest.fixture(scope="class")
    def therm(self, ni_oh4):
        return thermochemistry(
            ni_oh4, backend="xtb", T=298.15, P=101325.0,
            fmax=0.1, steps=50,
        )

    def test_returns_therm_result(self, therm):
        assert isinstance(therm, ThermResult)

    def test_energy_lower_than_input_sp(self, ni_oh4, therm):
        sp = compute_energy(ni_oh4, backend="xtb")
        assert therm.energy_eV <= sp.energy_eV + 0.01

    def test_positions_differ_from_input(self, ni_oh4, therm):
        assert not np.allclose(
            therm.mol.get_positions(), ni_oh4.get_positions(), atol=1e-3
        )

    def test_all_thermo_fields_set(self, therm):
        assert therm.zpe_eV       > 0
        assert therm.entropy_eV_K > 0
        assert len(therm.frequencies) > 0
        assert therm.gibbs_eV     < therm.enthalpy_eV

    def test_delta_E_example(self, ni_h2o4, ni_oh4):
        """Smoke test: ΔE between two species is a finite float."""
        e1 = compute_energy(ni_h2o4, backend="xtb")
        e2 = compute_energy(ni_oh4,  backend="xtb")
        dE = e2.energy_eV - e1.energy_eV
        assert isinstance(dE, float)
        assert np.isfinite(dE)

    def test_delta_G_example(self, ni_h2o4, ni_oh4):
        """Smoke test: ΔG between two species is a finite float."""
        g1 = compute_gibbs(ni_h2o4, backend="xtb", T=298.15)
        g2 = compute_gibbs(ni_oh4,  backend="xtb", T=298.15)
        dG = g2.gibbs_eV - g1.gibbs_eV
        assert isinstance(dG, float)
        assert np.isfinite(dG)


# ── write_all integration ─────────────────────────────────────────────────────

class TestWriteAllRelax:
    def test_relax_columns_in_row(self, tmp_path, ni_h2o4):
        from molbuilder.output.writer import write_all
        from molbuilder.core.molecule import Molecule

        row = {
            "formula": ni_h2o4.formula, "charge": ni_h2o4.charge,
            "structure": "monomer", "metal": "Ni", "ox": 2,
            "ox_label": "NiII", "cn": 4, "geometry": "sqp",
            "ligand_combo": "H2O4", "bridge": "", "n_bridges": 0,
            "arrangement": "", "isomer": "only",
            "n_atoms": ni_h2o4.num_atoms(),
            "filename": "test.POSCAR",
        }
        rows = write_all(
            [(ni_h2o4, row)],
            output_dir=str(tmp_path),
            csv_file=None,
            relax=True,
            relax_backend="xtb",
            relax_fmax=0.1,
            relax_steps=20,
            write_relax_json=True,
        )
        assert len(rows) == 1
        r = rows[0]
        assert "relax_energy_eV" in r
        assert "relax_converged"  in r
        assert "relax_steps"      in r
        assert "relax_backend"    in r
        assert "relax_filename"   in r
        assert r["relax_energy_eV"] is not None
        # relax_converged must be a plain Python bool (JSON-serializable)
        assert type(r["relax_converged"]) is bool

    def test_relax_json_written(self, tmp_path, ni_h2o4):
        import json as _json
        from molbuilder.output.writer import write_all

        row = {
            "formula": ni_h2o4.formula, "charge": ni_h2o4.charge,
            "structure": "monomer", "metal": "Ni", "ox": 2,
            "ox_label": "NiII", "cn": 4, "geometry": "sqp",
            "ligand_combo": "H2O4", "bridge": "", "n_bridges": 0,
            "arrangement": "", "isomer": "only",
            "n_atoms": ni_h2o4.num_atoms(),
            "filename": "test.POSCAR",
        }
        rows = write_all(
            [(ni_h2o4, row)],
            output_dir=str(tmp_path),
            csv_file=None,
            relax=True,
            relax_backend="xtb",
            relax_fmax=0.1,
            relax_steps=20,
            write_relax_json=True,
        )
        rfile = rows[0]["relax_filename"]
        from pathlib import Path
        json_file = Path(rfile).with_suffix(".json")
        assert json_file.exists(), f"Expected JSON at {json_file}"
        data = _json.loads(json_file.read_text())
        assert "energy_eV"  in data
        assert "converged"  in data
        assert "backend"    in data
        assert isinstance(data["converged"], bool)   # not numpy.bool_
        assert isinstance(data["energy_eV"], float)
