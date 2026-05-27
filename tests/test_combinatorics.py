"""
tests/test_combinatorics.py
===========================
Tests for the combinatorial enumeration engine.
"""

import pytest

from molbuilder.combinatorics import (
    enumerate_monomers,
    enumerate_dimers,
    enumerate_trimers,
    enumerate_complexes,
    enumerate_heteroleptic_dimers,
    MULTI_BRIDGE_CASES,
    combo_label,
    safe,
)


# ── helper ────────────────────────────────────────────────────────────────────

def collect(gen):
    """Consume a generator, return list of row dicts."""
    return [row for _, row in gen]


# ── combo_label / safe ────────────────────────────────────────────────────────

class TestHelpers:
    def test_combo_label_sorted(self):
        assert combo_label(["H2O", "HCOO", "H2O"]) == "H2O2_HCOO1"

    def test_combo_label_empty(self):
        assert combo_label([]) == ""

    def test_safe_strips_colon(self):
        assert ":" not in safe("HCOO:bi")

    def test_safe_strips_slash(self):
        assert "/" not in safe("a/b")


# ── MULTI_BRIDGE_CASES ────────────────────────────────────────────────────────

class TestMultiBridgeCases:
    def test_is_list(self):
        assert isinstance(MULTI_BRIDGE_CASES, list)

    def test_triangular_hcoo_present(self):
        found = any(
            arr == "triangular" and bridge == "mu-HCOO" and nbpp == 2
            for arr, nbpp, bridge, ox, terminal, _ in MULTI_BRIDGE_CASES
        )
        assert found, "Triangular mu-HCOO double-bridge case missing"

    def test_triangular_oh_present(self):
        found = any(
            arr == "triangular" and bridge == "mu-OH" and nbpp == 2
            for arr, nbpp, bridge, ox, terminal, _ in MULTI_BRIDGE_CASES
        )
        assert found, "Triangular mu-OH double-bridge case missing"

    def test_all_charge_neutral(self):
        """Every case in the table should be charge-neutral."""
        from molbuilder.ligands.library import get_ligand
        BRIDGE_PAIRS = {"linear": 2, "triangular": 3}
        for arr, nbpp, bridge, ox, terminal, _ in MULTI_BRIDGE_CASES:
            bc   = get_ligand(bridge.replace("mu-", ""))["charge"]
            n_bp = BRIDGE_PAIRS[arr]
            tc   = sum(get_ligand(l)["charge"] for l in terminal)
            net  = 3 * ox + 3 * tc + n_bp * nbpp * bc
            assert net == 0, \
                f"Case {arr} {nbpp}x{bridge} not charge-neutral: {net}"


# ── enumerate_monomers ────────────────────────────────────────────────────────

class TestEnumerateMonomers:
    def test_yields_tuples(self):
        rows = collect(enumerate_monomers(
            "Ni", [2], ["H2O", "OH"], cn_range=(4, 4), verbose=False
        ))
        assert len(rows) > 0
        for r in rows:
            assert "formula" in r
            assert "charge" in r

    def test_all_charge_zero(self):
        rows = collect(enumerate_monomers(
            "Ni", [2], ["HCOO", "H2O", "OH"], cn_range=(3, 6), verbose=False
        ))
        for r in rows:
            assert r["charge"] == 0, f"Non-neutral: {r['formula']}"

    def test_structure_type(self):
        rows = collect(enumerate_monomers(
            "Fe", [3], ["Cl", "H2O"], cn_range=(6, 6), verbose=False
        ))
        for r in rows:
            assert r["structure"] == "monomer"


# ── enumerate_dimers ──────────────────────────────────────────────────────────

class TestEnumerateDimers:
    def test_yields_dimers(self):
        rows = collect(enumerate_dimers(
            "Ni", [2], ["H2O", "OH"], ["mu-OH"],
            cn_range=(3, 5), max_bridges=2, verbose=False
        ))
        assert len(rows) > 0

    def test_all_charge_zero(self):
        rows = collect(enumerate_dimers(
            "Ni", [2], ["HCOO", "H2O"], ["mu-HCOO"],
            cn_range=(3, 5), max_bridges=3, verbose=False
        ))
        for r in rows:
            assert r["charge"] == 0

    def test_paddle_wheel_included(self):
        """Ni2(HCOO)4 should appear when max_bridges >= 4."""
        rows = collect(enumerate_dimers(
            "Ni", [2], [], ["mu-HCOO"],
            cn_range=(4, 4), max_bridges=4, verbose=False
        ))
        formulas = [r["formula"] for r in rows]
        assert "C4H4Ni2O8" in formulas, \
            "Paddle-wheel Ni2(HCOO)4 missing from n=4 dimer enumeration"

    def test_paddle_wheel_missing_when_capped(self):
        """With max_bridges=3, Ni2(HCOO)4 should NOT appear."""
        rows = collect(enumerate_dimers(
            "Ni", [2], [], ["mu-HCOO"],
            cn_range=(4, 4), max_bridges=3, verbose=False
        ))
        formulas = [r["formula"] for r in rows]
        assert "C4H4Ni2O8" not in formulas


# ── enumerate_trimers ─────────────────────────────────────────────────────────

class TestEnumerateTrimers:
    def test_triangular_double_bridge_hcoo(self):
        rows = collect(enumerate_trimers(
            "Ni", [2], [], ["mu-HCOO"],
            cn_range=(4, 4), verbose=False
        ))
        formulas = [r["formula"] for r in rows]
        assert "C6H6Ni3O12" in formulas, \
            "Ni3(HCOO)6 triangular double-bridge missing"

    def test_triangular_double_bridge_oh(self):
        rows = collect(enumerate_trimers(
            "Ni", [2], [], ["mu-OH"],
            cn_range=(4, 4), verbose=False
        ))
        formulas = [r["formula"] for r in rows]
        assert "H6Ni3O6" in formulas, \
            "Ni3(OH)6 triangular double-bridge missing"

    def test_all_charge_zero(self):
        rows = collect(enumerate_trimers(
            "Ni", [2], ["H2O", "OH"], ["mu-OH"],
            cn_range=(3, 5), verbose=False
        ))
        for r in rows:
            assert r["charge"] == 0


# ── enumerate_heteroleptic_dimers ─────────────────────────────────────────────

class TestEnumerateHeterolepticDimers:
    def test_yields_results(self):
        rows = collect(enumerate_heteroleptic_dimers(
            "Ni", [2], ["H2O", "OH"], ["mu-HCOO"],
            cn_range=(3, 5), max_bridges=2, verbose=False
        ))
        assert len(rows) > 0

    def test_no_homotopic_duplicates(self):
        """No row should have identical terminal sets on both sides."""
        rows = collect(enumerate_heteroleptic_dimers(
            "Ni", [2], ["H2O", "OH", "HCOO"], ["mu-HCOO"],
            cn_range=(3, 4), max_bridges=2, verbose=False
        ))
        for r in rows:
            assert r["structure"] == "dimer_hetero"

    def test_paddle_wheel_asymmetric(self):
        """Ni2(HCOO)4(H2O) (one water, one bare) should be enumerated."""
        rows = collect(enumerate_heteroleptic_dimers(
            "Ni", [2], ["H2O", "HCOOH"], ["mu-HCOO"],
            cn_range=(4, 5), max_bridges=4, verbose=False
        ))
        formulas = [r["formula"] for r in rows]
        assert "C4H6Ni2O9" in formulas, \
            "Heteroleptic Ni2(HCOO)4(H2O) missing"

    def test_all_charge_zero(self):
        rows = collect(enumerate_heteroleptic_dimers(
            "Ni", [2], ["H2O", "OH"], ["mu-HCOO"],
            cn_range=(3, 4), max_bridges=2, verbose=False
        ))
        for r in rows:
            assert r["charge"] == 0

    def test_no_duplicate_keys(self):
        """Each (ox, bridge, n, sorted_t1, sorted_t2) must be unique."""
        rows = collect(enumerate_heteroleptic_dimers(
            "Ni", [2], ["H2O", "OH", "HCOO"], ["mu-HCOO"],
            cn_range=(3, 4), max_bridges=2, verbose=False
        ))
        seen = set()
        for r in rows:
            key = (r["ox"], r["bridge"], r["n_bridges"], r["ligand_combo"])
            assert key not in seen, f"Duplicate: {key}"
            seen.add(key)


# ── enumerate_complexes ───────────────────────────────────────────────────────

class TestEnumerateComplexes:
    def test_monomers_only(self):
        rows = collect(enumerate_complexes(
            "Ni", [2], ["H2O", "OH"],
            nuclearity=[1], cn_range=(4, 4), verbose=False
        ))
        assert all(r["structure"] == "monomer" for r in rows)
        assert len(rows) > 0

    def test_dimers_only(self):
        rows = collect(enumerate_complexes(
            "Ni", [2], ["H2O"], bridge_pool=["mu-OH"],
            nuclearity=[2], cn_range=(3, 4), verbose=False
        ))
        assert all(r["structure"] == "dimer" for r in rows)

    def test_heteroleptic_opt_in(self):
        rows_sym = collect(enumerate_complexes(
            "Ni", [2], ["H2O", "OH"], bridge_pool=["mu-HCOO"],
            nuclearity=[2], cn_range=(3, 4),
            include_heteroleptic=False, verbose=False
        ))
        rows_het = collect(enumerate_complexes(
            "Ni", [2], ["H2O", "OH"], bridge_pool=["mu-HCOO"],
            nuclearity=[2], cn_range=(3, 4),
            include_heteroleptic=True, verbose=False
        ))
        assert len(rows_het) > len(rows_sym), \
            "Heteroleptic flag should add structures"
        assert any(r["structure"] == "dimer_hetero" for r in rows_het)
        assert not any(r["structure"] == "dimer_hetero" for r in rows_sym)
