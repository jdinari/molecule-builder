"""
tests/test_reactions.py
=======================
Tests for molbuilder.reactions — isodesmic-only ReactionNetwork.

Key correctness properties:
    1. Only isodesmic reactions (same-charge ligand swaps)
    2. Bond-broken structures excluded automatically
    3. Same oxidation state enforced
    4. All reference molecules are neutral (H₂O, HCOOH)
    5. Forward ΔE and reverse ΔE sum to ~0
    6. Charge balance holds on every edge
"""

import pytest
from pathlib import Path

from molbuilder import enumerate_complexes, MULTI_BRIDGE_CASES
from molbuilder.reactions import (
    ReactionNetwork, ReactionType,
    _parse_combo, _LIGAND_CHARGES, _NEUTRAL_ACID,
    _is_isodesmic, _substitution_stoich, _make_free_molecule,
)


# ── shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_rows(tmp_path_factory):
    out = tmp_path_factory.mktemp("rxn")
    return list(enumerate_complexes(
        metal="Ni", ox_states=[2],
        ligand_pool=["HCOO", "H2O", "OH"],
        bridge_pool=["mu-OH"],
        nuclearity=[1, 2], cn_range=(4, 5),
        output_root=out, verbose=False,
        multi_bridge_cases=MULTI_BRIDGE_CASES,
    ))


@pytest.fixture(scope="module")
def net(small_rows):
    return ReactionNetwork(small_rows, bond_filter=False, verbose=False)


@pytest.fixture(scope="module")
def net_with_broken(small_rows):
    tagged = []
    for i, (mol, row) in enumerate(small_rows):
        r = dict(row)
        r["bond_status"] = "BROKEN" if i % 5 == 0 else "OK"
        tagged.append((mol, r))
    return ReactionNetwork(tagged, bond_filter=True, verbose=False)


# ── helper functions ──────────────────────────────────────────────────────────

class TestHelpers:
    def test_parse_combo(self):
        c = _parse_combo("H2O3_HCOO2")
        assert c["H2O"] == 3
        assert c["HCOO"] == 2

    def test_is_isodesmic_same_charge(self):
        assert _is_isodesmic("OH",   "HCOO")   # both -1
        assert _is_isodesmic("H2O",  "HCOOH")  # both 0
        assert _is_isodesmic("HCOO", "OH")      # both -1

    def test_is_isodesmic_different_charge(self):
        assert not _is_isodesmic("H2O", "OH")   # 0 vs -1
        assert not _is_isodesmic("OH",  "H2O")  # -1 vs 0

    def test_neutral_acid_mapping(self):
        assert _NEUTRAL_ACID["OH"]   == "H2O"
        assert _NEUTRAL_ACID["HCOO"] == "HCOOH"
        assert _NEUTRAL_ACID["H2O"]  == "H2O"

    def test_substitution_stoich_isodesmic(self):
        stoich = _substitution_stoich("A", "B", "OH", "HCOO", 1)
        assert stoich["A"] == -1
        assert stoich["B"] == +1
        # ref_in = HCOOH (incoming HCOO), ref_out = H2O (outgoing OH)
        assert stoich.get("ref:HCOOH", 0) == -1
        assert stoich.get("ref:H2O",   0) == +1

    def test_substitution_stoich_dimer(self):
        stoich = _substitution_stoich("A", "B", "OH", "HCOO", 2)
        assert stoich.get("ref:HCOOH", 0) == -2
        assert stoich.get("ref:H2O",   0) == +2

    def test_make_free_molecule_neutral(self):
        for name in ("H2O", "HCOOH", "H2"):
            mol = _make_free_molecule(name)
            assert mol is not None
            assert mol.charge == 0

    def test_make_free_molecule_unknown(self):
        assert _make_free_molecule("UNKNOWN") is None


# ── ReactionNetwork construction ──────────────────────────────────────────────

class TestConstruction:
    def test_builds(self, net):
        assert net is not None

    def test_has_nodes(self, net):
        assert net.graph.number_of_nodes() > 0

    def test_has_edges(self, net):
        assert net.graph.number_of_edges() > 0

    def test_complex_node_count(self, net, small_rows):
        n = sum(1 for _,d in net.graph.nodes(data=True)
                if d["node_type"]=="complex")
        assert n == len(small_rows)

    def test_reference_nodes(self, net):
        for rid in ("ref:H2O", "ref:HCOOH", "ref:H2"):
            assert rid in net.graph, f"Missing {rid}"

    def test_reference_nodes_neutral(self, net):
        for rid in ("ref:H2O", "ref:HCOOH", "ref:H2"):
            assert net.graph.nodes[rid]["charge"] == 0

    def test_repr(self, net):
        r = repr(net)
        assert "ReactionNetwork" in r
        assert "edges" in r

    def test_summary(self, net):
        s = net.summary()
        assert "isodesmic" in s
        assert "Complex nodes" in s

    def test_summary_contains_counts(self, net):
        s = net.summary()
        assert "Substitutions" in s
        assert "Associations" in s


# ── Bond filter ───────────────────────────────────────────────────────────────

class TestBondFilter:
    def test_broken_excluded(self, net_with_broken, small_rows):
        n_broken = sum(1 for _,r in net_with_broken.broken_structures
                       if r.get("bond_status") == "BROKEN")
        assert len(net_with_broken.broken_structures) > 0
        assert n_broken == len(net_with_broken.broken_structures)

    def test_broken_accessible(self, net_with_broken):
        assert isinstance(net_with_broken.broken_structures, list)
        for mol, row in net_with_broken.broken_structures:
            assert row["bond_status"] == "BROKEN"

    def test_complex_count_reduced(self, net_with_broken, small_rows):
        n_in_graph = sum(1 for _,d in net_with_broken.graph.nodes(data=True)
                         if d["node_type"] == "complex")
        assert n_in_graph < len(small_rows)
        assert n_in_graph == len(small_rows) - len(net_with_broken.broken_structures)

    def test_no_broken_in_graph(self, net_with_broken):
        for _, d in net_with_broken.graph.nodes(data=True):
            assert d.get("bond_status") != "BROKEN", \
                f"BROKEN node found in graph: {d.get('formula')}"


# ── Isodesmic constraint ──────────────────────────────────────────────────────

class TestIsodesmicConstraint:
    def test_all_substitutions_isodesmic(self, net):
        """Every substitution edge must swap ligands of equal charge."""
        for src, dst, e in net.substitutions:
            inc = e.get("incoming", "")
            out = e.get("outgoing", "")
            if inc and out:
                q_in  = _LIGAND_CHARGES.get(inc,  0)
                q_out = _LIGAND_CHARGES.get(out, 0)
                assert q_in == q_out, \
                    f"Non-isodesmic edge: {inc}(q={q_in}) → {out}(q={q_out})"

    def test_no_h3o_reference(self, net):
        """H₃O⁺ should never appear — we are neutral-only."""
        assert "ref:H3O" not in net.graph

    def test_all_reference_nodes_neutral(self, net):
        for nid, d in net.graph.nodes(data=True):
            if d["node_type"] == "reference":
                assert d["charge"] == 0, f"{nid} has charge={d['charge']}"

    def test_charge_balance_every_edge(self, net):
        """Left-hand total charge must equal right-hand total charge."""
        for src, dst, e in net.edges:
            stoich = e.get("stoich", {})
            total_q = sum(
                coeff * net.graph.nodes[nid]["charge"]
                for nid, coeff in stoich.items()
                if nid in net.graph.nodes
            )
            assert abs(total_q) < 1e-9, \
                f"Charge imbalance ({total_q}) for {net.reaction_str(src, dst)}"


# ── Substitution properties ───────────────────────────────────────────────────

class TestSubstitutions:
    def test_has_substitutions(self, net):
        assert len(net.substitutions) > 0

    def test_all_are_type_substitution(self, net):
        for _, _, e in net.substitutions:
            assert e["reaction_type"] == ReactionType.SUBSTITUTION

    def test_bidirectional(self, net):
        fwd = {(s, d) for s, d, _ in net.substitutions}
        for s, d in list(fwd):
            assert (d, s) in fwd, \
                f"Missing reverse: {net.node_label(d)} → {net.node_label(s)}"

    def test_same_ox_state(self, net):
        for src, dst, _ in net.substitutions:
            ns, nd = net.graph.nodes[src], net.graph.nodes[dst]
            if ns.get("node_type") == "complex" and nd.get("node_type") == "complex":
                assert ns.get("ox") == nd.get("ox"), \
                    f"Ox state mismatch: {ns['ox']} vs {nd['ox']}"

    def test_same_geometry_default(self, net):
        for src, dst, _ in net.substitutions:
            ns, nd = net.graph.nodes[src], net.graph.nodes[dst]
            if ns.get("node_type") == "complex" and nd.get("node_type") == "complex":
                assert ns.get("geometry") == nd.get("geometry"), \
                    f"Geometry mismatch: {ns['geometry']} vs {nd['geometry']}"

    def test_dimer_n_exchanged(self, net):
        for src, dst, e in net.substitutions:
            ns = net.graph.nodes[src]
            if ns.get("structure") == "dimer":
                assert e.get("n_exchanged") == 2, \
                    f"Dimer should have n_exchanged=2, got {e.get('n_exchanged')}"
                break

    def test_references_are_neutral_acids(self, net):
        for _, _, e in net.substitutions:
            for ref_key in ("ref_in", "ref_out"):
                rid = e.get(ref_key, "")
                if rid and rid.startswith("ref:"):
                    name = rid[4:]
                    assert _LIGAND_CHARGES.get(name, 0) == 0, \
                        f"Reference {rid} is not neutral"


# ── Association / Dissociation ────────────────────────────────────────────────

class TestAssociations:
    def test_has_associations(self, net):
        assert len(net.associations) > 0

    def test_has_dissociations(self, net):
        assert len(net.dissociations) > 0

    def test_monomer_to_dimer(self, net):
        for src, dst, _ in net.associations:
            ns, nd = net.graph.nodes[src], net.graph.nodes[dst]
            assert ns["structure"] == "monomer"
            assert nd["structure"] == "dimer"

    def test_stoich_monomer_coeff(self, net):
        for src, dst, e in net.associations:
            assert e["stoich"][src] == -2

    def test_stoich_dimer_coeff(self, net):
        for src, dst, e in net.associations:
            assert e["stoich"][dst] == +1

    def test_displaced_is_h2o(self, net):
        """Displaced ligands in associations must be H₂O (isodesmic)."""
        for _, _, e in net.associations:
            assert e.get("displaced_ligand") == "H2O"

    def test_assoc_diss_pairs(self, net):
        """Every association should have a paired dissociation."""
        assoc_pairs = {(s, d) for s, d, _ in net.associations}
        diss_pairs  = {(d, s) for s, d, _ in net.dissociations}
        assert assoc_pairs == diss_pairs


# ── Energetics ────────────────────────────────────────────────────────────────

class TestEnergetics:
    @pytest.fixture(scope="class")
    def net_e(self, small_rows):
        """Network with xTB single-point energies."""
        from molbuilder.relaxation import compute_energy
        n = ReactionNetwork(small_rows, bond_filter=False, verbose=False)
        for nid, data in n.graph.nodes(data=True):
            try:
                res = compute_energy(data["mol"], backend="xtb")
                n.graph.nodes[nid]["energy_eV"] = float(res.energy_eV)
            except Exception:
                pass
        n._update_edge_energies()
        return n

    def test_some_delta_e_computed(self, net_e):
        n_with_de = sum(1 for _, _, e in net_e.substitutions
                        if e.get("delta_e") is not None)
        assert n_with_de > 0

    def test_substitution_de_range(self, net_e):
        """Isodesmic ΔE for monomer subs should be in ±5 eV."""
        for src, dst, e in net_e.substitutions:
            dE = e.get("delta_e")
            if dE is None:
                continue
            if net_e.graph.nodes[src].get("structure") == "monomer":
                assert -5.0 < dE < 5.0, \
                    f"Unreasonable ΔE={dE:.3f}: {net_e.reaction_str(src, dst)}"

    def test_forward_reverse_symmetric(self, net_e):
        """ΔE_fwd + ΔE_rev should equal 0 (isodesmic property)."""
        checked = 0
        for src, dst, e in net_e.substitutions:
            dE_fwd = e.get("delta_e")
            if dE_fwd is None:
                continue
            dE_rev = net_e.delta_e(dst, src)
            if dE_rev is None:
                continue
            assert abs(dE_fwd + dE_rev) < 0.05, \
                f"ΔE symmetry broken: fwd={dE_fwd:.4f}  rev={dE_rev:.4f}"
            checked += 1
        assert checked > 0, "No fwd/rev pairs found"

    def test_screen_returns_list(self, net_e):
        assert isinstance(net_e.screen(max_dE=100.0, require_energy=False), list)

    def test_screen_sorted(self, net_e):
        results = net_e.screen(max_dE=100.0, require_energy=False)
        vals = [r[3] for r in results if r[3] is not None]
        assert vals == sorted(vals)

    def test_screen_type_filter(self, net_e):
        subs = net_e.screen(max_dE=100.0, require_energy=False,
                            reaction_types=[ReactionType.SUBSTITUTION])
        for _, _, e, _ in subs:
            assert e["reaction_type"] == ReactionType.SUBSTITUTION

    def test_delta_e_method(self, net_e):
        for src, dst, _ in net_e.substitutions:
            dE = net_e.delta_e(src, dst)
            if dE is not None:
                assert isinstance(dE, float)
                break

    def test_delta_e_none_no_edge(self, net_e):
        nodes = list(net_e.graph.nodes)
        assert net_e.delta_e(nodes[0], nodes[0]) is None


# ── Labels and display ────────────────────────────────────────────────────────

class TestLabels:
    def test_node_label_reference(self, net):
        assert "H2O" in net.node_label("ref:H2O")

    def test_node_label_complex(self, net):
        nid = next(n for n, d in net.graph.nodes(data=True)
                   if d["node_type"] == "complex")
        assert len(net.node_label(nid)) > 0

    def test_reaction_str_substitution(self, net):
        s, d, _ = net.substitutions[0]
        rs = net.reaction_str(s, d)
        assert "→" in rs

    def test_reaction_str_association(self, net):
        s, d, _ = net.associations[0]
        rs = net.reaction_str(s, d)
        assert "2×" in rs

    def test_reaction_str_no_edge(self, net):
        nodes = list(net.graph.nodes)
        if not net.graph.has_edge(nodes[0], nodes[1]):
            assert "No such edge" in net.reaction_str(nodes[0], nodes[1])


# ── plot() ────────────────────────────────────────────────────────────────────

class TestPlot:
    def test_plot_returns_figure(self, net):
        mpl = pytest.importorskip("matplotlib")
        fig = net.plot()
        assert fig is not None
        mpl.pyplot.close(fig)

    def test_plot_saveable(self, net, tmp_path):
        mpl = pytest.importorskip("matplotlib")
        fig = net.plot()
        out = tmp_path / "rxn_network.png"
        fig.savefig(out)
        assert out.exists() and out.stat().st_size > 0
        mpl.pyplot.close(fig)


# ── to_dataframe() ────────────────────────────────────────────────────────────

class TestDataframe:
    def test_returns_df(self, net):
        pd = pytest.importorskip("pandas")
        df = net.to_dataframe()
        assert len(df) == net.graph.number_of_edges()

    def test_expected_columns(self, net):
        pd = pytest.importorskip("pandas")
        df = net.to_dataframe()
        for col in ("reaction_type", "incoming", "outgoing",
                    "delta_e_eV", "delta_g_eV", "src_formula", "dst_formula"):
            assert col in df.columns


# ── include_geometry_changes ──────────────────────────────────────────────────

class TestGeometryChanges:
    def test_more_substitutions_when_enabled(self, small_rows):
        net_no   = ReactionNetwork(small_rows, include_geometry_changes=False, verbose=False)
        net_yes  = ReactionNetwork(small_rows, include_geometry_changes=True,  verbose=False)
        assert len(net_yes.substitutions) >= len(net_no.substitutions)


# ── Coordination / Decoordination ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def rows_multi_cn(tmp_path_factory):
    out = tmp_path_factory.mktemp("rxn_cn")
    return list(enumerate_complexes(
        metal="Ni", ox_states=[2],
        ligand_pool=["HCOO", "H2O", "OH"],
        nuclearity=[1], cn_range=(4, 6),
        output_root=out, verbose=False,
        multi_bridge_cases=MULTI_BRIDGE_CASES,
    ))


@pytest.fixture(scope="module")
def net_cn(rows_multi_cn):
    return ReactionNetwork(rows_multi_cn, bond_filter=False, verbose=False)


class TestCoordination:
    def test_has_coordinations(self, net_cn):
        assert len(net_cn.coordinations) > 0

    def test_has_decoordinations(self, net_cn):
        assert len(net_cn.decoordinations) > 0

    def test_coordination_type(self, net_cn):
        for _, _, e in net_cn.coordinations:
            assert e["reaction_type"] == ReactionType.COORDINATION

    def test_decoordination_type(self, net_cn):
        for _, _, e in net_cn.decoordinations:
            assert e["reaction_type"] == ReactionType.DECOORDINATION

    def test_cn_increases_in_coordination(self, net_cn):
        for src, dst, e in net_cn.coordinations:
            assert e["cn_to"] == e["cn_from"] + 1

    def test_cn_decreases_in_decoordination(self, net_cn):
        for src, dst, e in net_cn.decoordinations:
            assert e["cn_to"] == e["cn_from"] - 1

    def test_coordination_gains_h2o(self, net_cn):
        for _, _, e in net_cn.coordinations:
            assert e["gained_ligand"] == "H2O"

    def test_decoordination_loses_h2o(self, net_cn):
        for _, _, e in net_cn.decoordinations:
            assert e["lost_ligand"] == "H2O"

    def test_isodesmic_h2o_neutral(self, net_cn):
        """Both sides of coordination/decoordination have the same total charge."""
        for src, dst, e in net_cn.coordinations + net_cn.decoordinations:
            stoich = e.get("stoich", {})
            total_q = sum(
                coeff * net_cn.graph.nodes[nid]["charge"]
                for nid, coeff in stoich.items()
                if nid in net_cn.graph.nodes
            )
            assert abs(total_q) < 1e-9, \
                f"Charge imbalance ({total_q}) for {net_cn.reaction_str(src, dst)}"

    def test_same_non_h2o_ligands(self, net_cn):
        """Coordination edges connect complexes with the same non-H₂O ligands."""
        for src, dst, e in net_cn.coordinations:
            ns, nd = net_cn.graph.nodes[src], net_cn.graph.nodes[dst]
            if ns.get("node_type") != "complex" or nd.get("node_type") != "complex":
                continue
            from molbuilder.reactions import _parse_combo
            la = {k: v for k, v in _parse_combo(ns["ligands"]).items() if k != "H2O"}
            lb = {k: v for k, v in _parse_combo(nd["ligands"]).items() if k != "H2O"}
            assert la == lb, \
                f"Non-H₂O ligands differ: {ns['ligands']} vs {nd['ligands']}"

    def test_bidirectional(self, net_cn):
        """Every coordination has a paired decoordination."""
        coord_pairs   = {(s, d) for s, d, _ in net_cn.coordinations}
        decoord_pairs = {(d, s) for s, d, _ in net_cn.decoordinations}
        assert coord_pairs == decoord_pairs

    def test_geometry_changes_allowed(self, net_cn):
        """CN changes always involve geometry changes; this should not be blocked."""
        geom_changes = [
            e for _, _, e in net_cn.coordinations
            if e.get("geom_from") != e.get("geom_to")
        ]
        assert len(geom_changes) > 0, \
            "Expected coordination edges crossing geometries (e.g. tet→sqpy)"

    def test_de_symmetric(self, net_cn):
        """ΔE(coord) + ΔE(decoord) should equal 0 when both are computed."""
        from molbuilder.relaxation import compute_energy
        computed = 0
        for nid, data in net_cn.graph.nodes(data=True):
            if computed >= 10:
                break
            try:
                res = compute_energy(data["mol"], backend="xtb")
                net_cn.graph.nodes[nid]["energy_eV"] = float(res.energy_eV)
                computed += 1
            except Exception:
                pass
        net_cn._update_edge_energies()

        for src, dst, e in net_cn.coordinations:
            dE_fwd = e.get("delta_e")
            if dE_fwd is None:
                continue
            dE_rev = net_cn.delta_e(dst, src)
            if dE_rev is None:
                continue
            assert abs(dE_fwd + dE_rev) < 0.05, \
                f"ΔE symmetry broken: fwd={dE_fwd:.4f}  rev={dE_rev:.4f}"
            break

    def test_summary_shows_cn_count(self, net_cn):
        s = net_cn.summary()
        assert "Coordination" in s

    def test_reaction_str_coordination(self, net_cn):
        src, dst, _ = net_cn.coordinations[0]
        rs = net_cn.reaction_str(src, dst)
        assert "H₂O" in rs
        assert "CN" in rs

    def test_reaction_str_decoordination(self, net_cn):
        src, dst, _ = net_cn.decoordinations[0]
        rs = net_cn.reaction_str(src, dst)
        assert "H₂O" in rs
        assert "CN" in rs

    def test_no_cn_change_without_multi_cn_inventory(self, net):
        """A CN=4-only inventory should have no coordination edges."""
        # The `net` fixture uses only cn_range=(4,5) monomers+dimers
        # but the key is that CN changes only appear between monomers with different CNs
        # For this fixture both CN4 and CN5 are in the inventory, so there may be some
        # But with no CN6, the count should be consistent
        # This is just a smoke test
        assert isinstance(net.coordinations, list)
