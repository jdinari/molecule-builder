"""
generate_ni_complexes.py
========================
Generate all neutral Ni(II)/Ni(III) coordination complexes, optionally relax
with xTB or MACE, and build an isodesmic reaction network with ΔG screening.

Workflow
--------
1.  Enumerate all monomers, dimers, and trimers from the ligand pool.
2.  (optional) Relax each structure with xTB or MACE and compute ΔG(T,P).
3.  Structures with broken bonds are written to poscar/broken/ for review.
4.  (optional) Build an isodesmic reaction network and screen by ΔG.
5.  Write CSV, Excel, and a reaction-network CSV for downstream use.

Flip the boolean flags below to enable each stage.
"""

from pathlib import Path
from molbuilder import enumerate_complexes, write_all, MULTI_BRIDGE_CASES
from molbuilder.energetics import run_energetics, write_broken_report

# ── Chemistry ─────────────────────────────────────────────────────────────────
METAL       = "Ni"
OX_STATES   = [2, 3]
LIGAND_POOL = ["HCOO", "HCOOH", "H2O", "OH"]
BI_LIGANDS  = ["HCOO:bi"]
BRIDGE_POOL = ["mu-OH", "mu-HCOO"]

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR      = Path("poscar")
CSV_FILE        = Path("ni_complexes_summary.csv")
EXCEL_FILE      = Path("ni_energetics.xlsx")
REACTIONS_CSV   = Path("ni_reaction_network.csv")
REACTIONS_PLOT  = Path("ni_reaction_network.png")

# ── Stage 1: Energy computation ───────────────────────────────────────────────
#
# COMPUTE_ENERGY   True  →  relax every structure and compute ΔG(T,P)
#
# ENERGY_BACKEND   "xtb"      GFN2-xTB via tblite (fast, ~0.2–0.5 eV accuracy)
#                             pip install tblite ase
#                  "mace"     MACE-MH-1 universal MLIP (better energies, needs GPU)
#                             pip install mace-torch ase
#                  "xtb+mace" RECOMMENDED HYBRID: xTB geometry + frequencies,
#                             MACE single-point energy, thermal correction transfer.
#                             G_hybrid = E_MACE + (G_xTB - E_xTB)
#                             ~DFT-quality energies with xTB-quality geometries.
#                             pip install tblite mace-torch ase
#                  "both"     Run both backends fully and compare
#
# COMPUTE_THERMO   True  →  freq calculation → ΔG(T,P)  (default: True)
#                  False →  geometry relax only → ΔE
#
# BROKEN STRUCTURES: structures where a bond stretched > 1.35× its initial
# length during xTB relaxation are automatically written to poscar/broken/
# with a review report.  They are excluded from the reaction network.

COMPUTE_ENERGY  = False
COMPUTE_THERMO  = True           # ΔG by default (set False for ΔE only)
ENERGY_BACKEND  = "xtb+mace"        # xTB relax + MACE SP energy (recommended hybrid)
XTB_MODEL       = None           # None → GFN2-xTB
MACE_MODEL      = None           # None → mh-1; or "/path/to/mace-mh-1.model"
MACE_DEVICE     = "cpu"          # "cpu" or "cuda"
CONSTRAIN_BONDS = False
TEMPERATURE_K   = 298.15
PRESSURE_PA     = 101325.0
RELAX_FMAX      = 0.05
RELAX_STEPS     = 300

# ── Stage 2: Reaction network ─────────────────────────────────────────────────
#
# COMPUTE_REACTIONS  True  →  build isodesmic reaction network and screen
#
# The reaction network requires COMPUTE_ENERGY = True to produce meaningful
# ΔG values.  Without energetics it still shows connectivity but all ΔG = None.
#
# Reactions included (all isodesmic — neutral refs only, same total charge):
#   SUBSTITUTION:   HCOOH + [Ni-OH]  →  [Ni-HCOO] + H₂O      (formic acid route)
#                   H₂O   + [Ni-OH]  →  [Ni-H₂O]  + OH        (water exchange)
#   COORDINATION:   [Ni(L)_n] + H₂O  →  [Ni(L)_n(H₂O)]       (CN increases)
#   ASSOCIATION:    2×monomer         →  dimer  + n×H₂O
#
# Structures with bond_status == BROKEN are excluded from the network.
# They appear in broken_structures and are written to poscar/broken/.

COMPUTE_REACTIONS   = False
REACTION_MAX_DG     = 0.5        # eV — screen threshold for "accessible" reactions
REACTION_TYPES      = None       # None = all; or e.g. ["substitution"]

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mols_in_order = []

    def _results():
        for mol, row in enumerate_complexes(
            metal=METAL, ox_states=OX_STATES,
            ligand_pool=LIGAND_POOL, bridge_pool=BRIDGE_POOL,
            bi_ligands=BI_LIGANDS, nuclearity=[1, 2, 3],
            arrangements=["linear", "triangular"], cn_range=(3, 7),
            multi_bridge_cases=MULTI_BRIDGE_CASES,
            include_heteroleptic=True,
            include_heteroleptic_trimers=True,
            max_terminals_per_metal=1,
            output_root=OUTPUT_DIR, verbose=True,
        ):
            mols_in_order.append(mol)
            yield mol, row

    rows = write_all(_results(), output_dir=OUTPUT_DIR,
                     csv_file=None, fmt="poscar")

    n = lambda s: sum(1 for r in rows if s in r["structure"])
    print(f"\n{'='*55}")
    print(f"  Monomers              : {n('monomer')}")
    print(f"  Dimers (symmetric)    : {n('dimer') - n('hetero')}")
    print(f"  Dimers (heteroleptic) : {n('dimer_hetero')}")
    print(f"  Trimers (symmetric)   : {n('trimer_') - n('hetero')}")
    print(f"  Trimers (heteroleptic): {sum(1 for r in rows if 'hetero' in r['structure'] and 'trimer' in r['structure'])}")
    print(f"  Total                 : {len(rows)}")
    print(f"{'='*55}")

    # ── Stage 1: energetics ───────────────────────────────────────────────────
    if COMPUTE_ENERGY:
        mol_lookup = {r["filename"]: m for m, r in zip(mols_in_order, rows)}
        rows = run_energetics(
            rows=rows, mols=mol_lookup,
            backend=ENERGY_BACKEND, compute_thermo=COMPUTE_THERMO,
            T=TEMPERATURE_K, P=PRESSURE_PA,
            fmax=RELAX_FMAX, steps=RELAX_STEPS,
            xtb_model=XTB_MODEL, mace_model=MACE_MODEL,
            mace_device=MACE_DEVICE, constrain_bonds=CONSTRAIN_BONDS,
            output_dir=OUTPUT_DIR, csv_file=CSV_FILE, excel_file=EXCEL_FILE,
        )

        # ── broken-structure report ───────────────────────────────────────────
        broken = [(mols_in_order[i], rows[i])
                  for i, r in enumerate(rows)
                  if r.get("bond_status") == "BROKEN"]
        if broken:
            write_broken_report(broken, OUTPUT_DIR, verbose=True)
    else:
        from molbuilder.output.writer import write_csv
        write_csv(rows, CSV_FILE)
        print(f"  CSV → {CSV_FILE}")

    # ── Stage 2: reaction network ─────────────────────────────────────────────
    if COMPUTE_REACTIONS:
        from molbuilder.reactions import ReactionNetwork, ReactionType

        print(f"\nBuilding reaction network …")
        mol_lookup = {r["filename"]: m for m, r in zip(mols_in_order, rows)}
        mols_and_rows = [(mol_lookup[r["filename"]], r) for r in rows
                         if r.get("filename") in mol_lookup]

        net = ReactionNetwork(mols_and_rows, bond_filter=True, verbose=True)
        print(net.summary())

        # Compute ΔG on every node if not already done via run_energetics
        if not COMPUTE_ENERGY:
            print("\nComputing xTB ΔG for reaction network nodes …")
            net.compute_energies(
                backend        = ENERGY_BACKEND,
                compute_thermo = COMPUTE_THERMO,
                T              = TEMPERATURE_K,
                P              = PRESSURE_PA,
                fmax           = RELAX_FMAX,
                steps          = RELAX_STEPS,
                verbose        = True,
            )
        else:
            # Energies already in rows — attach directly from row data
            for nid, data in net.graph.nodes(data=True):
                if data.get("node_type") == "complex":
                    row = data.get("row", {})
                    e   = row.get("relax_energy_eV")
                    g   = row.get("relax_gibbs_eV")
                    if e: net.graph.nodes[nid]["energy_eV"] = e
                    if g: net.graph.nodes[nid]["gibbs_eV"]  = g
            # Still need reference molecule energies (HCOOH, H2O, H2).
            # Use the same backend as the complexes for consistency.
            # For xtb+mace hybrid: use xtb_relax_mace_singlepoint so that
            # isodesmic cancellation works properly (same level on both sides).
            print("  Computing reference molecule energies (HCOOH, H2O, H2) …")
            _eff_backend = ENERGY_BACKEND.lower()
            if COMPUTE_THERMO:
                if _eff_backend == "xtb+mace":
                    from molbuilder.relaxation import xtb_relax_mace_singlepoint as _ref_fn
                    _ref_kwargs = dict(
                        xtb_model=XTB_MODEL, mace_model=MACE_MODEL,
                        mace_device=MACE_DEVICE, compute_thermo=True,
                        T=TEMPERATURE_K, P=PRESSURE_PA,
                        fmax=RELAX_FMAX, steps=RELAX_STEPS,
                    )
                else:
                    from molbuilder.relaxation import thermochemistry as _ref_fn
                    _ref_kwargs = dict(
                        backend=_eff_backend,
                        T=TEMPERATURE_K, P=PRESSURE_PA,
                        fmax=RELAX_FMAX, steps=RELAX_STEPS,
                    )
                for nid, data in net.graph.nodes(data=True):
                    if data.get("node_type") == "reference":
                        try:
                            res = _ref_fn(data["mol"], **_ref_kwargs)
                            net.graph.nodes[nid].update(
                                energy_eV = float(res.energy_eV),
                                gibbs_eV  = float(res.gibbs_eV),
                                _therm    = res,
                            )
                            print(f"    ref {data['formula']:8s}  "
                                  f"E={res.energy_eV:.3f}  G={res.gibbs_eV:.3f} eV")
                        except Exception as exc:
                            import warnings
                            warnings.warn(f"Ref energy failed for {data.get('formula','?')}: {exc}")
            else:
                from molbuilder.relaxation import compute_energy as _ce
                for nid, data in net.graph.nodes(data=True):
                    if data.get("node_type") == "reference":
                        try:
                            res = _ce(data["mol"], backend=_eff_backend.replace("+mace","").replace("xtb","xtb"))
                            net.graph.nodes[nid]["energy_eV"] = float(res.energy_eV)
                        except Exception: pass
            net._update_edge_energies()

        # ── screen ────────────────────────────────────────────────────────────
        print(f"\nScreening reactions with ΔG ≤ {REACTION_MAX_DG} eV …")
        hits = net.screen(
            max_dE          = REACTION_MAX_DG,
            use_gibbs       = COMPUTE_THERMO,
            reaction_types  = REACTION_TYPES,
            require_energy  = True,
        )
        e_or_g = "ΔG" if COMPUTE_THERMO else "ΔE"
        print(f"  {len(hits)} reaction(s) with {e_or_g} ≤ {REACTION_MAX_DG} eV:")
        for src, dst, e, val in hits[:20]:
            print(f"    {val:+.3f} eV  {net.reaction_str(src, dst)}")
        if len(hits) > 20:
            print(f"    … and {len(hits)-20} more")

        # ── export ────────────────────────────────────────────────────────────
        df = net.to_dataframe()
        df.to_csv(REACTIONS_CSV, index=False)
        print(f"\n  Reaction network CSV → {REACTIONS_CSV}")

        try:
            fig = net.plot(
                title=f"Ni reaction network  ({e_or_g})",
                edge_label="delta_g" if COMPUTE_THERMO else "delta_e",
            )
            fig.savefig(REACTIONS_PLOT, dpi=150, bbox_inches="tight")
            import matplotlib.pyplot as plt; plt.close(fig)
            print(f"  Reaction network plot → {REACTIONS_PLOT}")
        except Exception as exc:
            print(f"  (plot skipped: {exc})")

        # ── broken-structure reminder ─────────────────────────────────────────
        if net.broken_structures:
            write_broken_report(net.broken_structures, OUTPUT_DIR, verbose=True)
