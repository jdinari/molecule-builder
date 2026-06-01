"""
generate_ni_complexes.py
========================
Generate all neutral Ni(II)/Ni(III) coordination complexes, optionally
relax them with xTB or MACE, and build an isodesmic reaction network
with ΔG screening.

Workflow
--------
1.  Enumerate all monomers, dimers, and trimers from the ligand pool.
2.  (optional) Relax each structure with xTB or MACE and compute ΔG(T, P).
3.  Structures with broken bonds are written to poscar/broken/ for review.
4.  (optional) Build an isodesmic reaction network and screen by ΔG.
5.  Write CSV, Excel, and a reaction-network CSV for downstream use.

Enable each stage by flipping the boolean flags in the settings below.

Install
-------
    pip install git+https://github.com/jdinari/molecule-builder.git

    # For geometry relaxation and thermochemistry:
    pip install tblite ase          # xTB (recommended for Ni)
    pip install mace-torch ase      # MACE (faster on GPU)
    pip install openpyxl            # Excel output

Run
---
    python generate_ni_complexes.py
"""

from pathlib import Path
from molbuilder import enumerate_complexes, write_all, MULTI_BRIDGE_CASES
from molbuilder.energetics import run_energetics, write_broken_report


# ═══════════════════════════════════════════════════════════════════════════════
# CHEMISTRY SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

METAL       = "Ni"
OX_STATES   = [2, 3]

# Monodentate ligands available for mononuclear complexes
LIGAND_POOL = ["HCOO", "HCOOH", "H2O", "OH"]

# Bidentate chelating ligands (count as CN=2 each)
BI_LIGANDS  = ["HCOO:bi"]

# Bridging ligands for dinuclear and trinuclear complexes
BRIDGE_POOL = ["mu-OH", "mu-HCOO"]


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT PATHS
# ═══════════════════════════════════════════════════════════════════════════════

OUTPUT_DIR      = Path("poscar")
CSV_FILE        = Path("ni_complexes_summary.csv")
EXCEL_FILE      = Path("ni_energetics.xlsx")
REACTIONS_CSV   = Path("ni_reaction_network.csv")
REACTIONS_PLOT  = Path("ni_reaction_network.png")


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: GEOMETRY RELAXATION AND ENERGETICS
# ═══════════════════════════════════════════════════════════════════════════════

# Set COMPUTE_ENERGY = True to relax every structure and compute ΔG(T, P).
COMPUTE_ENERGY = False

# ENERGY_BACKEND options:
#
#   "xtb"       GFN2-xTB via tblite.  Fast (~5 s/structure on CPU), explicit
#               charge and spin — recommended for Ni(II)/Ni(III).
#               Install: pip install tblite ase
#
#   "mace"      MACE-MH-1 universal potential.  Better energetics on GPU
#               (~0.5 s/structure), but charge/spin are implicit only.
#               Install: pip install mace-torch ase
#
#   "xtb+mace"  Hybrid (RECOMMENDED): xTB geometry + frequencies,
#               MACE single-point energy.
#               G_hybrid = E_MACE + (G_xTB - E_xTB)
#               Near-DFT energy quality with xTB-quality geometries.
#               Install: pip install tblite mace-torch ase
#
#   "both"      Run both backends fully and compare.
#
ENERGY_BACKEND = "xtb+mace"

# Set COMPUTE_THERMO = True to calculate vibrational frequencies and ΔG(T, P).
# Set False to run geometry relaxation only and report ΔE.
COMPUTE_THERMO = True

# Temperature and pressure for thermochemical corrections
TEMPERATURE_K = 298.15   # Kelvin
PRESSURE_PA   = 101325.0  # Pascal (1 atm)

# Geometry relaxation convergence settings
RELAX_FMAX  = 0.05   # eV/Å — maximum force component at convergence
RELAX_STEPS = 300    # maximum number of optimisation steps

# Model paths (None = use defaults)
#   XTB_MODEL  : None → GFN2-xTB
#   MACE_MODEL : None → download mace-mh-1 automatically (requires internet).
#                On a cluster without internet, set this to the absolute path
#                of your local .model file, e.g.:
#                MACE_MODEL = "/scratch/yourname/models/mace-mh-1.model"
XTB_MODEL  = None
MACE_MODEL = None

# Device for MACE: "cpu" for local testing, "cuda" on a GPU node
MACE_DEVICE = "cpu"

# Keep bonds constrained during relaxation?
# False (default): bond breaking is reported, not suppressed.
#   If xTB says a ligand departs, the coordination is genuinely strained.
#   Broken structures are flagged and written to poscar/broken/ for review.
# True: forces bonds to stay near their initial length.
#   Use only if you need to preserve a specific coordination motif.
CONSTRAIN_BONDS = False


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: ISODESMIC REACTION NETWORK
# ═══════════════════════════════════════════════════════════════════════════════

# Set COMPUTE_REACTIONS = True to build and screen the reaction network.
# Requires COMPUTE_ENERGY = True for meaningful ΔG values (works without,
# but all reaction ΔG will be None).
COMPUTE_REACTIONS = False

# Reaction types modelled (all isodesmic — both sides have the same charge):
#
#   SUBSTITUTION  HCOOH + [Ni-OH]  →  [Ni-HCOO] + H₂O   (formic acid route)
#                 H₂O   + [Ni-OH]  →  [Ni-H₂O]  + OH     (water exchange)
#   COORDINATION  [Ni(L)_n] + H₂O  →  [Ni(L)_n(H₂O)]    (coordination number increases)
#   ASSOCIATION   2×monomer         →  dimer + n×H₂O
#
# Structures where a bond stretched > 1.35× during relaxation (bond_status
# == BROKEN) are automatically excluded from the network.

# Screen threshold: only report reactions with |ΔG| ≤ this value (eV)
REACTION_MAX_DG = 0.5

# Reaction types to include: None = all, or e.g. ["substitution"]
REACTION_TYPES = None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mols_in_order = []

    def _enumerate():
        """Generator: enumerate all complexes and collect molecule objects."""
        for mol, row in enumerate_complexes(
            metal       = METAL,
            ox_states   = OX_STATES,
            ligand_pool = LIGAND_POOL,
            bridge_pool = BRIDGE_POOL,
            bi_ligands  = BI_LIGANDS,
            nuclearity  = [1, 2, 3],
            arrangements = ["linear", "triangular"],
            cn_range    = (3, 7),
            multi_bridge_cases = MULTI_BRIDGE_CASES,
            include_heteroleptic         = True,
            include_heteroleptic_trimers = True,
            max_terminals_per_metal      = 1,
            output_root = OUTPUT_DIR,
            verbose     = True,
        ):
            mols_in_order.append(mol)
            yield mol, row

    # ── Enumerate and write POSCARs ───────────────────────────────────────────
    rows = write_all(
        _enumerate(),
        output_dir = OUTPUT_DIR,
        csv_file   = None,
        fmt        = "poscar",
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    def count(tag):
        return sum(1 for r in rows if tag in r["structure"])

    n_dimers_sym    = count("dimer") - count("hetero")
    n_dimers_hetero = count("dimer_hetero")
    n_trimers_sym   = count("trimer_") - count("hetero")
    n_trimers_hetero = sum(1 for r in rows if "hetero" in r["structure"] and "trimer" in r["structure"])

    print(f"\n{'='*55}")
    print(f"  Monomers              : {count('monomer')}")
    print(f"  Dimers (symmetric)    : {n_dimers_sym}")
    print(f"  Dimers (heteroleptic) : {n_dimers_hetero}")
    print(f"  Trimers (symmetric)   : {n_trimers_sym}")
    print(f"  Trimers (heteroleptic): {n_trimers_hetero}")
    print(f"  Total                 : {len(rows)}")
    print(f"{'='*55}")

    # ── Stage 1: Energetics ───────────────────────────────────────────────────
    if COMPUTE_ENERGY:
        mol_lookup = {r["filename"]: m for m, r in zip(mols_in_order, rows)}

        rows = run_energetics(
            rows           = rows,
            mols           = mol_lookup,
            backend        = ENERGY_BACKEND,
            compute_thermo = COMPUTE_THERMO,
            T              = TEMPERATURE_K,
            P              = PRESSURE_PA,
            fmax           = RELAX_FMAX,
            steps          = RELAX_STEPS,
            xtb_model      = XTB_MODEL,
            mace_model     = MACE_MODEL,
            mace_device    = MACE_DEVICE,
            constrain_bonds = CONSTRAIN_BONDS,
            output_dir     = OUTPUT_DIR,
            csv_file       = CSV_FILE,
            excel_file     = EXCEL_FILE,
        )

        # Report broken structures (bond stretched > 1.35× during relaxation)
        broken = [
            (mols_in_order[i], rows[i])
            for i, r in enumerate(rows)
            if r.get("bond_status") == "BROKEN"
        ]
        if broken:
            write_broken_report(broken, OUTPUT_DIR, verbose=True)

    else:
        # No energetics — just write the structure CSV
        from molbuilder.output.writer import write_csv
        write_csv(rows, CSV_FILE)
        print(f"  CSV → {CSV_FILE}")

    # ── Stage 2: Reaction network ─────────────────────────────────────────────
    if COMPUTE_REACTIONS:
        from molbuilder.reactions import ReactionNetwork, ReactionType

        print("\nBuilding reaction network …")

        mol_lookup   = {r["filename"]: m for m, r in zip(mols_in_order, rows)}
        mols_and_rows = [(mol_lookup[r["filename"]], r) for r in rows
                         if r.get("filename") in mol_lookup]

        net = ReactionNetwork(mols_and_rows, bond_filter=True, verbose=True)
        print(net.summary())

        # Attach energies: either from run_energetics (already in rows),
        # or compute them now if COMPUTE_ENERGY was False.
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
            # Energies are already in rows — attach them directly to graph nodes.
            for node_id, node_data in net.graph.nodes(data=True):
                if node_data.get("node_type") == "complex":
                    row   = node_data.get("row", {})
                    e_ev  = row.get("relax_energy_eV")
                    g_ev  = row.get("relax_gibbs_eV")
                    if e_ev:
                        net.graph.nodes[node_id]["energy_eV"] = e_ev
                    if g_ev:
                        net.graph.nodes[node_id]["gibbs_eV"]  = g_ev

            # Compute reference molecule energies (HCOOH, H2O, H2) at the
            # same level of theory so that isodesmic cancellation is valid.
            print("  Computing reference molecule energies (HCOOH, H2O, H2) …")

            backend_lower = ENERGY_BACKEND.lower()

            if COMPUTE_THERMO:
                if backend_lower == "xtb+mace":
                    from molbuilder.relaxation import xtb_relax_mace_singlepoint as _ref_fn
                    ref_kwargs = dict(
                        xtb_model     = XTB_MODEL,
                        mace_model    = MACE_MODEL,
                        mace_device   = MACE_DEVICE,
                        compute_thermo = True,
                        T             = TEMPERATURE_K,
                        P             = PRESSURE_PA,
                        fmax          = RELAX_FMAX,
                        steps         = RELAX_STEPS,
                    )
                else:
                    from molbuilder.relaxation import thermochemistry as _ref_fn
                    ref_kwargs = dict(
                        backend = backend_lower,
                        T       = TEMPERATURE_K,
                        P       = PRESSURE_PA,
                        fmax    = RELAX_FMAX,
                        steps   = RELAX_STEPS,
                    )

                for node_id, node_data in net.graph.nodes(data=True):
                    if node_data.get("node_type") == "reference":
                        try:
                            res = _ref_fn(node_data["mol"], **ref_kwargs)
                            net.graph.nodes[node_id].update(
                                energy_eV = float(res.energy_eV),
                                gibbs_eV  = float(res.gibbs_eV),
                                _therm    = res,
                            )
                            print(f"    ref {node_data['formula']:8s}  "
                                  f"E={res.energy_eV:.3f}  G={res.gibbs_eV:.3f} eV")
                        except Exception as exc:
                            import warnings
                            warnings.warn(
                                f"Ref energy failed for {node_data.get('formula', '?')}: {exc}"
                            )
            else:
                from molbuilder.relaxation import compute_energy as _compute_energy
                eff_backend = backend_lower.replace("+mace", "").replace("xtb", "xtb")
                for node_id, node_data in net.graph.nodes(data=True):
                    if node_data.get("node_type") == "reference":
                        try:
                            res = _compute_energy(node_data["mol"], backend=eff_backend)
                            net.graph.nodes[node_id]["energy_eV"] = float(res.energy_eV)
                        except Exception:
                            pass

            net._update_edge_energies()

        # ── Screen reactions ──────────────────────────────────────────────────
        e_or_g = "ΔG" if COMPUTE_THERMO else "ΔE"
        print(f"\nScreening reactions with {e_or_g} ≤ {REACTION_MAX_DG} eV …")

        hits = net.screen(
            max_dE         = REACTION_MAX_DG,
            use_gibbs      = COMPUTE_THERMO,
            reaction_types = REACTION_TYPES,
            require_energy = True,
        )
        print(f"  {len(hits)} reaction(s) with {e_or_g} ≤ {REACTION_MAX_DG} eV:")
        for src, dst, e, val in hits[:20]:
            print(f"    {val:+.3f} eV  {net.reaction_str(src, dst)}")
        if len(hits) > 20:
            print(f"    … and {len(hits) - 20} more")

        # ── Export reaction network ───────────────────────────────────────────
        df = net.to_dataframe()
        df.to_csv(REACTIONS_CSV, index=False)
        print(f"\n  Reaction network CSV → {REACTIONS_CSV}")

        try:
            import matplotlib.pyplot as plt
            fig = net.plot(
                title      = f"Ni reaction network  ({e_or_g})",
                edge_label = "delta_g" if COMPUTE_THERMO else "delta_e",
            )
            fig.savefig(REACTIONS_PLOT, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Reaction network plot → {REACTIONS_PLOT}")
        except Exception as exc:
            print(f"  (plot skipped: {exc})")

        # ── Broken-structure reminder ─────────────────────────────────────────
        if net.broken_structures:
            write_broken_report(net.broken_structures, OUTPUT_DIR, verbose=True)
