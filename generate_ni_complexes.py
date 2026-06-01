"""
generate_ni_complexes.py
========================
Generate all neutral Ni(II)/Ni(III) coordination complexes, optionally
relax them and compute DeltaG, then screen an isodesmic reaction network.

Stages (enable by setting the flags below):
  0. Enumerate all monomers, dimers, and trimers -> POSCARs + CSV
  1. Relax every structure and compute DeltaG(T, P) -> Excel workbook
  2. (optional) Keep only the lowest-energy isomer per stoichiometry
  3. Build isodesmic reaction network and screen by DeltaG -> CSV + plot

Each stage saves its results to disk.  Stages can be run independently:

    Run everything at once:
        Set all flags to True, run once.

    Run in separate jobs (e.g. on a cluster):
        Job 1: COMPUTE_ENERGY = False, COMPUTE_REACTIONS = False  -> POSCARs
        Job 2: COMPUTE_ENERGY = True,  COMPUTE_REACTIONS = False  -> energetics
        Job 3: COMPUTE_ENERGY = False, COMPUTE_REACTIONS = True   -> network
               (reads rows from CSV_FILE and mols from MOLS_CACHE automatically)

Install:
    pip install git+https://github.com/jdinari/molecule-builder.git
    pip install tblite ase          # xTB backend
    pip install mace-torch ase      # MACE backend
    pip install openpyxl            # Excel output

Run:
    python generate_ni_complexes.py
"""

import csv
import pickle
import warnings
from pathlib import Path
from molbuilder import enumerate_complexes, write_all, MULTI_BRIDGE_CASES
from molbuilder.energetics import run_energetics, write_broken_report


# ==============================================================================
# SETTINGS  --  edit these before running
# ==============================================================================

# --- Chemistry ----------------------------------------------------------------
METAL       = "Ni"
OX_STATES   = [2, 3]
LIGAND_POOL = ["HCOO", "HCOOH", "H2O", "OH"]   # monodentate
BI_LIGANDS  = ["HCOO:bi"]                        # bidentate chelating
BRIDGE_POOL = ["mu-OH", "mu-HCOO"]              # bridging (dimers/trimers)

# --- Output paths -------------------------------------------------------------
OUTPUT_DIR     = Path("poscar")
CSV_FILE       = Path("ni_complexes_summary.csv")
EXCEL_FILE     = Path("ni_energetics.xlsx")
REACTIONS_CSV  = Path("ni_reaction_network.csv")
REACTIONS_PLOT = Path("ni_reaction_network.png")
MOLS_CACHE     = Path("ni_mols_cache.pkl")      # molecule objects saved here

# --- Stage flags --------------------------------------------------------------
COMPUTE_ENERGY    = False   # Stage 1: relax structures and compute DeltaE / DeltaG
BEST_ISOMER_ONLY  = False   # Stage 2: keep only lowest-energy isomer per stoichiometry
COMPUTE_REACTIONS = False   # Stage 3: build isodesmic reaction network

# --- Energy backend -----------------------------------------------------------
# "xtb"      -- GFN2-xTB, explicit charge/spin, ~5 s/structure on CPU
#               recommended for Ni(II)/Ni(III)
# "mace"     -- MACE-MH-1, better energetics, ~0.5 s/structure on GPU
# "xtb+mace" -- xTB geometry + MACE single-point (best quality, needs both)
#               requires COMPUTE_ENERGY = True; reaction network falls back to xTB
ENERGY_BACKEND = "xtb+mace"

# --- Thermochemistry ----------------------------------------------------------
COMPUTE_THERMO = True      # True -> DeltaG(T,P);  False -> DeltaE only
TEMPERATURE_K  = 298.15
PRESSURE_PA    = 101325.0

# --- Relaxation ---------------------------------------------------------------
RELAX_FMAX      = 0.05    # eV/Angstrom convergence threshold
RELAX_STEPS     = 300     # max optimisation steps
CONSTRAIN_BONDS = False   # True -> suppress bond breaking (not recommended)

# --- Model paths (None = use built-in defaults) -------------------------------
# On a cluster, set MACE_MODEL to your local model file path.
# Find it by running:  from mace.calculators import mace_mp; mace_mp("mh-1")
# and reading the printed cache path.
XTB_MODEL   = None
MACE_MODEL  = None
MACE_DEVICE = "cpu"   # "cuda" on a GPU node

# --- Reaction network ---------------------------------------------------------
REACTION_MAX_DG = 0.5    # only report reactions with |DeltaG| <= this (eV)
REACTION_TYPES  = None   # None = all types; or e.g. ["substitution"]


# ==============================================================================
# HELPERS
# ==============================================================================

def _load_rows_from_csv(csv_file):
    """Load rows from a previously written CSV, coercing numeric columns."""
    FLOAT_COLS = {
        "relax_energy_eV", "relax_mace_energy_eV", "relax_gibbs_eV",
        "relax_zpe_eV", "relax_enthalpy_eV", "relax_entropy_eV_K",
        "relax_T_K", "relax_P_Pa", "bond_max_elongation",
        "relax_dE_mace_xtb_eV", "relax_mace_gibbs_eV",
    }
    INT_COLS = {"ox", "cn", "spin_multiplicity", "relax_steps", "bond_n_broken"}
    BOOL_COLS = {"relax_converged"}

    rows = []
    with open(csv_file, newline="") as f:
        for row in csv.DictReader(f):
            for col in FLOAT_COLS:
                if col in row and row[col] not in ("", "None", None):
                    try:
                        row[col] = float(row[col])
                    except ValueError:
                        row[col] = None
                else:
                    row[col] = None
            for col in INT_COLS:
                if col in row and row[col] not in ("", "None", None):
                    try:
                        row[col] = int(float(row[col]))
                    except ValueError:
                        pass
            for col in BOOL_COLS:
                if col in row:
                    row[col] = row[col] in ("True", "true", "1")
            rows.append(row)
    return rows


def _save_mols_cache(mols_in_order, rows, cache_file):
    """Save {filename -> Molecule} mapping to a pickle cache."""
    mol_lookup = {r["filename"]: m for m, r in zip(mols_in_order, rows)}
    with open(cache_file, "wb") as f:
        pickle.dump(mol_lookup, f)
    print(f"  Molecule cache -> {cache_file}")


def _load_mols_cache(cache_file):
    """Load {filename -> Molecule} from pickle cache."""
    with open(cache_file, "rb") as f:
        return pickle.load(f)


def _best_energy(row):
    """Return the best available energy for ranking: MACE > xTB > None."""
    mace_e = row.get("relax_mace_energy_eV")
    xtb_e  = row.get("relax_energy_eV")
    # Prefer MACE energy (higher quality); fall back to xTB
    if mace_e is not None:
        return mace_e
    if xtb_e is not None:
        return xtb_e
    return None


def _filter_best_isomers(rows):
    """Keep only the lowest-energy isomer per unique stoichiometry group.

    Ranks by MACE energy when available (xtb+mace backend), otherwise by
    xTB energy.  Groups with no energy data are kept in full.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for i, r in enumerate(rows):
        key = (r.get("metal"), r.get("ox"), r.get("cn"),
               r.get("geometry"), r.get("ligand_combo"), r.get("structure"))
        groups[key].append((i, r))

    keep = set()
    for key, members in groups.items():
        scored = [(i, _best_energy(r))
                  for i, r in members if _best_energy(r) is not None]
        if scored:
            keep.add(min(scored, key=lambda x: x[1])[0])
        else:
            for i, _ in members:
                keep.add(i)

    filtered = [r for i, r in enumerate(rows) if i in keep]
    n_dropped = len(rows) - len(filtered)
    if n_dropped:
        print(f"  Best-isomer filter: kept {len(filtered)} / {len(rows)} "
              f"({n_dropped} higher-energy isomers removed)")
    return filtered


def _attach_energies_from_rows(net, rows):
    """Attach pre-computed energies from rows into reaction network graph nodes."""
    for node_id, node_data in net.graph.nodes(data=True):
        if node_data.get("node_type") != "complex":
            continue
        row  = node_data.get("row", {})
        e_ev = row.get("relax_energy_eV")
        g_ev = row.get("relax_gibbs_eV")
        if e_ev is not None:
            net.graph.nodes[node_id]["energy_eV"] = e_ev
        if g_ev is not None:
            net.graph.nodes[node_id]["gibbs_eV"] = g_ev

    # Reference molecules (H2O, HCOOH, H2) at the same level of theory
    print("  Computing reference molecule energies (H2O, HCOOH, H2) ...")
    backend = ENERGY_BACKEND.lower()
    if COMPUTE_THERMO:
        if backend == "xtb+mace":
            from molbuilder.relaxation import xtb_relax_mace_singlepoint as _fn
            kwargs = dict(xtb_model=XTB_MODEL, mace_model=MACE_MODEL,
                          mace_device=MACE_DEVICE, compute_thermo=True,
                          T=TEMPERATURE_K, P=PRESSURE_PA,
                          fmax=RELAX_FMAX, steps=RELAX_STEPS)
        else:
            from molbuilder.relaxation import thermochemistry as _fn
            kwargs = dict(backend=backend, T=TEMPERATURE_K, P=PRESSURE_PA,
                          fmax=RELAX_FMAX, steps=RELAX_STEPS)
    else:
        from molbuilder.relaxation import compute_energy as _fn
        kwargs = dict(backend=backend.replace("+mace", ""))

    for node_id, node_data in net.graph.nodes(data=True):
        if node_data.get("node_type") != "reference":
            continue
        try:
            res = _fn(node_data["mol"], **kwargs)
            update = {"energy_eV": float(res.energy_eV)}
            if COMPUTE_THERMO:
                update["gibbs_eV"] = float(res.gibbs_eV)
                update["_therm"]   = res
            net.graph.nodes[node_id].update(update)
            g_str = f"  G={res.gibbs_eV:.3f}" if COMPUTE_THERMO else ""
            print(f"    {node_data['formula']:8s}  E={res.energy_eV:.3f}{g_str} eV")
        except Exception as exc:
            warnings.warn(f"Ref energy failed for {node_data.get('formula', '?')}: {exc}")

    net._update_edge_energies()


def _run_network_energies(net):
    """Compute all energies fresh via net.compute_energies() (no Stage 1 needed)."""
    backend = ENERGY_BACKEND.lower()
    if backend == "xtb+mace":
        backend = "xtb"
        print("  Note: xtb+mace hybrid requires COMPUTE_ENERGY = True.")
        print("        Using pure xTB for reaction network energies.")
    net.compute_energies(
        backend=backend, compute_thermo=COMPUTE_THERMO,
        T=TEMPERATURE_K, P=PRESSURE_PA,
        fmax=RELAX_FMAX, steps=RELAX_STEPS, verbose=True,
    )


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":

    # --------------------------------------------------------------------------
    # Stage 0: Enumerate structures
    #
    # Always runs. Re-enumerating is fast (no QM); it just rebuilds the
    # template geometries and writes POSCARs.  Previously written POSCARs are
    # overwritten, but this is safe -- they are deterministic.
    # --------------------------------------------------------------------------
    mols_in_order = []

    def _gen():
        for mol, row in enumerate_complexes(
            metal=METAL, ox_states=OX_STATES,
            ligand_pool=LIGAND_POOL, bridge_pool=BRIDGE_POOL,
            bi_ligands=BI_LIGANDS,
            nuclearity=[1, 2, 3],
            arrangements=["linear", "triangular"],
            cn_range=(3, 7),
            multi_bridge_cases=MULTI_BRIDGE_CASES,
            include_heteroleptic=True,
            include_heteroleptic_trimers=True,
            max_terminals_per_metal=1,
            output_root=OUTPUT_DIR,
            verbose=True,
        ):
            mols_in_order.append(mol)
            yield mol, row

    rows = write_all(_gen(), output_dir=OUTPUT_DIR, csv_file=None, fmt="poscar")

    n_mono = sum(1 for r in rows if "monomer" in r["structure"])
    n_dim  = sum(1 for r in rows if "dimer"   in r["structure"])
    n_tri  = sum(1 for r in rows if "trimer"  in r["structure"])
    print(f"\n{'='*45}")
    print(f"  Monomers : {n_mono}")
    print(f"  Dimers   : {n_dim}")
    print(f"  Trimers  : {n_tri}")
    print(f"  Total    : {len(rows)}")
    print(f"{'='*45}\n")

    # Save molecule objects to cache so later stages can reload them
    _save_mols_cache(mols_in_order, rows, MOLS_CACHE)

    # --------------------------------------------------------------------------
    # Stage 1: Energetics
    #
    # Relaxes every structure and writes CSV + Excel.
    # If this stage was run in a previous job, its results are loaded from
    # CSV_FILE so Stage 3 can use them without re-running the QM.
    # --------------------------------------------------------------------------
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
        broken = [(mols_in_order[i], rows[i])
                  for i, r in enumerate(rows) if r.get("bond_status") == "BROKEN"]
        if broken:
            write_broken_report(broken, OUTPUT_DIR, verbose=True)

    elif CSV_FILE.exists():
        # Stage 1 was run in a previous job -- reload its results
        print(f"  Loading energetics from {CSV_FILE} ...")
        rows = _load_rows_from_csv(CSV_FILE)
        print(f"  Loaded {len(rows)} rows")
    else:
        # No energetics yet -- write the plain structure CSV
        from molbuilder.output.writer import write_csv
        write_csv(rows, CSV_FILE)
        print(f"  CSV -> {CSV_FILE}")

    # --------------------------------------------------------------------------
    # Stage 2: Best-isomer filter
    #
    # Keeps only the lowest-energy isomer per stoichiometry.
    # Only meaningful after Stage 1 has provided energies.
    # --------------------------------------------------------------------------
    if BEST_ISOMER_ONLY:
        has_energies = any(r.get("relax_energy_eV") is not None for r in rows)
        if has_energies:
            rows = _filter_best_isomers(rows)
        else:
            print("  BEST_ISOMER_ONLY skipped: no energies available yet.")
            print("  Run with COMPUTE_ENERGY = True first.")

    # --------------------------------------------------------------------------
    # Stage 3: Reaction network
    #
    # Builds the isodesmic reaction graph and screens by DeltaG.
    # Uses energies from Stage 1 if available, otherwise runs xTB fresh.
    # Molecule objects are loaded from MOLS_CACHE if needed.
    # --------------------------------------------------------------------------
    if COMPUTE_REACTIONS:
        from molbuilder.reactions import ReactionNetwork

        # Load molecule objects -- from cache if the current rows came from CSV
        if MOLS_CACHE.exists():
            mol_lookup = _load_mols_cache(MOLS_CACHE)
        else:
            mol_lookup = {r["filename"]: m for m, r in zip(mols_in_order, rows)}

        mols_and_rows = [
            (mol_lookup[r["filename"]], r)
            for r in rows
            if r.get("filename") in mol_lookup
        ]

        if not mols_and_rows:
            print("ERROR: No molecule objects found for reaction network.")
            print(f"       Expected cache at {MOLS_CACHE}.")
            print("       Run Stage 0 first to generate the cache.")
        else:
            print(f"Building reaction network from {len(mols_and_rows)} structures ...")
            net = ReactionNetwork(
                mols_and_rows,
                bond_filter=True,
                include_geometry_changes=True,
                verbose=True,
            )
            print(net.summary())

            has_energies = any(r.get("relax_energy_eV") is not None for r in rows)
            if has_energies:
                _attach_energies_from_rows(net, rows)
            else:
                _run_network_energies(net)

            # Screen and print
            e_or_g = "DeltaG" if COMPUTE_THERMO else "DeltaE"
            hits = net.screen(max_dE=REACTION_MAX_DG, use_gibbs=COMPUTE_THERMO,
                              reaction_types=REACTION_TYPES, require_energy=True)
            print(f"\n{len(hits)} reaction(s) with {e_or_g} <= {REACTION_MAX_DG} eV:")
            for src, dst, e, val in hits[:20]:
                print(f"  {val:+.3f} eV  {net.reaction_str(src, dst)}")
            if len(hits) > 20:
                print(f"  ... and {len(hits) - 20} more")

            df = net.to_dataframe()
            df.to_csv(REACTIONS_CSV, index=False)

            # Report how many reactions actually have DeltaG vs None
            n_total  = len(df)
            n_with_g = int(df["delta_g_eV"].notna().sum()) if "delta_g_eV" in df.columns else 0
            n_with_e = int(df["delta_e_eV"].notna().sum()) if "delta_e_eV" in df.columns else 0
            print(f"\nReaction network -> {REACTIONS_CSV}")
            print(f"  {n_total} reactions total")
            print(f"  {n_with_e} have DeltaE,  {n_with_g} have DeltaG")
            if n_with_g == 0 and COMPUTE_THERMO:
                print("  WARNING: no DeltaG values computed.")
                print("  Reference molecule energies (H2O, HCOOH) may have failed.")
                print("  Check above for 'Ref energy failed' warnings.")
            elif n_with_g < n_total:
                print(f"  {n_total - n_with_g} reactions missing DeltaG (node energies absent)")

            try:
                import matplotlib.pyplot as plt
                fig = net.plot(
                    title=f"Ni reaction network ({e_or_g})",
                    edge_label="delta_g" if COMPUTE_THERMO else "delta_e",
                )
                fig.savefig(REACTIONS_PLOT, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Network plot     -> {REACTIONS_PLOT}")
            except Exception as exc:
                print(f"(plot skipped: {exc})")

            if net.broken_structures:
                write_broken_report(net.broken_structures, OUTPUT_DIR, verbose=True)
