"""
find_best_products.py
=====================
Direct ΔG ranking of Ni(II) coordination complexes — no reaction network needed.

Instead of building a full reaction network, this script:

  1. Enumerates every Ni(II) complex from your ligand pool.
  2. Relaxes each one with xTB and computes ΔG(T,P).
  3. Computes ΔG_form relative to free metal + free ligands (isodesmic
     references: HCOOH, H₂O, H₂ as neutral gas-phase species).
  4. Ranks all structures by ΔG_form and prints the best products.
  5. Optionally saves a bar-chart of the top-N and a CSV.

This answers "which complexes are thermodynamically most stable?" directly
without building or plotting a reaction network.

The formation reaction used is:
    Ni²⁺(aq)  +  ligand_pool  →  [Ni(ligands)]  +  displaced_refs

Because all species are neutral and isodesmic, the xTB ΔG is reliable at the
gas-phase level.  For more quantitative results use COMPUTE_THERMO=True.

Usage
-----
    python find_best_products.py

Flip the flags below as needed.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

from molbuilder import enumerate_complexes, MULTI_BRIDGE_CASES

# ── Chemistry ─────────────────────────────────────────────────────────────────
METAL        = "Ni"
OX_STATES    = [2]
LIGAND_POOL  = ["HCOO", "H2O", "OH"]
BRIDGE_POOL  = ["mu-OH", "mu-HCOO"]
NUCLEARITY   = [1, 2]          # 1 = monomers only; [1,2] = monomers + dimers
CN_RANGE     = (4, 6)

# ── Energetics ────────────────────────────────────────────────────────────────
COMPUTE_ENERGY  = True         # set False to just enumerate (no ΔG)
ENERGY_BACKEND  = "xtb+mace"   # "xtb", "mace", or "xtb+mace" (recommended hybrid)
COMPUTE_THERMO  = True         # True → ΔG(T,P);  False → ΔE only (faster)
TEMPERATURE_K   = 298.15
PRESSURE_PA     = 101325.0
RELAX_FMAX      = 0.05
RELAX_STEPS     = 300

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("poscar")
CSV_OUT     = Path("best_products.csv")
PLOT_OUT    = Path("best_products.png")
TOP_N       = 20               # how many complexes to show in the bar chart

# ── Reference molecule geometries (inline — no network dependency) ────────────

import numpy as np
from molbuilder.core.molecule import Molecule, Atom


def _make_ref(name: str) -> Molecule:
    if name == "H2O":
        a = np.radians(104.5 / 2); oh = 0.958
        return Molecule(
            atoms=[Atom("O", np.zeros(3)),
                   Atom("H", np.array([ oh*np.sin(a), oh*np.cos(a), 0.])),
                   Atom("H", np.array([-oh*np.sin(a), oh*np.cos(a), 0.]))],
            formula="H2O", charge=0, spin_multiplicity=1,
            metal_symbol="", metal_ox=0,
        )
    if name == "HCOOH":
        a = np.radians(124.0 / 2); co_d=1.20; co_s=1.34; oh=0.972; ch=1.09
        return Molecule(
            atoms=[Atom("C", np.zeros(3)),
                   Atom("O", np.array([ co_d*np.sin(a),  co_d*np.cos(a), 0.])),
                   Atom("O", np.array([-co_s*np.sin(a),  co_s*np.cos(a), 0.])),
                   Atom("H", np.array([0., -ch, 0.])),
                   Atom("H", np.array([-co_s*np.sin(a)-oh*0.77,
                                        co_s*np.cos(a)+oh*0.64, 0.]))],
            formula="HCOOH", charge=0, spin_multiplicity=1,
            metal_symbol="", metal_ox=0,
        )
    if name == "H2":
        return Molecule(
            atoms=[Atom("H", np.array([-0.371, 0., 0.])),
                   Atom("H", np.array([ 0.371, 0., 0.]))],
            formula="H2", charge=0, spin_multiplicity=1,
            metal_symbol="", metal_ox=0,
        )
    raise ValueError(f"Unknown reference: {name}")


# Neutral acid references for each anionic ligand
# HCOO⁻ → HCOOH;  OH⁻ → H2O
_REF_FOR_LIGAND = {
    "H2O":   ("H2O",   1),    # neutral  → 1 H2O consumed
    "HCOO":  ("HCOOH", 1),    # -1 charge → HCOOH consumed, H2O produced
    "OH":    ("H2O",   1),    # -1 charge → H2O consumed
    "HCOOH": ("HCOOH", 1),
}


def _compute_ref_energies(backend: str, compute_thermo: bool,
                           T: float, P: float,
                           fmax: float, steps: int,
                           mace_model: str = None,
                           mace_device: str = "cpu",
                           xtb_model: str = None) -> dict:
    """Return {name: {energy_eV, gibbs_eV}} for H2O, HCOOH, H2.
    Uses the same backend as the complexes so isodesmic cancellation is valid.
    """
    from molbuilder.relaxation import thermochemistry, relax, xtb_relax_mace_singlepoint

    eff = backend.lower()
    results = {}
    for name in ("H2O", "HCOOH", "H2"):
        mol = _make_ref(name)
        print(f"  Reference {name:8s} … ", end="", flush=True)
        try:
            if eff == "xtb+mace":
                if compute_thermo:
                    res = xtb_relax_mace_singlepoint(
                        mol, xtb_model=xtb_model, mace_model=mace_model,
                        mace_device=mace_device, compute_thermo=True,
                        T=T, P=P, fmax=fmax, steps=steps,
                    )
                    results[name] = {"energy_eV": float(res.energy_eV),
                                      "gibbs_eV":  float(res.gibbs_eV)}
                    print(f"E_MACE={res.energy_eV:.3f}  G_hybrid={res.gibbs_eV:.3f} eV")
                else:
                    res = xtb_relax_mace_singlepoint(
                        mol, xtb_model=xtb_model, mace_model=mace_model,
                        mace_device=mace_device, compute_thermo=False,
                        fmax=fmax, steps=steps,
                    )
                    results[name] = {"energy_eV": float(res.energy_eV), "gibbs_eV": None}
                    print(f"E_MACE={res.energy_eV:.3f} eV")
            elif compute_thermo:
                res = thermochemistry(mol, backend=eff,
                                      T=T, P=P, fmax=fmax, steps=steps)
                results[name] = {"energy_eV": float(res.energy_eV),
                                  "gibbs_eV":  float(res.gibbs_eV)}
                print(f"E={res.energy_eV:.3f}  G={res.gibbs_eV:.3f} eV")
            else:
                res = relax(mol, backend=eff, fmax=fmax, steps=steps)
                results[name] = {"energy_eV": float(res.energy_eV), "gibbs_eV": None}
                print(f"E={res.energy_eV:.3f} eV")
        except Exception as exc:
            warnings.warn(f"Reference {name} failed: {exc}")
            results[name] = {"energy_eV": None, "gibbs_eV": None}
    return results


def _formation_dg(row: dict, complex_val: float,
                  ref_vals: dict, key: str) -> float | None:
    """
    Compute ΔG_form for one complex (isodesmic, ligand-exchange reference).

    Formation reaction example for [Ni(HCOO)₂(H₂O)₂]:
        2 HCOOH  +  2 H₂O  →  [Ni(HCOO)₂(H₂O)₂]  +  2 H₂O   (net: +2 HCOOH consumed)

    Here we use the convention:
        ΔG_form = G(complex) − Σ G(ref_in)
    where ref_in is one reference molecule per coordinated anionic ligand.
    Neutral ligands (H₂O, HCOOH) contribute their own G directly.
    """
    from collections import Counter
    import re

    combo = row.get("ligand_combo", "")
    # Parse e.g. "HCOO2_H2O2_OH1" → Counter
    counts: Counter = Counter()
    for part in combo.split("_"):
        if not part:
            continue
        m = re.match(r"([A-Za-z0-9:]+?)(\d+)$", part)
        if m:
            counts[m.group(1)] += int(m.group(2))
        else:
            counts[part] += 1

    cost = 0.0
    for lig, n in counts.items():
        ref_name, _ = _REF_FOR_LIGAND.get(lig, ("H2O", 1))
        rv = ref_vals.get(ref_name, {}).get(key)
        if rv is None:
            return None
        cost += n * rv

    if complex_val is None:
        return None
    return complex_val - cost


def _make_label(row: dict) -> str:
    structure = row.get("structure", "")
    geom      = row.get("geometry", "")
    combo     = row.get("ligand_combo", "")
    cn        = row.get("cn", "")
    ox        = row.get("ox_label", "")
    parts = [f"[Ni{ox}]", combo.replace("_", "/"), geom, f"CN{cn}"]
    if "dimer" in structure:
        parts.insert(0, "dimer")
    return "  ".join(p for p in parts if p)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Enumerate
    print("=" * 60)
    print("Step 1: Enumerate structures")
    print("=" * 60)
    mols_and_rows = list(enumerate_complexes(
        metal=METAL, ox_states=OX_STATES,
        ligand_pool=LIGAND_POOL, bridge_pool=BRIDGE_POOL,
        nuclearity=NUCLEARITY, cn_range=CN_RANGE,
        output_root=OUTPUT_DIR / "poscar",
        multi_bridge_cases=MULTI_BRIDGE_CASES,
        verbose=False,
    ))
    print(f"  {len(mols_and_rows)} structures enumerated.")

    if not COMPUTE_ENERGY:
        # Just print the inventory and exit
        print("\nCOMPUTE_ENERGY=False — listing structures only:\n")
        for i, (mol, row) in enumerate(mols_and_rows, 1):
            print(f"  {i:4d}.  {_make_label(row)}")
        sys.exit(0)

    # 2. Compute reference molecule energies first
    print("\n" + "=" * 60)
    print("Step 2: Compute reference molecule energies (HCOOH, H₂O, H₂)")
    print("=" * 60)
    MACE_MODEL  = None   # None → mh-1; set to path for custom model
    MACE_DEVICE = "cpu"  # "cpu" or "cuda"
    XTB_MODEL   = None   # None → GFN2-xTB
    try:
        ref_vals = _compute_ref_energies(
            backend=ENERGY_BACKEND, compute_thermo=COMPUTE_THERMO,
            T=TEMPERATURE_K, P=PRESSURE_PA,
            fmax=RELAX_FMAX, steps=RELAX_STEPS,
            mace_model=MACE_MODEL, mace_device=MACE_DEVICE, xtb_model=XTB_MODEL,
        )
    except ImportError:
        print("  ✗ xTB not installed — pip install tblite ase")
        sys.exit(1)

    # 3. Relax / thermochem each complex
    print("\n" + "=" * 60)
    print("Step 3: Compute complex energies")
    print("=" * 60)
    try:
        from molbuilder.relaxation import (thermochemistry, relax,
                                           check_bonds_intact,
                                           xtb_relax_mace_singlepoint)
    except ImportError:
        print("  ✗ pip install tblite ase  (+ mace-torch for xtb+mace)")
        sys.exit(1)

    _eff = ENERGY_BACKEND.lower()

    results = []  # list of dicts
    n_total = len(mols_and_rows)
    for idx, (mol, row) in enumerate(mols_and_rows, 1):
        label = _make_label(row)
        print(f"  [{idx}/{n_total}] {label[:55]:<55s} … ", end="", flush=True)
        r = dict(row)
        r["label"] = label
        try:
            if _eff == "xtb+mace":
                res = xtb_relax_mace_singlepoint(
                    mol,
                    xtb_model=XTB_MODEL, mace_model=MACE_MODEL,
                    mace_device=MACE_DEVICE,
                    compute_thermo=COMPUTE_THERMO,
                    T=TEMPERATURE_K, P=PRESSURE_PA,
                    fmax=RELAX_FMAX, steps=RELAX_STEPS,
                )
                r["energy_eV"]     = float(getattr(res, "_xtb_energy_eV", res.energy_eV))
                r["mace_energy_eV"]= float(res.energy_eV)
                r["gibbs_eV"]      = float(res.gibbs_eV) if COMPUTE_THERMO else None
                tag = (f"E_MACE={res.energy_eV:.3f}  G_hybrid={res.gibbs_eV:.3f} eV"
                       if COMPUTE_THERMO else f"E_MACE={res.energy_eV:.3f} eV")
            elif COMPUTE_THERMO:
                res = thermochemistry(
                    mol, backend=_eff,
                    T=TEMPERATURE_K, P=PRESSURE_PA,
                    fmax=RELAX_FMAX, steps=RELAX_STEPS,
                )
                r["energy_eV"] = float(res.energy_eV)
                r["gibbs_eV"]  = float(res.gibbs_eV)
                tag = f"E={res.energy_eV:.3f}  G={res.gibbs_eV:.3f} eV"
            else:
                res = relax(mol, backend=_eff,
                            fmax=RELAX_FMAX, steps=RELAX_STEPS)
                r["energy_eV"] = float(res.energy_eV)
                r["gibbs_eV"]  = None
                tag = f"E={res.energy_eV:.3f} eV"

            # Bond check on the relaxed mol
            bc = check_bonds_intact(mol, res.mol)
            r["bond_status"]       = "BROKEN" if not bc["intact"] else "OK"
            r["bond_max_elongation"] = bc["max_elongation"]
            r["bond_n_broken"]     = len(bc["broken_bonds"])
            bs_tag = "  ⚠ BROKEN" if not bc["intact"] else ""
            print(tag + bs_tag)
        except Exception as exc:
            r["energy_eV"] = None
            r["gibbs_eV"]  = None
            r["bond_status"] = "ERROR"
            print(f"FAILED: {exc}")
        results.append(r)

    # 4. Compute ΔG_form and rank
    key = "gibbs_eV" if COMPUTE_THERMO else "energy_eV"
    for r in results:
        r["dg_form_eV"] = _formation_dg(r, r.get(key), ref_vals, key)

    ranked = sorted(
        [r for r in results if r.get("dg_form_eV") is not None],
        key=lambda x: x["dg_form_eV"],
    )
    failed = [r for r in results if r.get("dg_form_eV") is None]

    label_col = "ΔG_form (eV)" if COMPUTE_THERMO else "ΔE_form (eV)"
    print("\n" + "=" * 60)
    print(f"Step 4: Best products ranked by {label_col}")
    print("=" * 60)
    print(f"  {'Rank':>4}  {label_col:>14}  Structure")
    print("  " + "-" * 56)
    for i, r in enumerate(ranked[:TOP_N], 1):
        val = r["dg_form_eV"]
        bs  = "⚠ " if r.get("bond_status") == "BROKEN" else "  "
        print(f"  {i:4d}  {val:>+14.4f}  {bs}{r['label']}")
    if len(ranked) > TOP_N:
        print(f"  … and {len(ranked)-TOP_N} more (see CSV)")
    if failed:
        print(f"\n  ⚠ {len(failed)} structure(s) failed energy computation.")

    # 5. Save CSV
    try:
        import pandas as pd
        df = pd.DataFrame(results)[
            ["label", "structure", "geometry", "cn", "ligand_combo",
             "energy_eV", "gibbs_eV", "dg_form_eV", "bond_status"]
        ].sort_values("dg_form_eV")
        df.to_csv(CSV_OUT, index=False)
        print(f"\n  CSV → {CSV_OUT}  ({len(df)} rows)")
    except ImportError:
        pass

    # 6. Bar chart of top-N
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = ranked[:TOP_N]
        labels = [r["label"].split("  ")[0] + "\n" + "  ".join(r["label"].split("  ")[1:])
                  for r in top]
        values = [r["dg_form_eV"] for r in top]
        colors = ["#C62828" if r.get("bond_status") == "BROKEN" else
                  "#2E7D32" if r["dg_form_eV"] < 0 else "#1565C0"
                  for r in top]

        fig, ax = plt.subplots(figsize=(max(10, len(top) * 0.55), 6))
        bars = ax.bar(range(len(top)), values, color=colors, alpha=0.85,
                      edgecolor="#333", linewidth=0.6)
        ax.axhline(0, color="#333", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(top)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(label_col, fontsize=10)
        ax.set_title(f"Best Ni(II) products (top {len(top)} by {label_col})",
                     fontsize=12, fontweight="bold")
        ax.set_facecolor("#F5F7FA")

        import matplotlib.patches as mpatches
        legend_handles = [
            mpatches.Patch(color="#2E7D32", label="ΔG < 0  (thermodynamically favoured)"),
            mpatches.Patch(color="#1565C0", label="ΔG ≥ 0"),
            mpatches.Patch(color="#C62828", label="BROKEN bond — exclude from DFT"),
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc="upper left")
        plt.tight_layout()
        fig.savefig(PLOT_OUT, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot → {PLOT_OUT}")
    except Exception as exc:
        print(f"  (plot skipped: {exc})")

    print()
    print("Done.  The most negative ΔG_form values are your best candidate products.")
    print("Structures with BROKEN bonds should be reviewed before DFT submission.")
