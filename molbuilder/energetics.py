"""
energetics.py
=============
High-level pipeline for computing xTB / MACE energetics on sets of
molbuilder structures, with broken-bond detection and Excel/CSV output.

This module is the main interface used by both the generation script and
the standalone compute_energetics.py CLI.

Public API
----------
    run_energetics(rows, mols, ...)   → updated rows list with energy columns
    molecule_name(row)                → human-readable label from a row dict
    BondStatus                        → "OK" | "STRETCHED" | "BROKEN"

Bond status definitions
-----------------------
    OK          all M-L distances within 1.20× initial
    STRETCHED   longest M-L bond is 1.20–1.35× initial  (possible strain)
    BROKEN      at least one M-L bond > 1.35× initial   (ligand dissociated)

The threshold values are conservative and reflect that xTB optimised
M-L distances typically change by < 10% from the idealised template.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from molbuilder.core.molecule import Molecule
from molbuilder.output.writer import write_poscar, write_csv, write_json


# ── bond status constants ─────────────────────────────────────────────────────

class BondStatus:
    OK        = "OK"
    STRETCHED = "STRETCHED"
    BROKEN    = "BROKEN"

_STRETCHED_THRESHOLD = 1.20   # > this → STRETCHED
_BROKEN_THRESHOLD    = 1.35   # > this → BROKEN


def _bond_status(max_elongation: float) -> str:
    """Convert a max-elongation ratio to a BondStatus string."""
    if max_elongation > _BROKEN_THRESHOLD:
        return BondStatus.BROKEN
    if max_elongation > _STRETCHED_THRESHOLD:
        return BondStatus.STRETCHED
    return BondStatus.OK


# ── label helper ──────────────────────────────────────────────────────────────

def molecule_name(row: dict) -> str:
    """Build a concise human-readable name from a row dict."""
    parts = [
        row.get("ox_label", ""),
        f"CN{row['cn']}" if row.get("cn") else "",
        row.get("geometry", ""),
        row.get("ligand_combo", ""),
    ]
    if row.get("bridge"):
        parts.append(row["bridge"])
    if row.get("arrangement"):
        parts.append(row["arrangement"])
    return "  ".join(p for p in parts if p)


# ── internal helpers ──────────────────────────────────────────────────────────

def _write_relaxed(mol: Molecule, row: dict, output_dir: Path,
                   suffix: str) -> Path:
    """Write a relaxed POSCAR + JSON, return the POSCAR path."""
    orig  = Path(row.get("filename", "unknown.POSCAR"))
    if not orig.is_absolute():
        orig = output_dir / orig
    rpath = orig.with_stem(orig.stem + suffix)
    write_poscar(mol, rpath)
    write_json({
        "formula":        mol.formula,
        "charge":         int(mol.charge),
        "spin_mult":      int(mol.spin_multiplicity),
        "energy_eV":      row.get("relax_energy_eV"),
        "gibbs_eV":       row.get("relax_gibbs_eV"),
        "enthalpy_eV":    row.get("relax_enthalpy_eV"),
        "zpe_eV":         row.get("relax_zpe_eV"),
        "entropy_eV_K":   row.get("relax_entropy_eV_K"),
        "T_K":            row.get("relax_T_K"),
        "P_Pa":           row.get("relax_P_Pa"),
        "converged":      row.get("relax_converged"),
        "steps":          row.get("relax_steps"),
        "backend":        row.get("relax_backend"),
        "model":          row.get("relax_model"),
        "bond_status":    row.get("bond_status"),
        "bond_max_elong": row.get("bond_max_elongation"),
        "n_broken_bonds": row.get("bond_n_broken"),
    }, rpath.with_suffix(".json"))
    return rpath


def _empty_energy_cols() -> dict:
    """Return a dict of all energy columns initialised to None."""
    return {
        "relax_energy_eV":      None,
        "relax_gibbs_eV":       None,
        "relax_zpe_eV":         None,
        "relax_enthalpy_eV":    None,
        "relax_entropy_eV_K":   None,
        "relax_T_K":            None,
        "relax_P_Pa":           None,
        "relax_mace_energy_eV": None,
        "relax_mace_gibbs_eV":  None,
        "relax_dE_mace_xtb_eV": None,
        "relax_converged":      None,
        "relax_steps":          None,
        "relax_backend":        None,
        "relax_model":          None,
        "relax_filename":       None,
        "spin_multiplicity":    None,
        # bond columns
        "bond_status":          BondStatus.OK,
        "bond_max_elongation":  None,
        "bond_n_broken":        0,
    }


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_energetics(
    rows:            List[dict],
    mols:            Dict[str, Molecule],
    backend:         str              = "xtb",
    compute_thermo:  bool             = False,
    T:               float            = 298.15,
    P:               float            = 101325.0,
    fmax:            float            = 0.05,
    steps:           int              = 300,
    xtb_model:       Optional[str]    = None,
    mace_model:      Optional[str]    = None,
    mace_device:     str              = "cpu",
    constrain_bonds: bool             = False,
    output_dir:      Optional[Path]   = None,
    csv_file:        Optional[Path]   = None,
    excel_file:      Optional[Path]   = None,
    verbose:         bool             = True,
) -> List[dict]:
    """
    Compute xTB and/or MACE energetics for a list of structures.

    Adds energy columns to each row dict and optionally writes relaxed
    POSCAR files, a CSV summary, and a formatted Excel workbook.

    Broken bonds are automatically detected and flagged in the
    ``bond_status`` column as ``"OK"``, ``"STRETCHED"``, or ``"BROKEN"``.
    The thresholds are 1.20× (STRETCHED) and 1.35× (BROKEN) of the
    initial M-L bond length.  Structures with broken bonds are highlighted
    in orange in the Excel output and can be filtered in the CSV.

    Parameters
    ----------
    rows            : Row dicts from write_all() or enumerate_complexes().
    mols            : Mapping from row["filename"] → Molecule.
    backend         : "xtb", "mace", or "both".
    compute_thermo  : If True, run freq → G(T, P).
                      If False, geometry relax only → E.
    T, P            : Temperature (K) and pressure (Pa) for ΔG.
    fmax, steps     : Geometry optimiser settings.
    xtb_model       : Override xTB model  (None → GFN2-xTB).
    mace_model      : Override MACE model (None → mh-1).
    mace_device     : "cpu" or "cuda".
    constrain_bonds : Prevent M-L bond dissociation.  Default False — bond
                      breaking is meaningful information.  See relaxation.py
                      for guidance on when to enable this.
    output_dir      : Where to write relaxed POSCARs and JSON sidecars.
    csv_file        : Path for the CSV summary (None to skip).
    excel_file      : Path for the Excel workbook (None to skip).
    verbose         : Print per-structure progress.

    Returns
    -------
    Updated list of row dicts.  New columns include:
        molecule_name, relax_energy_eV, relax_gibbs_eV, relax_zpe_eV,
        relax_enthalpy_eV, relax_entropy_eV_K, relax_T_K, relax_P_Pa,
        relax_mace_energy_eV, relax_mace_gibbs_eV, relax_dE_mace_xtb_eV,
        relax_converged, relax_steps, relax_backend, relax_model,
        relax_filename, spin_multiplicity,
        bond_status, bond_max_elongation, bond_n_broken.
    """
    try:
        from molbuilder.relaxation import (
            relax, thermochemistry, compare_backends, check_bonds_intact,
        )
    except ImportError as exc:
        raise ImportError(
            "Relaxation backend not available. "
            "Install with: pip install tblite ase  (xTB) or  "
            "pip install mace-torch ase  (MACE)"
        ) from exc

    backend = backend.lower()
    updated: List[dict] = []

    for idx, row in enumerate(rows):
        fname = row.get("filename", "")
        mol   = mols.get(fname)

        row = dict(row)
        row["molecule_name"] = molecule_name(row)
        row.update(_empty_energy_cols())

        if mol is None:
            updated.append(row)
            continue

        row["spin_multiplicity"] = int(mol.spin_multiplicity)

        if verbose:
            label = row["molecule_name"] or mol.formula
            print(f"  [{idx+1}/{len(rows)}] {mol.formula:20s} "
                  f"{label[:35]:35s} ", end="", flush=True)

        try:
            if backend == "both":
                comp = compare_backends(
                    mol, T=T, P=P, fmax=fmax, steps=steps,
                    xtb_model=xtb_model, mace_model=mace_model,
                    mace_device=mace_device,
                    compute_thermo=compute_thermo,
                    constrain_bonds=constrain_bonds,
                )
                # xTB results
                if comp["xtb"]:
                    xtb = comp["xtb"]
                    row["relax_energy_eV"] = comp["delta_E_xtb_eV"]
                    row["relax_converged"] = bool(xtb.converged)
                    row["relax_steps"]     = int(xtb.steps)
                    row["relax_backend"]   = "both"
                    row["relax_model"]     = f"xtb={xtb.model}"
                    if compute_thermo and hasattr(xtb, "gibbs_eV"):
                        row["relax_gibbs_eV"]    = round(xtb.gibbs_eV, 6)
                        row["relax_zpe_eV"]      = round(xtb.zpe_eV, 6)
                        row["relax_enthalpy_eV"] = round(xtb.enthalpy_eV, 6)
                        row["relax_entropy_eV_K"]= round(xtb.entropy_eV_K, 8)
                        row["relax_T_K"] = T; row["relax_P_Pa"] = P
                    bc = comp.get("bond_check_xtb") or {}
                    row["bond_max_elongation"] = bc.get("max_elongation")
                    row["bond_n_broken"]       = len(bc.get("broken_bonds", []))
                    row["bond_status"]         = _bond_status(
                        bc.get("max_elongation") or 1.0)
                    if output_dir:
                        rp = _write_relaxed(xtb.mol, row, output_dir, "_relaxed_xtb")
                        row["relax_filename"] = str(rp)
                # MACE results
                if comp["mace"]:
                    mace = comp["mace"]
                    row["relax_mace_energy_eV"] = comp["delta_E_mace_eV"]
                    if compute_thermo and hasattr(mace, "gibbs_eV"):
                        row["relax_mace_gibbs_eV"] = comp["delta_G_mace_eV"]
                    row["relax_dE_mace_xtb_eV"] = comp.get("dE_xtb_vs_mace_eV")
                    if output_dir:
                        _write_relaxed(mace.mol, row, output_dir, "_relaxed_mace")

            else:
                # Single backend
                m = xtb_model if backend == "xtb" else mace_model
                if compute_thermo:
                    res = thermochemistry(mol, backend=backend, model=m,
                                         device=mace_device, T=T, P=P,
                                         fmax=fmax, steps=steps,
                                         constrain_bonds=constrain_bonds)
                    row["relax_gibbs_eV"]    = round(res.gibbs_eV, 6)
                    row["relax_zpe_eV"]      = round(res.zpe_eV, 6)
                    row["relax_enthalpy_eV"] = round(res.enthalpy_eV, 6)
                    row["relax_entropy_eV_K"]= round(res.entropy_eV_K, 8)
                    row["relax_T_K"] = T; row["relax_P_Pa"] = P
                else:
                    res = relax(mol, backend=backend, model=m,
                                device=mace_device, fmax=fmax, steps=steps,
                                constrain_bonds=constrain_bonds)

                row["relax_energy_eV"] = round(float(res.energy_eV), 6)
                row["relax_converged"] = bool(res.converged)
                row["relax_steps"]     = int(res.steps)
                row["relax_backend"]   = res.backend
                row["relax_model"]     = res.model
                if backend == "mace":
                    row["relax_mace_energy_eV"] = row.pop("relax_energy_eV")
                    row["relax_energy_eV"] = None

                bc = check_bonds_intact(mol, res.mol)
                row["bond_max_elongation"] = bc["max_elongation"]
                row["bond_n_broken"]       = len(bc["broken_bonds"])
                row["bond_status"]         = _bond_status(bc["max_elongation"])

                if output_dir:
                    rp = _write_relaxed(res.mol, row, output_dir,
                                        f"_relaxed_{backend}")
                    row["relax_filename"] = str(rp)

            # ── verbose output ────────────────────────────────────────────────
            if verbose:
                e = row.get("relax_energy_eV")
                g = row.get("relax_gibbs_eV")
                bs = row.get("bond_status", BondStatus.OK)
                cv = row.get("relax_converged", True)
                e_str = f"E={e:.3f}eV" if e is not None else ""
                g_str = f"  G={g:.3f}eV" if g is not None else ""
                b_str = (f"  ⚠ {bs}" if bs != BondStatus.OK else "")
                c_str = ("  ! not converged" if cv is False else "")
                print(f"{e_str}{g_str}{b_str}{c_str}")

        except Exception as exc:
            if verbose:
                print(f"FAILED: {exc}")
            warnings.warn(f"Energetics failed for {mol.formula}: {exc}", stacklevel=2)
            row["bond_status"] = "ERROR"

        updated.append(row)

    # ── write outputs ─────────────────────────────────────────────────────────
    n_broken    = sum(1 for r in updated if r.get("bond_status") == BondStatus.BROKEN)
    n_stretched = sum(1 for r in updated if r.get("bond_status") == BondStatus.STRETCHED)
    n_conv      = sum(1 for r in updated if r.get("relax_converged") is True)
    n_done      = sum(1 for r in updated if r.get("relax_energy_eV") is not None
                      or r.get("relax_mace_energy_eV") is not None)

    if verbose and n_done:
        print(f"\n  Completed    : {n_done} / {len(rows)}")
        print(f"  Converged    : {n_conv} / {n_done}")
        if n_broken:
            print(f"  ⚠ BROKEN     : {n_broken}  (bond dissociated during relaxation)")
        if n_stretched:
            print(f"  ~ STRETCHED  : {n_stretched}  (bond elongated > 1.20×)")

    if csv_file and updated:
        write_csv(updated, Path(csv_file))
        if verbose:
            print(f"  CSV  → {csv_file}")

    if excel_file and updated:
        from molbuilder.output.excel_writer import write_energetics_excel
        write_energetics_excel(updated, Path(excel_file))
        if verbose:
            print(f"  XLSX → {excel_file}")

    return updated


# ── broken-structure reporting ────────────────────────────────────────────────

def write_broken_report(
    broken: list,
    output_dir,
    verbose: bool = True,
) -> None:
    """
    Write a dedicated report for structures with bond_status == BROKEN.

    Creates:
        <output_dir>/broken/broken_structures.csv   — CSV with all broken rows
        <output_dir>/broken/*.POSCAR                — original (pre-relax) POSCARs

    Parameters
    ----------
    broken  : list of (mol, row) pairs where bond_status == "BROKEN".
              Typically from ReactionNetwork.broken_structures or
              filtered from run_energetics output rows.
    output_dir : root output directory (Path or str).
    verbose : print a summary.
    """
    from pathlib import Path
    from molbuilder.output.writer import write_csv, write_poscar

    if not broken:
        return

    broken_dir = Path(output_dir) / "broken"
    broken_dir.mkdir(parents=True, exist_ok=True)

    rows_out = []
    for mol, row in broken:
        # Copy POSCAR to broken/ subdir
        orig = Path(row.get("filename", "unknown.POSCAR"))
        dest = broken_dir / orig.name
        try:
            write_poscar(mol, dest)
        except Exception:
            pass

        rows_out.append({**row,
                         "bond_status":    "BROKEN",
                         "review_needed":  True,
                         "broken_poscar":  str(dest),
                         })

    csv_path = broken_dir / "broken_structures.csv"
    write_csv(rows_out, csv_path)

    if verbose:
        print()
        print("=" * 60)
        print(f"  ⚠  {len(broken)} BROKEN STRUCTURE(S) — REVIEW BEFORE DFT")
        print("=" * 60)
        for mol, row in broken:
            geom  = row.get("geometry", "?")
            ligs  = row.get("ligand_combo", "?")
            elong = row.get("bond_max_elongation")
            e_str = f"  max_elong={elong:.2f}×" if elong else ""
            print(f"  • {mol.formula:15s}  [{geom}] {ligs}{e_str}")
        print(f"\n  POSCARs → {broken_dir}/")
        print(f"  CSV     → {csv_path}")
        print("=" * 60)
        print()
        print("  These structures had at least one M-L bond stretch > 1.35× its")
        print("  initial length during xTB relaxation, suggesting the coordination")
        print("  mode is strained or a ligand dissociated.  Options:")
        print("   1. Inspect the POSCAR manually and fix the geometry.")
        print("   2. Re-run with constrain_bonds=True to force the topology.")
        print("   3. Discard — if xTB says it's unstable, DFT likely will too.")
        print("=" * 60)
