"""
writer.py
=========
Convenience I/O helpers for writing enumerated complexes to disk.

    write_poscar(mol, path)          – write one POSCAR
    write_xyz(mol, path)             – write one XYZ
    write_all(results, ...)          – write every (mol, row) pair + CSV summary
    write_csv(rows, csv_file)        – write just the CSV
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple
import csv

from molbuilder.core.molecule import Molecule
from molbuilder.output.poscar_writer import poscar_to_string
from molbuilder.output.xyz_writer import xyz_to_string


# ── single-file writers ───────────────────────────────────────────────────────

def write_poscar(mol: Molecule, path: Path) -> None:
    """Write *mol* to a VASP POSCAR file at *path*, creating directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(poscar_to_string(mol))


def write_xyz(mol: Molecule, path: Path) -> None:
    """Write *mol* to an XYZ file at *path*, creating directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xyz_to_string(mol))


def write_json(data: dict, path: Path) -> None:
    """Write a metadata dict as a pretty-printed JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ── CSV helper ────────────────────────────────────────────────────────────────

def write_csv(rows: List[Dict[str, Any]], csv_file: Path) -> None:
    """Write a list of metadata dicts to *csv_file*."""
    if not rows:
        return
    csv_file = Path(csv_file)
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    with csv_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── bulk writer ───────────────────────────────────────────────────────────────

def write_all(
    results: Iterable[Tuple[Molecule, Dict[str, Any]]],
    output_dir: str | Path = "poscar",
    csv_file: Optional[str | Path] = "complexes_summary.csv",
    fmt: str = "poscar",
    verbose: bool = False,
    # ── optional relaxation ───────────────────────────────────────────────────
    relax: bool = False,
    relax_backend: str = "xtb",
    relax_model: Optional[str] = None,
    relax_fmax: float = 0.05,
    relax_steps: int = 300,
    relax_device: str = "cpu",
    relax_suffix: str = "_relaxed",
    write_relax_json: bool = True,
) -> List[Dict[str, Any]]:
    """
    Consume an ``enumerate_complexes`` iterator, write every structure to disk,
    and optionally write a CSV summary.

    Parameters
    ----------
    results        : Iterator of (Molecule, row_dict) from enumerate_complexes().
    output_dir     : Root directory for output files.
    csv_file       : Path for the CSV summary.  Pass None to skip.
    fmt            : "poscar" (default) or "xyz".
    verbose        : Print each file path as it is written.

    Relaxation options (all ignored when relax=False)
    -------------------------------------------------
    relax          : If True, run geometry relaxation on every structure.
                     Off by default.
    relax_backend  : "xtb" (default) or "mace".
    relax_model    : Override the default model (None = backend default).
                     xTB default: "GFN2-xTB".  MACE default: "mh-1".
    relax_fmax     : Force convergence threshold in eV/Å.  Default 0.05.
    relax_steps    : Maximum optimiser steps.  Default 300.
    relax_device   : "cpu" or "cuda" (MACE only).
    relax_suffix   : Suffix appended to relaxed POSCAR filenames.
                     Default "_relaxed"  →  ``Ni_HCOO4_relaxed.POSCAR``.
    write_relax_json : If True, write a companion JSON alongside each relaxed
                       POSCAR containing energy, convergence, and backend
                       provenance.  Default True.

    Returns
    -------
    List of all row dicts.  When relax=True each dict gains extra fields:
        relax_energy_eV   float | None
        relax_converged   bool  | None
        relax_steps       int   | None
        relax_backend     str   | None
        relax_model       str   | None
        relax_filename    str   | None   (path to the relaxed POSCAR)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import relaxation module lazily — only needed when relax=True
    _relax_fn = None
    if relax:
        try:
            from molbuilder.relaxation import relax as _relax_import
            _relax_fn = _relax_import
        except ImportError as e:
            warnings.warn(
                f"relax=True requested but relaxation backend not available: {e}. "
                "Writing unrelaxed structures.",
                stacklevel=2,
            )
            relax = False

    rows: List[Dict[str, Any]] = []

    for mol, row in results:
        # ── write idealised POSCAR ──────────────────────────────────────────
        fname = Path(row["filename"])
        path  = fname if fname.is_absolute() else output_dir / fname
        if fmt == "xyz":
            path = path.with_suffix(".xyz")
            write_xyz(mol, path)
        else:
            write_poscar(mol, path)
        row["filename"] = str(path)

        # ── initialise relaxation columns ──────────────────────────────────
        row["relax_energy_eV"] = None
        row["relax_converged"] = None
        row["relax_steps"]     = None
        row["relax_backend"]   = None
        row["relax_model"]     = None
        row["relax_filename"]  = None

        # ── optional relaxation ────────────────────────────────────────────
        if relax and _relax_fn is not None:
            try:
                res = _relax_fn(
                    mol,
                    backend = relax_backend,
                    model   = relax_model,
                    fmax    = relax_fmax,
                    steps   = relax_steps,
                    device  = relax_device,
                )
                # Write relaxed structure with suffix
                rpath = path.with_stem(path.stem + relax_suffix)
                if fmt == "xyz":
                    write_xyz(res.mol, rpath.with_suffix(".xyz"))
                    rpath = rpath.with_suffix(".xyz")
                else:
                    write_poscar(res.mol, rpath)

                row["relax_energy_eV"] = round(float(res.energy_eV), 6)
                row["relax_converged"] = bool(res.converged)
                row["relax_steps"]     = int(res.steps)
                row["relax_backend"]   = res.backend
                row["relax_model"]     = res.model
                row["relax_filename"]  = str(rpath)

            except Exception as exc:
                warnings.warn(
                    f"Relaxation failed for {mol.formula}: {exc}. "
                    "Keeping unrelaxed structure.",
                    stacklevel=2,
                )

            if verbose and row["relax_energy_eV"] is not None:
                conv = "✓" if row["relax_converged"] else "!"
                print(f"  {conv} relaxed → {row['relax_filename']}  "
                      f"E={row['relax_energy_eV']:.4f} eV"
                      + ("" if row["relax_converged"] else "  (NOT converged)"))

        # JSON companion: written after the try/except so serialization errors
        # don't suppress the relaxation result that was already captured above.
        if relax and row["relax_energy_eV"] is not None and write_relax_json:
            try:
                rpath_stored = Path(row["relax_filename"])
                json_path    = rpath_stored.with_suffix(".json")
                write_json({
                    "formula":         mol.formula,
                    "charge":          int(mol.charge),
                    "spin_mult":       int(mol.spin_multiplicity),
                    "energy_eV":       row["relax_energy_eV"],
                    "converged":       row["relax_converged"],
                    "steps":           row["relax_steps"],
                    "backend":         row["relax_backend"],
                    "model":           row["relax_model"],
                    "fmax_eV_A":       float(relax_fmax),
                    "original_poscar": row["filename"],
                }, json_path)
            except Exception as exc:
                warnings.warn(f"Failed to write relaxation JSON for "
                              f"{mol.formula}: {exc}", stacklevel=2)

        rows.append(row)
        if verbose and not relax:
            print(f"  → {path}")

    if csv_file is not None and rows:
        write_csv(rows, Path(csv_file))
        if verbose:
            print(f"\n  CSV summary → {csv_file}  ({len(rows)} entries)")

    return rows
