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

import csv
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple

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
) -> List[Dict[str, Any]]:
    """
    Consume an ``enumerate_complexes`` iterator, write every structure to disk,
    and optionally write a CSV summary.

    Parameters
    ----------
    results    : Iterator of (Molecule, row_dict) from enumerate_complexes().
    output_dir : Root directory for output files.
                 Each file is placed at ``output_dir/<filename>`` where
                 *filename* is taken from the ``row["filename"]`` metadata key.
                 If that path is already absolute it is used as-is.
    csv_file   : Path for the CSV summary.  Pass None to skip.
    fmt        : "poscar" (default) or "xyz".
    verbose    : Print each file path as it is written.

    Returns
    -------
    List of all row dicts (same as what the iterator emits).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []

    for mol, row in results:
        fname = Path(row["filename"])
        path  = fname if fname.is_absolute() else output_dir / fname
        if fmt == "xyz":
            path = path.with_suffix(".xyz")
            write_xyz(mol, path)
        else:
            write_poscar(mol, path)
        row["filename"] = str(path)
        rows.append(row)
        if verbose:
            print(f"  → {path}")

    if csv_file is not None and rows:
        write_csv(rows, Path(csv_file))
        if verbose:
            print(f"\n  CSV summary → {csv_file}  ({len(rows)} entries)")

    return rows
