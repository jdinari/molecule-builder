"""
generate_ni_complexes.py
========================
Generate all neutral (charge = 0) Ni(II) and Ni(III) mononuclear,
dinuclear, and trinuclear complexes.

Ligands
-------
  HCOO    formate,      monodentate, charge -1
  HCOOH   formic acid,  monodentate, charge  0
  H2O     water,                     charge  0
  OH      hydroxide,    terminal,    charge -1

Bridging ligands (dimers/trimers only)
  mu-OH    bridging hydroxide,  charge -1
  mu-HCOO  bridging formate,    charge -1

Constraints
-----------
  - Net complex charge = 0
  - Coordination number per metal: 3–7
  - All symmetry-distinct isomers generated automatically

Output
------
  poscar/
    monomer/NiII/CN3/ … CN7/
    monomer/NiIII/CN3/ … CN7/
    dimer/NiII/CN3/ … CN7/
    dimer/NiIII/CN3/ … CN7/
    trimer/NiII/CN3/ … CN7/   (linear and triangular)
    trimer/NiIII/CN3/ … CN7/

  ni_complexes_summary.csv
"""

import csv
import sys
from itertools import combinations_with_replacement
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from molbuilder.api import build, dimer, trimer
from molbuilder.core.geometry import infer_geometry
from molbuilder.core.isomers import enumerate_isomers
from molbuilder.output.poscar_writer import poscar_to_string

# ── ligand charges ────────────────────────────────────────────────────────────

TERMINAL_LIGANDS = ["HCOO", "HCOOH", "H2O", "OH"]
TERMINAL_CHARGE  = {"HCOO": -1, "HCOOH": 0, "H2O": 0, "OH": -1}

BRIDGE_LIGANDS   = ["mu-OH", "mu-HCOO"]
BRIDGE_CHARGE    = {"mu-OH": -1, "mu-HCOO": -1}

GEOMETRY_FOR_CN  = {3: "tp", 4: "tet", 5: "sqpy", 6: "oct", 7: "pbp"}
EXTRA_GEOM       = {4: ["sqp"], 5: ["tbp"]}   # additional geometries per CN

OUTPUT_ROOT = Path("poscar")
CSV_FILE    = Path("ni_complexes_summary.csv")

# ── helpers ───────────────────────────────────────────────────────────────────

def geometries_for_cn(cn):
    g = [GEOMETRY_FOR_CN[cn]] + EXTRA_GEOM.get(cn, [])
    return g

def combo_label(combo):
    from collections import Counter
    c = Counter(combo)
    return "_".join(f"{l}{n}" for l, n in sorted(c.items()))

def safe(s):
    return s.replace(":", "-").replace("/", "-").replace(" ", "_")

def write_poscar(mol, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(poscar_to_string(mol))

def mol_to_row(mol, structure_type, ox, cn, geom, lig_combo, bridge, n_bridges,
               arrangement, isomer_label, filename):
    return {
        "structure":    structure_type,
        "ox":           ox,
        "ox_label":     f"Ni{'II' if ox == 2 else 'III'}",
        "cn":           cn,
        "geometry":     geom,
        "ligand_combo": lig_combo,
        "bridge":       bridge or "",
        "n_bridges":    n_bridges,
        "arrangement":  arrangement or "",
        "isomer":       isomer_label,
        "formula":      mol.formula,
        "charge":       mol.charge,
        "n_atoms":      mol.num_atoms(),
        "filename":     str(filename),
    }

# ── MONOMERS ──────────────────────────────────────────────────────────────────

def generate_monomers(rows):
    print("\n" + "="*60)
    print("  MONOMERS")
    print("="*60)
    n = 0

    for ox in [2, 3]:
        ox_label = "NiII" if ox == 2 else "NiIII"
        for cn in range(3, 8):
            for combo in combinations_with_replacement(TERMINAL_LIGANDS, cn):
                lig_charge = sum(TERMINAL_CHARGE[l] for l in combo)
                if ox + lig_charge != 0:
                    continue

                for geom in geometries_for_cn(cn):
                    iso_list = enumerate_isomers(list(combo), geom)
                    for iso in iso_list:
                        try:
                            mol = build("Ni", ox=ox,
                                        ligands=iso["site_assignment"],
                                        geometry=geom)
                            if isinstance(mol, list):
                                mol = mol[0]
                        except Exception as e:
                            print(f"  ✗ {ox_label} CN{cn} {geom} {combo}: {e}")
                            continue

                        label    = iso["label"]
                        cl       = combo_label(combo)
                        out_path = (OUTPUT_ROOT / "monomer" / ox_label / f"CN{cn}" /
                                    f"{safe(cl)}_{geom}_{safe(label)}.POSCAR")
                        write_poscar(mol, out_path)
                        n += 1
                        print(f"  ✓ monomer {ox_label} CN{cn} {geom:5s}  "
                              f"{cl:28s}  {label:10s}  {mol.formula}")
                        rows.append(mol_to_row(mol, "monomer", ox, cn, geom,
                                               cl, None, 0, None, label, out_path))
    return n

# ── DIMERS ────────────────────────────────────────────────────────────────────

def generate_dimers(rows):
    print("\n" + "="*60)
    print("  DIMERS")
    print("="*60)
    n = 0

    for ox in [2, 3]:
        ox_label = "NiII" if ox == 2 else "NiIII"
        for bridge in BRIDGE_LIGANDS:
            bc = BRIDGE_CHARGE[bridge]
            for n_bridges in [1, 2, 3]:
                for n_term in range(0, 8 - n_bridges):
                    for terminal in combinations_with_replacement(TERMINAL_LIGANDS, n_term):
                        tc = sum(TERMINAL_CHARGE[l] for l in terminal)
                        # per-metal charge must be 0
                        if ox + n_bridges * bc + tc != 0:
                            continue
                        cn = n_term + n_bridges
                        if not (3 <= cn <= 7):
                            continue

                        try:
                            mol = dimer("Ni", ox=ox,
                                        terminal=list(terminal),
                                        bridge=bridge,
                                        n=n_bridges)
                        except Exception as e:
                            print(f"  ✗ dimer {ox_label} {n_bridges}x{bridge} "
                                  f"term={terminal}: {e}")
                            continue

                        if mol.charge != 0:
                            continue  # safety check

                        cl       = f"{combo_label(terminal)}_{'_'+str(n_bridges)+'x'+bridge}"
                        geom     = GEOMETRY_FOR_CN[cn]
                        out_path = (OUTPUT_ROOT / "dimer" / ox_label / f"CN{cn}" /
                                    f"{safe(combo_label(terminal))}_{n_bridges}x{safe(bridge)}.POSCAR")
                        write_poscar(mol, out_path)
                        n += 1
                        print(f"  ✓ dimer  {ox_label} CN{cn}  {n_bridges}x{bridge:10s}  "
                              f"term={terminal}  {mol.formula}")
                        rows.append(mol_to_row(mol, "dimer", ox, cn, geom,
                                               combo_label(terminal), bridge, n_bridges,
                                               None, "only", out_path))
    return n

# ── TRIMERS ───────────────────────────────────────────────────────────────────

def generate_trimers(rows):
    """
    trimer() internally adds the bridge ligand TWICE to each metal's list,
    so per-metal charge = ox + 2*bridge_charge + terminal_charge.
    Arrangements: linear and triangular.
    """
    print("\n" + "="*60)
    print("  TRIMERS")
    print("="*60)
    n = 0

    for ox in [2, 3]:
        ox_label = "NiII" if ox == 2 else "NiIII"
        for bridge in BRIDGE_LIGANDS:
            bc = BRIDGE_CHARGE[bridge]
            for n_term in range(0, 6):
                for terminal in combinations_with_replacement(TERMINAL_LIGANDS, n_term):
                    tc = sum(TERMINAL_CHARGE[l] for l in terminal)
                    # trimer() adds bridge twice per metal
                    if ox + 2 * bc + tc != 0:
                        continue
                    cn = n_term + 2
                    if not (3 <= cn <= 7):
                        continue

                    for arrangement in ["linear", "triangular"]:
                        try:
                            mol = trimer("Ni", ox=ox,
                                         terminal=list(terminal),
                                         bridge=bridge,
                                         arrangement=arrangement)
                        except Exception as e:
                            print(f"  ✗ trimer {ox_label} {bridge} "
                                  f"term={terminal} {arrangement}: {e}")
                            continue

                        if mol.charge != 0:
                            continue

                        geom     = GEOMETRY_FOR_CN[cn]
                        out_path = (OUTPUT_ROOT / "trimer" / ox_label / f"CN{cn}" /
                                    f"{safe(combo_label(terminal))}_{safe(bridge)}_{arrangement}.POSCAR")
                        write_poscar(mol, out_path)
                        n += 1
                        print(f"  ✓ trimer {ox_label} CN{cn}  {bridge:10s}  "
                              f"term={terminal}  {arrangement:12s}  {mol.formula}")
                        rows.append(mol_to_row(mol, f"trimer_{arrangement}", ox, cn, geom,
                                               combo_label(terminal), bridge, 2,
                                               arrangement, "only", out_path))
    return n

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)
    rows = []

    n_mono   = generate_monomers(rows)
    n_dimer  = generate_dimers(rows)
    n_trimer = generate_trimers(rows)

    # write CSV
    if rows:
        with CSV_FILE.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"  Monomers : {n_mono}")
    print(f"  Dimers   : {n_dimer}")
    print(f"  Trimers  : {n_trimer}")
    print(f"  Total    : {n_mono + n_dimer + n_trimer} POSCAR files")
    print(f"  CSV      : {CSV_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
