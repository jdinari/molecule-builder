"""
generate_ni_complexes.py
========================
Generate all neutral (charge = 0) Ni(II) and Ni(III) mononuclear,
dinuclear, and trinuclear complexes.

Ligands
-------
  HCOO      formate, monodentate,          charge -1
  HCOO:bi   formate, bidentate chelating,  charge -1  (O,O, bite ~55°)
  HCOOH     formic acid, monodentate,      charge  0
  H2O       water,                         charge  0
  OH        hydroxide, terminal,           charge -1

Bridging ligands (dimers/trimers only)
  mu-OH    bridging hydroxide,  charge -1
  mu-HCOO  bridging formate,    charge -1

Constraints
-----------
  - Net complex charge = 0
  - Coordination number per metal: 3–7
  - Bidentate formate (HCOO:bi) counts as CN 2 per ligand
  - All symmetry-distinct isomers generated (for monodentate-only complexes)

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
from itertools import combinations_with_replacement, product
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from molbuilder.api import build, dimer, trimer
from molbuilder.core.geometry import infer_geometry
from molbuilder.core.isomers import enumerate_isomers
from molbuilder.output.poscar_writer import poscar_to_string

# ── ligand charges ────────────────────────────────────────────────────────────

# Monodentate terminal ligands
MONO_LIGANDS  = ["HCOO", "HCOOH", "H2O", "OH"]
MONO_CHARGE   = {"HCOO": -1, "HCOOH": 0, "H2O": 0, "OH": -1}
MONO_CN       = {"HCOO": 1, "HCOOH": 1, "H2O": 1, "OH": 1}

# Bidentate chelating ligands
BI_LIGANDS    = ["HCOO:bi"]
BI_CHARGE     = {"HCOO:bi": -1}
BI_CN         = {"HCOO:bi": 2}

BRIDGE_LIGANDS = ["mu-OH", "mu-HCOO"]
BRIDGE_CHARGE  = {"mu-OH": -1, "mu-HCOO": -1}

GEOMETRY_FOR_CN = {3: "tp", 4: "tet", 5: "sqpy", 6: "oct", 7: "pbp"}
EXTRA_GEOM      = {4: ["sqp"], 5: ["tbp"]}

OUTPUT_ROOT = Path("poscar")
CSV_FILE    = Path("ni_complexes_summary.csv")

# Set to False to skip isomer enumeration and generate only one
# representative structure per ligand combination / geometry.
GENERATE_ISOMERS = False

# ── helpers ───────────────────────────────────────────────────────────────────

def geometries_for_cn(cn):
    return [GEOMETRY_FOR_CN[cn]] + EXTRA_GEOM.get(cn, [])

def combo_label(ligands):
    c = Counter(ligands)
    return "_".join(f"{l.replace(':','')}{n}" for l, n in sorted(c.items()))

def safe(s):
    return s.replace(":", "-").replace("/", "-").replace(" ", "_")

def write_poscar(mol, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(poscar_to_string(mol))

def mol_to_row(mol, structure_type, ox, cn, geom, lig_combo,
               bridge, n_bridges, arrangement, isomer_label, filename):
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


# ── MONOMERS ─────────────────────────────────────────────────────────────────

def generate_monomers(rows):
    print("\n" + "="*60)
    print("  MONOMERS")
    print("="*60)
    n = 0

    for ox in [2, 3]:
        ox_label = "NiII" if ox == 2 else "NiIII"

        # ── pure monodentate combos ────────────────────────────────────────────
        for cn in range(3, 8):
            for combo in combinations_with_replacement(MONO_LIGANDS, cn):
                lig_charge = sum(MONO_CHARGE[l] for l in combo)
                if ox + lig_charge != 0:
                    continue
                for geom in geometries_for_cn(cn):
                    if GENERATE_ISOMERS:
                        iso_list = enumerate_isomers(list(combo), geom)
                    else:
                        iso_list = [{"site_assignment": list(combo), "label": "only"}]
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
                        label = iso["label"]
                        cl    = combo_label(combo)
                        path  = (OUTPUT_ROOT / "monomer" / ox_label / f"CN{cn}" /
                                 f"{safe(cl)}_{geom}_{safe(label)}.POSCAR")
                        write_poscar(mol, path)
                        n += 1
                        print(f"  ✓ monomer {ox_label} CN{cn} {geom:5s}  "
                              f"{cl:35s}  {label:10s}  {mol.formula}")
                        rows.append(mol_to_row(mol, "monomer", ox, cn, geom,
                                               cl, None, 0, None, label, path))

        # ── combos including bidentate HCOO:bi ────────────────────────────────
        # n_bi bidentate + mono_combo monodentate, total CN = n_bi*2 + len(mono) ∈ 3-7
        for n_bi in range(1, 4):
            for n_mono in range(0, 8 - n_bi * 2):
                cn = n_bi * 2 + n_mono
                if not (3 <= cn <= 7):
                    continue
                for mono_combo in (combinations_with_replacement(MONO_LIGANDS, n_mono)
                                   if n_mono > 0 else [()]):
                    lig_charge = (n_bi * BI_CHARGE["HCOO:bi"] +
                                  sum(MONO_CHARGE[l] for l in mono_combo))
                    if ox + lig_charge != 0:
                        continue
                    full_combo = ["HCOO:bi"] * n_bi + list(mono_combo)
                    for geom in geometries_for_cn(cn):
                        try:
                            mol = build("Ni", ox=ox,
                                        ligands=full_combo,
                                        geometry=geom)
                            if isinstance(mol, list):
                                mol = mol[0]
                        except Exception as e:
                            print(f"  ✗ {ox_label} CN{cn} {geom} {full_combo}: {e}")
                            continue
                        label = getattr(mol, "label", "only")
                        cl    = combo_label(full_combo)
                        path  = (OUTPUT_ROOT / "monomer" / ox_label / f"CN{cn}" /
                                 f"{safe(cl)}_{geom}_{safe(label)}.POSCAR")
                        write_poscar(mol, path)
                        n += 1
                        print(f"  ✓ monomer {ox_label} CN{cn} {geom:5s}  "
                              f"{cl:35s}  {label:10s}  {mol.formula}")
                        rows.append(mol_to_row(mol, "monomer", ox, cn, geom,
                                               cl, None, 0, None, label, path))
    return n


# ── DIMERS ────────────────────────────────────────────────────────────────────

# Minimum number of bridge ligands required for dimers (enforces bridging character)
MIN_BRIDGES_DIMER = 2

def generate_dimers(rows):
    print("\n" + "="*60)
    print("  DIMERS")
    print("="*60)
    n = 0

    for ox in [2, 3]:
        ox_label = "NiII" if ox == 2 else "NiIII"
        for bridge in BRIDGE_LIGANDS:
            bc = BRIDGE_CHARGE[bridge]
            for n_bridges in range(MIN_BRIDGES_DIMER, 4):   # 2 or 3 bridges
                # Pure monodentate terminal ligands
                for n_term in range(0, 8 - n_bridges):
                    for terminal in (combinations_with_replacement(MONO_LIGANDS, n_term)
                                     if n_term > 0 else [()]):
                        tc = sum(MONO_CHARGE[l] for l in terminal)
                        # Charge: 2*ox (two metals) + 2*terminal_charge + n_bridges*bc = 0
                        if 2 * ox + 2 * tc + n_bridges * bc != 0:
                            continue
                        cn = n_term + n_bridges
                        if not (3 <= cn <= 7):
                            continue
                        try:
                            mol = dimer("Ni", ox=ox,
                                        terminal=list(terminal),
                                        bridge=bridge, n=n_bridges)
                        except Exception as e:
                            print(f"  ✗ dimer {ox_label} {n_bridges}x{bridge}: {e}")
                            continue
                        if mol.charge != 0:
                            continue
                        geom  = GEOMETRY_FOR_CN[cn]
                        cl    = combo_label(list(terminal))
                        path  = (OUTPUT_ROOT / "dimer" / ox_label / f"CN{cn}" /
                                 f"{safe(cl)}_{n_bridges}x{safe(bridge)}.POSCAR")
                        write_poscar(mol, path)
                        n += 1
                        clash_note = f"  ⚠ {len(mol._clash_warnings)} clashes" if getattr(mol, '_clash_warnings', None) else ""
                        print(f"  ✓ dimer  {ox_label} CN{cn}  {n_bridges}x{bridge:10s}  "
                              f"term={list(terminal)}  {mol.formula}{clash_note}")
                        rows.append(mol_to_row(mol, "dimer", ox, cn, geom,
                                               cl, bridge, n_bridges, None,
                                               "only", path))

                # Terminal with bidentate formate
                for n_bi in range(1, 3):
                    for n_mono in range(0, 6 - n_bridges - n_bi * 2):
                        cn = n_bridges + n_bi * 2 + n_mono
                        if not (3 <= cn <= 7):
                            continue
                        for mono_combo in (combinations_with_replacement(MONO_LIGANDS, n_mono)
                                           if n_mono > 0 else [()]):
                            tc = (n_bi * BI_CHARGE["HCOO:bi"] +
                                  sum(MONO_CHARGE[l] for l in mono_combo))
                            if 2 * ox + 2 * tc + n_bridges * bc != 0:
                                continue
                            terminal = ["HCOO:bi"] * n_bi + list(mono_combo)
                            try:
                                mol = dimer("Ni", ox=ox,
                                            terminal=terminal,
                                            bridge=bridge, n=n_bridges)
                            except Exception as e:
                                continue
                            if mol.charge != 0:
                                continue
                            geom = GEOMETRY_FOR_CN[cn]
                            cl   = combo_label(terminal)
                            path = (OUTPUT_ROOT / "dimer" / ox_label / f"CN{cn}" /
                                    f"{safe(cl)}_{n_bridges}x{safe(bridge)}.POSCAR")
                            write_poscar(mol, path)
                            n += 1
                            clash_note = f"  ⚠ {len(mol._clash_warnings)} clashes" if getattr(mol, '_clash_warnings', None) else ""
                            print(f"  ✓ dimer  {ox_label} CN{cn}  {n_bridges}x{bridge:10s}  "
                                  f"term={terminal}  {mol.formula}{clash_note}")
                            rows.append(mol_to_row(mol, "dimer", ox, cn, geom,
                                                   cl, bridge, n_bridges, None,
                                                   "only", path))
    return n


# ── TRIMERS ───────────────────────────────────────────────────────────────────

# Minimum bridge count per edge for trimers
# triangular: 3 edges, linear: 2 edges — each edge needs at least 1 bridge
MIN_BRIDGES_PER_EDGE_TRIMER = 1

def generate_trimers(rows):
    print("\n" + "="*60)
    print("  TRIMERS")
    print("="*60)
    n = 0

    for ox in [2, 3]:
        ox_label = "NiII" if ox == 2 else "NiIII"
        for bridge in BRIDGE_LIGANDS:
            bc = BRIDGE_CHARGE[bridge]
            for n_term in range(0, 6):
                # Pure monodentate terminal
                for terminal in (combinations_with_replacement(MONO_LIGANDS, n_term)
                                 if n_term > 0 else [()]):
                    tc = sum(MONO_CHARGE[l] for l in terminal)
                    for arrangement in ["linear", "triangular"]:
                        # triangular: 3 edges → 3 bridge ligands total
                        # linear:     2 edges → 2 bridge ligands total
                        n_bridge_edges = 3 if arrangement == "triangular" else 2
                        # Charge: 3*ox + 3*tc + n_bridge_edges*bc = 0
                        if 3 * ox + 3 * tc + n_bridge_edges * bc != 0:
                            continue
                        cn_bridges_per_metal = 2 if arrangement == "triangular" else 1
                        cn = n_term + cn_bridges_per_metal
                        if not (3 <= cn <= 7):
                            continue
                        try:
                            mol = trimer("Ni", ox=ox,
                                         terminal=list(terminal),
                                         bridge=bridge,
                                         arrangement=arrangement)
                        except Exception as e:
                            continue
                        if mol.charge != 0:
                            continue
                        geom = GEOMETRY_FOR_CN[cn]
                        cl   = combo_label(list(terminal))
                        path = (OUTPUT_ROOT / "trimer" / ox_label / f"CN{cn}" /
                                f"{safe(cl)}_{safe(bridge)}_{arrangement}.POSCAR")
                        write_poscar(mol, path)
                        n += 1
                        clash_note = f"  ⚠ {len(mol._clash_warnings)} clashes" if getattr(mol, '_clash_warnings', None) else ""
                        print(f"  ✓ trimer {ox_label} CN{cn}  {bridge:10s}  "
                              f"term={list(terminal)}  {arrangement:12s}  {mol.formula}{clash_note}")
                        rows.append(mol_to_row(mol, f"trimer_{arrangement}", ox, cn, geom,
                                               cl, bridge, n_bridge_edges, arrangement, "only", path))

                # With bidentate terminal
                for n_bi in range(1, 3):
                    cn_bridges_per_metal = 2   # triangular always
                    cn = cn_bridges_per_metal + n_bi * 2 + n_term
                    if not (3 <= cn <= 7):
                        continue
                    for mono_combo in (combinations_with_replacement(MONO_LIGANDS, n_term)
                                       if n_term > 0 else [()]):
                        tc = (n_bi * BI_CHARGE["HCOO:bi"] +
                              sum(MONO_CHARGE[l] for l in mono_combo))
                        for arrangement in ["linear", "triangular"]:
                            n_bridge_edges = 3 if arrangement == "triangular" else 2
                            if 3 * ox + 3 * tc + n_bridge_edges * bc != 0:
                                continue
                            terminal = ["HCOO:bi"] * n_bi + list(mono_combo)
                            try:
                                mol = trimer("Ni", ox=ox,
                                             terminal=terminal,
                                             bridge=bridge,
                                             arrangement=arrangement)
                            except Exception as e:
                                continue
                            if mol.charge != 0:
                                continue
                            geom = GEOMETRY_FOR_CN[cn]
                            cl   = combo_label(terminal)
                            path = (OUTPUT_ROOT / "trimer" / ox_label / f"CN{cn}" /
                                    f"{safe(cl)}_{safe(bridge)}_{arrangement}.POSCAR")
                            write_poscar(mol, path)
                            n += 1
                            clash_note = f"  ⚠ {len(mol._clash_warnings)} clashes" if getattr(mol, '_clash_warnings', None) else ""
                            print(f"  ✓ trimer {ox_label} CN{cn}  {bridge:10s}  "
                                  f"term={terminal}  {arrangement}  {mol.formula}{clash_note}")
                            rows.append(mol_to_row(mol, f"trimer_{arrangement}", ox, cn, geom,
                                                   cl, bridge, n_bridge_edges, arrangement, "only", path))
    return n


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)
    rows = []

    n_mono   = generate_monomers(rows)
    n_dimer  = generate_dimers(rows)
    n_trimer = generate_trimers(rows)

    if rows:
        with CSV_FILE.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n{'='*60}")
    print(f"  Monomers : {n_mono}")
    print(f"  Dimers   : {n_dimer}")
    print(f"  Trimers  : {n_trimer}")
    print(f"  Total    : {n_mono + n_dimer + n_trimer} POSCAR files")
    print(f"  CSV      : {CSV_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
