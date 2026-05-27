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
  NH3       ammonia, monodentate,          charge  0

Bridging ligands (dimers/trimers only)
  mu-OH    bridging hydroxide,  charge -1
  mu-HCOO  bridging formate,    charge -1

Constraints
-----------
  - Net complex charge = 0
  - Coordination number per metal: 3–7
  - Bidentate formate (HCOO:bi) counts as CN 2 per ligand
  - All symmetry-distinct isomers generated (for monodentate-only complexes)
  - Multi-bridge trimers: linear trimers with n_bridges_per_pair = 2 or 3

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

from molbuilder.api import build, build_isomers, dimer, trimer
from molbuilder.core.geometry import infer_geometry
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
                    iso_list = build_isomers("Ni", ox=ox,
                                             ligands=list(combo),
                                             geometry=geom)
                    for mol in iso_list:
                        try:
                            pass
                        except Exception as e:
                            print(f"  ✗ {ox_label} CN{cn} {geom} {combo}: {e}")
                            continue
                        label = getattr(mol, "label", "only")
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
                        except Exception as e:
                            print(f"  ✗ {ox_label} CN{cn} {geom} {full_combo}: {e}")
                            continue
                        label = "only"
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
                # Pure monodentate terminal ligands
                # Charge = 2*ox (two metals) + 2*tc (terminal on each metal)
                #        + n_bridges*bc (bridge ligands shared between metals)
                for n_term in range(0, 8 - n_bridges):
                    for terminal in (combinations_with_replacement(MONO_LIGANDS, n_term)
                                     if n_term > 0 else [()]):
                        tc = sum(MONO_CHARGE[l] for l in terminal)
                        if 2 * ox + 2 * tc + n_bridges * bc != 0:
                            continue
                        cn = n_term + n_bridges
                        if not (3 <= cn <= 7):
                            continue
                        try:
                            mol = dimer("Ni", ox=ox,
                                        terminal=list(terminal),
                                        bridge=bridge, n=n_bridges)
                        except ValueError as e:
                            print(f"  ✗ dimer {ox_label} CN{cn} {n_bridges}x{bridge} "
                                  f"term={list(terminal)}: {e}")
                            continue
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
                        print(f"  ✓ dimer  {ox_label} CN{cn}  {n_bridges}x{bridge:10s}  "
                              f"term={list(terminal)}  {mol.formula}")
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
                            print(f"  ✓ dimer  {ox_label} CN{cn}  {n_bridges}x{bridge:10s}  "
                                  f"term={terminal}  {mol.formula}")
                            rows.append(mol_to_row(mol, "dimer", ox, cn, geom,
                                                   cl, bridge, n_bridges, None,
                                                   "only", path))
    return n


# ── TRIMERS ───────────────────────────────────────────────────────────────────

def generate_trimers(rows):
    print("\n" + "="*60)
    print("  TRIMERS")
    print("="*60)
    n = 0

    # Charge formula: 3*ox + 3*tc + n_bridge_pairs * n_bridges_per_pair * bc == 0
    # CN per metal: n_term + 2 * n_bridges_per_pair  (each metal touches 2 pairs)
    # Note: for linear arrangement, endpoint metals touch only 1 pair, so their real
    # CN = n_term + n_bridges_per_pair. But we use 2*nbpp for the geometry inference
    # to avoid under-estimating for the middle metal.

    BRIDGE_PAIRS = {"linear": 2, "triangular": 3}

    # ── standard trimers: n_bridges_per_pair = 1 ──────────────────────────────
    for ox in [2, 3]:
        ox_label = "NiII" if ox == 2 else "NiIII"
        for bridge in BRIDGE_LIGANDS:
            bc = BRIDGE_CHARGE[bridge]
            for n_term in range(0, 6):
                # Pure monodentate terminal
                for terminal in (combinations_with_replacement(MONO_LIGANDS, n_term)
                                 if n_term > 0 else [()]):
                    tc  = sum(MONO_CHARGE[l] for l in terminal)
                    cn  = n_term + 2   # 1 bridge per pair × 2 pairs per metal
                    if not (3 <= cn <= 7):
                        continue
                    for arrangement in ["linear", "triangular"]:
                        n_bp = BRIDGE_PAIRS[arrangement]
                        if 3 * ox + 3 * tc + n_bp * bc != 0:
                            continue
                        try:
                            mol = trimer("Ni", ox=ox,
                                         terminal=list(terminal),
                                         bridge=bridge,
                                         arrangement=arrangement)
                        except Exception:
                            continue
                        if mol.charge != 0:
                            continue
                        geom = GEOMETRY_FOR_CN[cn]
                        cl   = combo_label(list(terminal))
                        path = (OUTPUT_ROOT / "trimer" / ox_label / f"CN{cn}" /
                                f"{safe(cl)}_{safe(bridge)}_{arrangement}.POSCAR")
                        write_poscar(mol, path)
                        n += 1
                        print(f"  ✓ trimer {ox_label} CN{cn}  {bridge:10s}  "
                              f"term={list(terminal)}  {arrangement:12s}  {mol.formula}")
                        rows.append(mol_to_row(mol, f"trimer_{arrangement}", ox, cn, geom,
                                               cl, bridge, n_bp, arrangement, "only", path))

                # With bidentate terminal
                for n_bi in range(1, 3):
                    cn = 2 + n_bi * 2 + n_term
                    if not (3 <= cn <= 7):
                        continue
                    for mono_combo in (combinations_with_replacement(MONO_LIGANDS, n_term)
                                       if n_term > 0 else [()]):
                        tc = (n_bi * BI_CHARGE["HCOO:bi"] +
                              sum(MONO_CHARGE[l] for l in mono_combo))
                        terminal = ["HCOO:bi"] * n_bi + list(mono_combo)
                        for arrangement in ["linear", "triangular"]:
                            n_bp = BRIDGE_PAIRS[arrangement]
                            if 3 * ox + 3 * tc + n_bp * bc != 0:
                                continue
                            try:
                                mol = trimer("Ni", ox=ox,
                                             terminal=terminal,
                                             bridge=bridge,
                                             arrangement=arrangement)
                            except Exception:
                                continue
                            if mol.charge != 0:
                                continue
                            geom = GEOMETRY_FOR_CN[cn]
                            cl   = combo_label(terminal)
                            path = (OUTPUT_ROOT / "trimer" / ox_label / f"CN{cn}" /
                                    f"{safe(cl)}_{safe(bridge)}_{arrangement}.POSCAR")
                            write_poscar(mol, path)
                            n += 1
                            print(f"  ✓ trimer {ox_label} CN{cn}  {bridge:10s}  "
                                  f"term={terminal}  {arrangement}  {mol.formula}")
                            rows.append(mol_to_row(mol, f"trimer_{arrangement}", ox, cn, geom,
                                                   cl, bridge, n_bp, arrangement, "only", path))

    # ── multi-bridge trimers: n_bridges_per_pair = 2 or 3 ────────────────────
    # Charge = 3*ox + 3*tc + BRIDGE_PAIRS[arr] * nbpp * bc == 0
    # Feasible cases verified by geometry (triangular nbpp=2 with syn-syn HCOO/OH
    # causes inter-edge O overlap; only linear nbpp=2,3 and triangular nbpp≥4 work):
    #   - Linear, nbpp=3, OH:   3*2 + 0 + 2*3*(-1)=0, CN=6 (oct) ✓
    #   - Linear, nbpp=3, HCOO: 3*2 + 0 + 2*3*(-1)=0, CN=6 (oct) ✓
    #   - Triangular nbpp=2 is excluded — inter-edge O atoms always overlap
    #     in the syn-syn geometry regardless of M-M distance.
    MULTI_BRIDGE_CASES = [
        # (arrangement, nbpp, bridge, ox, terminal, label_suffix)
        # Linear, 3 bridges per pair, no terminal → Ni3(OH)6, Ni3(HCOO)6
        ("linear", 3, "mu-OH",   2, [], "triplebridge"),
        ("linear", 3, "mu-HCOO", 2, [], "triplebridge"),
        # NiIII linear 2-per-pair cases that are charge-neutral
        ("linear", 2, "mu-OH",   3, ["OH"], "doublebridge"),
        ("linear", 2, "mu-HCOO", 3, ["HCOO"], "doublebridge"),
    ]

    for arr, nbpp, bridge, ox, terminal, suffix in MULTI_BRIDGE_CASES:
        ox_label = "NiII" if ox == 2 else "NiIII"
        bc    = BRIDGE_CHARGE[bridge]
        n_bp  = BRIDGE_PAIRS[arr]
        tc    = sum(MONO_CHARGE.get(l, BI_CHARGE.get(l, 0)) for l in terminal)
        charge_check = 3 * ox + 3 * tc + n_bp * nbpp * bc
        if charge_check != 0:
            continue
        cn = len(terminal) + 2 * nbpp
        if not (3 <= cn <= 7):
            continue
        try:
            mol = trimer("Ni", ox=ox,
                         terminal=list(terminal),
                         bridge=bridge,
                         arrangement=arr,
                         n_bridges_per_pair=nbpp)
        except Exception as e:
            print(f"  ✗ trimer {ox_label} {arr} {nbpp}x{bridge} term={terminal}: {e}")
            continue
        if mol.charge != 0:
            continue
        geom = GEOMETRY_FOR_CN.get(cn, "oct")
        cl   = combo_label(list(terminal))
        fname = f"{safe(cl)}_{nbpp}x{safe(bridge)}_{arr}.POSCAR" if cl else f"{nbpp}x{safe(bridge)}_{arr}.POSCAR"
        path  = OUTPUT_ROOT / "trimer" / ox_label / f"CN{cn}" / fname
        write_poscar(mol, path)
        n += 1
        print(f"  ✓ trimer {ox_label} CN{cn}  {nbpp}x{bridge:10s}  "
              f"term={list(terminal)}  {arr:12s}  {mol.formula}")
        rows.append(mol_to_row(mol, f"trimer_{arr}", ox, cn, geom,
                               cl, bridge, n_bp * nbpp, arr, "only", path))
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
