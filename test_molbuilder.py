#!/usr/bin/env python3
"""
test_molbuilder.py
==================
Run all tests and print example outputs.

Usage:
    cd /path/to/  # parent of molbuilder/
    python test_molbuilder.py
"""

import sys
import os
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from molbuilder.api import build, dimer, trimer, poscar, xyz, info, list_ligands
from molbuilder.output.poscar_writer import poscar_to_string
from molbuilder.core.bond_lengths import get_bond_length
from molbuilder.core.geometry import suggest_geometry


PASS = "✓"
FAIL = "✗"
results = []

def test(name, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"  {FAIL}  {name}")
        print(f"     → {e}")
        traceback.print_exc()
        results.append((name, False, str(e)))

outdir = Path("/tmp/molbuilder_tests")
outdir.mkdir(exist_ok=True)

print()
print("=" * 60)
print("  molbuilder test suite")
print("=" * 60)

# ── Bond length database ────────────────────────────────────────────
print("\n[1] Bond length database")

def t_bl_exact():
    bl = get_bond_length("Fe", 3, "O", "oct")
    assert abs(bl - 2.00) < 0.05, f"Expected ~2.00, got {bl}"

def t_bl_fallback():
    bl = get_bond_length("Fe", 3, "N", "oct")
    assert 1.8 < bl < 2.5, f"Unexpected value {bl}"

def t_bl_cov_fallback():
    bl = get_bond_length("Fe", 2, "Xe", "oct")  # exotic: should use cov radii
    assert bl > 0, "Should return positive value"

test("Exact lookup Fe(III)-O-oct",     t_bl_exact)
test("Fallback Fe(III)-N-oct",         t_bl_fallback)
test("Covalent radii fallback (exotic)", t_bl_cov_fallback)

# ── Geometry engines ────────────────────────────────────────────────
print("\n[2] Geometry engines")

from molbuilder.core.geometry import get_geometry_vectors
import numpy as np

def t_oct():
    vecs = get_geometry_vectors("oct")
    assert len(vecs) == 6
    # All unit vectors
    for v in vecs:
        assert abs(np.linalg.norm(v) - 1.0) < 1e-6
    # All mutually 90° or 180°
    for i in range(6):
        for j in range(i+1, 6):
            d = abs(np.dot(vecs[i], vecs[j]))
            assert d < 0.01 or abs(d - 1.0) < 0.01, f"Unexpected angle {d}"

def t_sqp():
    vecs = get_geometry_vectors("sqp")
    assert len(vecs) == 4

def t_tet():
    vecs = get_geometry_vectors("tet")
    assert len(vecs) == 4
    # All angles should be tetrahedral (~109.47°)
    cos_tet = -1/3
    for i in range(4):
        for j in range(i+1, 4):
            d = np.dot(vecs[i], vecs[j])
            assert abs(d - cos_tet) < 0.01, f"Expected cos({cos_tet:.3f}), got {d:.3f}"

def t_tbp():
    vecs = get_geometry_vectors("tbp")
    assert len(vecs) == 5

def t_lin():
    vecs = get_geometry_vectors("lin")
    assert len(vecs) == 2
    assert abs(np.dot(vecs[0], vecs[1]) + 1.0) < 1e-6  # antiparallel

test("Octahedral geometry",            t_oct)
test("Square planar geometry",         t_sqp)
test("Tetrahedral geometry",           t_tet)
test("Trigonal bipyramidal geometry",  t_tbp)
test("Linear geometry",                t_lin)

# ── Ligand library ──────────────────────────────────────────────────
print("\n[3] Ligand library")

from molbuilder.ligands.library import get_ligand

def t_lig_CO():
    l = get_ligand("CO")
    assert l.donor_atoms == ["C"]
    assert l.charge == 0

def t_lig_Cl():
    l = get_ligand("Cl")
    assert l.charge == -1
    assert l.donor_atoms == ["Cl"]

def t_lig_bpy():
    l = get_ligand("bpy")
    assert l.denticity == 2
    assert l.bite_angle is not None

def t_lig_alias():
    l = get_ligand("aqua")
    assert l.name == "H2O"

def t_lig_EDTA():
    l = get_ligand("EDTA")
    assert l.denticity == 6
    assert l.charge == -4

test("CO ligand",       t_lig_CO)
test("Cl- ligand",      t_lig_Cl)
test("bpy bidentate",   t_lig_bpy)
test("Alias 'aqua'",    t_lig_alias)
test("EDTA hexadentate",t_lig_EDTA)

# ── Complex builder ─────────────────────────────────────────────────
print("\n[4] Mononuclear complex builder")

def t_FeIII_oct():
    mol = build("Fe", ox=3, ligands=["Cl","Cl","Cl","H2O","H2O","H2O"])
    assert mol is not None
    # 1 Fe + 6 donors
    assert len(mol.atoms) >= 7
    syms = [a.symbol for a in mol.atoms]
    assert "Fe" in syms
    assert syms.count("Cl") == 3
    assert syms.count("O") == 3

def t_PdII_sqp():
    mol = build("Pd", ox=2, ligands=["Cl","Cl","NH3","NH3"], geometry="sqp")
    assert "Pd" in [a.symbol for a in mol.atoms]

def t_CrIII_all_CO():
    mol = build("Cr", ox=0, ligands=["CO"]*6, geometry="oct")
    syms = [a.symbol for a in mol.atoms]
    assert syms.count("C") == 6
    assert syms.count("O") == 6  # CO body atoms

def t_NiII_tet():
    mol = build("Ni", ox=2, ligands=["Cl","Cl","Cl","Cl"], geometry="tet")
    assert mol.charge == -2   # Ni(II) + 4 Cl-

def t_PtII_sqp():
    mol = build("Pt", ox=2, ligands=["Cl","Cl","NH3","NH3"], geometry="sqp")
    assert "Pt" in [a.symbol for a in mol.atoms]

def t_CoIII_tris_en():
    mol = build("Co", ox=3, ligands=["en","en","en"], geometry="oct")
    assert "Co" in [a.symbol for a in mol.atoms]

def t_charge_calc():
    # [Fe(H2O)6]3+: Fe(III) + 6 H2O → charge = +3
    mol = build("Fe", ox=3, ligands=["H2O"]*6)
    assert mol.charge == 3, f"Expected +3, got {mol.charge}"

def t_charge_anionic():
    # [FeCl6]3-: Fe(III) + 6 Cl- → charge = 3-6 = -3
    mol = build("Fe", ox=3, ligands=["Cl"]*6)
    assert mol.charge == -3, f"Expected -3, got {mol.charge}"

test("[Fe(Cl)3(H2O)3] octahedral",  t_FeIII_oct)
test("[Pd(Cl)2(NH3)2] square planar", t_PdII_sqp)
test("[Cr(CO)6] octahedral",        t_CrIII_all_CO)
test("[NiCl4]2- tetrahedral",       t_NiII_tet)
test("[Pt(NH3)2Cl2] square planar", t_PtII_sqp)
test("[Co(en)3]3+ octahedral",      t_CoIII_tris_en)
test("Charge +3 for [Fe(H2O)6]3+",  t_charge_calc)
test("Charge -3 for [FeCl6]3-",     t_charge_anionic)

# ── POSCAR output ───────────────────────────────────────────────────
print("\n[5] POSCAR output")

def t_poscar_writes():
    mol = build("Fe", ox=3, ligands=["Cl","Cl","Cl","H2O","H2O","H2O"])
    path = poscar(mol, str(outdir / "FeCl3_H2O3.POSCAR"))
    assert Path(path).exists()
    content = open(path).read()
    assert "Fe" in content
    assert "Cl" in content
    assert len(content.splitlines()) >= 10

def t_poscar_structure():
    mol = build("Pd", ox=2, ligands=["NH3","NH3","Cl","Cl"], geometry="sqp")
    content = poscar_to_string(mol)
    lines = content.splitlines()
    # Line 2 should be scaling factor ~1.0
    assert abs(float(lines[1].strip()) - 1.0) < 0.01
    # Lines 3-5 should be lattice vectors (3 floats each)
    for i in [2, 3, 4]:
        vals = lines[i].split()
        assert len(vals) == 3
        floats = [float(v) for v in vals]
        assert all(f >= 0 for f in floats)

def t_poscar_vacuum():
    mol = build("Fe", ox=2, ligands=["H2O"]*6)
    c15 = poscar_to_string(mol, vacuum=15.0)
    c20 = poscar_to_string(mol, vacuum=20.0)
    # Larger vacuum → larger box → larger lattice vector
    def box_size(content):
        lines = content.splitlines()
        return float(lines[2].split()[0])
    assert box_size(c20) > box_size(c15), "Larger vacuum should give larger box"

def t_xyz_writes():
    mol = build("Co", ox=3, ligands=["NH3"]*6)
    path = xyz(mol, str(outdir / "Co_NH3_6.xyz"))
    assert Path(path).exists()
    content = open(path).read()
    lines = content.splitlines()
    n_atoms = int(lines[0].strip())
    assert n_atoms == len(mol.atoms)

test("POSCAR file written",          t_poscar_writes)
test("POSCAR structure valid",       t_poscar_structure)
test("POSCAR vacuum box scaling",    t_poscar_vacuum)
test("XYZ file written",             t_xyz_writes)

# ── Dimers ──────────────────────────────────────────────────────────
print("\n[6] Dimer builder")

def t_dimer_Rh():
    mol = dimer("Rh", ox=1, terminal=["CO","CO"], bridge="mu-Cl", n=2)
    syms = [a.symbol for a in mol.atoms]
    assert syms.count("Rh") == 2

def t_dimer_Pd():
    mol = dimer("Pd", ox=2, terminal=["Cl"], bridge="mu-Cl", n=2, geometry="sqp")
    syms = [a.symbol for a in mol.atoms]
    assert syms.count("Pd") == 2

def t_dimer_poscar():
    mol = dimer("Rh", ox=1, terminal=["CO","CO"], bridge="mu-Cl", n=2)
    path = poscar(mol, str(outdir / "Rh2_dimer.POSCAR"))
    assert Path(path).exists()

test("[Rh(CO)2(μ-Cl)]2 dimer",   t_dimer_Rh)
test("[Pd2(μ-Cl)2Cl2] dimer",    t_dimer_Pd)
test("Dimer POSCAR output",       t_dimer_poscar)

# ── Trimers ─────────────────────────────────────────────────────────
print("\n[7] Trimer builder")

def t_trimer_tri():
    mol = trimer("Ru", ox=0, terminal=["CO","CO","CO","CO"],
                 bridge="mu-Cl", arrangement="triangular")
    syms = [a.symbol for a in mol.atoms]
    assert syms.count("Ru") == 3

def t_trimer_lin():
    mol = trimer("Fe", ox=3, terminal=["Cl","Cl"],
                 bridge="mu-OH", arrangement="linear")
    syms = [a.symbol for a in mol.atoms]
    assert syms.count("Fe") == 3

def t_trimer_poscar():
    mol = trimer("Ru", ox=0, terminal=["CO","CO"],
                 bridge="mu-Cl", arrangement="triangular")
    path = poscar(mol, str(outdir / "Ru3_trimer.POSCAR"))
    assert Path(path).exists()

test("Ru3 triangular trimer",     t_trimer_tri)
test("Fe3 linear trimer",         t_trimer_lin)
test("Trimer POSCAR output",      t_trimer_poscar)

# ── Summary ─────────────────────────────────────────────────────────
print()
print("=" * 60)
n_pass = sum(1 for _, ok, _ in results if ok)
n_fail = sum(1 for _, ok, _ in results if not ok)
print(f"  Results: {n_pass}/{len(results)} passed  |  {n_fail} failed")
print("=" * 60)

if n_pass == len(results):
    print("\n  All tests passed! Output files in:", outdir)
    print()
    # Print a sample POSCAR
    mol = build("Fe", ox=3, ligands=["Cl","Cl","Cl","H2O","H2O","H2O"])
    print("  Sample POSCAR for [FeCl3(H2O)3]:")
    print("  " + "-"*56)
    for line in poscar_to_string(mol).splitlines():
        print("  " + line)
else:
    print(f"\n  {n_fail} test(s) failed. See output above.")
    sys.exit(1)
