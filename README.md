# molbuilder

**Gas-phase transition metal complex builder → POSCAR (VASP)**

Build mononuclear and polynuclear transition metal complexes from a simple Python API or CLI, with automatic bond lengths, full 3D ligand geometry (including H atoms), coordination geometries, isomer enumeration, and VASP-ready POSCAR output.

---

## Installation

```bash
pip install rdkit numpy scipy
# Then drop the molbuilder/ directory into your project
```

Dependencies: `rdkit`, `numpy`, `scipy`

---

## Python API

```python
from molbuilder.api import build, build_all_isomers, dimer, trimer, poscar, xyz, info
```

### Mononuclear complexes

```python
# Octahedral Fe(III): [FeCl3(H2O)3]  — includes H atoms on water
mol = build("Fe", ox=3, ligands=["Cl","Cl","Cl","H2O","H2O","H2O"])
poscar(mol, "FeCl3_H2O3.POSCAR")

# Square planar Pd(II): [Pd(bpy)Cl2]
mol = build("Pd", ox=2, ligands=["bpy","Cl","Cl"], geometry="sqp")
poscar(mol, "Pd_bpy_Cl2.POSCAR")

# Tetrahedral Ni(II): [NiCl4]2-
mol = build("Ni", ox=2, ligands=["Cl","Cl","Cl","Cl"], geometry="tet")
poscar(mol, "NiCl4_tet.POSCAR")

# Cr(0) hexacarbonyl: [Cr(CO)6]
mol = build("Cr", ox=0, ligands=["CO"]*6, geometry="oct")
poscar(mol, "Cr_CO6.POSCAR")

# Custom SMILES ligand (piperazine N-donor)
mol = build("Cu", ox=2, ligands=["N1CCNCC1","N1CCNCC1","H2O","H2O"])
poscar(mol, "Cu_piperazine_H2O2.POSCAR")

# Check the structure
info(mol)
```

### All isomers

`build_all_isomers` automatically enumerates all symmetry-distinct configurations:

```python
from molbuilder.api import build_all_isomers, poscar

# [FeCl3(H2O)3] has 2 isomers: fac and mer
isomers = build_all_isomers("Fe", ox=3,
    ligands=["Cl","Cl","Cl","H2O","H2O","H2O"])

for iso in isomers:
    print(iso["label"], iso["molecule"].formula)
    poscar(iso["molecule"], f"FeCl3_H2O3_{iso['label']}.POSCAR")
# → fac  Cl3FeH6O3  → FeCl3_H2O3_fac.POSCAR
# → mer  Cl3FeH6O3  → FeCl3_H2O3_mer.POSCAR

# [PtCl2(NH3)2]: cis-platin and trans-platin
isomers = build_all_isomers("Pt", ox=2,
    ligands=["Cl","Cl","NH3","NH3"], geometry="sqp")
for iso in isomers:
    poscar(iso["molecule"], f"PtCl2_NH3_2_{iso['label']}.POSCAR")
# → cis   → PtCl2_NH3_2_cis.POSCAR
# → trans → PtCl2_NH3_2_trans.POSCAR

# [CoCl4(H2O)2]: cis and trans
isomers = build_all_isomers("Co", ox=2,
    ligands=["Cl","Cl","Cl","Cl","H2O","H2O"])
```

### Dimers

```python
# [Rh(CO)2(μ-Cl)]2
mol = dimer("Rh", ox=1, terminal=["CO","CO"], bridge="mu-Cl", n=2)
poscar(mol, "Rh2_mu-Cl2_CO4.POSCAR")

# [Pd2(μ-Cl)2Cl2] square planar
mol = dimer("Pd", ox=2, terminal=["Cl"], bridge="mu-Cl", n=2, geometry="sqp")
poscar(mol, "Pd2_mu-Cl2_Cl2.POSCAR")

# M–M bonded dimer with custom distance
mol = dimer("Re", ox=3, terminal=["Cl","Cl","Cl"], bridge="mu-Cl", n=2,
            mm_bond=True, mm_distance=2.22)
poscar(mol, "Re2_quadruple_bond.POSCAR")
```

### Trimers

```python
# Triangular [Ru3(CO)12]-style
mol = trimer("Ru", ox=0, terminal=["CO","CO","CO","CO"],
             bridge="mu-CO", arrangement="triangular")
poscar(mol, "Ru3_triangular_CO12.POSCAR")

# Linear Fe3 with μ-OH bridges
mol = trimer("Fe", ox=3, terminal=["H2O","H2O"],
             bridge="mu-OH", arrangement="linear")
poscar(mol, "Fe3_linear_mu-OH.POSCAR")
```

### Also write XYZ

```python
mol = build("Fe", ox=3, ligands=["Cl"]*6)
poscar(mol, "FeCl6.POSCAR")
xyz(mol,   "FeCl6.xyz")
```

---

## CLI

```bash
PYTHONPATH=/path/to/project python molbuilder/cli.py [options]
```

### Examples

```bash
# [FeCl3(H2O)3] octahedral — print POSCAR to stdout and save file
PYTHONPATH=. python molbuilder/cli.py \
    --metal Fe --ox 3 \
    --ligands Cl Cl Cl H2O H2O H2O \
    --out FeCl3_H2O3.POSCAR --print

# [Pd(bpy)Cl2] square planar
PYTHONPATH=. python molbuilder/cli.py \
    --metal Pd --ox 2 \
    --ligands bpy Cl Cl --geometry sqp \
    --out Pd_bpy_Cl2.POSCAR

# All isomers of [FeCl3(H2O)3] → FeCl3_H2O3_fac.POSCAR + FeCl3_H2O3_mer.POSCAR
PYTHONPATH=. python molbuilder/cli.py \
    --metal Fe --ox 3 \
    --ligands Cl Cl Cl H2O H2O H2O \
    --all-isomers --out FeCl3_H2O3.POSCAR

# cis/trans isomers of [PtCl2(NH3)2] → PtCl2_NH3_2_cis.POSCAR + ..._trans.POSCAR
PYTHONPATH=. python molbuilder/cli.py \
    --metal Pt --ox 2 \
    --ligands Cl Cl NH3 NH3 --geometry sqp \
    --all-isomers --out PtCl2_NH3_2.POSCAR

# [Rh(CO)2(μ-Cl)]2 dimer
PYTHONPATH=. python molbuilder/cli.py \
    --dimer --metal Rh --ox 1 \
    --ligands CO CO --bridge mu-Cl --n-bridges 2 \
    --out Rh2_mu-Cl2_CO4.POSCAR

# Ru3 triangular trimer
PYTHONPATH=. python molbuilder/cli.py \
    --trimer --metal Ru --ox 0 \
    --ligands CO CO CO CO --bridge mu-CO \
    --arrangement triangular --out Ru3_triangular_CO12.POSCAR

# List available ligands / geometries
PYTHONPATH=. python molbuilder/cli.py --list-ligands
PYTHONPATH=. python molbuilder/cli.py --list-geometries

# Custom SMILES ligand + standard ligands, also write XYZ
PYTHONPATH=. python molbuilder/cli.py \
    --metal Fe --ox 3 \
    --smiles-ligands "N1CCNCC1" --smiles-count 3 \
    --ligands Cl Cl Cl --xyz --out Fe_piperazine3_Cl3.POSCAR
```

### Extended CLI (custom POSCAR ligands, denticity modes, bridging)

```bash
# Standard build (same as cli.py)
PYTHONPATH=. python molbuilder/cli_extended.py \
    --metal Fe --ox 3 \
    --ligands Cl Cl Cl H2O H2O H2O \
    --out FeCl3_H2O3.POSCAR --print

# All isomers via extended CLI
PYTHONPATH=. python molbuilder/cli_extended.py \
    --metal Fe --ox 3 \
    --ligands Cl Cl Cl H2O H2O H2O \
    --all-isomers --out FeCl3_H2O3.POSCAR

# Custom POSCAR ligand
PYTHONPATH=. python molbuilder/cli_extended.py \
    --metal Fe --ox 2 \
    --ligands Cl Cl Cl Cl \
    --custom-ligand my_ligand.POSCAR \
    --custom-donor-atoms 0,2 \
    --custom-ligand-charge 0 \
    --custom-ligand-name myligand \
    --out Fe_custom_ligand.POSCAR

# Bidentate bpy with denticity mode
PYTHONPATH=. python molbuilder/cli_extended.py \
    --metal Fe --ox 2 \
    --ligand-mode bpy:bi \
    --ligand-mode Cl \
    --ligand-mode Cl \
    --geometry oct \
    --out Fe_bpy_Cl2.POSCAR

# Dimer with bridging formate
PYTHONPATH=. python molbuilder/cli_extended.py \
    --dimer --metal Fe --ox 3 \
    --ligands H2O H2O \
    --bridge-ligand HCOO:bi \
    --bridge-count 2 \
    --geometry oct \
    --out Fe2_formate_dimer.POSCAR
```

---

## Isomer Reference

| Formula | Geometry | Isomers |
|---------|----------|---------|
| MA₆     | oct      | 1 |
| MA₅B    | oct      | 1 |
| MA₄B₂   | oct      | 2 (cis, trans) |
| MA₃B₃   | oct      | 2 (fac, mer) |
| MA₄BC   | oct      | 2 |
| MA₂B₂C₂ | oct     | 5 |
| MA₃B₂C  | oct      | 3 |
| MA₄     | sqp      | 1 |
| MA₃B    | sqp      | 1 |
| MA₂B₂   | sqp      | 2 (cis, trans) |
| MA₂BC   | sqp      | 2 |

---

## Supported Geometries

| Key | Name | CN |
|-----|------|-----|
| `lin` | Linear | 2 |
| `bent` | Bent | 2 |
| `tp` | Trigonal planar | 3 |
| `tshaped` | T-shaped | 3 |
| `tet` / `td` | Tetrahedral | 4 |
| `sqp` / `sp` | Square planar | 4 |
| `seesaw` | See-saw | 4 |
| `tbp` | Trigonal bipyramidal | 5 |
| `sqpy` / `spy` | Square pyramidal | 5 |
| `oct` / `oh` | Octahedral | 6 |
| `tpr` | Trigonal prismatic | 6 |
| `pbp` | Pentagonal bipyramidal | 7 |
| `sapr` | Square antiprismatic | 8 |

---

## Ligand Library (selection)

| Name | Donor | Charge | Denticity |
|------|-------|--------|-----------|
| `CO` | C | 0 | 1 |
| `H2O` / `aqua` | O | 0 | 1 |
| `NH3` / `ammine` | N | 0 | 1 |
| `Cl`, `Br`, `I`, `F` | halide | −1 | 1 |
| `CN` | C | −1 | 1 |
| `OH` | O | −1 | 1 |
| `SCN` | S | −1 | 1 |
| `PPh3` | P | 0 | 1 |
| `py` | N | 0 | 1 |
| `MeCN` | N | 0 | 1 |
| `en` | N,N | 0 | 2 |
| `bpy` / `bipy` | N,N | 0 | 2 |
| `phen` | N,N | 0 | 2 |
| `acac` | O,O | −1 | 2 |
| `ox` | O,O | −2 | 2 |
| `tpy` / `terpy` | N,N,N | 0 | 3 |
| `EDTA` | N,N,O,O,O,O | −4 | 6 |
| `Cp` | C×5 (η⁵) | −1 | cyclopentadienyl |
| `mu-Cl`, `mu-OH`, `mu-O` | bridging | varies | bridging |

---

## Bond Length Sources

Bond lengths are drawn from a database of CSD-averaged values (Orpen et al. 1989) keyed by `(metal, oxidation_state, donor_atom, geometry)`. The fallback hierarchy is:

1. Exact match
2. Same metal/ox/donor, any geometry (averaged)
3. Same metal/donor, any oxidation state (averaged)
4. Sum of Alvarez (2008) covalent radii + 0.1 Å

---

## POSCAR Format

The output is a standard VASP POSCAR file:
- Atoms centered in a cubic vacuum box (default 15 Å padding per side → 30 Å total per direction)
- Species sorted by atomic number (heaviest first: metal → donor atoms → H)
- Cartesian coordinates in Ångströms
- Formal charge and spin multiplicity in the comment line
