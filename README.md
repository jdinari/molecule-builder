# molbuilder

**Gas-phase transition metal complex builder → POSCAR (VASP)**

Build mononuclear and polynuclear transition metal complexes with automatic bond lengths,
full 3D ligand geometry (H atoms included), and VASP-ready POSCAR output.

Isomers are **always generated automatically** — when a ligand set has multiple
symmetry-distinct arrangements, `build()` returns a list of Molecules (one per isomer)
and the CLI writes one POSCAR per isomer. No extra flags needed.

---

## Installation

```bash
pip install rdkit numpy scipy
# then add the molbuilder/ directory to your project
```

---

## Python API

```python
from molbuilder.api import build, dimer, trimer, poscar, xyz, info, load_ligand_from_poscar
```

### `build()` — automatic isomers

`build()` returns a **single `Molecule`** when there is only one isomer,
or a **list of `Molecule` objects** when multiple exist.
Each molecule has a `.label` attribute (`"fac"`, `"mer"`, `"cis"`, `"trans"`, …).

```python
# One isomer → Molecule
mol = build("Ni", ox=2, ligands=["H2O"]*6)
poscar(mol, "Ni_H2O6.POSCAR")

# Two isomers → list[Molecule]
mols = build("Ni", ox=2, ligands=["Cl","Cl","H2O","H2O","H2O","H2O"])
for mol in mols:
    poscar(mol, f"NiCl2_H2O4_{mol.label}.POSCAR")
# → NiCl2_H2O4_cis.POSCAR
# → NiCl2_H2O4_trans.POSCAR
```

### Denticity modes — colon notation

Append `:mono`, `:bi`, or `:bridge` to a ligand name to specify how it binds:

| Name | Meaning |
|------|---------|
| `HCOO` | monodentate formate — one O donor (default) |
| `HCOO:bi` | bidentate chelating formate — both O donors, bite angle ~55° |
| `HCOO:bridge` | bridging formate (same as `mu-HCOO`) |
| `bpy` | bidentate bipyridine — both N donors (default) |
| `bpy:mono` | monodentate bipyridine — one N donor |

---

## Nickel examples — formate, water, hydroxide

### Monomers

```python
from molbuilder.api import build, poscar

# [Ni(H2O)6]²⁺  —  octahedral hexaaqua, one isomer
mol = build("Ni", ox=2, ligands=["H2O"]*6)
poscar(mol, "Ni_H2O6.POSCAR")

# [Ni(OH)₂(H2O)₄]  —  two terminal hydroxides, cis and trans
mols = build("Ni", ox=2, ligands=["OH","OH","H2O","H2O","H2O","H2O"])
for mol in mols:
    poscar(mol, f"Ni_OH2_H2O4_{mol.label}.POSCAR")
# → Ni_OH2_H2O4_cis.POSCAR
# → Ni_OH2_H2O4_trans.POSCAR

# [Ni(HCOO)₂(H2O)₄]  —  monodentate formate, cis and trans
mols = build("Ni", ox=2, ligands=["HCOO","HCOO","H2O","H2O","H2O","H2O"])
for mol in mols:
    poscar(mol, f"Ni_HCOO2_H2O4_{mol.label}.POSCAR")
# → Ni_HCOO2_H2O4_cis.POSCAR
# → Ni_HCOO2_H2O4_trans.POSCAR

# [Ni(HCOO-κ²)₂(H2O)₂]  —  bidentate chelating formate, cis and trans
mols = build("Ni", ox=2, ligands=["HCOO:bi","HCOO:bi","H2O","H2O"], geometry="sqp")
for mol in mols:
    poscar(mol, f"Ni_HCOObi2_H2O2_{mol.label}.POSCAR")
# → Ni_HCOObi2_H2O2_cis.POSCAR
# → Ni_HCOObi2_H2O2_trans.POSCAR

# [Ni(HCOO-κ²)(HCOO)(H2O)₃]  —  mixed mono+bidentate formate
mols = build("Ni", ox=2, ligands=["HCOO:bi","HCOO","H2O","H2O","H2O"])
for mol in (mols if isinstance(mols, list) else [mols]):
    poscar(mol, f"Ni_HCOObi_HCOO_H2O3_{mol.label}.POSCAR")

# [Ni(HCOO)₂(OH)₂(H2O)₂]  —  mixed formate, hydroxide, water
mols = build("Ni", ox=2, ligands=["HCOO","HCOO","OH","OH","H2O","H2O"])
for mol in mols:
    poscar(mol, f"Ni_HCOO2_OH2_H2O2_{mol.label}.POSCAR")
```

### Dimers

```python
from molbuilder.api import dimer, poscar

# [Ni₂(μ-OH)₂(H2O)₈]²⁺  —  di-μ-hydroxo, four waters per Ni
mol = dimer("Ni", ox=2, terminal=["H2O","H2O","H2O","H2O"], bridge="mu-OH", n=2)
poscar(mol, "Ni2_muOH2_H2O8.POSCAR")

# [Ni₂(μ-HCOO)₂(H2O)₆]²⁺  —  di-μ-formato, monodentate bridging formate
mol = dimer("Ni", ox=2, terminal=["H2O","H2O","H2O"], bridge="mu-HCOO", n=2)
poscar(mol, "Ni2_muHCOO2_H2O6.POSCAR")

# [Ni₂(μ-OH)(H2O)₈]³⁺  —  single bridging hydroxide
mol = dimer("Ni", ox=2, terminal=["H2O","H2O","H2O","H2O"], bridge="mu-OH", n=1)
poscar(mol, "Ni2_muOH_H2O8.POSCAR")

# [Ni₂(μ-OH)₂(HCOO)₂(H2O)₄]  —  mixed terminal formate + bridging hydroxide
mol = dimer("Ni", ox=2, terminal=["HCOO","H2O","H2O"], bridge="mu-OH", n=2)
poscar(mol, "Ni2_muOH2_HCOO2_H2O4.POSCAR")
```

---

## CLI

```bash
PYTHONPATH=. python molbuilder/cli.py --metal SYMBOL --ox N --ligands L1 L2 ... --out FILE.POSCAR
```

Isomers are automatic. When multiple isomers exist the `--out` base name is used
with the label appended: `Ni_HCOO2.POSCAR` → `Ni_HCOO2_cis.POSCAR` + `Ni_HCOO2_trans.POSCAR`.

### Ni + formate/water/hydroxide examples

```bash
# [Ni(H2O)6]²⁺
PYTHONPATH=. python molbuilder/cli.py \
    --metal Ni --ox 2 \
    --ligands H2O H2O H2O H2O H2O H2O \
    --out Ni_H2O6.POSCAR

# [Ni(HCOO)₂(H2O)₄]  →  Ni_HCOO2_H2O4_cis.POSCAR + Ni_HCOO2_H2O4_trans.POSCAR
PYTHONPATH=. python molbuilder/cli.py \
    --metal Ni --ox 2 \
    --ligands HCOO HCOO H2O H2O H2O H2O \
    --out Ni_HCOO2_H2O4.POSCAR

# [Ni(HCOO-κ²)₂(H2O)₂] bidentate formate, sqp  →  _cis + _trans
PYTHONPATH=. python molbuilder/cli.py \
    --metal Ni --ox 2 \
    --ligands HCOO:bi HCOO:bi H2O H2O \
    --geometry sqp \
    --out Ni_HCOObi2_H2O2.POSCAR

# [Ni(OH)₂(H2O)₄]  →  _cis + _trans
PYTHONPATH=. python molbuilder/cli.py \
    --metal Ni --ox 2 \
    --ligands OH OH H2O H2O H2O H2O \
    --out Ni_OH2_H2O4.POSCAR

# [Ni(HCOO)₂(OH)₂(H2O)₂]  →  multiple isomers
PYTHONPATH=. python molbuilder/cli.py \
    --metal Ni --ox 2 \
    --ligands HCOO HCOO OH OH H2O H2O \
    --out Ni_HCOO2_OH2_H2O2.POSCAR

# [Ni₂(μ-OH)₂(H2O)₈]²⁺
PYTHONPATH=. python molbuilder/cli.py \
    --dimer --metal Ni --ox 2 \
    --ligands H2O H2O H2O H2O \
    --bridge mu-OH --n-bridges 2 \
    --out Ni2_muOH2_H2O8.POSCAR

# [Ni₂(μ-HCOO)₂(H2O)₆]²⁺  monodentate bridging formate
PYTHONPATH=. python molbuilder/cli.py \
    --dimer --metal Ni --ox 2 \
    --ligands H2O H2O H2O \
    --bridge mu-HCOO --n-bridges 2 \
    --out Ni2_muHCOO2_H2O6.POSCAR

# [Ni₂(μ-OH)₂(HCOO)₂(H2O)₄]  terminal formate + bridging hydroxide
PYTHONPATH=. python molbuilder/cli.py \
    --dimer --metal Ni --ox 2 \
    --ligands HCOO H2O H2O \
    --bridge mu-OH --n-bridges 2 \
    --out Ni2_muOH2_HCOO2_H2O4.POSCAR
```

### Custom POSCAR ligand

```bash
PYTHONPATH=. python molbuilder/cli.py \
    --metal Ni --ox 2 \
    --ligands H2O H2O H2O H2O H2O \
    --custom-ligand my_ligand.POSCAR \
    --donor-atoms 0 \
    --ligand-charge -1 \
    --out Ni_custom_H2O5.POSCAR
```

### Other useful flags

```bash
# Print POSCAR to stdout as well as writing file
PYTHONPATH=. python molbuilder/cli.py --metal Ni --ox 2 \
    --ligands H2O H2O H2O H2O H2O H2O --out Ni_H2O6.POSCAR --print

# Also write XYZ alongside POSCAR
PYTHONPATH=. python molbuilder/cli.py --metal Ni --ox 2 \
    --ligands H2O H2O H2O H2O H2O H2O --out Ni_H2O6.POSCAR --xyz

# List all available ligands
PYTHONPATH=. python molbuilder/cli.py --list-ligands

# List all supported geometries
PYTHONPATH=. python molbuilder/cli.py --list-geometries
```

---

## Isomer reference

| Formula | Geometry | Isomers |
|---------|----------|---------|
| MA₆ | oct | 1 |
| MA₅B | oct | 1 |
| MA₄B₂ | oct | 2 (cis, trans) |
| MA₃B₃ | oct | 2 (fac, mer) |
| MA₄BC | oct | 2 |
| MA₂B₂C₂ | oct | 5 |
| MA₄ | sqp | 1 |
| MA₂B₂ | sqp | 2 (cis, trans) |

---

## Supported geometries

| Key | Name | CN |
|-----|------|----|
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

## Ligand library (selection)

| Name | Donor | Charge | Notes |
|------|-------|--------|-------|
| `H2O` / `aqua` | O | 0 | includes 2 H |
| `OH` | O | −1 | terminal hydroxide |
| `mu-OH` | O | −1 | bridging hydroxide |
| `NH3` / `ammine` | N | 0 | includes 3 H |
| `Cl`, `Br`, `I`, `F` | halide | −1 | |
| `CO` | C | 0 | |
| `CN` | C | −1 | |
| `HCOO` / `formate` | O | −1 | monodentate (default) |
| `HCOO:bi` | O,O | −1 | bidentate chelating, bite ~55° |
| `mu-HCOO` / `HCOO:bridge` | O | −1 | bridging |
| `HCOOH` | O | 0 | formic acid, monodentate |
| `HCOOH:bi` | O,O | 0 | formic acid bidentate |
| `OAc` / `acetate` | O | −1 | monodentate acetate |
| `OAc:bi` | O,O | −1 | bidentate acetate |
| `mu-OAc` | O | −1 | bridging acetate |
| `acac` | O,O | −1 | acetylacetonate |
| `ox` / `oxalate` | O,O | −2 | |
| `en` | N,N | 0 | ethylenediamine |
| `bpy` / `bipy` | N,N | 0 | bidentate (default) |
| `bpy:mono` | N | 0 | monodentate |
| `phen` | N,N | 0 | phenanthroline |
| `tpy` / `terpy` | N,N,N | 0 | |
| `EDTA` | N,N,O,O,O,O | −4 | |
| `Cp` | C×5 | −1 | cyclopentadienyl η⁵ |
| `mu-Cl`, `mu-O` | bridging | varies | |
| `PPh3`, `PMe3` | P | 0 | |
| `py` / `pyridine` | N | 0 | |
| `MeCN` | N | 0 | |
| `SCN` | S | −1 | |
| `dmso` / `DMSO` | S | 0 | |

---

## Bond length sources

CSD-averaged values (Orpen et al. 1989) keyed by `(metal, oxidation_state, donor_atom, geometry)`.
Fallback: same metal/donor averaged over geometries → averaged over oxidation states → Alvarez (2008) covalent radii sum.
