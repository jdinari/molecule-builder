# molbuilder

**Build transition metal complexes and export them as VASP POSCAR files.**

molbuilder generates 3D structures of mononuclear, dinuclear, and trinuclear transition metal complexes from a simple list of ligands. It handles bond lengths, ligand geometry (including H atoms), coordination geometry, and automatically produces all symmetry-distinct isomers.

---

## Installation

**From GitHub (recommended):**

```bash
pip install git+https://github.com/jdinari/molecule-builder.git
```

**To update to the latest version:**

```bash
pip install --upgrade git+https://github.com/jdinari/molecule-builder.git
```

**From source (if you have the files locally):**

```bash
git clone https://github.com/jdinari/molecule-builder.git
cd molecule-builder
pip install .
```

**Editable install for development** — changes to the source take effect immediately without reinstalling:

```bash
git clone https://github.com/jdinari/molecule-builder.git
cd molecule-builder
pip install -e .
```

Dependencies (`numpy`, `scipy`, `rdkit`) are installed automatically.

---

## Quick start

```python
from molbuilder.api import build, poscar

# Octahedral Ni(II) hexaaqua complex
mol = build("Ni", ox=2, ligands=["H2O"]*6)
poscar(mol, "Ni_H2O6.POSCAR")
```

Or from the command line:

```bash
molbuilder --metal Ni --ox 2 --ligands H2O H2O H2O H2O H2O H2O --out Ni_H2O6.POSCAR
```

---

## Core concepts

### Isomers are automatic

Whenever a ligand set has more than one symmetry-distinct arrangement, `build()` returns a **list** of molecules (one per isomer). If there is only one arrangement it returns a single molecule. Each molecule carries a `.label` attribute (`"fac"`, `"mer"`, `"cis"`, `"trans"`, …).

```python
# Single isomer → Molecule
mol = build("Ni", ox=2, ligands=["H2O"]*6)

# Two isomers → list[Molecule]
mols = build("Fe", ox=3, ligands=["Cl","Cl","Cl","H2O","H2O","H2O"])
for mol in mols:
    print(mol.label)          # "fac" or "mer"
    poscar(mol, f"FeCl3_H2O3_{mol.label}.POSCAR")
```

The CLI follows the same logic — when multiple isomers exist, it writes one file per isomer and appends the label to the filename:

```
FeCl3_H2O3.POSCAR  →  FeCl3_H2O3_fac.POSCAR
                       FeCl3_H2O3_mer.POSCAR
```

### Denticity modes

Append `:bi` or `:bridge` to a ligand name to change how it binds:

| Name | Binding mode |
|------|-------------|
| `HCOO` | Monodentate formate — one O donor |
| `HCOO:bi` | Bidentate chelating formate — both O donors, ~55° bite angle |
| `HCOO:bridge` or `mu-HCOO` | Bridging formate |
| `bpy` | Bidentate bipyridine (default) |
| `bpy:mono` | Monodentate bipyridine |

---

## Python API

### Monomers

```python
from molbuilder.api import build, poscar, xyz, info

# [Ni(H2O)6]²⁺  —  one isomer
mol = build("Ni", ox=2, ligands=["H2O"]*6)
poscar(mol, "Ni_H2O6.POSCAR")

# [FeCl3(H2O)3]  —  two isomers (fac and mer)
mols = build("Fe", ox=3, ligands=["Cl","Cl","Cl","H2O","H2O","H2O"])
for mol in mols:
    poscar(mol, f"FeCl3_H2O3_{mol.label}.POSCAR")

# [Ni(HCOO)2(H2O)4]  —  monodentate formate, cis and trans
mols = build("Ni", ox=2, ligands=["HCOO","HCOO","H2O","H2O","H2O","H2O"])
for mol in mols:
    poscar(mol, f"Ni_HCOO2_H2O4_{mol.label}.POSCAR")

# [Ni(HCOO:bi)2(H2O)2]  —  bidentate chelating formate
mol = build("Ni", ox=2, ligands=["HCOO:bi","HCOO:bi","H2O","H2O"], geometry="sqp")
poscar(mol, "Ni_HCOObi2_H2O2.POSCAR")

# Specify geometry explicitly (default is inferred from coordination number)
mol = build("Pd", ox=2, ligands=["Cl","Cl","NH3","NH3"], geometry="sqp")
for mol in (mol if isinstance(mol, list) else [mol]):
    poscar(mol, f"PdCl2_NH3_2_{mol.label}.POSCAR")  # cis-platin and trans-platin

# Print a structure summary
info(mol)

# Also write XYZ
xyz(mol, "Ni_H2O6.xyz")
```

### Dimers

Bridging ligands hold two metal centres together. There is no metal–metal bond unless you explicitly request one with `mm_bond=True`.

```python
from molbuilder.api import dimer, poscar

# [Ni2(μ-OH)2(H2O)8]  —  di-μ-hydroxo dimer
mol = dimer("Ni", ox=2, terminal=["H2O","H2O","H2O","H2O"], bridge="mu-OH", n=2)
poscar(mol, "Ni2_muOH2_H2O8.POSCAR")

# [Ni2(μ-HCOO)2(H2O)6]  —  bridging formate
mol = dimer("Ni", ox=2, terminal=["H2O","H2O","H2O"], bridge="mu-HCOO", n=2)
poscar(mol, "Ni2_muHCOO2_H2O6.POSCAR")

# Mixed terminal ligands
mol = dimer("Ni", ox=2, terminal=["HCOO","H2O","H2O"], bridge="mu-OH", n=2)
poscar(mol, "Ni2_muOH2_HCOO2_H2O4.POSCAR")

# Metal–metal bonded dimer (e.g. Re quadruple bond)
mol = dimer("Re", ox=3, terminal=["Cl","Cl","Cl"], bridge="mu-Cl", n=2,
            mm_bond=True, mm_distance=2.22)
poscar(mol, "Re2_quadruple_bond.POSCAR")
```

### Trimers

```python
from molbuilder.api import trimer, poscar

# Linear Fe3 with bridging hydroxides
mol = trimer("Fe", ox=3, terminal=["H2O","H2O"], bridge="mu-OH", arrangement="linear")
poscar(mol, "Fe3_linear_muOH.POSCAR")

# Triangular Ru3 carbonyl cluster
mol = trimer("Ru", ox=0, terminal=["CO","CO","CO","CO"], bridge="mu-CO",
             arrangement="triangular")
poscar(mol, "Ru3_triangular_CO12.POSCAR")
```

### Custom POSCAR ligands

Load an unusual ligand from an existing POSCAR file:

```python
from molbuilder.api import build, load_ligand_from_poscar, poscar

lig = load_ligand_from_poscar("my_ligand.POSCAR", donor_atom_indices=[0], charge=-1)
mol = build("Ni", ox=2, ligands=["H2O","H2O","H2O","H2O","H2O", lig])
poscar(mol, "Ni_custom.POSCAR")
```

---

## Energy & thermochemistry

molbuilder can relax structures and compute energetics using two backends:

| Backend | Method | Best for |
|---------|--------|----------|
| `xtb` | GFN2-xTB via tblite | Ni coordination chemistry; explicit charge/spin |
| `mace` | MACE-MH-1 MLIP | Larger clusters; faster on GPU |

### Setting your MACE model path

By default the MACE backend downloads `mh-1` (~500 MB) on first use. **On a cluster without internet access, or when using a custom fine-tuned model, you need to point to your local `.model` file.** There are three places to do this depending on how you're running the code:

#### 1. `generate_ni_complexes.py` (batch generation script)

Open the file and set `MACE_MODEL` near the top:

```python
# ── Energy computation ─────────────────────────────────────────────────────
COMPUTE_ENERGY  = True
ENERGY_BACKEND  = "mace"
MACE_MODEL      = "/path/to/your/mace-mh-1.model"   # ← set this
MACE_DEVICE     = "cuda"                              # "cpu" or "cuda"
```

#### 2. `compute_energetics.py` (CLI script for an existing POSCAR directory)

Pass the path via `--model`:

```bash
python compute_energetics.py \
    --poscar-dir poscar/ \
    --backend mace \
    --model /path/to/your/mace-mh-1.model \
    --mace-device cuda
```

#### 3. Python API directly

Pass the path as the `model` argument to any relaxation function:

```python
from molbuilder.relaxation import relax, thermochemistry

result = relax(mol, backend="mace",
               model="/path/to/your/mace-mh-1.model",
               device="cuda")

thermo = thermochemistry(mol, backend="mace",
                         model="/path/to/your/mace-mh-1.model",
                         device="cuda")
```

> **Tip:** The `model` argument accepts anything that `mace_mp(model=...)` accepts — a local file path, a URL, or a named preset like `"mh-1"`. For cluster use, a local path is always the safest option.

### Running energetics

```python
from molbuilder.api import build
from molbuilder.relaxation import relax, compute_energy, thermochemistry

mol = build("Ni", ox=2, ligands=["H2O"]*6)

# Geometry relaxation only
result = relax(mol, backend="xtb")
print(result.energy_eV, result.converged)

# Single-point (no geometry change)
result = compute_energy(mol, backend="mace", model="/path/to/mace.model", device="cuda")

# Full thermochemistry (relax + frequencies + ΔG)
thermo = thermochemistry(mol, backend="xtb", T=298.15, P=101325.0)
print(thermo.gibbs_eV)

# Recompute ΔG at a different temperature without re-running:
print(thermo.gibbs_at(T=350))
```

### ΔE and ΔG for reactions

```python
results = {name: thermochemistry(mol, backend="xtb") for name, mol in species.items()}
dE = results["product"].energy_eV - results["reactant"].energy_eV
dG = results["product"].gibbs_eV  - results["reactant"].gibbs_eV
```

---

## CLI reference

After `pip install .` the `molbuilder` command is available everywhere.

```
molbuilder --metal SYMBOL --ox N --ligands L1 L2 ... --out FILE.POSCAR [options]
```

| Flag | Description |
|------|-------------|
| `--metal` | Element symbol, e.g. `Ni`, `Fe`, `Pd` |
| `--ox` | Oxidation state, e.g. `2`, `3` |
| `--ligands` | Space-separated ligand names. Colon modes supported: `HCOO:bi`, `bpy:mono` |
| `--geometry` | Coordination geometry key (see table below). Auto-inferred if omitted |
| `--out` | Output POSCAR file. Multiple isomers → one file per isomer |
| `--dimer` | Build a dinuclear complex |
| `--trimer` | Build a trinuclear complex |
| `--bridge` | Bridging ligand for dimer/trimer, e.g. `mu-OH`, `mu-HCOO` |
| `--n-bridges` | Number of bridging ligand units (default: 2) |
| `--arrangement` | Trimer arrangement: `triangular` or `linear` (default: triangular) |
| `--mm-bond` | Include a metal–metal bond |
| `--mm-distance` | Override M–M distance in Å |
| `--xyz` | Also write an XYZ file alongside the POSCAR |
| `--print` | Print POSCAR to stdout |
| `--custom-ligand` | Path to a POSCAR file to use as a custom ligand |
| `--donor-atoms` | Comma-separated donor atom indices in the custom ligand |
| `--ligand-charge` | Formal charge of the custom ligand |
| `--list-ligands` | Print all available ligand names |
| `--list-geometries` | Print all supported geometry keys |

### Examples

```bash
# Octahedral Fe(III) — generates fac and mer isomers automatically
molbuilder --metal Fe --ox 3 \
    --ligands Cl Cl Cl H2O H2O H2O \
    --out FeCl3_H2O3.POSCAR
# → FeCl3_H2O3_fac.POSCAR  and  FeCl3_H2O3_mer.POSCAR

# Square planar Pd(II) — cis-platin and trans-platin
molbuilder --metal Pd --ox 2 \
    --ligands Cl Cl NH3 NH3 --geometry sqp \
    --out PdCl2_NH3_2.POSCAR

# Bidentate chelating formate
molbuilder --metal Ni --ox 2 \
    --ligands HCOO:bi HCOO:bi H2O H2O --geometry sqp \
    --out Ni_HCOObi2_H2O2.POSCAR

# Di-μ-hydroxo dimer
molbuilder --dimer --metal Ni --ox 2 \
    --ligands H2O H2O H2O H2O \
    --bridge mu-OH --n-bridges 2 \
    --out Ni2_muOH2_H2O8.POSCAR

# Linear Ni3 trimer with bridging formate
molbuilder --trimer --metal Ni --ox 2 \
    --ligands H2O H2O H2O \
    --bridge mu-HCOO --arrangement linear \
    --out Ni3_linear_muHCOO.POSCAR

# Print to stdout and save, also write XYZ
molbuilder --metal Ni --ox 2 --ligands H2O H2O H2O H2O H2O H2O \
    --out Ni_H2O6.POSCAR --print --xyz

# List everything available
molbuilder --list-ligands
molbuilder --list-geometries
```

---

## Batch generation

`generate_ni_complexes.py` generates all neutral Ni(II) and Ni(III) structures across coordination numbers 3–7, using formate (mono and bidentate), formic acid, water, and hydroxide as ligands, for monomers, dimers, and trimers:

```bash
python generate_ni_complexes.py
```

Output goes to `poscar/` organised by structure type, oxidation state, and CN. A summary CSV (`ni_complexes_summary.csv`) indexes every file with its formula, charge, atom count, geometry, and ligand combination.

---

## Geometry reference

| Key | Name | CN |
|-----|------|-----|
| `lin` | Linear | 2 |
| `bent` | Bent | 2 |
| `tp` | Trigonal planar | 3 |
| `tshaped` | T-shaped | 3 |
| `tet` | Tetrahedral | 4 |
| `sqp` | Square planar | 4 |
| `seesaw` | See-saw | 4 |
| `tbp` | Trigonal bipyramidal | 5 |
| `sqpy` | Square pyramidal | 5 |
| `oct` | Octahedral | 6 |
| `tpr` | Trigonal prismatic | 6 |
| `pbp` | Pentagonal bipyramidal | 7 |
| `sapr` | Square antiprismatic | 8 |

Aliases: `td` = `tet`, `sp` = `sqp`, `oh` = `oct`, `spy` = `sqpy`

---

## Ligand reference

### Common ligands

| Name | Formula | Donor | Charge | Denticity |
|------|---------|-------|--------|-----------|
| `H2O` / `aqua` | H₂O | O | 0 | 1 |
| `OH` | OH⁻ | O | −1 | 1 — terminal hydroxide |
| `mu-OH` | OH⁻ | O | −1 | 1 — bridging hydroxide |
| `NH3` / `ammine` | NH₃ | N | 0 | 1 |
| `CO` | CO | C | 0 | 1 |
| `CN` | CN⁻ | C | −1 | 1 |
| `Cl`, `Br`, `I`, `F` | X⁻ | X | −1 | 1 |
| `NO2` | NO₂⁻ | N | −1 | 1 |
| `SCN` | SCN⁻ | S | −1 | 1 |
| `py` / `pyridine` | C₅H₅N | N | 0 | 1 |
| `MeCN` | CH₃CN | N | 0 | 1 |
| `PPh3` | PPh₃ | P | 0 | 1 |
| `dmso` / `DMSO` | DMSO | S | 0 | 1 |

### Formate / formic acid

| Name | Binding mode | Charge |
|------|-------------|--------|
| `HCOO` / `formate` | Monodentate through one O | −1 |
| `HCOO:bi` | Bidentate chelating, O,O, ~55° bite | −1 |
| `mu-HCOO` / `HCOO:bridge` | Bridging, one O per metal | −1 |
| `HCOOH` | Monodentate formic acid, through C=O oxygen | 0 |
| `HCOOH:bi` | Bidentate formic acid | 0 |

### Bidentate and multidentate

| Name | Donors | Charge | Denticity |
|------|--------|--------|-----------|
| `en` | N, N | 0 | 2 |
| `bpy` / `bipy` | N, N | 0 | 2 |
| `phen` | N, N | 0 | 2 |
| `acac` | O, O | −1 | 2 |
| `ox` / `oxalate` | O, O | −2 | 2 |
| `OAc` / `acetate` | O | −1 | 1 |
| `OAc:bi` | O, O | −1 | 2 |
| `glycinate` | N, O | −1 | 2 |
| `tpy` / `terpy` | N, N, N | 0 | 3 |
| `EDTA` | N, N, O, O, O, O | −4 | 6 |
| `Cp` | C×5 (η⁵) | −1 | 5 |

---

## Isomer reference

| Composition | Geometry | Isomers |
|-------------|----------|---------|
| MA₆ | oct | 1 |
| MA₅B | oct | 1 |
| MA₄B₂ | oct | 2 — cis, trans |
| MA₃B₃ | oct | 2 — fac, mer |
| MA₄BC | oct | 2 |
| MA₂B₂C₂ | oct | 5 |
| MA₄ | sqp | 1 |
| MA₃B | sqp | 1 |
| MA₂B₂ | sqp | 2 — cis, trans |

---

## How it works

**Bond lengths** are drawn from a database of CSD-averaged M–L distances (Orpen et al. 1989), keyed by metal, oxidation state, donor atom, and geometry. The fallback hierarchy is: same metal/ox/donor averaged over geometries → same metal/donor averaged over oxidation states → Alvarez (2008) covalent radii sum.

**Ligand geometry** is built using internal coordinates (real bond lengths and angles). For each ligand, the M–donor–C angle places the first non-donor atom correctly (e.g. 120° for formate's sp² oxygen), and the torsion around the M–donor axis is optimised to point the ligand bulk into the largest gap between adjacent coordination sites.

**Isomers** are enumerated by generating all permutations of ligands across coordination sites and de-duplicating under the point-group symmetry of the geometry (Oₕ for octahedral, D₄ₕ for square planar, Tₐ for tetrahedral, etc.).

**POSCAR format** — atoms are centred in a cubic vacuum box (15 Å padding per side), species sorted by atomic number, Cartesian coordinates in Å. Charge and spin multiplicity are written in the comment line.
