# molbuilder

**Generate 3D VASP POSCAR structures of transition-metal coordination complexes.**

molbuilder builds mononuclear, dinuclear, and trinuclear complexes from a metal symbol, oxidation state, and a list of ligands. It handles bond lengths, ligand geometry (including all H atoms), coordination geometry, automatic isomer enumeration, and optional xTB or MACE-MH-1 geometry relaxation.

---

## Installation

```bash
pip install git+https://github.com/jdinari/molecule-builder.git
```

**Optional backends for geometry relaxation and thermochemistry:**

```bash
pip install tblite ase        # xTB — recommended for Ni coordination chemistry
pip install mace-torch ase    # MACE-MH-1 — faster on GPU
pip install openpyxl          # Excel output for energetics
```

---

## Quick start

```python
from molbuilder.api import build, poscar, info

mol = build("Ni", ox=2, ligands=["H2O"] * 6)
info(mol)
poscar(mol, "Ni_H2O6.POSCAR")
```

```bash
molbuilder --metal Ni --ox 2 --ligands H2O H2O H2O H2O H2O H2O --out Ni_H2O6.POSCAR
```

---

## Core concepts

### Isomers are automatic

`build()` returns a **list** when more than one symmetry-distinct arrangement exists, or a single `Molecule` otherwise:

```python
from molbuilder.api import build, poscar

# Single isomer → Molecule
mol  = build("Ni", ox=2, ligands=["H2O"] * 6)

# Two isomers → list[Molecule]
mols = build("Fe", ox=3, ligands=["Cl"]*3 + ["H2O"]*3)
for mol in mols:
    print(mol.label)              # "fac" or "mer"
    poscar(mol, f"Fe_{mol.label}.POSCAR")
```

### Ligand modes

| Name | Binding |
|------|---------|
| `HCOO` | Monodentate formate — one O donor |
| `HCOO:bi` | Bidentate chelating formate — both O, ~55° bite angle |
| `mu-HCOO` | Bridging formate — one O per metal centre |
| `H2O` | Water — O donor |
| `NH3` | Ammonia — N donor (H atoms correct, pointing away from metal) |
| `OH` | Hydroxide — O donor, charge −1 |
| `mu-OH` | Bridging hydroxide |

---

## API reference

### Monomers

```python
from molbuilder.api import build, poscar, info

# Octahedral Ni(II) hexaaqua
mol = build("Ni", ox=2, ligands=["H2O"] * 6)

# Square planar Pd(II) — generates cis (cisplatin) and trans isomers
mols = build("Pd", ox=2, ligands=["Cl", "Cl", "NH3", "NH3"], geometry="sqp")

# Bidentate chelating formate
mol = build("Ni", ox=2, ligands=["HCOO:bi", "HCOO:bi", "H2O", "H2O"], geometry="sqp")

# All geometries and coordination numbers
mol = build("Fe", ox=2, ligands=["CO"] * 6, geometry="oct")
mol = build("Cu", ox=2, ligands=["Cl"] * 4, geometry="sqp")
```

### Dimers

```python
from molbuilder.api import dimer, poscar

# Di-μ-hydroxo dimer — CN=4 per Ni
mol = dimer("Ni", ox=2, terminal=["H2O", "H2O"], bridge="mu-OH", n=2)

# Paddle-wheel — 4 bridging formates, no terminals, common MOF building unit
mol = dimer("Ni", ox=2, terminal=[], bridge="mu-HCOO", n=4)

# Heteroleptic — different terminal ligands on each metal centre
mol = dimer("Ni", ox=2,
            terminal_m1=["H2O"], terminal_m2=[],
            bridge="mu-HCOO", n=4)
```

### Trimers

```python
from molbuilder.api import trimer, poscar

# Linear Fe(III) trimer
mol = trimer("Fe", ox=3, terminal=["H2O", "H2O"], bridge="mu-OH",
             arrangement="linear")

# Triangular Ni3 — equilateral triangle, double-bridge per edge
mol = trimer("Ni", ox=2, terminal=[], bridge="mu-HCOO",
             arrangement="triangular", n_bridges_per_pair=2)

# Heteroleptic — H2O on one metal only
mol = trimer("Ni", ox=2, bridge="mu-HCOO",
             arrangement="triangular", n_bridges_per_pair=2,
             terminals_per_metal=[["H2O"], [], []])
```

### Geometry relaxation

Requires `pip install tblite ase`.

```python
from molbuilder.api import build
from molbuilder.relaxation import relax, thermochemistry, check_bonds_intact

mol = build("Ni", ox=2, ligands=["H2O"] * 6)

# Geometry relaxation
res = relax(mol, backend="xtb", fmax=0.05, steps=300)
print(f"E = {res.energy_eV:.4f} eV  converged={res.converged}")

# Bond integrity check — detects ligand dissociation
bc = check_bonds_intact(mol, res.mol)
print(f"Bond status: {'OK' if bc['intact'] else 'BROKEN'}")
print(f"Max elongation: {bc['max_elongation']:.2f}×")

# Full thermochemistry — relax + frequencies + ΔG
thermo = thermochemistry(mol, backend="xtb", T=298.15, P=101325)
print(f"G(298K) = {thermo.gibbs_eV:.4f} eV")
print(f"G(350K) = {thermo.gibbs_at(T=350):.4f} eV")  # no re-run needed
```

### ΔE and ΔG for reactions

Requires `pip install tblite ase`.

```python
import numpy as np
from molbuilder.api import build
from molbuilder.core.molecule import Molecule, Atom
from molbuilder.relaxation import thermochemistry

# Build the Ni complexes
Ni_H2O6      = build("Ni", ox=2, ligands=["H2O"] * 6)
Ni_HCOO_H2O5 = build("Ni", ox=2, ligands=["HCOO"] + ["H2O"] * 5)
if isinstance(Ni_HCOO_H2O5, list):
    Ni_HCOO_H2O5 = Ni_HCOO_H2O5[0]

# Build free-molecule references from known geometry parameters
def make_h2o():
    ah = np.radians(104.5 / 2); oh = 0.958
    return Molecule(
        atoms=[Atom("O", np.zeros(3)),
               Atom("H", np.array([ oh*np.sin(ah), oh*np.cos(ah), 0.])),
               Atom("H", np.array([-oh*np.sin(ah), oh*np.cos(ah), 0.]))],
        formula="H2O", charge=0, spin_multiplicity=1, metal_symbol="", metal_ox=0,
    )

def make_formate():
    ah = np.radians(126.0 / 2); co = 1.25; ch = 1.09
    return Molecule(
        atoms=[Atom("C", np.zeros(3)),
               Atom("O", np.array([ co*np.sin(ah), co*np.cos(ah), 0.])),
               Atom("O", np.array([-co*np.sin(ah), co*np.cos(ah), 0.])),
               Atom("H", np.array([0., -ch, 0.]))],
        formula="HCOO", charge=-1, spin_multiplicity=1, metal_symbol="", metal_ox=0,
    )

# Compute thermochemistry for all four species
r_hexaaqua = thermochemistry(Ni_H2O6,       backend="xtb", T=298.15)
r_formate   = thermochemistry(make_formate(), backend="xtb", T=298.15)
r_product   = thermochemistry(Ni_HCOO_H2O5, backend="xtb", T=298.15)
r_water     = thermochemistry(make_h2o(),    backend="xtb", T=298.15)

dE = r_product.energy_eV + r_water.energy_eV \
   - r_hexaaqua.energy_eV - r_formate.energy_eV

dG = r_product.gibbs_eV  + r_water.gibbs_eV  \
   - r_hexaaqua.gibbs_eV - r_formate.gibbs_eV

# Re-evaluate at any T without re-running:
dG_350 = (r_product.gibbs_at(350) + r_water.gibbs_at(350)
        - r_hexaaqua.gibbs_at(350) - r_formate.gibbs_at(350))

print(f"[Ni(H2O)6] + HCOO⁻  →  [Ni(HCOO)(H2O)5] + H2O")
print(f"  ΔE = {dE:+.4f} eV")
print(f"  ΔG(298K) = {dG:+.4f} eV")
print(f"  ΔG(350K) = {dG_350:+.4f} eV")
```

### Batch generation

The `generate_ni_complexes.py` script generates all neutral Ni(II)/Ni(III) complexes across CN 3–7 using formate, formic acid, water, hydroxide, and bidentate formate as ligands:

```bash
pip install git+https://github.com/jdinari/molecule-builder.git
python generate_ni_complexes.py
```

Open `generate_ni_complexes.py` and flip these flags at the top of the file to enable each stage:

```python
COMPUTE_ENERGY    = True   # relax every structure and compute ΔG(T, P)
COMPUTE_REACTIONS = True   # build isodesmic reaction network and screen by ΔG
```

With `COMPUTE_ENERGY = True`, every structure is relaxed with xTB or MACE and
an Excel workbook with colour-coded bond status is written to `ni_energetics.xlsx`.

---

## Bond status

After relaxation, every M-L bond is compared to its initial length:

| Status | M-L elongation | Meaning |
|--------|---------------|---------|
| `OK` | ≤ 1.20× | Clean structure |
| `STRETCHED` | 1.20–1.35× | Possible strain — review |
| `BROKEN` | > 1.35× | Ligand dissociated during relaxation |

Bond breaking is reported, not suppressed. If xTB says a ligand departs, the coordination is genuinely strained — this is useful information before DFT.

---

## Backend comparison

| | xTB (GFN2-xTB) | MACE-MH-1 |
|---|---|---|
| Install | `pip install tblite ase` | `pip install mace-torch ase` |
| Model size | < 1 MB | ~500 MB |
| Charge/spin | ✓ explicit | ✗ implicit only |
| Ni(III) open-shell | ✓ reliable | ✗ verify against xTB |
| Speed (CPU) | ~5 s / structure | ~8 s / structure |
| Speed (GPU) | — | ~0.5 s / structure |
| Thermochemistry | ✓ | ✓ (numerical freq) |
| Recommended for | All Ni coordination chemistry | Large batches on GPU |

**For Ni(II)/Ni(III) coordination complexes, xTB is the recommended default** — it has explicit Ni d-electron parameters and takes charge/spin directly.

MACE uses `head="omol"` automatically — the molecular head trained on OMOL/OC20/MATPES data with wB97M-V/R2SCAN references.

---

## Geometry reference

| Key | Geometry | CN |
|-----|----------|----|
| `lin` | Linear | 2 |
| `tp` | Trigonal planar | 3 |
| `tet` / `td` | Tetrahedral | 4 |
| `sqp` / `sp` | Square planar | 4 |
| `tbp` | Trigonal bipyramidal | 5 |
| `sqpy` / `spy` | Square pyramidal | 5 |
| `oct` / `oh` | Octahedral | 6 |
| `tpr` | Trigonal prismatic | 6 |
| `pbp` | Pentagonal bipyramidal | 7 |

---

## Ligand reference

| Name | Formula | Donor | Charge | Denticity |
|------|---------|-------|--------|-----------|
| `H2O` / `aqua` | H₂O | O | 0 | 1 |
| `OH` | OH⁻ | O | −1 | 1 |
| `NH3` / `ammine` | NH₃ | N | 0 | 1 |
| `CO` | CO | C | 0 | 1 |
| `Cl`, `Br`, `I`, `F` | X⁻ | X | −1 | 1 |
| `HCOO` | HCOO⁻ | O (mono) | −1 | 1 |
| `HCOO:bi` | HCOO⁻ | O,O (chelate) | −1 | 2 |
| `mu-HCOO` | HCOO⁻ | O (bridge) | −1 | 1 |
| `HCOOH` | HCOOH | O | 0 | 1 |
| `mu-OH` | OH⁻ (bridge) | O | −1 | 1 |
| `en` | ethylenediamine | N,N | 0 | 2 |
| `bpy` / `bipy` | bipyridine | N,N | 0 | 2 |
| `phen` | phenanthroline | N,N | 0 | 2 |
| `acac` | acetylacetonate | O,O | −1 | 2 |
| `ox` | oxalate | O,O | −2 | 2 |
| `tpy` | terpyridine | N,N,N | 0 | 3 |
| `EDTA` | EDTA | N,N,O,O,O,O | −4 | 6 |

---

## Tutorials

Seven step-by-step tutorials in `tutorials/`:

| # | Topic |
|---|-------|
| 01 | First complex — build, inspect, write POSCAR |
| 02 | Isomers and bidentate ligands |
| 03 | Dimers, paddle-wheels, heteroleptic dimers, triangular trimers |
| 04 | MACE relaxation on a cluster (local model file, GPU, SLURM) |
| 05 | xTB thermochemistry — ΔE and ΔG for ligand substitution |
| 06 | Batch enumeration with `enumerate_complexes()` and `run_energetics()` |
| 07 | Isodesmic reaction network, ΔG screening, broken-structure filtering |

Each tutorial can be run directly from the repository root after installing the package:

```bash
python tutorials/01_first_complex.py
python tutorials/02_isomers_and_bidentate.py
python tutorials/03_dimers_and_trimers.py
python tutorials/04_mace_relaxation.py            # requires: pip install mace-torch ase
python tutorials/05_thermochemistry.py            # requires: pip install tblite ase
python tutorials/06_enumeration_and_energetics.py # requires: pip install tblite ase openpyxl
python tutorials/07_reaction_network.py           # requires: pip install tblite ase matplotlib pandas
```

Short copy-paste examples are in `examples/`.

---

## Known limitations

- `dimer(..., mm_bond=True)` with terminal halides may produce Cl-on-Re placements. Use `mm_bond=True` with bare dimers (no terminals) for now.
- Ru₃(μ-CO) triangular trimers produce C-C clashes at ideal template geometry. Start from a crystallographic geometry and relax.
- CN=7 (pentagonal bipyramidal) structures with ≥2 H-bearing ligands are filtered out of batch enumeration automatically because equatorial ligands at 72° spacing cause unavoidable H-H contacts.

---

## How it works

**Bond lengths** are drawn from a database of CSD-averaged M-L distances (Orpen et al. 1989), keyed by metal, oxidation state, donor element, and geometry.

**Ligand geometry** is built using internal coordinates. The M-donor-H angle for NH₃, H₂O, and other H-bearing ligands is computed with the metal as the grandparent reference so H atoms always point away from the metal centre.

**Isomers** are enumerated by generating all permutations of ligands across coordination sites and de-duplicating under the point-group symmetry of the geometry (Oₕ for octahedral, D₄ₕ for square planar, Tₐ for tetrahedral).

**POSCAR format** — atoms centred in a cubic vacuum box (≥15 Å padding), species sorted by atomic number, Cartesian coordinates in Å, charge and spin multiplicity in the comment line.
