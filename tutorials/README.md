# molbuilder tutorials

Step-by-step examples covering the full molbuilder workflow from first structure
to thermochemistry on a cluster.

---

## Installation

```bash
# Core package
pip install git+https://github.com/jdinari/molecule-builder.git

# xTB relaxation and thermochemistry (tutorials 05, 06, 07)
pip install tblite ase

# MACE relaxation (tutorials 04) — downloads ~500 MB model on first use
pip install mace-torch ase

# Excel output (tutorials 06)
pip install openpyxl

# Reaction network plot (tutorial 07)
pip install matplotlib pandas
```

---

## Running tutorials

Run any tutorial from the repository root:

```bash
python tutorials/01_first_complex.py
python tutorials/02_isomers_and_bidentate.py
python tutorials/03_dimers_and_trimers.py
python tutorials/04_mace_relaxation.py       # requires: pip install mace-torch ase
python tutorials/05_thermochemistry.py       # requires: pip install tblite ase
python tutorials/06_enumeration_and_energetics.py  # requires: pip install tblite ase openpyxl
python tutorials/07_reaction_network.py      # requires: pip install tblite ase matplotlib pandas
```

Output files are written to `out_tutorial0N/`.

---

## Tutorials

| # | File | What it covers |
|---|------|---------------|
| 01 | `01_first_complex.py` | Build a Ni(II) hexaaqua complex, print a summary, write POSCAR and XYZ. Introduction to `build()`, `poscar()`, `info()`. |
| 02 | `02_isomers_and_bidentate.py` | Automatic isomer enumeration (fac/mer, cis/trans), bidentate chelating ligands (`HCOO:bi`, `en`, `bpy`), batch generation over a ligand pool. |
| 03 | `03_dimers_and_trimers.py` | Di-μ-hydroxo and paddle-wheel Ni dimers, heteroleptic dimers (different terminals per metal), linear and triangular trimers, heteroleptic Ni₃ with water on one metal only. |
| 04 | `04_mace_relaxation.py` | Full cluster workflow: generate structures, relax with a local MACE-MH-1 model, detect broken bonds, write ΔE CSV. SLURM submission snippet included. |
| 05 | `05_thermochemistry.py` | xTB thermochemistry (ΔE + ΔG) for ligand substitution; cis vs trans isomer comparison; `gibbs_at(T)` re-evaluation at multiple temperatures without re-running. |
| 06 | `06_enumeration_and_energetics.py` | Batch enumeration with `enumerate_complexes()`, `run_energetics()`, bond-status reporting, Excel output with colour-coded bond flags. |
| 07 | `07_reaction_network.py` | Isodesmic reaction network: the formic acid route, ΔG screening, broken-structure filtering, CSV and plot export. |

---

## Using a local MACE model on a cluster

Every tutorial that uses MACE has `MACE_MODEL` and `MACE_DEVICE` variables near the top:

```python
MACE_MODEL  = None          # ← set this to your local .model path
MACE_DEVICE = "cpu"         # ← "cuda" on a GPU node
```

Leaving `MACE_MODEL = None` triggers an automatic download from GitHub, which
will fail on cluster compute nodes without internet access.  Download the model
once on a login node and set the absolute path:

```python
from mace.calculators.foundations_models import download_mace_mp_checkpoint
path = download_mace_mp_checkpoint("mh-1")
print(path)   # copy this into MACE_MODEL in the tutorial
```

Then in the tutorial script:

```python
MACE_MODEL  = "/scratch/yourname/models/mace-mh-1.model"
MACE_DEVICE = "cuda"
```

The `head="omol"` argument is set automatically inside molbuilder.  This selects
the OMOL/OC20 molecular head trained with wB97M-V references, which is the most
appropriate choice for transition-metal coordination complexes.

---

## Backend comparison: xTB vs MACE

| Property | xTB (GFN2-xTB) | MACE-MH-1 |
|---|---|---|
| Install | `pip install tblite ase` | `pip install mace-torch ase` |
| Model size | < 1 MB | ~500 MB |
| Charge/spin | ✓ explicit | ✗ implicit only |
| Ni(III) open-shell | ✓ reliable | ✗ verify manually |
| Speed (CPU) | ~5 s / structure | ~8 s / structure |
| Speed (GPU) | N/A | ~0.5 s / structure |
| Thermochemistry (ΔG) | ✓ via harmonic freq | ✓ via numerical freq |
| Recommended for | All Ni coordination chemistry | Larger batches on GPU |

For Ni(II)/Ni(III) coordination complexes, **xTB is the recommended default**
because it has explicit d-electron parametrisation and handles charge and spin
multiplicity directly.

---

## Bond status

After relaxation, every structure is classified as:

| Status | Meaning |
|---|---|
| `OK` | All M-L bonds within 1.20× initial length |
| `STRETCHED` | Longest M-L bond 1.20–1.35× initial (possible strain) |
| `BROKEN` | At least one M-L bond > 1.35× initial (ligand likely dissociated) |

Bond breaking is physically meaningful — if xTB says a ligand departs, the
coordination is genuinely strained.  Review broken-bond structures before
passing them to DFT.  Do not use `constrain_bonds=True` to suppress them.

---

## Known limitations

- `dimer(..., mm_bond=True)` with terminal halides may place Cl on the metal
  due to the mm-axis direction not being excluded from terminal placement.
  Use `mm_bond=True` only with bare dimers (no terminals) for now.
- Ru₃(μ-CO) triangular trimers produce C-C clashes at ideal geometry.
  These are geometrically unfeasible as rigid-template structures; use MACE/xTB
  relaxation or start from a crystallographic geometry.
- CN=7 (pentagonal bipyramidal) structures with ≥2 H-bearing ligands may have
  unavoidable H-H contacts at ideal geometry and are filtered out of the
  enumeration automatically.
