# molbuilder tutorials

Step-by-step examples covering the full molbuilder workflow from first structure
to thermochemistry on a cluster.

---

## Quick start

```bash
# Install the package and dependencies
pip install git+https://github.com/jdinari/molecule-builder.git

# For xTB relaxation and thermochemistry
pip install tblite ase

# For MACE relaxation (requires ~500 MB model download on first use)
pip install mace-torch ase

# For Excel output
pip install openpyxl
```

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

Run any tutorial from the repository root:

```bash
python tutorials/01_first_complex.py
```

Output files go to `out_tutorial0N/`.

---

## Using a local MACE model on a cluster

Every tutorial that uses MACE has a `MACE_MODEL` variable near the top:

```python
MACE_MODEL  = None          # ← set this to your local .model path
MACE_DEVICE = "cpu"         # ← "cuda" on a GPU node
```

Set `MACE_MODEL` to the **absolute path** of your checkpoint before submitting.
Leaving it `None` causes an automatic download from GitHub, which will fail
on most cluster compute nodes without internet access.

```python
MACE_MODEL  = "/scratch/yourname/models/mace-mh-1.model"
MACE_DEVICE = "cuda"
```

The model file (~500 MB) can be downloaded once on a login node:

```python
from mace.calculators.foundations_models import download_mace_mp_checkpoint
path = download_mace_mp_checkpoint("mh-1")
print(path)   # copy this path into MACE_MODEL above
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
| Thermochemistry (ΔG) | ✓ via freq | ✓ via freq (numerical) |
| Recommended for | All Ni coordination chemistry | Larger clusters on GPU |

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
coordination is genuinely strained.  **Do not use `constrain_bonds=True`** as a
workaround; instead, review broken-bond structures before passing them to DFT.

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
