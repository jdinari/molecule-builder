# Tutorials

Step-by-step examples covering the main molbuilder workflows.

| File | What it covers |
|------|---------------|
| `01_first_complex_and_relaxation.py` | Build a Ni(II) hexaaqua complex, write a POSCAR, relax it with MACE. Good starting point for cluster use. |
| `02_isomers_and_batch.py` | Automatic isomer enumeration for octahedral and square-planar complexes; batch POSCAR generation over a ligand pool. |
| `03_dimers_and_trimers.py` | Di-μ-hydroxo and di-μ-formate Ni dimers; linear and triangular trimers; Re quadruple bond. |
| `04_mace_cluster_batch.py` | Full cluster workflow: generate structures, relax all with a **local** MACE model on GPU, write a ΔE CSV. Includes a SLURM submission snippet. |
| `05_thermochemistry_substitution.py` | Compute ΔG for cis/trans isomers and ligand substitution using xTB thermochemistry; re-evaluate at multiple temperatures. |

## Quick start

```bash
# install dependencies
pip install git+https://github.com/jdinari/molecule-builder.git
pip install tblite ase          # for xTB backend
pip install mace-torch ase      # for MACE backend

# run any tutorial
python tutorials/01_first_complex_and_relaxation.py
```

## Using your own MACE model on a cluster

Every tutorial that uses MACE has a `MACE_MODEL` variable near the top:

```python
MACE_MODEL  = None          # ← set this to your local .model path
MACE_DEVICE = "cpu"         # ← "cuda" on a GPU node
```

Set `MACE_MODEL` to the **absolute path** of your checkpoint file before
submitting. Leaving it as `None` causes mace-torch to download `mh-1`
from the internet, which will fail on most cluster compute nodes.

Example:
```python
MACE_MODEL  = "/scratch/yourname/models/mace-mh-1.model"
MACE_DEVICE = "cuda"
```
