# Examples

Short, self-contained scripts — one idea per file.
Run directly from the repository root after installing the package.

## Installation

```bash
pip install git+https://github.com/jdinari/molecule-builder.git

# For xTB relaxation and thermochemistry (xtb_* examples)
pip install tblite ase

# For Excel output (batch_enumerate.py)
pip install openpyxl

# For reaction network plot (reaction_network.py)
pip install matplotlib pandas
```

## Running examples

```bash
python examples/build_single.py
python examples/isomers.py
python examples/cisplatin.py
python examples/dimer_paddlewheel.py
python examples/heteroleptic_dimer.py
python examples/trimer_triangular.py
python examples/xtb_relax_and_energy.py    # requires: pip install tblite ase
python examples/xtb_delta_g.py             # requires: pip install tblite ase
python examples/batch_enumerate.py         # requires: pip install tblite ase openpyxl
python examples/reaction_network.py        # requires: pip install tblite ase matplotlib pandas
```

## File descriptions

| File | What it shows |
|------|--------------|
| `build_single.py` | Build one complex, print summary, write POSCAR |
| `isomers.py` | Enumerate fac/mer isomers of [FeCl₃(H₂O)₃] |
| `cisplatin.py` | cis and trans [PdCl₂(NH₃)₂] — cisplatin and transplatin |
| `dimer_paddlewheel.py` | Ni₂(HCOO)₄ paddle-wheel (MOF secondary building unit) |
| `heteroleptic_dimer.py` | Asymmetric dimer: H₂O on one Ni only |
| `trimer_triangular.py` | Ni₃(HCOO)₆ equilateral trimer; heteroleptic variant with H₂O on one metal |
| `xtb_relax_and_energy.py` | Relax cis/trans isomers with xTB, compare ΔE |
| `xtb_delta_g.py` | Full ΔG for a ligand substitution reaction |
| `batch_enumerate.py` | Enumerate a set of complexes and run batch xTB energetics |
| `reaction_network.py` | Build and screen an isodesmic reaction network |
