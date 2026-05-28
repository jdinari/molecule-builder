# Examples

Short, self-contained scripts — one idea per file.
Copy-paste or run directly from the repository root.

```bash
python examples/build_single.py
python examples/isomers.py
# etc.
```

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

## Dependencies

```bash
pip install git+https://github.com/jdinari/molecule-builder.git

# For xTB (tutorials 05, 06 and examples xtb_*)
pip install tblite ase

# For MACE (tutorial 04)
pip install mace-torch ase

# For Excel output
pip install openpyxl
```
