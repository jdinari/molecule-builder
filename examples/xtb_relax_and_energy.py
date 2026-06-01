"""
xtb_relax_and_energy.py -- relax a complex with xTB and compute DeltaE

Install:  pip install tblite ase
"""
from molbuilder.api import build, poscar
from molbuilder.relaxation import relax, compute_energy, check_bonds_intact

# Two isomers of [Ni(HCOO)2(H2O)4]
mols = build("Ni", ox=2, ligands=["HCOO", "HCOO", "H2O", "H2O", "H2O", "H2O"])

energies = {}
for mol in mols:
    print(f"Relaxing {mol.label} ({mol.formula}) ...", flush=True)
    res = relax(mol, backend="xtb", fmax=0.05, steps=300)
    bc  = check_bonds_intact(mol, res.mol)
    print(f"  E={res.energy_eV:.4f} eV  converged={res.converged}"
          f"  bond_status={'OK' if bc['intact'] else 'BROKEN'}")
    energies[mol.label] = res.energy_eV
    poscar(res.mol, f"Ni_HCOO2_H2O4_{mol.label}_relaxed.POSCAR")

if len(energies) == 2:
    labels = list(energies)
    dE = energies[labels[1]] - energies[labels[0]]
    print(f"\nDeltaE({labels[1]} - {labels[0]}) = {dE:+.4f} eV")
