"""
heteroleptic_dimer.py -- asymmetric dimer with different ligands on each Ni
"""
import numpy as np
from molbuilder.api import dimer, poscar

# Water on Ni1 only; Ni2 is bare.  Both share 4 bridging formates.
mol = dimer("Ni", ox=2,
            terminal_m1=["H2O"], terminal_m2=[],
            bridge="mu-HCOO", n=4)

print(f"Formula : {mol.formula}  charge={mol.charge}")
ni_atoms = [a for a in mol.atoms if a.symbol == "Ni"]
for i, ni in enumerate(ni_atoms):
    o_donors = [a for a in mol.atoms
                if a.symbol == "O" and np.linalg.norm(a.position - ni.position) < 2.5]
    print(f"Ni{i}: CN={len(o_donors)}")

poscar(mol, "Ni2_HCOO4_H2O_hetero.POSCAR")
