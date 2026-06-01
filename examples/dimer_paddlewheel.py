"""
dimer_paddlewheel.py -- Ni2(HCOO)4 paddle-wheel (common MOF building unit)
"""
import numpy as np
from molbuilder.api import dimer, poscar, info

mol = dimer("Ni", ox=2, terminal=[], bridge="mu-HCOO", n=4)
ni_pos = [a.position for a in mol.atoms if a.symbol == "Ni"]
ni_ni  = float(np.linalg.norm(ni_pos[0] - ni_pos[1]))

print(f"Formula : {mol.formula}  charge={mol.charge}")
print(f"Ni-Ni   : {ni_ni:.3f} Angstrom")
info(mol)
poscar(mol, "Ni2_HCOO4_paddlewheel.POSCAR")
