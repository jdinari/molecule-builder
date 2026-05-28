"""
trimer_triangular.py — Ni3(HCOO)6 triangular trimer with optional terminal water
"""
import numpy as np
from molbuilder.api import trimer, poscar

# Bare triangular Ni3 — equilateral triangle, each Ni is CN=4
mol1 = trimer("Ni", ox=2, terminal=[], bridge="mu-HCOO",
              arrangement="triangular", n_bridges_per_pair=2)
ni   = [a.position for a in mol1.atoms if a.symbol == "Ni"]
d_nn = [np.linalg.norm(ni[i]-ni[j]) for i in range(3) for j in range(i+1,3)]
print(f"Ni3(HCOO)6:  {mol1.formula}  Ni-Ni = {sorted(d_nn)[0]:.3f} Å (equilateral)")
poscar(mol1, "Ni3_HCOO6.POSCAR")

# H2O on Ni0 only — asymmetric coordination
mol2 = trimer("Ni", ox=2, bridge="mu-HCOO",
              arrangement="triangular", n_bridges_per_pair=2,
              terminals_per_metal=[["H2O"], [], []])
print(f"Ni3(HCOO)6(H2O):  {mol2.formula}  charge={mol2.charge}")
for i, ni in enumerate([a for a in mol2.atoms if a.symbol=="Ni"]):
    cn = len([a for a in mol2.atoms if a.symbol=="O"
              and np.linalg.norm(a.position-ni.position)<2.5])
    print(f"  Ni{i}: CN={cn}")
poscar(mol2, "Ni3_HCOO6_H2O_hetero.POSCAR")
