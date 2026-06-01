"""
reaction_network.py -- Build and screen an isodesmic reaction network

Demonstrates:
    * Formic acid route: HCOOH + [Ni-OH] -> [Ni-HCOO] + H2O
    * DeltaG screening to find accessible reactions
    * Broken-structure flagging
    * Network plot and CSV export

Install:
    pip install git+https://github.com/jdinari/molecule-builder.git
    pip install tblite ase matplotlib pandas

Run from the repository root:
    python examples/reaction_network.py
"""

from pathlib import Path

from molbuilder import enumerate_complexes, MULTI_BRIDGE_CASES
from molbuilder.reactions import ReactionNetwork, ReactionType
from molbuilder.energetics import write_broken_report

OUT = Path("out_rxn_example")
OUT.mkdir(exist_ok=True)

# -- Build inventory -----------------------------------------------------------
rows = list(enumerate_complexes(
    metal="Ni", ox_states=[2],
    ligand_pool=["HCOO", "H2O", "OH"],
    nuclearity=[1], cn_range=(4, 6),
    output_root=OUT / "poscar",
    verbose=False,
    multi_bridge_cases=MULTI_BRIDGE_CASES,
))
print(f"Inventory: {len(rows)} structures")

# -- Build reaction network ----------------------------------------------------
net = ReactionNetwork(rows, bond_filter=True, verbose=True)

# -- Compute DeltaG with xTB ------------------------------------------------------
try:
    net.compute_energies(backend="xtb", compute_thermo=True,
                         T=298.15, fmax=0.05, verbose=True)
    has_g = True
except ImportError:
    print("xTB not installed -- run: pip install tblite ase")
    has_g = False

# -- Print formic acid reactions -----------------------------------------------
print("\n--- Formic acid route (unique reactions) ---")
seen = set()
for src, dst, e in net.substitutions:
    if not e.get("hcoo_source"):
        continue
    if net.graph.nodes[src].get("structure") != "monomer":
        continue
    key = (net.graph.nodes[src]["ligands"], net.graph.nodes[dst]["ligands"],
           net.graph.nodes[src]["geometry"])
    if key in seen:
        continue
    seen.add(key)
    print(f"  {net.reaction_str(src, dst)}")

# -- Screen --------------------------------------------------------------------
if has_g:
    print("\n--- Most accessible reactions (DeltaG <= 0.5 eV) ---")
    for src, dst, e, dG in net.screen(max_dE=0.5, use_gibbs=True)[:5]:
        print(f"  {dG:+.3f} eV  {net.reaction_str(src, dst)}")

# -- Report broken structures --------------------------------------------------
if net.broken_structures:
    write_broken_report(net.broken_structures, OUT, verbose=True)

# -- Export --------------------------------------------------------------------
net.to_dataframe().to_csv(OUT / "reactions.csv", index=False)
print(f"\nCSV -> {OUT}/reactions.csv")

try:
    import matplotlib.pyplot as plt
    fig = net.plot(title="Ni reaction network", edge_label="delta_g" if has_g else "delta_e")
    fig.savefig(OUT / "network.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot -> {OUT}/network.png")
except ImportError:
    print("(plot skipped -- pip install matplotlib)")
