"""
Tutorial 07 — Isodesmic Reaction Network and ΔG Screening
===========================================================

What you will learn
-------------------
1.  Build a reaction network from a set of Ni(II) monomers and dimers.
2.  Understand which reactions are already modelled (formic acid route!).
3.  Compute ΔG for every node with xTB thermochemistry.
4.  Screen for the most accessible reactions (ΔG < threshold).
5.  Handle broken-bond structures before they reach DFT.
6.  Export the network as a CSV and a plot.

Background: the formic acid route
-----------------------------------
In your chemistry, formic acid (HCOOH) delivers formate (HCOO⁻) to the Ni
complex while simultaneously protonating a coordinated hydroxide (OH⁻) to
release water.  In one step:

    HCOOH  +  [Ni–OH]  →  [Ni–HCOO]  +  H₂O

This is the canonical isodesmic substitution:
  • Both OH⁻ and HCOO⁻ carry charge −1  →  same total charge on both sides.
  • The references are HCOOH and H₂O, both neutral  →  no charged free species.
  • ΔG is fully reliable at the xTB level in gas phase.

The reaction network captures this automatically for every Ni complex in
your inventory that contains OH⁻.

Install:
    pip install git+https://github.com/jdinari/molecule-builder.git
    pip install tblite ase matplotlib pandas

Run:
    python tutorials/07_reaction_network.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

from molbuilder import enumerate_complexes, MULTI_BRIDGE_CASES
from molbuilder.reactions import ReactionNetwork, ReactionType
from molbuilder.energetics import write_broken_report

OUT = Path("out_tutorial07")
OUT.mkdir(exist_ok=True)

# ── 1. Generate a small inventory ─────────────────────────────────────────────
# CN 4–6 Ni(II) monomers and dimers with HCOO, H2O, OH ligands.

print("=== Step 1: Enumerate structures ===")
rows = list(enumerate_complexes(
    metal="Ni", ox_states=[2],
    ligand_pool=["HCOO", "H2O", "OH"],
    bridge_pool=["mu-OH", "mu-HCOO"],
    nuclearity=[1, 2], cn_range=(4, 6),
    output_root=OUT / "poscar",
    verbose=False,
    multi_bridge_cases=MULTI_BRIDGE_CASES,
))

n_mono  = sum(1 for _, r in rows if r["structure"] == "monomer")
n_dimer = sum(1 for _, r in rows if "dimer" in r["structure"])
print(f"  Monomers: {n_mono}   Dimers: {n_dimer}   Total: {len(rows)}")

# ── 2. Tag some structures as BROKEN to demonstrate the filter ────────────────
# In a real run, bond_status comes from run_energetics().  Here we simulate it.
rows_tagged = []
for i, (mol, row) in enumerate(rows):
    r = dict(row)
    r["bond_status"] = "BROKEN" if i % 8 == 0 else "OK"
    rows_tagged.append((mol, r))

# ── 3. Build the reaction network ─────────────────────────────────────────────
print("\n=== Step 2: Build reaction network ===")
net = ReactionNetwork(
    rows_tagged,
    bond_filter=True,   # exclude BROKEN structures automatically
    verbose=True,
)
print()
print(net.summary())

# ── 4. Inspect the formic acid route ──────────────────────────────────────────
print("\n=== Step 3: The formic acid route (HCOOH + [Ni-OH] → [Ni-HCOO] + H₂O) ===")
hcooh_edges = [
    (s, d, e) for s, d, e in net.substitutions
    if e.get("hcoo_source") and
       net.graph.nodes[s].get("structure") == "monomer"
]
# Deduplicate by (src_ligands, dst_ligands, geometry)
seen = set()
for src, dst, e in hcooh_edges:
    ns = net.graph.nodes[src]
    key = (ns["ligands"], net.graph.nodes[dst]["ligands"], ns["geometry"])
    if key in seen:
        continue
    seen.add(key)
    print(f"  {net.reaction_str(src, dst)}")

# ── 5. Compute xTB ΔG ─────────────────────────────────────────────────────────
print("\n=== Step 4: xTB thermochemistry (ΔG at 298.15 K) ===")
print("  (this takes ~2 min for a small inventory)")
try:
    net.compute_energies(
        backend        = "xtb",
        compute_thermo = True,    # ← ΔG is the default
        T              = 298.15,
        P              = 101325.0,
        fmax           = 0.1,     # loose for tutorial speed
        steps          = 100,
        verbose        = True,
    )
    has_energies = True
except ImportError:
    print("  xTB not installed — skipping (pip install tblite ase)")
    has_energies = False

# ── 6. Screen for thermodynamically accessible reactions ──────────────────────
if has_energies:
    print("\n=== Step 5: Screen reactions (ΔG ≤ 1.0 eV) ===")
    print()

    hits = net.screen(max_dE=1.0, use_gibbs=True, require_energy=True)
    print(f"  {len(hits)} reaction(s) with ΔG ≤ 1.0 eV:")
    for src, dst, e, dG in hits[:10]:
        rtype = e["reaction_type"]
        print(f"    {dG:+.3f} eV  [{rtype:12s}]  {net.reaction_str(src, dst)}")
    if len(hits) > 10:
        print(f"    … and {len(hits)-10} more")

    print()
    print("  Formic acid substitutions (HCOOH + [Ni-OH] → [Ni-HCOO] + H₂O):")
    hcooh_hits = [
        (s, d, e, dG) for s, d, e, dG in hits
        if e.get("hcoo_source") and
           net.graph.nodes[s].get("structure") == "monomer"
    ]
    if hcooh_hits:
        for src, dst, e, dG in hcooh_hits[:5]:
            print(f"    {dG:+.3f} eV  {net.reaction_str(src, dst)}")
    else:
        print("    (none within ΔG ≤ 1.0 eV threshold — try increasing max_dE)")

# ── 7. Broken-structure report ─────────────────────────────────────────────────
print("\n=== Step 6: Broken-structure report ===")
if net.broken_structures:
    write_broken_report(net.broken_structures, OUT, verbose=True)
else:
    print("  No broken structures.")

# ── 8. Export ─────────────────────────────────────────────────────────────────
print("\n=== Step 7: Export ===")

df = net.to_dataframe()
csv_path = OUT / "reaction_network.csv"
df.to_csv(csv_path, index=False)
print(f"  Reaction network CSV → {csv_path}  ({len(df)} edges)")

cols = ["src_ligands", "dst_ligands", "reaction_type",
        "incoming", "outgoing", "delta_e_eV", "delta_g_eV"]
available_cols = [c for c in cols if c in df.columns]
if available_cols:
    print()
    print("  CSV columns (first 5 rows):")
    print(df[available_cols].head().to_string(index=False))

# Plot (requires matplotlib)
try:
    import matplotlib.pyplot as plt
    fig = net.plot(
        edge_label    = "delta_g" if has_energies else "delta_e",
        title         = "Ni coordination network (isodesmic, ΔG eV)",
        show_broken   = True,
    )
    plot_path = OUT / "reaction_network.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Network plot → {plot_path}")
except ImportError:
    print("  (plot skipped — pip install matplotlib)")

print()
print("=== Summary ===")
print(f"  Total structures in network : {sum(1 for _, d in net.graph.nodes(data=True) if d['node_type'] == 'complex')}")
print(f"  Excluded (BROKEN)           : {len(net.broken_structures)}")
print(f"  Substitution reactions      : {len(net.substitutions) // 2}")
print(f"  Coordination/Decoord        : {len(net.coordinations) // 2}")
print(f"  Associations                : {len(net.associations)}")
print()
print("Key takeaways:")
print("  • HCOOH + [Ni-OH] → [Ni-HCOO] + H₂O is the formic acid route.")
print("    It is isodesmic (ΔG fully reliable) and already in the network.")
print("  • Broken structures are excluded from reactions and written to out/broken/.")
print("  • screen(max_dE=X) returns accessible reactions sorted by ΔG.")
print("  • to_dataframe() gives a flat CSV for downstream analysis.")
