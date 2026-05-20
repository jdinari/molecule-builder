# Extended Ligand Features

This document describes the enhancements added to molbuilder for:
1. Custom ligands from POSCAR files
2. Variable denticity binding modes
3. Bridging ligands for polynuclear complexes
4. New ligands: formate (HCOO⁻) and formic acid (HCOOH)

---

## 1. Custom Ligands from POSCAR Files

### Python API

```python
from molbuilder.ligands.custom_poscar import load_custom_ligand
from molbuilder.api_extended import build_with_custom_ligands

# Load custom ligand from POSCAR
custom_lig = load_custom_ligand(
    "my_custom_ligand.POSCAR",
    donor_atom_indices=[0, 2],  # Atoms 0 and 2 are donors
    charge=0,
    name="myligand"
)

# Build complex with custom ligand
mol = build_with_custom_ligands(
    "Fe", ox=2,
    ligands=["Cl", "Cl"],
    custom_ligands=[custom_lig],
    geometry="oct"
)
```

### CLI

```bash
python molbuilder/cli_extended.py \
    --metal Fe --ox 2 \
    --ligands Cl Cl Cl Cl \
    --custom-ligand custom_ligand.POSCAR \
    --custom-donor-atoms 0,2 \
    --custom-ligand-charge 0 \
    --custom-ligand-name myligand \
    --out Fe_complex.POSCAR
```

**Key Points:**
- The POSCAR file should contain only the ligand molecule
- Donor atoms are specified by 0-indexed atom numbers
- Multiple custom ligands can be used (repeat `--custom-ligand` and `--custom-donor-atoms`)
- Charge should reflect the formal charge of the free ligand

---

## 2. Variable Denticity Binding Modes

Ligands can now be specified with binding modes using the colon notation:
- `ligand:mode` format
- Examples: `bpy:bi`, `OH:bridge`, `HCOO:mono`

### Supported Modes

#### Bipyridine (bpy / bipy)
- `bpy:mono` - Monodentate (1 N donor)
- `bpy:bi` - Bidentate (2 N donors) [default]
- `bpy:bridge` - Bridging between two metals

#### Ethylenediamine (en)
- `en:bi` - Bidentate (2 N donors) [default]
- `en:bridge` - Bridging between two metals

#### Hydroxide (OH)
- `OH:mono` - Monodentate (terminal) [default]
- `OH:bridge` or `mu-OH` - Bridging between metals

#### Carbonyl (CO)
- `CO:mono` - Monodentate (terminal) [default]
- `CO:bridge` or `mu-CO` - Bridging between metals

#### Formate (HCOO)
- `HCOO:mono` - Monodentate (1 O donor)
- `HCOO:bi` - Bidentate chelating (2 O donors)
- `HCOO:bridge` - Bridging between two metals

### Python API

```python
from molbuilder.api_extended import build_with_denticity_modes

# Build [Fe(bpy)2(Cl)2] with bidentate bpy
mol = build_with_denticity_modes(
    "Fe", ox=2,
    ligands_with_modes={
        "bpy": "bi",   # bidentate
        "Cl": None,    # use default
    },
    geometry="oct"
)

# Build [Pd(bpy)Cl2] with monodentate bpy
mol = build_with_denticity_modes(
    "Pd", ox=2,
    ligands_with_modes={
        "bpy": "mono",  # monodentate
        "Cl": None,
    },
    geometry="sqp"
)
```

### CLI

```bash
# Bidentate bipyridine
python molbuilder/cli_extended.py \
    --metal Fe --ox 2 \
    --ligand-mode bpy:bi \
    --ligand-mode Cl \
    --ligand-mode Cl \
    --geometry oct \
    --out Fe_bpy2_Cl2.POSCAR

# Mixed monodentate and bidentate
python molbuilder/cli_extended.py \
    --metal Pd --ox 2 \
    --ligand-mode bpy:mono \
    --ligand-mode Cl \
    --ligand-mode Cl \
    --geometry sqp \
    --out Pd_complex.POSCAR
```

---

## 3. Bridging Ligands for Polynuclear Complexes

Bridging ligands can connect two or more metal centers using the `--bridge-ligand` argument.

### Python API

```python
from molbuilder.api_extended import dimer_with_bridging_ligands

# Build [Fe2(μ-HCOO)2(H2O)4]
mol = dimer_with_bridging_ligands(
    "Fe", ox=3,
    terminal_ligands=["H2O", "H2O"],
    bridging_ligands={"HCOO": "bi"},  # bidentate bridging formate
    bridging_count=2,  # Two bridging formates
    geometry="oct"
)

# Build [Rh2(μ-OH)2(CO)2] with Rh-Rh bonding
mol = dimer_with_bridging_ligands(
    "Rh", ox=1,
    terminal_ligands=["CO"],
    bridging_ligands={"OH": "bridge"},
    bridging_count=2,
    mm_bond=True,  # Metal-metal bonding
)
```

### CLI

```bash
# [Fe2(μ-HCOO)2(H2O)4] dimer
python molbuilder/cli_extended.py \
    --dimer \
    --metal Fe --ox 3 \
    --ligands H2O H2O \
    --bridge-ligand HCOO:bi \
    --bridge-count 2 \
    --geometry oct \
    --out Fe2_formate_dimer.POSCAR

# [Rh2(μ-OH)2(CO)2] dimer with M-M bond
python molbuilder/cli_extended.py \
    --dimer \
    --metal Rh --ox 1 \
    --ligands CO \
    --bridge-ligand OH:bridge \
    --bridge-count 2 \
    --out Rh2_dimer.POSCAR
```

---

## 4. New Ligands: Formate and Formic Acid

### Formate (HCOO⁻)

Structure: `H-C(=O)-O⁻`

**Available binding modes:**
- **Monodentate**: Single O donor (charge -1)
- **Bidentate**: Chelating (O,O) with bite angle ~55° (charge -1)
- **Bridging**: Bridges two metals via both oxygens (charge -1)

```python
mol = build_with_denticity_modes(
    "Fe", ox=2,
    ligands_with_modes={
        "HCOO": "mono",  # monodentate formate
        "H2O": None,
    },
    geometry="oct"
)

mol = build_with_denticity_modes(
    "Cu", ox=2,
    ligands_with_modes={
        "HCOO": "bi",  # bidentate chelating formate
        "H2O": None,
    },
    geometry="sqp"
)
```

### Formic Acid (HCOOH)

Structure: `H-C(=O)-OH`

**Available binding modes:**
- **Monodentate**: Via carbonyl O donor (charge 0)
- **Bidentate**: Chelating (O,O) with bite angle ~52° (charge 0)

```python
mol = build_with_denticity_modes(
    "Fe", ox=3,
    ligands_with_modes={
        "HCOOH": "mono",  # monodentate formic acid
        "H2O": None,
    },
    geometry="oct"
)
```

---

## Integration with Existing Code

The extended features are designed to be backward compatible:

1. **Original `build()` function** still works unchanged
2. **Extended functions** provide additional capabilities
3. **Mode modifiers are optional** - if not specified, defaults are used
4. **Existing ligand library** is preserved

### Backward Compatibility

```python
# Original code still works
from molbuilder.api import build, poscar

mol = build("Fe", ox=3, ligands=["Cl"]*6)
poscar(mol, "FeCl6.POSCAR")

# New features available via extended API
from molbuilder.api_extended import build_with_denticity_modes

mol = build_with_denticity_modes(
    "Fe", ox=3,
    ligands_with_modes={"HCOO": "bi"},
)
```

---

## File Structure

```
molbuilder/
├── ligands/
│   ├── denticity_modes.py       # Denticity mode classes
│   ├── library_extended.py      # New ligands and mode definitions
│   └── custom_poscar.py         # Custom POSCAR ligand loader
├── api_extended.py              # Extended build functions
├── cli_extended.py              # Extended CLI
└── EXTENDED_FEATURES.md         # This file
```

---

## Examples

### Example 1: [Fe(HCOO)3]³⁻ with Monodentate Formate

```python
from molbuilder.api_extended import build_with_denticity_modes
from molbuilder.output.poscar_writer import poscar_to_string

mol = build_with_denticity_modes(
    "Fe", ox=3,
    ligands_with_modes={
        "HCOO": "mono",
    },
    geometry="oct"
)
print(poscar_to_string(mol))
```

### Example 2: [Fe2(μ-HCOO)2(H2O)4] with Bridging Formate

```python
from molbuilder.api_extended import dimer_with_bridging_ligands

mol = dimer_with_bridging_ligands(
    "Fe", ox=3,
    terminal_ligands=["H2O", "H2O"],
    bridging_ligands={"HCOO": "bi"},
    bridging_count=2,
    geometry="oct"
)
```

### Example 3: Custom Norbornane Ligand

```python
from molbuilder.ligands.custom_poscar import load_custom_ligand
from molbuilder.api_extended import build_with_custom_ligands

# Create norbornane.POSCAR with donor atoms at indices 0 and 2
norbornane = load_custom_ligand(
    "norbornane.POSCAR",
    donor_atom_indices=[0, 2],
    charge=0,
    name="norbornane"
)

mol = build_with_custom_ligands(
    "Pd", ox=2,
    custom_ligands=[norbornane],
    ligands=["Cl", "Cl"],
    geometry="sqp"
)
```

---

## Future Enhancements

- [ ] Support for polydentate ligands (tridentate, hexadentate EDTA variants)
- [ ] Bridging via different atom combinations (e.g., one O and one N)
- [ ] Automatic donor atom detection from POSCAR via distance analysis
- [ ] Geometry optimization for bridging ligand arrangements
- [ ] Bite angle validation and warnings
