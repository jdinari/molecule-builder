"""
geometry.py
===========
Ideal coordination geometry vectors for transition metal complexes.
All vectors are unit vectors pointing FROM the metal TO the donor atom.
"""

import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────
_s2 = 1 / np.sqrt(2)
_s3 = 1 / np.sqrt(3)

def _norm(v):
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)

# ── Geometry definitions ─────────────────────────────────────────────────────
# Each key maps to a list of unit vectors (one per coordination site).
# The 'aliases' dict maps alternative names to canonical names.

GEOMETRIES = {
    # CN 2
    "lin": [
        np.array([1., 0., 0.]),
        np.array([-1., 0., 0.]),
    ],
    "bent": [
        _norm([1., 0., 0.]),
        _norm([-0.5, 0.866, 0.]),      # ~120° H₂O-like
    ],

    # CN 3
    "tp": [
        np.array([1., 0., 0.]),
        _norm([-0.5,  0.866, 0.]),
        _norm([-0.5, -0.866, 0.]),
    ],
    "tshaped": [
        np.array([1., 0., 0.]),
        np.array([-1., 0., 0.]),
        np.array([0., 1., 0.]),
    ],

    # CN 4
    "tet": [
        _norm([ 1.,  1.,  1.]),
        _norm([ 1., -1., -1.]),
        _norm([-1.,  1., -1.]),
        _norm([-1., -1.,  1.]),
    ],
    "sqp": [
        np.array([1., 0., 0.]),
        np.array([0., 1., 0.]),
        np.array([-1., 0., 0.]),
        np.array([0., -1., 0.]),
    ],
    "seesaw": [
        np.array([0., 0., 1.]),
        np.array([1., 0., 0.]),
        np.array([-1., 0., 0.]),
        np.array([0., 0., -1.]),
    ],

    # CN 5
    "tbp": [
        np.array([0., 0., 1.]),
        np.array([0., 0., -1.]),
        np.array([1., 0., 0.]),
        _norm([-0.5,  0.866, 0.]),
        _norm([-0.5, -0.866, 0.]),
    ],
    "sqpy": [
        np.array([0., 0., 1.]),
        np.array([1., 0., 0.]),
        np.array([0., 1., 0.]),
        np.array([-1., 0., 0.]),
        np.array([0., -1., 0.]),
    ],

    # CN 6
    "oct": [
        np.array([1., 0., 0.]),
        np.array([-1., 0., 0.]),
        np.array([0., 1., 0.]),
        np.array([0., -1., 0.]),
        np.array([0., 0., 1.]),
        np.array([0., 0., -1.]),
    ],
    "tpr": [   # Trigonal prismatic
        _norm([ 1., 0.,  1.]),
        _norm([-0.5,  0.866,  1.]),
        _norm([-0.5, -0.866,  1.]),
        _norm([ 1., 0., -1.]),
        _norm([-0.5,  0.866, -1.]),
        _norm([-0.5, -0.866, -1.]),
    ],

    # CN 7
    "pbp": [   # Pentagonal bipyramidal
        np.array([0., 0., 1.]),
        np.array([0., 0., -1.]),
        np.array([1., 0., 0.]),
        _norm([np.cos(2*np.pi/5), np.sin(2*np.pi/5), 0.]),
        _norm([np.cos(4*np.pi/5), np.sin(4*np.pi/5), 0.]),
        _norm([np.cos(6*np.pi/5), np.sin(6*np.pi/5), 0.]),
        _norm([np.cos(8*np.pi/5), np.sin(8*np.pi/5), 0.]),
    ],

    # CN 8
    "sapr": [  # Square antiprismatic
        _norm([ 1.,  1.,  1.]),
        _norm([-1.,  1.,  1.]),
        _norm([ 1., -1.,  1.]),
        _norm([-1., -1.,  1.]),
        _norm([ 1.,  0., -1.]),
        _norm([-1.,  0., -1.]),
        _norm([ 0.,  1., -1.]),
        _norm([ 0., -1., -1.]),
    ],
}

# Aliases
GEOMETRY_ALIASES = {
    "oh":                "oct",
    "octahedral":        "oct",
    "sp":                "sqp",
    "square_planar":     "sqp",
    "square-planar":     "sqp",
    "td":                "tet",
    "tetrahedral":       "tet",
    "linear":            "lin",
    "spy":               "sqpy",
    "square_pyramidal":  "sqpy",
    "tbp":               "tbp",
    "trig_bipy":         "tbp",
    "trigonal_bipyramidal": "tbp",
    "trigonal_prismatic": "tpr",
    "pentagonal_bipyramidal": "pbp",
    "square_antiprismatic": "sapr",
}

GEOMETRY_DESCRIPTIONS = {
    "lin":   ("Linear",                  2),
    "bent":  ("Bent",                    2),
    "tp":    ("Trigonal planar",         3),
    "tshaped": ("T-shaped",              3),
    "tet":   ("Tetrahedral",             4),
    "sqp":   ("Square planar",           4),
    "seesaw": ("See-saw",                4),
    "tbp":   ("Trigonal bipyramidal",    5),
    "sqpy":  ("Square pyramidal",        5),
    "oct":   ("Octahedral",             6),
    "tpr":   ("Trigonal prismatic",      6),
    "pbp":   ("Pentagonal bipyramidal",  7),
    "sapr":  ("Square antiprismatic",    8),
}


def resolve_geometry(name: str) -> str:
    """Return canonical geometry key from any alias."""
    if name is None:
        return None
    k = name.lower().strip()
    return GEOMETRY_ALIASES.get(k, k)


def get_geometry_vectors(geometry: str) -> list:
    """Return list of unit vectors for the given geometry."""
    canon = resolve_geometry(geometry)
    if canon not in GEOMETRIES:
        available = ", ".join(sorted(GEOMETRIES.keys()))
        raise KeyError(f"Unknown geometry '{geometry}'. Available: {available}")
    return list(GEOMETRIES[canon])


def infer_geometry(cn: int) -> str:
    """Best-guess geometry for a given coordination number."""
    defaults = {
        1: "lin", 2: "lin", 3: "tp", 4: "tet",
        5: "sqpy", 6: "oct", 7: "pbp", 8: "sapr",
    }
    return defaults.get(cn, "oct")


def list_geometries() -> list:
    """Return list of (key, name, CN) tuples."""
    return [(k, v[0], v[1]) for k, v in GEOMETRY_DESCRIPTIONS.items()]
