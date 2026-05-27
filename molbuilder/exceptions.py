"""
exceptions.py
=============
Custom exception hierarchy for molbuilder.

All public-facing errors raised by the library are subclasses of
``MolbuilderError`` so callers can catch the whole family with one clause:

    try:
        mol = build(...)
    except MolbuilderError as e:
        ...

Hierarchy
---------
MolbuilderError
├── InvalidLigandError      – unknown or malformed ligand name/spec
├── GeometryError           – impossible or unsupported geometry
│   └── ClashError          – atom-atom overlap that cannot be resolved
├── CoordinationError       – CN / donor count / oxidation-state inconsistency
├── ChargeError             – charge balance cannot be satisfied
└── ValidationError         – structure failed post-build validation checks
"""


class MolbuilderError(Exception):
    """Base class for all molbuilder errors."""


# ── ligand errors ─────────────────────────────────────────────────────────────

class InvalidLigandError(MolbuilderError):
    """
    Raised when a ligand name cannot be resolved from the library and does
    not look like a valid SMILES string.

    Example
    -------
    >>> build("Ni", ox=2, ligands=["TYPO"])
    InvalidLigandError: Unknown ligand 'TYPO'. ...
    """


# ── geometry errors ───────────────────────────────────────────────────────────

class GeometryError(MolbuilderError):
    """
    Raised when a requested or inferred geometry is impossible or unsupported
    for the given coordination number.

    Example
    -------
    >>> build("Fe", ox=2, ligands=["H2O"]*6, geometry="tet")
    GeometryError: Geometry 'tet' requires CN=4, but 6 ligands were supplied.
    """


class ClashError(GeometryError):
    """
    Raised when atom-atom overlaps in the built structure exceed hard thresholds
    and cannot be resolved by the placement algorithm.

    Carries the list of clashing atom-pair descriptions from the validator.

    Attributes
    ----------
    clashes : list[str]
        Human-readable descriptions of each problematic contact.
    """

    def __init__(self, message: str, clashes: list[str] | None = None):
        super().__init__(message)
        self.clashes: list[str] = clashes or []


# ── coordination errors ───────────────────────────────────────────────────────

class CoordinationError(MolbuilderError):
    """
    Raised when the coordination environment is chemically implausible:
    wrong number of donors, incompatible oxidation state, or unsupported CN.

    Example
    -------
    >>> build("Ni", ox=2, ligands=["H2O"]*9)  # CN=9 not supported
    CoordinationError: ...
    """


# ── charge errors ─────────────────────────────────────────────────────────────

class ChargeError(MolbuilderError):
    """
    Raised when a charge-neutral complex cannot be constructed from the
    supplied combination of metal oxidation state and ligands.

    Example
    -------
    >>> # enumerator strict mode: no neutral complex exists for this combo
    ChargeError: No charge-neutral combination found for Ni(II) + [HCOO, HCOO].
    """


# ── validation errors ─────────────────────────────────────────────────────────

class ValidationError(MolbuilderError):
    """
    Raised when a built structure fails post-construction validation.

    Wraps the ValidationResult from molbuilder.core.validation so callers
    can inspect individual issues programmatically.

    Attributes
    ----------
    summary  : str   – human-readable validation report
    n_errors : int   – number of error-severity issues
    """

    def __init__(self, message: str, summary: str = "", n_errors: int = 0):
        super().__init__(message)
        self.summary  = summary
        self.n_errors = n_errors
