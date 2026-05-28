"""
relaxation.py
=============
Geometry relaxation and thermochemistry for molbuilder structures.

Two backends are supported and can be used interchangeably:

    backend="xtb"   GFN2-xTB via tblite (fast, excellent for Ni coordination
                    chemistry, explicit d-electron parametrisation)

    backend="mace"  MACE-MH-1 universal MLIP (fast on GPU, good for larger
                    clusters and intermolecular geometry; trained on
                    OMAT/OMOL/OC20/MATPES with PBE/R2SCAN/wB97M-V references)

Public API
----------
    relax(mol, ...)              → RelaxResult  (geometry only)
    compute_energy(mol, ...)     → RelaxResult  (single-point, no geometry change)
    compute_gibbs(mol, ...)      → ThermResult  (freq + thermo at one T/P)
    thermochemistry(mol, ...)    → ThermResult  (relax + freq + thermo)

ThermResult extends RelaxResult and adds:
    .zpe_eV           zero-point energy
    .enthalpy_eV      H(T) = E + ZPE + H_thermal
    .entropy_eV_per_K S(T, P)  (gas-phase, harmonic/rigid-rotor/PIAB)
    .gibbs_eV         G(T, P) = H - T·S  at the construction T, P
    .gibbs_at(T, P)   recompute G at any other T, P without re-running

All energies are in eV for consistency with VASP outputs.

Gas-phase thermochemistry
-------------------------
The harmonic oscillator / rigid-rotor / particle-in-a-box model is used:
    H(T) = E_elec + ZPE + integral_0^T Cv dT
    S(T,P) = S_trans(T,P) + S_rot(T) + S_vib(T)

The pressure P enters only through the translational partition function
(S_trans ∝ -R·ln(P)), so it matters when reactions change the number of
molecules (e.g. ligand dissociation Ni-L → Ni + L releases one molecule).
Standard conditions: T = 298.15 K, P = 101325 Pa (1 atm).

ΔE and ΔG for reactions
-----------------------
Compute the relevant quantity for each species, then subtract:

    results = {name: thermochemistry(mol, ...) for name, mol in species.items()}
    dE = results["product"].energy_eV - results["reactant"].energy_eV
    dG = results["product"].gibbs_eV  - results["reactant"].gibbs_eV

    # Re-evaluate ΔG at a different temperature without re-running:
    dG_350K = (results["product"].gibbs_at(T=350)
               - results["reactant"].gibbs_at(T=350))
"""

from __future__ import annotations

import os
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from molbuilder.core.molecule import Atom, Molecule

# ── physical constants ────────────────────────────────────────────────────────
_eV_per_Hartree = 27.211386245988
_eV_per_kcal    = 0.043364104  # kcal/mol → eV
_kB_eV          = 8.617333262e-5   # Boltzmann in eV/K
_kB_J           = 1.380649e-23     # Boltzmann in J/K
_hbar_eV_s      = 6.582119569e-16  # ħ in eV·s
_u_kg           = 1.66053906660e-27  # atomic mass unit in kg
_NA             = 6.02214076e23
_R_eV_per_K     = _kB_eV * _NA      # gas constant in eV/K/mol (= 8.617e-5 * 6.022e23)


# ── result dataclasses ────────────────────────────────────────────────────────

@dataclass
class RelaxResult:
    """
    Result of a geometry relaxation or single-point energy calculation.

    Attributes
    ----------
    mol          : Relaxed (or input) Molecule with updated coordinates.
    energy_eV    : Electronic energy in eV.
    converged    : True if the geometry optimiser converged within *steps*.
    steps        : Number of optimiser steps taken.
    backend      : "xtb" or "mace".
    model        : Model identifier (e.g. "GFN2-xTB", "mh-1").
    """
    mol:        Molecule
    energy_eV:  float
    converged:  bool
    steps:      int
    backend:    str
    model:      str

    def __repr__(self) -> str:
        conv = "converged" if self.converged else "NOT converged"
        return (f"RelaxResult({self.mol.formula}, "
                f"E={self.energy_eV:.4f} eV, {conv}, "
                f"backend={self.backend}/{self.model})")


@dataclass
class ThermResult(RelaxResult):
    """
    Result of a thermochemistry calculation (relaxation + frequency analysis).

    All thermal quantities are in eV (per molecule, not per mole).

    Attributes
    ----------
    T_K          : Temperature used for the primary G calculation (K).
    P_Pa         : Pressure used for the primary G calculation (Pa).
    zpe_eV       : Zero-point vibrational energy.
    enthalpy_eV  : H(T) = E_elec + ZPE + H_thermal(T).
    entropy_eV_K : S(T, P) in eV/K (= S in eV when multiplied by T).
    gibbs_eV     : G(T, P) = H(T) - T·S(T, P).
    frequencies  : Vibrational frequencies in cm⁻¹ (imaginary → negative).
    _vib_energies: Internal list of real vibrational energies in eV (for
                   recomputing G at different T/P via gibbs_at()).
    """
    T_K:           float            = 298.15
    P_Pa:          float            = 101325.0
    zpe_eV:        float            = 0.0
    enthalpy_eV:   float            = 0.0
    entropy_eV_K:  float            = 0.0
    gibbs_eV:      float            = 0.0
    frequencies:   List[float]      = field(default_factory=list)
    _vib_energies: list             = field(default_factory=list)  # complex energies from ASE Vibrations

    def gibbs_at(self, T: float, P: float = 101325.0) -> float:
        """
        Recompute the Gibbs free energy at a different temperature and/or
        pressure **without re-running the frequency calculation**.

        Uses the same harmonic oscillator / rigid-rotor / PIAB model as the
        original calculation but evaluated at (T, P).
        Passes the stored complex vibrational energies to ASE, which handles
        imaginary-mode filtering internally.

        Parameters
        ----------
        T : Temperature in K.
        P : Pressure in Pa.  Defaults to 1 atm (101325 Pa).

        Returns
        -------
        G(T, P) in eV.
        """
        if not self._vib_energies:
            warnings.warn(
                "gibbs_at() called but no vibrational data stored. "
                "Returning E_elec + ZPE as a rough estimate.",
                stacklevel=2,
            )
            return self.energy_eV + self.zpe_eV

        return _harmonic_gibbs(
            e_elec   = self.energy_eV,
            vib_eV   = self._vib_energies,
            mol      = self.mol,
            T        = T,
            P        = P,
        )

    def __repr__(self) -> str:
        return (f"ThermResult({self.mol.formula}, "
                f"E={self.energy_eV:.4f} eV, "
                f"G({self.T_K:.0f}K)={self.gibbs_eV:.4f} eV, "
                f"backend={self.backend}/{self.model})")


# ── ASE conversion helpers ────────────────────────────────────────────────────

def _mol_to_ase(mol: Molecule):
    """Convert a molbuilder Molecule to an ASE Atoms object."""
    from ase import Atoms
    symbols   = [a.symbol for a in mol.atoms]
    positions = np.array([a.position for a in mol.atoms])
    # Large vacuum cell so periodic-image interactions are negligible
    cell_size = max(
        max(positions[:, i].max() - positions[:, i].min() for i in range(3))
        + 25.0,
        30.0,
    )
    atoms = Atoms(symbols=symbols, positions=positions,
                  cell=[cell_size] * 3, pbc=False)
    return atoms


def _ase_to_mol(atoms, original_mol: Molecule) -> Molecule:
    """
    Copy relaxed positions from an ASE Atoms object back into a new Molecule,
    preserving all metadata from the original.
    """
    new_atoms = []
    for orig, pos in zip(original_mol.atoms, atoms.get_positions()):
        new_atoms.append(Atom(
            symbol   = orig.symbol,
            position = pos.copy(),
            label    = orig.label,
            charge   = orig.charge,
        ))
    d = original_mol.to_dict()
    d["atoms"] = [a.to_dict() for a in new_atoms]
    return Molecule.from_dict(d)


# ── thermochemistry helper ────────────────────────────────────────────────────

def _harmonic_gibbs(
    e_elec:  float,
    vib_eV:  list,   # List[complex] from ASE Vibrations, or List[float] for compat
    mol:     Molecule,
    T:       float,
    P:       float,
) -> float:
    """
    Compute G(T, P) using ASE's HarmonicThermo model (gas phase).

    Uses the harmonic oscillator approximation for vibrations,
    rigid-rotor for rotations, and particle-in-a-box for translations.
    All three contributions to entropy are included.

    Parameters
    ----------
    e_elec : Electronic energy in eV.
    vib_eV : Real (positive) vibrational energies in eV.
             Imaginary frequencies must already be filtered out by the caller.
    mol    : Molecule (used for mass, number of atoms for thermo type).
    T      : Temperature in K.
    P      : Pressure in Pa.
    """
    from ase.thermochemistry import HarmonicThermo, IdealGasThermo
    from ase import Atoms

    atoms   = _mol_to_ase(mol)
    n_atoms = len(atoms)

    if n_atoms == 1:
        thermo = HarmonicThermo(vib_energies=vib_eV, potentialenergy=e_elec,
                                ignore_imag_modes=True)
        H = thermo.get_enthalpy(T, verbose=False)
        S = thermo.get_entropy(T, P, verbose=False)
    else:
        pos = atoms.get_positions()
        if n_atoms == 2:
            linear = True
        else:
            v0 = pos[1] - pos[0]; v0 /= np.linalg.norm(v0)
            linear = all(
                abs(abs(np.dot((pos[i] - pos[0]) / max(np.linalg.norm(pos[i] - pos[0]), 1e-9), v0)) - 1.0) < 0.01
                for i in range(2, n_atoms)
            )

        spin = (mol.spin_multiplicity - 1) / 2.0

        thermo = IdealGasThermo(
            vib_energies      = vib_eV,
            potentialenergy   = e_elec,
            atoms             = atoms,
            geometry          = "linear" if linear else "nonlinear",
            symmetrynumber    = 1,
            spin              = spin,
            ignore_imag_modes = True,
        )
        H = thermo.get_enthalpy(T, verbose=False)
        S = thermo.get_entropy(T, P, verbose=False)

    return H - T * S


def _run_vibrations(atoms, calc, mol: Molecule,
                    workdir: Optional[Path] = None) -> List[float]:
    """
    Compute vibrational frequencies via numerical differentiation of forces.
    Returns real vibrational energies in eV (imaginary modes excluded).
    Writes scratch files to *workdir* (temp dir if None).
    """
    from ase.vibrations import Vibrations

    cleanup = workdir is None
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="molbuilder_vib_"))

    atoms.calc = calc
    vib = Vibrations(atoms, name=str(workdir / "vib"))
    vib.run()
    energies_eV = vib.get_energies()   # complex array; imaginary → imag part

    if cleanup:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)

    # Return the full complex array.  ASE's IdealGasThermo (3.22+) handles
    # imaginary and near-zero modes internally via _clean_vib_energies.
    # We return complex values so the caller can pass them directly to ASE.
    return list(energies_eV)


# ── backend: xTB ─────────────────────────────────────────────────────────────

def _xtb_calculator(mol: Molecule, method: str = "GFN2-xTB"):
    """Return a tblite ASE calculator configured for this molecule."""
    from tblite.ase import TBLite
    return TBLite(
        method  = method,
        charge  = mol.charge,
        uhf     = mol.spin_multiplicity - 1,   # number of unpaired electrons
        verbosity = 0,
    )


def _relax_xtb(mol: Molecule, method: str, fmax: float,
               steps: int, constrain_bonds: bool = False) -> RelaxResult:
    from ase.optimize import BFGS

    atoms = _mol_to_ase(mol)
    calc  = _xtb_calculator(mol, method)
    atoms.calc = calc

    if constrain_bonds:
        _add_ml_constraints(atoms, mol)

    opt = BFGS(atoms, logfile=os.devnull)
    opt.run(fmax=fmax, steps=steps)

    energy_eV = float(atoms.get_potential_energy())
    relaxed   = _ase_to_mol(atoms, mol)
    return RelaxResult(
        mol       = relaxed,
        energy_eV = energy_eV,
        converged = bool(opt.converged()),
        steps     = int(opt.nsteps),
        backend   = "xtb",
        model     = method,
    )


def _singlepoint_xtb(mol: Molecule, method: str) -> RelaxResult:
    atoms = _mol_to_ase(mol)
    calc  = _xtb_calculator(mol, method)
    atoms.calc = calc
    energy_eV = float(atoms.get_potential_energy())
    return RelaxResult(
        mol       = mol,
        energy_eV = energy_eV,
        converged = True,
        steps     = 0,
        backend   = "xtb",
        model     = method,
    )


def _thermo_xtb(mol: Molecule, method: str, fmax: float,
                steps: int, T: float, P: float,
                relax_first: bool,
                constrain_bonds: bool = False) -> ThermResult:
    # Step 1: geometry
    if relax_first:
        relax_res = _relax_xtb(mol, method, fmax, steps,
                               constrain_bonds=constrain_bonds)
        mol_opt   = relax_res.mol
        e_eV      = relax_res.energy_eV
        conv      = relax_res.converged
        n_steps   = relax_res.steps
    else:
        sp        = _singlepoint_xtb(mol, method)
        mol_opt   = mol
        e_eV      = sp.energy_eV
        conv      = True
        n_steps   = 0

    # Step 2: frequencies via numerical differentiation
    atoms = _mol_to_ase(mol_opt)
    calc  = _xtb_calculator(mol_opt, method)
    vib_eV = _run_vibrations(atoms, calc, mol_opt)

    # Step 3: thermochemistry
    # ZPE: sum only real, positive-frequency modes
    zpe_eV = sum(0.5 * float(e.real) for e in vib_eV
                 if float(getattr(e, "real", e)) > 1e-6)
    H   = _harmonic_gibbs.__wrapped__  if hasattr(_harmonic_gibbs, '__wrapped__') else None

    # Use ASE's IdealGasThermo for H and S separately
    from ase.thermochemistry import IdealGasThermo, HarmonicThermo
    from ase import Atoms

    atoms_thermo = _mol_to_ase(mol_opt)
    n = len(atoms_thermo)
    if n == 1:
        thermo = HarmonicThermo(vib_energies=vib_eV, potentialenergy=e_eV, ignore_imag_modes=True)
    else:
        spin   = (mol_opt.spin_multiplicity - 1) / 2.0
        pos    = atoms_thermo.get_positions()
        linear = (n == 2) or all(
            abs(abs(np.dot((pos[i]-pos[0])/max(np.linalg.norm(pos[i]-pos[0]),1e-9),
                           (pos[1]-pos[0])/max(np.linalg.norm(pos[1]-pos[0]),1e-9)))-1.0) < 0.01
            for i in range(2, n)
        )
        thermo = IdealGasThermo(
            vib_energies      = vib_eV,
            potentialenergy   = e_eV,
            atoms             = atoms_thermo,
            geometry          = "linear" if linear else "nonlinear",
            symmetrynumber    = 1,
            spin              = spin,
            ignore_imag_modes = True,
        )

    H_eV   = thermo.get_enthalpy(T, verbose=False)
    S_eV_K = thermo.get_entropy(T, P, verbose=False)   # eV/K
    G_eV   = H_eV - T * S_eV_K

    # Frequencies in cm⁻¹ for the result object
    freqs_cm1 = [e / (_hbar_eV_s * 2 * np.pi * 2.998e10) for e in vib_eV]

    return ThermResult(
        mol           = mol_opt,
        energy_eV     = e_eV,
        converged     = conv,
        steps         = n_steps,
        backend       = "xtb",
        model         = method,
        T_K           = T,
        P_Pa          = P,
        zpe_eV        = zpe_eV,
        enthalpy_eV   = H_eV,
        entropy_eV_K  = S_eV_K,
        gibbs_eV      = G_eV,
        frequencies   = freqs_cm1,
        _vib_energies = vib_eV,
    )


# ── backend: MACE ─────────────────────────────────────────────────────────────

def _mace_calculator(mol: Molecule, model: str, device: str,
                     dtype: str = "float64"):
    """Return a MACE ASE calculator."""
    from mace.calculators import mace_mp
    calc = mace_mp(
        model         = model,
        device        = device,
        default_dtype = dtype,
        dispersion    = False,   # dispersion less critical for small molecules
    )
    # MACE does not natively accept charge/spin in the ASE calculator interface
    # for mace_mp models; these are encoded in the training data implicitly.
    # For charged systems the user should verify results against xTB/DFT.
    if mol.charge != 0:
        warnings.warn(
            f"MACE backend: molecule has charge={mol.charge}. "
            "mace_mp models do not explicitly account for charge. "
            "Consider using backend='xtb' for charged species.",
            stacklevel=3,
        )
    return calc


def _relax_mace(mol: Molecule, model: str, device: str,
                fmax: float, steps: int,
                constrain_bonds: bool = False) -> RelaxResult:
    from ase.optimize import BFGS

    atoms = _mol_to_ase(mol)
    calc  = _mace_calculator(mol, model, device)
    atoms.calc = calc

    if constrain_bonds:
        _add_ml_constraints(atoms, mol)

    opt = BFGS(atoms, logfile=os.devnull)
    opt.run(fmax=fmax, steps=steps)

    energy_eV = float(atoms.get_potential_energy())
    relaxed   = _ase_to_mol(atoms, mol)
    return RelaxResult(
        mol       = relaxed,
        energy_eV = energy_eV,
        converged = bool(opt.converged()),
        steps     = int(opt.nsteps),
        backend   = "mace",
        model     = model,
    )


def _singlepoint_mace(mol: Molecule, model: str, device: str) -> RelaxResult:
    atoms = _mol_to_ase(mol)
    calc  = _mace_calculator(mol, model, device)
    atoms.calc = calc
    energy_eV = float(atoms.get_potential_energy())
    return RelaxResult(
        mol       = mol,
        energy_eV = energy_eV,
        converged = True,
        steps     = 0,
        backend   = "mace",
        model     = model,
    )


def _thermo_mace(mol: Molecule, model: str, device: str,
                 fmax: float, steps: int,
                 T: float, P: float, relax_first: bool,
                 constrain_bonds: bool = False) -> ThermResult:
    # Step 1: geometry
    if relax_first:
        relax_res = _relax_mace(mol, model, device, fmax, steps,
                                constrain_bonds=constrain_bonds)
        mol_opt   = relax_res.mol
        e_eV      = relax_res.energy_eV
        conv      = relax_res.converged
        n_steps   = relax_res.steps
    else:
        sp      = _singlepoint_mace(mol, model, device)
        mol_opt = mol
        e_eV    = sp.energy_eV
        conv    = True
        n_steps = 0

    # Step 2: frequencies (numerical differentiation via ASE)
    atoms = _mol_to_ase(mol_opt)
    calc  = _mace_calculator(mol_opt, model, device)
    vib_eV = _run_vibrations(atoms, calc, mol_opt)

    # Step 3: thermochemistry
    # ZPE: sum only real, positive-frequency modes
    zpe_eV = sum(0.5 * float(e.real) for e in vib_eV
                 if float(getattr(e, "real", e)) > 1e-6)
    from ase.thermochemistry import IdealGasThermo, HarmonicThermo

    atoms_thermo = _mol_to_ase(mol_opt)
    n    = len(atoms_thermo)
    spin = (mol_opt.spin_multiplicity - 1) / 2.0
    pos  = atoms_thermo.get_positions()

    if n == 1:
        thermo = HarmonicThermo(vib_energies=vib_eV, potentialenergy=e_eV, ignore_imag_modes=True)
    else:
        linear = (n == 2) or all(
            abs(abs(np.dot((pos[i]-pos[0])/max(np.linalg.norm(pos[i]-pos[0]),1e-9),
                           (pos[1]-pos[0])/max(np.linalg.norm(pos[1]-pos[0]),1e-9)))-1.0) < 0.01
            for i in range(2, n)
        )
        thermo = IdealGasThermo(
            vib_energies      = vib_eV,
            potentialenergy   = e_eV,
            atoms             = atoms_thermo,
            geometry          = "linear" if linear else "nonlinear",
            symmetrynumber    = 1,
            spin              = spin,
            ignore_imag_modes = True,
        )

    H_eV   = thermo.get_enthalpy(T, verbose=False)
    S_eV_K = thermo.get_entropy(T, P, verbose=False)
    G_eV   = H_eV - T * S_eV_K

    freqs_cm1 = [e / (_hbar_eV_s * 2 * np.pi * 2.998e10) for e in vib_eV]

    return ThermResult(
        mol           = mol_opt,
        energy_eV     = e_eV,
        converged     = conv,
        steps         = n_steps,
        backend       = "mace",
        model         = model,
        T_K           = T,
        P_Pa          = P,
        zpe_eV        = zpe_eV,
        enthalpy_eV   = H_eV,
        entropy_eV_K  = S_eV_K,
        gibbs_eV      = G_eV,
        frequencies   = freqs_cm1,
        _vib_energies = vib_eV,
    )


# ── public API ────────────────────────────────────────────────────────────────

def relax(
    mol: Molecule,
    backend: str  = "xtb",
    model: Optional[str] = None,
    fmax: float   = 0.05,
    steps: int    = 300,
    device: str   = "cpu",
    constrain_bonds: bool = False,
) -> RelaxResult:
    """
    Geometry optimisation.

    Parameters
    ----------
    mol             : Input Molecule (unrelaxed).
    backend         : "xtb" or "mace".
    model           : Override the default model.
    fmax            : Force convergence threshold in eV/Å.
    steps           : Maximum optimiser steps.
    device          : "cpu" or "cuda" (MACE only).
    constrain_bonds : If True, add Hookean spring constraints on all M-L bonds
                      so that ligands cannot dissociate during optimisation.
                      The springs only activate when a bond is stretched
                      >1.35× its initial length; normal geometry changes are
                      unaffected.

                      **Default is False.**  Bond dissociation during xTB/MACE
                      relaxation is physically meaningful — if a ligand
                      departs, xTB is telling you that coordination is
                      genuinely strained.  The recommended workflow is:

                          1. Relax without constraints (default).
                          2. Call check_bonds_intact() on the result.
                          3. Use the bond_status flag in the CSV/Excel to
                             identify strained structures before DFT.
                          4. Set constrain_bonds=True only if you specifically
                             want to preserve the designed coordination motif
                             as a DFT starting point regardless of stability.

    Returns
    -------
    RelaxResult with relaxed coordinates, electronic energy, and convergence info.
    """
    backend = backend.lower()
    if backend == "xtb":
        m = model or "GFN2-xTB"
        return _relax_xtb(mol, m, fmax, steps,
                          constrain_bonds=constrain_bonds)
    elif backend == "mace":
        m = model or "mh-1"
        return _relax_mace(mol, m, device, fmax, steps,
                           constrain_bonds=constrain_bonds)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'xtb' or 'mace'.")


def compute_energy(
    mol: Molecule,
    backend: str  = "xtb",
    model: Optional[str] = None,
    device: str   = "cpu",
) -> RelaxResult:
    """
    Single-point electronic energy at the input geometry (no relaxation).

    Useful for computing ΔE between structures that have already been
    relaxed, or for a quick energy estimate without geometry change.

    Parameters
    ----------
    mol     : Input Molecule.
    backend : "xtb" or "mace".
    model   : Override the default model.
    device  : "cpu" or "cuda" (MACE only).

    Returns
    -------
    RelaxResult with energy_eV set, steps=0, converged=True.
    """
    backend = backend.lower()
    if backend == "xtb":
        return _singlepoint_xtb(mol, model or "GFN2-xTB")
    elif backend == "mace":
        return _singlepoint_mace(mol, model or "mh-1", device)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'xtb' or 'mace'.")


def compute_gibbs(
    mol: Molecule,
    backend: str  = "xtb",
    model: Optional[str] = None,
    T: float      = 298.15,
    P: float      = 101325.0,
    device: str   = "cpu",
) -> ThermResult:
    """
    Thermochemistry at the input geometry (no relaxation, just frequencies).

    Use this when you already have a relaxed structure and only want the
    thermal corrections and Gibbs free energy.

    Parameters
    ----------
    mol     : Input Molecule (should be at or near a stationary point).
    backend : "xtb" or "mace".
    model   : Override the default model.
    T       : Temperature in K.  Default 298.15 K.
    P       : Pressure in Pa.    Default 101325 Pa (1 atm).
    device  : "cpu" or "cuda" (MACE only).

    Returns
    -------
    ThermResult with ZPE, H(T), S(T,P), G(T,P), and .gibbs_at(T,P).
    """
    backend = backend.lower()
    if backend == "xtb":
        return _thermo_xtb(mol, model or "GFN2-xTB",
                           fmax=0.05, steps=0, T=T, P=P, relax_first=False)
    elif backend == "mace":
        return _thermo_mace(mol, model or "mh-1", device,
                            fmax=0.05, steps=0, T=T, P=P, relax_first=False)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'xtb' or 'mace'.")


def thermochemistry(
    mol: Molecule,
    backend: str  = "xtb",
    model: Optional[str] = None,
    T: float      = 298.15,
    P: float      = 101325.0,
    fmax: float   = 0.05,
    steps: int    = 300,
    device: str   = "cpu",
    constrain_bonds: bool = False,
) -> ThermResult:
    """
    Full thermochemistry pipeline: geometry relaxation + frequency analysis.

    This is the recommended function for obtaining ΔE and ΔG for reactions.
    It combines relax() and compute_gibbs() in one call.

    Parameters
    ----------
    mol     : Input (unrelaxed) Molecule.
    backend : "xtb" (recommended for Ni coordination chemistry) or "mace".
    model   : Override the default model.
              xTB default "GFN2-xTB"; MACE default "mh-1" (MACE-MH-1).
    T       : Temperature in K.  Default 298.15 K (25 °C).
    P       : Pressure in Pa.    Default 101325 Pa (1 atm).
    fmax    : Force convergence threshold in eV/Å.
    steps   : Maximum optimiser steps.
    device  : "cpu" or "cuda" (MACE only).

    Returns
    -------
    ThermResult with:
        .energy_eV     — electronic energy of the relaxed structure
        .zpe_eV        — zero-point energy
        .enthalpy_eV   — H(T)
        .entropy_eV_K  — S(T, P) in eV/K
        .gibbs_eV      — G(T, P) = H(T) - T·S(T, P)
        .gibbs_at(T,P) — recompute G at any other (T, P) without re-running
        .frequencies   — vibrational frequencies in cm⁻¹

    Example
    -------
    Compute ΔG for Ni(H₂O)₆ → Ni(HCOO)(H₂O)₅ + H₂O at 350 K:

        from molbuilder import build, trimer
        from molbuilder.relaxation import thermochemistry

        reactant = thermochemistry(Ni_H2O6,  backend="xtb", T=350)
        formate  = thermochemistry(HCOO_mol, backend="xtb", T=350)
        product  = thermochemistry(Ni_HCOO,  backend="xtb", T=350)
        H2O      = thermochemistry(H2O_mol,  backend="xtb", T=350)

        dE = product.energy_eV + H2O.energy_eV - reactant.energy_eV - formate.energy_eV
        dG = product.gibbs_eV  + H2O.gibbs_eV  - reactant.gibbs_eV  - formate.gibbs_eV

        # Re-evaluate at 400 K without re-running:
        dG_400 = (product.gibbs_at(400) + H2O.gibbs_at(400)
                  - reactant.gibbs_at(400) - formate.gibbs_at(400))
    """
    backend = backend.lower()
    if backend == "xtb":
        return _thermo_xtb(mol, model or "GFN2-xTB",
                           fmax=fmax, steps=steps, T=T, P=P,
                           relax_first=True,
                           constrain_bonds=constrain_bonds)
    elif backend == "mace":
        return _thermo_mace(mol, model or "mh-1", device,
                            fmax=fmax, steps=steps, T=T, P=P,
                            relax_first=True,
                            constrain_bonds=constrain_bonds)
    else:
        raise ValueError(f"Unknown backend '{backend}'. Choose 'xtb' or 'mace'.")


# ── bond-integrity checker ────────────────────────────────────────────────────

def check_bonds_intact(
    mol_before: Molecule,
    mol_after:  Molecule,
    threshold_factor: float = 1.4,
) -> dict:
    """
    Check whether any metal-ligand bonds were broken during relaxation.

    A bond is considered broken if the M-L distance after relaxation exceeds
    ``threshold_factor`` × the original distance (default 1.4×, i.e. 40%
    elongation).

    Parameters
    ----------
    mol_before       : Structure before relaxation.
    mol_after        : Structure after relaxation (same atom ordering).
    threshold_factor : Fraction above which a bond is flagged as broken.

    Returns
    -------
    dict with keys:
        "intact"         : bool   – True if no bonds were broken
        "broken_bonds"   : list   – [(atom_i, atom_j, d_before, d_after), ...]
        "max_elongation" : float  – largest fractional elongation observed
    """
    metal = mol_before.metal_symbol
    if not metal:
        return {"intact": True, "broken_bonds": [], "max_elongation": 0.0}

    metal_indices = mol_before.metal_indices
    broken = []
    max_elong = 0.0

    for mi in metal_indices:
        m_pos_b = mol_before.atoms[mi].position
        m_pos_a = mol_after.atoms[mi].position

        for j, (ab, aa) in enumerate(zip(mol_before.atoms, mol_after.atoms)):
            if j == mi or ab.symbol == metal:
                continue
            d_b = float(np.linalg.norm(ab.position - m_pos_b))
            d_a = float(np.linalg.norm(aa.position - m_pos_a))
            if d_b < 2.8:   # only check atoms that were originally bonded
                elong = d_a / d_b
                max_elong = max(max_elong, elong)
                if elong > threshold_factor:
                    broken.append({
                        "metal_idx":    mi,
                        "ligand_idx":   j,
                        "metal_sym":    ab.symbol if j == mi else metal,
                        "ligand_sym":   ab.symbol,
                        "d_before_A":   round(d_b, 4),
                        "d_after_A":    round(d_a, 4),
                        "elongation":   round(elong, 3),
                    })

    return {
        "intact":         len(broken) == 0,
        "broken_bonds":   broken,
        "max_elongation": round(max_elong, 3),
    }


def _add_ml_constraints(atoms, mol: Molecule,
                        force_constant: float = 5.0,
                        max_stretch_factor: float = 1.35):
    """
    Add Hookean spring constraints on all metal-ligand bonds.

    Each M-L bond gets a spring that activates when the bond is stretched
    beyond ``max_stretch_factor`` × its initial length.  This prevents
    dissociation without constraining the bond geometry for normal relaxation.

    Parameters
    ----------
    atoms            : ASE Atoms object (modified in-place).
    mol              : Original Molecule (used to identify metal and bonding).
    force_constant   : Spring constant in eV/Å².
    max_stretch_factor : Fraction above equilibrium at which spring kicks in.
    """
    from ase.constraints import Hookean

    metal = mol.metal_symbol
    if not metal:
        return

    constraints = []
    for mi in mol.metal_indices:
        m_pos = mol.atoms[mi].position
        for j, atom in enumerate(mol.atoms):
            if j == mi or atom.symbol == metal:
                continue
            d0 = float(np.linalg.norm(atom.position - m_pos))
            if d0 < 2.8:    # only constrain atoms that are originally bonded
                rt = d0 * max_stretch_factor
                constraints.append(
                    Hookean(a1=mi, a2=j, rt=rt, k=force_constant)
                )

    if constraints:
        existing = atoms.constraints or []
        atoms.set_constraint(list(existing) + constraints)


# ── compare_backends ──────────────────────────────────────────────────────────

def compare_backends(
    mol: Molecule,
    T: float   = 298.15,
    P: float   = 101325.0,
    fmax: float = 0.05,
    steps: int  = 300,
    xtb_model:  Optional[str] = None,
    mace_model: Optional[str] = None,
    mace_device: str = "cpu",
    compute_thermo: bool = True,
    constrain_bonds: bool = False,
) -> dict:
    """
    Run both xTB and MACE on the same molecule and return a comparison dict.

    Parameters
    ----------
    mol            : Input Molecule.
    T, P           : Temperature (K) and pressure (Pa) for ΔG.
    fmax, steps    : Geometry optimiser settings.
    xtb_model      : Override xTB model (default GFN2-xTB).
    mace_model     : Override MACE model (default mh-1).
    mace_device    : "cpu" or "cuda".
    compute_thermo : If True, run full thermochemistry (relax + freq).
                     If False, only compute single-point energies.
    constrain_bonds: If True, add M-L bond constraints to prevent dissociation.

    Returns
    -------
    dict with keys:
        "xtb"                 : RelaxResult or ThermResult
        "mace"                : RelaxResult or ThermResult (or None if unavailable)
        "delta_E_xtb_eV"      : xTB electronic energy
        "delta_G_xtb_eV"      : xTB G(T, P)  (None if compute_thermo=False)
        "delta_E_mace_eV"     : MACE electronic energy (None if unavailable)
        "delta_G_mace_eV"     : MACE G(T, P)  (None if unavailable or no thermo)
        "bond_check_xtb"      : dict from check_bonds_intact for xTB result
        "bond_check_mace"     : dict from check_bonds_intact for MACE result
        "dE_xtb_vs_mace_eV"   : E_mace - E_xtb  (None if MACE unavailable)
        "T_K"                 : T
        "P_Pa"                : P
    """
    result: dict = {
        "xtb": None, "mace": None,
        "delta_E_xtb_eV": None, "delta_G_xtb_eV": None,
        "delta_E_mace_eV": None, "delta_G_mace_eV": None,
        "bond_check_xtb": None, "bond_check_mace": None,
        "dE_xtb_vs_mace_eV": None,
        "T_K": T, "P_Pa": P,
    }

    # ── xTB ──────────────────────────────────────────────────────────────────
    try:
        if compute_thermo:
            xtb_res = thermochemistry(
                mol, backend="xtb", model=xtb_model, T=T, P=P,
                fmax=fmax, steps=steps,
            )
            result["delta_G_xtb_eV"] = round(xtb_res.gibbs_eV, 6)
        else:
            xtb_res = relax(mol, backend="xtb", model=xtb_model,
                            fmax=fmax, steps=steps,
                            constrain_bonds=constrain_bonds)
        result["xtb"]             = xtb_res
        result["delta_E_xtb_eV"] = round(xtb_res.energy_eV, 6)
        result["bond_check_xtb"] = check_bonds_intact(mol, xtb_res.mol)
    except Exception as exc:
        warnings.warn(f"xTB failed for {mol.formula}: {exc}", stacklevel=2)

    # ── MACE ─────────────────────────────────────────────────────────────────
    try:
        if compute_thermo:
            mace_res = thermochemistry(
                mol, backend="mace", model=mace_model,
                device=mace_device, T=T, P=P, fmax=fmax, steps=steps,
            )
            result["delta_G_mace_eV"] = round(mace_res.gibbs_eV, 6)
        else:
            mace_res = relax(mol, backend="mace", model=mace_model,
                             device=mace_device, fmax=fmax, steps=steps,
                             constrain_bonds=constrain_bonds)
        result["mace"]              = mace_res
        result["delta_E_mace_eV"]  = round(mace_res.energy_eV, 6)
        result["bond_check_mace"]  = check_bonds_intact(mol, mace_res.mol)
    except Exception as exc:
        warnings.warn(f"MACE failed for {mol.formula}: {exc}", stacklevel=2)

    # ── cross-backend comparison ──────────────────────────────────────────────
    if result["delta_E_xtb_eV"] is not None and result["delta_E_mace_eV"] is not None:
        result["dE_xtb_vs_mace_eV"] = round(
            result["delta_E_mace_eV"] - result["delta_E_xtb_eV"], 6
        )

    return result
