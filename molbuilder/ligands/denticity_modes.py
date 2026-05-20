"""
denticity_modes.py
==================
Support for variable denticity binding modes.
Allows ligands to bind in monodentate, bidentate, tridentate, etc. forms.
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np


@dataclass
class DenticitySite:
    """
    Represents a single binding site (donor atom) in a ligand.
    """
    atom_index: int  # Index in the ligand's atom list
    atom_symbol: str  # Element symbol (C, N, O, S, P, etc.)
    geometry_vector: Optional[np.ndarray] = None  # Preferred orientation from metal center
    is_bridging: bool = False  # Can this site act as a bridge in polynuclear complexes?


@dataclass
class DenticityMode:
    """
    Defines how a ligand can bind to a metal center.
    Example: bpy can bind in monodentate (1 N) or bidentate (2 N) modes.
    """
    name: str  # e.g., "monodentate", "bidentate-N,N"
    donor_sites: List[DenticitySite]  # List of atoms involved in binding
    bite_angle: Optional[float] = None  # Chelating angle (only for multi-dentate)
    charge: int = 0  # Charge contribution from this binding mode
    is_bridging: bool = False  # Can bridge between multiple metals?

    @property
    def denticity(self):
        """Return the denticity (number of binding sites)."""
        return len(self.donor_sites)


class VariableDenticitySMARTS:
    """
    Utility for defining ligands with variable denticity via SMARTS patterns.
    """
    
    # Define standard donor atom patterns
    DONOR_PATTERNS = {
        # Single O donors (monodentate or bidentate)
        "O_mono": "[O;X2,X1]"  # Oxygen with 1-2 connections
        # Two O donors (bidentate chelation like acac, ox)
        "O2_chelate": "[O;X2].[O;X2]",
        # Single N donor
        "N_mono": "[N;X3,X2]",
        # Two N donors (like bpy)
        "N2_chelate": "[N;X3].[N;X3]",
        # Sulfur donor (monodentate)
        "S_mono": "[S;X2]",
        # Phosphorus donor
        "P_mono": "[P;X3,X4]",
    }


def create_variable_denticity_ligand(name: str, smiles: str, 
                                     possible_modes: List[DenticityMode],
                                     charge: int = 0):
    """
    Create a ligand that can bind in multiple denticity modes.
    
    Args:
        name: Ligand name (e.g., 'bpy')
        smiles: SMILES string
        possible_modes: List of DenticityMode objects representing binding possibilities
        charge: Formal charge of the free ligand
    
    Returns:
        Dictionary with ligand properties
    """
    return {
        "name": name,
        "smiles": smiles,
        "possible_modes": possible_modes,
        "base_charge": charge,
        "variable_denticity": True,
    }
