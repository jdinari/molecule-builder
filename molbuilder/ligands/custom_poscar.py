"""
custom_poscar.py
================
Support for loading custom ligands from POSCAR files.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np


class CustomLigandPOSCAR:
    """
    Represents a custom ligand loaded from a POSCAR file.
    """
    
    def __init__(self, poscar_path: Union[str, Path], 
                 donor_atom_indices: List[int],
                 charge: int = 0,
                 name: Optional[str] = None):
        """
        Load a custom ligand from a POSCAR file.
        
        Args:
            poscar_path: Path to the POSCAR file
            donor_atom_indices: List of atomic indices that serve as donor atoms
            charge: Formal charge of the ligand
            name: Optional name (defaults to filename without extension)
        """
        self.poscar_path = Path(poscar_path)
        self.donor_atom_indices = donor_atom_indices
        self.charge = charge
        self.name = name or self.poscar_path.stem
        
        # Will be populated by parse_poscar()
        self.atoms = None
        self.positions = None
        self.symbols = None
        self.cell = None
        self.pbc = None
        
        self._parse_poscar()
    
    def _parse_poscar(self):
        """
        Parse POSCAR file and store atomic structure.
        Uses a minimal parser to avoid external dependencies beyond rdkit/ase.
        """
        try:
            from ase.io import read
            atoms_obj = read(str(self.poscar_path))
            self.atoms = atoms_obj
            self.positions = atoms_obj.get_positions()
            self.symbols = atoms_obj.get_chemical_symbols()
            self.cell = atoms_obj.get_cell()
            self.pbc = atoms_obj.get_pbc()
        except ImportError:
            # Fallback: minimal POSCAR parser without ASE
            self._parse_poscar_minimal()
    
    def _parse_poscar_minimal(self):
        """
        Minimal POSCAR parser (no external dependencies).
        Parses standard VASP POSCAR format.
        """
        lines = self.poscar_path.read_text().strip().split('\n')
        
        # POSCAR format:
        # Line 1: Comment
        # Line 2: Scaling factor
        # Lines 3-5: Lattice vectors
        # Line 6: Element symbols
        # Line 7: Counts
        # Line 8+: Atomic positions
        
        if len(lines) < 8:
            raise ValueError(f"Invalid POSCAR format in {self.poscar_path}")
        
        scale = float(lines[1].strip())
        
        # Lattice vectors
        cell = []
        for i in range(2, 5):
            cell.append([float(x) for x in lines[i].split()])
        self.cell = np.array(cell) * scale
        self.pbc = np.array([True, True, True])
        
        # Element symbols and counts
        symbols_line = lines[5].split()
        counts_line = [int(x) for x in lines[6].split()]
        
        self.symbols = []
        for sym, count in zip(symbols_line, counts_line):
            self.symbols.extend([sym] * count)
        
        # Atomic positions
        positions = []
        coord_type = lines[7].strip()[0].upper()
        is_cartesian = coord_type == 'C'
        
        for i in range(8, 8 + len(self.symbols)):
            if i < len(lines):
                coords = [float(x) for x in lines[i].split()[:3]]
                if is_cartesian:
                    coords = np.array(coords) * scale
                else:  # Direct/fractional coordinates
                    coords = np.dot(np.array(coords), self.cell)
                positions.append(coords)
        
        self.positions = np.array(positions)
    
    @property
    def donor_atoms(self):
        """Get symbols of donor atoms."""
        return [self.symbols[i] for i in self.donor_atom_indices]
    
    @property
    def denticity(self):
        """Get the denticity (number of donor atoms)."""
        return len(self.donor_atom_indices)
    
    def to_dict(self):
        """
        Convert to dictionary representation for compatibility with ligand library.
        """
        return {
            "name": self.name,
            "custom": True,
            "poscar_path": str(self.poscar_path),
            "donor_atom_indices": self.donor_atom_indices,
            "donor_atoms": self.donor_atoms,
            "denticity": self.denticity,
            "charge": self.charge,
            "atoms_obj": self.atoms,
            "positions": self.positions,
            "symbols": self.symbols,
        }
    
    def __repr__(self):
        return (f"CustomLigandPOSCAR(name={self.name!r}, "
                f"denticity={self.denticity}, charge={self.charge})")


def load_custom_ligand(poscar_path: Union[str, Path],
                       donor_atom_indices: List[int],
                       charge: int = 0,
                       name: Optional[str] = None) -> CustomLigandPOSCAR:
    """
    Convenience function to load a custom ligand from POSCAR.
    
    Args:
        poscar_path: Path to POSCAR file
        donor_atom_indices: Atom indices of donor atoms (0-indexed)
        charge: Formal charge
        name: Optional ligand name
    
    Returns:
        CustomLigandPOSCAR instance
    
    Example:
        ligand = load_custom_ligand(
            "my_ligand.POSCAR",
            donor_atom_indices=[0, 2],  # Atoms 0 and 2 donate
            charge=0,
            name="myligand"
        )
    """
    return CustomLigandPOSCAR(
        poscar_path=poscar_path,
        donor_atom_indices=donor_atom_indices,
        charge=charge,
        name=name,
    )
