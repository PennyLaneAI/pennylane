"""Contains the HOState class which represents a wavefunction in the Harmonic Oscillator basis"""
from __future__ import annotations

from typing import Dict, Tuple

from scipy.sparse import coo_matrix, coo_array


class HOState:
    """Representation of a wavefunction in the Harmonic Oscillator basis"""

    def __init__(self, modes: int, gridpoints: int, coeffs: Dict[Tuple[int], float]):
        self.dim = gridpoints ** modes

        self.vector = coo_array()

    def apply_momentum(self, mode: int) -> HOState:
        """Apply momentum operator on specified mode"""

    def apply_position(self, mode: int) -> HOState:
        """Apply position operator on specified mode"""

    def apply_creation(self, mode: int) -> HOState:
        """Apply creation operator on specified mode"""

    def apply_annihilation(self, mode: int) -> HOState:
        """Apply annihilation operator on specified mode"""

    def __add__(self, other: HOState) -> HOState:
        pass

    def __mul__(self, scalar: float) -> HOState:
        pass

    def to_dict(self) -> Dict[Tuple[int], float]:
        """Return the dictionary representation"""

def _validate_indices(modes, indices) -> None:
    for index in indices:
        if len(index) != modes:
            raise ValueError(f"Index {index} does not operate on {modes} modes.")

        for i in index:
            if not isinstance(i, int):
                raise TypeError(f"Index {index} contains non-integral values.")
            if i < 0:
                raise ValueError(f"Index {index} contains negative values.")
