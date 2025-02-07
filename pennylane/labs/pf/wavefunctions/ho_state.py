"""Contains the HOState class which represents a wavefunction in the Harmonic Oscillator basis"""

from typing import Dict, Tuple

class HOState:
    """Representation of a wavefunction in the Harmonic Oscillator basis"""

    def __init__(self, modes: int, coeffs: Dict[Tuple[int], float]):
        _validate_indices(modes, coeffs.keys())

        self.modes = modes

    def to_dict(self) -> Dict[Tuple[int], float]:
        """Returns a dictionary representation of the wavefunction"""

    def annihilation(self, mode) -> None:
        """Apply annihilation operator on specified mode"""

    def creation(self, mode) -> None:
        """Apply creation operator on specified mode"""

def _validate_indices(modes, indices) -> None:
    for index in indices:
        if len(index) != modes:
            raise ValueError(f"Index {index} does not operate on {modes} modes.")

        for i in index:
            if not isinstance(i, int):
                raise TypeError(f"Index {index} contains non-integral values.")
            if i < 0:
                raise ValueError(f"Index {index} contains negative values.")
