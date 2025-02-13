"""Contains the HOState class which represents a wavefunction in the Harmonic Oscillator basis"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.sparse import csr_array, identity, kron


class HOState:
    """Representation of a wavefunction in the Harmonic Oscillator basis"""

    def __init__(self, modes: int, gridpoints: int, vector: csr_array):
        self.gridpoints = gridpoints
        self.modes = modes
        self.dim = gridpoints**modes
        self.vector = vector

    @classmethod
    def from_dict(cls, modes: int, gridpoints: int, coeffs: Dict[Tuple[int], float]) -> HOState:
        """Construct an HOState from a dictionary"""
        rows, cols, vals = [], [], []
        for index, val in coeffs.items():
            if len(index) != modes:
                raise ValueError(
                    f"Number of modes given was {modes}, but {index} contains {len(index)} modes."
                )

            row = sum(i * (gridpoints**exp) for exp, i in enumerate(reversed(index)))
            rows.append(row)
            cols.append(0)
            vals.append(val)

        rows = np.array(rows)
        cols = np.array(cols)
        vals = np.array(vals)
        vector = csr_array((vals, (rows, cols)), shape=(gridpoints**modes, 1))

        return cls(modes, gridpoints, vector)

    @classmethod
    def from_scipy(cls, modes: int, gridpoints: int, vector: csr_array) -> HOState:
        """Construct an HOState from a scipy array"""
        if vector.shape != (gridpoints**modes, 1):
            raise ValueError(
                f"Dimension mismatch. Expected vector of shape {(gridpoints ** modes, 1)} but got shape {vector.shape}."
            )

        return cls(modes, gridpoints, vector)

    def apply_momentum(self, mode: int) -> HOState:
        """Apply momentum operator on specified mode"""
        rows = np.array(list(range(1, self.gridpoints)) + list(range(0, self.gridpoints - 1)))
        cols = np.array(list(range(0, self.gridpoints - 1)) + list(range(1, self.gridpoints)))
        vals = np.array([np.sqrt(i) for i in range(1, self.gridpoints)] * 2)

        momentum = csr_array((vals, (rows, cols)), shape=(self.gridpoints, self.gridpoints))
        op = _tensor_with_identity(momentum, self.gridpoints, self.modes, mode)

        return HOState.from_scipy(self.modes, self.gridpoints, op @ self.vector)

    def apply_position(self, mode: int) -> HOState:
        """Apply position operator on specified mode"""
        rows = np.array(list(range(1, self.gridpoints)) + list(range(0, self.gridpoints - 1)))
        cols = np.array(list(range(0, self.gridpoints - 1)) + list(range(1, self.gridpoints)))
        vals = np.array(
            [np.sqrt(i) for i in range(1, self.gridpoints)]
            + [-np.sqrt(i) for i in range(1, self.gridpoints)]
        )

        position = csr_array((vals, (rows, cols)), shape=(self.gridpoints, self.gridpoints))
        op = _tensor_with_identity(position, self.gridpoints, self.modes, mode)

        return HOState.from_scipy(self.modes, self.gridpoints, op @ self.vector)

    def apply_creation(self, mode: int) -> HOState:
        """Apply creation operator on specified mode"""
        rows = np.array(range(1, self.gridpoints))
        cols = np.array(range(0, self.gridpoints - 1))
        vals = np.array([np.sqrt(i) for i in range(1, self.gridpoints)])

        creation = csr_array((vals, (rows, cols)), shape=(self.gridpoints, self.gridpoints))
        op = _tensor_with_identity(creation, self.gridpoints, self.modes, mode)

        return HOState.from_scipy(self.modes, self.gridpoints, op @ self.vector)

    def apply_annihilation(self, mode: int) -> HOState:
        """Apply annihilation operator on specified mode"""
        rows = np.array(range(0, self.gridpoints - 1))
        cols = np.array(range(1, self.gridpoints))
        vals = np.array([np.sqrt(i) for i in range(1, self.gridpoints)])

        annihilation = csr_array((vals, (rows, cols)), shape=(self.gridpoints, self.gridpoints))
        op = _tensor_with_identity(annihilation, self.gridpoints, self.modes, mode)

        return HOState.from_scipy(self.modes, self.gridpoints, op @ self.vector)

    def apply_operator(self, op: csr_array) -> HOState:
        """Apply an operator to the state"""
        return HOState.from_scipy(self.modes, self.gridpoints, op @ self.vector)

    def __add__(self, other: HOState) -> HOState:
        return HOState.from_scipy(self.modes, self.gridpoints, self.vector + other.vector)

    def __mul__(self, scalar: float) -> HOState:
        return HOState.from_scipy(self.modes, self.gridpoints, scalar * self.vector)

    def to_dict(self) -> Dict[Tuple[int], float]:
        """Return the dictionary representation"""
        raise NotImplementedError

    def dot(self, other: HOState) -> float:
        """Return the inner product"""

        if self.dim != other.dim:
            raise ValueError(
                f"Dimension mismatch. Attempting to dot product vectors of dimension {self.dim} and {other.dim}."
            )

        return self.vector.dot(other.vector)


def _tensor_with_identity(op: csr_array, gridpoints: int, n_modes: int, mode: int) -> csr_array:
    if mode == 0:
        return kron(op, identity(gridpoints ** (n_modes - 1)))

    if mode == n_modes - 1:
        return kron(identity(gridpoints ** (n_modes - 1)), op)

    id_left = identity(gridpoints**mode)
    id_right = identity(gridpoints ** (n_modes - mode - 1))

    return kron(id_left, kron(op, id_right))
