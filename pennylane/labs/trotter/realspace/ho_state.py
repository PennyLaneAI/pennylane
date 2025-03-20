# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains the HOState class which represents a wavefunction in the Harmonic Oscillator basis"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

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

    @classmethod
    def zero_state(cls, modes: int, gridpoints: int) -> HOState:
        """Construct an HOState whose vector is zero"""
        return cls(modes, gridpoints, csr_array((gridpoints**modes, 1)))

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
        if not isinstance(other, HOState):
            raise TypeError(f"Can only add HOState with another HOState, got {type(other)}.")

        return HOState.from_scipy(self.modes, self.gridpoints, self.vector + other.vector)

    def __mul__(self, scalar: float) -> HOState:
        return HOState.from_scipy(self.modes, self.gridpoints, scalar * self.vector)

    __rmul__ = __mul__

    def to_dict(self) -> Dict[Tuple[int], float]:
        """Return the dictionary representation"""
        raise NotImplementedError

    def dot(self, other: HOState) -> float:
        """Return the inner product"""
        if self.dim != other.dim:
            raise ValueError(
                f"Dimension mismatch. Attempting to dot product vectors of dimension {self.dim} and {other.dim}."
            )

        return ((self.vector.T).dot(other.vector))[0, 0]


class VibronicHO:
    """Class representing a harmonic oscilator vibronic state"""

    def __init__(self, states: int, modes: int, gridpoints: int, ho_states: Sequence[HOState]):

        if len(ho_states) != states:
            raise ValueError(
                f"Got {len(ho_states)} harmonic oscillator states, but expected {states}."
            )

        for ho_state in ho_states:
            if ho_state.modes != modes:
                raise ValueError(
                    f"Mode mismatch: given {modes} modes, but found an HOState on {ho_state.modes} modes."
                )

            if ho_state.gridpoints != gridpoints:
                raise ValueError(
                    f"Gridpoint mismatch: given {gridpoints} gridpoints, but found an HOState on {ho_state.gridpoints} gridpoints."
                )

        self.states = states
        self.modes = modes
        self.gridpoints = gridpoints
        self.ho_states = ho_states

    def __add__(self, other: VibronicHO) -> VibronicHO:
        if self.states != other.states:
            raise ValueError(
                f"Cannot add VibronicHO on {self.states} states with VibronicHO on {other.states} states."
            )

        if self.modes != other.modes:
            raise ValueError(
                f"Cannot add VibronicHO on {self.modes} modes with VibronicHO on {other.modes} modes."
            )

        if self.gridpoints != other.gridpoints:
            raise ValueError(
                f"Cannot add VibronicHO on {self.gridpoints} gridpoints with VibronicHO on {other.gridpoints} gridpoints."
            )

        return VibronicHO(
            states=self.states,
            modes=self.modes,
            gridpoints=self.gridpoints,
            ho_states=[x + y for x, y in zip(self.ho_states, other.ho_states)],
        )

    def __mul__(self, scalar: float) -> VibronicHO:
        return VibronicHO(
            states=self.states,
            modes=self.modes,
            gridpoints=self.gridpoints,
            ho_states=[scalar * ho for ho in self.ho_states],
        )

    __rmul__ = __mul__

    @classmethod
    def zero_state(cls, states: int, modes: int, gridpoints: int) -> VibronicHO:
        """Return an all zero state"""
        return cls(
            states=states,
            modes=modes,
            gridpoints=gridpoints,
            ho_states=[HOState.zero_state(modes, gridpoints)] * states,
        )

    def dot(self, other: VibronicHO):
        """Return the inner product"""

        return sum(x.dot(y) for x, y in zip(self.ho_states, other.ho_states))


def _tensor_with_identity(op: csr_array, gridpoints: int, n_modes: int, mode: int) -> csr_array:
    if mode == 0:
        return kron(op, identity(gridpoints ** (n_modes - 1)))

    if mode == n_modes - 1:
        return kron(identity(gridpoints ** (n_modes - 1)), op)

    id_left = identity(gridpoints**mode)
    id_right = identity(gridpoints ** (n_modes - mode - 1))

    return kron(id_left, kron(op, id_right))
