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
"""Contains the HOState class which represents a wavefunction in the harmonic oscillator basis"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_array, identity, kron


class HOState:
    """Represent a wavefunction in the harmonic oscillator basis.

    Args:
        modes (int): the number of vibrational modes
        gridpoints (int): the number of gridpoints used to discretize the state
        state: (Union[scipy.sparse.csr_array, Dict[Tuple[int], float]]): a sparse state vector for the full wavefunction or a dictionary containing the interacting modes and their non-zero coefficients


    **Examples**

    Building an :class:`~.pennylane.labs.trotter_error.HOState` from a dictionary

    >>> from pennylane.labs.trotter_error import HOState
    >>> n_modes = 3
    >>> gridpoints = 5
    >>> state_dict = {(1, 2, 3): 1, (0, 3, 2): 1}
    >>> HOState(n_modes, gridpoints, state_dict)
    HOState(modes=3, gridpoints=5, <Compressed Sparse Row sparse array of dtype 'int64'
        with 2 stored elements and shape (125, 1)>
      Coords	Values
      (17, 0)	1
      (38, 0)	1)

    Building an :class:`~.pennylane.labs.trotter_error.HOState` from a ``scipy.sparse.csr_array``

    >>> from scipy.sparse import csr_array
    >>> import numpy as np
    >>> gridpoints = 2
    >>> n_modes = 2
    >>> state_vector = csr_array(np.array([0, 1, 0, 0]))
    >>> HOState(n_modes, gridpoints, state_vector)
    HOState(modes=2, gridpoints=2, <COOrdinate sparse array of dtype 'int64'
        with 1 stored elements and shape (4, 1)>
      Coords	Values
      (1, 0)	1)
    """

    def __init__(self, modes: int, gridpoints: int, state: csr_array):
        if isinstance(state, csr_array):
            if state.shape == (gridpoints**modes,):
                state = state.reshape((gridpoints**modes, 1))

            if state.shape != (gridpoints**modes, 1):
                raise ValueError(
                    f"Dimension mismatch. Expected vector of shape {(gridpoints ** modes, 1)} but got shape {state.shape}."
                )

            self.vector = state
        elif isinstance(state, dict):
            self.vector = self._vector_from_dict(modes, gridpoints, state)
        else:
            raise TypeError(f"state must be of type csr_array or dict, got {type(state)}.")

        self.gridpoints = gridpoints
        self.modes = modes
        self.dim = gridpoints**modes

    @classmethod
    def _vector_from_dict(
        cls, modes: int, gridpoints: int, coeffs: dict[tuple[int], float]
    ) -> csr_array:
        """Construct an :class:`~.pennylane.labs.trotter_error.HOState` from a dictionary.

        Args:
            modes (int): the number of vibrational modes
            gridpoints (int): the number of gridpoints used to discretize the state
            coeffs (Dict[Tuple[int]]): a dictionary representation of the state

        Returns:
            HOState: an :class:`~.pennylane.labs.trotter_error.HOState` representation of the state vector

        **Example**

        >>> from pennylane.labs.trotter_error import HOState
        >>> n_modes = 3
        >>> gridpoints = 5
        >>> state_dict = {(1, 2, 3): 1, (0, 3, 2): 1}
        >>> HOState.from_dict(n_modes, gridpoints, state_dict)
        HOState(modes=3, gridpoints=5, <Compressed Sparse Row sparse array of dtype 'int64'
            with 2 stored elements and shape (125, 1)>
          Coords	Values
          (17, 0)	1
          (38, 0)	1)
        """
        rows, cols, vals = [], [], []
        for index, val in coeffs.items():
            if len(index) != modes:
                raise ValueError(
                    f"Number of modes given was {modes}, but {index} contains {len(index)} modes."
                )

            row = sum(i * (gridpoints**exp) for exp, i in enumerate(reversed(index)))
            rows.append(row)
            vals.append(val)

        rows = np.array(rows)
        cols = np.zeros_like(rows)
        vals = np.array(vals)
        return csr_array((vals, (rows, cols)), shape=(gridpoints**modes, 1))

    @classmethod
    def zero_state(cls, modes: int, gridpoints: int) -> HOState:
        """Construct an :class:`~.pennylane.labs.trotter_error.HOState` whose vector is zero.

        Args:
            modes (int): the number of vibrational modes
            gridpoints(int): the number of gridpoints used to discretize the state

        Returns:
            HOState: an :class:`~.pennylane.labs.trotter_error.HOState` representing the zero state

        **Example**

        >>> from pennylane.labs.trotter_error import HOState
        >>> HOState.zero_state(5, 10)
        HOState(modes=5, gridpoints=10, <Compressed Sparse Row sparse array of dtype 'float64'
            with 0 stored elements and shape (100000, 1)>)
        """
        return cls(modes, gridpoints, csr_array((gridpoints**modes, 1)))

    def __add__(self, other: HOState) -> HOState:
        if not isinstance(other, HOState):
            raise TypeError(f"Can only add HOState with another HOState, got {type(other)}.")

        return HOState(self.modes, self.gridpoints, self.vector + other.vector)

    def __mul__(self, scalar: float) -> HOState:
        return HOState(self.modes, self.gridpoints, scalar * self.vector)

    __rmul__ = __mul__

    def dot(self, other: HOState) -> float:
        """Return the dot product of two :class:`~.pennylane.labs.trotter_error.HOState` objects.

        Args:
            other (HOState): the state to take the dot product with

        Returns:
            float: the dot product of the two states

        **Example**

        >>> from pennylane.labs.trotter_error import HOState
        >>> n_modes = 3
        >>> gridpoints = 5
        >>> state_dict = {(1, 2, 3): 1, (0, 3, 2): 1}
        >>> state1 = HOState(n_modes, gridpoints, state_dict)
        >>> state1.dot(state1)
        2
        """
        if self.dim != other.dim:
            raise ValueError(
                f"Dimension mismatch. Attempting to dot product vectors of dimension {self.dim} and {other.dim}."
            )

        return ((self.vector.T).dot(other.vector))[0, 0]

    def __repr__(self):
        return f"HOState(modes={self.modes}, gridpoints={self.gridpoints}, {self.vector})"


class VibronicHO:
    """Represent the tensor product of harmonic oscillator states.

    Args:
        states (int): the number of electronic states
        modes (int): the number of vibrational modes
        gridpoints (int): the number of gridpoints used to discretize the state
        ho_states (Sequence[HOState]): a sequence of :class:`~.pennylane.labs.trotter_error.HOState` objects representing the harmonic oscillator states

    **Example**

    >>> from pennylane.labs.trotter_error import HOState, VibronicHO
    >>> n_modes = 3
    >>> n_states = 2
    >>> gridpoints = 5
    >>> state_dict = {(1, 2, 3): 1, (0, 3, 2): 1}
    >>> state = HOState(n_modes, gridpoints, state_dict)
    >>> VibronicHO(n_states, n_modes, gridpoints, [state, state])
    VibronicHO([HOState(modes=3, gridpoints=5, <Compressed Sparse Row sparse array of dtype 'int64'
        with 2 stored elements and shape (125, 1)>
      Coords	Values
      (17, 0)	1
      (38, 0)	1), HOState(modes=3, gridpoints=5, <Compressed Sparse Row sparse array of dtype 'int64'
        with 2 stored elements and shape (125, 1)>
      Coords	Values
      (17, 0)	1
      (38, 0)	1)])
    """

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

    def __repr__(self):
        return f"VibronicHO({self.ho_states})"

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
        """Construct a :class:`~.pennylane.labs.trotter_error.VibronicHO` representing the zero state.

        Args:
            states (int): the number of electronic states
            modes (int): the number of vibrational modes
            gridpoints(int): the number of gridpoints used to discretize the state

        Returns:
            VibronicHO: a :class:`~.pennylane.labs.trotter_error.VibronicHO` representing the zero state

        **Example**

        >>> from pennylane.labs.trotter_error import VibronicHO
        >>> VibronicHO.zero_state(2, 3, 5)
        VibronicHO([HOState(modes=3, gridpoints=5, <Compressed Sparse Row sparse array of dtype 'float64'
            with 0 stored elements and shape (125, 1)>), HOState(modes=3, gridpoints=5, <Compressed Sparse Row sparse array of dtype 'float64'
            with 0 stored elements and shape (125, 1)>)])
        """
        return cls(
            states=states,
            modes=modes,
            gridpoints=gridpoints,
            ho_states=[HOState.zero_state(modes, gridpoints)] * states,
        )

    def dot(self, other: VibronicHO):
        """Return the dot product of two :class:`~.pennylane.labs.trotter_error.VibronicHO` objects.

        Args:
            other (VibronicHO): the state to take the dot product with

        Returns:
            float: the dot product of the two states

        **Example**

        >>> from pennylane.labs.trotter_error import HOState, VibronicHO
        >>> n_modes = 3
        >>> n_states = 2
        >>> gridpoints = 5
        >>> state_dict = {(1, 2, 3): 1, (0, 3, 2): 1}
        >>> state = HOState(n_modes, gridpoints, state_dict)
        >>> vo_state = VibronicHO(n_states, n_modes, gridpoints, [state, state])
        >>> vo_state.dot(vo_state)
        4
        """

        return np.real(sum(x.dot(y) for x, y in zip(self.ho_states, other.ho_states)))


def _tensor_with_identity(op: csr_array, gridpoints: int, n_modes: int, mode: int) -> csr_array:
    """Return the tensor product of ``op`` with the ``gridpoints**mode`` dimensional identity matrix on the left, and the ``gridpoints ** (n_modes - mode - 1)`` dimensional identity matrix on the right."""
    if mode == 0:
        return kron(op, identity(gridpoints ** (n_modes - 1)))

    if mode == n_modes - 1:
        return kron(identity(gridpoints ** (n_modes - 1)), op)

    id_left = identity(gridpoints**mode)
    id_right = identity(gridpoints ** (n_modes - mode - 1))

    return kron(id_left, kron(op, id_right))
