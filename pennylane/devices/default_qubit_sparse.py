# Copyright 2018-2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module contains a sparse implementation of the :class:`~.DefaultQubit`
reference plugin.
"""
from pennylane.operation import DiagonalOperation
from pennylane.devices import DefaultQubit
import numpy as np

try:
    import sparse

except ImportError as e:  # pragma: no cover
    raise ImportError("default.qubit.sparse device requires installing sparse") from e


class DefaultQubitSparse(DefaultQubit):
    """Simulator plugin based on ``"default.qubit"``, written using Sparse.

    **Short name:** ``default.qubit.sparse``

    This device provides a pure-state qubit simulator where states, operations, and observables
    are represented as sparse tensors.

    To use this device, you will need to install sparse:

    .. code-block:: console

        pip install sparse

    **Example**


    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means
            that the device returns analytical results.
            If ``shots > 0`` is used, this is not supported right now.
    """

    name = "Default qubit (Sparse) PennyLane plugin"
    short_name = "default.qubit.sparse"

    C_DTYPE = np.complex128
    R_DTYPE = np.float64
    _asarray = staticmethod(sparse.COO.from_numpy)
    _dot = staticmethod(sparse.dot)
    _reshape = staticmethod(sparse.COO.reshape)
    _flatten = staticmethod(sparse.COO.flatten)
    _cast = staticmethod(sparse.COO.from_numpy)
    _transpose = staticmethod(sparse.COO.transpose)
    _tensordot = staticmethod(sparse.tensordot)
    _conj = staticmethod(sparse.COO.conj)
    _imag = staticmethod(sparse.COO.imag)
    _roll = staticmethod(sparse.roll)
    _stack = staticmethod(sparse.stack)
    _real = staticmethod(sparse.COO.real)
    # _einsum won't work and not supported in sparse, need to override methods that use it
    # _gather may be an issue
    # _abs should work with just numpy
    # _reduce_sum only used for marginal_probs, which should be fine
    # _outer only used for default.mixed
    # _diag not used as far as I can tell

    def __init__(self, wires, *, shots=None):
        super().__init__(wires, shots=shots, cache=0)

        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def _get_unitary_matrix(self, unitary):  # pylint: disable=no-self-use
        """Return the matrix representing a unitary operation.

        Args:
            unitary (~.Operation): a PennyLane unitary operation

        Returns:
            array[complex]: Returns a 2D matrix representation of
            the unitary in the computational basis, or, in the case of a diagonal unitary,
            a 1D array representing the matrix diagonal.
        """
        if isinstance(unitary, DiagonalOperation):
            return _cast(unitary.eigvals)

        return _cast(unitary.matrix)

    def _create_basis_state(self, index):
        """Return a computational basis state over all wires.

        Args:
            index (int): integer representing the computational basis state

        Returns:
            array[complex]: complex array of shape ``[2]*self.num_wires``
            representing the statevector of the basis state
        """
        state = np.zeros(2 ** self.num_wires, dtype=np.complex128)
        state[index] = 1
        state = self._asarray(state)
        return self._reshape(state, [2] * self.num_wires)
