# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
The default.qutrit device is PennyLane's standard qutrit-based device.

It implements the :class:`~pennylane._device.Device` methods as well as some built-in
:mod:`qutrit operations <pennylane.ops.qutrit>`, and provides simple pure state
simulation of qutrit-based quantum computing.
"""
import functools
import numpy as np

import pennylane as qml  # pylint: disable=unused-import
from pennylane import QutritDevice
from pennylane.wires import WireError
from pennylane.devices.default_qubit import _get_slice
from .._version import __version__

# tolerance for numerical errors
tolerance = 1e-10

OMEGA = np.exp(2 * np.pi * 1j / 3)


class DefaultQutrit(QutritDevice):
    """Default qutrit device for PennyLane.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means that the device
            returns analytical results.
    """

    name = "Default qutrit PennyLane plugin"
    short_name = "default.qutrit"
    pennylane_requires = __version__
    version = __version__
    author = "Mudit Pandey, UBC Quantum Software and Algorithms Research Group, and Xanadu"

    # TODO: Update list of operations and observables once more are added
    operations = {
        "Identity",
        "QutritUnitary",
        "TShift",
        "TClock",
    }

    # Identity is supported as an observable for qml.state() to work correctly. However, any
    # measurement types that rely on eigenvalue decomposition will not work with qml.Identity
    observables = {
        "THermitian",
        "GellMannObs",
        "Identity",
    }

    def __init__(
        self,
        wires,
        *,
        r_dtype=np.float64,
        c_dtype=np.complex128,
        shots=None,
        analytic=None,
    ):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self._debugger = None

        # Create the initial state. Internally, we store the
        # state as an array of dimension [3]*wires.
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

        # TODO: Add operations
        self._apply_ops = {
            # All operations that can be applied on the `default.qutrit` device by directly
            # manipulating the internal state array will be included in this dictionary
            "TShift": self._apply_tshift,
            "TClock": self._apply_tclock,
        }

    @functools.lru_cache()
    def map_wires(self, wires):
        # temporarily overwrite this method to bypass
        # wire map that produces Wires objects
        try:
            mapped_wires = [self.wire_map[w] for w in wires]
        except KeyError as e:
            raise WireError(
                f"Did not find some of the wires {wires.labels} on device with wires {self.wires.labels}."
            ) from e

        return mapped_wires

    def define_wire_map(self, wires):
        # temporarily overwrite this method to bypass
        # wire map that produces Wires objects
        consecutive_wires = range(self.num_wires)
        wire_map = zip(wires, consecutive_wires)
        return dict(wire_map)

    def apply(self, operations, rotations=None, **kwargs):  # pylint: disable=arguments-differ
        rotations = rotations or []

        # apply the circuit operations

        # Operations are enumerated so that the order of operations can eventually be used
        # for correctly applying basis state / state vector / snapshot operations which will
        # be added later.
        for i, operation in enumerate(operations):  # pylint: disable=unused-variable
            self._state = self._apply_operation(self._state, operation)

        # store the pre-rotated state
        self._pre_rotated_state = self._state

        # apply the circuit rotations
        for operation in rotations:
            self._state = self._apply_operation(self._state, operation)

    def _apply_operation(self, state, operation):
        """Applies operations to the input state.

        Args:
            state (array[complex]): input state
            operation (~.Operation): operation to apply on the device

        Returns:
            array[complex]: output state
        """
        if operation.base_name == "Identity":
            return state
        wires = operation.wires

        if operation.base_name in self._apply_ops:
            axes = self.wires.indices(wires)
            return self._apply_ops[operation.base_name](state, axes, inverse=operation.inverse)

        matrix = self._asarray(self._get_unitary_matrix(operation), dtype=self.C_DTYPE)

        return self._apply_unitary(state, matrix, wires)

    def _apply_tshift(self, state, axes, inverse=False):
        """Applies a ternary Shift gate by rolling 1 unit along the axis specified in ``axes``.

        Rolling by 1 unit along the axis means that the :math:`|0 \rangle` state with index ``0`` is
        shifted to the :math:`|1 \rangle` state with index ``1``. Likewise, since rolling beyond
        the last index loops back to the first, :math:`|2 \rangle` is transformed to
        :math:`|0 \rangle`.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            inverse (bool): whether to apply the inverse operation

        Returns:
            array[complex]: output state
        """
        shift = -1 if inverse else 1
        return self._roll(state, shift, axes[0])

    def _apply_tclock(self, state, axes, inverse=False):
        """Applies a ternary Clock gate by adding appropriate phases to the 1 and 2 indices
        along the axis specified in ``axes``

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            inverse (bool): whether to apply the inverse operation

        Returns:
            array[complex]: output state
        """
        partial_state = self._apply_phase(state, axes, 1, OMEGA, inverse)
        return self._apply_phase(partial_state, axes, 2, OMEGA**2, inverse)

    def _apply_phase(
        self, state, axes, index, phase, inverse=False
    ):  # pylint: disable=too-many-arguments
        """Applies a phase onto the specified index along the axis specified in ``axes``.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            index (int): target index of axis to apply phase to
            phase (float): phase to apply
            inverse (bool): whether to apply the inverse phase

        Returns:
            array[complex]: output state
        """
        num_wires = len(state.shape)
        slices = [_get_slice(i, axes[0], num_wires) for i in range(3)]

        phase = self._conj(phase) if inverse else phase
        state_slices = [
            self._const_mul(phase if i == index else 1, state[slices[i]]) for i in range(3)
        ]
        return self._stack(state_slices, axis=axes[0])

    def _get_unitary_matrix(self, unitary):  # pylint: disable=no-self-use
        """Return the matrix representing a unitary operation.

        Args:
            unitary (~.Operation): a PennyLane unitary operation

        Returns:
            array[complex]: Returns a 2D matrix representation of
            the unitary in the computational basis.
        """
        return unitary.matrix()

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qutrit",
            supports_inverse_operations=True,
            supports_analytic_computation=True,
            returns_state=True,
        )
        return capabilities

    def _create_basis_state(self, index):
        """Return a computational basis state over all wires.

        Args:
            index (int): integer representing the computational basis state

        Returns:
            array[complex]: complex array of shape ``[3]*self.num_wires``
            representing the statevector of the basis state
        """
        state = np.zeros(3**self.num_wires, dtype=np.complex128)
        state[index] = 1
        state = self._asarray(state, dtype=self.C_DTYPE)
        return self._reshape(state, [3] * self.num_wires)

    @property
    def state(self):
        return self._flatten(self._pre_rotated_state)

    def density_matrix(self, wires):
        """Returns the reduced density matrix of a given set of wires.

        Args:
            wires (Wires): wires of the reduced system.

        Returns:
            array[complex]: complex tensor of shape ``(3 ** len(wires), 3 ** len(wires))``
            representing the reduced density matrix.
        """
        dim = self.num_wires
        state = self._pre_rotated_state

        # Return the full density matrix by using numpy tensor product
        if wires == self.wires:
            density_matrix = self._tensordot(state, self._conj(state), axes=0)
            density_matrix = self._reshape(density_matrix, (3 ** len(wires), 3 ** len(wires)))
            return density_matrix

        complete_system = list(range(0, dim))
        traced_system = [x for x in complete_system if x not in wires.labels]

        # Return the reduced density matrix by using numpy tensor product
        density_matrix = self._tensordot(
            state, self._conj(state), axes=(traced_system, traced_system)
        )
        density_matrix = self._reshape(density_matrix, (3 ** len(wires), 3 ** len(wires)))

        return density_matrix

    def _apply_unitary(self, state, mat, wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        Args:
            state (array[complex]): input state
            mat (array): matrix to multiply
            wires (Wires): target wires

        Returns:
            array[complex]: output state
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        mat = self._cast(self._reshape(mat, [3] * len(device_wires) * 2), dtype=self.C_DTYPE)
        axes = (np.arange(len(device_wires), 2 * len(device_wires)), device_wires)
        tdot = self._tensordot(mat, state, axes=axes)

        # tensordot causes the axes given in `wires` to end up in the first positions
        # of the resulting tensor. This corresponds to a (partial) transpose of
        # the correct output state
        # We'll need to invert this permutation to put the indices in the correct place
        unused_idxs = [idx for idx in range(self.num_wires) if idx not in device_wires]
        perm = list(device_wires) + unused_idxs
        inv_perm = np.argsort(perm)  # argsort gives inverse permutation
        return self._transpose(tdot, inv_perm)

    def reset(self):
        """Reset the device"""
        super().reset()

        # init the state vector to |00..0>
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def analytic_probability(self, wires=None):

        if self._state is None:
            return None

        flat_state = self._flatten(self._state)
        real_state = self._real(flat_state)
        imag_state = self._imag(flat_state)
        prob = self.marginal_prob(real_state**2 + imag_state**2, wires)
        return prob
