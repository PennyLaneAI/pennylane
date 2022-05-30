r"""
The default.qutrit device is Pennylane's standard qutrit-based device.

It implements the :class:`~pennylane._device.Device` methods as well as some built-in
:mod:`qutrit operations <pennylane.ops.qutrit>`, and provides a simple pure state
simulation of qutrit-based quantum circuit architecture
"""
import functools
import numpy as np

import pennylane as qml
from pennylane import QutritDevice, DeviceError
from pennylane.wires import Wires, WireError
from .._version import __version__
from default_qubit import _get_slice

from pennylane.measurements import MeasurementProcess

# tolerance for numerical errors
tolerance = 1e-10


class DefaultQutrit(QutritDevice):
    """Default qutrit device for PennyLane

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
    # TODO: Add author

    # TODO: Add list of operations and observables
    operations = {}

    observables = {}

    def __init__(
        self, wires, *, r_dtype=np.float64, c_dtype=np.complex128, shots=None, analytic=None,
    ):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self._debugger = None

        # Create the initial state. Internally, we store the
        # state as an array of dimension [3]*wires.
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

        # TODO: Add operations
        self._apply_ops = {}

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

    def apply(self, operations, rotations=None, **kwargs):
        rotations = rotations or []

        # apply the circuit operations
        for i, operation in enumerate(operations):

            # if i > 0 and isinstance(operation, (QubitStateVector, BasisState)):
            #     raise DeviceError(
            #         f"Operation {operation.name} cannot be used after other Operations have already been applied "
            #         f"on a {self.short_name} device."
            #     )

            # if isinstance(operation, QubitStateVector):
            #     self._apply_state_vector(operation.parameters[0], operation.wires)
            # elif isinstance(operation, BasisState):
            #     self._apply_basis_state(operation.parameters[0], operation.wires)
            # elif isinstance(operation, Snapshot):
            #     if self._debugger and self._debugger.active:
            #         state_vector = np.array(self._flatten(self._state))
            #         if operation.tag:
            #             self._debugger.snapshots[operation.tag] = state_vector
            #         else:
            #             self._debugger.snapshots[len(self._debugger.snapshots)] = state_vector
            if False:
                # DO NOTHING
                continue
            else:
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

        if len(wires) <= 2:
            # Einsum is faster for small gates
            return self._apply_unitary_einsum(state, matrix, wires)

        return self._apply_unitary(state, matrix, wires)

    def expval(self, observable, shot_range=None, bin_size=None):
        # TODO: Update later
        return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

    def _get_unitary_matrix(self, unitary):
        """Return the matrix representing a unitary operation.

        Args:
            unitary (~.Operation): a PennyLane unitary operation

        Returns:
            array[complex]: Returns a 2D matrix representation of
            the unitary in the computational basis
        """
        return unitary.matrix()

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qutrit",
            supports_reversible_diff=True,
            supports_inverse_operations=True,
            supports_analytic_computation=True,
            returns_state=True
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

    # TODO: Implement function
    def _apply_state_vector(self, state, device_wires):
        pass

    # TODO: Implement function
    def _apply_basis_state(self, state, wires):
        pass

    # TODO: Implement function
    def _apply_unitary(self, state, mat, wires):
        pass

    # TODO: Implement function
    def _apply_unitary_einsum(self, state, mat, wires):
        pass

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

