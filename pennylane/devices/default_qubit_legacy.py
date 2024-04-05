# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This module contains the legacy implementation of default.qubit.

It implements the necessary :class:`~pennylane._device.Device` methods as well as some built-in
:mod:`qubit operations <pennylane.ops.qubit>`, and provides a very simple pure state
simulation of a qubit-based quantum circuit architecture.
"""
import functools
import itertools
from string import ascii_letters as ABC
from typing import List

import numpy as np
from scipy.sparse import csr_matrix

import pennylane as qml
from pennylane import BasisState, DeviceError, QubitDevice, StatePrep, Snapshot
from pennylane.devices.qubit import measure
from pennylane.operation import Operation
from pennylane.ops import Sum
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane.pulse import ParametrizedEvolution
from pennylane.measurements import ExpectationMP
from pennylane.typing import TensorLike
from pennylane.wires import WireError

from .._version import __version__

ABC_ARRAY = np.array(list(ABC))

# tolerance for numerical errors
tolerance = 1e-10
SQRT2INV = 1 / np.sqrt(2)
TPHASE = np.exp(1j * np.pi / 4)


def _get_slice(index, axis, num_axes):
    """Allows slicing along an arbitrary axis of an array or tensor.

    Args:
        index (int): the index to access
        axis (int): the axis to slice into
        num_axes (int): total number of axes

    Returns:
        tuple[slice or int]: a tuple that can be used to slice into an array or tensor

    **Example:**

    Accessing the 2 index along axis 1 of a 3-axis array:

    >>> sl = _get_slice(2, 1, 3)
    >>> sl
    (slice(None, None, None), 2, slice(None, None, None))
    >>> a = np.arange(27).reshape((3, 3, 3))
    >>> a[sl]
    array([[ 6,  7,  8],
           [15, 16, 17],
           [24, 25, 26]])
    """
    idx = [slice(None)] * num_axes
    idx[axis] = index
    return tuple(idx)


# pylint: disable=unused-argument
class DefaultQubitLegacy(QubitDevice):
    """Default qubit device for PennyLane.

    .. warning::

        This is the legacy implementation of DefaultQubit. It has been replaced by
        ``qml.devices.DefaultQubit``, which can be accessed with the familiar constructor,
        ``qml.device("default.qubit")``.

        This change will not alter device behaviour for most workflows, but may have implications for
        plugin developers and users who directly interact with device methods. Please consult
        :class:`pennylane.devices.Device` and the implementation in
        :class:`pennylane.devices.DefaultQubit` for more information on what the new
        interface will look like and be prepared to make updates in a coming release. If you have any
        feedback on these changes, please create an
        `issue <https://github.com/PennyLaneAI/pennylane/issues>`_ or post in our
        `discussion forum <https://discuss.pennylane.ai/>`_.

    Args:
        wires (int, Iterable[Number, str]): Number of subsystems represented by the device,
            or iterable that contains unique labels for the subsystems as numbers (i.e., ``[-1, 0, 2]``)
            or strings (``['ancilla', 'q1', 'q2']``). Default 1 if not specified.
        shots (None, int): How many times the circuit should be evaluated (or sampled) to estimate
            the expectation values. Defaults to ``None`` if not specified, which means that the device
            returns analytical results.
    """

    name = "Default qubit PennyLane plugin (Legacy)"
    short_name = "default.qubit.legacy"
    pennylane_requires = __version__
    version = __version__
    author = "Xanadu Inc."

    operations = {
        "GlobalPhase",
        "Identity",
        "Snapshot",
        "BasisState",
        "StatePrep",
        "QubitStateVector",
        "QubitUnitary",
        "ControlledQubitUnitary",
        "BlockEncode",
        "MultiControlledX",
        "IntegerComparator",
        "DiagonalQubitUnitary",
        "PauliX",
        "X",
        "PauliY",
        "Y",
        "PauliZ",
        "Z",
        "MultiRZ",
        "Hadamard",
        "S",
        "Adjoint(S)",
        "T",
        "Adjoint(T)",
        "SX",
        "Adjoint(SX)",
        "CNOT",
        "SWAP",
        "ISWAP",
        "PSWAP",
        "Adjoint(ISWAP)",
        "SISWAP",
        "Adjoint(SISWAP)",
        "SQISW",
        "CSWAP",
        "Toffoli",
        "CCZ",
        "CY",
        "CZ",
        "CH",
        "PhaseShift",
        "PCPhase",
        "ControlledPhaseShift",
        "CPhaseShift00",
        "CPhaseShift01",
        "CPhaseShift10",
        "CPhase",
        "RX",
        "RY",
        "RZ",
        "Rot",
        "CRX",
        "CRY",
        "CRZ",
        "CRot",
        "IsingXX",
        "IsingYY",
        "IsingZZ",
        "IsingXY",
        "SingleExcitation",
        "SingleExcitationPlus",
        "SingleExcitationMinus",
        "DoubleExcitation",
        "DoubleExcitationPlus",
        "DoubleExcitationMinus",
        "QubitCarry",
        "QubitSum",
        "OrbitalRotation",
        "QFT",
        "ECR",
    }

    observables = {
        "PauliX",
        "X",
        "PauliY",
        "Y",
        "PauliZ",
        "Z",
        "Hadamard",
        "Hermitian",
        "Identity",
        "Projector",
        "SparseHamiltonian",
        "Hamiltonian",
        "LinearCombination",
        "Sum",
        "SProd",
        "Prod",
        "Exp",
        "Evolution",
    }

    def __init__(
        self, wires, *, r_dtype=np.float64, c_dtype=np.complex128, shots=None, analytic=None
    ):
        super().__init__(wires, shots, r_dtype=r_dtype, c_dtype=c_dtype, analytic=analytic)
        self._debugger = None

        # Create the initial state. Internally, we store the
        # state as an array of dimension [2]*wires.
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

        self._apply_ops = {
            "PauliX": self._apply_x,
            "PauliY": self._apply_y,
            "PauliZ": self._apply_z,
            "Hadamard": self._apply_hadamard,
            "S": self._apply_s,
            "T": self._apply_t,
            "SX": self._apply_sx,
            "CNOT": self._apply_cnot,
            "SWAP": self._apply_swap,
            "CZ": self._apply_cz,
            "Toffoli": self._apply_toffoli,
        }

    @property
    def stopping_condition(self):
        def accepts_obj(obj):
            if obj.name == "QFT" and len(obj.wires) >= 6:
                return False
            if obj.name == "GroverOperator" and len(obj.wires) >= 13:
                return False
            if getattr(obj, "has_matrix", False):
                # pow operations dont work with backprop or adjoint without decomposition
                # use class name string so we don't need to use isinstance check
                return not (obj.__class__.__name__[:3] == "Pow" and qml.operation.is_trainable(obj))
            return obj.name in self.observables.union(self.operations)

        return qml.BooleanFn(accepts_obj)

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

    # pylint: disable=arguments-differ
    def _get_batch_size(self, tensor, expected_shape, expected_size):
        """Determine whether a tensor has an additional batch dimension for broadcasting,
        compared to an expected_shape."""
        size = self._size(tensor)
        if self._ndim(tensor) > len(expected_shape) or size > expected_size:
            return size // expected_size

        return None

    # pylint: disable=arguments-differ
    def apply(self, operations, rotations=None, **kwargs):
        rotations = rotations or []

        # apply the circuit operations
        for i, operation in enumerate(operations):
            if i > 0 and isinstance(operation, (StatePrep, BasisState)):
                raise DeviceError(
                    f"Operation {operation.name} cannot be used after other Operations have already been applied "
                    f"on a {self.short_name} device."
                )

            if isinstance(operation, StatePrep):
                self._apply_state_vector(operation.parameters[0], operation.wires)
            elif isinstance(operation, BasisState):
                self._apply_basis_state(operation.parameters[0], operation.wires)
            elif isinstance(operation, Snapshot):
                if self._debugger and self._debugger.active:
                    state_vector = np.array(self._flatten(self._state))
                    if operation.tag:
                        self._debugger.snapshots[operation.tag] = state_vector
                    else:
                        self._debugger.snapshots[len(self._debugger.snapshots)] = state_vector
            elif isinstance(operation, ParametrizedEvolution):
                self._state = self._apply_parametrized_evolution(self._state, operation)
            else:
                self._state = self._apply_operation(self._state, operation)

        # store the pre-rotated state
        self._pre_rotated_state = self._state

        # apply the circuit rotations
        for operation in rotations:
            self._state = self._apply_operation(self._state, operation)

    def _apply_parametrized_evolution(self, state: TensorLike, operation: ParametrizedEvolution):
        """Applies a parametrized evolution to the input state.

        Args:
            state (array[complex]): input state
            operation (ParametrizedEvolution): operation to apply on the state
        """
        raise NotImplementedError(
            f"The device {self.short_name} cannot execute a ParametrizedEvolution operation. "
            "Please use the jax interface."
        )

    def _apply_operation(self, state, operation):
        """Applies operations to the input state.

        Args:
            state (array[complex]): input state
            operation (~.Operation): operation to apply on the device

        Returns:
            array[complex]: output state
        """
        if operation.__class__.__name__ == "Identity":
            return state
        if operation.name == "GlobalPhase":
            return self._apply_global_phase(state, operation)
        wires = operation.wires

        if str(operation.name) in self._apply_ops:  # cast to string because of Tensor
            shift = int(self._ndim(state) > self.num_wires)
            axes = [ax + shift for ax in self.wires.indices(wires)]
            return self._apply_ops[operation.name](state, axes)

        matrix = self._asarray(self._get_unitary_matrix(operation), dtype=self.C_DTYPE)

        if operation in diagonal_in_z_basis:
            return self._apply_diagonal_unitary(state, matrix, wires)
        if len(wires) <= 2:
            # Einsum is faster for small gates
            return self._apply_unitary_einsum(state, matrix, wires)

        return self._apply_unitary(state, matrix, wires)

    def _apply_global_phase(self, state, operation: qml.GlobalPhase):  # pylint: disable=no-self-use
        """Applies a :class:`~.GlobalPhase` operation to the state."""
        return qml.math.exp(-1j * operation.data[0]) * state

    def _apply_x(self, state, axes, **kwargs):
        """Applies a PauliX gate by rolling 1 unit along the axis specified in ``axes``.

        Rolling by 1 unit along the axis means that the :math:`|0 \rangle` state with index ``0`` is
        shifted to the :math:`|1 \rangle` state with index ``1``. Likewise, since rolling beyond
        the last index loops back to the first, :math:`|1 \rangle` is transformed to
        :math:`|0\rangle`.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        return self._roll(state, 1, axes[0])

    def _apply_y(self, state, axes, **kwargs):
        """Applies a PauliY gate by adding a negative sign to the 1 index along the axis specified
        in ``axes``, rolling one unit along the same axis, and multiplying the result by 1j.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        return 1j * self._apply_x(self._apply_z(state, axes), axes)

    def _apply_z(self, state, axes, **kwargs):
        """Applies a PauliZ gate by adding a negative sign to the 1 index along the axis specified
        in ``axes``.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        return self._apply_phase(state, axes, -1)

    def _apply_hadamard(self, state, axes, **kwargs):
        """Apply the Hadamard gate by combining the results of applying the PauliX and PauliZ gates.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        state_x = self._apply_x(state, axes)
        state_z = self._apply_z(state, axes)
        return self._const_mul(SQRT2INV, state_x + state_z)

    def _apply_s(self, state, axes, inverse=False):
        return self._apply_phase(state, axes, 1j, inverse)

    def _apply_t(self, state, axes, inverse=False):
        return self._apply_phase(state, axes, TPHASE, inverse)

    def _apply_sx(self, state, axes, inverse=False):
        """Apply the Square Root X gate.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        return (
            0.5 * ((1 - 1j) * state + (1 + 1j) * self._apply_x(state, axes))
            if inverse
            else 0.5 * ((1 + 1j) * state + (1 - 1j) * self._apply_x(state, axes))
        )

    def _apply_cnot(self, state, axes, **kwargs):
        """Applies a CNOT gate by slicing along the first axis specified in ``axes`` and then
        applying an X transformation along the second axis.

        By slicing along the first axis, we are able to select all of the amplitudes with a
        corresponding :math:`|1\rangle` for the control qubit. This means we then just need to apply
        a :class:`~.PauliX` (NOT) gate to the result.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        ndim = self._ndim(state)
        sl_0 = _get_slice(0, axes[0], ndim)
        sl_1 = _get_slice(1, axes[0], ndim)

        # We will be slicing into the state according to state[sl_1], giving us all of the
        # amplitudes with a |1> for the control qubit. The resulting array has lost an axis
        # relative to state and we need to be careful about the axis we apply the PauliX rotation
        # to. If axes[1] is larger than axes[0], then we need to shift the target axis down by
        # one, otherwise we can leave as-is. For example: a state has [0, 1, 2, 3], control=1,
        # target=3. Then, state[sl_1] has 3 axes and target=3 now corresponds to the second axis.
        if axes[1] > axes[0]:
            target_axes = [axes[1] - 1]
        else:
            target_axes = [axes[1]]

        state_x = self._apply_x(state[sl_1], axes=target_axes)
        return self._stack([state[sl_0], state_x], axis=axes[0])

    def _apply_toffoli(self, state, axes, **kwargs):
        """Applies a Toffoli gate by slicing along the axis of the greater control qubit, slicing
        each of the resulting sub-arrays along the axis of the smaller control qubit, and then applying
        an X transformation along the axis of the target qubit of the fourth sub-sub-array.

        By performing two consecutive slices in this way, we are able to select all of the amplitudes with
        a corresponding :math:`|11\rangle` for the two control qubits. This means we then just need to apply
        a :class:`~.PauliX` (NOT) gate to the result.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        cntrl_max = np.argmax(axes[:2])
        cntrl_min = cntrl_max ^ 1
        ndim = self._ndim(state)
        sl_a0 = _get_slice(0, axes[cntrl_max], ndim)
        sl_a1 = _get_slice(1, axes[cntrl_max], ndim)
        sl_b0 = _get_slice(0, axes[cntrl_min], ndim - 1)
        sl_b1 = _get_slice(1, axes[cntrl_min], ndim - 1)

        # If both controls are smaller than the target, shift the target axis down by two. If one
        # control is greater and one control is smaller than the target, shift the target axis
        # down by one. If both controls are greater than the target, leave the target axis as-is.
        if axes[cntrl_min] > axes[2]:
            target_axes = [axes[2]]
        elif axes[cntrl_max] > axes[2]:
            target_axes = [axes[2] - 1]
        else:
            target_axes = [axes[2] - 2]

        # state[sl_a1][sl_b1] gives us all of the amplitudes with a |11> for the two control qubits.
        state_x = self._apply_x(state[sl_a1][sl_b1], axes=target_axes)
        state_stacked_a1 = self._stack([state[sl_a1][sl_b0], state_x], axis=axes[cntrl_min])
        return self._stack([state[sl_a0], state_stacked_a1], axis=axes[cntrl_max])

    def _apply_swap(self, state, axes, **kwargs):
        """Applies a SWAP gate by performing a partial transposition along the specified axes.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        all_axes = list(range(len(state.shape)))
        all_axes[axes[0]] = axes[1]
        all_axes[axes[1]] = axes[0]
        return self._transpose(state, all_axes)

    def _apply_cz(self, state, axes, **kwargs):
        """Applies a CZ gate by slicing along the first axis specified in ``axes`` and then
        applying a Z transformation along the second axis.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation

        Returns:
            array[complex]: output state
        """
        ndim = self._ndim(state)
        sl_0 = _get_slice(0, axes[0], ndim)
        sl_1 = _get_slice(1, axes[0], ndim)

        if axes[1] > axes[0]:
            target_axes = [axes[1] - 1]
        else:
            target_axes = [axes[1]]

        state_z = self._apply_z(state[sl_1], axes=target_axes)
        return self._stack([state[sl_0], state_z], axis=axes[0])

    def _apply_phase(self, state, axes, parameters, inverse=False):
        """Applies a phase onto the 1 index along the axis specified in ``axes``.

        Args:
            state (array[complex]): input state
            axes (List[int]): target axes to apply transformation
            parameters (float): phase to apply
            inverse (bool): whether to apply the inverse phase

        Returns:
            array[complex]: output state
        """
        ndim = self._ndim(state)
        sl_0 = _get_slice(0, axes[0], ndim)
        sl_1 = _get_slice(1, axes[0], ndim)

        phase = self._conj(parameters) if inverse else parameters
        return self._stack([state[sl_0], self._const_mul(phase, state[sl_1])], axis=axes[0])

    def expval(self, observable, shot_range=None, bin_size=None):
        """Returns the expectation value of a Hamiltonian observable. When the observable is a
        ``Hamiltonian`` or ``SparseHamiltonian`` object, the expectation value is computed directly
        from the sparse matrix representation, which leads to faster execution.

        Args:
            observable (~.Observable): a PennyLane observable
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Returns:
            float: returns the expectation value of the observable

        .. warning::

            This function does not support broadcasting if ``observable`` is a
            :class:``~.Hamiltonian`` and the device interface or the interface of the
            Hamiltonian is not NumPy or Autograd

        """
        is_state_batched = self._ndim(self.state) == 2
        # intercept Sums
        if isinstance(observable, Sum) and not self.shots:
            return measure(
                ExpectationMP(observable.map_wires(self.wire_map)),
                self._pre_rotated_state,
                is_state_batched,
            )

        # intercept other Hamiltonians
        # TODO: Ideally, this logic should not live in the Device, but be moved
        # to a component that can be re-used by devices as needed.
        if observable.name not in ("Hamiltonian", "SparseHamiltonian", "LinearCombination"):
            return super().expval(observable, shot_range=shot_range, bin_size=bin_size)

        assert self.shots is None, f"{observable.name} must be used with shots=None"

        self.map_wires(observable.wires)
        backprop_mode = (
            not isinstance(self.state, np.ndarray)
            or any(not isinstance(d, (float, np.ndarray)) for d in observable.data)
        ) and observable.name in ["Hamiltonian", "LinearCombination"]

        if backprop_mode:
            # TODO[dwierichs]: This branch is not adapted to broadcasting yet
            if is_state_batched:
                raise NotImplementedError(
                    "Expectation values of Hamiltonians for interface!=None are "
                    "not supported together with parameter broadcasting yet"
                )
            # We must compute the expectation value assuming that the Hamiltonian
            # coefficients *and* the quantum states are tensor objects.

            # Compute  <psi| H |psi> via sum_i coeff_i * <psi| PauliWord |psi> using a sparse
            # representation of the Pauliword
            res = qml.math.cast(qml.math.convert_like(0.0, observable.data), dtype=complex)
            interface = qml.math.get_interface(self.state)

            # Note: it is important that we use the Hamiltonian's data and not the coeffs
            # attribute. This is because the .data attribute may be 'unwrapped' as required by
            # the interfaces, whereas the .coeff attribute will always be the same input dtype
            # that the user provided.
            for op, coeff in zip(observable.ops, observable.data):
                # extract a scipy.sparse.coo_matrix representation of this Pauli word
                coo = qml.operation.Tensor(op).sparse_matrix(wire_order=self.wires, format="coo")
                Hmat = qml.math.cast(qml.math.convert_like(coo.data, self.state), self.C_DTYPE)

                product = (
                    self._gather(self._conj(self.state), coo.row)
                    * Hmat
                    * self._gather(self.state, coo.col)
                )
                c = qml.math.convert_like(coeff, product)

                if interface == "tensorflow":
                    c = qml.math.cast(c, "complex128")

                res = qml.math.convert_like(res, product) + qml.math.sum(c * product)

        else:
            # Coefficients and the state are not trainable, we can be more
            # efficient in how we compute the Hamiltonian sparse matrix.
            Hmat = observable.sparse_matrix(wire_order=self.wires)

            state = qml.math.toarray(self.state)
            if is_state_batched:
                res = qml.math.array(
                    [
                        csr_matrix.dot(
                            csr_matrix(self._conj(_state)),
                            csr_matrix.dot(Hmat, csr_matrix(_state[..., None])),
                        ).toarray()[0]
                        for _state in state
                    ]
                )
            else:
                res = csr_matrix.dot(
                    csr_matrix(self._conj(state)),
                    csr_matrix.dot(Hmat, csr_matrix(state[..., None])),
                ).toarray()[0]

        if observable.name in ["Hamiltonian", "LinearCombination"]:
            res = qml.math.squeeze(res)

        return self._real(res)

    def _get_unitary_matrix(self, unitary):  # pylint: disable=no-self-use
        """Return the matrix representing a unitary operation.

        Args:
            unitary (~.Operation): a PennyLane unitary operation

        Returns:
            array[complex]: Returns a 2D matrix representation of
            the unitary in the computational basis, or, in the case of a diagonal unitary,
            a 1D array representing the matrix diagonal.
        """
        if unitary in diagonal_in_z_basis:
            return unitary.eigvals()

        return unitary.matrix()

    @classmethod
    def capabilities(cls):
        capabilities = super().capabilities().copy()
        capabilities.update(
            model="qubit",
            supports_inverse_operations=True,
            supports_analytic_computation=True,
            supports_broadcasting=True,
            returns_state=True,
            passthru_devices={
                "tf": "default.qubit.tf",
                "torch": "default.qubit.torch",
                "autograd": "default.qubit.autograd",
                "jax": "default.qubit.jax",
            },
        )
        return capabilities

    def _create_basis_state(self, index):
        """Return a computational basis state over all wires.

        Args:
            index (int): integer representing the computational basis state

        Returns:
            array[complex]: complex array of shape ``[2]*self.num_wires``
            representing the statevector of the basis state

        Note: This function does not support broadcasted inputs yet.
        """
        state = np.zeros(2**self.num_wires, dtype=np.complex128)
        state[index] = 1
        state = self._asarray(state, dtype=self.C_DTYPE)
        return self._reshape(state, [2] * self.num_wires)

    @property
    def state(self):
        dim = 2**self.num_wires
        batch_size = self._get_batch_size(self._pre_rotated_state, (2,) * self.num_wires, dim)
        # Do not flatten the state completely but leave the broadcasting dimension if there is one
        shape = (batch_size, dim) if batch_size is not None else (dim,)
        return self._reshape(self._pre_rotated_state, shape)

    def _apply_state_vector(self, state, device_wires):
        """Initialize the internal state vector in a specified state.

        Args:
            state (array[complex]): normalized input state of length ``2**len(wires)``
                or broadcasted state of shape ``(batch_size, 2**len(wires))``
            device_wires (Wires): wires that get initialized in the state
        """

        # translate to wire labels used by device
        device_wires = self.map_wires(device_wires)
        dim = 2 ** len(device_wires)

        state = self._asarray(state, dtype=self.C_DTYPE)
        batch_size = self._get_batch_size(state, (dim,), dim)
        output_shape = [2] * self.num_wires
        if batch_size is not None:
            output_shape.insert(0, batch_size)

        if len(device_wires) == self.num_wires and sorted(device_wires) == device_wires:
            # Initialize the entire device state with the input state
            self._state = self._reshape(state, output_shape)
            return

        # generate basis states on subset of qubits via the cartesian product
        basis_states = np.array(list(itertools.product([0, 1], repeat=len(device_wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(device_wires), self.num_wires), dtype=int)
        unravelled_indices[:, device_wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)

        if batch_size is not None:
            state = self._scatter(
                (slice(None), ravelled_indices), state, [batch_size, 2**self.num_wires]
            )
        else:
            state = self._scatter(ravelled_indices, state, [2**self.num_wires])
        state = self._reshape(state, output_shape)
        self._state = self._asarray(state, dtype=self.C_DTYPE)

    def _apply_basis_state(self, state, wires):
        """Initialize the state vector in a specified computational basis state.

        Args:
            state (array[int]): computational basis state of shape ``(wires,)``
                consisting of 0s and 1s.
            wires (Wires): wires that the provided computational state should be initialized on

        Note: This function does not support broadcasted inputs yet.
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        # length of basis state parameter
        n_basis_state = len(state)

        if not set(state.tolist()).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")

        if n_basis_state != len(device_wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # get computational basis state number
        basis_states = 2 ** (self.num_wires - 1 - np.array(device_wires))
        basis_states = qml.math.convert_like(basis_states, state)
        num = int(qml.math.dot(state, basis_states))

        self._state = self._create_basis_state(num)

    def _apply_unitary(self, state, mat, wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        Args:
            state (array[complex]): input state
            mat (array): matrix to multiply
            wires (Wires): target wires

        Returns:
            array[complex]: output state

        Note: This function does not support simultaneously broadcasted states and matrices yet.
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        dim = 2 ** len(device_wires)
        mat_batch_size = self._get_batch_size(mat, (dim, dim), dim**2)
        state_batch_size = self._get_batch_size(state, (2,) * self.num_wires, 2**self.num_wires)

        shape = [2] * (len(device_wires) * 2)
        state_axes = device_wires
        # If the matrix is broadcasted, it is reshaped to have leading axis of size mat_batch_size
        if mat_batch_size:
            shape.insert(0, mat_batch_size)
            if state_batch_size:
                raise NotImplementedError(
                    "Applying a broadcasted unitary to an already broadcasted state via "
                    "_apply_unitary is not supported. Broadcasting sizes are "
                    f"({mat_batch_size}, {state_batch_size})."
                )
        # If the state is broadcasted, the affected state axes need to be shifted by 1.
        if state_batch_size:
            state_axes = [ax + 1 for ax in state_axes]
        mat = self._cast(self._reshape(mat, shape), dtype=self.C_DTYPE)
        axes = (np.arange(-len(device_wires), 0), state_axes)
        tdot = self._tensordot(mat, state, axes=axes)

        # tensordot causes the axes given in `wires` to end up in the first positions
        # of the resulting tensor. This corresponds to a (partial) transpose of
        # the correct output state
        # We'll need to invert this permutation to put the indices in the correct place
        unused_idxs = [idx for idx in range(self.num_wires) if idx not in device_wires]
        perm = list(device_wires) + unused_idxs
        # If the matrix is broadcasted, all but the first dimension are shifted by 1
        if mat_batch_size:
            perm = [idx + 1 for idx in perm]
            perm.insert(0, 0)
        if state_batch_size:
            # As the state broadcasting dimension always is the first in the state, it always
            # ends up in position `len(device_wires)` after the tensordot. The -1 causes it
            # being permuted to the leading dimension after transposition
            perm.insert(len(device_wires), -1)

        inv_perm = np.argsort(perm)  # argsort gives inverse permutation
        return self._transpose(tdot, inv_perm)

    def _apply_unitary_einsum(self, state, mat, wires):
        r"""Apply multiplication of a matrix to subsystems of the quantum state.

        This function uses einsum instead of tensordot. This approach is only
        faster for single- and two-qubit gates.

        Args:
            state (array[complex]): input state
            mat (array): matrix to multiply
            wires (Wires): target wires

        Returns:
            array[complex]: output state
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)

        dim = 2 ** len(device_wires)
        batch_size = self._get_batch_size(mat, (dim, dim), dim**2)

        # If the matrix is broadcasted, it is reshaped to have leading axis of size mat_batch_size
        shape = [2] * (len(device_wires) * 2)
        if batch_size is not None:
            shape.insert(0, batch_size)
        mat = self._cast(self._reshape(mat, shape), dtype=self.C_DTYPE)

        # Tensor indices of the quantum state
        state_indices = ABC[: self.num_wires]

        # Indices of the quantum state affected by this operation
        affected_indices = "".join(ABC_ARRAY[list(device_wires)].tolist())

        # All affected indices will be summed over, so we need the same number of new indices
        new_indices = ABC[self.num_wires : self.num_wires + len(device_wires)]

        # The new indices of the state are given by the old ones with the affected indices
        # replaced by the new_indices
        new_state_indices = functools.reduce(
            lambda old_string, idx_pair: old_string.replace(idx_pair[0], idx_pair[1]),
            zip(affected_indices, new_indices),
            state_indices,
        )

        # We now put together the indices in the notation numpy's einsum requires
        # This notation allows for the state, the matrix, or both to be broadcasted
        einsum_indices = (
            f"...{new_indices}{affected_indices},...{state_indices}->...{new_state_indices}"
        )

        return self._einsum(einsum_indices, mat, state)

    def _apply_diagonal_unitary(self, state, phases, wires):
        r"""Apply multiplication of a phase vector to subsystems of the quantum state.

        This represents the multiplication with diagonal gates in a more efficient manner.

        Args:
            state (array[complex]): input state
            phases (array): vector to multiply
            wires (Wires): target wires

        Returns:
            array[complex]: output state
        """
        # translate to wire labels used by device
        device_wires = self.map_wires(wires)
        dim = 2 ** len(device_wires)
        batch_size = self._get_batch_size(phases, (dim,), dim)

        # reshape vectors
        shape = [2] * len(device_wires)
        if batch_size is not None:
            shape.insert(0, batch_size)
        phases = self._cast(self._reshape(phases, shape), dtype=self.C_DTYPE)

        state_indices = ABC[: self.num_wires]
        affected_indices = "".join(ABC_ARRAY[list(device_wires)].tolist())

        einsum_indices = f"...{affected_indices},...{state_indices}->...{state_indices}"
        return self._einsum(einsum_indices, phases, state)

    def reset(self):
        """Reset the device"""
        super().reset()

        # init the state vector to |00..0>
        self._state = self._create_basis_state(0)
        self._pre_rotated_state = self._state

    def analytic_probability(self, wires=None):
        if self._state is None:
            return None

        dim = 2**self.num_wires
        batch_size = self._get_batch_size(self._state, [2] * self.num_wires, dim)
        flat_state = self._reshape(
            self._state, (batch_size, dim) if batch_size is not None else (dim,)
        )
        real_state = self._real(flat_state)
        imag_state = self._imag(flat_state)
        return self.marginal_prob(real_state**2 + imag_state**2, wires)

    def classical_shadow(self, obs, circuit):
        """
        Returns the measured bits and recipes in the classical shadow protocol.

        The protocol is described in detail in the `classical shadows paper <https://arxiv.org/abs/2002.08953>`_.
        This measurement process returns the randomized Pauli measurements (the ``recipes``)
        that are performed for each qubit and snapshot as an integer:

        - 0 for Pauli X,
        - 1 for Pauli Y, and
        - 2 for Pauli Z.

        It also returns the measurement results (the ``bits``); 0 if the 1 eigenvalue
        is sampled, and 1 if the -1 eigenvalue is sampled.

        The device shots are used to specify the number of snapshots. If ``T`` is the number
        of shots and ``n`` is the number of qubits, then both the measured bits and the
        Pauli measurements have shape ``(T, n)``.

        This implementation leverages vectorization and offers a significant speed-up over
        the generic implementation.

        .. Note::

            This method internally calls ``np.einsum`` which supports at most 52 indices,
            thus the classical shadow measurement for this device supports at most 52
            qubits.

        .. seealso:: :func:`~pennylane.classical_shadow`

        Args:
            obs (~.pennylane.measurements.ClassicalShadowMP): The classical shadow measurement process
            circuit (~.tape.QuantumTape): The quantum tape that is being executed

        Returns:
            tensor_like[int]: A tensor with shape ``(2, T, n)``, where the first row represents
            the measured bits and the second represents the recipes used.
        """
        wires = obs.wires
        seed = obs.seed

        n_qubits = len(wires)
        n_snapshots = self.shots
        device_qubits = len(self.wires)
        mapped_wires = np.array(self.map_wires(wires))

        # seed the random measurement generation so that recipes
        # are the same for different executions with the same seed
        rng = np.random.RandomState(seed)
        recipes = rng.randint(0, 3, size=(n_snapshots, n_qubits))

        obs_list = self._stack(
            [
                qml.X.compute_matrix(),
                qml.Y.compute_matrix(),
                qml.Z.compute_matrix(),
            ]
        )
        uni_list = self._stack(
            [
                qml.Hadamard.compute_matrix(),
                qml.Hadamard.compute_matrix() @ qml.RZ.compute_matrix(-np.pi / 2),
                qml.Identity.compute_matrix(),
            ]
        )
        obs = obs_list[recipes]
        uni = uni_list[recipes]

        # There's a significant speedup if we use the following iterative
        # process to perform the randomized Pauli measurements:
        #   1. Randomly generate Pauli observables for all snapshots for
        #      a single qubit (e.g. the first qubit).
        #   2. Compute the expectation of each Pauli observable on the first
        #      qubit by tracing out all other qubits.
        #   3. Sample the first qubit based on each Pauli expectation.
        #   4. For all snapshots, determine the collapsed state of the remaining
        #      qubits based on the sample result.
        #   4. Repeat iteratively until no qubits are remaining.
        #
        # Observe that after the first iteration, the second qubit will become the
        # "first" qubit in the process. The advantage to this approach as opposed to
        # simulataneously computing the Pauli expectations for each qubit is that
        # the partial traces are computed over iteratively smaller subsystems, leading
        # to a significant speed-up.

        # transpose the state so that the measured wires appear first
        unmeasured_wires = [i for i in range(len(self.wires)) if i not in mapped_wires]
        transposed_state = np.transpose(self._state, axes=mapped_wires.tolist() + unmeasured_wires)

        outcomes = np.zeros((n_snapshots, n_qubits))
        stacked_state = self._stack([transposed_state for _ in range(n_snapshots)])

        for i in range(n_qubits):
            # trace out every qubit except the first
            first_qubit_state = self._einsum(
                f"{ABC[device_qubits - i + 1]}{ABC[:device_qubits - i]},{ABC[device_qubits - i + 1]}{ABC[device_qubits - i]}{ABC[1:device_qubits - i]}"
                f"->{ABC[device_qubits - i + 1]}a{ABC[device_qubits - i]}",
                stacked_state,
                self._conj(stacked_state),
            )

            # sample the observables on the first qubit
            probs = (self._einsum("abc,acb->a", first_qubit_state, obs[:, i]) + 1) / 2
            samples = np.random.uniform(0, 1, size=probs.shape) > probs
            outcomes[:, i] = samples

            # collapse the state of the remaining qubits; the next qubit in line
            # becomes the first qubit for the next iteration
            rotated_state = self._einsum("ab...,acb->ac...", stacked_state, uni[:, i])
            stacked_state = rotated_state[np.arange(n_snapshots), self._cast(samples, np.int8)]

            # re-normalize the collapsed state
            norms = np.sqrt(
                np.sum(
                    np.abs(stacked_state) ** 2, tuple(range(1, device_qubits - i)), keepdims=True
                )
            )
            stacked_state /= norms

        return self._cast(self._stack([outcomes, recipes]), dtype=np.int8)

    def _get_diagonalizing_gates(self, circuit: qml.tape.QuantumTape) -> List[Operation]:
        meas_filtered = [
            m
            for m in circuit.measurements
            if m.obs is None
            or not isinstance(m.obs, (qml.ops.Hamiltonian, qml.ops.LinearCombination))
        ]
        return super()._get_diagonalizing_gates(qml.tape.QuantumScript(measurements=meas_filtered))
