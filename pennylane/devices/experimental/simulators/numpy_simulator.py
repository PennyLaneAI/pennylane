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
"""module docstring"""


from string import ascii_letters as ABC

import numpy as np

import pennylane as qml
from pennylane.tape import QuantumScript


def _get_slice(index, axis, num_axes):
    idx = [slice(None)] * num_axes
    idx[axis] = index
    return tuple(idx)


class PlainNumpySimulator:
    """

    Current Restrictions:
    * No batching

    * No support for state preparation yet
    * No sampling yet
    * state based measurements only

    Preprocessing restrictions:
    * Quantum Script wires must be adjacent integers starting from zero
    * All operations must have matrices

    """

    def __init__(self):
        pass

    @classmethod
    def execute(cls, qs: QuantumScript, dtype=np.complex128):
        """
        Execute a single quantum script.

        """
        num_indices = len(qs.wires)
        state = cls.create_zeroes_state(num_indices, dtype=dtype)
        for op in qs.operations:  # assume no state prep here
            state = cls.apply_operation(state, op)

        measurements = tuple(cls.measure_state(state, m) for m in qs.measurements)
        return measurements[0] if len(measurements) == 1 else measurements

    @staticmethod
    def create_zeroes_state(num_indices: int, dtype=np.complex128) -> np.ndarray:
        """Create a zeroes state with ``num_indices`` wires and of type ``dtype``."""
        state = np.zeros(2**num_indices, dtype=dtype)
        state[0] = 1
        state.shape = [2] * num_indices
        return state

    @classmethod
    def apply_operation(cls, state: np.ndarray, operation: qml.operation.Operator) -> np.ndarray:
        """Apply ``operation`` to the input ``state`` and return a new ``state``."""
        matrix = operation.matrix()
        if operation.name == "CNOT":
            return cls.apply_cnot(state, operation.wires)
        if len(operation.wires) < 3:
            return cls.apply_matrix_einsum(state, matrix, operation.wires)
        return cls.apply_matrix_tensordot(state, matrix, operation.wires)

    @staticmethod
    def apply_x(state: np.ndarray, index: int) -> np.ndarray:
        """Apply an X gate at position ``index``"""
        return np.roll(state, 1, index)

    @classmethod
    def apply_cnot(cls, state: np.ndarray, indices: tuple(int, int)) -> np.ndarray:
        """Apply a CNOT gate on the state at ``indices``."""
        ndim = np.ndim(state)
        sl_0 = _get_slice(0, indices[0], ndim)
        sl_1 = _get_slice(1, indices[0], ndim)

        target_axes = [indices[1] - 1] if indices[1] > indices[0] else [indices[1]]
        state_x = cls.apply_x(state[sl_1], index=target_axes)
        return np.stack([state[sl_0], state_x], axis=indices[0])

    @classmethod
    def apply_matrix(cls, state: np.ndarray, matrix: np.ndarray, indices: tuple) -> np.ndarray:
        """Apply ``matrix`` to ``state`` at ``indices``. Dispatches between using einsum and tensordot
        based on the number of indices."""
        if len(indices) < 3:
            return cls.apply_matrix_einsum(state, matrix, indices)
        return cls.apply_matrix_tensordot(state, matrix, indices)

    @staticmethod
    def apply_matrix_tensordot(state: np.ndarray, matrix: np.ndarray, indices: tuple) -> np.ndarray:
        """Apply ``matrix`` to ``state`` at ``indices`` using ``np.tensordot``. More efficient at higher numbers
        of indices."""
        total_indices = len(state.shape)
        num_indices = len(indices)
        reshaped_mat = np.reshape(matrix, [2] * (num_indices * 2))
        axes = (tuple(range(num_indices, 2 * num_indices)), indices)

        tdot = np.tensordot(reshaped_mat, state, axes=axes)

        unused_idxs = [i for i in range(total_indices) if i not in indices]
        perm = list(indices) + unused_idxs
        inv_perm = np.argsort(perm)

        return np.transpose(tdot, inv_perm)

    @staticmethod
    def apply_matrix_einsum(state: np.ndarray, matrix: np.ndarray, indices: tuple) -> np.ndarray:
        """Apply ``matrix`` to ``state`` at ``indices``. Uses ``np.einsum`` and is more efficent at lower qubit
        numbers.

        Args:
            state (array[complex]): input state
            mat (array): matrix to multiply
            indices (Iterable[integer]): indices to apply the matrix on

        Returns:
            array[complex]: output_state
        """
        total_indices = len(state.shape)
        num_indices = len(indices)

        state_indices = ABC[:total_indices]
        affected_indices = "".join(ABC[i] for i in indices)

        new_indices = ABC[total_indices : total_indices + num_indices]

        new_state_indices = state_indices
        for old, new in zip(affected_indices, new_indices):
            new_state_indices = new_state_indices.replace(old, new)

        einsum_indices = f"{new_indices}{affected_indices},{state_indices}->{new_state_indices}"

        reshaped_mat = np.reshape(matrix, [2] * (num_indices * 2))

        return np.einsum(einsum_indices, reshaped_mat, state)

    @classmethod
    def measure_state(
        cls, state: np.ndarray, measurementprocess: qml.measurements.MeasurementProcess
    ):
        """Measure ``state`` using ``measurementprocess``."""
        if isinstance(measurementprocess, qml.measurements.StateMeasurement):
            total_indices = len(state.shape)
            wires = qml.wires.Wires(range(total_indices))
            if (
                measurementprocess.obs is not None
                and measurementprocess.obs.has_diagonalizing_gates
            ):
                for op in measurementprocess.obs.diagonalizing_gates():
                    state = cls.apply_operation(state, op)
            return measurementprocess.process_state(state.flatten(), wires)
        return state

    # pylint: disable=protected-access
    @classmethod
    def generate_samples(
        cls, state: np.ndarray, rng: np.random._generator.Generator, shots: int = 1
    ):
        """UNFINISHED!

        Generate ``shots`` samples from ``state`` using the provided random number generator.
        """
        total_indices = len(state.shape)
        probs = np.real(state) ** 2 + np.imag(state) ** 2
        basis_states = np.arange(2**total_indices)
        samples = rng.choice(basis_states, shots, p=probs.flatten())

        powers_of_two = 1 << np.arange(total_indices, dtype=np.int64)
        # `samples` typically is one-dimensional, but can be two-dimensional with broadcasting.
        # In any case we want to append a new axis at the *end* of the shape.
        states_sampled_base_ten = samples[..., None] & powers_of_two
        # `states_sampled_base_ten` can be two- or three-dimensional. We revert the *last* axis.
        return (states_sampled_base_ten > 0).astype(np.int64)[..., ::-1]
