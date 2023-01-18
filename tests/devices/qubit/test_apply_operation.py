# Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import pennylane as qml
import numpy as np

from pennylane.devices.qubit.apply_operation import (
    apply_operation,
    apply_operation_einsum,
    apply_operation_tensordot,
)


@pytest.mark.parametrize(
    "method", (apply_operation_einsum, apply_operation_tensordot, apply_operation)
)
@pytest.mark.parametrize("wire", (0, 1))
class TestTwoQubitStateSpecialCases:
    """Test the special cases on a two qubit state.  Also tests the special cases for einsum and tensor application methods
    for additional testing of these generic matrix application methods."""

    def test_paulix(self, method, wire):
        """Test the application of a paulix gate on a two qubit state."""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )

        new_state = method(qml.PauliX(wire), initial_state)

        initial0dim = np.take(initial_state, 0, axis=wire)
        new1dim = np.take(new_state, 1, axis=wire)

        assert qml.math.allclose(initial0dim, new1dim)

        initial1dim = np.take(initial_state, 1, axis=wire)
        new0dim = np.take(new_state, 0, axis=wire)
        assert qml.math.allclose(initial1dim, new0dim)

    def test_pauliz(self, method, wire):
        """Test the application of a pauliz gate on a two qubit state."""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )

        new_state = method(qml.PauliZ(wire), initial_state)

        initial0 = np.take(initial_state, 0, axis=wire)
        new0 = np.take(new_state, 0, axis=wire)
        assert qml.math.allclose(initial0, new0)

        initial1 = np.take(initial_state, 1, axis=wire)
        new1 = np.take(new_state, 1, axis=wire)
        assert qml.math.allclose(initial1, -new1)

    def test_pauliy(self, method, wire):
        """Test the application of a pauliy gate on a two qubit state."""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )

        new_state = method(qml.PauliY(wire), initial_state)

        initial0 = np.take(initial_state, 0, axis=wire)
        new1 = np.take(new_state, 1, axis=wire)
        assert qml.math.allclose(1j * initial0, new1)

        initial1 = np.take(initial_state, 1, axis=wire)
        new0 = np.take(new_state, 0, axis=wire)
        assert qml.math.allclose(-1j * initial1, new0)

    def test_hadamard(self, method, wire):
        """Test the application of a hadamard on a two qubit state."""
        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )

        new_state = method(qml.Hadamard(wire), initial_state)

        inv_sqrt2 = 1 / np.sqrt(2)

        initial0 = np.take(initial_state, 0, axis=wire)
        initial1 = np.take(initial_state, 1, axis=wire)

        expected0 = inv_sqrt2 * (initial0 + initial1)
        new0 = np.take(new_state, 0, axis=wire)
        assert qml.math.allclose(new0, expected0)

        expected1 = inv_sqrt2 * (initial0 - initial1)
        new1 = np.take(new_state, 1, axis=wire)
        assert qml.math.allclose(new1, expected1)

    def test_phaseshift(self, method, wire):
        """test the application of a phaseshift gate on a two qubit state."""

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )

        phase = -2.3
        shift = np.exp(1j * phase)

        new_state = method(qml.PhaseShift(phase, wire), initial_state)

        new0 = np.take(new_state, 0, axis=wire)
        initial0 = np.take(initial_state, 0, axis=wire)
        assert qml.math.allclose(new0, initial0)

        initial1 = np.take(initial_state, 1, axis=wire)
        new1 = np.take(new_state, 1, axis=wire)
        assert qml.math.allclose(shift * initial1, new1)

    def test_cnot(self, method, wire):
        """Test the application of a cnot gate on a two qubit state."""

        initial_state = np.array(
            [
                [0.04624539 + 0.3895457j, 0.22399401 + 0.53870339j],
                [-0.483054 + 0.2468498j, -0.02772249 - 0.45901669j],
            ]
        )

        control = wire
        target = int(not control)

        new_state = method(qml.CNOT((control, target)), initial_state)

        initial0 = np.take(initial_state, 0, axis=control)
        new0 = np.take(new_state, 0, axis=control)
        assert qml.math.allclose(initial0, new0)

        initial1_reversed = np.take(initial_state, 1, axis=control)[::-1]
        new1 = np.take(new_state, 1, axis=control)
        assert qml.math.allclose(initial1_reversed, new1)
