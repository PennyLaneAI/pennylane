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
"""
Tests for the CommutingEvolution template.
"""
import pytest
import numpy as np
from pennylane import numpy as pnp
import pennylane as qml


class TestInputs:
    """Tests for input validation of `CommutingEvolution`."""

    def test_invalid_hamiltonian(self):
        """Tests TypeError is raised if `hamiltonian` is not type `qml.Hamiltonian`."""

        invalid_operator = qml.PauliX(0)

        assert pytest.raises(TypeError, qml.CommutingEvolution, invalid_operator, 1)


class TestGradients:
    """Tests that correct gradients are obtained for `CommutingEvolution` when frequencies
    are specified."""

    def test_two_term_case(self):
        """Tests the paramer shift rules for `CommutingEvolution` equal the
        finite difference result for a two term shift rule case."""

        n_wires = 1
        dev = qml.device("default.qubit", wires=n_wires)

        hamiltonian = qml.Hamiltonian([1], [qml.PauliX(0)])
        frequencies = [2]

        @qml.qnode(dev)
        def circuit(time):
            qml.PauliX(0)
            qml.CommutingEvolution(hamiltonian, time, frequencies)
            return qml.expval(qml.PauliZ(0))

        x_vals = np.linspace(-np.pi, np.pi, num=10)

        grads_finite_diff = [qml.gradients.finite_diff(circuit)(x) for x in x_vals]
        grads_param_shift = [qml.gradients.param_shift(circuit)(x) for x in x_vals]

        assert all(np.isclose(grads_finite_diff, grads_param_shift, atol=1e-7))

    def test_four_term_case(self):
        """Tests the paramer shift rules for `CommutingEvolution` equal the
        finite difference result for a four term shift rule case."""

        n_wires = 2
        dev = qml.device("default.qubit", wires=n_wires)

        coeffs = [1, -1]
        obs = [qml.PauliX(0) @ qml.PauliY(1), qml.PauliY(0) @ qml.PauliX(1)]
        hamiltonian = qml.Hamiltonian(coeffs, obs)
        frequencies = [2, 4]

        @qml.qnode(dev)
        def circuit(time):
            qml.PauliX(0)
            qml.CommutingEvolution(hamiltonian, time, frequencies)
            return qml.expval(qml.PauliZ(0))

        x_vals = np.linspace(-np.pi, np.pi, num=10)

        grads_finite_diff = [qml.gradients.finite_diff(circuit)(x) for x in x_vals]
        grads_param_shift = [qml.gradients.param_shift(circuit)(x) for x in x_vals]

        assert all(np.isclose(grads_finite_diff, grads_param_shift, atol=1e-7))
