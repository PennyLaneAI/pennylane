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
"""
Tests for the QDrift template.
"""
import copy
import pytest
from functools import reduce

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.math import allclose, get_interface
from pennylane.templates.subroutines.qdrift import _sample_decomposition


test_hamiltonians = (
    (
        [1, 1, 1],
        [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(1)],
    ),
    (
        [1.23, -0.45],
        [qml.s_prod(0.1, qml.PauliX(0)), qml.prod(qml.PauliX(0), qml.PauliZ(1))],
    ),  # op arith
    (
        [1, -0.5, 0.5],
        [qml.Identity(wires=[0, 1]), qml.PauliZ(0), qml.PauliZ(1)],
    ),
)


def mocked_choice(*args, **kwargs):
    """Fix randomness"""
    n, t, normalization = (10, 1.23, 2)
    print("Here!")

    return [
        qml.exp(qml.Identity([0, 1]), 1 * normalization * t * 1j / n),
        qml.exp(qml.PauliZ(0), -1 * normalization * t * 1j / n),
        qml.exp(qml.PauliZ(1), 1 * normalization * t * 1j / n),
        qml.exp(qml.Identity([0, 1]), 1 * normalization * t * 1j / n),
        qml.exp(qml.PauliZ(0), -1 * normalization * t * 1j / n),
        qml.exp(qml.PauliZ(1), 1 * normalization * t * 1j / n),
        qml.exp(qml.Identity([0, 1]), 1 * normalization * t * 1j / n),
        qml.exp(qml.PauliZ(0), -1 * normalization * t * 1j / n),
        qml.exp(qml.PauliZ(1), 1 * normalization * t * 1j / n),
        qml.exp(qml.Identity([0, 1]), 1 * normalization * t * 1j / n),
    ]


class TestInitialization:
    """Test that the class is intialized correctly."""

    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    @pytest.mark.parametrize("seed", (None, 1234, 42))
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_init_correctly(self, coeffs, ops, time, n, seed):
        """Test that all of the attributes are initalized correctly."""
        h = qml.dot(coeffs, ops)
        op = qml.QDrift(h, time, n=n, seed=seed)

        assert op.wires == h.wires
        assert op.parameters == [time]
        assert op.data == (time,)

        assert op.hyperparameters["n"] == n
        assert op.hyperparameters["seed"] == seed
        assert op.hyperparameters["base"] == h

        for term in op.hyperparameters["decomposition"]:
            # the decomposition is solely made up of exponentials of ops sampled from hamiltonian terms
            assert term.base in ops

    def test_set_decomp(self):
        """Test that setting the decomposition works correctly."""
        h = qml.dot([1.23, -0.45], [qml.PauliX(0), qml.PauliY(0)])
        decomposition = [
            qml.exp(qml.PauliX(0), 0.5j * 1.68 / 3),
            qml.exp(qml.PauliY(0), -0.5j * 1.68 / 3),
            qml.exp(qml.PauliX(0), 0.5j * 1.68 / 3),
        ]
        op = qml.QDrift(h, 0.5, n=3, decomposition=decomposition)

        assert op.hyperparameters["decomposition"] == decomposition

    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    @pytest.mark.parametrize("seed", (None, 1234, 42))
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_copy(self, coeffs, ops, time, n, seed):
        """Test that we can make copies of QDrift correctly."""
        h = qml.dot(coeffs, ops)
        op = qml.QDrift(h, time, n=n, seed=seed)
        new_op = copy.copy(op)

        assert op.wires == new_op.wires
        assert op.parameters == new_op.parameters
        assert op.data == new_op.data
        assert op.hyperparameters == new_op.hyperparameters
        assert op is not new_op

    @pytest.mark.parametrize(
        "hamiltonian, raise_error",
        (
            (qml.PauliX(0), True),
            (qml.prod(qml.PauliX(0), qml.PauliZ(1)), True),
            (qml.Hamiltonian([1.23, 3.45], [qml.PauliX(0), qml.PauliZ(1)]), False),
            (qml.dot([1.23, 3.45], [qml.PauliX(0), qml.PauliZ(1)]), False),
        ),
    )
    def test_error_type(self, hamiltonian, raise_error):
        """Test an error is raised of an incorrect type is passed"""
        if raise_error:
            with pytest.raises(
                TypeError, match="The given operator must be a PennyLane ~.Hamiltonian or ~.Sum"
            ):
                qml.QDrift(hamiltonian, time=1.23)
        else:
            try:
                qml.QDrift(hamiltonian, time=1.23)
            except TypeError:
                assert False  # test should fail if an error was raised when we expect it not to

    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_flatten_and_unflatten(self, coeffs, ops):
        """Test that the flatten and unflatten methods work correctly."""
        time, n, seed = (0.5, 2, 1234)
        hamiltonian = qml.dot(coeffs, ops)
        op = qml.QDrift(hamiltonian, time, n=n, seed=seed)
        decomp = op.decomposition()

        data, metadata = op._flatten()
        assert data[0] == time
        assert metadata[0] == op.wires
        assert dict(metadata[1]) == {
            "n": n,
            "seed": seed,
            "base": hamiltonian,
            "decomposition": decomp,
        }

        new_op = type(op)._unflatten(data, metadata)
        assert qml.equal(op, new_op)
        assert new_op is not op


class TestDecomposition:
    """Test decompositions are generated correctly."""

    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    @pytest.mark.parametrize("seed", (None, 1234, 42))
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_private_sample(self, coeffs, ops, time, seed, n):
        """Test the private function which samples the decomposition"""
        ops_to_coeffs = dict(zip(ops, coeffs))
        normalization = qnp.sum(qnp.abs(coeffs))
        decomp = _sample_decomposition(coeffs, ops, time, n, seed)

        assert len(decomp) == n
        for term in decomp:
            exponent_coeff_sign = qml.math.sign(ops_to_coeffs[term.base])
            assert term.base in ops  # sample from ops
            assert term.coeff == (
                exponent_coeff_sign * normalization * time * 1j / n
            )  # with this exponent

    @pytest.mark.parametrize("seed", (1234, 42))
    def test_compute_decomposition(self, seed):
        """Test that the decomposition is computed and queues correctly."""
        coeffs = [1, -0.5, 0.5]
        ops = [qml.Identity(wires=[0, 1]), qml.PauliZ(0), qml.PauliZ(1)]

        h = qml.dot(coeffs, ops)
        op = qml.QDrift(h, time=1.23, n=10, seed=seed)

        expected_decomp = _sample_decomposition(coeffs, ops, 1.23, 10, seed=seed)

        with qml.tape.QuantumTape() as tape:
            decomp = op.compute_decomposition(*op.parameters, **op.hyperparameters)

        assert all(decomp == tape.operations)  # queue matches decomp with circuit ordering
        assert all(decomp == expected_decomp)  # sample the same ops


class TestIntegration:
    """Test that the QDrift template integrates well with the rest of PennyLane"""

    @pytest.mark.parametrize("n", (1, 2, 3))
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    @pytest.mark.parametrize("seed", (1234, 42))
    @pytest.mark.parametrize("coeffs, ops", test_hamiltonians)
    def test_execution(self, coeffs, ops, time, n, seed):
        """Test that the circuit executes as expected"""
        hamiltonian = qml.dot(coeffs, ops)
        wires = hamiltonian.wires
        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def circ():
            qml.QDrift(hamiltonian, time, n=n, seed=seed)
            return qml.state()

        expected_decomp = _sample_decomposition(coeffs, ops, time, n=n, seed=seed)

        initial_state = qnp.zeros(2 ** (len(wires)))
        initial_state[0] = 1

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=wires) for op in expected_decomp],
            )
            @ initial_state
        )
        state = circ()

        assert allclose(expected_state, state)

    def test_error_gradient_workflow(self):
        """Test that an error is raised if we require a gradient of QDrift with respect to hamiltonian coefficients."""
        pass

    @pytest.mark.autograd
    def test_autograd_gradient(self):
        """Test that the gradient is computed correctly"""
        pass
