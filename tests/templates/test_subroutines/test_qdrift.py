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
from unittest.mock import patch

import pytest

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.math import allclose, get_interface
from pennylane.templates.subroutines.qdrift import QDrift


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
        [qml.Identity(wires=[0, 1]), qml.PauliZ(0), qml.PauliZ(0)],
    ),  # H = Identity
)


def mocked_choice(exps, p, size, replace=True):
    """Fix randomness"""
    return []


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

    @patch("numpy.random.choice", mocked_choice)
    def test_private_sample(self):
        """Test the private function which samples the decomposition"""

    pass


class TestIntegration:
    def test_error_gradient_workflow(self):
        """Test that an error is raised if we require a gradient of QDrift with respect to hamiltonian coefficients."""
        pass
