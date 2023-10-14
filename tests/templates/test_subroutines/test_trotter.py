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
Tests for the TrotterProduct template and helper functions.
"""
import copy
import pytest
from functools import reduce

import pennylane as qml
from pennylane.templates.subroutines.trotter import (
    _recursive_expression,
    _scalar,
)  # pylint: disable=private-access
from pennylane import numpy as qnp

test_hamiltonians = (
    qml.dot([1, 1, 1], [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(1)]),
    qml.dot(
        [1.23, -0.45], [qml.s_prod(0.1, qml.PauliX(0)), qml.prod(qml.PauliX(0), qml.PauliZ(1))]
    ),  # op arith
    qml.dot(
        [1, -0.5, 0.5], [qml.Identity(wires=[0, 1]), qml.PauliZ(0), qml.PauliZ(0)]
    ),  # H = Identity
)

p_4 = (4 - 4 ** (1 / 3)) ** -1

test_decompositions = {  # (hamiltonian_index, order): decomposition assuming t = 4.2
    (0, 1): [
        qml.exp(qml.PauliX(0), 4.2j),
        qml.exp(qml.PauliY(0), 4.2j),
        qml.exp(qml.PauliZ(1), 4.2j),
    ],
    (0, 2): [
        qml.exp(qml.PauliX(0), 4.2j / 2),
        qml.exp(qml.PauliY(0), 4.2j / 2),
        qml.exp(qml.PauliZ(1), 4.2j / 2),
        qml.exp(qml.PauliZ(1), 4.2j / 2),
        qml.exp(qml.PauliY(0), 4.2j / 2),
        qml.exp(qml.PauliX(0), 4.2j / 2),
    ],
    (0, 4): [
        qml.exp(qml.PauliX(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliY(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliZ(1), p_4 * 4.2j / 2),
        qml.exp(qml.PauliZ(1), p_4 * 4.2j / 2),
        qml.exp(qml.PauliY(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliX(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliX(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliY(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliZ(1), p_4 * 4.2j / 2),
        qml.exp(qml.PauliZ(1), p_4 * 4.2j / 2),
        qml.exp(qml.PauliY(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliX(0), p_4 * 4.2j / 2),  # S_2(p * t) ^ 2
        qml.exp(qml.PauliX(0), (1 - 4 * p_4) * 4.2j / 2),
        qml.exp(qml.PauliY(0), (1 - 4 * p_4) * 4.2j / 2),
        qml.exp(qml.PauliZ(1), (1 - 4 * p_4) * 4.2j / 2),
        qml.exp(qml.PauliZ(1), (1 - 4 * p_4) * 4.2j / 2),
        qml.exp(qml.PauliY(0), (1 - 4 * p_4) * 4.2j / 2),
        qml.exp(qml.PauliX(0), (1 - 4 * p_4) * 4.2j / 2),  # S_2((1 - 4p) * t)
        qml.exp(qml.PauliX(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliY(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliZ(1), p_4 * 4.2j / 2),
        qml.exp(qml.PauliZ(1), p_4 * 4.2j / 2),
        qml.exp(qml.PauliY(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliX(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliX(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliY(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliZ(1), p_4 * 4.2j / 2),
        qml.exp(qml.PauliZ(1), p_4 * 4.2j / 2),
        qml.exp(qml.PauliY(0), p_4 * 4.2j / 2),
        qml.exp(qml.PauliX(0), p_4 * 4.2j / 2),  # S_2(p * t) ^ 2
    ],
    (1, 1): [
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), 1.23 * 4.2j),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), -0.45 * 4.2j),
    ],
    (1, 2): [
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), 1.23 * 4.2j / 2),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), -0.45 * 4.2j / 2),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), -0.45 * 4.2j / 2),
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), 1.23 * 4.2j / 2),
    ],
    (1, 4): [
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * 4.2j / 2),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * 4.2j / 2),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * 4.2j / 2),
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * 4.2j / 2),
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * 4.2j / 2),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * 4.2j / 2),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * 4.2j / 2),
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * 4.2j / 2),
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), (1 - 4 * p_4) * 1.23 * 4.2j / 2),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), (1 - 4 * p_4) * -0.45 * 4.2j / 2),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), (1 - 4 * p_4) * -0.45 * 4.2j / 2),
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), (1 - 4 * p_4) * 1.23 * 4.2j / 2),
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * 4.2j / 2),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * 4.2j / 2),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * 4.2j / 2),
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * 4.2j / 2),
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * 4.2j / 2),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * 4.2j / 2),
        qml.exp(qml.prod(qml.PauliX(0), qml.PauliZ(1)), p_4 * -0.45 * 4.2j / 2),
        qml.exp(qml.s_prod(0.1, qml.PauliX(0)), p_4 * 1.23 * 4.2j / 2),
    ],
    (2, 1): [
        qml.exp(qml.Identity(wires=[0, 1]), 4.2j),
        qml.exp(qml.PauliZ(0), -0.5 * 4.2j),
        qml.exp(qml.PauliZ(0), 0.5 * 4.2j),
    ],
    (2, 2): [
        qml.exp(qml.Identity(wires=[0, 1]), 4.2j / 2),
        qml.exp(qml.PauliZ(0), -0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), 0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), 0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), -0.5 * 4.2j / 2),
        qml.exp(qml.Identity(wires=[0, 1]), 4.2j / 2),
    ],
    (2, 4): [
        qml.exp(qml.Identity(wires=[0, 1]), p_4 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * -0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * 0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * 0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * -0.5 * 4.2j / 2),
        qml.exp(qml.Identity(wires=[0, 1]), p_4 * 4.2j / 2),
        qml.exp(qml.Identity(wires=[0, 1]), p_4 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * -0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * 0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * 0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * -0.5 * 4.2j / 2),
        qml.exp(qml.Identity(wires=[0, 1]), p_4 * 4.2j / 2),
        qml.exp(qml.Identity(wires=[0, 1]), (1 - 4 * p_4) * 4.2j / 2),
        qml.exp(qml.PauliZ(0), (1 - 4 * p_4) * -0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), (1 - 4 * p_4) * 0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), (1 - 4 * p_4) * 0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), (1 - 4 * p_4) * -0.5 * 4.2j / 2),
        qml.exp(qml.Identity(wires=[0, 1]), (1 - 4 * p_4) * 4.2j / 2),
        qml.exp(qml.Identity(wires=[0, 1]), p_4 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * -0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * 0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * 0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * -0.5 * 4.2j / 2),
        qml.exp(qml.Identity(wires=[0, 1]), p_4 * 4.2j / 2),
        qml.exp(qml.Identity(wires=[0, 1]), p_4 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * -0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * 0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * 0.5 * 4.2j / 2),
        qml.exp(qml.PauliZ(0), p_4 * -0.5 * 4.2j / 2),
        qml.exp(qml.Identity(wires=[0, 1]), p_4 * 4.2j / 2),
    ],
}


class TestInitialization:
    """Test the TrotterProduct class initializes correctly."""

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
                qml.TrotterProduct(hamiltonian, time=1.23)

        else:
            try:
                qml.TrotterProduct(hamiltonian, time=1.23)
            except TypeError:
                assert False  # test should fail if an error was raised when we expect it not to

    @pytest.mark.parametrize(
        "hamiltonian",
        (
            qml.Hamiltonian([1.23, 4 + 5j], [qml.PauliX(0), qml.PauliZ(1)]),
            qml.dot([1.23, 4 + 5j], [qml.PauliX(0), qml.PauliZ(1)]),
            qml.dot([1.23, 0.5], [qml.RY(1.23, 0), qml.RZ(3.45, 1)]),
        ),
    )
    def test_error_hermiticity(self, hamiltonian):
        """Test that an error is raised if any terms in
        the Hamiltonian are not Hermitian and check_hermitian is True."""

        with pytest.raises(
            ValueError, match="One or more of the terms in the Hamiltonian may not be Hermitian"
        ):
            qml.TrotterProduct(hamiltonian, time=0.5)

        try:
            qml.TrotterProduct(hamiltonian, time=0.5, check_hermitian=False)
        except ValueError:
            assert False  # No error should be raised if the check_hermitian flag is disabled

    @pytest.mark.parametrize("order", (-1, 0, 0.5, 3, 7.0))
    def test_error_order(self, order):
        """Test that an error is raised if 'order' is not one or positive even number."""
        time = 0.5
        hamiltonian = qml.dot([1.23, 3.45], [qml.PauliX(0), qml.PauliZ(1)])

        with pytest.raises(
            ValueError, match="The order of a TrotterProduct must be 1 or a positive even integer,"
        ):
            qml.TrotterProduct(hamiltonian, time, order=order)

    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_init_correctly(self, hamiltonian):
        """Test that all of the attributes are initalized correctly."""
        time, n, order = (4.2, 10, 4)
        op = qml.TrotterProduct(hamiltonian, time, n=n, order=order, check_hermitian=False)

        assert op.wires == hamiltonian.wires
        assert op.parameters == [time]
        assert op.data == (time,)
        assert op.hyperparameters == {
            "base": hamiltonian,
            "n": n,
            "order": order,
            "check_hermitian": False,
        }

    @pytest.mark.parametrize("n", (1, 2, 5, 10))
    @pytest.mark.parametrize("time", (0.5, 1.2))
    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_copy(self, hamiltonian, time, n, order):
        """Test that we can make deep and shallow copies of TrotterProduct correctly."""
        op = qml.TrotterProduct(hamiltonian, time, n=n, order=order)
        new_op = copy.copy(op)

        assert op.wires == new_op.wires
        assert op.parameters == new_op.parameters
        assert op.data == new_op.data
        assert op.hyperparameters == new_op.hyperparameters
        assert op is not new_op

    @pytest.mark.parametrize("hamiltonian", test_hamiltonians)
    def test_flatten_and_unflatten(self, hamiltonian):
        """Test that the flatten and unflatten methods work correctly."""
        time, n, order = (4.2, 10, 4)
        op = qml.TrotterProduct(hamiltonian, time, n=n, order=order)

        data, metadata = op._flatten()
        assert qml.equal(data[0], hamiltonian)
        assert data[1] == time
        assert dict(metadata) == {"n": n, "order": order, "check_hermitian": True}

        new_op = type(op)._unflatten(data, metadata)
        assert qml.equal(op, new_op)
        assert new_op is not op


class TestPrivateFunctions:
    """Test the private helper functions."""

    @pytest.mark.parametrize(
        "order, result",
        (
            (4, 0.4144907717943757),
            (6, 0.3730658277332728),
            (8, 0.35958464934999224),
        ),
    )  # Computed by hand
    def test_private_scalar(self, order, result):
        """Test the _scalar function correctly computes the parameter scalar."""
        s = _scalar(order)
        assert qnp.isclose(s, result)

    expected_expansions = (  # for H = X0 + Y0 + Z1, t = 1.23, computed by hand
        [  # S_1(t)
            qml.exp(qml.PauliX(0), 1.23j),
            qml.exp(qml.PauliY(0), 1.23j),
            qml.exp(qml.PauliZ(1), 1.23j),
        ],
        [  # S_2(t)
            qml.exp(qml.PauliX(0), 1.23j / 2),
            qml.exp(qml.PauliY(0), 1.23j / 2),
            qml.exp(qml.PauliZ(1), 1.23j / 2),
            qml.exp(qml.PauliZ(1), 1.23j / 2),
            qml.exp(qml.PauliY(0), 1.23j / 2),
            qml.exp(qml.PauliX(0), 1.23j / 2),
        ],
        [  # S_4(t)
            qml.exp(qml.PauliX(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliY(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 1.23j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 1.23j / 2),
            qml.exp(qml.PauliY(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliX(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliX(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliY(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 1.23j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 1.23j / 2),
            qml.exp(qml.PauliY(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliX(0), p_4 * 1.23j / 2),  # S_2(p * t) ^ 2
            qml.exp(qml.PauliX(0), (1 - 4 * p_4) * 1.23j / 2),
            qml.exp(qml.PauliY(0), (1 - 4 * p_4) * 1.23j / 2),
            qml.exp(qml.PauliZ(1), (1 - 4 * p_4) * 1.23j / 2),
            qml.exp(qml.PauliZ(1), (1 - 4 * p_4) * 1.23j / 2),
            qml.exp(qml.PauliY(0), (1 - 4 * p_4) * 1.23j / 2),
            qml.exp(qml.PauliX(0), (1 - 4 * p_4) * 1.23j / 2),  # S_2((1 - 4p) * t)
            qml.exp(qml.PauliX(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliY(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 1.23j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 1.23j / 2),
            qml.exp(qml.PauliY(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliX(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliX(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliY(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 1.23j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 1.23j / 2),
            qml.exp(qml.PauliY(0), p_4 * 1.23j / 2),
            qml.exp(qml.PauliX(0), p_4 * 1.23j / 2),  # S_2(p * t) ^ 2
        ],
    )

    @pytest.mark.parametrize("order, expected_expansion", zip((1, 2, 4), expected_expansions))
    def test_recursive_expression_no_queue(self, order, expected_expansion):
        """Test the _recursive_expression function correctly generates the decomposition"""
        ops = [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(1)]

        with qml.tape.QuantumTape() as tape:
            decomp = _recursive_expression(1.23, order, ops)

        assert tape.operations == []  # No queuing!
        assert all(
            qml.equal(op1, op2) for op1, op2 in zip(decomp, expected_expansion)
        )  # Expected expression


class TestDecomposition:
    """Test the decomposition of the TrotterProduct class."""

    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("hamiltonian_index, hamiltonian", enumerate(test_hamiltonians))
    def test_compute_decomposition(self, hamiltonian, hamiltonian_index, order):
        """Test the decomposition is correct and queues"""
        op = qml.TrotterProduct(hamiltonian, 4.2, order=order)
        with qml.tape.QuantumTape() as tape:
            decomp = op.compute_decomposition(*op.parameters, **op.hyperparameters)

        assert decomp == tape.operations  # queue matches decomp with circuit ordering

        decomp = [qml.simplify(op) for op in decomp]
        true_decomp = [
            qml.simplify(op) for op in test_decompositions[(hamiltonian_index, order)][-1::-1]
        ]
        assert all(
            qml.equal(op1, op2) for op1, op2 in zip(decomp, true_decomp)
        )  # decomp is correct

    @pytest.mark.parametrize("order", (1, 2))
    @pytest.mark.parametrize("num_steps", (1, 2, 3))
    def test_compute_decomposition_n_steps(self, num_steps, order):
        """Test the decomposition is correct when we set the number of trotter steps"""
        time = 0.5
        hamiltonian = qml.sum(qml.PauliX(0), qml.PauliZ(0))

        if order == 1:
            base_decomp = [
                qml.exp(qml.PauliZ(0), 0.5j / num_steps),
                qml.exp(qml.PauliX(0), 0.5j / num_steps),
            ]
        if order == 2:
            base_decomp = [
                qml.exp(qml.PauliX(0), 0.25j / num_steps),
                qml.exp(qml.PauliZ(0), 0.25j / num_steps),
                qml.exp(qml.PauliZ(0), 0.25j / num_steps),
                qml.exp(qml.PauliX(0), 0.25j / num_steps),
            ]

        true_decomp = base_decomp * num_steps

        op = qml.TrotterProduct(hamiltonian, time, n=num_steps, order=order)
        decomp = op.compute_decomposition(*op.parameters, **op.hyperparameters)
        assert all(qml.equal(op1, op2) for op1, op2 in zip(decomp, true_decomp))


class TestIntegration:
    """Test that the TrotterProduct can be executed and differentiated
    through all interfaces."""

    @pytest.mark.parametrize("order", (1, 2, 4))
    @pytest.mark.parametrize("hamiltonian_index, hamiltonian", enumerate(test_hamiltonians))
    def test_execute_circuit(self, hamiltonian, hamiltonian_index, order):
        """Test that the gate executes correctly in a circuit."""
        wires = hamiltonian.wires
        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def circ():
            qml.TrotterProduct(hamiltonian, time=4.2, order=order)
            return qml.state()

        initial_state = qnp.zeros(2 ** (len(wires)))
        initial_state[0] = 1

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [
                    qml.matrix(op, wire_order=wires)
                    for op in test_decompositions[(hamiltonian_index, order)]
                ],
            )
            @ initial_state
        )
        state = circ()

        assert qnp.allclose(expected_state, state)

    @pytest.mark.parametrize("order", (1, 2))
    @pytest.mark.parametrize("num_steps", (1, 2, 3))
    def test_execute_circuit_n_steps(self, num_steps, order):
        """Test that the circuit executes as expected when we set the number of trotter steps"""
        time = 0.5
        hamiltonian = qml.sum(qml.PauliX(0), qml.PauliZ(0))

        if order == 1:
            base_decomp = [
                qml.exp(qml.PauliZ(0), 0.5j / num_steps),
                qml.exp(qml.PauliX(0), 0.5j / num_steps),
            ]
        if order == 2:
            base_decomp = [
                qml.exp(qml.PauliX(0), 0.25j / num_steps),
                qml.exp(qml.PauliZ(0), 0.25j / num_steps),
                qml.exp(qml.PauliZ(0), 0.25j / num_steps),
                qml.exp(qml.PauliX(0), 0.25j / num_steps),
            ]

        true_decomp = base_decomp * num_steps

        wires = hamiltonian.wires
        dev = qml.device("default.qubit", wires=wires)

        @qml.qnode(dev)
        def circ():
            qml.TrotterProduct(hamiltonian, time, n=num_steps, order=order)
            return qml.state()

        initial_state = qnp.zeros(2 ** (len(wires)))
        initial_state[0] = 1

        expected_state = (
            reduce(
                lambda x, y: x @ y, [qml.matrix(op, wire_order=wires) for op in true_decomp[-1::-1]]
            )
            @ initial_state
        )
        state = circ()
        assert qnp.allclose(expected_state, state)
