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
# pylint: disable=private-access, protected-access
import copy
from functools import reduce

import pytest

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.math import allclose, get_interface
from pennylane.resource.error import SpectralNormError
from pennylane.templates.subroutines.trotter import _recursive_expression, _scalar

test_hamiltonians = (
    qml.dot([1, 1, 1], [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(1)]),
    qml.dot(
        [1.23, -0.45], [qml.s_prod(0.1, qml.PauliX(0)), qml.prod(qml.PauliX(0), qml.PauliZ(1))]
    ),  # op arith
    qml.dot(
        [1, -0.5, 0.5], [qml.Identity(wires=[0, 1]), qml.PauliZ(0), qml.PauliZ(0)]
    ),  # H = Identity
    qml.dot([2, 2, 2], [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(1)]),
)

p_4 = (4 - 4 ** (1 / 3)) ** -1

test_decompositions = (
    {  # (hamiltonian_index, order): decomposition assuming t = 4.2, computed by hand
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
        (3, 1): [
            qml.exp(qml.PauliX(0), 8.4j),
            qml.exp(qml.PauliY(0), 8.4j),
            qml.exp(qml.PauliZ(1), 8.4j),
        ],
        (3, 2): [
            qml.exp(qml.PauliX(0), 8.4j / 2),
            qml.exp(qml.PauliY(0), 8.4j / 2),
            qml.exp(qml.PauliZ(1), 8.4j / 2),
            qml.exp(qml.PauliZ(1), 8.4j / 2),
            qml.exp(qml.PauliY(0), 8.4j / 2),
            qml.exp(qml.PauliX(0), 8.4j / 2),
        ],
        (3, 4): [
            qml.exp(qml.PauliX(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliY(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 8.4j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 8.4j / 2),
            qml.exp(qml.PauliY(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliX(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliX(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliY(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 8.4j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 8.4j / 2),
            qml.exp(qml.PauliY(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliX(0), p_4 * 8.4j / 2),  # S_2(p * t) ^ 2
            qml.exp(qml.PauliX(0), (1 - 4 * p_4) * 8.4j / 2),
            qml.exp(qml.PauliY(0), (1 - 4 * p_4) * 8.4j / 2),
            qml.exp(qml.PauliZ(1), (1 - 4 * p_4) * 8.4j / 2),
            qml.exp(qml.PauliZ(1), (1 - 4 * p_4) * 8.4j / 2),
            qml.exp(qml.PauliY(0), (1 - 4 * p_4) * 8.4j / 2),
            qml.exp(qml.PauliX(0), (1 - 4 * p_4) * 8.4j / 2),  # S_2((1 - 4p) * t)
            qml.exp(qml.PauliX(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliY(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 8.4j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 8.4j / 2),
            qml.exp(qml.PauliY(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliX(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliX(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliY(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 8.4j / 2),
            qml.exp(qml.PauliZ(1), p_4 * 8.4j / 2),
            qml.exp(qml.PauliY(0), p_4 * 8.4j / 2),
            qml.exp(qml.PauliX(0), p_4 * 8.4j / 2),  # S_2(p * t) ^ 2
        ],
    }
)


def _generate_simple_decomp(coeffs, ops, time, order, n):
    """Given coeffs, ops and a time argument in a given framework, generate the
    Trotter product for order and number of trotter steps."""
    decomp = []
    if order == 1:
        decomp.extend(qml.exp(op, coeff * (time / n) * 1j) for coeff, op in zip(coeffs, ops))

    coeffs_ops = zip(coeffs, ops)

    if get_interface(coeffs) == "torch":
        import torch

        coeffs_ops_reversed = zip(torch.flip(coeffs, dims=(0,)), ops[::-1])
    else:
        coeffs_ops_reversed = zip(coeffs[::-1], ops[::-1])

    if order == 2:
        decomp.extend(qml.exp(op, coeff * (time / n) * 1j / 2) for coeff, op in coeffs_ops)
        decomp.extend(qml.exp(op, coeff * (time / n) * 1j / 2) for coeff, op in coeffs_ops_reversed)

    if order == 4:
        s_2 = []
        s_2_p = []

        for coeff, op in coeffs_ops:
            s_2.append(qml.exp(op, (p_4 * coeff) * (time / n) * 1j / 2))
            s_2_p.append(qml.exp(op, ((1 - (4 * p_4)) * coeff) * (time / n) * 1j / 2))

        for coeff, op in coeffs_ops_reversed:
            s_2.append(qml.exp(op, (p_4 * coeff) * (time / n) * 1j / 2))
            s_2_p.append(qml.exp(op, ((1 - (4 * p_4)) * coeff) * (time / n) * 1j / 2))

        decomp = (s_2 * 2) + s_2_p + (s_2 * 2)

    return decomp * n


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
                TypeError,
                match="The given operator must be a PennyLane ~.Hamiltonian, ~.Sum or ~.SProd",
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

    @pytest.mark.parametrize(
        "hamiltonian",
        (
            qml.Hamiltonian([1.0], [qml.PauliX(0)]),
            qml.dot([2.0], [qml.PauliY(0)]),
        ),
    )
    def test_error_hamiltonian(self, hamiltonian):
        """Test that an error is raised if the input Hamiltonian has only 1 term."""
        with pytest.raises(
            ValueError, match="There should be at least 2 terms in the Hamiltonian."
        ):
            qml.TrotterProduct(hamiltonian, 1.23, n=2, order=4)

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

        if isinstance(hamiltonian, qml.ops.op_math.SProd):
            hamiltonian = hamiltonian.simplify()

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

        if isinstance(hamiltonian, qml.ops.op_math.SProd):
            hamiltonian = hamiltonian.simplify()

        data, metadata = op._flatten()
        assert qml.equal(data[0], hamiltonian)
        assert data[1] == time
        assert dict(metadata) == {"n": n, "order": order, "check_hermitian": True}

        new_op = type(op)._unflatten(data, metadata)
        assert qml.equal(op, new_op)
        assert new_op is not op

    # TODO: Remove test when we deprecate ApproxTimeEvolution
    @pytest.mark.parametrize("n", (1, 2, 5, 10))
    @pytest.mark.parametrize("time", (0.5, 1.2))
    def test_convention_approx_time_evolv(self, time, n):
        """Test that TrotterProduct matches ApproxTimeEvolution"""
        hamiltonian = qml.Hamiltonian(
            [1.23, -0.45, 6], [qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
        )
        op1 = qml.TrotterProduct(hamiltonian, time, order=1, n=n)
        op2 = qml.adjoint(qml.ApproxTimeEvolution(hamiltonian, time, n=n))

        assert qnp.allclose(
            qml.matrix(op1, wire_order=hamiltonian.wires),
            qml.matrix(op2, wire_order=hamiltonian.wires),
        )

        op1 = qml.adjoint(qml.TrotterProduct(hamiltonian, time, order=1, n=n))
        op2 = qml.ApproxTimeEvolution(hamiltonian, time, n=n)

        assert qnp.allclose(
            qml.matrix(op1, wire_order=hamiltonian.wires),
            qml.matrix(op2, wire_order=hamiltonian.wires),
        )


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


class TestError:
    """Test the error method of the TrotterProduct class"""

    @pytest.mark.parametrize("fast", (True, False))
    def test_invalid_method(self, fast):
        """Test that passing an invalid method raises an error."""
        method = "crazy"
        op = qml.TrotterProduct(qml.sum(qml.X(0), qml.Y(0)), 1.23)
        with pytest.raises(ValueError, match=f"The '{method}' method is not supported"):
            _ = op.error(method, fast=fast)

    def test_one_norm_error_method(self):
        """Test that the one-norm error method works as expected."""
        op = qml.TrotterProduct(qml.sum(qml.X(0), qml.Y(0)), time=0.05, order=4)
        expected_error = ((10**5 + 1) / 120) * (0.1**5)

        for computed_error in (
            op.error(method="one-norm"),
            op.error(method="one-norm", fast=False),
        ):
            assert isinstance(computed_error, SpectralNormError)
            assert qnp.isclose(computed_error.error, expected_error)

    def test_commutator_error_method(self):
        """Test that the commutator error method works as expected."""
        op = qml.TrotterProduct(qml.sum(qml.X(0), qml.Y(0)), time=0.05, order=2, n=10)
        expected_error = (32 / 3) * (0.05**3) * (1 / 100)

        for computed_error in (
            op.error(method="commutator"),
            op.error(method="commutator", fast=False),
        ):
            assert isinstance(computed_error, SpectralNormError)
            assert qnp.isclose(computed_error.error, expected_error)

    @pytest.mark.all_interfaces
    @pytest.mark.parametrize(
        "method, expected_error", (("one-norm", 0.001265625), ("commutator", 0.001))
    )
    @pytest.mark.parametrize("interface", ("autograd", "jax", "torch"))
    def test_error_interfaces(self, method, interface, expected_error):
        """Test that the error method works with all interfaces"""

        time = qml.math.array(0.1, like=interface)
        coeffs = qml.math.array([1.0, 0.5], like=interface)

        hamiltonian = qml.dot(coeffs, [qml.X(0), qml.Y(0)])

        op = qml.TrotterProduct(hamiltonian, time, n=2, order=2)
        computed_error = op.error(method=method)

        assert isinstance(computed_error, SpectralNormError)
        assert qml.math.get_interface(computed_error.error) == interface

        assert qnp.isclose(computed_error.error, qml.math.array(expected_error, like=interface))

    @pytest.mark.tf
    def test_tensorflow_interface(self):
        """Test that an error is raised if a TrotterProduct with
        tensorflow parameters is used to compute error."""

        coeffs = qml.math.array([1.0, 0.5], like="tensorflow")
        hamiltonian = qml.dot(coeffs, [qml.X(0), qml.Y(0)])

        op = qml.TrotterProduct(hamiltonian, 1.23, order=2, n=5)
        with pytest.raises(TypeError, match="Calculating error bound for Tensorflow objects"):
            _ = op.error()


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
            qml.simplify(op) for op in test_decompositions[(hamiltonian_index, order)][::-1]
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

    #   Circuit execution tests:
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
                lambda x, y: x @ y, [qml.matrix(op, wire_order=wires) for op in true_decomp[::-1]]
            )
            @ initial_state
        )
        state = circ()
        assert qnp.allclose(expected_state, state)

    @pytest.mark.jax
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    def test_jax_execute(self, time):
        """Test that the gate executes correctly in the jax interface."""
        from jax import numpy as jnp

        time = jnp.array(time)
        coeffs = jnp.array([1.23, -0.45])
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.dot(coeffs, terms)
            qml.TrotterProduct(h, time, n=2, order=2)
            return qml.state()

        initial_state = jnp.array([1.0, 0.0, 0.0, 0.0])

        expected_product_sequence = _generate_simple_decomp(coeffs, terms, time, order=2, n=2)

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=range(2)) for op in expected_product_sequence],
            )
            @ initial_state
        )

        state = circ(time, coeffs)
        assert allclose(expected_state, state)

    @pytest.mark.jax
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    def test_jaxjit_execute(self, time):
        """Test that the gate executes correctly in the jax interface with jit."""
        import jax
        from jax import numpy as jnp

        time = jnp.array(time)
        c1 = jnp.array(1.23)
        c2 = jnp.array(-0.45)
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("default.qubit", wires=2)

        @jax.jit
        @qml.qnode(dev, interface="jax")
        def circ(time, c1, c2):
            h = qml.sum(
                qml.s_prod(c1, terms[0]),
                qml.s_prod(c2, terms[1]),
            )
            qml.TrotterProduct(h, time, n=2, order=2, check_hermitian=False)
            return qml.state()

        initial_state = jnp.array([1.0, 0.0, 0.0, 0.0])

        expected_product_sequence = _generate_simple_decomp([c1, c2], terms, time, order=2, n=2)

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=range(2)) for op in expected_product_sequence],
            )
            @ initial_state
        )

        state = circ(time, c1, c2)
        assert allclose(expected_state, state)

    @pytest.mark.tf
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    def test_tf_execute(self, time):
        """Test that the gate executes correctly in the tensorflow interface."""
        import tensorflow as tf

        time = tf.Variable(time, dtype=tf.complex128)
        coeffs = tf.Variable([1.23, -0.45], dtype=tf.complex128)
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.sum(
                qml.s_prod(coeffs[0], terms[0]),
                qml.s_prod(coeffs[1], terms[1]),
            )
            qml.TrotterProduct(h, time, n=2, order=2)

            return qml.state()

        initial_state = tf.Variable([1.0, 0.0, 0.0, 0.0], dtype=tf.complex128)

        expected_product_sequence = _generate_simple_decomp(coeffs, terms, time, order=2, n=2)

        expected_state = tf.linalg.matvec(
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=range(2)) for op in expected_product_sequence],
            ),
            initial_state,
        )

        state = circ(time, coeffs)
        assert allclose(expected_state, state)

    @pytest.mark.torch
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    def test_torch_execute(self, time):
        """Test that the gate executes correctly in the torch interface."""
        import torch

        time = torch.tensor(time, dtype=torch.complex64, requires_grad=True)
        coeffs = torch.tensor([1.23, -0.45], dtype=torch.complex64, requires_grad=True)
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.dot(coeffs, terms)
            qml.TrotterProduct(h, time, n=2, order=2)
            return qml.state()

        initial_state = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.complex64)

        expected_product_sequence = _generate_simple_decomp(coeffs, terms, time, order=2, n=2)

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=range(2)) for op in expected_product_sequence],
            )
            @ initial_state
        )

        state = circ(time, coeffs)
        assert allclose(expected_state, state)

    @pytest.mark.autograd
    @pytest.mark.parametrize("time", (0.5, 1, 2))
    def test_autograd_execute(self, time):
        """Test that the gate executes correctly in the autograd interface."""
        time = qnp.array(time)
        coeffs = qnp.array([1.23, -0.45])
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.dot(coeffs, terms)
            qml.TrotterProduct(h, time, n=2, order=2)
            return qml.state()

        initial_state = qnp.array([1.0, 0.0, 0.0, 0.0])

        expected_product_sequence = _generate_simple_decomp(coeffs, terms, time, order=2, n=2)

        expected_state = (
            reduce(
                lambda x, y: x @ y,
                [qml.matrix(op, wire_order=range(2)) for op in expected_product_sequence],
            )
            @ initial_state
        )

        state = circ(time, coeffs)
        assert qnp.allclose(expected_state, state)

    @pytest.mark.autograd
    @pytest.mark.parametrize("order, n", ((1, 1), (1, 2), (2, 1), (4, 1)))
    def test_autograd_gradient(self, order, n):
        """Test that the gradient is computed correctly"""
        time = qnp.array(1.5)
        coeffs = qnp.array([1.23, -0.45])
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.dot(coeffs, terms)
            qml.TrotterProduct(h, time, n=n, order=order)
            return qml.expval(qml.Hadamard(0))

        @qml.qnode(dev)
        def reference_circ(time, coeffs):
            with qml.QueuingManager.stop_recording():
                decomp = _generate_simple_decomp(coeffs, terms, time, order, n)

            for op in decomp[::-1]:
                qml.apply(op)

            return qml.expval(qml.Hadamard(0))

        measured_time_grad, measured_coeff_grad = qml.grad(circ)(time, coeffs)
        reference_time_grad, reference_coeff_grad = qml.grad(reference_circ)(time, coeffs)
        assert allclose(measured_time_grad, reference_time_grad)
        assert allclose(measured_coeff_grad, reference_coeff_grad)

    @pytest.mark.torch
    @pytest.mark.parametrize("order, n", ((1, 1), (1, 2), (2, 1), (4, 1)))
    def test_torch_gradient(self, order, n):
        """Test that the gradient is computed correctly using torch"""
        import torch

        time = torch.tensor(1.5, dtype=torch.complex64, requires_grad=True)
        coeffs = torch.tensor([1.23, -0.45], dtype=torch.complex64, requires_grad=True)
        time_reference = torch.tensor(1.5, dtype=torch.complex64, requires_grad=True)
        coeffs_reference = torch.tensor([1.23, -0.45], dtype=torch.complex64, requires_grad=True)
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.dot(coeffs, terms)
            qml.TrotterProduct(h, time, n=n, order=order)
            return qml.expval(qml.Hadamard(0))

        @qml.qnode(dev)
        def reference_circ(time, coeffs):
            with qml.QueuingManager.stop_recording():
                decomp = _generate_simple_decomp(coeffs, terms, time, order, n)

            for op in decomp[::-1]:
                qml.apply(op)

            return qml.expval(qml.Hadamard(0))

        res_circ = circ(time, coeffs)
        res_circ.backward()
        measured_time_grad = time.grad
        measured_coeff_grad = coeffs.grad

        ref_circ = reference_circ(time_reference, coeffs_reference)
        ref_circ.backward()
        reference_time_grad = time_reference.grad
        reference_coeff_grad = coeffs_reference.grad

        assert allclose(measured_time_grad, reference_time_grad)
        assert allclose(measured_coeff_grad, reference_coeff_grad)

    @pytest.mark.tf
    @pytest.mark.parametrize("order, n", ((1, 1), (1, 2), (2, 1), (4, 1)))
    def test_tf_gradient(self, order, n):
        """Test that the gradient is computed correctly using tensorflow"""
        import tensorflow as tf

        time = tf.Variable(1.5, dtype=tf.complex128)
        coeffs = tf.Variable([1.23, -0.45], dtype=tf.complex128)
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.sum(
                qml.s_prod(coeffs[0], terms[0]),
                qml.s_prod(coeffs[1], terms[1]),
            )
            qml.TrotterProduct(h, time, n=n, order=order)
            return qml.expval(qml.Hadamard(0))

        @qml.qnode(dev)
        def reference_circ(time, coeffs):
            with qml.QueuingManager.stop_recording():
                decomp = _generate_simple_decomp(coeffs, terms, time, order, n)

            for op in decomp[::-1]:
                qml.apply(op)

            return qml.expval(qml.Hadamard(0))

        with tf.GradientTape() as tape:
            result = circ(time, coeffs)

        measured_time_grad, measured_coeff_grad = tape.gradient(result, (time, coeffs))

        with tf.GradientTape() as tape:
            result = reference_circ(time, coeffs)

        reference_time_grad, reference_coeff_grad = tape.gradient(result, (time, coeffs))
        assert allclose(measured_time_grad, reference_time_grad)
        assert allclose(measured_coeff_grad, reference_coeff_grad)

    @pytest.mark.jax
    @pytest.mark.parametrize("order, n", ((1, 1), (1, 2), (2, 1), (4, 1)))
    def test_jax_gradient(self, order, n):
        """Test that the gradient is computed correctly"""
        import jax
        from jax import numpy as jnp

        time = jnp.array(1.5)
        coeffs = jnp.array([1.23, -0.45])
        terms = [qml.PauliX(0), qml.PauliZ(0)]

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circ(time, coeffs):
            h = qml.dot(coeffs, terms)
            qml.TrotterProduct(h, time, n=n, order=order)
            return qml.expval(qml.Hadamard(0))

        @qml.qnode(dev)
        def reference_circ(time, coeffs):
            with qml.QueuingManager.stop_recording():
                decomp = _generate_simple_decomp(coeffs, terms, time, order, n)

            for op in decomp[::-1]:
                qml.apply(op)

            return qml.expval(qml.Hadamard(0))

        measured_time_grad, measured_coeff_grad = jax.grad(circ, argnums=[0, 1])(time, coeffs)
        reference_time_grad, reference_coeff_grad = jax.grad(reference_circ, argnums=[0, 1])(
            time, coeffs
        )
        assert allclose(measured_time_grad, reference_time_grad)
        assert allclose(measured_coeff_grad, reference_coeff_grad)
