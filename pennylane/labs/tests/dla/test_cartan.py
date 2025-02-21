# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pennylane/dla/lie_closure_dense.py functionality"""
# pylint: disable=no-self-use,too-few-public-methods,missing-class-docstring
import pytest

import pennylane as qml
from pennylane import X, Y, Z
from pennylane.labs.dla import cartan_decomp, concurrence_involution, even_odd_involution


def check_commutation(ops1, ops2, vspace):
    """Helper function to check things like [k, m] subspace m; expensive"""
    for o1 in ops1:
        for o2 in ops2:
            com = o1.commutator(o2)
            assert not vspace.is_independent(com)

    return True


Ising2 = qml.lie_closure([X(0), X(1), Z(0) @ Z(1)])
Ising3 = qml.lie_closure([X(0), X(1), X(2), Z(0) @ Z(1), Z(1) @ Z(2)])
Heisenberg3 = qml.lie_closure(
    [X(0) @ X(1), X(1) @ X(2), Y(0) @ Y(1), Y(1) @ Y(2), Z(0) @ Z(1), Z(1) @ Z(2)]
)


class TestCartanDecomposition:
    @pytest.mark.parametrize("involution", [even_odd_involution, concurrence_involution])
    @pytest.mark.parametrize("g", [Ising2, Ising3, Heisenberg3])
    def test_cartan_decomp(self, g, involution):
        """Test basic properties and Cartan decomposition definitions"""

        g = [op.pauli_rep for op in g]
        k, m = cartan_decomp(g, involution)

        assert all(involution(op) is True for op in k)
        assert all(involution(op) is False for op in m)

        k_space = qml.pauli.PauliVSpace(k)
        m_space = qml.pauli.PauliVSpace(m)

        # Commutation relations for Cartan pair
        assert check_commutation(k, k, k_space)
        assert check_commutation(k, m, m_space)
        assert check_commutation(m, m, k_space)

    @pytest.mark.parametrize("involution", [even_odd_involution, concurrence_involution])
    @pytest.mark.parametrize("g", [Ising2, Ising3, Heisenberg3])
    def test_cartan_decomp_dense(self, g, involution):
        """Test basic properties and Cartan decomposition definitions using dense representations"""

        g = [qml.matrix(op, wire_order=range(3)) for op in g]
        k, m = cartan_decomp(g, involution)

        assert all(involution(op) is True for op in k)
        assert all(involution(op) is False for op in m)

        # check currently only works with pauli sentences
        k_space = qml.pauli.PauliVSpace([qml.pauli_decompose(op).pauli_rep for op in k])
        m_space = qml.pauli.PauliVSpace([qml.pauli_decompose(op).pauli_rep for op in m])

        # Commutation relations for Cartan pair
        assert check_commutation(k_space.basis, k_space.basis, k_space)
        assert check_commutation(k_space.basis, m_space.basis, m_space)
        assert check_commutation(m_space.basis, m_space.basis, k_space)


involution_ops = [
    X(0) @ X(1),
    X(0) @ X(1) + Y(0) @ Y(1),
    X(1) + Z(0),
    Y(0) - Y(0) @ Y(1) @ Y(2),
    Y(0) @ X(1),
]


class TestInvolutions:
    """Test involutions"""

    @pytest.mark.parametrize("op", involution_ops)
    def test_concurrence_involution_inputs(self, op):
        """Test different input types yield consistent results"""
        res_op = concurrence_involution(op)
        res_ps = concurrence_involution(op.pauli_rep)
        res_m = concurrence_involution(op.matrix())

        assert isinstance(res_op, bool)
        assert res_op is res_ps
        assert res_op is res_m

    @pytest.mark.parametrize("op", involution_ops)
    def test_even_odd_involution_inputs(self, op):
        """Test different input types yield consistent results"""
        res_op = even_odd_involution(op)
        res_ps = even_odd_involution(op.pauli_rep)
        res_m = even_odd_involution(op.matrix())

        assert isinstance(res_op, bool)
        assert res_op is res_ps
        assert res_op is res_m
