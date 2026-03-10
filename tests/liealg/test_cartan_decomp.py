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
"""Tests for pennylane/labs/dla/cartan.py functionality"""
# pylint: disable=no-self-use,too-few-public-methods,missing-class-docstring
import pytest

import pennylane as qml
from pennylane import X, Y, Z
from pennylane.liealg import (
    cartan_decomp,
    check_cartan_decomp,
    check_commutation_relation,
    concurrence_involution,
    even_odd_involution,
)

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
        assert check_commutation_relation(k, k, k_space)
        assert check_commutation_relation(k, m, m_space)
        assert check_commutation_relation(m, m, k_space)

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
        assert check_commutation_relation(k_space.basis, k_space.basis, k_space)
        assert check_commutation_relation(k_space.basis, m_space.basis, m_space)
        assert check_commutation_relation(m_space.basis, m_space.basis, k_space)


involution_ops = [
    X(0) @ X(1),
    X(0) @ X(1) + Y(0) @ Y(1),
    X(1) + Z(0),
    Y(0) - Y(0) @ Y(1) @ Y(2),
    Y(0) @ X(1),
]

k0 = [Z(0) @ Y(1), Y(0) @ Z(1)]
m0 = [Z(0) @ Z(1), Y(0) @ Y(1), X(0), X(1)]
k0_m = [qml.matrix(op, wire_order=range(2)) for op in k0]
m0_m = [qml.matrix(op, wire_order=range(2)) for op in m0]


class TestCheckFunctions:
    """Test check functions for cartan decompositions"""

    def test_check_cartan_decomp_mixed_inputs_raises_TypeError(self):
        """Test that mixing operators and matrices raises an error in check_cartan_decomp"""
        with pytest.raises(TypeError, match=r"All inputs `k`, `m`"):
            _ = check_cartan_decomp(m0, k0_m)

        with pytest.raises(TypeError, match=r"All inputs `k`, `m`"):
            _ = check_cartan_decomp(m0, qml.numpy.array(k0_m))

        with pytest.raises(TypeError, match=r"All inputs `k`, `m`"):
            _ = check_cartan_decomp(m0_m, k0)

        with pytest.raises(TypeError, match=r"All inputs `k`, `m`"):
            _ = check_cartan_decomp(qml.numpy.array(m0_m), k0)

    def test_check_commutation_relation_mixed_inputs_raises_TypeError(self):
        """Test that mixing operators and matrices raises an error in check_cartan_decomp"""
        with pytest.raises(TypeError, match=r"All inputs `ops1`, `ops2`"):
            _ = check_commutation_relation(m0, k0_m, m0_m)

        with pytest.raises(TypeError, match=r"All inputs `ops1`, `ops2`"):
            _ = check_commutation_relation(m0_m, k0, m0_m)

        with pytest.raises(TypeError, match=r"All inputs `ops1`, `ops2`"):
            _ = check_commutation_relation(m0_m, k0_m, m0)

    def test_check_cartan_decomp(self):
        """Test that check_cartan_decomp correctly checks Ising cartan decomp
        from fdhs paper (https://arxiv.org/abs/2104.00728)"""

        assert check_cartan_decomp(k0, m0)

    def test_check_cartan_decomp_arrays(self):
        """Test that check_cartan_decomp correctly checks Ising cartan decomp
        from fdhs paper (https://arxiv.org/abs/2104.00728) when using matrix inputs."""

        assert check_cartan_decomp(k0_m, m0_m)

    def test_check_commutation_relation(self):
        """Test that check_commutation_relation returns false correctly"""

        assert check_commutation_relation(k0, k0, k0)
        assert not check_commutation_relation(m0, m0, m0)
        assert check_commutation_relation(k0, m0, m0)
        assert not check_commutation_relation(k0, m0, k0)
        assert check_commutation_relation(m0, k0, m0)
        assert not check_commutation_relation(m0, k0, k0)

    def test_check_commutation_relation_matrix(self):
        """Test that check_commutation_relation returns false correctly when using matrix inputs"""

        assert check_commutation_relation(k0_m, k0_m, k0_m)
        assert not check_commutation_relation(m0_m, m0_m, m0_m)
        assert check_commutation_relation(k0_m, m0_m, m0_m)
        assert not check_commutation_relation(k0_m, m0_m, k0_m)
        assert check_commutation_relation(m0_m, k0_m, m0_m)
        assert not check_commutation_relation(m0_m, k0_m, k0_m)

    def test_check_cartan_decomp_verbose(self, capsys):
        """Test the verbose output of check_cartan_decomp"""
        _ = check_cartan_decomp(k=m0, m=m0, verbose=True)
        captured = capsys.readouterr()
        assert "[k, k] sub k not fulfilled" in captured.out
        assert "[k, m] sub m not fulfilled" in captured.out
        assert "[m, m] sub k not fulfilled" in captured.out

    def test_check_cartan_decomp_verbose_matrix(self, capsys):
        """Test the verbose output of check_cartan_decomp"""
        _ = check_cartan_decomp(k=m0_m, m=m0_m, verbose=True)
        captured = capsys.readouterr()
        assert "[k, k] sub k not fulfilled" in captured.out
        assert "[k, m] sub m not fulfilled" in captured.out
        assert "[m, m] sub k not fulfilled" in captured.out


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
