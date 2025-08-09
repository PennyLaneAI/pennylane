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
"""Unit tests for the Identity Operator."""
import numpy as np
import pytest

import pennylane as qml
from pennylane import Identity

op_wires = [[], [0], ["a"], [0, 1], ["a", "b", "c"], [100, "xasd", 12]]
op_repr = ["I()", "I(0)", "I('a')", "I([0, 1])", "I(['a', 'b', 'c'])", "I([100, 'xasd', 12])"]
op_params = tuple(zip(op_wires, op_repr))


@pytest.mark.parametrize("wires", op_wires)
class TestIdentity:
    # pylint: disable=protected-access
    def test_flatten_unflatten(self, wires):
        """Test the flatten and unflatten methods of identity."""
        op = Identity(wires)
        data, metadata = op._flatten()
        assert data == tuple()
        assert metadata[0] == qml.wires.Wires(wires)
        assert metadata[1] == tuple()

        new_op = Identity._unflatten(*op._flatten())
        qml.assert_equal(op, new_op)

    def test_class_name(self, wires):
        """Test the class name of either I and Identity is by default 'Identity'"""
        assert qml.I.__name__ == "Identity"
        assert qml.Identity.__name__ == "Identity"

        assert qml.I(wires).name == "Identity"
        assert qml.Identity(wires).name == "Identity"

    @pytest.mark.jax
    def test_jax_pytree_integration(self, wires):
        """Test that identity is a pytree by jitting a function of it."""
        import jax

        op = qml.Identity(wires)

        adj_op = jax.jit(lambda op: qml.adjoint(op, lazy=False))(op)

        qml.assert_equal(op, adj_op)

    def test_identity_eigvals(self, wires, tol):
        """Test identity eigenvalues are correct"""
        res = Identity(wires).eigvals()
        expected = np.ones(2 ** len(wires))
        assert np.allclose(res, expected, atol=tol, rtol=0)

    def test_decomposition(self, wires):
        """Test the decomposition of the identity operation."""

        assert Identity.compute_decomposition(wires=wires) == []
        assert Identity(wires=wires).decomposition() == []

    def test_label_method(self, wires):
        """Test the label method for the Identity Operator"""
        assert Identity(wires=wires).label() == "I"

    @pytest.mark.parametrize("n", (2, -3, 3.455, -1.29))
    def test_identity_pow(self, wires, n):
        """Test that the identity raised to any power is simply a single copy."""
        op = Identity(wires)
        pow_ops = op.pow(n)
        assert len(pow_ops) == 1
        assert pow_ops[0].__class__ is Identity
        assert pow_ops[0].wires == op.wires

    def test_matrix_representation(self, wires, tol):
        """Test the matrix representation"""
        res_static = Identity.compute_matrix(n_wires=len(wires))
        res_dynamic = Identity(wires=wires).matrix()
        expected = np.eye(int(2 ** len(wires)))
        assert np.allclose(res_static, expected, atol=tol)
        assert np.allclose(res_dynamic, expected, atol=tol)

    def test_sparse_matrix_format(self, wires):
        from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, lil_matrix

        op = qml.Identity(wires=wires)
        assert isinstance(op.sparse_matrix(), csr_matrix)
        assert isinstance(op.sparse_matrix(format="csc"), csc_matrix)
        assert isinstance(op.sparse_matrix(format="lil"), lil_matrix)
        assert isinstance(op.sparse_matrix(format="coo"), coo_matrix)
        assert qml.math.allclose(op.matrix(), op.sparse_matrix().toarray())


@pytest.mark.parametrize("wires, expected_repr", op_params)
def test_repr(wires, expected_repr):
    """Test the operator's repr"""
    op = Identity(wires=wires)
    assert repr(op) == expected_repr
