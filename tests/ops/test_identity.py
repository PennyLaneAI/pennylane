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

from pennylane import Identity


@pytest.mark.parametrize("wires", [[0], [0, 1], ["a", "b", "c"], [100, "xasd", 12]])
class TestIdentity:
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
