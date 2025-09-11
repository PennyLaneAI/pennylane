# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for non parametric resource operators."""
import pytest

from pennylane.estimator.ops import SWAP, Hadamard, Identity, S, T, X, Y, Z
from pennylane.estimator.resource_operator import (
    CompressedResourceOp,
    GateCount,
    ResourcesNotDefined,
)

# pylint: disable=no-self-use,use-implicit-booleaness-not-comparison


class TestHadamard:
    """Tests for Hadamard resource operator"""

    def test_resources(self):
        """Test that Hadamard resource operator does not implement a decomposition"""
        op = Hadamard()
        with pytest.raises(ResourcesNotDefined):
            op.resource_decomp()

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = Hadamard()
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = CompressedResourceOp(Hadamard, 1, {})
        assert Hadamard.resource_rep() == expected

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct."""
        h = Hadamard()
        h_dag = h.adjoint_resource_decomp()

        expected = [GateCount(Hadamard.resource_rep(), 1)]
        assert h_dag == expected

    ctrl_data = (
        (
            ["c1"],
            [1],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = Hadamard(0)
        with pytest.raises(ResourcesNotDefined):
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = (
        (1, [GateCount(Hadamard.resource_rep(), 1)]),
        (2, [GateCount(Identity.resource_rep(), 1)]),
        (3, [GateCount(Hadamard.resource_rep(), 1)]),
        (4, [GateCount(Identity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = Hadamard(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestSWAP:
    """Tests for SWAP resource operator"""

    def test_resources(self):
        """Test that SWAP decomposes into three CNOTs"""
        op = SWAP([0, 1])
        with pytest.raises(ResourcesNotDefined):
            op.resource_decomp()

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = SWAP([0, 1])
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test the compact representation"""
        expected = CompressedResourceOp(SWAP, 2, {})
        assert SWAP.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""

        op = SWAP([0, 1])
        expected = CompressedResourceOp(SWAP, 2, {})
        assert op.resource_rep_from_op() == expected

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct."""
        swap = SWAP([0, 1])
        swap_dag = swap.adjoint_resource_decomp()
        expected = [GateCount(SWAP.resource_rep(), 1)]
        assert swap_dag == expected

    ctrl_data = (
        (
            ["c1"],
            [1],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = SWAP([0, 1])
        with pytest.raises(ResourcesNotDefined):
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = (
        (1, [GateCount(SWAP.resource_rep(), 1)]),
        (2, [GateCount(Identity.resource_rep(), 1)]),
        (3, [GateCount(SWAP.resource_rep(), 1)]),
        (4, [GateCount(Identity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = SWAP([0, 1])
        assert op.pow_resource_decomp(z) == expected_res


class TestS:
    """Tests for S resource operator"""

    def test_resources(self):
        """Test that S decomposes into two Ts"""
        op = S(0)
        expected = [GateCount(T.resource_rep(), 2)]
        assert op.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = S(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct"""
        expected = CompressedResourceOp(S, 1, {})
        assert S.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""
        op = S(0)
        expected = [GateCount(T.resource_rep(), 2)]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [
            GateCount(Z.resource_rep(), 1),
            GateCount(S.resource_rep(), 1),
        ]
        assert S.adjoint_resource_decomp() == expected

    ctrl_data = (
        (
            ["c1"],
            [1],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = S(0)
        with pytest.raises(ResourcesNotDefined):
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = (
        (1, [GateCount(S.resource_rep(), 1)]),
        (2, [GateCount(Z.resource_rep(), 1)]),
        (
            3,
            [
                GateCount(Z.resource_rep(), 1),
                GateCount(S.resource_rep(), 1),
            ],
        ),
        (4, [GateCount(Identity.resource_rep(), 1)]),
        (
            7,
            [
                GateCount(Z.resource_rep(), 1),
                GateCount(S.resource_rep(), 1),
            ],
        ),
        (8, [GateCount(Identity.resource_rep(), 1)]),
        (14, [GateCount(Z.resource_rep(), 1)]),
        (
            15,
            [
                GateCount(Z.resource_rep(), 1),
                GateCount(S.resource_rep(), 1),
            ],
        ),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = S(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestT:
    """Tests for T resource operator"""

    def test_resources(self):
        """Test that there is no further decomposition of the T gate."""
        op = T(0)
        with pytest.raises(ResourcesNotDefined):
            op.resource_decomp()

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = T(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = CompressedResourceOp(T, 1, {})
        assert T.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [
            GateCount(T.resource_rep(), 1),
            GateCount(S.resource_rep(), 1),
            GateCount(Z.resource_rep(), 1),
        ]
        assert T.adjoint_resource_decomp() == expected

    ctrl_data = ((["c1"], [1]),)

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = T(0)
        with pytest.raises(ResourcesNotDefined):
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = (
        (1, [GateCount(T.resource_rep(), 1)]),
        (2, [GateCount(S.resource_rep(), 1)]),
        (
            3,
            [
                GateCount(S.resource_rep(), 1),
                GateCount(T.resource_rep(), 1),
            ],
        ),
        (
            7,
            [
                GateCount(Z.resource_rep(), 1),
                GateCount(S.resource_rep(), 1),
                GateCount(T.resource_rep(), 1),
            ],
        ),
        (8, [GateCount(Identity.resource_rep(), 1)]),
        (
            14,
            [
                GateCount(Z.resource_rep(), 1),
                GateCount(S.resource_rep(), 1),
            ],
        ),
        (
            15,
            [
                GateCount(Z.resource_rep(), 1),
                GateCount(S.resource_rep(), 1),
                GateCount(T.resource_rep(), 1),
            ],
        ),
        (16, [GateCount(Identity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = T(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestX:
    """Tests for the X resource operator gate"""

    def test_resources(self):
        """Tests for the X resource operator gate"""
        expected = [
            GateCount(Hadamard.resource_rep(), 2),
            GateCount(S.resource_rep(), 2),
        ]
        assert X.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = X(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = CompressedResourceOp(X, 1, {})
        assert X.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [GateCount(X.resource_rep(), 1)]
        assert X.adjoint_resource_decomp() == expected

    ctrl_data = ((["c1"], [1]),)

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = X(0)
        with pytest.raises(ResourcesNotDefined):
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = (
        (1, [GateCount(X.resource_rep(), 1)]),
        (2, [GateCount(Identity.resource_rep(), 1)]),
        (3, [GateCount(X.resource_rep(), 1)]),
        (4, [GateCount(Identity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = X(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestY:
    """Tests for the resource Y gate"""

    def test_resources(self):
        """Test that T does not implement a decomposition"""
        expected = [
            GateCount(S.resource_rep(), 2),
            GateCount(Z.resource_rep(), 2),
            GateCount(Hadamard.resource_rep(), 2),
        ]
        assert Y.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = Y(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = CompressedResourceOp(Y, 1, {})
        assert Y.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [GateCount(Y.resource_rep(), 1)]
        assert Y.adjoint_resource_decomp() == expected

    ctrl_data = ((["c1"], [1]),)

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = Y(0)
        with pytest.raises(ResourcesNotDefined):
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = (
        (1, [GateCount(Y.resource_rep())]),
        (2, [GateCount(Identity.resource_rep())]),
        (3, [GateCount(Y.resource_rep())]),
        (4, [GateCount(Identity.resource_rep())]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = Y(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestZ:
    """Tests for the Z resource operator gate"""

    def test_resources(self):
        """Test that Z resource operator implements the correct decomposition"""
        expected = [GateCount(S.resource_rep(), 2)]
        assert Z.resource_decomp() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = Z(0)
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = CompressedResourceOp(Z, 1, {})
        assert Z.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = [GateCount(Z.resource_rep(), 1)]
        assert Z.adjoint_resource_decomp() == expected

    ctrl_data = ((["c1"], [1]),)

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = Z(0)
        with pytest.raises(ResourcesNotDefined):
            op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)

    pow_data = (
        (1, [GateCount(Z.resource_rep())]),
        (2, [GateCount(Identity.resource_rep())]),
        (3, [GateCount(Z.resource_rep())]),
        (4, [GateCount(Identity.resource_rep())]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = Z(0)
        assert op.pow_resource_decomp(z) == expected_res
