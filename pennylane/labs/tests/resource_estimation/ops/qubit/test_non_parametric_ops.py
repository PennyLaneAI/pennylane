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
"""
Tests for non parametric resource operators.
"""
import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use,use-implicit-booleaness-not-comparison


class TestHadamard:
    """Tests for ResourceHadamard"""

    def test_resources(self):
        """Test that ResourceHadamard does not implement a decomposition"""
        op = re.ResourceHadamard(0)
        with pytest.raises(re.ResourcesNotDefined):
            op.resources()

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceHadamard(0)
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = re.CompressedResourceOp(re.ResourceHadamard, {})
        assert re.ResourceHadamard.resource_rep() == expected

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct."""
        h = re.ResourceHadamard(0)
        h_dag = re.ResourceAdjoint(re.ResourceHadamard(0))

        assert re.get_resources(h) == re.get_resources(h_dag)

    def test_controlled_decomp(self):
        """Test that the controlled decomposition is correct."""
        expected = {re.ResourceCH.resource_rep(): 1}
        assert re.ResourceHadamard.controlled_resource_decomp(1, 0, 0) == expected

        controlled_h = re.ResourceControlled(re.ResourceHadamard(0), control_wires=[1])
        ch = re.ResourceCH([0, 1])

        r1 = re.get_resources(controlled_h)
        r2 = re.get_resources(ch)
        assert r1 == r2

    @pytest.mark.parametrize("z", list(range(10)))
    def test_pow_decomp(self, z):
        """Test that the pow decomposition is correct."""
        expected = {re.ResourceHadamard.resource_rep(): z % 2}
        assert re.ResourceHadamard.pow_resource_decomp(z) == expected

        h = re.ResourceHadamard(0)
        pow_h = re.ResourcePow(re.ResourceHadamard(0), z)

        r1 = re.get_resources(h) * (z % 2)
        r2 = re.get_resources(pow_h)

        assert r1 == r2


class TestSWAP:
    """Tests for ResourceSWAP"""

    def test_resources(self):
        """Test that SWAP decomposes into three CNOTs"""
        op = re.ResourceSWAP([0, 1])
        cnot = re.ResourceCNOT.resource_rep()
        expected = {cnot: 3}

        assert op.resources() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceSWAP([0, 1])
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test the compact representation"""
        expected = re.CompressedResourceOp(re.ResourceSWAP, {})
        assert re.ResourceSWAP.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""

        op = re.ResourceSWAP([0, 1])
        expected = {re.ResourceCNOT.resource_rep(): 3}

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct."""
        swap = re.ResourceSWAP([0, 1])
        swap_dag = re.ResourceAdjoint(re.ResourceSWAP([0, 1]))

        assert re.get_resources(swap) == re.get_resources(swap_dag)

    def test_controlled_decomp(self):
        """Test that the controlled decomposition is correct."""
        expected = {re.ResourceCSWAP.resource_rep(): 1}
        assert re.ResourceSWAP.controlled_resource_decomp(1, 0, 0) == expected

        controlled_swap = re.ResourceControlled(re.ResourceSWAP([0, 1]), control_wires=[2])
        cswap = re.ResourceCSWAP([0, 1, 2])

        r1 = re.get_resources(controlled_swap)
        r2 = re.get_resources(cswap)
        assert r1 == r2

    @pytest.mark.parametrize("z", list(range(10)))
    def test_pow_decomp(self, z):
        """Test that the pow decomposition is correct."""
        expected = {re.ResourceSWAP.resource_rep(): z % 2}
        assert re.ResourceSWAP.pow_resource_decomp(z) == expected

        swap = re.ResourceSWAP([0, 1])
        pow_swap = re.ResourcePow(re.ResourceSWAP([0, 1]), z)

        r1 = re.get_resources(swap) * (z % 2)
        r2 = re.get_resources(pow_swap)

        assert r1 == r2


class TestS:
    """Tests for ResourceS"""

    def test_resources(self):
        """Test that S decomposes into two Ts"""
        op = re.ResourceS(0)
        expected = {re.CompressedResourceOp(re.ResourceT, {}): 2}
        assert op.resources() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceS(0)
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compressed representation is correct"""
        expected = re.CompressedResourceOp(re.ResourceS, {})
        assert re.ResourceS.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""

        op = re.ResourceS(0)
        expected = {re.ResourceT.resource_rep(): 2}

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = {re.ResourceS.resource_rep(): 3}
        assert re.ResourceS.adjoint_resource_decomp() == expected

        s = re.ResourceS(0)
        s_dag = re.ResourceAdjoint(s)

        r1 = re.get_resources(s) * 3
        r2 = re.get_resources(s_dag)
        assert r1 == r2

    @pytest.mark.parametrize("z", list(range(10)))
    def test_pow_decomp(self, z):
        """Test that the pow decomposition is correct."""
        expected = {re.ResourceS.resource_rep(): z % 4}
        assert re.ResourceS.pow_resource_decomp(z) == expected

        s = re.ResourceS(0)
        pow_s = re.ResourcePow(s, z)

        r1 = re.get_resources(s) * (z % 4)
        r2 = re.get_resources(pow_s)

        assert r1 == r2


class TestT:
    """Tests for ResourceT"""

    def test_resources(self):
        """Test that ResourceT does not implement a decomposition"""
        op = re.ResourceT(0)
        with pytest.raises(re.ResourcesNotDefined):
            op.resources()

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceT(0)
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = re.CompressedResourceOp(re.ResourceT, {})
        assert re.ResourceT.resource_rep() == expected

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = {re.ResourceT.resource_rep(): 7}
        assert re.ResourceT.adjoint_resource_decomp() == expected

        t = re.ResourceT(0)
        t_dag = re.ResourceAdjoint(t)

        r1 = re.get_resources(t) * 7
        r2 = re.get_resources(t_dag)
        assert r1 == r2

    @pytest.mark.parametrize("z", list(range(10)))
    def test_pow_decomp(self, z):
        """Test that the pow decomposition is correct."""
        expected = {re.ResourceT.resource_rep(): z % 8}
        assert re.ResourceT.pow_resource_decomp(z) == expected

        t = re.ResourceT(0)
        pow_t = re.ResourcePow(t, z)

        r1 = re.get_resources(t) * (z % 8)
        r2 = re.get_resources(pow_t)

        assert r1 == r2


class TestX:
    """Tests for ResourceX"""

    def test_resources(self):
        """Test that ResourceT does not implement a decomposition"""
        expected = {
            re.ResourceS.resource_rep(): 2,
            re.ResourceHadamard.resource_rep(): 2,
        }
        assert re.ResourceX.resources() == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceX(0)
        assert op.resource_params() == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = re.CompressedResourceOp(re.ResourceX, {})
        assert re.ResourceX.resource_rep() == expected

    def test_single_controlled_resources(self):
        """Test that the controlled_resource_decomp method dispatches correctly."""
        controlled_op = re.ResourceControlled(re.ResourceX(0), control_wires=[1])

        cnot = re.ResourceCNOT([0, 1])
        assert re.get_resources(controlled_op) == re.get_resources(cnot)

    def test_double_controlled_resources(self):
        """Test that the controlled_resource_decomp method dispatches correctly."""
        controlled_op = re.ResourceControlled(re.ResourceX(0), control_wires=[1, 2])
        expected_op = re.ResourceToffoli([0, 1, 2])

        r1 = re.get_resources(controlled_op)
        r2 = re.get_resources(expected_op)

        assert r1 == r2

    def test_adjoint_decomposition(self):
        """Test that the adjoint resources are correct."""
        expected = {re.ResourceX.resource_rep(): 1}
        assert re.ResourceX.adjoint_resource_decomp() == expected

        x = re.ResourceX(0)
        x_dag = re.ResourceAdjoint(x)

        r1 = re.get_resources(x)
        r2 = re.get_resources(x_dag)
        assert r1 == r2

    @pytest.mark.parametrize("z", list(range(10)))
    def test_pow_decomp(self, z):
        """Test that the pow decomposition is correct."""
        expected = {re.ResourceX.resource_rep(): z % 2}
        assert re.ResourceX.pow_resource_decomp(z) == expected

        x = re.ResourceX(0)
        pow_x = re.ResourcePow(x, z)

        r1 = re.get_resources(x) * (z % 2)
        r2 = re.get_resources(pow_x)

        assert r1 == r2
