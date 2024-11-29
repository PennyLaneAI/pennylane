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
        controlled_cp = controlled_op.resource_rep_from_op()

        with pytest.raises(re.ResourcesNotDefined):
            controlled_cp.op_type.resources(**controlled_cp.params)

    def test_double_controlled_resources(self):
        """Test that the controlled_resource_decomp method dispatches correctly."""
        controlled_op = re.ResourceControlled(re.ResourceX(0), control_wires=[1, 2])
        expected_op = re.ResourceToffoli([0, 1, 2])

        controlled_cp = controlled_op.resource_rep_from_op()
        controlled_resources = controlled_cp.op_type.resources(**controlled_cp.params)

        expected_cp = expected_op.resource_rep_from_op()
        expected_resources = expected_cp.op_type.resources(**expected_cp.params)

        assert controlled_resources == expected_resources
