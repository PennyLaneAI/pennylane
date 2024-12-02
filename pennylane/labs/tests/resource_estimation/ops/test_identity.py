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
Tests for identity resource operators
"""
import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use,use-implicit-booleaness-not-comparison


class TestIdentity:
    """Test ResourceIdentity"""

    def test_resources(self):
        """ResourceIdentity should have empty resources"""
        op = re.ResourceIdentity()
        assert op.resources() == {}

    def test_resource_rep(self):
        """Test the compressed representation"""
        expected = re.CompressedResourceOp(re.ResourceIdentity, {})
        assert re.ResourceIdentity.resource_rep() == expected

    def test_resource_params(self):
        """Test the resource params are correct"""
        op = re.ResourceIdentity(0)
        assert op.resource_params() == {}

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""
        op = re.ResourceIdentity()
        expected = {}

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected


class TestGlobalPhase:
    """Test ResourceGlobalPhase"""

    def test_resources(self):
        """ResourceGlobalPhase should have empty resources"""
        op = re.ResourceGlobalPhase(0.1, wires=0)
        assert op.resources() == {}

    def test_resource_rep(self):
        """Test the compressed representation"""
        expected = re.CompressedResourceOp(re.ResourceGlobalPhase, {})
        assert re.ResourceGlobalPhase.resource_rep() == expected

    def test_resource_params(self):
        """Test the resource params are correct"""
        op = re.ResourceGlobalPhase(0.1, wires=0)
        assert op.resource_params() == {}

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""
        op = re.ResourceGlobalPhase(0.1, wires=0)
        expected = {}

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resources(**op_resource_params) == expected
