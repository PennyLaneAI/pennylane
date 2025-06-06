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
"""
Tests for identity resource operators
"""
import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use,use-implicit-booleaness-not-comparison


class TestIdentity:
    """Test ResourceIdentity"""

    def test_resources(self):
        """ResourceIdentity should have empty resources"""
        op = re.ResourceIdentity()
        assert op.resource_decomp() == []

    def test_resource_rep(self):
        """Test the compressed representation"""
        expected = re.CompressedResourceOp(re.ResourceIdentity, {})
        assert re.ResourceIdentity.resource_rep() == expected

    def test_resource_params(self):
        """Test the resource params are correct"""
        op = re.ResourceIdentity(0)
        assert op.resource_params == {}

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""
        op = re.ResourceIdentity()
        expected = []

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        op = re.ResourceIdentity(0)
        assert op.adjoint_resource_decomp() == [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]

    identity_ctrl_data = (
        ([1], [1], [], {re.ResourceIdentity.resource_rep(): 1}),
        ([1, 2], [1, 1], ["w1"], {re.ResourceIdentity.resource_rep(): 1}),
        ([1, 2, 3], [1, 0, 0], ["w1", "w2"], {re.ResourceIdentity.resource_rep(): 1}),
    )

    identity_pow_data = (
        (1, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
        (2, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
        (5, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", identity_pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = re.ResourceIdentity(0)
        assert op.pow_resource_decomp(z) == expected_res

        # op2 = re.ResourcePow(op, z)
        # assert op2.resources(**op2.resource_params) == expected_res


class TestGlobalPhase:
    """Test ResourceGlobalPhase"""

    def test_resources(self):
        """ResourceGlobalPhase should have empty resources"""
        op = re.ResourceGlobalPhase(wires=0)
        assert op.resource_decomp() == []

    def test_resource_rep(self):
        """Test the compressed representation"""
        expected = re.CompressedResourceOp(re.ResourceGlobalPhase, {})
        assert re.ResourceGlobalPhase.resource_rep() == expected

    def test_resource_params(self):
        """Test the resource params are correct"""
        op = re.ResourceGlobalPhase()
        assert op.resource_params == {}

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""
        op = re.ResourceGlobalPhase(wires=0)
        expected = []

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        op = re.ResourceGlobalPhase(wires=0)
        assert op.adjoint_resource_decomp() == [
            re.GateCount(re.ResourceGlobalPhase.resource_rep(), 1)
        ]

    globalphase_pow_data = (
        (1, [re.GateCount(re.ResourceGlobalPhase.resource_rep(), 1)]),
        (2, [re.GateCount(re.ResourceGlobalPhase.resource_rep(), 1)]),
        (5, [re.GateCount(re.ResourceGlobalPhase.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", globalphase_pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = re.ResourceGlobalPhase()
        assert op.pow_resource_decomp(z) == expected_res
