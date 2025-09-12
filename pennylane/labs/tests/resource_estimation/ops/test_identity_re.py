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

import pennylane.labs.resource_estimation as plre

# pylint: disable=no-self-use,use-implicit-booleaness-not-comparison


class TestIdentity:
    """Test ResourceIdentity"""

    def test_resources(self):
        """ResourceIdentity should have empty resources"""
        op = plre.ResourceIdentity()
        assert op.resource_decomp() == []

    def test_resource_rep(self):
        """Test the compressed representation"""
        expected = plre.CompressedResourceOp(plre.ResourceIdentity, 1, {})
        assert plre.ResourceIdentity.resource_rep() == expected

    def test_resource_params(self):
        """Test the resource params are correct"""
        op = plre.ResourceIdentity(0)
        assert op.resource_params == {}

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""
        op = plre.ResourceIdentity()
        expected = []

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        op = plre.ResourceIdentity(0)
        assert op.adjoint_resource_decomp() == [
            plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)
        ]

    identity_ctrl_data = (
        ([1], [1], [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
        ([1, 2], [1, 1], [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
        ([1, 2, 3], [1, 0, 0], [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("ctrl_wires, ctrl_values, expected_res", identity_ctrl_data)
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = plre.ResourceIdentity(0)
        op2 = plre.ResourceControlled(op, num_ctrl_wires, num_ctrl_values)

        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == expected_res
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    identity_pow_data = (
        (1, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
        (2, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
        (5, [plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", identity_pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = plre.ResourceIdentity(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestGlobalPhase:
    """Test ResourceGlobalPhase"""

    def test_resources(self):
        """ResourceGlobalPhase should have empty resources"""
        op = plre.ResourceGlobalPhase(wires=0)
        assert op.resource_decomp() == []

    def test_resource_rep(self):
        """Test the compressed representation"""
        expected = plre.CompressedResourceOp(plre.ResourceGlobalPhase, 1, {})
        assert plre.ResourceGlobalPhase.resource_rep() == expected

    def test_resource_params(self):
        """Test the resource params are correct"""
        op = plre.ResourceGlobalPhase()
        assert op.resource_params == {}

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""
        op = plre.ResourceGlobalPhase(wires=0)
        expected = []

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        op = plre.ResourceGlobalPhase(wires=0)
        assert op.adjoint_resource_decomp() == [
            plre.GateCount(plre.ResourceGlobalPhase.resource_rep(), 1)
        ]

    globalphase_ctrl_data = (
        ([1], [1], [plre.GateCount(plre.ResourcePhaseShift.resource_rep(), 1)]),
        (
            [1, 2],
            [1, 1],
            [
                plre.GateCount(plre.ResourcePhaseShift.resource_rep(), 1),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(2, 0), 2),
            ],
        ),
        (
            [1, 2, 3],
            [1, 0, 0],
            [
                plre.GateCount(plre.ResourcePhaseShift.resource_rep(), 1),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(3, 2), 2),
            ],
        ),
    )

    @pytest.mark.parametrize("ctrl_wires, ctrl_values, expected_res", globalphase_ctrl_data)
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = plre.ResourceGlobalPhase(wires=0)
        op2 = plre.ResourceControlled(op, num_ctrl_wires, num_ctrl_values)
        assert repr(op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)) == repr(
            expected_res
        )
        assert repr(op2.resource_decomp(**op2.resource_params)) == repr(expected_res)

    globalphase_pow_data = (
        (1, [plre.GateCount(plre.ResourceGlobalPhase.resource_rep(), 1)]),
        (2, [plre.GateCount(plre.ResourceGlobalPhase.resource_rep(), 1)]),
        (5, [plre.GateCount(plre.ResourceGlobalPhase.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", globalphase_pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = plre.ResourceGlobalPhase()
        assert op.pow_resource_decomp(z) == expected_res
