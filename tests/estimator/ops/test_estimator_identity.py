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

import pennylane.estimator as qre
from pennylane.estimator.ops import GlobalPhase, Identity
from pennylane.estimator.resource_operator import CompressedResourceOp, GateCount
from pennylane.wires import Wires

# pylint: disable=no-self-use,use-implicit-booleaness-not-comparison


class TestIdentity:
    """Test ResourceIdentity"""

    @pytest.mark.parametrize(
        "wire_labels", [0, [0], [1, 0, 4], ["a", "b", "c"], [0, 1, None], ["a", 1, "auxiliary"]]
    )
    def test_wires(self, wire_labels):
        """Test that common PL wires are accepted."""
        op = Identity(wire_labels)
        assert op.wires == Wires(wire_labels)
        assert op.num_wires == len(Wires(wire_labels))

    def test_resources(self):
        """ResourceIdentity should have empty resources"""
        op = Identity()
        assert op.resource_decomp() == []

    def test_resource_rep(self):
        """Test the compressed representation"""
        expected = CompressedResourceOp(Identity, 1, {})
        assert Identity.resource_rep() == expected

    def test_resource_params(self):
        """Test the resource params are correct"""
        op = Identity(0)
        assert op.resource_params == {}

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""
        op = Identity()
        expected = []

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        op = Identity(0)
        assert op.adjoint_resource_decomp() == [GateCount(Identity.resource_rep(), 1)]

    identity_ctrl_data = (
        ([1], [1], [GateCount(Identity.resource_rep(), 1)]),
        ([1, 2], [1, 1], [GateCount(Identity.resource_rep(), 1)]),
        ([1, 2, 3], [1, 0, 0], [GateCount(Identity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("ctrl_wires, ctrl_values, expected_res", identity_ctrl_data)
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = Identity(0)
        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == expected_res

    identity_pow_data = (
        (1, [GateCount(Identity.resource_rep(), 1)]),
        (2, [GateCount(Identity.resource_rep(), 1)]),
        (5, [GateCount(Identity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", identity_pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = Identity(0)
        assert op.pow_resource_decomp(z) == expected_res


class TestGlobalPhase:
    """Test ResourceGlobalPhase"""

    @pytest.mark.parametrize(
        "wire_labels", [0, [0], [1, 0, 4], ["a", "b", "c"], ["a", 1, "auxiliary"]]
    )
    def test_wires(self, wire_labels):
        """Test that common PL wires are accepted."""
        op = GlobalPhase(wire_labels)
        assert op.wires == Wires(wire_labels)
        assert op.num_wires == len(Wires(wire_labels))

    def test_resources(self):
        """ResourceGlobalPhase should have empty resources"""
        op = GlobalPhase(wires=0)
        assert op.resource_decomp() == []

    def test_resource_rep(self):
        """Test the compressed representation"""
        expected = CompressedResourceOp(GlobalPhase, 1, {})
        assert GlobalPhase.resource_rep() == expected

    def test_resource_params(self):
        """Test the resource params are correct"""
        op = GlobalPhase()
        assert op.resource_params == {}

    def test_resources_from_rep(self):
        """Test that the resources can be computed from the compressed representation"""
        op = GlobalPhase(wires=[0])
        expected = []

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        op = GlobalPhase(wires=[0])
        assert op.adjoint_resource_decomp() == [GateCount(GlobalPhase.resource_rep(), 1)]

    globalphase_ctrl_data = (
        ([1], [1], [GateCount(qre.PhaseShift.resource_rep(), 1)]),
        (
            [1],
            [0],
            [GateCount(qre.PhaseShift.resource_rep(), 1), GateCount(qre.X.resource_rep(), 2)],
        ),
        (
            [1, 2],
            [1, 1],
            [
                GateCount(qre.PhaseShift.resource_rep(), 1),
                GateCount(qre.MultiControlledX.resource_rep(2, 0), 2),
            ],
        ),
        (
            [1, 2, 3],
            [1, 0, 0],
            [
                GateCount(qre.PhaseShift.resource_rep(), 1),
                GateCount(qre.MultiControlledX.resource_rep(3, 2), 2),
            ],
        ),
    )

    @pytest.mark.parametrize("ctrl_wires, ctrl_values, expected_res", globalphase_ctrl_data)
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = GlobalPhase(wires=0)
        op2 = qre.Controlled(op, num_ctrl_wires, num_ctrl_values)
        assert repr(op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)) == repr(
            expected_res
        )
        assert repr(op2.resource_decomp(**op2.resource_params)) == repr(expected_res)

    globalphase_pow_data = (
        (1, [GateCount(GlobalPhase.resource_rep(), 1)]),
        (2, [GateCount(GlobalPhase.resource_rep(), 1)]),
        (5, [GateCount(GlobalPhase.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", globalphase_pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = GlobalPhase()
        assert op.pow_resource_decomp(z) == expected_res
