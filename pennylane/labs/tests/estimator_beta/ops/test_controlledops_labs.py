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
"""Tests for controlled resource operators."""
import pytest

import pennylane.labs.estimator_beta as qre
from pennylane.estimator.resource_operator import CompressedResourceOp, GateCount, resource_rep
from pennylane.labs.estimator_beta.ops.op_math.controlled_ops import CH

# pylint: disable=no-self-use, use-implicit-booleaness-not-comparison,too-many-arguments,too-many-positional-arguments


class TestCH:
    """Test the Resource CH operation"""

    op = CH(wires=[0, 1])

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 2 wires, got 3"):
            CH(wires=[0, 1, 2])

    def test_resources(self):
        """Test that the resources method produces the expected resources."""

        expected_resources = [
            GateCount(qre.Hadamard.resource_rep(), 4),
            GateCount(qre.T.resource_rep(), 1),
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.T)}), 1),
            GateCount(qre.S.resource_rep(), 2),
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 2),
            GateCount(qre.CNOT.resource_rep(), 1),
        ]
        assert self.op.resource_decomp(**self.op.resource_params) == expected_resources

    def test_toffoli_based_resources(self):
        """Test that the resources method produces the expected resources when using a Toffoli-based decomposition."""

        expected_resources = [
            qre.Allocate(1),
            GateCount(qre.Hadamard.resource_rep(), 5),
            GateCount(qre.S.resource_rep(), 2),
            GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
            GateCount(resource_rep(qre.Toffoli), 1),
            GateCount(resource_rep(qre.CNOT), 5),
            GateCount(resource_rep(qre.CZ), 1),
            qre.Deallocate(1),
        ]
        assert self.op.toffoli_based_resource_decomp() == expected_resources

    def test_resource_rep(self):
        """Test the resource_rep produces the correct compressed representation."""
        expected_rep = CompressedResourceOp(CH, 2, {})
        assert self.op.resource_rep(**self.op.resource_params) == expected_rep

    def test_resource_params(self):
        """Test that the resource_params are produced as expected."""
        expected_params = {}
        assert self.op.resource_params == expected_params

    def test_resource_adjoint(self):
        """Test that the adjoint resources are as expected"""
        expected_res = [GateCount(self.op.resource_rep(), 1)]
        assert self.op.adjoint_resource_decomp() == expected_res

    def test_resource_controlled(self):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = 3
        num_zero_ctrl = 1

        expected_op = qre.Controlled(
            qre.Hadamard(),
            num_ctrl_wires=4,
            num_zero_ctrl=1,
        )
        expected_res = [GateCount(expected_op.resource_rep_from_op())]

        assert self.op.controlled_resource_decomp(num_ctrl_wires, num_zero_ctrl) == expected_res

    pow_data = (
        (1, [GateCount(op.resource_rep(), 1)]),
        (2, [GateCount(qre.Identity.resource_rep(), 1)]),
        (5, [GateCount(op.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""

        assert self.op.pow_resource_decomp(z) == expected_res
