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

import pennylane.labs.estimator_beta as qre
from pennylane.exceptions import ResourcesUndefinedError
from pennylane.labs.estimator_beta import CompressedResourceOp, GateCount, resource_rep

# pylint: disable=no-self-use,use-implicit-booleaness-not-comparison


class TestHadamard:
    """Tests for Hadamard resource operator"""

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 1 wires, got 2"):
            qre.Hadamard(wires=[0, 1])

    def test_resources(self):
        """Test that Hadamard resource operator does not implement a decomposition"""
        op = qre.Hadamard()
        with pytest.raises(ResourcesUndefinedError):
            op.resource_decomp()

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = qre.Hadamard()
        assert op.resource_params == {}

    def test_resource_rep(self):
        """Test that the compact representation is correct"""
        expected = CompressedResourceOp(qre.Hadamard, 1, {})
        assert qre.Hadamard.resource_rep() == expected

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct."""
        h = qre.Hadamard()
        h_dag = h.adjoint_resource_decomp()

        expected = [GateCount(qre.Hadamard.resource_rep(), 1)]
        assert h_dag == expected

    ctrl_data = (
        (
            ["c1"],
            [1],
            [
                GateCount(qre.CH.resource_rep(), 1),
            ],
        ),
        (
            ["c1"],
            [0],
            [
                GateCount(qre.CH.resource_rep(), 1),
                GateCount(qre.X.resource_rep(), 2),
            ],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [
                GateCount(qre.Hadamard.resource_rep(), 2),
                GateCount(qre.T.resource_rep(), 1),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.T)}), 1),
                GateCount(qre.S.resource_rep(), 1),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
                GateCount(qre.MultiControlledX.resource_rep(2, 0), 1),
            ],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [
                GateCount(qre.Hadamard.resource_rep(), 2),
                GateCount(qre.T.resource_rep(), 1),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.T)}), 1),
                GateCount(qre.S.resource_rep(), 1),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
                GateCount(qre.MultiControlledX.resource_rep(3, 2), 1),
            ],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = qre.Hadamard(0)
        op2 = qre.Controlled(op, num_ctrl_wires, num_ctrl_values)

        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == expected_res
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    ctrl_data = (
        (
            ["c1"],
            [1],
            [
                qre.Allocate(1),
                GateCount(qre.Hadamard.resource_rep(), 5),
                GateCount(qre.S.resource_rep(), 2),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
                GateCount(resource_rep(qre.Toffoli), 1),
                GateCount(resource_rep(qre.CNOT), 4),
                GateCount(resource_rep(qre.CZ), 1),
                GateCount(
                    resource_rep(qre.MultiControlledX, {"num_ctrl_wires": 1, "num_zero_ctrl": 0}), 1
                ),
                qre.Deallocate(1),
            ],
        ),
        (
            ["c1"],
            [0],
            [
                qre.Allocate(1),
                GateCount(qre.Hadamard.resource_rep(), 5),
                GateCount(qre.S.resource_rep(), 2),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
                GateCount(resource_rep(qre.Toffoli), 1),
                GateCount(resource_rep(qre.CNOT), 4),
                GateCount(resource_rep(qre.CZ), 1),
                GateCount(
                    resource_rep(qre.MultiControlledX, {"num_ctrl_wires": 1, "num_zero_ctrl": 1}), 1
                ),
                qre.Deallocate(1),
            ],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [
                qre.Allocate(1),
                GateCount(qre.Hadamard.resource_rep(), 5),
                GateCount(qre.S.resource_rep(), 2),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
                GateCount(resource_rep(qre.Toffoli), 1),
                GateCount(resource_rep(qre.CNOT), 4),
                GateCount(resource_rep(qre.CZ), 1),
                GateCount(
                    resource_rep(qre.MultiControlledX, {"num_ctrl_wires": 2, "num_zero_ctrl": 0}), 1
                ),
                qre.Deallocate(1),
            ],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [
                qre.Allocate(1),
                GateCount(qre.Hadamard.resource_rep(), 5),
                GateCount(qre.S.resource_rep(), 2),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
                GateCount(resource_rep(qre.Toffoli), 1),
                GateCount(resource_rep(qre.CNOT), 4),
                GateCount(resource_rep(qre.CZ), 1),
                GateCount(
                    resource_rep(qre.MultiControlledX, {"num_ctrl_wires": 3, "num_zero_ctrl": 2}), 1
                ),
                qre.Deallocate(1),
            ],
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_toffoli_based_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = qre.Hadamard(0)
        print(op.toffoli_based_controlled_decomp(num_ctrl_wires, num_ctrl_values))
        assert op.toffoli_based_controlled_decomp(num_ctrl_wires, num_ctrl_values) == expected_res

    pow_data = (
        (1, [GateCount(qre.Hadamard.resource_rep(), 1)]),
        (2, [GateCount(qre.Identity.resource_rep(), 1)]),
        (3, [GateCount(qre.Hadamard.resource_rep(), 1)]),
        (4, [GateCount(qre.Identity.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_pow_decomp(self, z, expected_res):
        """Test that the pow decomposition is correct."""
        op = qre.Hadamard(0)
        assert op.pow_resource_decomp(z) == expected_res
