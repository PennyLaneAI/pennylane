# Copyright 2026 Xanadu Quantum Technologies Inc.

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

from collections import defaultdict

import pytest

import pennylane.labs.estimator_beta as qre
from pennylane.labs.estimator_beta import GateCount, resource_rep

# pylint: disable=no-self-use, use-implicit-booleaness-not-comparison


class TestHadamard:
    """Tests for Hadamard resource operator"""

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
                qre.Allocate(1),
                GateCount(qre.Hadamard.resource_rep(), 2),
                GateCount(qre.T.resource_rep(), 1),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.T)}), 1),
                GateCount(qre.S.resource_rep(), 1),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
                GateCount(qre.MultiControlledX.resource_rep(2, 0), 2),
                qre.Deallocate(1),
            ],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [
                qre.Allocate(1),
                GateCount(qre.Hadamard.resource_rep(), 2),
                GateCount(qre.T.resource_rep(), 1),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.T)}), 1),
                GateCount(qre.S.resource_rep(), 1),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
                GateCount(qre.MultiControlledX.resource_rep(3, 2), 2),
                qre.Deallocate(1),
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
        num_zero_ctrl = len([v for v in ctrl_values if not v])

        result = qre.hadamard_controlled_resource_decomp(
            num_ctrl_wires=num_ctrl_wires, num_zero_ctrl=num_zero_ctrl
        )

        for r, e in zip(result, expected_res):
            if hasattr(r, "equal"):
                assert r.equal(e)
            else:
                assert r == e

    ctrl_data = (
        (
            ["c1"],
            [1],
            qre.Resources(
                zeroed_wires=0,
                any_state_wires=0,
                algo_wires=2,
                gate_types=defaultdict(
                    int,
                    {
                        resource_rep(qre.Hadamard): 2,
                        resource_rep(qre.T): 2,
                        resource_rep(qre.Z): 2,
                        resource_rep(qre.S): 3,
                        resource_rep(qre.CNOT): 1,
                    },
                ),
            ),
        ),
        (
            ["c1"],
            [0],
            qre.Resources(
                zeroed_wires=0,
                any_state_wires=0,
                algo_wires=2,
                gate_types=defaultdict(
                    int,
                    {
                        resource_rep(qre.T): 2,
                        resource_rep(qre.X): 2,
                        resource_rep(qre.Z): 2,
                        resource_rep(qre.S): 3,
                        resource_rep(qre.Hadamard): 2,
                        resource_rep(qre.CNOT): 1,
                    },
                ),
            ),
        ),
    )

    @pytest.mark.parametrize(
        "ctrl_wires, ctrl_values, expected_res",
        ctrl_data,
    )
    def test_resource_controlled_estimate(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected when estimate is used."""
        num_ctrl_wires = len(ctrl_wires)
        num_zero_ctrl = len([v for v in ctrl_values if not v])

        result = qre.estimate(
            qre.Controlled(
                qre.Hadamard(), num_ctrl_wires=num_ctrl_wires, num_zero_ctrl=num_zero_ctrl
            )
        )
        print(result)
        assert result == expected_res

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
                GateCount(resource_rep(qre.CNOT), 5),
                GateCount(resource_rep(qre.CZ), 1),
                GateCount(resource_rep(qre.X), 4),
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
                GateCount(resource_rep(qre.CNOT), 5),
                GateCount(resource_rep(qre.CZ), 1),
                GateCount(resource_rep(qre.X), 4),
                qre.Deallocate(1),
            ],
        ),
        (
            ["c1", "c2"],
            [1, 1],
            [
                qre.Allocate(1),
                qre.Allocate(1),
                GateCount(qre.Hadamard.resource_rep(), 5),
                GateCount(qre.S.resource_rep(), 2),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
                GateCount(resource_rep(qre.Toffoli), 1),
                GateCount(resource_rep(qre.CNOT), 5),
                GateCount(resource_rep(qre.CZ), 1),
                GateCount(resource_rep(qre.X), 4),
                GateCount(
                    resource_rep(qre.MultiControlledX, {"num_ctrl_wires": 2, "num_zero_ctrl": 0}), 2
                ),
                qre.Deallocate(1),
                qre.Deallocate(1),
            ],
        ),
        (
            ["c1", "c2", "c3"],
            [1, 0, 0],
            [
                qre.Allocate(1),
                qre.Allocate(1),
                GateCount(qre.Hadamard.resource_rep(), 5),
                GateCount(qre.S.resource_rep(), 2),
                GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)}), 1),
                GateCount(resource_rep(qre.Toffoli), 1),
                GateCount(resource_rep(qre.CNOT), 5),
                GateCount(resource_rep(qre.CZ), 1),
                GateCount(resource_rep(qre.X), 4),
                GateCount(
                    resource_rep(qre.MultiControlledX, {"num_ctrl_wires": 3, "num_zero_ctrl": 2}), 2
                ),
                qre.Deallocate(1),
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

        result = qre.hadamard_toffoli_based_controlled_decomp(
            num_ctrl_wires=num_ctrl_wires, num_zero_ctrl=num_ctrl_values
        )

        for r, e in zip(result, expected_res):
            if hasattr(r, "equal"):
                assert r.equal(e)
            else:
                assert r == e
