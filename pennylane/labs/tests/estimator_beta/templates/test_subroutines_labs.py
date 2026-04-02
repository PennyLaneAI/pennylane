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
"""
Tests for quantum algorithmic subroutines resource operators.
"""

from collections import defaultdict

import pytest

import pennylane as qml
import pennylane.labs.estimator_beta as qre
from pennylane.estimator import GateCount, ResourceConfig, resource_rep


# pylint: disable=too-few-public-methods, too-many-arguments, no-self-use
class TestResourceSelectPauliRot:
    """Test the custom controlled decomposition for ResourceSelectPauliRot template"""

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, num_ctrl_wires_base, rot_axis, precision, expected_res",
        (
            (
                1,
                0,
                1,
                "X",
                None,
                [
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": resource_rep(qre.RX, {"precision": 1e-9}),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        ),
                        2,
                    ),
                    GateCount(resource_rep(qre.CNOT), 2),
                ],
            ),
            (
                2,
                0,
                2,
                "Y",
                1e-3,
                [
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": resource_rep(qre.RY, {"precision": 1e-3}),
                                "num_ctrl_wires": 2,
                                "num_zero_ctrl": 0,
                            },
                        ),
                        2**2,
                    ),
                    GateCount(resource_rep(qre.CNOT), 2**2),
                ],
            ),
            (
                2,
                2,
                5,
                "Z",
                1e-5,
                [
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": resource_rep(qre.RZ, {"precision": 1e-5}),
                                "num_ctrl_wires": 2,
                                "num_zero_ctrl": 2,
                            },
                        ),
                        2**5,
                    ),
                    GateCount(resource_rep(qre.CNOT), 2**5),
                ],
            ),
        ),
    )
    def test_controlled_resources(
        self, num_ctrl_wires, num_zero_ctrl, num_ctrl_wires_base, rot_axis, precision, expected_res
    ):
        """Test that the controlled resources are correct."""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[qml.estimator.SelectPauliRot]
            assert (
                qre.selectpaulirot_controlled_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    num_zero_ctrl=num_zero_ctrl,
                    target_resource_params={
                        "num_ctrl_wires": num_ctrl_wires_base,
                        "rot_axis": rot_axis,
                        "precision": kwargs["precision"],
                    },
                )
                == expected_res
            )
        else:
            assert (
                qre.selectpaulirot_controlled_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    num_zero_ctrl=num_zero_ctrl,
                    target_resource_params={
                        "num_ctrl_wires": num_ctrl_wires_base,
                        "rot_axis": rot_axis,
                        "precision": precision,
                    },
                )
                == expected_res
            )

    @pytest.mark.parametrize(
        "num_ctrl_wires, num_zero_ctrl, num_ctrl_wires_base, rot_axis, precision, expected_res",
        (
            (
                2,
                0,
                2,
                "Y",
                1e-3,
                qre.Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=5,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.CNOT): 4,
                            resource_rep(qre.Toffoli, {"elbow": None}): 8,
                            resource_rep(qre.T): 168,
                        },
                    ),
                ),
            ),
            (
                2,
                2,
                5,
                "Z",
                1e-5,
                qre.Resources(
                    zeroed_wires=0,
                    any_state_wires=0,
                    algo_wires=8,
                    gate_types=defaultdict(
                        int,
                        {
                            resource_rep(qre.CNOT): 32,
                            resource_rep(qre.Toffoli, {"elbow": None}): 64,
                            resource_rep(qre.T): 1792,
                            resource_rep(qre.X): 256,
                        },
                    ),
                ),
            ),
        ),
    )
    def test_controlled_resources_estimate(
        self, num_ctrl_wires, num_zero_ctrl, num_ctrl_wires_base, rot_axis, precision, expected_res
    ):
        """Test that the controlled resources are correct when estimate is used."""
        op = qre.Controlled(
            qre.SelectPauliRot(
                rot_axis=rot_axis, num_ctrl_wires=num_ctrl_wires_base, precision=precision
            ),
            num_ctrl_wires=num_ctrl_wires,
            num_zero_ctrl=num_zero_ctrl,
        )
        assert qre.estimate(op) == expected_res
