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

import pytest

import pennylane as qml
import pennylane.labs.estimator_beta as qre
from pennylane.estimator import GateCount, ResourceConfig, resource_rep


# pylint: disable=no-self-use, too-many-arguments
class TestResourceSelectPauliRot:
    """Test the ResourceSelectPauliRot template"""

    def test_rot_axis_errors(self):
        """Test that the correct error is raised when invalid rotation axis argument is provided."""
        with pytest.raises(ValueError, match="The `rot_axis` argument must be one of"):
            qre.SelectPauliRot(rot_axis="A", num_ctrl_wires=1, precision=1e-3)

    @pytest.mark.parametrize("precision", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("rot_axis", ("X", "Y", "Z"))
    @pytest.mark.parametrize("num_ctrl_wires", (1, 2, 3, 4, 5))
    def test_resource_params(self, num_ctrl_wires, rot_axis, precision):
        """Test that the resource params are correct."""
        op = (
            qre.SelectPauliRot(rot_axis, num_ctrl_wires, precision)
            if precision
            else qre.SelectPauliRot(rot_axis, num_ctrl_wires)
        )
        assert op.resource_params == {
            "rot_axis": rot_axis,
            "num_ctrl_wires": num_ctrl_wires,
            "precision": precision,
        }

    @pytest.mark.parametrize("precision", (None, 1e-3, 1e-5))
    @pytest.mark.parametrize("rot_axis", ("X", "Y", "Z"))
    @pytest.mark.parametrize("num_ctrl_wires", (1, 2, 3, 4, 5))
    def test_resource_rep(self, num_ctrl_wires, rot_axis, precision):
        """Test that the compressed representation is correct."""
        expected = qre.CompressedResourceOp(
            qre.SelectPauliRot,
            num_ctrl_wires + 1,
            {
                "rot_axis": rot_axis,
                "num_ctrl_wires": num_ctrl_wires,
                "precision": precision,
            },
        )
        assert qre.SelectPauliRot.resource_rep(num_ctrl_wires, rot_axis, precision) == expected

    @pytest.mark.parametrize(
        "num_ctrl_wires, rot_axis, precision, expected_res",
        (
            (
                1,
                "X",
                None,
                [
                    GateCount(resource_rep(qre.RX, {"precision": 1e-9}), 2),
                    GateCount(resource_rep(qre.CNOT), 2),
                ],
            ),
            (
                2,
                "Y",
                1e-3,
                [
                    GateCount(resource_rep(qre.RY, {"precision": 1e-3}), 2**2),
                    GateCount(resource_rep(qre.CNOT), 2**2),
                ],
            ),
            (
                5,
                "Z",
                1e-5,
                [
                    GateCount(resource_rep(qre.RZ, {"precision": 1e-5}), 2**5),
                    GateCount(resource_rep(qre.CNOT), 2**5),
                ],
            ),
        ),
    )
    def test_default_resources(self, num_ctrl_wires, rot_axis, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[qml.estimator.SelectPauliRot]
            assert (
                qre.SelectPauliRot.resource_decomp(
                    num_ctrl_wires=num_ctrl_wires, rot_axis=rot_axis, **kwargs
                )
                == expected_res
            )
        else:
            assert (
                qre.SelectPauliRot.resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    rot_axis=rot_axis,
                    precision=precision,
                )
                == expected_res
            )

    @pytest.mark.parametrize(
        "num_ctrl_wires, rot_axis, precision, expected_res",
        (
            (
                1,
                "X",
                None,
                [
                    qre.Allocate(33),
                    GateCount(qre.QROM.resource_rep(2, 33, 33, False)),
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": qre.SemiAdder.resource_rep(33),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {
                                "base_cmpr_op": qre.QROM.resource_rep(2, 33, 33, False),
                            },
                        )
                    ),
                    qre.Deallocate(33),
                    GateCount(resource_rep(qre.Hadamard), 2),
                ],
            ),
            (
                2,
                "Y",
                1e-3,
                [
                    qre.Allocate(13),
                    GateCount(qre.QROM.resource_rep(4, 13, 26, False)),
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": qre.SemiAdder.resource_rep(13),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {
                                "base_cmpr_op": qre.QROM.resource_rep(4, 13, 26, False),
                            },
                        )
                    ),
                    qre.Deallocate(13),
                    GateCount(resource_rep(qre.Hadamard), 2),
                    GateCount(resource_rep(qre.S)),
                    GateCount(resource_rep(qre.Adjoint, {"base_cmpr_op": resource_rep(qre.S)})),
                ],
            ),
            (
                5,
                "Z",
                1e-5,
                [
                    qre.Allocate(20),
                    GateCount(qre.QROM.resource_rep(32, 20, 320, False)),
                    GateCount(
                        resource_rep(
                            qre.Controlled,
                            {
                                "base_cmpr_op": qre.SemiAdder.resource_rep(20),
                                "num_ctrl_wires": 1,
                                "num_zero_ctrl": 0,
                            },
                        )
                    ),
                    GateCount(
                        resource_rep(
                            qre.Adjoint,
                            {
                                "base_cmpr_op": qre.QROM.resource_rep(32, 20, 320, False),
                            },
                        )
                    ),
                    qre.Deallocate(20),
                ],
            ),
        ),
    )
    def test_phase_gradient_resources(self, num_ctrl_wires, rot_axis, precision, expected_res):
        """Test that the resources are correct."""
        if precision is None:
            config = ResourceConfig()
            kwargs = config.resource_op_precisions[qml.estimator.SelectPauliRot]
            assert (
                qre.SelectPauliRot.phase_grad_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires, rot_axis=rot_axis, **kwargs
                )
                == expected_res
            )
        else:
            assert (
                qre.SelectPauliRot.phase_grad_resource_decomp(
                    num_ctrl_wires=num_ctrl_wires,
                    rot_axis=rot_axis,
                    precision=precision,
                )
                == expected_res
            )

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
                qre.SelectPauliRot.controlled_resource_decomp(
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
                qre.SelectPauliRot.controlled_resource_decomp(
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
