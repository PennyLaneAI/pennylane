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
Tests for parametric single qubit resource operators.
"""
import copy

import pytest

import pennylane.labs.resource_estimation as plre
from pennylane.labs.resource_estimation.ops.qubit.parametric_ops_single_qubit import (
    _rotation_resources,
)
from pennylane.labs.resource_estimation.resource_config import ResourceConfig

# pylint: disable=no-self-use, use-implicit-booleaness-not-comparison,too-many-arguments

params = list(zip([10e-3, 10e-4, 10e-5], [17, 21, 24]))


@pytest.mark.parametrize("precision, expected", params)
def test_rotation_resources(precision, expected):
    """Test the hardcoded resources used for RX, RY, RZ"""
    gate_types = [plre.GateCount(plre.CompressedResourceOp(plre.ResourceT, 1, {}), expected)]

    assert gate_types == _rotation_resources(precision=precision)


class TestPauliRotation:
    """Test ResourceRX, ResourceRY, and ResourceRZ"""

    params_classes = [plre.ResourceRX, plre.ResourceRY, plre.ResourceRZ]
    params_errors = [10e-3, 10e-4, 10e-5]
    params_ctrl_res = [
        [
            plre.GateCount(plre.ResourceHadamard.resource_rep(), 2),
            plre.GateCount(plre.ResourceRZ.resource_rep(), 2),
        ],
        [
            plre.GateCount(plre.ResourceRY.resource_rep(), 2),
        ],
        [
            plre.GateCount(plre.ResourceRZ.resource_rep(), 2),
        ],
    ]

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("precision", params_errors)
    def test_resources(self, resource_class, precision):
        """Test the resources method"""

        config = {"precision": precision}
        op = resource_class(wires=0)
        assert op.resource_decomp(**config) == _rotation_resources(precision=precision)

    @pytest.mark.parametrize("resource_class", params_classes)
    def test_resource_rep(self, resource_class):
        """Test the compact representation"""
        op = resource_class(wires=0)
        expected = plre.CompressedResourceOp(resource_class, 1, {"precision": None})
        assert op.resource_rep() == expected

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("precision", params_errors)
    def test_resources_from_rep(self, resource_class, precision):
        """Test the resources can be obtained from the compact representation"""

        op = resource_class(wires=0, precision=precision)
        expected = _rotation_resources(precision=precision)

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("precision", params_errors)
    def test_resource_params(self, resource_class, precision):  # pylint: disable=unused-argument
        """Test that the resource params are correct"""
        op = resource_class(precision, wires=0)
        assert op.resource_params == {"precision": precision}

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("precision", params_errors)
    def test_adjoint_decomposition(self, resource_class, precision):
        """Test that the adjoint decompositions are correct."""

        expected = [plre.GateCount(resource_class(precision).resource_rep(), 1)]
        assert resource_class(precision).adjoint_resource_decomp() == expected

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("precision", params_errors)
    @pytest.mark.parametrize("z", list(range(1, 10)))
    def test_pow_decomposition(self, resource_class, precision, z):
        """Test that the pow decompositions are correct."""

        expected = [
            (
                plre.GateCount(resource_class(precision).resource_rep(), 1)
                if z
                else plre.GateCount(plre.ResourceIdentity.resource_rep(), 1)
            )
        ]
        assert resource_class(precision).pow_resource_decomp(z) == expected

    params_ctrl_classes = (
        (plre.ResourceRX, plre.ResourceCRX),
        (plre.ResourceRY, plre.ResourceCRY),
        (plre.ResourceRZ, plre.ResourceCRZ),
    )

    @pytest.mark.parametrize("resource_class, controlled_class", params_ctrl_classes)
    @pytest.mark.parametrize("precision", params_errors)
    def test_controlled_decomposition_single_control(
        self, resource_class, controlled_class, precision
    ):
        """Test that the controlled decompositions are correct."""
        expected = [plre.GateCount(controlled_class.resource_rep(), 1)]
        assert resource_class.controlled_resource_decomp(1, 0) == expected

        expected = [
            plre.GateCount(controlled_class.resource_rep(), 1),
            plre.GateCount(plre.ResourceX.resource_rep(), 2),
        ]
        assert resource_class.controlled_resource_decomp(1, 1) == expected

        op = resource_class(wires=0)
        c_op = plre.ResourceControlled(op, 1, 0)

        c = controlled_class(wires=[0, 1])

        config = ResourceConfig()
        config.resource_op_precisions[plre.ResourceRX]["precision"] = precision
        config.resource_op_precisions[plre.ResourceRY]["precision"] = precision
        config.resource_op_precisions[plre.ResourceRZ]["precision"] = precision
        config.resource_op_precisions[plre.ResourceCRX]["precision"] = precision
        config.resource_op_precisions[plre.ResourceCRY]["precision"] = precision
        config.resource_op_precisions[plre.ResourceCRZ]["precision"] = precision

        r1 = plre.estimate(c, config=config)
        r2 = plre.estimate(c_op, config=config)

        assert r1 == r2

    ctrl_res_data = (
        (
            [1, 2],
            [1, 1],
            [plre.GateCount(plre.ResourceMultiControlledX.resource_rep(2, 0), 2)],
        ),
        (
            [1, 2],
            [1, 0],
            [plre.GateCount(plre.ResourceMultiControlledX.resource_rep(2, 1), 2)],
        ),
        (
            [1, 2, 3],
            [1, 0, 0],
            [plre.GateCount(plre.ResourceMultiControlledX.resource_rep(3, 2), 2)],
        ),
    )

    @pytest.mark.parametrize("resource_class, local_res", zip(params_classes, params_ctrl_res))
    @pytest.mark.parametrize("ctrl_wires, ctrl_values, general_res", ctrl_res_data)
    def test_controlled_decomposition_multi_controlled(
        self, resource_class, local_res, ctrl_wires, ctrl_values, general_res
    ):
        """Test that the controlled docomposition is correct when controlled on multiple wires."""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = resource_class(wires=0)
        op2 = plre.ResourceControlled(op, num_ctrl_wires, num_ctrl_values)

        expected_resources = copy.copy(local_res)
        expected_resources.extend(general_res)

        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == expected_resources
        assert op2.resource_decomp(**op2.resource_params) == expected_resources


class TestRot:
    """Test ResourceRot"""

    def test_resources(self):
        """Test the resources method"""
        op = plre.ResourceRot(wires=0)
        ry = plre.ResourceRY.resource_rep()
        rz = plre.ResourceRZ.resource_rep()
        expected = [plre.GateCount(ry, 1), plre.GateCount(rz, 2)]

        assert op.resource_decomp() == expected

    def test_resource_rep(self):
        """Test the compressed representation"""
        op = plre.ResourceRot(wires=0)
        expected = plre.CompressedResourceOp(plre.ResourceRot, 1, {"precision": None})
        assert op.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compact representation"""
        op = plre.ResourceRot(wires=0)
        ry = plre.ResourceRY.resource_rep()
        rz = plre.ResourceRZ.resource_rep()
        expected = [plre.GateCount(ry, 1), plre.GateCount(rz, 2)]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = plre.ResourceRot(wires=0)
        assert op.resource_params == {"precision": None}

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct"""

        expected = [plre.GateCount(plre.ResourceRot.resource_rep(), 1)]
        assert plre.ResourceRot.adjoint_resource_decomp() == expected

    ctrl_data = (
        ([1], [1], [plre.GateCount(plre.ResourceCRot.resource_rep(), 1)]),
        (
            [1],
            [0],
            [
                plre.GateCount(plre.ResourceCRot.resource_rep(), 1),
                plre.GateCount(plre.ResourceX.resource_rep(), 2),
            ],
        ),
        (
            [1, 2],
            [1, 1],
            [
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(2, 0), 2),
                plre.GateCount(plre.ResourceRZ.resource_rep(), 3),
                plre.GateCount(plre.ResourceRY.resource_rep(), 2),
            ],
        ),
        (
            [1, 2, 3],
            [1, 0, 0],
            [
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(3, 2), 2),
                plre.GateCount(plre.ResourceRZ.resource_rep(), 3),
                plre.GateCount(plre.ResourceRY.resource_rep(), 2),
            ],
        ),
    )

    @pytest.mark.parametrize("ctrl_wires, ctrl_values, expected_res", ctrl_data)
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = plre.ResourceRot(wires=0)
        op2 = plre.ResourceControlled(op, num_ctrl_wires, num_ctrl_values)

        assert op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values) == expected_res
        assert op2.resource_decomp(**op2.resource_params) == expected_res

    pow_data = (
        (1, [plre.GateCount(plre.ResourceRot.resource_rep(), 1)]),
        (2, [plre.GateCount(plre.ResourceRot.resource_rep(), 1)]),
        (5, [plre.GateCount(plre.ResourceRot.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = plre.ResourceRot()
        assert op.pow_resource_decomp(z) == expected_res


class TestPhaseShift:
    """Test ResourcePhaseShift"""

    def test_resources(self):
        """Test the resources method"""
        op = plre.ResourcePhaseShift(0.1, wires=0)
        rz = plre.ResourceRZ.resource_rep()
        global_phase = plre.ResourceGlobalPhase.resource_rep()

        expected = [plre.GateCount(rz, 1), plre.GateCount(global_phase, 1)]

        assert op.resource_decomp() == expected

    def test_resource_rep(self):
        """Test the compressed representation"""
        op = plre.ResourcePhaseShift(wires=0)
        expected = plre.CompressedResourceOp(plre.ResourcePhaseShift, 1, {"precision": None})
        assert op.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compact representation"""
        op = plre.ResourcePhaseShift(0.1)
        global_phase = plre.ResourceGlobalPhase.resource_rep()
        rz = plre.ResourceRZ.resource_rep(0.1)
        expected = [plre.GateCount(rz, 1), plre.GateCount(global_phase, 1)]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = plre.ResourcePhaseShift()
        assert op.resource_params == {"precision": None}

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct"""

        expected = [plre.GateCount(plre.ResourcePhaseShift.resource_rep(), 1)]
        assert plre.ResourcePhaseShift.adjoint_resource_decomp() == expected

    ctrl_data = (
        ([1], [1], [plre.GateCount(plre.ResourceControlledPhaseShift.resource_rep(), 1)]),
        (
            [1],
            [0],
            [
                plre.GateCount(plre.ResourceControlledPhaseShift.resource_rep(), 1),
                plre.GateCount(plre.ResourceX.resource_rep(), 2),
            ],
        ),
        (
            [1, 2],
            [1, 1],
            [
                plre.AllocWires(1),
                plre.GateCount(plre.ResourceControlledPhaseShift.resource_rep(), 1),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(2, 0), 2),
                plre.FreeWires(1),
            ],
        ),
        (
            [1, 2, 3],
            [1, 0, 0],
            [
                plre.AllocWires(1),
                plre.GateCount(plre.ResourceControlledPhaseShift.resource_rep(), 1),
                plre.GateCount(plre.ResourceMultiControlledX.resource_rep(3, 2), 2),
                plre.FreeWires(1),
            ],
        ),
    )

    @pytest.mark.parametrize("ctrl_wires, ctrl_values, expected_res", ctrl_data)
    def test_resource_controlled(self, ctrl_wires, ctrl_values, expected_res):
        """Test that the controlled resources are as expected"""
        num_ctrl_wires = len(ctrl_wires)
        num_ctrl_values = len([v for v in ctrl_values if not v])

        op = plre.ResourcePhaseShift(wires=0)
        op2 = plre.ResourceControlled(op, num_ctrl_wires, num_ctrl_values)

        assert repr(op.controlled_resource_decomp(num_ctrl_wires, num_ctrl_values)) == repr(
            expected_res
        )
        assert repr(op2.resource_decomp(**op2.resource_params)) == repr(expected_res)

    pow_data = (
        (1, [plre.GateCount(plre.ResourcePhaseShift.resource_rep(), 1)]),
        (2, [plre.GateCount(plre.ResourcePhaseShift.resource_rep(), 1)]),
        (5, [plre.GateCount(plre.ResourcePhaseShift.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = plre.ResourcePhaseShift()
        assert op.pow_resource_decomp(z) == expected_res
