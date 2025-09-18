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

import pytest

from pennylane.estimator.ops import GlobalPhase, Hadamard, Identity, T
from pennylane.estimator.ops.qubit.parametric_ops_single_qubit import (
    RX,
    RY,
    RZ,
    PhaseShift,
    Rot,
    _rotation_resources,
)
from pennylane.estimator.resource_operator import CompressedResourceOp, GateCount
from pennylane.exceptions import ResourcesUndefinedError

# pylint: disable=no-self-use, use-implicit-booleaness-not-comparison,too-many-arguments

params = list(zip([10e-3, 10e-4, 10e-5], [17, 21, 24]))


@pytest.mark.parametrize("precision, expected", params)
def test_rotation_resources(precision, expected):
    """Test the hardcoded resources used for RX, RY, RZ"""
    gate_types = [GateCount(CompressedResourceOp(T, 1, {}), expected)]

    assert gate_types == _rotation_resources(precision=precision)


class TestPauliRotation:
    """Test Resource operators for RX, RY, and RZ"""

    params_classes = [RX, RY, RZ]
    params_errors = [10e-3, 10e-4, 10e-5]
    params_ctrl_res = [
        [
            GateCount(CompressedResourceOp(Hadamard, 1, {}), 2),
            GateCount(CompressedResourceOp(RZ, 1, {}), 2),
        ],
        [
            GateCount(CompressedResourceOp(RY, 1, {}), 2),
        ],
        [
            GateCount(CompressedResourceOp(RZ, 1, {}), 2),
        ],
    ]

    @pytest.mark.parametrize("resource_op_class", params_classes)
    @pytest.mark.parametrize("precision", params_errors)
    def test_resources(self, resource_op_class, precision):
        """Test the resources method matches the expected number of rotation resources"""
        config = {"precision": precision}
        op = resource_op_class(wires=0)
        assert op.resource_decomp(**config) == _rotation_resources(precision=precision)

    @pytest.mark.parametrize("resource_op_class", params_classes)
    @pytest.mark.parametrize("precision", params_errors)
    def test_resource_rep(self, resource_op_class, precision):
        """Test the compact representation"""
        op = resource_op_class(wires=0, precision=precision)
        expected = CompressedResourceOp(resource_op_class, 1, {"precision": precision})
        assert op.resource_rep(precision=precision) == expected

    @pytest.mark.parametrize("resource_op_class", params_classes)
    @pytest.mark.parametrize("precision", params_errors)
    def test_resources_from_rep(self, resource_op_class, precision):
        """Test the resources can be obtained from the compact representation"""

        op = resource_op_class(wires=0, precision=precision)
        expected = _rotation_resources(precision=precision)

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    @pytest.mark.parametrize("resource_op_class", params_classes)
    @pytest.mark.parametrize("precision", params_errors)
    def test_resource_params(self, resource_op_class, precision):
        """Test that the resource params are correct"""
        op = resource_op_class(precision, wires=0)
        assert op.resource_params == {"precision": precision}

    @pytest.mark.parametrize("resource_op_class", params_classes)
    @pytest.mark.parametrize("precision", params_errors)
    def test_adjoint_decomposition(self, resource_op_class, precision):
        """Test that the adjoint decompositions are correct."""

        expected = [GateCount(resource_op_class(precision).resource_rep(), 1)]
        assert resource_op_class(precision).adjoint_resource_decomp({"precision": None}) == expected

    @pytest.mark.parametrize("resource_op_class", params_classes)
    @pytest.mark.parametrize("precision", params_errors)
    @pytest.mark.parametrize("z", list(range(1, 10)))
    def test_pow_decomposition(self, resource_op_class, precision, z):
        """Test that the pow decompositions are correct."""

        expected = [
            (
                GateCount(resource_op_class(precision).resource_rep(), 1)
                if z
                else GateCount(CompressedResourceOp(Identity, 1, {}), 1)
            )
        ]
        assert resource_op_class(precision).pow_resource_decomp(z, {"precision": None}) == expected

    params_ctrl_classes = (
        (RX),
        (RY),
        (RZ),
    )

    @pytest.mark.parametrize("resource_op_class", params_ctrl_classes)
    @pytest.mark.parametrize("precision", params_errors)
    def test_controlled_decomposition_single_control(
        self, resource_op_class, precision
    ):  # pylint: disable=unused-argument
        """Test that the controlled decompositions are correct."""
        with pytest.raises(ResourcesUndefinedError):
            resource_op_class.controlled_resource_decomp(1, 0, resource_op_class.resource_params)

    ctrl_res_data = (([1, 2], [1, 1]),)

    @pytest.mark.parametrize("resource_op_class", params_classes)
    @pytest.mark.parametrize("ctrl_wires, ctrl_values", ctrl_res_data)
    def test_controlled_decomposition_multi_controlled(
        self, resource_op_class, ctrl_wires, ctrl_values
    ):
        """Test that the controlled docomposition is correct when controlled on multiple wires."""
        with pytest.raises(ResourcesUndefinedError):
            resource_op_class.controlled_resource_decomp(
                ctrl_wires, ctrl_values, resource_op_class.resource_params
            )

    @pytest.mark.parametrize("resource_op_class", params_classes)
    def test_wire_error(self, resource_op_class):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 1 wires, got 2"):
            resource_op_class(wires=[0, 1])


class TestRot:
    """Test ResourceRot"""

    def test_resources(self):
        """Test the resources method matches the expected resources"""
        op = Rot(wires=0)
        ry = RY.resource_rep()
        rz = RZ.resource_rep()
        expected = [GateCount(ry, 1), GateCount(rz, 2)]

        assert op.resource_decomp() == expected

    def test_resource_rep(self):
        """Test the compressed representation"""
        op = Rot(wires=0)
        expected = CompressedResourceOp(Rot, 1, {"precision": None})
        assert op.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compact representation"""
        op = Rot(wires=0)
        ry = RY.resource_rep()
        rz = RZ.resource_rep()
        expected = [GateCount(ry, 1), GateCount(rz, 2)]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = Rot(wires=0)
        assert op.resource_params == {"precision": None}

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct"""

        expected = [GateCount(Rot.resource_rep(), 1)]
        assert Rot.adjoint_resource_decomp({"precision": None}) == expected

    params_ctrl_classes = ((Rot),)
    ctrl_data = ([1, 0], [1, 1])

    @pytest.mark.parametrize("resource_op_class", params_ctrl_classes)
    @pytest.mark.parametrize("ctrl_wires, ctrl_values", ctrl_data)
    def test_resource_controlled(self, resource_op_class, ctrl_wires, ctrl_values):
        """Test that the controlled resources are as expected"""
        with pytest.raises(ResourcesUndefinedError):
            resource_op_class.controlled_resource_decomp(
                ctrl_wires, ctrl_values, resource_op_class.resource_params
            )

    pow_data = (
        (1, [GateCount(Rot.resource_rep(), 1)]),
        (2, [GateCount(Rot.resource_rep(), 1)]),
        (5, [GateCount(Rot.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = Rot()
        assert op.pow_resource_decomp(z, op.resource_params) == expected_res

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 1 wires, got 2"):
            Rot(wires=[0, 1])


class TestPhaseShift:
    """Test ResourcePhaseShift"""

    def test_resources(self):
        """Test the resources method matches the expected resources"""
        op = PhaseShift(0.1, wires=0)
        rz = RZ.resource_rep()
        global_phase = GlobalPhase.resource_rep()

        expected = [GateCount(rz, 1), GateCount(global_phase, 1)]

        assert op.resource_decomp() == expected

    def test_resource_rep(self):
        """Test the compressed representation"""
        op = PhaseShift(wires=0)
        expected = CompressedResourceOp(PhaseShift, 1, {"precision": None})
        assert op.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compact representation"""
        op = PhaseShift(0.1)
        global_phase = GlobalPhase.resource_rep()
        rz = RZ.resource_rep(0.1)
        expected = [GateCount(rz, 1), GateCount(global_phase, 1)]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = PhaseShift()
        assert op.resource_params == {"precision": None}

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct"""

        expected = [GateCount(PhaseShift.resource_rep(), 1)]
        assert PhaseShift.adjoint_resource_decomp({"precision": None}) == expected

    params_ctrl_classes = ((PhaseShift),)
    ctrl_data = ([1, 0], [1, 1])

    @pytest.mark.parametrize("resource_op_class", params_ctrl_classes)
    @pytest.mark.parametrize("ctrl_wires, ctrl_values", ctrl_data)
    def test_resource_controlled(self, resource_op_class, ctrl_wires, ctrl_values):
        """Test that the controlled resources are as expected"""
        with pytest.raises(ResourcesUndefinedError):
            resource_op_class.controlled_resource_decomp(
                ctrl_wires, ctrl_values, resource_op_class.resource_params
            )

    pow_data = (
        (1, [GateCount(PhaseShift.resource_rep(), 1)]),
        (2, [GateCount(PhaseShift.resource_rep(), 1)]),
        (5, [GateCount(PhaseShift.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = PhaseShift()
        assert op.pow_resource_decomp(z, op.resource_params) == expected_res

    def test_wire_error(self):
        """Test that an error is raised when wrong number of wires is provided."""
        with pytest.raises(ValueError, match="Expected 1 wires, got 2"):
            PhaseShift(wires=[0, 1])
