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

import pennylane.labs.resource_estimation as re
from pennylane.labs.resource_estimation.ops.qubit.parametric_ops_single_qubit import (
    _rotation_resources,
)

# pylint: disable=no-self-use, use-implicit-booleaness-not-comparison,too-many-arguments

params = list(zip([10e-3, 10e-4, 10e-5], [17, 21, 24]))


@pytest.mark.parametrize("epsilon, expected", params)
def test_rotation_resources(epsilon, expected):
    """Test the hardcoded resources used for RX, RY, RZ"""
    gate_types = [re.GateCount(re.CompressedResourceOp(re.ResourceT, {}), expected)]

    assert gate_types == _rotation_resources(epsilon=epsilon)


class TestPauliRotation:
    """Test ResourceRX, ResourceRY, and ResourceRZ"""

    params_classes = [re.ResourceRX, re.ResourceRY, re.ResourceRZ]
    params_errors = [10e-3, 10e-4, 10e-5]

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    def test_resources(self, resource_class, epsilon):
        """Test the resources method"""

        label = "error_" + resource_class.__name__.replace("Resource", "").lower()
        config = {label: epsilon}
        op = resource_class(wires=0)
        assert op.resource_decomp(config=config) == _rotation_resources(epsilon=epsilon)

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    def test_resource_rep(self, resource_class, epsilon):  # pylint: disable=unused-argument
        """Test the compact representation"""
        op = resource_class(wires=0)
        expected = re.CompressedResourceOp(resource_class, {"eps": None})
        assert op.resource_rep() == expected

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    def test_resources_from_rep(self, resource_class, epsilon):
        """Test the resources can be obtained from the compact representation"""

        label = "error_" + resource_class.__name__.replace("Resource", "").lower()
        config = {label: epsilon}
        op = resource_class(wires=0)
        expected = _rotation_resources(epsilon=epsilon)

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params, config=config) == expected

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    def test_resource_params(self, resource_class, epsilon):  # pylint: disable=unused-argument
        """Test that the resource params are correct"""
        op = resource_class(epsilon, wires=0)
        assert op.resource_params == {"eps": epsilon}

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    def test_adjoint_decomposition(self, resource_class, epsilon):
        """Test that the adjoint decompositions are correct."""

        expected = [re.GateCount(resource_class(epsilon).resource_rep(), 1)]
        assert resource_class(epsilon).adjoint_resource_decomp() == expected

    @pytest.mark.parametrize("resource_class", params_classes)
    @pytest.mark.parametrize("epsilon", params_errors)
    @pytest.mark.parametrize("z", list(range(0, 10)))
    def test_pow_decomposition(self, resource_class, epsilon, z):
        """Test that the pow decompositions are correct."""

        expected = [
            (
                re.GateCount(resource_class(epsilon).resource_rep(), 1)
                if z
                else re.GateCount(re.ResourceIdentity.resource_rep(), 1)
            )
        ]
        assert resource_class(epsilon).pow_resource_decomp(z) == expected


class TestRot:
    """Test ResourceRot"""

    def test_resources(self):
        """Test the resources method"""
        op = re.ResourceRot(wires=0)
        ry = re.ResourceRY.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        expected = [re.GateCount(ry, 1), re.GateCount(rz, 2)]

        assert op.resource_decomp() == expected

    def test_resource_rep(self):
        """Test the compressed representation"""
        op = re.ResourceRot(wires=0)
        expected = re.CompressedResourceOp(re.ResourceRot, {"eps": None})
        assert op.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compact representation"""
        op = re.ResourceRot(wires=0)
        ry = re.ResourceRY.resource_rep()
        rz = re.ResourceRZ.resource_rep()
        expected = [re.GateCount(ry, 1), re.GateCount(rz, 2)]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourceRot(wires=0)
        assert op.resource_params == {"eps": None}

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct"""

        expected = [re.GateCount(re.ResourceRot.resource_rep(), 1)]
        assert re.ResourceRot.adjoint_resource_decomp() == expected

    pow_data = (
        (0, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
        (1, [re.GateCount(re.ResourceRot.resource_rep(), 1)]),
        (2, [re.GateCount(re.ResourceRot.resource_rep(), 1)]),
        (5, [re.GateCount(re.ResourceRot.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = re.ResourceRot()
        assert op.pow_resource_decomp(z) == expected_res


class TestPhaseShift:
    """Test ResourcePhaseShift"""

    def test_resources(self):
        """Test the resources method"""
        op = re.ResourcePhaseShift(0.1, wires=0)
        rz = re.ResourceRZ.resource_rep()
        global_phase = re.ResourceGlobalPhase.resource_rep()

        expected = [re.GateCount(rz, 1), re.GateCount(global_phase, 1)]

        assert op.resource_decomp() == expected

    def test_resource_rep(self):
        """Test the compressed representation"""
        op = re.ResourcePhaseShift(wires=0)
        expected = re.CompressedResourceOp(re.ResourcePhaseShift, {"eps": None})
        assert op.resource_rep() == expected

    def test_resources_from_rep(self):
        """Test that the resources can be obtained from the compact representation"""
        op = re.ResourcePhaseShift(0.1)
        global_phase = re.ResourceGlobalPhase.resource_rep()
        rz = re.ResourceRZ.resource_rep(0.1)
        expected = [re.GateCount(rz, 1), re.GateCount(global_phase, 1)]

        op_compressed_rep = op.resource_rep_from_op()
        op_resource_type = op_compressed_rep.op_type
        op_resource_params = op_compressed_rep.params
        assert op_resource_type.resource_decomp(**op_resource_params) == expected

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = re.ResourcePhaseShift()
        assert op.resource_params == {"eps": None}

    def test_adjoint_decomp(self):
        """Test that the adjoint decomposition is correct"""

        expected = [re.GateCount(re.ResourcePhaseShift.resource_rep(), 1)]
        assert re.ResourcePhaseShift.adjoint_resource_decomp() == expected

    pow_data = (
        (0, [re.GateCount(re.ResourceIdentity.resource_rep(), 1)]),
        (1, [re.GateCount(re.ResourcePhaseShift.resource_rep(), 1)]),
        (2, [re.GateCount(re.ResourcePhaseShift.resource_rep(), 1)]),
        (5, [re.GateCount(re.ResourcePhaseShift.resource_rep(), 1)]),
    )

    @pytest.mark.parametrize("z, expected_res", pow_data)
    def test_resource_pow(self, z, expected_res):
        """Test that the pow resources are as expected"""
        op = re.ResourcePhaseShift()
        assert op.pow_resource_decomp(z) == expected_res
