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
Test the ResourceBasisRotation class
"""
import pytest

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use


class TestBasisRotation:
    """Test the ResourceBasisRotation class"""

    op_data = (
        re.ResourceBasisRotation(unitary_matrix=qml.matrix(qml.X(0) @ qml.Y(1)), wires=range(4)),
        re.ResourceBasisRotation(
            unitary_matrix=qml.matrix(qml.RX(1.23, 0) @ qml.RY(4.56, 1) @ qml.Z(2)), wires=range(8)
        ),
        re.ResourceBasisRotation(
            unitary_matrix=qml.matrix(
                qml.Hadamard(0) @ qml.Hadamard(1) @ qml.Hadamard(2) @ qml.Hadamard(3)
            ),
            wires=range(16),
        ),
    )

    resource_data = (
        {
            re.ResourcePhaseShift.resource_rep(): 10,
            re.ResourceSingleExcitation.resource_rep(): 6,
        },
        {
            re.ResourcePhaseShift.resource_rep(): 36,
            re.ResourceSingleExcitation.resource_rep(): 28,
        },
        {
            re.ResourcePhaseShift.resource_rep(): 136,
            re.ResourceSingleExcitation.resource_rep(): 120,
        },
    )

    resource_params_data = (
        {
            "dim_N": 4,
        },
        {
            "dim_N": 8,
        },
        {
            "dim_N": 16,
        },
    )

    name_data = (
        "BasisRotation(4)",
        "BasisRotation(8)",
        "BasisRotation(16)",
    )

    @pytest.mark.parametrize(
        "op, params, expected_res", zip(op_data, resource_params_data, resource_data)
    )
    def test_resources(self, op, params, expected_res):
        """Test the resources method returns the correct dictionary"""
        res_from_op = op.resources(**op.resource_params)
        res_from_func = re.ResourceBasisRotation.resources(**params)

        assert res_from_op == expected_res
        assert res_from_func == expected_res

    @pytest.mark.parametrize("op, expected_params", zip(op_data, resource_params_data))
    def test_resource_params(self, op, expected_params):
        """Test that the resource params are correct"""
        assert op.resource_params == expected_params

    @pytest.mark.parametrize("expected_params", resource_params_data)
    def test_resource_rep(self, expected_params):
        """Test the resource_rep returns the correct CompressedResourceOp"""
        expected = re.CompressedResourceOp(re.ResourceBasisRotation, expected_params)
        assert re.ResourceBasisRotation.resource_rep(**expected_params) == expected

    @pytest.mark.parametrize("params, expected_name", zip(resource_params_data, name_data))
    def test_tracking_name(self, params, expected_name):
        """Test that the tracking name is correct."""
        assert re.ResourceBasisRotation.tracking_name(**params) == expected_name
