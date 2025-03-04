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
Test the ResourceQuantumPhaseEstimation class
"""
import pytest

import pennylane as qml
import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use


class TestQuantumPhaseEstimation:
    """Test the ResourceQuantumPhaseEstimation class"""

    input_data = (
        (
            re.ResourceHadamard(0),
            [1, 2],
        ),
        (
            re.ResourceRX(1.23, 1),
            [2, 3, 4],
        ),
        (
            re.ResourceCRY(1.23, [0, 1]),
            [2, 3, 4, 5],
        ),
        (
            re.ResourceQFT([0, 1, 2]),
            [4, 5],
        ),
    )

    resource_data = (
        {
            re.ResourceHadamard.resource_rep(): 2,
            re.ResourceAdjoint.resource_rep(re.ResourceQFT, {"num_wires": 2}): 1,
            re.ResourceControlled.resource_rep(re.ResourceHadamard, {}, 1, 0, 0): 3,
        },
        {
            re.ResourceRX.resource_rep(): 3,
            re.ResourceAdjoint.resource_rep(re.ResourceQFT, {"num_wires": 3}): 1,
            re.ResourceControlled.resource_rep(re.ResourceRX, {}, 1, 0, 0): 7,
        },
        {
            re.ResourceCRY.resource_rep(): 4,
            re.ResourceAdjoint.resource_rep(re.ResourceQFT, {"num_wires": 4}): 1,
            re.ResourceControlled.resource_rep(re.ResourceCRY, {}, 1, 0, 0): 15,
        },
        {
            re.ResourceQFT.resource_rep(num_wires=3): 2,
            re.ResourceAdjoint.resource_rep(re.ResourceQFT, {"num_wires": 2}): 1,
            re.ResourceControlled.resource_rep(re.ResourceQFT, {"num_wires": 3}, 1, 0, 0): 3,
        },
    )

    resource_params_data = (
        {
            "base_class": re.ResourceHadamard,
            "base_params": {},
            "num_estimation_wires": 2,
        },
        {
            "base_class": re.ResourceRX,
            "base_params": {},
            "num_estimation_wires": 3,
        },
        {
            "base_class": re.ResourceCRY,
            "base_params": {},
            "num_estimation_wires": 4,
        },
        {
            "base_class": re.ResourceQFT,
            "base_params": {"num_wires": 3},
            "num_estimation_wires": 2,
        },
    )

    name_data = (
        "QPE(Hadamard, 2)",
        "QPE(RX, 3)",
        "QPE(CRY, 4)",
        "QPE(QFT(3), 2)",
    )

    @pytest.mark.parametrize(
        "num_wires, num_hadamard, num_swap, num_ctrl_phase_shift",
        [
            (1, 1, 0, 0),
            (2, 2, 1, 1),
            (3, 3, 1, 3),
            (4, 4, 2, 6),
        ],
    )
    def test_resources(self, num_wires, num_hadamard, num_swap, num_ctrl_phase_shift):
        """Test the resources method returns the correct dictionary"""
        hadamard = re.CompressedResourceOp(re.ResourceHadamard, {})
        swap = re.CompressedResourceOp(re.ResourceSWAP, {})
        ctrl_phase_shift = re.CompressedResourceOp(re.ResourceControlledPhaseShift, {})

        expected = {hadamard: num_hadamard, swap: num_swap, ctrl_phase_shift: num_ctrl_phase_shift}

        assert re.ResourceQFT.resources(num_wires) == expected

    @pytest.mark.parametrize(
        "unitary_and_wires, expected_params", zip(input_data, resource_params_data)
    )
    def test_resource_params(self, unitary_and_wires, expected_params):
        """Test that the resource params are correct"""
        unitary, estimation_wires = unitary_and_wires
        op = re.ResourceQuantumPhaseEstimation(unitary, estimation_wires=estimation_wires)
        assert op.resource_params == expected_params

    def test_resource_params_error(self):
        """Test that an error is raised if a resource operator is not provided."""
        with pytest.raises(TypeError, match="Can't obtain QPE resources when"):
            op = re.ResourceQuantumPhaseEstimation(qml.Hadamard(0), estimation_wires=[1, 2, 3])
            op.resource_params  # pylint: disable=pointless-statement

    @pytest.mark.parametrize("expected_params", resource_params_data)
    def test_resource_rep(self, expected_params):
        """Test the resource_rep returns the correct CompressedResourceOp"""
        expected = re.CompressedResourceOp(re.ResourceQuantumPhaseEstimation, expected_params)
        assert re.ResourceQuantumPhaseEstimation.resource_rep(**expected_params) == expected

    @pytest.mark.parametrize("params, expected_name", zip(resource_params_data, name_data))
    def test_tracking_name(self, params, expected_name):
        """Test that the tracking name is correct."""
        assert re.ResourceQuantumPhaseEstimation.tracking_name(**params) == expected_name
