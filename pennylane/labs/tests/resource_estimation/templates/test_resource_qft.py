# Copyright 2024 Xanadu Quantum Technologies Inc.

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
Test the ResourceQFT class
"""
import pytest

import pennylane.labs.resource_estimation as re

# pylint: disable=no-self-use


class TestQFT:
    """Test the ResourceQFT class"""

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

    @pytest.mark.parametrize("wires", [range(1), range(2), range(3), range(4)])
    def test_resource_params(self, wires):
        """Test that the resource params are correct"""
        op = re.ResourceQFT(wires)
        assert op.resource_params() == {"num_wires": len(wires)}

    @pytest.mark.parametrize("num_wires", [1, 2, 3, 4])
    def test_resource_rep(self, num_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = re.CompressedResourceOp(re.ResourceQFT, {"num_wires": num_wires})
        assert re.ResourceQFT.resource_rep(num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires, num_hadamard, num_swap, num_ctrl_phase_shift",
        [
            (1, 1, 0, 0),
            (2, 2, 1, 1),
            (3, 3, 1, 3),
            (4, 4, 2, 6),
        ],
    )
    def test_resources_from_rep(self, num_wires, num_hadamard, num_swap, num_ctrl_phase_shift):
        """Test that computing the resources from a compressed representation works"""

        hadamard = re.CompressedResourceOp(re.ResourceHadamard, {})
        swap = re.CompressedResourceOp(re.ResourceSWAP, {})
        ctrl_phase_shift = re.CompressedResourceOp(re.ResourceControlledPhaseShift, {})

        expected = {hadamard: num_hadamard, swap: num_swap, ctrl_phase_shift: num_ctrl_phase_shift}

        rep = re.ResourceQFT.resource_rep(num_wires)
        actual = rep.op_type.resources(**rep.params)

        assert actual == expected

    @pytest.mark.parametrize("num_wires", range(10))
    def test_tracking_name(self, num_wires):
        """Test that the tracking name is correct."""
        assert re.ResourceQFT.tracking_name(num_wires + 1) == "QFT"
