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
Tests for the state preparation subroutines resource operators.
"""
import pytest

import pennylane.labs.estimator_beta as qre
from pennylane.estimator import GateCount, resource_rep

# pylint: disable=no-self-use,too-many-arguments

class TestMottonenStatePreparation:
    """Test the MottonenStatePreparation class"""

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resources(self, num_wires):
        """Test that the resources are correct"""
        rz = resource_rep(qre.RZ)
        cnot = resource_rep(qre.CNOT)

        r_count = 2 ** (num_wires + 2) - 5
        cnot_count = 2 ** (num_wires + 2) - 4 * num_wires - 4

        expected = [GateCount(rz, r_count), GateCount(cnot, cnot_count)]

        assert qre.MottonenStatePreparation.resource_decomp(num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct"""
        op = qre.MottonenStatePreparation(num_wires)

        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_rep(self, num_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = qre.CompressedResourceOp(
            qre.MottonenStatePreparation,
            num_wires,
            {"num_wires": num_wires},
        )
        assert expected == qre.MottonenStatePreparation.resource_rep(num_wires)


class TestCosineWindow:
    """Test the CosineWindow class"""

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resources(self, num_wires):
        """Test that the resources are correct"""
        hadamard = resource_rep(qre.Hadamard)
        rz = resource_rep(qre.RZ)
        iqft = resource_rep(
            qre.Adjoint,
            {"base_cmpr_op": resource_rep(qre.QFT, {"num_wires": num_wires})},
        )
        phase_shift = resource_rep(qre.PhaseShift)

        expected = [
            GateCount(hadamard, 1),
            GateCount(rz, 1),
            GateCount(iqft, 1),
            GateCount(phase_shift, num_wires),
        ]

        assert qre.CosineWindow.resource_decomp(num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct"""
        op = qre.CosineWindow(num_wires)

        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_rep(self, num_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = qre.CompressedResourceOp(
            qre.CosineWindow,
            num_wires,
            {"num_wires": num_wires},
        )
        assert expected == qre.CosineWindow.resource_rep(num_wires)