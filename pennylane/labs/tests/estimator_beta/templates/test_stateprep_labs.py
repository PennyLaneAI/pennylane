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

        assert qre.ResourceMottonenStatePreparation.resource_decomp(num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct"""
        op = qre.ResourceMottonenStatePreparation(num_wires)

        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_rep(self, num_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = qre.CompressedResourceOp(
            qre.ResourceMottonenStatePreparation,
            num_wires,
            {"num_wires": num_wires},
        )
        assert expected == qre.ResourceMottonenStatePreparation.resource_rep(num_wires)


class TestCosineWindow:
    """Test the ResourceCosineWindow class"""

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

        assert qre.ResourceCosineWindow.resource_decomp(num_wires) == expected

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_params(self, num_wires):
        """Test that the resource params are correct"""
        op = qre.ResourceCosineWindow(num_wires)

        assert op.resource_params == {"num_wires": num_wires}

    @pytest.mark.parametrize(
        "num_wires",
        [(4), (5), (6)],
    )
    def test_resource_rep(self, num_wires):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = qre.CompressedResourceOp(
            qre.ResourceCosineWindow,
            num_wires,
            {"num_wires": num_wires},
        )
        assert expected == qre.ResourceCosineWindow.resource_rep(num_wires)


class TestSumOfSlatersPrep:
    """Test the ResourceSumOfSlatersPrep class"""

    @pytest.mark.parametrize(
        "num_coeffs, num_wires, stateprep_op, select_swap_depth, expected_resources",
        [
            (
                40,
                16,
                None,
                1,
                [
                    qre.Allocate(6),
                    GateCount(
                        resource_rep(qre.ResourceMottonenStatePreparation, {"num_wires": 6}), 1
                    ),
                    GateCount(
                        resource_rep(
                            qre.QROM,
                            {
                                "num_bitstrings": 40,
                                "size_bitstring": 16,
                                "restored": False,
                                "select_swap_depth": 1,
                            },
                        ),
                        1,
                    ),
                    qre.Allocate(11),
                    GateCount(resource_rep(qre.CNOT), 352),
                    GateCount(resource_rep(qre.X), 440),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX, {"num_ctrl_wires": 11, "num_zero_ctrl": 0}
                        ),
                        39,
                    ),
                    qre.Deallocate(11),
                    qre.Deallocate(6),
                ],
            ),
            (
                56,
                20,
                None,
                1,
                [
                    qre.Allocate(6),
                    GateCount(
                        resource_rep(qre.ResourceMottonenStatePreparation, {"num_wires": 6}), 1
                    ),
                    GateCount(
                        resource_rep(
                            qre.QROM,
                            {
                                "num_bitstrings": 56,
                                "size_bitstring": 20,
                                "restored": False,
                                "select_swap_depth": 1,
                            },
                        ),
                        1,
                    ),
                    qre.Allocate(11),
                    GateCount(resource_rep(qre.CNOT), 440),
                    GateCount(resource_rep(qre.X), 616),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX, {"num_ctrl_wires": 11, "num_zero_ctrl": 0}
                        ),
                        55,
                    ),
                    qre.Deallocate(11),
                    qre.Deallocate(6),
                ],
            ),
            (
                100,
                10,
                resource_rep(qre.QROMStatePreparation, {"num_state_qubits": 7}),
                2,
                [
                    qre.Allocate(7),
                    GateCount(resource_rep(qre.QROMStatePreparation, {"num_state_qubits": 7}), 1),
                    GateCount(
                        resource_rep(
                            qre.QROM,
                            {
                                "num_bitstrings": 100,
                                "size_bitstring": 10,
                                "restored": False,
                                "select_swap_depth": 2,
                            },
                        ),
                        1,
                    ),
                    GateCount(resource_rep(qre.X), 1000),
                    GateCount(
                        resource_rep(
                            qre.MultiControlledX, {"num_ctrl_wires": 10, "num_zero_ctrl": 0}
                        ),
                        99,
                    ),
                    qre.Deallocate(7),
                ],
            ),
        ],
    )
    def test_resources(
        self, num_coeffs, num_wires, stateprep_op, select_swap_depth, expected_resources
    ):
        """Test that the resources are correct"""
        res = qre.ResourceSumOfSlatersPrep.resource_decomp(
            num_coeffs, num_wires, stateprep_op, select_swap_depth
        )

        for r, e in zip(res, expected_resources):
            if hasattr(r, "equal"):
                assert r.equal(e)
            else:
                assert r == e

    def test_resource_params(self):
        """Test that the resource params are correct"""
        op = qre.ResourceSumOfSlatersPrep(num_coeffs=100, num_wires=10)

        assert op.resource_params == {
            "num_wires": 10,
            "num_coeffs": 100,
            "stateprep_cmpr_op": None,
            "select_swap_depth": None,
        }

    def test_resource_rep(self):
        """Test the resource_rep returns the correct CompressedResourceOp"""

        expected = qre.CompressedResourceOp(
            qre.ResourceSumOfSlatersPrep,
            10,
            {
                "num_wires": 10,
                "num_coeffs": 100,
                "stateprep_cmpr_op": None,
                "select_swap_depth": None,
            },
        )
        assert expected == qre.ResourceSumOfSlatersPrep.resource_rep(num_coeffs=100, num_wires=10)
